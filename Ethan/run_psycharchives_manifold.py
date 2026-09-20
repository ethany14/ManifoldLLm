"""Local, license-conscious PsychArchives slow/fast manifold prototype.

Reads source CSVs in place. Never writes raw records or participant IDs.
The participant-level test split is reserved and never scored here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TRAITS = ["NPAR", "EPAR", "OPAR", "CPAR", "APAR"]
CUES = [
    "Activity_prob_still_IS", "Activity_prob_inVehicle_IS", "Activity_prob_onFoot_IS",
    "App_totalNum_usages_allApps_CS", "App_totalDur_usages_allApps_CS",
    "App_totalNum_usages_differentApps_CS", "GPS_location_home_IS",
    "GPS_location_work_IS", "Phone_totalNum_calls_CS", "Phone_totalDur_calls_CS",
    "Screen_totalNum_sessions_CS", "Screen_totalDur_sessions_CS",
    "Screen_totalNum_checks_CS", "Notification_totalNum_notific_CS",
    "Power_status_connected_IS", "Wifi_status_on_connected_IS",
    "Timestamp_status_weekend_IS", "Timestamp_status_morning_IS",
    "Timestamp_status_noon_IS", "Timestamp_status_afternoon_IS",
    "Timestamp_status_evening_IS", "valence",
]
LOG_CUES = [x for x in CUES if "totalNum" in x or "totalDur" in x]
SEEDS = (1, 2, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(source_dir: Path):
    paths = {name: source_dir / name for name in
             ("Codebook.csv", "person_variables.csv", "es_sensing_variables.csv")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    codebook = pd.read_csv(paths["Codebook.csv"], sep=";")
    people = pd.read_csv(paths["person_variables.csv"])
    rows = pd.read_csv(paths["es_sensing_variables.csv"],
                       usecols=["user_id", "es_questionnaire_id"] + CUES)
    if not {"Feature", "Description"}.issubset(codebook.columns):
        raise ValueError("Unexpected codebook schema")
    if not set(CUES + TRAITS + ["es_questionnaire_id"]).issubset(set(codebook.Feature)):
        raise ValueError("Required cues or traits absent from codebook")
    if people["user_id"].duplicated().any():
        raise ValueError("Duplicate person records")
    if rows.duplicated(["user_id", "es_questionnaire_id"]).any():
        raise ValueError("Duplicate questionnaire records")
    if not rows["user_id"].isin(people["user_id"]).all():
        raise ValueError("Questionnaire with no person record")
    if not rows["valence"].dropna().between(1, 6).all():
        raise ValueError("Unexpected valence scale")
    if rows[CUES].select_dtypes(exclude="number").shape[1]:
        raise ValueError("Selected cues must be numeric")
    rows = rows.sort_values(["user_id", "es_questionnaire_id"]).reset_index(drop=True)
    for col in LOG_CUES:
        if (rows[col].dropna() < 0).any():
            raise ValueError(f"Negative count/duration: {col}")
        rows[col] = np.log1p(rows[col])
    groups = rows.groupby("user_id", sort=False)
    for col in CUES:
        observed = rows[col].notna().astype(int)
        total = rows[col].fillna(0).groupby(rows["user_id"]).cumsum()
        count = observed.groupby(rows["user_id"]).cumsum()
        rows[f"history_{col}"] = total / count.replace(0, np.nan)
    rows["next_valence"] = groups["valence"].shift(-1)
    rows["prefix_count"] = groups.cumcount() + 1
    return rows, people, {name: sha256(path) for name, path in paths.items()}


def split_people(people: pd.DataFrame):
    complete = people[TRAITS].notna().all(axis=1).astype(int)
    train, remaining = train_test_split(
        np.arange(len(people)), test_size=0.30, random_state=42, stratify=complete)
    dev, test = train_test_split(
        remaining, test_size=0.50, random_state=42, stratify=complete.iloc[remaining])
    return {name: set(people.iloc[index]["user_id"].tolist()) for name, index in
            (("train", train), ("dev", dev), ("test", test))}


def make_blocks(rows, people, split):
    train_rows = rows[rows.user_id.isin(split["train"]) & rows.next_valence.notna()]
    imputer = SimpleImputer(strategy="median").fit(train_rows[CUES])
    scaler = StandardScaler().fit(imputer.transform(train_rows[CUES]))
    trait_people = people[people.user_id.isin(split["train"]) & people[TRAITS].notna().all(axis=1)]
    trait_scaler = StandardScaler().fit(trait_people[TRAITS].to_numpy())
    person_traits = people.set_index("user_id")[TRAITS]
    blocks = {}
    history_cols = [f"history_{col}" for col in CUES]
    for name in ("train", "dev"):
        part = rows[rows.user_id.isin(split[name]) & rows.next_valence.notna()].copy()
        current = scaler.transform(imputer.transform(part[CUES])).astype("float32")
        history = scaler.transform(imputer.transform(
            part[history_cols].set_axis(CUES, axis=1))).astype("float32")
        traits = person_traits.loc[part.user_id, TRAITS].to_numpy(dtype="float32")
        trait_mask = np.isfinite(traits).all(axis=1)
        traits[~trait_mask] = trait_scaler.mean_.astype("float32")
        traits = trait_scaler.transform(traits).astype("float32")
        last = ~part.user_id.duplicated(keep="last").to_numpy()
        blocks[name] = {
            "history": torch.from_numpy(history),
            "deviation": torch.from_numpy(current - history),
            "current": torch.from_numpy(current),
            "y": torch.tensor((part.next_valence.to_numpy(dtype="float32") - 1) / 5),
            "traits": torch.from_numpy(traits),
            "trait_last": torch.from_numpy(trait_mask & last),
            "adjacent": torch.tensor(np.flatnonzero(
                part.user_id.to_numpy()[:-1] == part.user_id.to_numpy()[1:]), dtype=torch.long),
            "person_count": part.user_id.nunique(),
            "row_count": len(part),
        }
    preprocessing = {"cues": CUES, "log1p_cues": LOG_CUES,
                     "imputer_medians": imputer.statistics_.tolist(),
                     "feature_mean": scaler.mean_.tolist(),
                     "feature_scale": scaler.scale_.tolist(),
                     "trait_mean": trait_scaler.mean_.tolist(),
                     "trait_scale": trait_scaler.scale_.tolist()}
    return blocks, preprocessing


class SlowFastManifold(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.slow = torch.nn.Sequential(torch.nn.Linear(dimension, 32), torch.nn.Tanh(),
                                        torch.nn.Linear(32, 8))
        self.fast = torch.nn.Sequential(torch.nn.Linear(dimension, 32), torch.nn.Tanh(),
                                        torch.nn.Linear(32, 8))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(),
                                           torch.nn.Linear(32, dimension))
        self.next_valence = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(),
                                                torch.nn.Linear(32, 1))
        self.trait_head = torch.nn.Linear(4, 5)

    @staticmethod
    def draw(stats, sample):
        mean, logvar = stats.chunk(2, dim=1)
        logvar = logvar.clamp(-8, 5)
        value = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(1).mean()
        return value, mean, kl

    def forward(self, block, sample):
        phi, phi_mean, phi_kl = self.draw(self.slow(block["history"]), sample)
        z, _, z_kl = self.draw(self.fast(block["deviation"]), sample)
        combined = torch.cat([phi, z], dim=1)
        return self.decoder(combined), self.next_valence(combined).squeeze(1), \
            self.trait_head(phi_mean), phi_mean, phi_kl + z_kl

    def decode_person(self, phi):
        return self.decoder(torch.cat([phi, torch.zeros_like(phi)], dim=1))


def dev_rmse(model, block):
    model.eval()
    with torch.no_grad():
        _, prediction, _, _, _ = model(block, False)
    return float(torch.sqrt(F.mse_loss(prediction, block["y"])) * 5)


def fit(blocks, seed):
    torch.manual_seed(seed)
    model = SlowFastManifold(len(CUES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    train, dev = blocks["train"], blocks["dev"]
    best, best_epoch, best_score, stale = None, 0, float("inf"), 0
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstruction, prediction, traits, phi, kl = model(train, True)
        trait_loss = F.mse_loss(traits[train["trait_last"]],
                                train["traits"][train["trait_last"]])
        adjacent = train["adjacent"]
        stability = F.mse_loss(phi[adjacent], phi[adjacent + 1])
        loss = (F.mse_loss(prediction, train["y"])
                + 0.05 * F.mse_loss(reconstruction, train["current"])
                + 0.05 * trait_loss + 0.002 * kl + 0.01 * stability)
        loss.backward()
        optimizer.step()
        score = dev_rmse(model, dev)
        if score < best_score - 1e-5:
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= 30:
                break
    model.load_state_dict(best)
    model.eval()
    return model, best_epoch, best_score


def metric_eigenvalues(model, phi_values):
    """G(phi) = J_phi decoder(phi,0)' J_phi decoder(phi,0) + 1e-4 I."""
    eigenvalues = []
    for phi in phi_values[:10]:
        point = phi.detach().clone().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            lambda p: model.decode_person(p.unsqueeze(0)).squeeze(0), point)
        metric = jacobian.T @ jacobian + 1e-4 * torch.eye(4)
        eigenvalues.append(torch.linalg.eigvalsh(metric).detach().numpy())
    return np.asarray(eigenvalues)


def infer_history(checkpoint: Path, chronological_cues: pd.DataFrame):
    """Recompute one person's state after a new questionnaire is appended.

    Caller supplies rows in temporal order with all CUES columns. No source
    participant ID is required or retained. This is an in-memory interface.
    """
    if chronological_cues.empty or not set(CUES).issubset(chronological_cues.columns):
        raise ValueError("Expected at least one row and the exact cue schema")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    pre = saved["preprocessing"]
    model = SlowFastManifold(len(CUES))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    raw = chronological_cues[CUES].copy()
    if not raw["valence"].dropna().between(1, 6).all():
        raise ValueError("Unexpected valence scale")
    for col in LOG_CUES:
        if (raw[col].dropna() < 0).any():
            raise ValueError(f"Negative count/duration: {col}")
        raw[col] = np.log1p(raw[col])
    current = raw.iloc[-1].to_numpy(dtype=float)
    history = raw.mean(skipna=True).to_numpy(dtype=float)
    median = np.asarray(pre["imputer_medians"])
    center = np.asarray(pre["feature_mean"])
    scale = np.asarray(pre["feature_scale"])
    current = (np.where(np.isfinite(current), current, median) - center) / scale
    history = (np.where(np.isfinite(history), history, median) - center) / scale
    block = {"history": torch.tensor(history[None, :], dtype=torch.float32),
             "deviation": torch.tensor((current - history)[None, :], dtype=torch.float32)}
    with torch.no_grad():
        _, next_valence, traits, phi, _ = model(block, False)
        z = model.fast(block["deviation"])[:, :4]
    eig = metric_eigenvalues(model, phi)
    trait_original = (traits.detach().numpy()[0] * np.asarray(pre["trait_scale"])
                      + np.asarray(pre["trait_mean"]))
    return {"observations_seen": len(raw), "slow_phi": phi.numpy()[0].tolist(),
            "fast_z": z.numpy()[0].tolist(),
            "estimated_traits": dict(zip(TRAITS, trait_original.tolist())),
            "next_questionnaire_valence_estimate": float(next_valence[0] * 5 + 1),
            "local_metric_eigenvalues": eig[0].tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "psycharchives")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, people, hashes = load_data(args.source_dir)
    split = split_people(people)
    blocks, preprocessing = make_blocks(rows, people, split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "exploratory implementation; not psychological validation",
        "source_sha256": hashes,
        "protocol": {"participant_split_seed": 42, "seeds": SEEDS,
                     "test": "reserved; never scored or used for preprocessing",
                     "target": "next questionnaire valence (1-6); not a fixed time horizon",
                     "selection": "development RMSE", "max_epochs": 200,
                     "trait_convention": "NPAR is emotional stability, not neuroticism"},
        "cohort": {"participants": len(people), "questionnaires": len(rows),
                   "complete_trait_participants": int(people[TRAITS].notna().all(axis=1).sum()),
                   "split_participants": {k: len(v) for k, v in split.items()},
                   "train_pairs": blocks["train"]["row_count"],
                   "dev_pairs": blocks["dev"]["row_count"]},
        "feature_columns": CUES, "runs": [],
    }
    dev_rows = rows[rows.user_id.isin(split["dev"]) & rows.next_valence.notna()]
    train_outcomes = rows[rows.user_id.isin(split["train"]) & rows.next_valence.notna()]
    report["descriptive_baselines"] = {
        "train_mean_valence_on_dev_rmse": float(np.sqrt(np.mean(
            (dev_rows.next_valence.to_numpy() - train_outcomes.next_valence.mean()) ** 2))),
        "current_valence_on_dev_rmse": float(np.sqrt(np.mean(
            (dev_rows.next_valence.to_numpy() - dev_rows.valence.to_numpy()) ** 2))),
        "interpretation": "Context only; the experiment selects an implementation, not a superiority claim",
    }
    for seed in SEEDS:
        model, epoch, score = fit(blocks, seed)
        with torch.no_grad():
            phi = model.slow(blocks["dev"]["history"])[:, :4]
        eig = metric_eigenvalues(model, phi)
        torch.save({"state_dict": model.state_dict(), "preprocessing": preprocessing,
                    "seed": seed, "best_epoch": epoch,
                    "training_contract": "PsychArchives variable-prefix slow/fast manifold"},
                   args.output_dir / f"seed_{seed}.pt")
        report["runs"].append({"seed": seed, "best_epoch": epoch,
                               "dev_next_valence_rmse_1_to_6": score,
                               "sampled_metric_eigenvalue_min": float(eig.min()),
                               "sampled_metric_eigenvalue_max": float(eig.max())})
    report["mean_dev_rmse"] = float(np.mean(
        [run["dev_next_valence_rmse_1_to_6"] for run in report["runs"]]))
    output = args.output_dir / "development_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cohort": report["cohort"],
                      "mean_dev_rmse": report["mean_dev_rmse"],
                      "runs": report["runs"]}, indent=2))


if __name__ == "__main__":
    main()
