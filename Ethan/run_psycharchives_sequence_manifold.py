"""Causal, missingness-aware sequence manifold on licensed PsychArchives data.

The test-person split is reserved. Raw rows and person IDs are never saved.
The previous prefix-mean prototype remains in run_psycharchives_manifold.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from run_psycharchives_manifold import CUES, LOG_CUES, TRAITS, load_data, split_people


SEEDS = (1, 2, 3)
MAX_EPOCHS = 120
PATIENCE = 18
BATCH_PEOPLE = 32
WEIGHTS = {"reconstruction": 0.05, "traits": 0.05,
           "transition": 0.02, "slow_stability": 0.01, "kl": 0.002}


def prepare(rows: pd.DataFrame, people: pd.DataFrame, split: dict, cues=None):
    """Fit transformations on training people and retain missingness masks."""
    cues = list(CUES if cues is None else cues)
    if not cues or len(cues) != len(set(cues)) or not set(cues).issubset(CUES):
        raise ValueError("Expected a nonempty, unique subset of selected cues")
    training_pairs = rows[rows.user_id.isin(split["train"]) & rows.next_valence.notna()]
    imputer = SimpleImputer(strategy="median").fit(training_pairs[cues])
    scaler = StandardScaler().fit(imputer.transform(training_pairs[cues]))
    labeled_train = people[people.user_id.isin(split["train"]) & people[TRAITS].notna().all(axis=1)]
    trait_scaler = StandardScaler().fit(labeled_train[TRAITS].to_numpy())
    traits_by_person = people.set_index("user_id")[TRAITS]
    blocks = {}
    for name in ("train", "dev"):
        part = rows[rows.user_id.isin(split[name])].copy()
        groups = list(part.groupby("user_id", sort=True))
        max_length = max(len(group) for _, group in groups)
        n_people, d = len(groups), len(cues)
        x = np.zeros((n_people, max_length, d), dtype=np.float32)
        observed = np.zeros_like(x)
        valid = np.zeros((n_people, max_length), dtype=bool)
        y = np.zeros((n_people, max_length), dtype=np.float32)
        trait_y = np.zeros((n_people, len(TRAITS)), dtype=np.float32)
        trait_valid = np.zeros(n_people, dtype=bool)
        lengths = []
        for i, (user_id, group) in enumerate(groups):
            length = len(group)
            lengths.append(length)
            raw = group[cues]
            x[i, :length] = scaler.transform(imputer.transform(raw)).astype(np.float32)
            observed[i, :length] = raw.notna().to_numpy(dtype=np.float32)
            valid[i, :length] = True
            y[i, :length] = ((group.valence.to_numpy(dtype=np.float32) - 1) / 5)
            person_trait = traits_by_person.loc[user_id].to_numpy(dtype=np.float32)
            if np.isfinite(person_trait).all():
                trait_valid[i] = True
                trait_y[i] = trait_scaler.transform(person_trait[None, :])[0]
        blocks[name] = {"x": torch.from_numpy(x),
                        "observed": torch.from_numpy(observed),
                        "valid": torch.from_numpy(valid),
                        "y": torch.from_numpy(y),
                        "traits": torch.from_numpy(trait_y),
                        "trait_valid": torch.from_numpy(trait_valid),
                        "lengths": torch.tensor(lengths, dtype=torch.long),
                        "people": n_people, "pairs": int(sum(length - 1 for length in lengths))}
    preprocessing = {"cues": cues, "log1p_cues": [cue for cue in cues if cue in LOG_CUES],
                     "imputer_medians": imputer.statistics_.tolist(),
                     "feature_mean": scaler.mean_.tolist(),
                     "feature_scale": scaler.scale_.tolist(),
                     "trait_mean": trait_scaler.mean_.tolist(),
                     "trait_scale": trait_scaler.scale_.tolist()}
    return blocks, preprocessing


class SequenceManifold(torch.nn.Module):
    """Ordered history -> slow phi; observed state -> fast z; z transition."""

    def __init__(self, dimension: int, use_transition: bool = True):
        super().__init__()
        self.use_transition = use_transition
        self.history = torch.nn.GRU(2 * dimension, 32, batch_first=True)
        self.slow = torch.nn.Linear(32, 8)
        self.fast = torch.nn.Sequential(torch.nn.Linear(2 * dimension + 4, 32),
                                        torch.nn.Tanh(), torch.nn.Linear(32, 8))
        self.transition = torch.nn.Sequential(torch.nn.Linear(8, 32),
                                              torch.nn.Tanh(), torch.nn.Linear(32, 4))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(),
                                           torch.nn.Linear(32, dimension))
        self.next_valence = torch.nn.Sequential(torch.nn.Linear(8, 32),
                                                torch.nn.Tanh(), torch.nn.Linear(32, 1))
        self.trait_head = torch.nn.Linear(4, 5)

    @staticmethod
    def draw(stats, sample):
        mean, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-8, 5)
        value = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1)
        return value, mean, kl

    def forward(self, x, observed, sample=False):
        masked_input = torch.cat([x * observed, observed], dim=-1)
        states, _ = self.history(masked_input)
        phi, phi_mean, phi_kl = self.draw(self.slow(states), sample)
        fast_input = torch.cat([masked_input, phi_mean], dim=-1)
        z, z_mean, z_kl = self.draw(self.fast(fast_input), sample)
        reconstruction = self.decoder(torch.cat([phi, z], dim=-1))
        predicted_z = (self.transition(torch.cat([phi_mean, z_mean], dim=-1))
                       if self.use_transition else z_mean)
        predicted_y = self.next_valence(
            torch.cat([phi_mean, predicted_z], dim=-1)).squeeze(-1)
        return {"phi": phi_mean, "z": z_mean, "predicted_z": predicted_z,
                "reconstruction": reconstruction, "predicted_y": predicted_y,
                "kl": phi_kl + z_kl}

    def decode_person(self, phi):
        return self.decoder(torch.cat([phi, torch.zeros_like(phi)], dim=-1))


def batch_view(block, indices):
    return {key: value[indices] for key, value in block.items() if isinstance(value, torch.Tensor)}


def next_mask(valid):
    return valid[:, :-1] & valid[:, 1:]


def train_loss(model, block, weights=None):
    weights = WEIGHTS if weights is None else weights
    result = model(block["x"], block["observed"], sample=True)
    pairs = next_mask(block["valid"])
    prediction = F.mse_loss(result["predicted_y"][:, :-1][pairs], block["y"][:, 1:][pairs])
    observed = block["observed"] * block["valid"].unsqueeze(-1)
    reconstruction = ((result["reconstruction"] - block["x"]).square() * observed).sum() / observed.sum()
    labeled = block["trait_valid"]
    last = block["lengths"] - 1
    last_phi = result["phi"][torch.arange(len(last)), last]
    traits = (F.mse_loss(model.trait_head(last_phi[labeled]), block["traits"][labeled])
              if weights["traits"] else prediction.new_zeros(()))
    transition = (F.mse_loss(result["predicted_z"][:, :-1][pairs],
                             result["z"][:, 1:][pairs].detach())
                  if weights["transition"] else prediction.new_zeros(()))
    stability = (result["phi"][:, 1:] - result["phi"][:, :-1]).square().mean(-1)[pairs].mean()
    kl = result["kl"][block["valid"]].mean()
    return (prediction + weights["reconstruction"] * reconstruction
            + weights["traits"] * traits + weights["transition"] * transition
            + weights["slow_stability"] * stability + weights["kl"] * kl)


def development_metrics(model, block):
    model.eval()
    with torch.no_grad():
        result = model(block["x"], block["observed"])
        pairs = next_mask(block["valid"])
        difference = (result["predicted_y"][:, :-1][pairs] - block["y"][:, 1:][pairs]) * 5
        rmse = float(torch.sqrt(difference.square().mean()))
        mae = float(difference.abs().mean())
        trait_mask = block["trait_valid"]
        last_phi = result["phi"][torch.arange(block["people"]), block["lengths"] - 1]
        trait_rmse = float(torch.sqrt(F.mse_loss(
            model.trait_head(last_phi[trait_mask]), block["traits"][trait_mask])))
    return {"next_valence_rmse_1_to_6": rmse,
            "next_valence_mae_1_to_6": mae,
            "trait_rmse_training_scaled": trait_rmse}


def metric_check(model, block):
    with torch.no_grad():
        phi = model(block["x"], block["observed"])["phi"][:, 0]
    eigenvalues = []
    metrics = []
    for point in phi[:10]:
        jacobian = torch.autograd.functional.jacobian(
            lambda coordinate: model.decode_person(coordinate.unsqueeze(0)).squeeze(0),
            point.detach().requires_grad_(True))
        metric = jacobian.T @ jacobian + 1e-4 * torch.eye(4)
        metrics.append(metric.detach())
        eigenvalues.extend(torch.linalg.eigvalsh(metric).detach().tolist())
    return {"sampled_metric_eigenvalue_min": min(eigenvalues),
            "sampled_metric_eigenvalue_max": max(eigenvalues),
            "sampled_metric_change_from_first_median_frobenius": float(torch.median(
                torch.stack([torch.linalg.matrix_norm(item - metrics[0]) for item in metrics[1:]])))}


def update_diagnostics(model, block):
    """Coordinate-scale diagnostics, not validated psychological magnitudes."""
    with torch.no_grad():
        result = model(block["x"], block["observed"])
        pairs = next_mask(block["valid"])
        slow_step = torch.linalg.vector_norm(result["phi"][:, 1:] - result["phi"][:, :-1], dim=-1)
        fast_step = torch.linalg.vector_norm(result["z"][:, 1:] - result["z"][:, :-1], dim=-1)
    return {"median_adjacent_slow_coordinate_change": float(slow_step[pairs].median()),
            "median_adjacent_fast_coordinate_change": float(fast_step[pairs].median())}


def fit(blocks, seed):
    torch.manual_seed(seed)
    model = SequenceManifold(len(CUES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    best_state, best_epoch, best_score, stale = None, 0, float("inf"), 0
    train = blocks["train"]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(train["people"])
        for indices in order.split(BATCH_PEOPLE):
            mini = batch_view(train, indices)
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss(model, mini)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        score = development_metrics(model, blocks["dev"])["next_valence_rmse_1_to_6"]
        if score < best_score - 1e-5:
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def infer_history(checkpoint: Path, chronological_cues: pd.DataFrame):
    """Stateless online readout; appending a row updates phi and z causally."""
    if chronological_cues.empty or not set(CUES).issubset(chronological_cues.columns):
        raise ValueError("Expected nonempty chronological history with all cue columns")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    pre = saved["preprocessing"]
    raw = chronological_cues[CUES].copy()
    if not raw["valence"].dropna().between(1, 6).all():
        raise ValueError("Unexpected valence scale")
    for cue in LOG_CUES:
        if (raw[cue].dropna() < 0).any():
            raise ValueError(f"Negative count/duration: {cue}")
        raw[cue] = np.log1p(raw[cue])
    values = raw.to_numpy(dtype=float)
    observed = np.isfinite(values).astype(np.float32)
    median = np.asarray(pre["imputer_medians"])
    values = np.where(np.isfinite(values), values, median)
    x = ((values - np.asarray(pre["feature_mean"])) /
         np.asarray(pre["feature_scale"])).astype(np.float32)
    model = SequenceManifold(len(CUES))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        result = model(torch.from_numpy(x[None]), torch.from_numpy(observed[None]))
        phi = result["phi"][0, -1]
        z = result["z"][0, -1]
        trait_scaled = model.trait_head(phi).numpy()
    jacobian = torch.autograd.functional.jacobian(
        lambda p: model.decode_person(p.unsqueeze(0)).squeeze(0),
        phi.detach().requires_grad_(True))
    metric = jacobian.T @ jacobian + 1e-4 * torch.eye(4)
    trait_values = trait_scaled * np.asarray(pre["trait_scale"]) + np.asarray(pre["trait_mean"])
    return {"observations_seen": len(raw), "slow_phi": phi.tolist(), "fast_z": z.tolist(),
            "estimated_traits": dict(zip(TRAITS, trait_values.tolist())),
            "next_questionnaire_valence_estimate": float(result["predicted_y"][0, -1] * 5 + 1),
            "local_metric_eigenvalues": torch.linalg.eigvalsh(metric).detach().tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "psycharchives" / "sequence")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, people, hashes = load_data(args.source_dir)
    split = split_people(people)
    blocks, preprocessing = prepare(rows, people, split)
    previous = Path(__file__).resolve().parent / "artifacts" / "psycharchives" / "development_results.json"
    previous_metrics = None
    if previous.is_file():
        saved = json.loads(previous.read_text(encoding="utf-8"))
        if saved.get("source_sha256") == hashes:
            previous_metrics = {"mean_dev_rmse": saved["mean_dev_rmse"],
                                "per_seed": [run["dev_next_valence_rmse_1_to_6"] for run in saved["runs"]]}
    report = {"status": "exploratory ordered-history manifold implementation",
              "source_sha256": hashes,
              "protocol": {"participant_split_seed": 42, "seeds": SEEDS,
                           "selection": "development next-valence RMSE",
                           "test": "reserved; never processed for modeling or scored",
                           "target": "next questionnaire valence; elapsed time unknown",
                           "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                           "batch_people": BATCH_PEOPLE, "loss_weights": WEIGHTS},
              "cohort": {"train_people": blocks["train"]["people"],
                         "dev_people": blocks["dev"]["people"],
                         "reserved_test_people": len(split["test"]),
                         "train_pairs": blocks["train"]["pairs"],
                         "dev_pairs": blocks["dev"]["pairs"]},
              "previous_prefix_mean_dev": previous_metrics, "runs": []}
    dev_labeled = blocks["dev"]["traits"][blocks["dev"]["trait_valid"]]
    report["descriptive_baselines"] = {
        "train_mean_trait_dev_rmse_training_scaled": float(torch.sqrt(dev_labeled.square().mean())),
        "interpretation": "Trait head is auxiliary; this baseline predicts the training trait mean"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        model, epoch = fit(blocks, seed)
        metrics = development_metrics(model, blocks["dev"])
        metric = metric_check(model, blocks["dev"])
        updates = update_diagnostics(model, blocks["dev"])
        torch.save({"state_dict": model.state_dict(), "preprocessing": preprocessing,
                    "seed": seed, "best_epoch": epoch,
                    "training_contract": "ordered-masked-history-and-fast-transition"},
                   args.output_dir / f"seed_{seed}.pt")
        report["runs"].append({"seed": seed, "best_epoch": epoch,
                               **metrics, **metric, **updates})
    report["mean_dev_rmse"] = float(np.mean(
        [run["next_valence_rmse_1_to_6"] for run in report["runs"]]))
    output = args.output_dir / "development_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cohort": report["cohort"],
                      "mean_dev_rmse": report["mean_dev_rmse"],
                      "runs": report["runs"]}, indent=2))


if __name__ == "__main__":
    main()
