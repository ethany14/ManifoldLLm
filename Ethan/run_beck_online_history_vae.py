"""Train a variable-history slow/fast VAE on the existing Beck person split.

The slow posterior reads only the expanding prefix mean available at each
prediction time. The outcome is next-prompt self-reported studying. This is
an engineering prototype, evaluated on previously inspected dev people.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from run_beck_behavior_pilot import OUT, SOURCE, build_pairs, scores
from run_beck_graph_geometry import STATE, TIME
from run_beck_riemannian_vae import SlowFastVAE


SEEDS = (1, 2, 3)
EPOCHS = 300
PATIENCE = 40
STABILITY_WEIGHT = 0.01


def make_blocks():
    pairs, audit = build_pairs()
    saved = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    split_ids = saved["protocol"]["split_person_ids"]
    if sorted(int(x) for ids in split_ids.values() for x in ids) != sorted(map(int, pairs["id"].unique())):
        raise ValueError("Saved participant split does not match current eligible cohort")
    raw = pd.read_csv(SOURCE, sep="\t", usecols=["id", "date", "hour", "minute"] + STATE)
    raw["timestamp"] = (pd.to_datetime(raw["date"])
                        + pd.to_timedelta(raw["hour"], unit="h")
                        + pd.to_timedelta(raw["minute"], unit="m"))
    raw = raw.sort_values(["id", "timestamp"]).reset_index(drop=True)
    if raw.duplicated(["id", "timestamp"]).any():
        raise ValueError("Duplicate participant/time records")
    prefix_cols = []
    for col in STATE:
        observed = raw[col].notna().astype(int)
        total = raw[col].fillna(0).groupby(raw["id"]).cumsum()
        count = observed.groupby(raw["id"]).cumsum()
        name = f"prefix_{col}"
        raw[name] = total / count.replace(0, np.nan)
        prefix_cols.append(name)
    raw["prefix_count"] = raw.groupby("id").cumcount() + 1
    selected = raw[["id", "timestamp", "prefix_count"] + prefix_cols]
    pairs = pairs.merge(selected, on=["id", "timestamp"], validate="one_to_one")
    pairs = pairs.sort_values(["id", "timestamp"]).reset_index(drop=True)
    parts = {name: pairs[pairs["id"].isin(ids)].copy() for name, ids in split_ids.items()}
    imputer = SimpleImputer(strategy="median").fit(parts["train"][STATE].to_numpy())
    scaler = StandardScaler().fit(imputer.transform(parts["train"][STATE].to_numpy()))
    time_scaler = StandardScaler().fit(parts["train"][TIME].to_numpy())
    blocks = {}
    for name in ("train", "dev"):
        part = parts[name]
        current = scaler.transform(imputer.transform(part[STATE].to_numpy())).astype(np.float32)
        history = scaler.transform(imputer.transform(part[prefix_cols].to_numpy())).astype(np.float32)
        time = time_scaler.transform(part[TIME].to_numpy()).astype(np.float32)
        blocks[name] = {
            "anchor": torch.from_numpy(history),
            "deviation": torch.from_numpy(current - history),
            "time": torch.from_numpy(time),
            "y": torch.tensor(part["next_target"].to_numpy(), dtype=torch.float32),
            "id": part["id"].to_numpy(),
            "prefix_count": part["prefix_count"].to_numpy(),
        }
    preproc = {
        "state_columns": STATE, "time_columns": TIME,
        "state_imputer_median": imputer.statistics_.tolist(),
        "state_scaler_mean": scaler.mean_.tolist(),
        "state_scaler_scale": scaler.scale_.tolist(),
        "time_scaler_mean": time_scaler.mean_.tolist(),
        "time_scaler_scale": time_scaler.scale_.tolist(),
        "history_rule": "per-feature mean of all observed prompts through and including prediction time",
    }
    return blocks, audit, split_ids, preproc


def forward(model, block, sample):
    phi, phi_mean, phi_kl = model.draw(model.slow(block["anchor"]), sample)
    z, _, z_kl = model.draw(model.fast(block["deviation"]), sample)
    latent = torch.cat((phi, z, block["time"]), dim=1)
    return model.behavior(latent).squeeze(1), model.decoder(latent), phi_mean, phi_kl, z_kl


def probabilities(model, block):
    model.eval()
    with torch.no_grad():
        logits, _, _, _, _ = forward(model, block, False)
        return torch.sigmoid(logits).numpy()


def fit(blocks, seed):
    torch.manual_seed(seed)
    train, dev = blocks["train"], blocks["dev"]
    model = SlowFastVAE(train["anchor"].shape[1], train["time"].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    target = train["anchor"] + train["deviation"]
    adjacent = np.flatnonzero(train["id"][:-1] == train["id"][1:])
    best_state, best_epoch, best_brier, stale = None, 0, float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, reconstruction, phi_mean, phi_kl, z_kl = forward(model, train, True)
        stability = F.mse_loss(phi_mean[adjacent], phi_mean[adjacent + 1])
        loss = (F.binary_cross_entropy_with_logits(logits, train["y"])
                + 0.05 * F.mse_loss(reconstruction, target)
                + 0.002 * (phi_kl + z_kl) + STABILITY_WEIGHT * stability)
        loss.backward()
        optimizer.step()
        dev_probability = probabilities(model, dev)
        brier = float(np.mean((dev_probability - dev["y"].numpy()) ** 2))
        if brier < best_brier - 1e-7:
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch, best_brier, stale = epoch, brier, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def person_change(model, block):
    """Descriptive change in slow posterior between early/late eligible prefixes."""
    with torch.no_grad():
        phi = model.slow(block["anchor"])[:, :4].numpy()
    change, counts = [], []
    for pid in np.unique(block["id"]):
        rows = np.flatnonzero(block["id"] == pid)
        if len(rows) >= 2:
            change.append(float(np.linalg.norm(phi[rows[-1]] - phi[rows[0]])))
            counts.append((int(block["prefix_count"][rows[0]]),
                           int(block["prefix_count"][rows[-1]])))
    return {"people_with_two_eligible_prefixes": len(change),
            "early_to_late_phi_euclidean_median": float(np.median(change)),
            "prefix_count_early_median": float(np.median([x[0] for x in counts])),
            "prefix_count_late_median": float(np.median([x[1] for x in counts]))}


def main():
    torch.set_num_threads(1)
    blocks, audit, ids, preproc = make_blocks()
    output_dir = OUT / "online_history_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "exploratory variable-prefix personality-method prototype",
              "protocol": {"seeds": SEEDS, "train_people": len(ids["train"]),
                           "dev_people": len(ids["dev"]),
                           "train_pairs": len(blocks["train"]["y"]),
                           "dev_pairs": len(blocks["dev"]["y"]),
                           "max_epochs": EPOCHS, "patience": PATIENCE,
                           "stability_weight": STABILITY_WEIGHT,
                           "history": preproc["history_rule"],
                           "selection": "development Brier; no test data used"},
              "audit": audit, "runs": []}
    for seed in SEEDS:
        model, epoch = fit(blocks, seed)
        checkpoint = output_dir / f"seed_{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "state_dim": len(STATE),
                    "time_dim": len(TIME), "seed": seed, "best_epoch": epoch,
                    "training_contract": "variable-prefix-history",
                    "preprocessing": preproc}, checkpoint)
        p = probabilities(model, blocks["dev"])
        run = {"seed": seed, "best_epoch": epoch,
               "dev": scores(blocks["dev"]["y"].numpy(), p),
               "person_change": person_change(model, blocks["dev"]),
               "checkpoint": str(checkpoint.relative_to(OUT))}
        result["runs"].append(run)
    result["mean_dev_brier"] = float(np.mean([r["dev"]["brier"] for r in result["runs"]]))
    output = OUT / "online_history_dev_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"mean_dev_brier": result["mean_dev_brier"],
                      "per_seed": [r["dev"]["brier"] for r in result["runs"]]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
