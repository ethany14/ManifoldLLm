"""Matched slow-fast vs person-distance-regularized models on PersDyn.

Exploratory deterministic prototype. The geometry term is a supervised
person-distance proxy, not a learned Riemannian metric or behavioral manifold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from run_persdyn_public_pilot import TRAITS, get_source, load_complete_rows, metrics


def prepare(frame: pd.DataFrame, history: int):
    rows = []
    anchors = {}
    future_means = {}
    for person, group in frame.groupby("ppn", sort=True):
        states = group[TRAITS].to_numpy(dtype=np.float32) / 100.0
        times = group["ExecutionTime"].to_numpy(dtype="datetime64[ns]")
        if len(states) <= history:
            continue
        anchor = states[:history].mean(axis=0)
        anchors[int(person)] = anchor
        future_means[int(person)] = states[history:].mean(axis=0)
        for t in range(history, len(states)):
            gap = float((times[t] - times[t - 1]) / np.timedelta64(1, "h"))
            if gap < 0:
                raise ValueError("Negative time gap")
            at = pd.Timestamp(times[t])
            hour = at.hour + at.minute / 60
            context = np.array([np.log1p(gap) / 6,
                                np.sin(2 * np.pi * hour / 24),
                                np.cos(2 * np.pi * hour / 24)], dtype=np.float32)
            rows.append((int(person), anchor, states[t - 1] - anchor, context, states[t]))
    return rows, anchors, future_means


class SlowFast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slow = torch.nn.Sequential(torch.nn.Linear(5, 16), torch.nn.Tanh(),
                                        torch.nn.Linear(16, 4))
        self.fast = torch.nn.Sequential(torch.nn.Linear(5, 16), torch.nn.Tanh(),
                                        torch.nn.Linear(16, 4))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(11, 32), torch.nn.Tanh(),
                                           torch.nn.Linear(32, 5))

    def forward(self, anchor, deviation, context):
        phi = self.slow(anchor)
        z = self.fast(deviation)
        delta = 0.25 * torch.tanh(self.decoder(torch.cat([phi, z, context], dim=-1)))
        return torch.clamp(anchor + delta, 0, 1)


def tensors(rows, person_ids, device):
    selected = [row for row in rows if row[0] in person_ids]
    if not selected:
        raise ValueError("Empty participant split")
    return tuple(torch.tensor(np.stack([row[j] for row in selected]),
                              dtype=torch.float32, device=device) for j in range(1, 5))


def geometry_loss(model, train_people, anchors, future_means, device):
    a = torch.tensor(np.stack([anchors[p] for p in train_people]), device=device)
    target = torch.tensor(np.stack([future_means[p] for p in train_people]), device=device)
    phi = model.slow(a)
    i, j = torch.triu_indices(len(train_people), len(train_people), offset=1, device=device)
    d_phi = torch.linalg.vector_norm(phi[i] - phi[j], dim=1)
    d_target = torch.linalg.vector_norm(target[i] - target[j], dim=1)
    # Scale-normalized pairwise loss is independent of arbitrary latent units.
    return torch.mean(((d_phi / (d_phi.mean().detach() + 1e-6)) -
                       (d_target / (d_target.mean() + 1e-6))) ** 2)


def train_variant(rows, anchors, future_means, splits, model_seed, geometry_weight,
                  epochs, device):
    torch.manual_seed(model_seed)
    model = SlowFast().to(device)
    train = tensors(rows, set(splits["train"]), device)
    dev = tensors(rows, set(splits["dev"]), device)
    test = tensors(rows, set(splits["test"]), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001)
    best_state, best_dev, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(*train[:3])
        loss = torch.mean((prediction - train[3]) ** 2)
        if geometry_weight:
            loss = loss + geometry_weight * geometry_loss(
                model, list(splits["train"]), anchors, future_means, device)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_mse = torch.mean((model(*dev[:3]) - dev[3]) ** 2).item()
        if dev_mse < best_dev - 1e-7:
            best_dev, best_epoch, stale = dev_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 40:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dev_pred = model(*dev[:3]).cpu().numpy()
        test_pred = model(*test[:3]).cpu().numpy()
        test_ids = list(splits["test"])
        test_anchor = torch.tensor(np.stack([anchors[p] for p in test_ids]), device=device)
        test_phi = model.slow(test_anchor).cpu().numpy()
        test_future_mean = np.stack([future_means[p] for p in test_ids])
    pair_i, pair_j = np.triu_indices(len(test_ids), k=1)
    latent_dist = np.linalg.norm(test_phi[pair_i] - test_phi[pair_j], axis=1)
    outcome_dist = np.linalg.norm(test_future_mean[pair_i] - test_future_mean[pair_j], axis=1)
    distance_rho = float(spearmanr(latent_dist, outcome_dist).statistic)
    return {
        "best_epoch": best_epoch,
        "dev": metrics(dev[3].cpu().numpy(), dev_pred),
        "test": metrics(test[3].cpu().numpy(), test_pred),
        "test_person_distance_spearman": round(distance_rho, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=base / "data/persdyn/source/dataES_copy.xlsx")
    parser.add_argument("--output", type=Path, default=base / "artifacts/persdyn_slow_fast/results.json")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--geometry-weight", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    if args.geometry_weight <= 0:
        raise ValueError("geometry-weight must be positive")

    get_source(args.source)
    frame = load_complete_rows(args.source)
    rows, anchors, future_means = prepare(frame, args.history)
    people = np.sort(np.array(list(anchors)))
    train_people, other = train_test_split(people, test_size=0.4, random_state=args.split_seed)
    dev_people, test_people = train_test_split(other, test_size=0.5, random_state=args.split_seed)
    splits = {"train": train_people, "dev": dev_people, "test": test_people}
    linear_x = np.stack([np.r_[row[1], row[2], row[3]] for row in rows])
    linear_y = np.stack([row[4] for row in rows])
    row_people = np.array([row[0] for row in rows])
    linear = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    linear.fit(linear_x[np.isin(row_people, train_people)],
               linear_y[np.isin(row_people, train_people)])
    linear_metrics = {}
    for name in ("dev", "test"):
        mask = np.isin(row_people, splits[name])
        linear_metrics[name] = metrics(linear_y[mask],
                                       np.clip(linear.predict(linear_x[mask]), 0, 1))
    test_ids = list(test_people)
    pair_i, pair_j = np.triu_indices(len(test_ids), k=1)
    raw_anchor = np.stack([anchors[p] for p in test_ids])
    raw_outcome = np.stack([future_means[p] for p in test_ids])
    raw_distance_rho = float(spearmanr(
        np.linalg.norm(raw_anchor[pair_i] - raw_anchor[pair_j], axis=1),
        np.linalg.norm(raw_outcome[pair_i] - raw_outcome[pair_j], axis=1),
    ).statistic)
    pca = PCA(n_components=4).fit(np.stack([anchors[p] for p in train_people]))
    pca_anchor = pca.transform(raw_anchor)
    pca_distance_rho = float(spearmanr(
        np.linalg.norm(pca_anchor[pair_i] - pca_anchor[pair_j], axis=1),
        np.linalg.norm(raw_outcome[pair_i] - raw_outcome[pair_j], axis=1),
    ).statistic)
    # CPU is fast for this small dataset and avoids device-dependent kernels.
    torch.set_num_threads(1)
    device = torch.device("cpu")
    runs = []
    for seed in args.model_seeds:
        flat = train_variant(rows, anchors, future_means, splits, seed, 0.0, args.epochs, device)
        geo = train_variant(rows, anchors, future_means, splits, seed,
                            args.geometry_weight, args.epochs, device)
        runs.append({"model_seed": seed, "no_geometry": flat, "person_distance_geometry": geo})
    summary = {}
    for variant in ("no_geometry", "person_distance_geometry"):
        for split in ("dev", "test"):
            values = [run[variant][split]["mean_mae_0_to_100"] for run in runs]
            summary[f"{variant}_{split}_mae_mean"] = round(float(np.mean(values)), 4)
            summary[f"{variant}_{split}_mae_sd"] = round(float(np.std(values, ddof=1)), 4)
        rho = [run[variant]["test_person_distance_spearman"] for run in runs]
        summary[f"{variant}_test_person_distance_spearman_mean"] = round(float(np.mean(rho)), 4)
    result = {
        "status": "exploratory; deterministic hierarchy and supervised person-distance proxy, not an H-VAE or Riemannian manifold",
        "source_sha256": "3b6c8ba9d02a57ac23e590e5db9b5a20c97c78531ce68e130a203315ac8cdbd1",
        "split_seed": args.split_seed,
        "split_person_ids": {name: [int(p) for p in sorted(ids)] for name, ids in splits.items()},
        "split_example_counts": {name: sum(row[0] in set(ids) for row in rows) for name, ids in splits.items()},
        "model_seeds": args.model_seeds,
        "geometry_weight": args.geometry_weight,
        "geometry_definition": "pairwise Euclidean distances between person phi embeddings fitted to pairwise future-state-mean distances, training participants only",
        "raw_first_ten_mean_test_distance_spearman": round(raw_distance_rho, 4),
        "pca4_first_ten_mean_test_distance_spearman": round(pca_distance_rho, 4),
        "selection": "best epoch by development MSE separately for each model; no hyperparameter search",
        "linear_fixed_history": linear_metrics,
        "summary": summary,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
