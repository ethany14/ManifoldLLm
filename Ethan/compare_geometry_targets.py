"""Compare semantic, affective, and hybrid geometry objectives on AE+VAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from compare_representation_routes import TARGETS, evaluate, seed_everything
from run_ablation_study import AblationModel, encode


def normalized_local_distance_loss(
    z: torch.Tensor, reference: torch.Tensor, neighbors: int = 8
) -> torch.Tensor:
    """Match latent distances to local distances defined by a reference space."""
    if reference.shape[0] < 2:
        return z.new_zeros(())
    with torch.no_grad():
        reference_dist = torch.cdist(reference, reference)
        k = min(neighbors + 1, reference.shape[0])
        indices = reference_dist.topk(k, largest=False).indices[:, 1:]
        mask = torch.zeros_like(reference_dist, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        expected = reference_dist[mask]
        expected = expected / expected.mean().clamp_min(1e-6)
    actual = torch.cdist(z, z)[mask]
    actual = actual / actual.mean().clamp_min(1e-6)
    return nn.functional.mse_loss(actual, expected)


def make_loader(x: np.ndarray, y: np.ndarray, idx: np.ndarray,
                batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    reconstruction_weight: float,
    geometry_weight: float,
    geometry_target: str,
    hybrid_alpha: float,
) -> AblationModel:
    model = AblationModel(x.shape[1], latent_dim, hidden_dim, True, True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = make_loader(x, y, train_idx, batch_size, True)
    dev_loader = make_loader(x, y, dev_idx, batch_size, False)
    best, best_state, stale = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            z, prediction, reconstruction = model(xb)
            loss = nn.functional.mse_loss(prediction, yb)
            loss = loss + reconstruction_weight * nn.functional.mse_loss(reconstruction, xb)
            if geometry_target == "semantic":
                geometry = normalized_local_distance_loss(z, xb)
            elif geometry_target == "affective":
                geometry = normalized_local_distance_loss(z, yb)
            elif geometry_target == "hybrid":
                semantic = normalized_local_distance_loss(z, xb)
                affective = normalized_local_distance_loss(z, yb)
                geometry = hybrid_alpha * semantic + (1.0 - hybrid_alpha) * affective
            elif geometry_target == "none":
                geometry = z.new_zeros(())
            else:
                raise ValueError(f"Unknown geometry target: {geometry_target}")
            loss = loss + geometry_weight * geometry
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Use the same validation VAD criterion for every geometry condition.
        model.eval()
        with torch.no_grad():
            values = []
            for xb, yb in dev_loader:
                _, prediction, _ = model(xb.to(device))
                values.append(nn.functional.mse_loss(prediction, yb.to(device)).item())
        score = float(np.mean(values))
        if score < best - 1e-6:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 12:
                break
    model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    args = parser.parse_args()
    seed_everything(args.seed)

    frame = pd.read_csv(args.csv)
    raw = np.load(args.embeddings).astype(np.float32)
    if len(frame) != len(raw):
        raise ValueError("CSV and embeddings must have identical row counts.")
    train = np.flatnonzero(frame["split"].eq("train").to_numpy())
    dev = np.flatnonzero(frame["split"].eq("dev").to_numpy())
    test = np.flatnonzero(frame["split"].eq("test").to_numpy())
    x = StandardScaler().fit(raw[train]).transform(raw).astype(np.float32)
    y_raw = frame[TARGETS].to_numpy(np.float32)
    y = StandardScaler().fit(y_raw[train]).transform(y_raw).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; train/dev/test={len(train)}/{len(dev)}/{len(test)}")

    rows = []
    for target in ["none", "semantic", "affective", "hybrid"]:
        seed_everything(args.seed)
        print(f"training geometry_target={target}")
        model = train_model(
            x, y, train, dev, args.dimensions, args.hidden_dim, args.epochs,
            args.batch_size, device, args.reconstruction_weight,
            args.geometry_weight, target, args.hybrid_alpha,
        )
        z = encode(model, x, device, args.batch_size)
        row = evaluate(f"AE_VAD_{target}", z, x, y_raw, train, test, args.seed, True)
        row["geometry_target"] = target
        row["geometry_weight"] = 0.0 if target == "none" else args.geometry_weight
        row["hybrid_alpha"] = args.hybrid_alpha if target == "hybrid" else None
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("mean_vad_r2", ascending=False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "geometry_target_results.csv", index=False)
    summary = {
        "design": {
            "dimensions": args.dimensions,
            "seed": args.seed,
            "geometry_weight": args.geometry_weight,
            "reconstruction_weight": args.reconstruction_weight,
            "hybrid_alpha": args.hybrid_alpha,
            "train_rows": len(train), "dev_rows": len(dev), "test_rows": len(test),
            "validation_criterion": "VAD MSE for every condition",
        },
        "results": results.to_dict(orient="records"),
    }
    (output / "geometry_target_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
