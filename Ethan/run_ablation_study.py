"""Ablate supervision, nonlinearity, reconstruction, and geometry constraints."""

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

from compare_representation_routes import (
    TARGETS,
    evaluate,
    local_geometry_loss,
    seed_everything,
)


class AblationModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int,
                 nonlinear: bool, reconstruct: bool) -> None:
        super().__init__()
        if nonlinear:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, latent_dim),
            )
        else:
            self.encoder = nn.Linear(input_dim, latent_dim)
        self.affect_head = nn.Linear(latent_dim, len(TARGETS))
        self.decoder = (
            nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, input_dim))
            if reconstruct else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z = self.encoder(x)
        return z, self.affect_head(z), self.decoder(z) if self.decoder is not None else None


def make_loader(x: np.ndarray, y: np.ndarray, idx: np.ndarray,
                batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_variant(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    nonlinear: bool,
    reconstruction_weight: float,
    geometry_weight: float,
) -> AblationModel:
    model = AblationModel(
        x.shape[1], latent_dim, hidden_dim, nonlinear,
        reconstruct=reconstruction_weight > 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = make_loader(x, y, train_idx, batch_size, True)
    dev_loader = make_loader(x, y, dev_idx, batch_size, False)
    best = float("inf")
    best_state = None
    stale = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            z, prediction, reconstruction = model(xb)
            loss = nn.functional.mse_loss(prediction, yb)
            if reconstruction is not None:
                loss = loss + reconstruction_weight * nn.functional.mse_loss(reconstruction, xb)
            if geometry_weight > 0:
                loss = loss + geometry_weight * local_geometry_loss(z, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            dev_losses = []
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, prediction, _ = model(xb)
                loss = nn.functional.mse_loss(prediction, yb)
                dev_losses.append(loss.item())
        score = float(np.mean(dev_losses))
        if score < best - 1e-6:
            best = score
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 12:
                break

    model.load_state_dict(best_state)
    return model


def encode(model: AblationModel, x: np.ndarray, device: torch.device,
           batch_size: int) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            result.append(model(xb)[0].cpu().numpy())
    return np.concatenate(result).astype(np.float32)


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
    args = parser.parse_args()
    seed_everything(args.seed)

    frame = pd.read_csv(args.csv)
    raw = np.load(args.embeddings).astype(np.float32)
    if len(frame) != len(raw):
        raise ValueError("CSV and embeddings must have identical row counts.")
    train = np.flatnonzero(frame["split"].eq("train").to_numpy())
    dev = np.flatnonzero(frame["split"].eq("dev").to_numpy())
    test = np.flatnonzero(frame["split"].eq("test").to_numpy())
    x_scaler = StandardScaler().fit(raw[train])
    x = x_scaler.transform(raw).astype(np.float32)
    y_raw = frame[TARGETS].to_numpy(np.float32)
    y_scaler = StandardScaler().fit(y_raw[train])
    y = y_scaler.transform(y_raw).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; train/dev/test={len(train)}/{len(dev)}/{len(test)}")

    variants = [
        ("Supervised_linear", False, 0.0, 0.0),
        ("Supervised_nonlinear", True, 0.0, 0.0),
        ("Supervised_nonlinear_geometry", True, 0.0, args.geometry_weight),
        ("AE_VAD", True, args.reconstruction_weight, 0.0),
        ("AE_VAD_geometry", True, args.reconstruction_weight, args.geometry_weight),
    ]
    rows = []
    for name, nonlinear, reconstruction_weight, geometry_weight in variants:
        seed_everything(args.seed)
        print(f"training {name}")
        model = train_variant(
            x, y, train, dev, args.dimensions, args.hidden_dim, args.epochs,
            args.batch_size, device, nonlinear, reconstruction_weight, geometry_weight,
        )
        z = encode(model, x, device, args.batch_size)
        row = evaluate(name, z, x, y_raw, train, test, args.seed, True)
        row["nonlinear"] = nonlinear
        row["reconstruction_weight"] = reconstruction_weight
        row["geometry_weight"] = geometry_weight
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("mean_vad_r2", ascending=False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "ablation_results.csv", index=False)
    summary = {
        "design": {
            "dimensions": args.dimensions,
            "seed": args.seed,
            "train_rows": len(train), "dev_rows": len(dev), "test_rows": len(test),
            "common_probe": "Ridge(alpha=10)",
            "interpretation": [
                "nonlinear > linear tests nonlinear capacity",
                "+geometry > no geometry tests incremental manifold constraint value",
                "+reconstruction > no reconstruction tests semantic-retention value",
            ],
        },
        "results": results.to_dict(orient="records"),
    }
    (output / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
