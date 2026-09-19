"""Compare latent-to-manifold and directly learned nonlinear representations.

The frozen LLM embeddings are never updated. All preprocessing and representation
models are fitted on the official training split only. The same Ridge probe is
used for every representation so that the comparison concerns the representation,
not the downstream regressor.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import Isomap, trustworthiness
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ["valence", "arousal", "dominance"]


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)


class SupervisedBottleneck(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.head = nn.Linear(latent_dim, len(TARGETS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.head(z)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loaders(x: np.ndarray, y: np.ndarray, train: np.ndarray, dev: np.ndarray, batch: int):
    def make(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)
    return make(train, True), make(dev, False)


def train_autoencoder(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, dev: np.ndarray,
    latent_dim: int, hidden_dim: int, epochs: int, batch: int, device: torch.device,
) -> Autoencoder:
    model = Autoencoder(x.shape[1], latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader, dev_loader = loaders(x, y, train, dev, batch)
    best, best_state, patience = float("inf"), None, 12
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, _ in train_loader:
            xb = xb.to(device)
            _, reconstruction = model(xb)
            loss = nn.functional.mse_loss(reconstruction, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = [nn.functional.mse_loss(model(xb.to(device))[1], xb.to(device)).item()
                      for xb, _ in dev_loader]
        score = float(np.mean(losses))
        if score < best - 1e-6:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model


def local_geometry_loss(z: torch.Tensor, x: torch.Tensor, neighbors: int = 8) -> torch.Tensor:
    """Match normalized distances to local neighbours in frozen-LLM space."""
    with torch.no_grad():
        dx = torch.cdist(x, x)
        k = min(neighbors + 1, x.shape[0])
        indices = dx.topk(k, largest=False).indices[:, 1:]
        mask = torch.zeros_like(dx, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        expected = dx[mask]
        expected = expected / expected.mean().clamp_min(1e-6)
    actual = torch.cdist(z, z)[mask]
    actual = actual / actual.mean().clamp_min(1e-6)
    return nn.functional.mse_loss(actual, expected)


def train_supervised(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, dev: np.ndarray,
    latent_dim: int, hidden_dim: int, epochs: int, batch: int, device: torch.device,
    geometry_weight: float,
) -> SupervisedBottleneck:
    model = SupervisedBottleneck(x.shape[1], latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader, dev_loader = loaders(x, y, train, dev, batch)
    best, best_state, patience = float("inf"), None, 12
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            z, prediction = model(xb)
            loss = nn.functional.mse_loss(prediction, yb)
            loss = loss + geometry_weight * local_geometry_loss(z, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = [nn.functional.mse_loss(model(xb.to(device))[1], yb.to(device)).item()
                      for xb, yb in dev_loader]
        score = float(np.mean(losses))
        if score < best - 1e-6:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model


def encode(model: nn.Module, x: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.from_numpy(x[start:start + batch]).to(device)
            chunks.append(model(xb)[0].cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def distance_correlation(z: np.ndarray, y: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    pairs = min(20000, len(z) * (len(z) - 1) // 2)
    i = rng.integers(0, len(z), pairs)
    j = rng.integers(0, len(z), pairs)
    keep = i != j
    dz = np.linalg.norm(z[i[keep]] - z[j[keep]], axis=1)
    dy = np.linalg.norm(y[i[keep]] - y[j[keep]], axis=1)
    return float(spearmanr(dz, dy).statistic)


def evaluate(name: str, z: np.ndarray, x: np.ndarray, y: np.ndarray,
             train: np.ndarray, test: np.ndarray, seed: int, supervised: bool) -> dict:
    scaler = StandardScaler().fit(z[train])
    train_z, test_z = scaler.transform(z[train]), scaler.transform(z[test])
    probe = Ridge(alpha=10.0).fit(train_z, y[train])
    prediction = probe.predict(test_z)
    row = {"method": name, "supervised_representation": supervised,
           "dimensions": int(z.shape[1])}
    for n, target in enumerate(TARGETS):
        row[f"{target}_r2"] = float(r2_score(y[test, n], prediction[:, n]))
        row[f"{target}_mae"] = float(mean_absolute_error(y[test, n], prediction[:, n]))
        row[f"{target}_rmse"] = float(mean_squared_error(y[test, n], prediction[:, n]) ** 0.5)
    row["mean_vad_r2"] = float(np.mean([row[f"{t}_r2"] for t in TARGETS]))
    k = min(10, max(1, (int(test.sum()) - 1) // 2))
    row["test_trustworthiness"] = float(trustworthiness(x[test], z[test], n_neighbors=k))
    row["test_vad_distance_spearman"] = distance_correlation(z[test], y[test], seed)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    frame = pd.read_csv(args.csv)
    x_raw = np.load(args.embeddings).astype(np.float32)
    if len(frame) != len(x_raw):
        raise ValueError("CSV and embeddings must have identical row counts.")
    if not set(TARGETS).issubset(frame.columns):
        raise ValueError(f"CSV must contain {TARGETS}.")
    y_raw = frame[TARGETS].to_numpy(np.float32)
    train = np.flatnonzero(frame["split"].eq("train").to_numpy())
    dev = np.flatnonzero(frame["split"].eq("dev").to_numpy())
    test = np.flatnonzero(frame["split"].eq("test").to_numpy())
    if min(len(train), len(dev), len(test)) == 0:
        raise ValueError("The CSV needs non-empty train, dev, and test splits.")

    x_scaler = StandardScaler().fit(x_raw[train])
    x = x_scaler.transform(x_raw).astype(np.float32)
    y_scaler = StandardScaler().fit(y_raw[train])
    y_scaled = y_scaler.transform(y_raw).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; train/dev/test={len(train)}/{len(dev)}/{len(test)}")

    representations: list[tuple[str, np.ndarray, bool]] = []
    raw_pca = PCA(n_components=args.dimensions, random_state=args.seed).fit(x[train])
    representations.append(("PCA", raw_pca.transform(x), False))

    iso = Isomap(n_neighbors=args.neighbors, n_components=args.dimensions).fit(x[train])
    iso_all = np.empty((len(x), args.dimensions), dtype=np.float32)
    iso_all[train] = iso.embedding_
    iso_all[dev] = iso.transform(x[dev])
    iso_all[test] = iso.transform(x[test])
    representations.append(("Isomap", iso_all, False))

    ae = train_autoencoder(x, y_scaled, train, dev, args.dimensions, args.hidden_dim,
                           args.epochs, args.batch_size, device)
    ae_z = encode(ae, x, device, args.batch_size)
    representations.append(("Autoencoder", ae_z, False))

    ae_iso = Isomap(n_neighbors=args.neighbors, n_components=args.dimensions).fit(ae_z[train])
    ae_iso_all = np.empty((len(x), args.dimensions), dtype=np.float32)
    ae_iso_all[train] = ae_iso.embedding_
    ae_iso_all[dev] = ae_iso.transform(ae_z[dev])
    ae_iso_all[test] = ae_iso.transform(ae_z[test])
    representations.append(("Autoencoder_to_Isomap", ae_iso_all, False))

    supervised = train_supervised(
        x, y_scaled, train, dev, args.dimensions, args.hidden_dim, args.epochs,
        args.batch_size, device, args.geometry_weight,
    )
    supervised_z = encode(supervised, x, device, args.batch_size)
    representations.append(("Supervised_nonlinear", supervised_z, True))

    # Raw embeddings are intentionally not dimension matched; they are the information ceiling.
    rows = [evaluate("Raw_Qwen", x, x, y_raw, train, test, args.seed, False)]
    rows.extend(evaluate(name, z, x, y_raw, train, test, args.seed, sup)
                for name, z, sup in representations)
    results = pd.DataFrame(rows).sort_values("mean_vad_r2", ascending=False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "representation_comparison.csv", index=False)
    summary = {
        "design": {
            "csv": str(Path(args.csv).resolve()),
            "embeddings": str(Path(args.embeddings).resolve()),
            "dimensions": args.dimensions,
            "neighbors": args.neighbors,
            "seed": args.seed,
            "train_rows": len(train), "dev_rows": len(dev), "test_rows": len(test),
            "note": "Supervised_nonlinear uses VAD labels; all other compressed representations are unsupervised.",
        },
        "results": results.to_dict(orient="records"),
    }
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        plot = results.set_index("method")[[f"{t}_r2" for t in TARGETS]]
        ax = plot.plot(kind="bar", figsize=(11, 5), color=["#2F6690", "#F2A65A", "#5B8E7D"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Held-out test R²")
        ax.set_xlabel("")
        ax.set_title("Frozen-Qwen affect representation comparison")
        ax.legend(["Valence", "Arousal", "Dominance"])
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output / "representation_comparison.png", dpi=180)
        plt.close()
    except ImportError:
        print("matplotlib is not installed; skipped the optional PNG chart.")
    print(results.to_string(index=False))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
