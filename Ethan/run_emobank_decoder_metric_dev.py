"""Full-EmoBank sentence-level VAE and decoder-path distance comparison.

Uses the existing frozen Qwen embeddings and official train/dev split.
Development-only: the official test split is not scored here. This is an
affect-representation experiment, not a person-level digital personality test.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
CSV = HERE / "data/emobank/emobank_manifold.csv"
EMBEDDINGS = HERE / "artifacts/emobank_full/embeddings.npy"
OUTPUT = HERE / "artifacts/emobank_full/decoder_metric_dev_results.json"
TARGETS = ["valence", "arousal", "dominance"]
SEEDS = (1, 2, 3)
LATENT = 16
HIDDEN = 256
BATCH = 128
MAX_EPOCHS = 60
PATIENCE = 10
K = 8
CANDIDATES = 32
PATH_STEPS = 5


class TextVAE(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, HIDDEN), nn.GELU(),
                                     nn.Linear(HIDDEN, 2 * LATENT))
        self.decoder = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.GELU(),
                                     nn.Linear(HIDDEN, input_dim))
        self.affect = nn.Linear(LATENT, 3)

    def forward(self, x, sample):
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        logvar = logvar.clamp(-8, 5)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mean.square() - logvar.exp(), dim=1))
        return mean, self.affect(z), self.decoder(z), kl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(x, y, train, dev, seed, device):
    set_seed(seed)
    model = TextVAE(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    ds = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train]))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    dev_x = torch.from_numpy(x[dev]).to(device)
    dev_y = torch.from_numpy(y[dev]).to(device)
    best_state, best_epoch, best_score, stale = None, 0, float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            _, pred, recon, kl = model(xb, True)
            loss = F.mse_loss(pred, yb) + 0.1 * F.mse_loss(recon, xb) + 0.001 * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(F.mse_loss(model(dev_x, False)[1], dev_y).item())
        if score < best_score - 1e-6:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def encode(model, x, device):
    z, pred = [], []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            mean, affect, _, _ = model(torch.from_numpy(x[start:start + 512]).to(device), False)
            z.append(mean.cpu().numpy())
            pred.append(affect.cpu().numpy())
    return np.concatenate(z), np.concatenate(pred)


def decoded_lengths(model, starts, ends, device):
    """Straight latent-path lengths in decoder output space, not global geodesics."""
    values = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), 256):
            a = torch.from_numpy(starts[offset:offset + 256]).to(device)
            b = torch.from_numpy(ends[offset:offset + 256]).to(device)
            fractions = torch.linspace(0, 1, PATH_STEPS + 1, device=device)
            path = a[:, None] * (1 - fractions[None, :, None]) + b[:, None] * fractions[None, :, None]
            decoded = model.decoder(path.reshape(-1, LATENT)).reshape(len(a), PATH_STEPS + 1, -1)
            length = torch.linalg.vector_norm(decoded[:, 1:] - decoded[:, :-1], dim=2).sum(1)
            values.append(length.cpu().numpy())
    return np.concatenate(values)


def neighbor_prediction(distances, candidates, y_train):
    order = np.argsort(distances, axis=1)[:, :K]
    idx = np.take_along_axis(candidates, order, axis=1)
    d = np.take_along_axis(distances, order, axis=1)
    scale = np.median(d, axis=1, keepdims=True).clip(1e-6)
    weights = np.exp(-(d - d[:, :1]) / scale)
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights[:, :, None] * y_train[idx]).sum(axis=1)


def evaluate(y, prediction):
    return {"mean_r2": float(np.mean([r2_score(y[:, j], prediction[:, j]) for j in range(3)])),
            "r2": {name: float(r2_score(y[:, j], prediction[:, j])) for j, name in enumerate(TARGETS)},
            "mean_mae": float(np.mean([mean_absolute_error(y[:, j], prediction[:, j]) for j in range(3)]))}


def metric_diagnostics(model, points, device):
    """Check local decoder Jacobian rank at a few training representations."""
    smallest, condition = [], []
    for point in points[:5]:
        p = torch.as_tensor(point, dtype=torch.float32, device=device)
        jac = torch.autograd.functional.jacobian(
            lambda v: model.decoder(v[None])[0], p,
            vectorize=True, strategy="forward-mode")
        singular = torch.linalg.svdvals(jac).detach().cpu().numpy()
        smallest.append(float(singular[-1]))
        condition.append(float(singular[0] / max(singular[-1], 1e-8)))
    return {"training_points_checked": len(smallest),
            "smallest_singular_median": float(np.median(smallest)),
            "condition_number_median": float(np.median(condition))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    frame = pd.read_csv(CSV)
    raw = np.load(EMBEDDINGS).astype(np.float32)
    if len(frame) != len(raw):
        raise ValueError("CSV/embedding row mismatch")
    train = np.flatnonzero(frame["split"].eq("train").to_numpy())
    dev = np.flatnonzero(frame["split"].eq("dev").to_numpy())
    if len(train) != 8062 or len(dev) != 999:
        raise ValueError("Unexpected official EmoBank split counts")
    x = StandardScaler().fit(raw[train]).transform(raw).astype(np.float32)
    y_raw = frame[TARGETS].to_numpy(np.float32)
    y_scaler = StandardScaler().fit(y_raw[train])
    y = y_scaler.transform(y_raw).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"status": "exploratory development-only; sentence affect, not personality",
              "protocol": {"seeds": args.seeds, "train_rows": len(train), "dev_rows": len(dev),
                           "latent_dim": LATENT, "candidate_neighbors": CANDIDATES,
                           "readout_neighbors": K, "path_steps": PATH_STEPS,
                           "selection": "direct-head development VAD MSE",
                           "geometry": "decoder-output length of straight latent paths; not optimized global geodesic"},
              "runs": []}
    for seed in args.seeds:
        model, epoch = train_model(x, y, train, dev, seed, device)
        z, pred = encode(model, x, device)
        z_scaler = StandardScaler().fit(z[train])
        z_scaled = z_scaler.transform(z).astype(np.float32)
        d_e, candidate = NearestNeighbors(n_neighbors=CANDIDATES).fit(
            z_scaled[train]).kneighbors(z_scaled[dev])
        starts = np.repeat(z[dev], CANDIDATES, axis=0)
        ends = z[train][candidate.reshape(-1)]
        d_g = decoded_lengths(model, starts, ends, device).reshape(len(dev), CANDIDATES)
        euclid = neighbor_prediction(d_e, candidate, y_raw[train])
        metric = neighbor_prediction(d_g, candidate, y_raw[train])
        direct = y_scaler.inverse_transform(pred[dev])
        result["runs"].append({"seed": seed, "best_epoch": epoch,
                               "direct_head": evaluate(y_raw[dev], direct),
                               "euclidean_knn": evaluate(y_raw[dev], euclid),
                               "decoder_path_knn": evaluate(y_raw[dev], metric),
                               "metric_diagnostics": metric_diagnostics(model, z[train], device),
                               "mean_absolute_prediction_change": float(np.mean(np.abs(metric - euclid)))})
        print(f"seed={seed} direct={result['runs'][-1]['direct_head']['mean_r2']:.4f} "
              f"euclidean={result['runs'][-1]['euclidean_knn']['mean_r2']:.4f} "
              f"decoder_path={result['runs'][-1]['decoder_path_knn']['mean_r2']:.4f}", flush=True)
    result["mean_dev_r2"] = {name: float(np.mean([run[name]["mean_r2"] for run in result["runs"]]))
                             for name in ("direct_head", "euclidean_knn", "decoder_path_knn")}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean_dev_r2"], indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
