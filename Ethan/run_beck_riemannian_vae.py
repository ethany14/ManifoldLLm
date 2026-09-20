"""Development-only slow/fast VAE with a decoder-induced person metric.

This reuses the existing Beck participant split and prediction task. It does not
touch test outcomes. Euclidean and decoder-metric readouts share a single
trained model, graph topology, labels, and fixed interpolation weight.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

from run_beck_behavior_pilot import OUT, scores
from run_beck_graph_geometry import STATE, TIME, inputs


SEEDS = (1, 2, 3)
MAX_EPOCHS = 300
PATIENCE = 40
K = 8
MIX = 0.2
PATH_STEPS = 8
RECON_WEIGHT = 0.05
KL_WEIGHT = 0.002


class SlowFastVAE(torch.nn.Module):
    def __init__(self, state_dim: int, time_dim: int):
        super().__init__()
        self.slow = torch.nn.Sequential(torch.nn.Linear(state_dim, 32), torch.nn.Tanh(),
                                        torch.nn.Linear(32, 8))
        self.fast = torch.nn.Sequential(torch.nn.Linear(state_dim, 32), torch.nn.Tanh(),
                                        torch.nn.Linear(32, 8))
        combined = 8 + time_dim
        self.decoder = torch.nn.Sequential(torch.nn.Linear(combined, 32), torch.nn.Tanh(),
                                           torch.nn.Linear(32, state_dim))
        self.behavior = torch.nn.Sequential(torch.nn.Linear(combined, 32), torch.nn.Tanh(),
                                            torch.nn.Linear(32, 1))
        self.time_dim = time_dim

    @staticmethod
    def draw(stats: torch.Tensor, sample: bool):
        mean, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-8, 5)
        value = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mean.square() - logvar.exp(), dim=-1))
        return value, mean, kl

    def forward(self, unique_anchors, row_to_person, deviations, time, sample: bool):
        phi, phi_mean, phi_kl = self.draw(self.slow(unique_anchors), sample)
        z, _, z_kl = self.draw(self.fast(deviations), sample)
        latent = torch.cat((phi[row_to_person], z, time), dim=1)
        return (self.behavior(latent).squeeze(1), self.decoder(latent),
                phi_mean, phi_kl, z_kl)

    def decode_person(self, phi):
        zeros = phi.new_zeros((len(phi), 4 + self.time_dim))
        return self.decoder(torch.cat((phi, zeros), dim=1))


def person_view(block):
    ids, first, inverse = np.unique(block["id"], return_index=True, return_inverse=True)
    return ids, block["anchor"][first], torch.as_tensor(inverse, dtype=torch.long)


def predict(model, block):
    ids, anchors, inverse = person_view(block)
    model.eval()
    with torch.no_grad():
        logits, _, _, _, _ = model(anchors, inverse, block["deviation"], block["time"], False)
        return torch.sigmoid(logits).numpy()


def fit(blocks, seed):
    torch.manual_seed(seed)
    train, dev = blocks["train"], blocks["dev"]
    ids, anchors, inverse = person_view(train)
    model = SlowFastVAE(train["anchor"].shape[1], train["time"].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    best_state, best_epoch, best_score, stale = None, 0, float("inf"), 0
    target = train["anchor"] + train["deviation"]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, reconstruction, _, phi_kl, z_kl = model(
            anchors, inverse, train["deviation"], train["time"], True)
        loss = (F.binary_cross_entropy_with_logits(logits, train["y"])
                + RECON_WEIGHT * F.mse_loss(reconstruction, target)
                + KL_WEIGHT * (phi_kl + z_kl))
        loss.backward()
        optimizer.step()
        dev_prob = predict(model, dev)
        dev_brier = float(np.mean((dev_prob - dev["y"].numpy()) ** 2))
        if dev_brier < best_score - 1e-7:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_score, best_epoch, stale = dev_brier, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("No valid training checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def path_length(model, start, end, metric):
    if metric == "euclidean":
        return float(np.linalg.norm(start - end))
    fractions = torch.linspace(0, 1, PATH_STEPS + 1)
    path = (torch.as_tensor(start, dtype=torch.float32)[None] * (1 - fractions[:, None])
            + torch.as_tensor(end, dtype=torch.float32)[None] * fractions[:, None])
    with torch.no_grad():
        decoded = model.decode_person(path).numpy()
    return float(np.linalg.norm(np.diff(decoded, axis=0), axis=1).sum())


def graph_rates(model, train_phi, dev_phi, train_rates, metric):
    all_phi = np.vstack((train_phi, dev_phi))
    n_train, n_dev = len(train_phi), len(dev_phi)
    links = set()
    train_neighbors = NearestNeighbors(n_neighbors=K + 1).fit(train_phi)
    for i, row in enumerate(train_neighbors.kneighbors(train_phi, return_distance=False)):
        links.update(tuple(sorted((i, int(j)))) for j in row if i != j)
    for q, row in enumerate(NearestNeighbors(n_neighbors=K).fit(train_phi).kneighbors(
            dev_phi, return_distance=False)):
        links.update((int(j), n_train + q) for j in row)
    rr, cc, vv = [], [], []
    for i, j in sorted(links):
        d = path_length(model, all_phi[i], all_phi[j], metric)
        if not np.isfinite(d) or d <= 0:
            raise ValueError("Invalid graph edge length")
        rr.extend((i, j)); cc.extend((j, i)); vv.extend((d, d))
    graph = csr_matrix((vv, (rr, cc)), shape=(n_train + n_dev, n_train + n_dev))
    distances = shortest_path(graph, directed=False, indices=np.arange(n_train, n_train + n_dev))[:, :n_train]
    if not np.isfinite(distances).all():
        raise ValueError("Disconnected person graph")
    nearest = np.argsort(distances, axis=1)[:, :K]
    selected = np.take_along_axis(distances, nearest, axis=1)
    edge_scale = np.median(vv)
    weights = np.exp(-(selected - selected[:, :1]) / max(edge_scale, 1e-6))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(weights * train_rates[nearest], axis=1)


def metric_diagnostics(model, phi):
    singular = []
    for point in phi[:min(30, len(phi))]:
        p = torch.tensor(point, dtype=torch.float32, requires_grad=True)
        jac = torch.autograd.functional.jacobian(lambda x: model.decode_person(x[None])[0], p)
        singular.append(torch.linalg.svdvals(jac).detach().numpy())
    values = np.stack(singular)
    return {"n_people_checked": len(values),
            "min_singular_median": float(np.median(values[:, -1])),
            "condition_median": float(np.median(values[:, 0] / np.maximum(values[:, -1], 1e-8)))}


def main():
    torch.set_num_threads(1)
    blocks, _, audit, split_ids = inputs(include_test=False)
    result = {
        "status": "exploratory development-only slow/fast VAE and metric-readout prototype",
        "limitations": [
            "Beck outcome is next-prompt self-reported studying, not independent behavior.",
            "The existing participant split has been inspected during previous development.",
            "Geometry enters the neighbor readout after joint VAE training; this is not an end-to-end geometry-aware M1 training objective.",
            "Graph topology is Euclidean kNN; only edge lengths differ.",
            "One fixed reference state/time (zero after standardization) defines the metric; sensitivity is untested.",
        ],
        "protocol": {"seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                     "train_people": len(split_ids["train"]), "dev_people": len(split_ids["dev"]),
                     "train_pairs": len(blocks["train"]["y"]), "dev_pairs": len(blocks["dev"]["y"]),
                     "state_columns": STATE, "time_columns": TIME, "neighbors": K,
                     "readout_mix": MIX, "path_steps": PATH_STEPS,
                     "reconstruction_weight": RECON_WEIGHT, "kl_weight": KL_WEIGHT,
                     "selection": "base predictor development Brier; identical checkpoint for both readouts"},
        "audit": audit, "runs": []}
    for seed in SEEDS:
        model, best_epoch = fit(blocks, seed)
        checkpoint_dir = OUT / "riemannian_vae_models"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"seed_{seed}.pt"
        torch.save({"state_dict": model.state_dict(),
                    "state_dim": blocks["train"]["anchor"].shape[1],
                    "time_dim": blocks["train"]["time"].shape[1],
                    "seed": seed, "best_epoch": best_epoch,
                    "input_contract": "standardized anchor, deviation and time from Beck inputs()"},
                   checkpoint_path)
        train_ids, train_anchors, _ = person_view(blocks["train"])
        dev_ids, dev_anchors, _ = person_view(blocks["dev"])
        with torch.no_grad():
            train_phi = model.slow(train_anchors)[:, :4].numpy()
            dev_phi = model.slow(dev_anchors)[:, :4].numpy()
        train_y = blocks["train"]["y"].numpy()
        train_people = blocks["train"]["id"]
        rates = np.array([train_y[train_people == pid].mean() for pid in train_ids])
        euclidean_rates = graph_rates(model, train_phi, dev_phi, rates, "euclidean")
        riemannian_rates = graph_rates(model, train_phi, dev_phi, rates, "riemannian")
        row_of = {int(pid): i for i, pid in enumerate(dev_ids)}
        row = np.array([row_of[int(pid)] for pid in blocks["dev"]["id"]])
        y = blocks["dev"]["y"].numpy()
        base = predict(model, blocks["dev"])
        euclidean = (1 - MIX) * base + MIX * euclidean_rates[row]
        riemannian = (1 - MIX) * base + MIX * riemannian_rates[row]
        result["runs"].append({
            "seed": seed, "best_epoch": best_epoch,
            "checkpoint": str(checkpoint_path.relative_to(OUT)),
            "base_vae": scores(y, base),
            "euclidean_readout": scores(y, euclidean),
            "riemannian_readout": scores(y, riemannian),
            "delta_riemannian_minus_euclidean_brier": float(np.mean(
                (riemannian - y) ** 2 - (euclidean - y) ** 2)),
            "person_readout_mean_absolute_difference": float(np.mean(
                np.abs(riemannian_rates - euclidean_rates))),
            "metric_diagnostics": metric_diagnostics(model, train_phi)})
    result["mean_dev_brier"] = {key: float(np.mean([r[key]["brier"] for r in result["runs"]]))
                                for key in ("base_vae", "euclidean_readout", "riemannian_readout")}
    output = OUT / "riemannian_vae_dev_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean_dev_brier"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
