"""Development-only decoder-metric engineering check for the M1 proposal.

This is a deterministic B2/M1 proxy, not the proposal's full hierarchical VAE.
Held-out test outcomes are not used for fitting, selection, or reported scores.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import brier_score_loss
from sklearn.neighbors import NearestNeighbors

from run_beck_behavior_pilot import OUT, scores
from run_beck_graph_geometry import STATE, TIME, inputs


SEEDS = [1, 2, 3]
EPOCHS = 250
PATIENCE = 30
K = 8
MIX = 0.2
STEPS = 6


class Model(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.slow = torch.nn.Sequential(torch.nn.Linear(state_dim, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
        self.fast = torch.nn.Sequential(torch.nn.Linear(state_dim, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.Tanh(), torch.nn.Linear(32, state_dim))
        self.behavior = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))

    def forward(self, anchor, deviation, time):
        phi = self.slow(anchor)
        z = self.fast(deviation)
        latent = torch.cat([phi, z, time], dim=1)
        return self.behavior(latent).squeeze(1), self.decoder(latent)

    def decode_person(self, phi):
        zeros = torch.zeros((len(phi), 8), dtype=phi.dtype, device=phi.device)
        return self.decoder(torch.cat([phi, zeros], dim=1))


def person_table(block):
    ids = np.unique(block["id"])
    anchors = torch.stack([block["anchor"][np.flatnonzero(block["id"] == i)[0]] for i in ids])
    rates = np.array([block["y"].numpy()[block["id"] == i].mean() for i in ids])
    return ids, anchors, rates


def probabilities(model, block):
    model.eval()
    with torch.no_grad():
        logits, _ = model(block["anchor"], block["deviation"], block["time"])
        return torch.sigmoid(logits).numpy()


def train(blocks, seed):
    torch.manual_seed(seed)
    train_block, dev = blocks["train"], blocks["dev"]
    model = Model(train_block["anchor"].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001)
    target = train_block["anchor"] + train_block["deviation"]
    best, best_epoch, best_dev, stale = None, 0, float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits, reconstruction = model(train_block["anchor"], train_block["deviation"], train_block["time"])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_block["y"])
        loss = loss + 0.05 * torch.mean((reconstruction - target) ** 2)
        loss.backward()
        optimizer.step()
        dev_p = probabilities(model, dev)
        dev_brier = brier_score_loss(dev["y"].numpy(), dev_p)
        if dev_brier < best_dev - 1e-7:
            best_dev, best_epoch, stale = dev_brier, epoch, 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best)
    model.eval()
    return model, best_epoch


def path_length(model, start, end):
    """Length of a straight latent segment measured after nonlinear decoding."""
    fractions = torch.linspace(0, 1, STEPS + 1)
    path = torch.from_numpy(start.astype("float32"))[None] * (1 - fractions[:, None])
    path = path + torch.from_numpy(end.astype("float32"))[None] * fractions[:, None]
    with torch.no_grad():
        decoded = model.decode_person(path).numpy()
    return float(np.linalg.norm(np.diff(decoded, axis=0), axis=1).sum())


def metric_diagnostics(model, phi):
    singular = []
    for point in phi[: min(30, len(phi))]:
        p = torch.tensor(point, dtype=torch.float32, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(lambda x: model.decode_person(x[None])[0], p)
        singular.append(torch.linalg.svdvals(jacobian).detach().numpy())
    values = np.stack(singular)
    return {"sampled_people": len(values), "smallest_singular_median": float(np.median(values[:, -1])),
            "condition_number_median": float(np.median(values[:, 0] / np.maximum(values[:, -1], 1e-8)))}


def neighbor_scores(model, train_phi, dev_phi, train_rates, geometry):
    all_phi = np.vstack([train_phi, dev_phi])
    n_train, n_dev = len(train_phi), len(dev_phi)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(train_phi)
    _, neighbors = nn.kneighbors(train_phi)
    edges = {}
    for i, row in enumerate(neighbors):
        for j in row:
            if i != j:
                a, b = sorted((i, int(j)))
                edges[(a, b)] = None
    query_neighbors = NearestNeighbors(n_neighbors=K).fit(train_phi).kneighbors(dev_phi, return_distance=False)
    for q, row in enumerate(query_neighbors):
        for j in row:
            edges[(int(j), n_train + q)] = None
    rows, cols, vals = [], [], []
    for i, j in edges:
        value = path_length(model, all_phi[i], all_phi[j]) if geometry else float(np.linalg.norm(all_phi[i] - all_phi[j]))
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Non-positive or nonfinite graph edge")
        rows.extend([i, j]); cols.extend([j, i]); vals.extend([value, value])
    graph = csr_matrix((vals, (rows, cols)), shape=(len(all_phi), len(all_phi)))
    distances = shortest_path(graph, directed=False, indices=np.arange(n_train, len(all_phi)))[:, :n_train]
    if not np.isfinite(distances).all():
        raise ValueError("Disconnected person graph")
    neighbor = np.argsort(distances, axis=1)[:, :K]
    selected = np.take_along_axis(distances, neighbor, axis=1)
    scale = np.median(np.array(vals))
    weights = np.exp(-(selected - selected[:, :1]) / max(scale, 1e-6))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(weights * train_rates[neighbor], axis=1)


def main():
    torch.set_num_threads(1)
    blocks, _, audit, split_ids = inputs(include_test=False)
    results = {"status": "development-only deterministic engineering proxy; not full B1/B2/M1 VAE", "config":
               {"seeds": SEEDS, "k": K, "mix": MIX, "path_subdivisions": STEPS,
                "state_columns": STATE, "time_columns": TIME, "dev_people": split_ids["dev"],
                "geometry": "decoder-induced lengths on kNN edges; shortest-path approximation"},
               "audit": {"train_people": len(split_ids["train"]), "dev_people": len(split_ids["dev"]),
                         "train_pairs": len(blocks["train"]["y"]), "dev_pairs": len(blocks["dev"]["y"])}, "runs": []}
    for seed in SEEDS:
        model, epoch = train(blocks, seed)
        train_ids, train_anchor, train_rates = person_table(blocks["train"])
        dev_ids, dev_anchor, _ = person_table(blocks["dev"])
        with torch.no_grad():
            train_phi = model.slow(train_anchor).numpy()
            dev_phi = model.slow(dev_anchor).numpy()
        rank = metric_diagnostics(model, train_phi)
        euclid_rate = neighbor_scores(model, train_phi, dev_phi, train_rates, geometry=False)
        manifold_rate = neighbor_scores(model, train_phi, dev_phi, train_rates, geometry=True)
        person_to_row = {int(pid): idx for idx, pid in enumerate(dev_ids)}
        row_index = np.array([person_to_row[int(pid)] for pid in blocks["dev"]["id"]])
        base = probabilities(model, blocks["dev"])
        y = blocks["dev"]["y"].numpy()
        euclid = (1 - MIX) * base + MIX * euclid_rate[row_index]
        manifold = (1 - MIX) * base + MIX * manifold_rate[row_index]
        results["runs"].append({"seed": seed, "best_epoch": epoch, "metric_diagnostics": rank,
                                "base_b2_proxy": scores(y, base), "euclidean_graph_readout": scores(y, euclid),
                                "decoder_metric_geodesic_readout": scores(y, manifold),
                                "dev_brier_delta_manifold_minus_euclidean": float(np.mean((manifold - y) ** 2 - (euclid - y) ** 2)),
                                "person_rate_readout_mean_absolute_difference": float(np.mean(np.abs(manifold_rate - euclid_rate)))})
    results["mean_dev_brier"] = {key: float(np.mean([r[key]["brier"] for r in results["runs"]]))
                                 for key in ["base_b2_proxy", "euclidean_graph_readout", "decoder_metric_geodesic_readout"]}
    output = OUT / "decoder_metric_dev_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["mean_dev_brier"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
