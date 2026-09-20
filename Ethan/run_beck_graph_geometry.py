"""Matched slow-fast models with/without unsupervised graph-geodesic stress.

The graph is built from the first 10 prompts of TRAIN people only. This is a
nonlinear graph-distance proxy, not a learned Riemannian metric or LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import shortest_path
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from run_beck_behavior_pilot import AFFECT, ACTIVITY, OUT, SITUATION, SOURCE, TARGET, build_pairs, scores


STATE = SITUATION + AFFECT + ACTIVITY + [TARGET]
TIME = ["history_rate", "hour_sin", "hour_cos", "day_of_week"]
SEEDS = [1, 2, 3]
EPOCHS = 350
PATIENCE = 40
GEOMETRY_WEIGHT = 0.01
K_NEIGHBORS = 8


class SlowFast(torch.nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.slow = torch.nn.Sequential(torch.nn.Linear(state_dim, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
        self.fast = torch.nn.Sequential(torch.nn.Linear(state_dim, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
        self.head = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))

    def forward(self, anchor, deviation, time):
        phi = self.slow(anchor)
        fast = self.fast(deviation)
        return self.head(torch.cat([phi, fast, time], dim=1)).squeeze(1)


def inputs(include_test=True):
    pairs, audit = build_pairs()
    baseline = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    ids = baseline["protocol"]["split_person_ids"]
    if sorted(int(i) for values in ids.values() for i in values) != sorted(int(i) for i in pairs["id"].unique()):
        raise ValueError("Saved split does not match current eligible cohort")
    split = {name: pairs[pairs["id"].isin(values)].copy() for name, values in ids.items()}
    raw = pd.read_csv(SOURCE, sep="\t", usecols=["id", "date", "hour", "minute"] + STATE)
    raw["timestamp"] = pd.to_datetime(raw["date"]) + pd.to_timedelta(raw["hour"], unit="h") + pd.to_timedelta(raw["minute"], unit="m")
    raw = raw.sort_values(["id", "timestamp"])
    anchors = raw.groupby("id").head(10).groupby("id")[STATE].mean()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_current = split["train"][STATE]
    scaler.fit(imputer.fit_transform(train_current))
    time_scaler = StandardScaler().fit(split["train"][TIME])
    blocks = {}
    for name, part in split.items():
        if name == "test" and not include_test:
            continue
        current = scaler.transform(imputer.transform(part[STATE])).astype("float32")
        anchor = scaler.transform(imputer.transform(anchors.loc[part["id"]])).astype("float32")
        time = time_scaler.transform(part[TIME]).astype("float32")
        blocks[name] = {
            "anchor": torch.from_numpy(anchor),
            "deviation": torch.from_numpy(current - anchor),
            "time": torch.from_numpy(time),
            "y": torch.tensor(part["next_target"].to_numpy(), dtype=torch.float32),
            "id": part["id"].to_numpy(),
        }
    train_ids = ids["train"]
    train_anchor = scaler.transform(imputer.transform(anchors.loc[train_ids])).astype("float32")
    graph = kneighbors_graph(train_anchor, n_neighbors=K_NEIGHBORS, mode="distance", include_self=False)
    graph = graph.maximum(graph.T)
    geodesic = shortest_path(graph, directed=False)
    if not np.isfinite(geodesic).all():
        raise ValueError("Person graph is disconnected; adjust predeclared k and rerun protocol")
    pair_i, pair_j = np.triu_indices(len(train_ids), k=1)
    target_distance = geodesic[pair_i, pair_j].astype("float32")
    target_distance /= target_distance.mean()
    geometry = {
        "anchor": torch.from_numpy(train_anchor),
        "i": torch.from_numpy(pair_i),
        "j": torch.from_numpy(pair_j),
        "distance": torch.from_numpy(target_distance),
    }
    return blocks, geometry, audit, ids


def predict(model, block):
    with torch.no_grad():
        return torch.sigmoid(model(block["anchor"], block["deviation"], block["time"])).numpy()


def train_one(blocks, geometry, seed, weight):
    torch.manual_seed(seed)
    model = SlowFast(blocks["train"]["anchor"].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001)
    train, dev = blocks["train"], blocks["dev"]
    best_state, best_dev, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train["anchor"], train["deviation"], train["time"])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train["y"])
        if weight:
            phi = model.slow(geometry["anchor"])
            distance = torch.linalg.vector_norm(phi[geometry["i"]] - phi[geometry["j"]], dim=1)
            relative = distance / distance.mean().clamp_min(1e-6)
            loss = loss + weight * torch.mean((relative - geometry["distance"]) ** 2)
        loss.backward()
        optimizer.step()
        model.eval()
        dev_p = predict(model, dev)
        dev_brier = brier_score_loss(dev["y"].numpy(), dev_p)
        if dev_brier < best_dev - 1e-7:
            best_dev, best_epoch, stale = dev_brier, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def bootstrap_delta(y, people, model_p, baseline_p, seed=42, repeats=2000):
    unique = np.unique(people)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(people == person) for person in sample])
        deltas.append(np.mean((model_p[indices] - y[indices]) ** 2 - (baseline_p[indices] - y[indices]) ** 2))
    return [float(v) for v in np.quantile(deltas, [0.025, 0.975])]


def main():
    torch.set_num_threads(1)
    blocks, geometry, audit, ids = inputs()
    result = {
        "protocol": {"seeds": SEEDS, "epochs_max": EPOCHS, "patience": PATIENCE, "geometry_weight": GEOMETRY_WEIGHT,
                     "neighbors": K_NEIGHBORS, "state_columns": STATE, "time_columns": TIME,
                     "split_person_ids": ids, "primary_metric": "Brier; lower is better",
                     "geometry_definition": "unsupervised shortest-path distances on an 8-NN graph of first-ten-prompt person anchors, train people only; normalized pairwise stress on 4-D slow embedding"},
        "audit": audit, "runs": [], "ensemble": {}, "paired_bootstrap": {},
    }
    predictions = {variant: {name: [] for name in ["dev", "test"]} for variant in ["euclidean", "graph_geodesic"]}
    for seed in SEEDS:
        run = {"seed": seed}
        for variant, weight in [("euclidean", 0.0), ("graph_geodesic", GEOMETRY_WEIGHT)]:
            model, epoch = train_one(blocks, geometry, seed, weight)
            run[variant] = {"best_epoch": epoch}
            for name in ["dev", "test"]:
                p = predict(model, blocks[name])
                predictions[variant][name].append(p)
                run[variant][name] = scores(blocks[name]["y"].numpy(), p)
        result["runs"].append(run)
    for variant in predictions:
        result["ensemble"][variant] = {}
        for name in ["dev", "test"]:
            p = np.mean(predictions[variant][name], axis=0)
            result["ensemble"][variant][name] = scores(blocks[name]["y"].numpy(), p)
    y = blocks["test"]["y"].numpy()
    people = blocks["test"]["id"]
    euclidean = np.mean(predictions["euclidean"]["test"], axis=0)
    graph = np.mean(predictions["graph_geodesic"]["test"], axis=0)
    result["paired_bootstrap"] = {"delta_graph_minus_euclidean_test_brier": float(np.mean((graph - y) ** 2 - (euclidean - y) ** 2)),
                                  "person_cluster_95_percentile_interval": bootstrap_delta(y, people, graph, euclidean)}
    result["interpretation_limit"] = "Graph shortest paths on observed first-ten-prompt anchors are an unsupervised nonlinear regularizer; this is not proof of an intrinsic personality manifold or LLM behavior."
    output = OUT / "graph_geometry_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"ensemble": result["ensemble"], "paired_bootstrap": result["paired_bootstrap"]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
