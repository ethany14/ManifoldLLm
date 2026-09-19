"""Test whether graph-geodesic distance adds value beyond latent Euclidean distance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from compare_geometry_targets import train_model
from compare_representation_routes import TARGETS, seed_everything
from run_ablation_study import encode


def sample_pairs(nodes: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    i = rng.choice(nodes, size=count, replace=True)
    j = rng.choice(nodes, size=count, replace=True)
    keep = i != j
    return i[keep], j[keep]


def fit_distance_models(
    euclidean: np.ndarray,
    geodesic: np.ndarray,
    vad: np.ndarray,
    fit_nodes: np.ndarray,
    eval_nodes: np.ndarray,
    seed: int,
    pair_count: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    fi, fj = sample_pairs(fit_nodes, pair_count, rng)
    ei, ej = sample_pairs(eval_nodes, pair_count, rng)

    def values(i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.column_stack([euclidean[i, j], geodesic[i, j]])
        target = np.linalg.norm(vad[i] - vad[j], axis=1)
        finite = np.isfinite(features).all(axis=1) & np.isfinite(target)
        return features[finite], target[finite]

    x_fit, y_fit = values(fi, fj)
    x_eval, y_eval = values(ei, ej)
    scaler = StandardScaler().fit(x_fit)
    x_fit = scaler.transform(x_fit)
    x_eval = scaler.transform(x_eval)

    euclidean_model = Ridge(alpha=1.0).fit(x_fit[:, [0]], y_fit)
    geodesic_model = Ridge(alpha=1.0).fit(x_fit[:, [1]], y_fit)
    combined_model = Ridge(alpha=1.0).fit(x_fit, y_fit)
    pred_e = euclidean_model.predict(x_eval[:, [0]])
    pred_g = geodesic_model.predict(x_eval[:, [1]])
    pred_both = combined_model.predict(x_eval)
    euclidean_r2 = float(r2_score(y_eval, pred_e))
    geodesic_r2 = float(r2_score(y_eval, pred_g))
    combined_r2 = float(r2_score(y_eval, pred_both))
    return {
        "fit_pairs": int(len(y_fit)),
        "eval_pairs": int(len(y_eval)),
        "euclidean_spearman": float(spearmanr(x_eval[:, 0], y_eval).statistic),
        "geodesic_spearman": float(spearmanr(x_eval[:, 1], y_eval).statistic),
        "euclidean_r2": euclidean_r2,
        "geodesic_r2": geodesic_r2,
        "combined_r2": combined_r2,
        "geodesic_incremental_r2": combined_r2 - euclidean_r2,
        "euclidean_rmse": float(mean_squared_error(y_eval, pred_e) ** 0.5),
        "geodesic_rmse": float(mean_squared_error(y_eval, pred_g) ** 0.5),
        "combined_rmse": float(mean_squared_error(y_eval, pred_both) ** 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 21, 42, 99])
    parser.add_argument("--neighbors", nargs="+", type=int, default=[10, 20, 40])
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--pair-count", type=int, default=50000)
    args = parser.parse_args()

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
    rows = []

    for seed in args.seeds:
        for target in ["none", "affective"]:
            seed_everything(seed)
            print(f"training seed={seed} geometry_target={target}")
            model = train_model(
                x, y, train, dev, args.dimensions, args.hidden_dim, args.epochs,
                args.batch_size, device, args.reconstruction_weight,
                args.geometry_weight, target, 0.5,
            )
            latent = encode(model, x, device, args.batch_size)
            latent_scaler = StandardScaler().fit(latent[train])
            test_z = latent_scaler.transform(latent[test]).astype(np.float64)
            test_y = y_raw[test].astype(np.float64)
            euclidean = np.linalg.norm(test_z[:, None, :] - test_z[None, :, :], axis=2)
            rng = np.random.default_rng(seed)
            nodes = rng.permutation(len(test))
            fit_nodes, eval_nodes = nodes[: len(nodes) // 2], nodes[len(nodes) // 2 :]

            for neighbors in args.neighbors:
                graph = kneighbors_graph(
                    test_z, n_neighbors=neighbors, mode="distance", include_self=False
                )
                graph = graph.maximum(graph.T)
                components, labels = connected_components(graph, directed=False)
                geodesic = shortest_path(graph, directed=False, unweighted=False)
                metrics = fit_distance_models(
                    euclidean, geodesic, test_y, fit_nodes, eval_nodes,
                    seed + neighbors, args.pair_count,
                )
                rows.append({
                    "seed": seed,
                    "geometry_target": target,
                    "neighbors": neighbors,
                    "connected_components": int(components),
                    "largest_component_fraction": float(np.bincount(labels).max() / len(labels)),
                    **metrics,
                })

    results = pd.DataFrame(rows)
    aggregate = results.groupby(["geometry_target", "neighbors"], as_index=False).agg(
        seeds=("seed", "count"),
        components_mean=("connected_components", "mean"),
        largest_component_fraction_mean=("largest_component_fraction", "mean"),
        euclidean_spearman_mean=("euclidean_spearman", "mean"),
        euclidean_spearman_sd=("euclidean_spearman", "std"),
        geodesic_spearman_mean=("geodesic_spearman", "mean"),
        geodesic_spearman_sd=("geodesic_spearman", "std"),
        euclidean_r2_mean=("euclidean_r2", "mean"),
        geodesic_r2_mean=("geodesic_r2", "mean"),
        combined_r2_mean=("combined_r2", "mean"),
        incremental_r2_mean=("geodesic_incremental_r2", "mean"),
        incremental_r2_sd=("geodesic_incremental_r2", "std"),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "geodesic_all_runs.csv", index=False)
    aggregate.to_csv(output / "geodesic_aggregate.csv", index=False)
    (output / "geodesic_summary.json").write_text(
        json.dumps({"design": vars(args), "aggregate": aggregate.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
