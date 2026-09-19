"""Leakage-safe PCA and Isomap comparison on saved frozen-LLM embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import Isomap, trustworthiness
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def affect_metrics(
    train_z: np.ndarray,
    test_z: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    z_scaler = StandardScaler().fit(train_z)
    model = Ridge(alpha=10.0).fit(z_scaler.transform(train_z), train_y)
    prediction = model.predict(z_scaler.transform(test_z))
    return {
        "valence_mae": float(mean_absolute_error(test_y[:, 0], prediction[:, 0])),
        "valence_r2": float(r2_score(test_y[:, 0], prediction[:, 0])),
        "arousal_mae": float(mean_absolute_error(test_y[:, 1], prediction[:, 1])),
        "arousal_r2": float(r2_score(test_y[:, 1], prediction[:, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--neighbors", nargs="+", type=int, default=[5, 10, 20, 40])
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    embeddings = np.load(args.embeddings).astype(np.float32)
    if len(frame) != len(embeddings):
        raise ValueError("CSV and embeddings must have identical row counts.")
    train = frame["split"].eq("train").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    y = frame[["valence", "arousal"]].to_numpy(np.float32)
    scaler = StandardScaler().fit(embeddings[train])
    x_train = scaler.transform(embeddings[train])
    x_test = scaler.transform(embeddings[test])
    k_eval = min(10, (int(train.sum()) - 1) // 2)
    rows: list[dict[str, float | int | str]] = []

    for dimensions in args.dimensions:
        pca = PCA(n_components=dimensions, random_state=42).fit(x_train)
        train_z = pca.transform(x_train)
        test_z = pca.transform(x_test)
        rows.append({
            "method": "PCA", "dimensions": dimensions, "neighbors": 0,
            "trustworthiness": float(trustworthiness(x_train, train_z, n_neighbors=k_eval)),
            **affect_metrics(train_z, test_z, y[train], y[test]),
        })
        for neighbors in args.neighbors:
            model = Isomap(n_neighbors=neighbors, n_components=dimensions)
            train_z = model.fit_transform(x_train)
            test_z = model.transform(x_test)
            rows.append({
                "method": "Isomap", "dimensions": dimensions, "neighbors": neighbors,
                "trustworthiness": float(trustworthiness(x_train, train_z, n_neighbors=k_eval)),
                **affect_metrics(train_z, test_z, y[train], y[test]),
            })
            print(f"finished Isomap dimensions={dimensions}, neighbors={neighbors}")

    results = pd.DataFrame(rows)
    results["mean_affect_r2"] = (results["valence_r2"] + results["arousal_r2"]) / 2
    results = results.sort_values(
        ["mean_affect_r2", "trustworthiness"], ascending=False
    ).reset_index(drop=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "manifold_benchmark.csv", index=False)
    best_isomap = results.loc[results["method"].eq("Isomap")].iloc[0].to_dict()
    best_pca = results.loc[results["method"].eq("PCA")].iloc[0].to_dict()
    summary = {"best_isomap": best_isomap, "best_pca": best_pca}
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
