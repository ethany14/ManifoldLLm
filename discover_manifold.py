from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.manifold import Isomap, trustworthiness
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover an Isomap affect manifold.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--csv")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-value", default="train")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(args.embeddings).astype(np.float32)
    if len(embeddings) <= args.neighbors:
        raise ValueError("Number of rows must be larger than --neighbors.")

    if args.csv:
        frame = pd.read_csv(args.csv)
        if len(frame) != len(embeddings):
            raise ValueError("CSV and embeddings must have identical row counts.")
        if args.split_column not in frame:
            raise ValueError(f"Missing split column: {args.split_column}")
        train_mask = frame[args.split_column].astype(str).eq(args.train_value).to_numpy()
        if train_mask.sum() <= args.neighbors:
            raise ValueError("Training rows must be larger than --neighbors.")
    else:
        train_mask = np.ones(len(embeddings), dtype=bool)

    scaler = StandardScaler()
    scaler.fit(embeddings[train_mask])
    normalized = scaler.transform(embeddings)
    isomap = Isomap(n_neighbors=args.neighbors, n_components=args.dimensions)
    raw_coordinates = np.empty((len(embeddings), args.dimensions), dtype=np.float32)
    raw_coordinates[train_mask] = isomap.fit_transform(normalized[train_mask]).astype(np.float32)
    if (~train_mask).any():
        raw_coordinates[~train_mask] = isomap.transform(normalized[~train_mask]).astype(np.float32)
    raw_score = trustworthiness(
        normalized[train_mask],
        raw_coordinates[train_mask],
        n_neighbors=min(args.neighbors, (int(train_mask.sum()) - 1) // 2),
    )
    coordinate_scaler = StandardScaler().fit(raw_coordinates[train_mask])
    coordinates = coordinate_scaler.transform(raw_coordinates).astype(np.float32)
    normalized_score = trustworthiness(
        normalized[train_mask],
        coordinates[train_mask],
        n_neighbors=min(args.neighbors, (int(train_mask.sum()) - 1) // 2),
    )

    np.save(output / "coordinates.npy", coordinates)
    joblib.dump(scaler, output / "embedding_scaler.joblib")
    joblib.dump(isomap, output / "isomap.joblib")
    joblib.dump(coordinate_scaler, output / "coordinate_scaler.joblib")
    (output / "discovery_metrics.json").write_text(
        json.dumps({
            "train_trustworthiness_raw": float(raw_score),
            "train_trustworthiness_standardized": float(normalized_score),
            "fit_rows": int(train_mask.sum()),
            "out_of_sample_rows": int((~train_mask).sum()),
        }, indent=2), encoding="utf-8"
    )
    print(
        f"Saved standardized {args.dimensions}D coordinates; "
        f"trustworthiness={normalized_score:.4f}"
    )


if __name__ == "__main__":
    main()
