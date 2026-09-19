"""Evaluate a trained manifold adapter on the untouched test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from affective_adapter.model import ManifoldAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    embeddings = np.load(args.embeddings).astype(np.float32)
    coordinates = np.load(args.coordinates).astype(np.float32)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ManifoldAdapter(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaled = (embeddings - checkpoint["input_mean"]) / checkpoint["input_scale"]
    with torch.inference_mode():
        result = model(torch.from_numpy(scaled))
    predicted_z = result["z"].numpy()
    predicted_affect = result["valence_arousal"].numpy()
    affect = frame[["valence", "arousal"]].to_numpy(np.float32)
    train = frame["split"].eq("train").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    baseline_z = np.repeat(coordinates[train].mean(0, keepdims=True), test.sum(), axis=0)
    expected_distances = np.linalg.norm(
        coordinates[test][:, None, :] - coordinates[test][None, :, :], axis=2
    )
    predicted_distances = np.linalg.norm(
        predicted_z[test][:, None, :] - predicted_z[test][None, :, :], axis=2
    )
    triangle = np.triu_indices(int(test.sum()), 1)
    metrics = {
        "coordinate_mse": float(mean_squared_error(coordinates[test], predicted_z[test])),
        "coordinate_mean_baseline_mse": float(
            mean_squared_error(coordinates[test], baseline_z)
        ),
        "coordinate_r2": float(r2_score(coordinates[test], predicted_z[test])),
        "pairwise_distance_spearman": float(
            spearmanr(expected_distances[triangle], predicted_distances[triangle]).statistic
        ),
        "valence_mae": float(
            mean_absolute_error(affect[test, 0], predicted_affect[test, 0])
        ),
        "valence_r2": float(r2_score(affect[test, 0], predicted_affect[test, 0])),
        "arousal_mae": float(
            mean_absolute_error(affect[test, 1], predicted_affect[test, 1])
        ),
        "arousal_r2": float(r2_score(affect[test, 1], predicted_affect[test, 1])),
        "test_rows": int(test.sum()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
