"""Post-hoc simple valence summaries for interpreting manifold features.

Development only. Uses no neural model and never exports participant rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from run_psycharchives_manifold import TRAITS, load_data, split_people
from run_psycharchives_evaluation_12 import matching_metrics


def person_summaries(rows, people, ids):
    traits = people.set_index("user_id")[TRAITS]
    records = []
    for user_id, group in rows[rows.user_id.isin(ids)].groupby("user_id", sort=True):
        values = group.valence.to_numpy(dtype=float)
        if len(values) < 10:
            raise ValueError("Requires disjoint five-prompt windows")
        records.append((values.mean(), values.std(),
                        values[:5].mean(), values[:5].std(),
                        values[-5:].mean(), values[-5:].std(),
                        traits.loc[user_id].to_numpy(dtype=float)))
    features = np.asarray([item[:2] for item in records])
    early = np.asarray([item[2:4] for item in records])
    late = np.asarray([item[4:6] for item in records])
    trait_values = np.asarray([item[6] for item in records])
    return features, early, late, trait_values


def trait_scores(train_features, dev_features, train_traits, dev_traits, columns):
    train_mask = np.isfinite(train_traits).all(axis=1)
    dev_mask = np.isfinite(dev_traits).all(axis=1)
    feature_scaler = StandardScaler().fit(train_features)
    x_train = feature_scaler.transform(train_features)[train_mask][:, columns]
    x_dev = feature_scaler.transform(dev_features)[dev_mask][:, columns]
    trait_scaler = StandardScaler().fit(train_traits[train_mask])
    y_train = trait_scaler.transform(train_traits[train_mask])
    y_dev = trait_scaler.transform(dev_traits[dev_mask])
    predicted = Ridge(alpha=1.0).fit(x_train, y_train).predict(x_dev)
    per_trait = {}
    for i, trait in enumerate(TRAITS):
        per_trait[trait] = {"r2": float(r2_score(y_dev[:, i], predicted[:, i])),
                            "rmse_train_standardized": float(np.sqrt(np.mean(
                                (y_dev[:, i] - predicted[:, i]) ** 2)))}
    return {"labeled_train_people": int(train_mask.sum()),
            "labeled_dev_people": int(dev_mask.sum()),
            "per_trait": per_trait,
            "macro_trait_r2": float(np.mean([value["r2"] for value in per_trait.values()])),
            "pooled_trait_rmse": float(np.sqrt(np.mean((y_dev - predicted) ** 2))),
            "train_mean_baseline_rmse": float(np.sqrt(np.mean(y_dev ** 2)))}


def window_match(train_early, train_late, dev_early, dev_late, columns):
    scaler = StandardScaler().fit(np.vstack([train_early, train_late]))
    early = scaler.transform(dev_early)[:, columns]
    late = scaler.transform(dev_late)[:, columns]
    return matching_metrics(torch.cdist(torch.from_numpy(early), torch.from_numpy(late)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "psycharchives" / "feature_groups")
    args = parser.parse_args()
    rows, people, hashes = load_data(args.source_dir)
    split = split_people(people)
    tr_x, tr_early, tr_late, tr_y = person_summaries(rows, people, split["train"])
    dv_x, dv_early, dv_late, dv_y = person_summaries(rows, people, split["dev"])
    report = {"status": "post-hoc development-only non-neural valence baselines",
              "source_sha256": hashes,
              "cohort": {"train_people": len(tr_x), "dev_people": len(dv_x),
                         "reserved_test_people": len(split["test"])},
              "definitions": {"mean": "mean self-reported valence over observed prompts",
                              "sd": "population standard deviation of self-reported valence",
                              "windows": "nonoverlapping first five and last five prompts"},
              "mean_only": {"trait_probe": trait_scores(tr_x, dv_x, tr_y, dv_y, [0]),
                            "independent_windows": window_match(tr_early, tr_late, dv_early, dv_late, [0])},
              "mean_and_sd": {"trait_probe": trait_scores(tr_x, dv_x, tr_y, dv_y, [0, 1]),
                              "independent_windows": window_match(tr_early, tr_late, dv_early, dv_late, [0, 1])}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "valence_summary_baselines.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "mean_only": report["mean_only"],
                      "mean_and_sd": report["mean_and_sd"]}, indent=2))


if __name__ == "__main__":
    main()
