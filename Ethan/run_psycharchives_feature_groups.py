"""Development-only cue-group sensitivity of unsupervised person coordinates.

No auxiliary trait loss; traits enter only the post-training training-person
linear probe and aggregate development evaluation. Licensed rows stay local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_psycharchives_manifold import CUES, load_data, sha256, split_people
from run_psycharchives_sequence_manifold import (
    BATCH_PEOPLE, SequenceManifold, WEIGHTS, batch_view, prepare, train_loss,
)
from run_psycharchives_geometry_training import identity_loss
from run_psycharchives_evaluation_12 import assess


SEEDS = (1, 2, 3)
EPOCHS = 30
IDENTITY_WEIGHT = 0.02
ACTIVITY = [cue for cue in CUES if cue.startswith("Activity_prob_")]
TIME = [cue for cue in CUES if cue.startswith("Timestamp_status_")]
GROUPS = {
    "full_22": list(CUES),
    "activity_time_valence_9": ACTIVITY + TIME + ["valence"],
    "time_valence_6": TIME + ["valence"],
    "valence_only_1": ["valence"],
}


def verify_groups():
    expected = {"full_22": 22, "activity_time_valence_9": 9,
                "time_valence_6": 6, "valence_only_1": 1}
    for name, cues in GROUPS.items():
        if len(cues) != expected[name] or len(cues) != len(set(cues)) or not set(cues).issubset(CUES):
            raise ValueError(f"Unexpected cue group {name}")
    if not set(GROUPS["valence_only_1"]).issubset(GROUPS["time_valence_6"]):
        raise ValueError("Feature groups must be nested")
    if not set(GROUPS["time_valence_6"]).issubset(GROUPS["activity_time_valence_9"]):
        raise ValueError("Feature groups must be nested")


def fit_group(blocks, seed):
    torch.manual_seed(seed)
    model = SequenceManifold(blocks["train"]["x"].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    weights = dict(WEIGHTS)
    weights["traits"] = 0.0
    train = blocks["train"]
    for _ in range(EPOCHS):
        model.train()
        for indices in torch.randperm(train["people"]).split(BATCH_PEOPLE):
            mini = batch_view(train, indices)
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss(model, mini, weights)
            loss = loss + IDENTITY_WEIGHT * identity_loss(model, mini, "euclidean_identity")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "psycharchives" / "feature_groups")
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_groups()
    rows, people, source_hashes = load_data(args.source_dir)
    split = split_people(people)
    protocol_path = Path(__file__).resolve().parent / "PSYCHARCHIVES_FEATURE_GROUP_PROTOCOL.md"
    report = {"status": "exploratory development-only cue-group sensitivity",
              "protocol_sha256": sha256(protocol_path), "source_sha256": source_hashes,
              "settings": {"seeds": SEEDS, "epochs": EPOCHS,
                           "auxiliary_trait_loss_weight": 0.0,
                           "identity_loss": "Euclidean, symmetric InfoNCE, weight 0.02",
                           "groups": GROUPS, "test": "reserved, never scored"},
              "cohort": {"train_people": len(split["train"]),
                         "dev_people": len(split["dev"]),
                         "reserved_test_people": len(split["test"])},
              "runs": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group_name, cues in GROUPS.items():
        blocks, preprocessing = prepare(rows, people, split, cues=cues)
        for seed in SEEDS:
            model = fit_group(blocks, seed)
            result = assess(model, blocks, "feature_group")
            report["runs"].append({"group": group_name, "seed": seed, **result})
            torch.save({"state_dict": model.state_dict(), "preprocessing": preprocessing,
                        "group": group_name, "seed": seed, "fixed_epochs": EPOCHS,
                        "protocol_sha256": report["protocol_sha256"]},
                       args.output_dir / f"{group_name}_seed_{seed}.pt")
            print(f"group={group_name} seed={seed} "
                  f"rank1={result['independent_windows_euclidean']['rank1_same_person_accuracy']:.3f} "
                  f"trait_R2={result['trait_linear_probe']['macro_trait_r2']:.3f}", flush=True)
    summary = {}
    for name in GROUPS:
        subset = [run for run in report["runs"] if run["group"] == name]
        summary[name] = {
            "mean_next_valence_rmse": float(np.mean([run["next_valence_rmse_1_to_6"] for run in subset])),
            "mean_macro_trait_r2": float(np.mean([run["trait_linear_probe"]["macro_trait_r2"]
                                                   for run in subset])),
            "mean_pooled_trait_rmse": float(np.mean([
                run["trait_linear_probe"]["pooled_trait_rmse_train_standardized"]
                for run in subset])),
            "mean_early_late_rank1": float(np.mean([
                run["independent_windows_euclidean"]["rank1_same_person_accuracy"]
                for run in subset])),
            "mean_within_over_between": float(np.mean([
                run["independent_windows_euclidean"]["within_over_between_median_distance"]
                for run in subset])),
        }
    report["summary"] = summary
    output = args.output_dir / "development_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
