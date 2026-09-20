"""Compare Euclidean versus decoder-path identity losses during training.

Development-only and exploratory. Licensed source rows are never exported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_psycharchives_manifold import CUES, load_data, sha256, split_people
from run_psycharchives_sequence_manifold import (
    BATCH_PEOPLE, SequenceManifold, batch_view, metric_check, prepare, train_loss,
)
from run_psycharchives_evaluation_12 import assess


SEEDS = (1, 2, 3)
CONDITIONS = ("base", "euclidean_identity", "decoder_path_identity")
EPOCHS = 30
WINDOW = 5
TRAIN_PATH_SEGMENTS = 4
IDENTITY_WEIGHT = 0.02
TEMPERATURE = 0.2


def independent_window_phi(model, block):
    """Encode disjoint early and late windows with gradients enabled."""
    if int(block["lengths"].min()) < 2 * WINDOW:
        raise ValueError("At least 10 records per person are required")
    n = len(block["lengths"])
    offsets = block["lengths"][:, None] - WINDOW + torch.arange(WINDOW)[None, :]
    row = torch.arange(n)[:, None]
    early = model(block["x"][:, :WINDOW], block["observed"][:, :WINDOW])["phi"][:, -1]
    late = model(block["x"][row, offsets], block["observed"][row, offsets])["phi"][:, -1]
    return early, late


def differentiable_decoder_path(model, early, late):
    """Straight coordinate path length in decoded standardized cue space."""
    n = len(early)
    start = early[:, None, :].expand(n, n, 4).reshape(-1, 4)
    end = late[None, :, :].expand(n, n, 4).reshape(-1, 4)
    previous = model.decode_person(start)
    length = torch.zeros(len(start), dtype=start.dtype, device=start.device)
    for step in range(1, TRAIN_PATH_SEGMENTS + 1):
        fraction = step / TRAIN_PATH_SEGMENTS
        current = model.decode_person((1 - fraction) * start + fraction * end)
        length = length + torch.linalg.vector_norm(current - previous, dim=1)
        previous = current
    return length.reshape(n, n)


def identity_loss(model, block, condition, detach_distance_scale=False):
    early, late = independent_window_phi(model, block)
    distances = (torch.cdist(early, late) if condition == "euclidean_identity"
                 else differentiable_decoder_path(model, early, late))
    n = len(early)
    off_diagonal = distances[~torch.eye(n, dtype=torch.bool, device=distances.device)]
    scale = off_diagonal.mean().clamp_min(1e-6)
    if detach_distance_scale:
        scale = scale.detach()  # Historical v1 numerical pilot only.
    logits = -distances / (scale * TEMPERATURE)
    target = torch.arange(n, device=distances.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def fit_condition(blocks, seed, condition, detach_distance_scale=False):
    torch.manual_seed(seed)
    model = SequenceManifold(blocks["train"]["x"].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    train = blocks["train"]
    for _ in range(EPOCHS):
        model.train()
        for indices in torch.randperm(train["people"]).split(BATCH_PEOPLE):
            mini = batch_view(train, indices)
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss(model, mini)
            if condition != "base":
                loss = loss + IDENTITY_WEIGHT * identity_loss(
                    model, mini, condition, detach_distance_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--detach-distance-scale", action="store_true",
                        help="Reproduce the historical v1 numerical pilot only")
    args = parser.parse_args()
    if args.output_dir is None:
        version = "geometry_training" if args.detach_distance_scale else "geometry_training_v2"
        args.output_dir = Path(__file__).resolve().parent / "artifacts" / "psycharchives" / version
    torch.set_num_threads(1)
    rows, people, source_hashes = load_data(args.source_dir)
    split = split_people(people)
    blocks, preprocessing = prepare(rows, people, split)
    protocol_name = ("PSYCHARCHIVES_GEOMETRY_TRAINING_PROTOCOL.md" if args.detach_distance_scale
                     else "PSYCHARCHIVES_GEOMETRY_TRAINING_PROTOCOL_V2.md")
    protocol_path = Path(__file__).resolve().parent / protocol_name
    report = {"status": "exploratory development-only geometry-in-training comparison",
              "protocol_sha256": sha256(protocol_path), "source_sha256": source_hashes,
              "cohort": {"train_people": blocks["train"]["people"],
                         "dev_people": blocks["dev"]["people"],
                         "reserved_test_people": len(split["test"]),
                         "train_pairs": blocks["train"]["pairs"],
                         "dev_pairs": blocks["dev"]["pairs"]},
              "settings": {"seeds": SEEDS, "conditions": CONDITIONS, "epochs": EPOCHS,
                           "window": WINDOW, "train_path_segments": TRAIN_PATH_SEGMENTS,
                           "identity_weight": IDENTITY_WEIGHT, "temperature": TEMPERATURE,
                           "detach_distance_scale": args.detach_distance_scale,
                           "test": "not scored"},
              "runs": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for condition in CONDITIONS:
            model = fit_condition(blocks, seed, condition, args.detach_distance_scale)
            result = assess(model, blocks, condition, include_path=True)
            result["metric_numerical_check"] = metric_check(model, blocks["dev"])
            report["runs"].append({"seed": seed, "condition": condition, **result})
            torch.save({"state_dict": model.state_dict(), "preprocessing": preprocessing,
                        "seed": seed, "condition": condition, "fixed_epochs": EPOCHS,
                        "protocol_sha256": report["protocol_sha256"]},
                       args.output_dir / f"{condition}_seed_{seed}.pt")
            print(f"seed={seed} condition={condition} "
                  f"rank1={result['independent_windows_euclidean']['rank1_same_person_accuracy']:.3f} "
                  f"trait_R2={result['trait_linear_probe']['macro_trait_r2']:.3f}", flush=True)
    summary = {}
    for condition in CONDITIONS:
        subset = [run for run in report["runs"] if run["condition"] == condition]
        summary[condition] = {
            "mean_next_valence_rmse": float(np.mean([run["next_valence_rmse_1_to_6"] for run in subset])),
            "mean_macro_trait_r2": float(np.mean([run["trait_linear_probe"]["macro_trait_r2"]
                                                   for run in subset])),
            "mean_pooled_trait_rmse": float(np.mean([
                run["trait_linear_probe"]["pooled_trait_rmse_train_standardized"]
                for run in subset])),
            "mean_early_late_rank1_euclidean": float(np.mean([
                run["independent_windows_euclidean"]["rank1_same_person_accuracy"]
                for run in subset])),
            "mean_early_late_rank1_decoder_path": float(np.mean([
                run["independent_windows_decoder_path"]["rank1_same_person_accuracy"]
                for run in subset])),
            "mean_within_over_between_euclidean": float(np.mean([
                run["independent_windows_euclidean"]["within_over_between_median_distance"]
                for run in subset])),
            "mean_within_over_between_decoder_path": float(np.mean([
                run["independent_windows_decoder_path"]["within_over_between_median_distance"]
                for run in subset])),
        }
    report["summary"] = summary
    output = args.output_dir / "development_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
