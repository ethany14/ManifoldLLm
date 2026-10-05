"""Run the fixed development protocol for personality readouts and ablations.

Licensed rows and participant IDs remain in memory only. No test scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from run_psycharchives_manifold import CUES, TRAITS, load_data, sha256, split_people
from run_psycharchives_sequence_manifold import (
    BATCH_PEOPLE, SequenceManifold, WEIGHTS, batch_view,
    development_metrics, prepare, train_loss,
)


SEEDS = (1, 2, 3)
EPOCHS = 30
WINDOW = 5
PATH_SEGMENTS = 8
CONDITIONS = ("full", "no_trait_loss", "no_transition")


def fit_fixed(blocks, seed, condition):
    torch.manual_seed(seed)
    model = SequenceManifold(len(CUES), use_transition=condition != "no_transition")
    weights = dict(WEIGHTS)
    if condition == "no_trait_loss":
        weights["traits"] = 0.0
    if condition == "no_transition":
        weights["transition"] = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    train = blocks["train"]
    for _ in range(EPOCHS):
        model.train()
        for indices in torch.randperm(train["people"]).split(BATCH_PEOPLE):
            mini = batch_view(train, indices)
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss(model, mini, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    model.eval()
    return model


def last_phi(model, block):
    with torch.no_grad():
        phi = model(block["x"], block["observed"])["phi"]
        return phi[torch.arange(block["people"]), block["lengths"] - 1].numpy()


def trait_probe(model, train, dev):
    train_phi, dev_phi = last_phi(model, train), last_phi(model, dev)
    train_mask = train["trait_valid"].numpy()
    dev_mask = dev["trait_valid"].numpy()
    ridge = Ridge(alpha=1.0).fit(train_phi[train_mask], train["traits"].numpy()[train_mask])
    prediction = ridge.predict(dev_phi[dev_mask])
    truth = dev["traits"].numpy()[dev_mask]
    per_trait = {}
    for index, trait in enumerate(TRAITS):
        per_trait[trait] = {
            "r2": float(r2_score(truth[:, index], prediction[:, index])),
            "rmse_train_standardized": float(np.sqrt(np.mean(
                (truth[:, index] - prediction[:, index]) ** 2))),
            "train_mean_baseline_rmse": float(np.sqrt(np.mean(truth[:, index] ** 2))),
        }
    return {"labeled_train_people": int(train_mask.sum()),
            "labeled_dev_people": int(dev_mask.sum()), "per_trait": per_trait,
            "macro_trait_r2": float(np.mean([value["r2"] for value in per_trait.values()])),
            "pooled_trait_rmse_train_standardized": float(np.sqrt(np.mean((truth - prediction) ** 2))),
            "pooled_train_mean_baseline_rmse": float(np.sqrt(np.mean(truth ** 2)))}


def independent_windows(model, block):
    if int(block["lengths"].min()) < 2 * WINDOW:
        raise ValueError("Independent windows require at least 10 records per person")
    first_x, first_observed = block["x"][:, :WINDOW], block["observed"][:, :WINDOW]
    offsets = block["lengths"][:, None] - WINDOW + torch.arange(WINDOW)[None, :]
    people = torch.arange(block["people"])[:, None]
    last_x, last_observed = block["x"][people, offsets], block["observed"][people, offsets]
    with torch.no_grad():
        early = model(first_x, first_observed)["phi"][:, -1]
        late = model(last_x, last_observed)["phi"][:, -1]
    return early, late


def matching_metrics(distances):
    d = distances.detach().cpu().numpy()
    n = len(d)
    if d.shape != (n, n):
        raise ValueError("Expected square person-distance matrix")
    diagonal = np.diag(d)
    cross_person = d[~np.eye(n, dtype=bool)]
    return {"rank1_same_person_accuracy": float(np.mean(d.argmin(axis=1) == np.arange(n))),
            "within_over_between_median_distance": float(np.median(diagonal) / np.median(cross_person)),
            "median_same_person_distance": float(np.median(diagonal)),
            "median_cross_person_distance": float(np.median(cross_person))}


def decoder_path_distances(model, early, late):
    """Length of decoded straight coordinate path, not a geodesic optimum."""
    n = len(early)
    start = early[:, None, :].expand(n, n, 4).reshape(-1, 4)
    end = late[None, :, :].expand(n, n, 4).reshape(-1, 4)
    length = torch.zeros(len(start))
    with torch.no_grad():
        previous = model.decode_person(start)
        for step in range(1, PATH_SEGMENTS + 1):
            alpha = step / PATH_SEGMENTS
            current = model.decode_person((1 - alpha) * start + alpha * end)
            length += torch.linalg.vector_norm(current - previous, dim=1)
            previous = current
    return length.reshape(n, n)


def assess(model, blocks, condition, include_path=False):
    train, dev = blocks["train"], blocks["dev"]
    valence = development_metrics(model, dev)
    early, late = independent_windows(model, dev)
    euclidean = matching_metrics(torch.cdist(early, late))
    result = {"next_valence_rmse_1_to_6": valence["next_valence_rmse_1_to_6"],
              "next_valence_mae_1_to_6": valence["next_valence_mae_1_to_6"],
              "trait_linear_probe": trait_probe(model, train, dev),
              "independent_windows_euclidean": euclidean}
    if condition == "full" or include_path:
        result["independent_windows_decoder_path"] = matching_metrics(
            decoder_path_distances(model, early, late))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "psycharchives" / "evaluation_12")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, people, source_hashes = load_data(args.source_dir)
    split = split_people(people)
    blocks, preprocessing = prepare(rows, people, split)
    protocol = Path(__file__).resolve().parent / "PSYCHARCHIVES_EVALUATION_PROTOCOL.md"
    report = {"status": "exploratory development-only evaluation; test unopened",
              "protocol_sha256": sha256(protocol), "source_sha256": source_hashes,
              "cohort": {"train_people": blocks["train"]["people"],
                         "dev_people": blocks["dev"]["people"],
                         "reserved_test_people": len(split["test"]),
                         "train_pairs": blocks["train"]["pairs"],
                         "dev_pairs": blocks["dev"]["pairs"]},
              "settings": {"seeds": SEEDS, "epochs": EPOCHS, "window_prompts": WINDOW,
                           "decoder_path_segments": PATH_SEGMENTS,
                           "conditions": CONDITIONS, "test": "never scored"},
              "runs": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for condition in CONDITIONS:
            model = fit_fixed(blocks, seed, condition)
            run = {"seed": seed, "condition": condition, **assess(model, blocks, condition)}
            report["runs"].append(run)
            torch.save({"state_dict": model.state_dict(), "preprocessing": preprocessing,
                        "seed": seed, "condition": condition, "fixed_epochs": EPOCHS,
                        "protocol_sha256": report["protocol_sha256"]},
                       args.output_dir / f"{condition}_seed_{seed}.pt")
            print(f"seed={seed} condition={condition} dev_RMSE={run['next_valence_rmse_1_to_6']:.3f}",
                  flush=True)
    summary = {}
    for condition in CONDITIONS:
        group = [run for run in report["runs"] if run["condition"] == condition]
        summary[condition] = {
            "mean_next_valence_rmse": float(np.mean([run["next_valence_rmse_1_to_6"] for run in group])),
            "mean_macro_trait_r2": float(np.mean([
                run["trait_linear_probe"]["macro_trait_r2"] for run in group])),
            "mean_pooled_trait_rmse": float(np.mean([
                run["trait_linear_probe"]["pooled_trait_rmse_train_standardized"] for run in group])),
            "mean_early_late_rank1_euclidean": float(np.mean([
                run["independent_windows_euclidean"]["rank1_same_person_accuracy"] for run in group])),
        }
        if condition == "full":
            summary[condition]["mean_early_late_rank1_decoder_path"] = float(np.mean([
                run["independent_windows_decoder_path"]["rank1_same_person_accuracy"] for run in group]))
    report["summary"] = summary
    output = args.output_dir / "development_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
