"""Demonstrate trained slow-posterior updating as a Beck history grows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from personality_manifold_runtime import PersonalityManifold
from run_beck_behavior_pilot import OUT, SOURCE
from run_beck_graph_geometry import STATE
from run_beck_online_history_vae import make_blocks


def main():
    torch.set_num_threads(1)
    blocks, _, _, _ = make_blocks()
    engine = PersonalityManifold(OUT / "online_history_models/seed_1.pt")
    dev = blocks["dev"]
    candidate_ids = np.unique(dev["id"])
    raw = pd.read_csv(SOURCE, sep="\t", usecols=["id", "date", "hour", "minute"] + STATE)
    raw["timestamp"] = (pd.to_datetime(raw["date"])
                        + pd.to_timedelta(raw["hour"], unit="h")
                        + pd.to_timedelta(raw["minute"], unit="m"))
    raw = raw.sort_values(["id", "timestamp"])
    chosen = None
    for pid in candidate_ids:
        first = np.flatnonzero(dev["id"] == pid)[0]
        n = int(dev["prefix_count"][first])
        records = raw[raw["id"] == pid].reset_index(drop=True)
        if n >= 10 and len(records) > n:
            chosen = (records, n, first)
            break
    if chosen is None:
        raise ValueError("No suitable development history")
    records, n, first_row = chosen
    first_history = records.iloc[:n][STATE].to_numpy(dtype=np.float32)
    initial = engine.infer_from_raw_history(
        first_history, records.iloc[n - 1]["timestamp"].to_pydatetime(), "person_A_prefix")
    standardized_check = engine.infer(
        dev["anchor"][first_row].numpy(), dev["deviation"][first_row].numpy(),
        dev["time"][first_row].numpy(), "same_preprocessed_row")
    preprocessing_difference = abs(
        initial["predicted_next_self_reported_studying"]
        - standardized_check["predicted_next_self_reported_studying"])
    if preprocessing_difference > 1e-5:
        raise AssertionError(f"Raw-history/preprocessed inference mismatch: {preprocessing_difference}")
    updated = engine.update_history(
        initial, records.iloc[n][STATE].to_numpy(dtype=np.float32),
        records.iloc[n]["timestamp"].to_pydatetime(), "person_A_one_more_prompt")
    other_id = next(pid for pid in candidate_ids if pid != records.iloc[0]["id"])
    other_first = np.flatnonzero(dev["id"] == other_id)[0]
    other_n = int(dev["prefix_count"][other_first])
    other_records = raw[raw["id"] == other_id].reset_index(drop=True)
    other = engine.infer_from_raw_history(
        other_records.iloc[:other_n][STATE].to_numpy(dtype=np.float32),
        other_records.iloc[other_n - 1]["timestamp"].to_pydatetime(), "person_B_prefix")
    train = blocks["train"]
    last_row = {int(pid): i for i, pid in enumerate(train["id"])}
    with torch.no_grad():
        reference = engine.model.slow(train["anchor"][list(last_row.values())])[:, :4].numpy()
    distance = engine.approximate_geodesic(
        updated["slow_person_posterior"]["mean"],
        other["slow_person_posterior"]["mean"], reference)
    phi_before = np.asarray(initial["slow_person_posterior"]["mean"])
    phi_after = np.asarray(updated["slow_person_posterior"]["mean"])
    result = {
        "status": "working variable-history manifold inference demo; development people only",
        "before": initial, "after_one_more_prompt": updated, "other_person": other,
        "slow_posterior_mean_change": float(np.linalg.norm(phi_after - phi_before)),
        "raw_history_preprocessing_probability_difference": preprocessing_difference,
        "updated_person_to_other_graph_distance": distance,
        "llm_condition_example": engine.llm_condition(updated),
        "limits": [
            "The history encoder sees prefix feature means, not raw text or a full sequence model.",
            "One new prompt changes the slow estimate numerically; its psychological calibration is unproven.",
            "All data are Beck self-reports and the demonstration uses previously inspected development people.",
        ],
    }
    path = OUT / "online_history_runtime_demo.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"history_before": initial["history_observation_count"],
                      "history_after": updated["history_observation_count"],
                      "slow_phi_change": result["slow_posterior_mean_change"],
                      "fast_z_change": float(np.linalg.norm(
                          np.asarray(updated["fast_state_posterior"]["mean"])
                          - np.asarray(initial["fast_state_posterior"]["mean"]))),
                      "metric_min_eigenvalue": min(updated["local_metric_eigenvalues"]),
                      "graph_distance": distance["approximate_graph_geodesic"]}, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
