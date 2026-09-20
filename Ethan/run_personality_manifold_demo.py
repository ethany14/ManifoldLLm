"""Run the Beck manifold-personality inference chain on development examples."""

from __future__ import annotations

import json

import numpy as np
import torch

from personality_manifold_runtime import PersonalityManifold
from run_beck_behavior_pilot import OUT
from run_beck_graph_geometry import inputs


def main():
    torch.set_num_threads(1)
    blocks, _, _, _ = inputs(include_test=False)
    checkpoint = OUT / "riemannian_vae_models/seed_1.pt"
    if not checkpoint.exists():
        raise FileNotFoundError("Run Ethan/run_beck_riemannian_vae.py first")
    engine = PersonalityManifold(checkpoint)
    dev = blocks["dev"]
    ids, counts = np.unique(dev["id"], return_counts=True)
    people = ids[counts >= 2]
    if len(people) < 2:
        raise ValueError("Need two development people and repeated observations")
    first = np.flatnonzero(dev["id"] == people[0])
    second = np.flatnonzero(dev["id"] == people[1])
    i, j = int(first[0]), int(second[0])
    initial = engine.infer(dev["anchor"][i].numpy(), dev["deviation"][i].numpy(),
                           dev["time"][i].numpy(), "person_A_observation_1")
    next_state = engine.update_fast_state(initial, dev["deviation"][int(first[1])].numpy(),
                                          dev["time"][int(first[1])].numpy(),
                                          "person_A_observation_2")
    other = engine.infer(dev["anchor"][j].numpy(), dev["deviation"][j].numpy(),
                         dev["time"][j].numpy(), "person_B_observation_1")
    train = blocks["train"]
    _, first_idx = np.unique(train["id"], return_index=True)
    with torch.no_grad():
        reference_phi = engine.model.slow(train["anchor"][first_idx])[:, :4].numpy()
    distance = engine.approximate_geodesic(
        initial["slow_person_posterior"]["mean"],
        other["slow_person_posterior"]["mean"], reference_phi)
    result = {
        "status": "working inference demonstration, not a validated digital personality",
        "initial_person_A": initial,
        "person_A_after_new_state": next_state,
        "person_B": other,
        "person_A_B_distance": distance,
        "llm_condition_example": engine.llm_condition(next_state),
        "important_limits": [
            "No text or actual LLM generation is used in this Beck demonstration.",
            "Fast state is updated; slow personality is held fixed. Long-term online updating is not trained.",
            "The graph geodesic is approximate and depends on the training-person reference graph.",
            "All examples come from previously inspected development people.",
        ],
    }
    output = OUT / "personality_manifold_runtime_demo.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"initial_probability": initial["predicted_next_self_reported_studying"],
                      "updated_probability": next_state["predicted_next_self_reported_studying"],
                      "metric_min_eigenvalue": min(initial["local_metric_eigenvalues"]),
                      "graph_distance": distance["approximate_graph_geodesic"]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
