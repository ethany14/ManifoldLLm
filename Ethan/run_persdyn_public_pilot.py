"""Reproducible person-held-out forecasting pilot on the PersDyn OSF workbook.

This is a personality-state dynamics test, not a test of text-conditioned LLM
generation or behavioral validity. The source workbook remains unmodified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE_URL = "https://osf.io/download/byr4u/"
SOURCE_SHA256 = "3b6c8ba9d02a57ac23e590e5db9b5a20c97c78531ce68e130a203315ac8cdbd1"
TRAITS = ["O", "C", "E", "A", "N"]


def get_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(SOURCE_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Source SHA-256 mismatch: {digest}")


def load_complete_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="only TIPI", engine="openpyxl")
    required = ["ppn", "ExecutionTime", *TRAITS]
    if any(column not in frame for column in required):
        raise ValueError("The OSF workbook schema has changed")
    frame = frame[required].copy()
    frame["ExecutionTime"] = pd.to_datetime(frame["ExecutionTime"], errors="coerce")
    frame = frame.dropna(subset=required)
    frame = frame.sort_values(["ppn", "ExecutionTime"], kind="stable")
    if frame.duplicated(["ppn", "ExecutionTime"]).any():
        raise ValueError("Duplicate participant/timestamp records need explicit resolution")
    if not frame[TRAITS].apply(lambda column: column.between(0, 100)).all().all():
        raise ValueError("Unexpected trait score outside 0-100")
    return frame


def make_examples(frame: pd.DataFrame, history: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, targets, people, persistence, running_mean = [], [], [], [], []
    for person, group in frame.groupby("ppn", sort=True):
        states = group[TRAITS].to_numpy(dtype=float) / 100.0
        times = group["ExecutionTime"].to_numpy(dtype="datetime64[ns]")
        if len(group) <= history:
            continue
        for t in range(history, len(group)):
            lag_hours = float((times[t] - times[t - 1]) / np.timedelta64(1, "h"))
            if lag_hours < 0:
                raise ValueError("Timestamps are not monotonic")
            previous = states[t - 1]
            average = states[:t].mean(axis=0)
            # All features are available strictly before the target observation.
            hour = pd.Timestamp(times[t]).hour + pd.Timestamp(times[t]).minute / 60
            vector = np.r_[previous, average, np.log1p(lag_hours),
                           np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
            features.append(vector)
            targets.append(states[t])
            people.append(person)
            persistence.append(previous)
            running_mean.append(average)
    return tuple(np.asarray(item) for item in (features, targets, people, persistence, running_mean))


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "mean_mae_0_to_100": round(float(mean_absolute_error(truth, prediction) * 100), 4),
        "mean_rmse_0_to_100": round(float(np.sqrt(mean_squared_error(truth, prediction)) * 100), 4),
        "mean_r2": round(float(np.mean([r2_score(truth[:, j], prediction[:, j]) for j in range(5)])), 4),
        "trait_mae_0_to_100": {trait: round(float(mean_absolute_error(truth[:, j], prediction[:, j]) * 100), 4)
                              for j, trait in enumerate(TRAITS)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "data/persdyn/source/dataES_copy.xlsx")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts/persdyn_public_pilot/metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history", type=int, default=10)
    args = parser.parse_args()

    get_source(args.source)
    frame = load_complete_rows(args.source)
    x, y, people, persistence, running_mean = make_examples(frame, args.history)
    unique_people = np.sort(np.unique(people))
    train_people, other_people = train_test_split(unique_people, test_size=0.4, random_state=args.seed)
    dev_people, test_people = train_test_split(other_people, test_size=0.5, random_state=args.seed)
    masks = {name: np.isin(people, ids) for name, ids in
             (("train", train_people), ("dev", dev_people), ("test", test_people))}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("A split has no forecasting examples")

    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "nonlinear_mlp": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32,), activation="tanh", alpha=1.0,
            max_iter=1000, early_stopping=False, random_state=args.seed)),
    }
    results = {}
    for split in ("dev", "test"):
        mask = masks[split]
        results[split] = {
            "persistence": metrics(y[mask], persistence[mask]),
            "person_running_mean": metrics(y[mask], running_mean[mask]),
        }
    for name, model in models.items():
        model.fit(x[masks["train"]], y[masks["train"]])
        for split in ("dev", "test"):
            mask = masks[split]
            results[split][name] = metrics(y[mask], np.clip(model.predict(x[mask]), 0, 1))

    report = {
        "question": "Predict the next Big Five personality-state rating after ten prior observations of a new person",
        "status": "exploratory first-stage pilot; no text, LLM, behavior outcome, or manifold model",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "data_rows_complete": int(len(frame)),
        "participants": int(frame.ppn.nunique()),
        "history_observations": args.history,
        "seed": args.seed,
        "split": {name: {"participants": int(len(ids)), "examples": int(masks[name].sum()), "person_ids": [int(v) for v in sorted(ids)]}
                  for name, ids in (("train", train_people), ("dev", dev_people), ("test", test_people))},
        "features": "prior state, prefix mean, log time gap, target time-of-day; no target state",
        "metrics": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"source_rows": len(frame), "participants": frame.ppn.nunique(),
                      "split_examples": {name: int(mask.sum()) for name, mask in masks.items()},
                      "test": results["test"]}, indent=2))


if __name__ == "__main__":
    main()
