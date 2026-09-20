"""Leakage-controlled next-prompt behavior baselines on openESM Beck data.

No training outcome or feature from the next prompt enters a predictor. This
experiment uses a new held-out-person cohort, not the published article cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data/person_situation/source/0060_beck_ts.tsv"
OUT = HERE / "artifacts/beck_behavior_pilot"
SEED = 42
TARGET = "studying"
MAX_GAP_HOURS = 8.0
MIN_PROMPTS = 15
CALIBRATION_PROMPTS = 10
SITUATION = ["mating", "positivity", "intellect", "negativity", "adversity", "sociality", "duty", "deception"]
AFFECT = ["guilty", "attentive", "angry", "content", "goal_directed", "happy", "afraid", "proud", "excited", "purposeful"]
ACTIVITY = ["in_class", "late", "sick", "procrastinating", "bored_schoolwork", "excited_schoolwork", "anxious_schoolwork", "internet", "tired", "sleeping"]
def scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def build_pairs() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(SOURCE, sep="\t")
    source_rows = len(raw)
    source_people = int(raw["id"].nunique())
    personality = raw.columns[raw.columns.get_loc("optimistic_setback") : raw.columns.get_loc("mating")].tolist()
    required = {"id", "date", "hour", "minute", TARGET, *SITUATION, *AFFECT, *ACTIVITY}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    ts = pd.to_datetime(raw["date"]) + pd.to_timedelta(raw["hour"], unit="h") + pd.to_timedelta(raw["minute"], unit="m")
    raw = raw.assign(timestamp=ts).sort_values(["id", "timestamp"]).reset_index(drop=True)
    if raw.duplicated(["id", "timestamp"]).any():
        raise ValueError("Duplicate participant/timestamp keys")
    eligible = raw.groupby("id").size()
    eligible = eligible[eligible >= MIN_PROMPTS].index
    raw = raw[raw["id"].isin(eligible)].copy()
    g = raw.groupby("id", sort=False)
    raw["prompt_index"] = g.cumcount()
    raw["next_target"] = g[TARGET].shift(-1)
    raw["gap_hours"] = (g["timestamp"].shift(-1) - raw["timestamp"]).dt.total_seconds() / 3600
    raw["history_rate"] = g[TARGET].transform(lambda s: s.fillna(0).cumsum() / s.notna().cumsum().replace(0, np.nan))
    raw["hour_sin"] = np.sin(2 * np.pi * raw["hour"] / 24)
    raw["hour_cos"] = np.cos(2 * np.pi * raw["hour"] / 24)
    raw["day_of_week"] = raw["timestamp"].dt.dayofweek
    pairs = raw[
        (raw["prompt_index"] >= CALIBRATION_PROMPTS - 1)
        & raw[TARGET].notna()
        & raw["next_target"].notna()
        & (raw["gap_hours"] > 0)
        & (raw["gap_hours"] <= MAX_GAP_HOURS)
    ].copy()
    pairs["next_target"] = pairs["next_target"].astype(int)
    audit = {
        "source_rows": source_rows,
        "source_people": source_people,
        "eligible_people_min_15_prompts": int(len(eligible)),
        "pairs_after_calibration_gap_and_target_rules": int(len(pairs)),
        "pair_people": int(pairs["id"].nunique()),
        "personality_item_columns": personality,
        "personality_item_nonmissing_fraction": float(raw[personality].notna().mean().mean()),
        "gap_hours_median": float(pairs["gap_hours"].median()),
        "target_is_self_report": True,
    }
    return pairs, audit


def main() -> None:
    pairs, audit = build_pairs()
    ids = sorted(pairs["id"].unique())
    train_ids, hold_ids = train_test_split(ids, test_size=0.4, random_state=SEED)
    dev_ids, test_ids = train_test_split(hold_ids, test_size=0.5, random_state=SEED)
    split_ids = {"train": sorted(map(int, train_ids)), "dev": sorted(map(int, dev_ids)), "test": sorted(map(int, test_ids))}
    assert not (set(train_ids) & set(dev_ids) | set(train_ids) & set(test_ids) | set(dev_ids) & set(test_ids))
    split = {k: pairs[pairs["id"].isin(v)].copy() for k, v in split_ids.items()}
    train_prevalence = float(split["train"]["next_target"].mean())
    basic = [TARGET, "history_rate", "hour_sin", "hour_cos", "day_of_week"]
    context = basic + SITUATION + AFFECT + ACTIVITY
    results = {"audit": audit, "protocol": {"seed": SEED, "target": TARGET, "gap_hours_max": MAX_GAP_HOURS, "minimum_prompts": MIN_PROMPTS, "calibration_prompts": CALIBRATION_PROMPTS, "primary_metric": "Brier; lower is better", "source": "openESM 0060_beck_ts.tsv", "split_person_ids": split_ids}, "metrics": {}}
    for name in ["train", "dev", "test"]:
        part = split[name]
        y = part["next_target"].to_numpy()
        results["metrics"][name] = {
            "global_prevalence": scores(y, np.full(len(y), train_prevalence)),
            "persistence": scores(y, part[TARGET].to_numpy() * 0.98 + 0.01),
            "person_history": scores(y, part["history_rate"].fillna(train_prevalence).to_numpy() * 0.98 + 0.01),
        }
        results["metrics"][name]["people"] = int(part["id"].nunique())
    for model_name, features in {"logistic_history_time": basic, "logistic_context": context}.items():
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=SEED))
        model.fit(split["train"][features], split["train"]["next_target"])
        for name, part in split.items():
            p = model.predict_proba(part[features])[:, 1]
            results["metrics"][name][model_name] = scores(part["next_target"].to_numpy(), p)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {m: round(v["brier"], 4) for m, v in results["metrics"][k].items() if isinstance(v, dict)} for k in ["dev", "test"]}, indent=2))
    print(f"Saved {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
