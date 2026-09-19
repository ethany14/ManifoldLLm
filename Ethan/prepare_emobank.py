"""Convert the official EmoBank CSV into the adapter's canonical schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SOURCE_URL = (
    "https://raw.githubusercontent.com/JULIELab/EmoBank/"
    "master/corpus/emobank.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/source/emobank_official.csv")
    parser.add_argument("--output-dir", default="data/emobank")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source)
    required = {"id", "split", "V", "A", "D", "text"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Official EmoBank columns missing: {missing}")
    annotation_fields = ["id", "split", "V", "A", "D"]
    if data[annotation_fields].isna().any().any():
        raise ValueError("Unexpected missing annotation values in official EmoBank fields.")
    missing_text_rows = int(data["text"].isna().sum())
    data = data.loc[data["text"].notna()].copy()
    if not data["split"].isin(["train", "dev", "test"]).all():
        raise ValueError("Unexpected split label in EmoBank.")
    for column in ["V", "A", "D"]:
        if not data[column].between(1.0, 5.0).all():
            raise ValueError(f"{column} contains a value outside the documented 1-5 range.")

    prepared = pd.DataFrame(
        {
            "id": data["id"].astype(str),
            "text": data["text"].astype(str),
            "valence": ((data["V"] - 3.0) / 2.0).round(4),
            "arousal": ((data["A"] - 3.0) / 2.0).round(4),
            "dominance": ((data["D"] - 3.0) / 2.0).round(4),
            "split": data["split"].astype(str),
            "source": "EmoBank",
        }
    )
    prepared.to_csv(output / "emobank_manifold.csv", index=False)
    for split in ["train", "dev", "test"]:
        prepared.loc[prepared["split"] == split].to_csv(
            output / f"emobank_{split}.csv", index=False
        )
    pilot_parts = []
    pilot_sizes = {"train": 800, "dev": 100, "test": 100}
    for split, count in pilot_sizes.items():
        rows = prepared.loc[prepared["split"] == split]
        pilot_parts.append(rows.sample(n=count, random_state=42))
    pd.concat(pilot_parts, ignore_index=True).to_csv(
        output / "emobank_pilot_1000.csv", index=False
    )

    summary = {
        "source_url": SOURCE_URL,
        "license": "CC BY-SA 4.0",
        "rows": int(len(prepared)),
        "excluded_missing_text_rows": missing_text_rows,
        "split_counts": prepared["split"].value_counts().to_dict(),
        "pilot_split_counts": pilot_sizes,
        "value_transform": "normalized = (official_score - 3) / 2",
        "normalized_range": [-1.0, 1.0],
        "citation": "Buechel and Hahn (2017), EmoBank, EACL 2017",
    }
    (output / "dataset_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
