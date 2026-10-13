"""Local EmoBank CSV ingestion with train-only vocabulary construction."""

from collections import Counter
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re

import torch
from torch.utils.data import TensorDataset


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TARGETS = ["valence", "arousal", "dominance"]
DEFAULT_CSV = Path(__file__).resolve().parents[2] / "Ethan" / "data" / "emobank" / "emobank_manifold.csv"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


@dataclass
class Corpus:
    vocabulary: list[str]
    datasets: dict[str, TensorDataset]
    ids: dict[str, list[str]]
    statistics: dict[str, dict]


def load_emobank(path: Path, max_tokens: int = 64, max_vocabulary: int = 4000,
                 min_frequency: int = 2) -> Corpus:
    if max_tokens < 1 or max_vocabulary < 5 or min_frequency < 1:
        raise ValueError("Require max_tokens >= 1, max_vocabulary >= 5 and min_frequency >= 1.")
    rows = {split: [] for split in ("train", "dev", "test")}
    seen_ids = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "text", "split", *TARGETS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            if row["split"] not in rows or not row["text"] or not row["text"].strip():
                raise ValueError(f"Invalid split or empty text on CSV line {line}.")
            if not row["id"] or row["id"] in seen_ids:
                raise ValueError(f"Missing or duplicate ID on CSV line {line}.")
            seen_ids.add(row["id"])
            try:
                ratings = [float(row[target]) for target in TARGETS]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid VAD on CSV line {line}.") from error
            if not all(math.isfinite(value) and -1 <= value <= 1 for value in ratings):
                raise ValueError(f"VAD must be finite and normalized to [-1, 1] on line {line}.")
            rows[row["split"]].append((row["id"], tokenize(row["text"]), ratings))
    if any(not records for records in rows.values()):
        raise ValueError("All three official splits must be nonempty.")
    counts = Counter(word for _, words, _ in rows["train"] for word in words[:max_tokens])
    ordered = sorted(counts, key=lambda word: (-counts[word], word))
    vocabulary = SPECIAL_TOKENS + [word for word in ordered if counts[word] >= min_frequency][
        :max_vocabulary - len(SPECIAL_TOKENS)
    ]
    lookup = {word: index for index, word in enumerate(vocabulary)}
    datasets, ids, statistics = {}, {}, {}
    for split, records in rows.items():
        inputs = torch.zeros(len(records), max_tokens + 1, dtype=torch.long)
        targets = torch.zeros_like(inputs)
        unknown, retained = 0, 0
        for index, (_, words, _) in enumerate(records):
            encoded = [lookup.get(word, 1) for word in words[:max_tokens]]
            inputs[index, :len(encoded) + 1] = torch.tensor([2] + encoded)
            targets[index, :len(encoded) + 1] = torch.tensor(encoded + [3])
            unknown += encoded.count(1)
            retained += len(encoded)
        ratings = torch.tensor([record[2] for record in records], dtype=torch.float32)
        datasets[split] = TensorDataset(inputs, targets, ratings)
        ids[split] = [record[0] for record in records]
        statistics[split] = {
            "rows": len(records), "retained_text_tokens": retained,
            "unknown_tokens": unknown, "unknown_rate": unknown / max(retained, 1),
            "truncated_rows": sum(len(words) > max_tokens for _, words, _ in records),
        }
    return Corpus(vocabulary, datasets, ids, statistics)