"""Download pinned official dair-ai/emotion `split` Parquet files.

The source data is kept local rather than redistributed in this repository.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
SOURCES = {
    "train": "10817f0f2ea42358bc62f69a09dfb8bd71701727df6d5a387bea742f3ea06417",
    "validation": "c70f0e660b5ebd1ea9a37d2a851f516f08a6d6477cdfc11be204e22a2f1102fd",
    "test": "6f8407fa1ca9c310f55781f082ed73812f6551e8dda2c61973123a121869245b",
}


def main() -> None:
    target = Path("data/dair_emotion/source")
    target.mkdir(parents=True, exist_ok=True)
    for split, expected in SOURCES.items():
        destination = target / f"{split}.parquet"
        url = (
            "https://huggingface.co/datasets/dair-ai/emotion/resolve/"
            f"{REVISION}/split/{split}-00000-of-00001.parquet"
        )
        if not destination.exists():
            urllib.request.urlretrieve(url, destination)
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Hash mismatch for {destination}: {actual}")
        print(f"Verified {destination}: {actual}")


if __name__ == "__main__":
    main()
