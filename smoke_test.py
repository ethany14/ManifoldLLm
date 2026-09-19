"""Small end-to-end deployment test that does not download an LLM."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run(*args: str, cwd: Path) -> None:
    print("+", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "artifacts" / "smoke_test"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    count, input_dim = 240, 96

    # A curved two-dimensional surface embedded in a larger observation space.
    valence = rng.uniform(-1, 1, count)
    arousal = rng.uniform(-1, 1, count)
    latent = np.column_stack(
        [valence, arousal, valence**2, arousal**2, valence * arousal,
         np.sin(np.pi * valence), np.cos(np.pi * arousal)]
    )
    projection = rng.normal(size=(latent.shape[1], input_dim))
    embeddings = (latent @ projection + rng.normal(0, 0.08, (count, input_dim))).astype(np.float32)
    emotion = np.where(
        valence > 0.25, np.where(arousal > 0, "joy", "calm"),
        np.where(valence < -0.25, np.where(arousal > 0, "anger", "sadness"), "neutral"),
    )
    frame = pd.DataFrame({
        "text": [f"synthetic example {i}" for i in range(count)],
        "valence": valence, "arousal": arousal, "emotion": emotion,
    })
    csv_path = output / "data.csv"
    embedding_path = output / "embeddings.npy"
    frame.to_csv(csv_path, index=False)
    np.save(embedding_path, embeddings)

    run("discover_manifold.py", "--embeddings", str(embedding_path),
        "--output-dir", str(output / "manifold"), "--neighbors", "12", cwd=root)
    run("train_adapter.py", "--csv", str(csv_path), "--embeddings", str(embedding_path),
        "--coordinates", str(output / "manifold" / "coordinates.npy"),
        "--output-dir", str(output / "model"), "--epochs", "20",
        "--hidden-dim", "128", "--batch-size", "64", cwd=root)
    run("predict.py", "--checkpoint", str(output / "model" / "adapter.pt"),
        "--embeddings", str(embedding_path), "--output", str(output / "predictions.json"), cwd=root)

    predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    assert len(predictions) == count
    assert len(predictions[0]["manifold"]) == 2
    print(f"PASS: {count} samples completed the full adapter pipeline on {output}")


if __name__ == "__main__":
    main()

