from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from affective_adapter.model import ManifoldAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Map saved LLM embeddings to affect space.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ManifoldAdapter(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = np.load(args.embeddings).astype(np.float32)
    x = (x - checkpoint["input_mean"]) / checkpoint["input_scale"]
    with torch.inference_mode():
        result = model(torch.from_numpy(x))
        z = result["z"].numpy()
        affect = result["valence_arousal"].numpy()
        emotion = None
        if "emotion_logits" in result:
            indices = result["emotion_logits"].argmax(1).numpy()
            emotion = [checkpoint["emotion_classes"][i] for i in indices]
    records = []
    for i in range(len(x)):
        records.append({
            "manifold": z[i].tolist(),
            "valence": float(affect[i, 0]),
            "arousal": float(affect[i, 1]),
            "emotion": emotion[i] if emotion else None,
        })
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(f"Saved {len(records)} predictions to {args.output}")


if __name__ == "__main__":
    main()

