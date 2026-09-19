from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep downloaded model files inside this deployed project unless the user has
# deliberately configured another Hugging Face cache.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "artifacts" / "hf_cache"))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract states from a frozen LLM.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    if args.text_column not in frame:
        raise ValueError(f"Missing text column: {args.text_column}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    model.requires_grad_(False)

    vectors: list[np.ndarray] = []
    texts = frame[args.text_column].fillna("").astype(str).tolist()
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.batch_size)):
            tokens = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            device = next(model.parameters()).device
            tokens = {key: value.to(device) for key, value in tokens.items()}
            result = model(**tokens, output_hidden_states=True, return_dict=True)
            hidden = result.hidden_states[args.layer]
            pooled = mean_pool(hidden, tokens["attention_mask"])
            vectors.append(pooled.float().cpu().numpy())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.concatenate(vectors, axis=0))
    print(f"Saved {len(texts)} frozen embeddings with shape {vectors[0].shape[1]} to {output}")


if __name__ == "__main__":
    main()
