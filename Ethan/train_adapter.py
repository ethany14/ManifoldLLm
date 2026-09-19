from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from affective_adapter.model import ManifoldAdapter, local_distance_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the nonlinear out-of-sample adapter.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance-weight", type=float, default=0.25)
    parser.add_argument("--affect-weight", type=float, default=0.50)
    parser.add_argument("--emotion-weight", type=float, default=0.25)
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-value", default="train")
    parser.add_argument("--valid-value", default="dev")
    parser.add_argument("--test-value", default="test")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = pd.read_csv(args.csv)
    x = np.load(args.embeddings).astype(np.float32)
    z = np.load(args.coordinates).astype(np.float32)
    if not (len(frame) == len(x) == len(z)):
        raise ValueError("CSV, embeddings, and coordinates must have identical row counts.")

    has_affect = {"valence", "arousal"}.issubset(frame.columns)
    affect = (
        frame[["valence", "arousal"]].to_numpy(np.float32)
        if has_affect
        else np.zeros((len(frame), 2), np.float32)
    )
    has_emotion = "emotion" in frame.columns
    label_encoder = LabelEncoder() if has_emotion else None
    labels = (
        label_encoder.fit_transform(frame["emotion"].astype(str)).astype(np.int64)
        if label_encoder is not None
        else np.zeros(len(frame), np.int64)
    )

    indices = np.arange(len(frame))
    if args.split_column in frame.columns:
        split = frame[args.split_column].astype(str)
        train_idx = indices[split.eq(args.train_value).to_numpy()]
        valid_idx = indices[split.eq(args.valid_value).to_numpy()]
        test_idx = indices[split.eq(args.test_value).to_numpy()]
        if not len(train_idx) or not len(valid_idx):
            raise ValueError("The official split must contain both training and validation rows.")
    else:
        train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)
        test_idx = np.array([], dtype=int)
    input_scaler = StandardScaler().fit(x[train_idx])
    x = input_scaler.transform(x).astype(np.float32)

    def dataset(idx: np.ndarray) -> TensorDataset:
        return TensorDataset(*[torch.from_numpy(a[idx]) for a in (x, z, affect, labels)])

    train_loader = DataLoader(dataset(train_idx), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset(valid_idx), batch_size=args.batch_size)
    test_loader = DataLoader(dataset(test_idx), batch_size=args.batch_size) if len(test_idx) else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ManifoldAdapter(
        input_dim=x.shape[1],
        manifold_dim=z.shape[1],
        hidden_dim=args.hidden_dim,
        num_emotions=len(label_encoder.classes_) if label_encoder is not None else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    def loss_for(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, dict[str, float]]:
        xb, zb, ab, yb = (item.to(device) for item in batch)
        result = model(xb)
        coordinate_loss = F.mse_loss(result["z"], zb)
        distance_loss = local_distance_loss(result["z"], zb)
        affect_loss = F.mse_loss(result["valence_arousal"], ab) if has_affect else xb.new_zeros(())
        emotion_loss = (
            F.cross_entropy(result["emotion_logits"], yb) if has_emotion else xb.new_zeros(())
        )
        total = coordinate_loss + args.distance_weight * distance_loss
        total = total + args.affect_weight * affect_loss + args.emotion_weight * emotion_loss
        metrics = {
            "total": total.item(), "coordinate": coordinate_loss.item(),
            "distance": distance_loss.item(), "affect": affect_loss.item(),
            "emotion": emotion_loss.item(),
        }
        return total, metrics

    best = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            values = [loss_for(batch)[0].item() for batch in valid_loader]
        validation = float(np.mean(values))
        if validation < best:
            best = validation
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} validation_loss={validation:.5f}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "config": {
            "input_dim": x.shape[1], "manifold_dim": z.shape[1],
            "hidden_dim": args.hidden_dim,
            "num_emotions": len(label_encoder.classes_) if label_encoder is not None else 0,
        },
        "input_mean": input_scaler.mean_.astype(np.float32),
        "input_scale": input_scaler.scale_.astype(np.float32),
        "emotion_classes": label_encoder.classes_.tolist() if label_encoder is not None else [],
    }
    torch.save(checkpoint, output / "adapter.pt")
    test_loss = None
    if test_loader is not None:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_loss = float(np.mean([loss_for(batch)[0].item() for batch in test_loader]))
    metrics = {
        "best_validation_loss": best,
        "test_loss": test_loss,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(valid_idx)),
        "test_rows": int(len(test_idx)),
    }
    (output / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved adapter to {output / 'adapter.pt'}")


if __name__ == "__main__":
    main()
