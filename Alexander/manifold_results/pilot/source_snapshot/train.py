"""Train a scratch GRU with text-only variational manifold losses on EmoBank."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from Alexander.manifold_losses import LossWeights
from .data import DEFAULT_CSV, TARGETS, load_emobank
from .model import EmotionLanguageModel, generate, training_loss

VARIANTS = ("baseline", "neighborhood", "decoder", "decoder_neighborhood")


@torch.inference_mode()
def evaluate(model: EmotionLanguageModel, loader: DataLoader,
             weights: LossWeights, neighborhood_weight: float = 0.0) -> dict:
    model.eval()
    sums = {name: 0.0 for name in (
        "language", "anchor", "decorrelation", "reconstruction", "kl", "cross", "lexicon", "neighborhood"
    )}
    examples, tokens = 0, 0
    predictions, labels = [], []
    for inputs, targets, vad in loader:
        outputs = model(inputs, vad)
        losses = training_loss(outputs, targets, vad, weights, neighborhood_weight)
        batch_tokens = int(targets.ne(0).sum())
        tokens += batch_tokens
        examples += len(vad)
        for name in sums:
            sums[name] += float(losses[name]) * (batch_tokens if name == "language" else len(vad))
        predictions.append(outputs["affect"])
        labels.append(vad)
    if examples == 0 or tokens == 0:
        raise ValueError("Evaluation requires a nonempty dataset with target tokens.")
    means = {name: value / (tokens if name == "language" else examples)
             for name, value in sums.items()}
    means["total"] = (
        means["language"] + means["reconstruction"]
        + weights.supervision * (means["anchor"] + weights.lexicon * means["lexicon"])
        + weights.cross * means["cross"]
        + weights.disentanglement * means["decorrelation"]
        + weights.kl * weights.beta * means["kl"]
    )
    means["selection_total"] = means["total"]
    means["total"] += neighborhood_weight * means["neighborhood"]
    expected, predicted = torch.cat(labels), torch.cat(predictions)
    error = predicted - expected
    affect = {}
    for index, name in enumerate(TARGETS):
        residual = error[:, index].square().sum().item()
        variance = (expected[:, index] - expected[:, index].mean()).square().sum().item()
        affect[name] = {
            "mae": error[:, index].abs().mean().item(),
            "rmse": error[:, index].square().mean().sqrt().item(),
            "r2": 1 - residual / variance if variance > 0 else None,
        }
    return {
        "rows": examples, "target_tokens": tokens, "losses": means,
        "vad_conditioned_perplexity": math.exp(means["language"]) if means["language"] < 700 else None,
        "vad": affect,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-vocabulary", type=int, default=4000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--supervision-weight", type=float, default=5.0)
    parser.add_argument("--disentanglement-weight", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--neighborhood-weight", type=float, default=0.1)
    parser.add_argument("--skip-test", action="store_true", help="Use during development comparisons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "emobank")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 2:
        parser.error("epochs/patience must be positive and batch-size must be at least 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (
        args.supervision_weight, args.disentanglement_weight, args.kl_weight, args.beta,
        args.neighborhood_weight
    )):
        parser.error("loss weights must be finite and nonnegative")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    corpus = load_emobank(args.csv, args.max_tokens, args.max_vocabulary, args.min_frequency)
    weights = LossWeights(cross=0.0, lexicon=0.0, supervision=args.supervision_weight,
                          disentanglement=args.disentanglement_weight, kl=args.kl_weight,
                          beta=args.beta)
    neighborhood_weight = args.neighborhood_weight if "neighborhood" in args.variant else 0.0
    config = {"vocabulary_size": len(corpus.vocabulary), "embedding_dim": 32,
              "hidden_dim": 64, "private_dim": 8,
              "conditioning_mode": "decoder" if "decoder" in args.variant else "linear"}
    model = EmotionLanguageModel(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(corpus.datasets["train"], batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(corpus.datasets["dev"], batch_size=args.batch_size)
    initial_dev = evaluate(model, dev_loader, weights, neighborhood_weight)
    initial_condition = model.emotion_condition.weight.detach().clone()
    initial_decoder = model.decoder[0].weight.detach().clone()
    best_score, best_epoch, stale = float("inf"), 0, 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, examples = 0.0, 0
        for inputs, targets, vad in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = training_loss(model(inputs, vad), targets, vad, weights, neighborhood_weight)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Training produced a non-finite loss.")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(losses["total"].detach()) * len(vad)
            examples += len(vad)
        dev_metrics = evaluate(model, dev_loader, weights, neighborhood_weight)
        score = dev_metrics["losses"]["selection_total"]
        if not math.isfinite(score):
            raise RuntimeError("Development evaluation produced a non-finite loss.")
        history.append({"epoch": epoch, "train_sampled_total": loss_sum / examples,
                        "dev": dev_metrics})
        print(f"epoch={epoch} train_sampled_total={loss_sum / examples:.4f} dev_selection={score:.4f}")
        if score < best_score - 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    dev_metrics = evaluate(model, dev_loader, weights, neighborhood_weight)
    test_metrics, baseline = None, None
    training_vad = corpus.datasets["train"].tensors[2]
    if not args.skip_test:
        test_loader = DataLoader(corpus.datasets["test"], batch_size=args.batch_size)
        test_metrics = evaluate(model, test_loader, weights, neighborhood_weight)
        test_vad = corpus.datasets["test"].tensors[2]
        baseline_error = training_vad.mean(dim=0) - test_vad
        baseline = {name: {"mae": baseline_error[:, index].abs().mean().item(),
                           "rmse": baseline_error[:, index].square().mean().sqrt().item()}
                    for index, name in enumerate(TARGETS)}
    settings = {name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()}
    data_metadata = {
        "csv": str(args.csv.resolve()), "sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "source": "EmoBank, Buechel and Hahn (2017)", "license": "CC BY-SA 4.0",
        "splits": corpus.statistics,
        "tokenizer": "lowercase Unicode words and individual punctuation; train-only vocabulary",
        "max_tokens": args.max_tokens,
    }
    provenance = {
        "article": "https://aclanthology.org/2026.acl-long.1929.pdf",
        "registry": "Alexander/small_language_model/ARTICLE_DIFFERENCES.md",
        "variant": args.variant,
        "active_departures": ["A1", "A2", "A3", "A4", "A5"]
            + (["N1"] if neighborhood_weight else [])
            + (["G1"] if config["conditioning_mode"] == "decoder" else []),
        "effective_neighborhood_weight": neighborhood_weight,
        "selection": "Development baseline objective, excluding N1 in every variant",
        "source_hashes": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("model.py", "geometry.py", "train.py", "data.py")},
        "paper_loss_source_sha256": hashlib.sha256(
            (Path(__file__).parent.parent / "manifold_losses.py").read_bytes()).hexdigest(),
        "difference_registry_sha256": hashlib.sha256(
            (Path(__file__).parent / "ARTICLE_DIFFERENCES.md").read_bytes()).hexdigest(),
        "software": {"python": sys.version, "torch": str(torch.__version__)},
    }
    report = {
        "provenance": provenance,
        "data": data_metadata, "settings": settings, "loss_weights": asdict(weights),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch, "initial_dev": initial_dev,
        "dev": dev_metrics, "test": test_metrics, "history": history,
        "test_training_mean_vad_baseline": baseline,
        "emotion_weight_change_l2": float(
            (model.emotion_condition.weight.detach() - initial_condition).norm()),
        "decoder_weight_change_l2": float((model.decoder[0].weight.detach() - initial_decoder).norm()),
        "inactive_paper_terms": {"cross": "No paired modalities", "lexicon": "No NRC-VAD pseudo-labels"},
        "evaluation": "Posterior means, not sampled ELBO; perplexity is VAD-conditioned and includes unknown tokens",
        "generated": {name: generate(model, corpus.vocabulary, ratings)
                      for name, ratings in {"positive": [0.7, 0.5, 0.5],
                                            "negative": [-0.7, -0.5, -0.5]}.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "state_dict": model.state_dict(), "config": config,
                "vocabulary": corpus.vocabulary, "data": data_metadata,
                "provenance": provenance,
                "settings": settings, "loss_weights": asdict(weights), "best_epoch": best_epoch},
               args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "parameters": report["parameters"],
                      "test": test_metrics, "generated": report["generated"]}, indent=2))
    print(f"Saved scratch-trained model to {args.output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()