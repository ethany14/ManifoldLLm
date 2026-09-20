"""Plot checkpoint posterior means against human EmoBank VAD annotations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import TARGETS, load_emobank
from ..model import EmotionLanguageModel


ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def extract_points(checkpoint_path: Path, split: str = "dev",
                   csv_path: Path | None = None, batch_size: int = 32) -> tuple[list[dict], dict]:
    if split not in ("train", "dev", "test") or batch_size < 1:
        raise ValueError("Choose train/dev/test and a positive batch size.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Use a version-2 EmoBank variational checkpoint, not the old toy model.")
    source = csv_path if csv_path is not None else Path(checkpoint["data"]["csv"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != checkpoint["data"]["sha256"]:
        raise ValueError("CSV hash differs from the checkpoint dataset; use its original CSV.")
    settings = checkpoint["settings"]
    corpus = load_emobank(source, settings["max_tokens"], settings["max_vocabulary"],
                          settings["min_frequency"])
    if corpus.vocabulary != checkpoint["vocabulary"]:
        raise ValueError("Reconstructed vocabulary does not match the checkpoint.")
    model = EmotionLanguageModel(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predicted = []
    for tokens, _, ratings in DataLoader(corpus.datasets[split], batch_size=batch_size):
        predicted.append(model(tokens, torch.zeros_like(ratings))["affect"])
    coordinates = torch.cat(predicted)
    if not torch.isfinite(coordinates).all():
        raise ValueError("Model produced non-finite affect coordinates.")
    expected = corpus.datasets[split].tensors[2]
    with source.open(encoding="utf-8", newline="") as handle:
        text_by_id = {row["id"]: row["text"] for row in csv.DictReader(handle)}
    records = []
    for index, identifier in enumerate(corpus.ids[split]):
        record = {"id": identifier, "split": split, "text": text_by_id[identifier]}
        for dimension, target in enumerate(TARGETS):
            record[f"learned_{target}"] = float(coordinates[index, dimension])
            record[f"human_{target}"] = float(expected[index, dimension])
        records.append(record)
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "csv": str(source.resolve()), "csv_sha256": digest,
        "split": split, "rows": len(records), "best_epoch": checkpoint["best_epoch"],
        "training_rows": checkpoint["data"]["splits"]["train"]["rows"],
        "coordinates": "Deterministic 3D affect posterior means; no PCA/Isomap or smoothing",
        "color": "Human-annotated valence, used for coloring only",
        "note": "A point cloud of the learned representation, not proof of a recovered manifold",
        "source": checkpoint["data"]["source"], "license": checkpoint["data"]["license"],
    }
    return records, metadata


def render(records: list[dict], metadata: dict, output_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    output_dir.mkdir(parents=True, exist_ok=True)
    learned = [[record[f"learned_{target}"] for target in TARGETS] for record in records]
    human = [[record[f"human_{target}"] for target in TARGETS] for record in records]
    limits = []
    for dimension in range(3):
        values = [point[dimension] for point in learned + human]
        lower, upper = min(-1.0, min(values)), max(1.0, max(values))
        padding = (upper - lower) * 0.03
        limits.append((lower - padding, upper + padding))
    colors = [point[0] for point in human]
    figure = plt.figure(figsize=(13, 10), facecolor="#f6f8fa", layout="constrained")
    label = "Pilot" if metadata["training_rows"] < 8062 else "EmoBank"
    figure.suptitle(
        f"{label} affect space | {metadata['split']} split | {metadata['rows']} sentences\n"
        f"Scratch-trained GRU | checkpoint epoch {metadata['best_epoch']}",
        fontsize=18, fontweight="bold",
    )
    axes = []
    for column, (points, title) in enumerate((
        (learned, "Learned affect coordinates"), (human, "Human emotion ratings")
    )):
        horizontal, vertical, depth = zip(*points)
        axis = figure.add_subplot(2, 2, column + 1, projection="3d")
        scatter = axis.scatter(horizontal, vertical, depth, c=colors, cmap="coolwarm",
                               norm=Normalize(-1, 1), s=25, alpha=0.8, edgecolors="none")
        axis.set(title=title, xlabel="Valence", ylabel="Arousal", zlabel="Dominance",
                 xlim=limits[0], ylim=limits[1], zlim=limits[2])
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=23, azim=-55)
        for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate_axis.set_major_locator(MaxNLocator(nbins=4))
        axes.append(axis)
        projection = figure.add_subplot(2, 2, column + 3)
        projection.scatter(horizontal, vertical, c=colors, cmap="coolwarm",
                           norm=Normalize(-1, 1), s=25, alpha=0.75, edgecolors="none")
        projection.set(title=f"{title}: valence/arousal view", xlabel="Valence",
                       ylabel="Arousal", xlim=limits[0], ylim=limits[1])
        projection.set_aspect("equal", adjustable="box")
        projection.axhline(0, color="#9ca3af", linewidth=0.6)
        projection.axvline(0, color="#9ca3af", linewidth=0.6)
        projection.grid(alpha=0.15)
        axes.append(projection)
    figure.colorbar(scatter, ax=axes, shrink=0.65, pad=0.03,
                    label="Human valence (color only): negative to positive")
    figure.supxlabel("Same axis limits; no smoothing. Point clouds are not proof of manifold recovery.",
                     fontsize=11)
    image_path = output_dir / "manifold.png"
    figure.savefig(image_path, dpi=170)
    plt.close(figure)
    with (output_dir / "coordinates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT.parent / "artifacts" / "emobank" / "model.pt")
    parser.add_argument("--csv", type=Path, help="Original CSV relocated to another path, with identical bytes")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint not found. Train the model first or provide --checkpoint.")
    torch.set_num_threads(1)
    records, metadata = extract_points(args.checkpoint, args.split, args.csv)
    image_path = render(records, metadata, args.output_dir)
    print(f"Saved {len(records)} sentence coordinates and image: {image_path}")


if __name__ == "__main__":
    main()