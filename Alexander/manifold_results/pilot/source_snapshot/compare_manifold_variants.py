"""Matched, offline experiments with archived methods and paired differences."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import torch

from .data import DEFAULT_CSV, TARGETS
from .train import VARIANTS
from .visualization.diagnose_geometry import diagnose


def summarize(rows):
    fields = ("dev_vad_rmse", "dev_perplexity", "full_rank_fraction", "neighborhood_overlap",
              "euclidean_human_spearman", "graph_human_spearman")
    aggregate = []
    for variant in dict.fromkeys(row["variant"] for row in rows):
        group = [row for row in rows if row["variant"] == variant]
        result = {"variant": variant, "seeds": len(group)}
        for field in fields:
            values = [row[field] for row in group if row[field] is not None]
            result[f"{field}_n"] = len(values)
            result[f"{field}_mean"] = statistics.mean(values) if values else None
            result[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else None
            paired = [row[field] - baseline[field] for row in group for baseline in rows
                      if baseline["variant"] == "baseline" and baseline["seed"] == row["seed"]
                      and row[field] is not None and baseline[field] is not None]
            result[f"{field}_paired_delta_mean"] = statistics.mean(paired) if paired else None
        aggregate.append(result)
    return aggregate


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).parent / "artifacts" / "manifold_runs")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).parent / "manifold_results")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_points < 3 or args.grid_size < 3:
        parser.error("epochs must be positive; max-points and grid-size must be at least 3")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("seeds and variants must be unique")
    source = args.csv.resolve()
    run_root, report_root = args.run_dir.resolve(), args.report_dir.resolve()
    if report_root.exists() and any(report_root.iterdir()):
        parser.error("Report directory is not empty; choose a new path to preserve prior results")
    for seed in args.seeds:
        for variant in args.variants:
            if (run_root / f"seed_{seed}" / variant / "model.pt").exists():
                parser.error("A run checkpoint already exists; choose a new --run-dir")
    report_root.mkdir(parents=True, exist_ok=True)
    archive = report_root / "source_snapshot"
    archive.mkdir()
    package = Path(__file__).parent
    for name in ("model.py", "train.py", "geometry.py", "data.py", "compare_manifold_variants.py",
                 "ARTICLE_DIFFERENCES.md", "requirements.txt"):
        shutil.copy2(package / name, archive / name)
    shutil.copy2(package.parent / "manifold_losses.py", archive / "manifold_losses.py")
    shutil.copy2(package / "visualization" / "diagnose_geometry.py", archive / "diagnose_geometry.py")
    shutil.copy2(package / "visualization" / "requirements.txt", archive / "visualization_requirements.txt")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (report_root / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.set_num_threads(1)
    rows = []
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = run_root / f"seed_{seed}" / variant
            command = [sys.executable, "-m", "Alexander.small_language_model.train", "--csv", str(source),
                       "--variant", variant, "--seed", str(seed), "--epochs", str(args.epochs),
                       "--output-dir", str(run_dir)]
            if not args.evaluate_test:
                command.append("--skip-test")
            subprocess.run(command, cwd=root, check=True)
            destination = report_root / f"seed_{seed}" / variant
            diagnostics = diagnose(run_dir / "model.pt", destination, max_points=args.max_points,
                                   grid_size=args.grid_size, seed=seed)
            shutil.copy2(run_dir / "metrics.json", destination / "training_metrics.json")
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            row = {
                "seed": seed, "variant": variant, "best_epoch": metrics["best_epoch"],
                "dev_vad_rmse": statistics.mean(metrics["dev"]["vad"][target]["rmse"] for target in TARGETS),
                "dev_perplexity": metrics["dev"]["vad_conditioned_perplexity"],
                "full_rank_fraction": diagnostics["private_references"]["train_mean"]["full_rank_fraction"],
                "neighborhood_overlap": diagnostics["neighborhood_overlap"],
                "euclidean_human_spearman": diagnostics["pair_distances"]["euclidean_human_spearman_same_finite_pairs"],
                "graph_human_spearman": diagnostics["pair_distances"]["graph_human_spearman"],
                "low_spread_warning": diagnostics["low_spread_warning"],
            }
            rows.append(row)
            with (report_root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            (report_root / "aggregate.json").write_text(
                json.dumps(summarize(rows), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()