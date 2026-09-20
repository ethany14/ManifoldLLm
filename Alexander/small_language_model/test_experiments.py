"""Experiment summary and diagnostic provenance regression checks."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from .compare_manifold_variants import summarize
from .data import load_emobank
from .model import EmotionLanguageModel
from .visualization.diagnose_geometry import diagnose


class ExperimentTests(unittest.TestCase):
    def test_paired_summary_uses_matching_seeds(self):
        rows = []
        for seed, variant, value in ((1, "baseline", 1.0), (2, "baseline", 3.0),
                                     (1, "decoder", 0.5), (2, "decoder", 2.0)):
            row = {"seed": seed, "variant": variant}
            for name in ("dev_vad_rmse", "dev_perplexity", "full_rank_fraction", "neighborhood_overlap",
                         "euclidean_human_spearman", "graph_human_spearman"):
                row[name] = value
            rows.append(row)
        summary = summarize(rows)[1]
        self.assertEqual(summary["dev_vad_rmse_mean"], 1.25)
        self.assertEqual(summary["dev_vad_rmse_paired_delta_mean"], -0.75)
        self.assertEqual(summarize(rows[2:])[0]["dev_vad_rmse_paired_delta_mean"], None)

    def test_diagnostic_outputs_do_not_mutate_checkpoint(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "text", "split", "valence", "arousal", "dominance"])
                for split in ("train", "dev", "test"):
                    for index, word in enumerate(("happy", "calm", "sad", "worried")):
                        writer.writerow([f"{split}_{index}", f"i feel {word}", split,
                                         index * 0.3 - 0.5, 0.5 - index * 0.2, index * 0.1])
            corpus = load_emobank(source, 8, 30, 1)
            config = {"vocabulary_size": len(corpus.vocabulary), "conditioning_mode": "decoder"}
            model = EmotionLanguageModel(**config)
            checkpoint = root / "model.pt"
            torch.save({
                "format_version": 2, "config": config, "state_dict": model.state_dict(),
                "vocabulary": corpus.vocabulary, "best_epoch": 1,
                "settings": {"max_tokens": 8, "max_vocabulary": 30, "min_frequency": 1},
                "data": {"csv": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "splits": corpus.statistics, "source": "fixture", "license": "fixture"},
            }, checkpoint)
            before = checkpoint.read_bytes()
            report = diagnose(checkpoint, root / "report", max_points=4, grid_size=3, neighbors=2)
            self.assertEqual(before, checkpoint.read_bytes())
            self.assertEqual(report["source"]["split"], "dev")
            self.assertEqual(report["diagnostic_departures"], ["D1", "D2", "D3"])
            self.assertEqual(len(report["graph_node_ids"]), 8)
            json.dumps(report, allow_nan=False)
            self.assertGreater((root / "report" / "decoder_geometry.png").stat().st_size, 10000)
            self.assertTrue((root / "report" / "local_geometry.csv").is_file())


if __name__ == "__main__":
    unittest.main()