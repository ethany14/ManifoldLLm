"""Check real coordinate extraction, provenance guards, and image rendering."""

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from ..data import load_emobank
from ..model import EmotionLanguageModel
from .plot_manifold import extract_points, render


class VisualizationTests(unittest.TestCase):
    def test_coordinates_rendering_and_csv_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sentences.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "text", "split", "valence", "arousal", "dominance"])
                for index, split in enumerate(("train", "dev", "test")):
                    writer.writerow([str(index), "a calm day", split, 0.2, -0.4, 0.1])
            corpus = load_emobank(source, 8, 20, 1)
            model = EmotionLanguageModel(len(corpus.vocabulary))
            model.eval()
            checkpoint = root / "model.pt"
            torch.save({
                "format_version": 2, "config": {"vocabulary_size": len(corpus.vocabulary)},
                "state_dict": model.state_dict(), "vocabulary": corpus.vocabulary,
                "settings": {"max_tokens": 8, "max_vocabulary": 20, "min_frequency": 1},
                "best_epoch": 1,
                "data": {"csv": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "splits": corpus.statistics, "source": "test fixture", "license": "test fixture"},
            }, checkpoint)
            records, metadata = extract_points(checkpoint)
            tokens, _, ratings = corpus.datasets["dev"].tensors
            expected = model(tokens, torch.zeros_like(ratings))["affect"][0].detach()
            actual = torch.tensor([records[0][f"learned_{name}"]
                                   for name in ("valence", "arousal", "dominance")])
            torch.testing.assert_close(actual, expected)
            self.assertEqual(records[0]["id"], "1")
            self.assertEqual(records, extract_points(checkpoint)[0])
            image = render(records, metadata, root / "output")
            self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(image.stat().st_size, 10000)
            self.assertTrue((root / "output" / "coordinates.csv").is_file())
            with source.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "CSV hash"):
                extract_points(checkpoint)


if __name__ == "__main__":
    unittest.main()