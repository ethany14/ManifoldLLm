"""Small fixture tests for split integrity, normalization and token alignment."""

import csv
from pathlib import Path
import tempfile
import unittest

import torch

from .data import load_emobank


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data.csv"
        self.rows = [
            ["train1", "Known words extra", "train", 0.5, -0.5, 0.1],
            ["train2", "Known words", "train", -0.5, 0.5, -0.1],
            ["dev1", "Unseenonly known", "dev", 0, 0, 0],
            ["test1", "Testonly known", "test", 1, -1, 1],
        ]

    def write_rows(self):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "text", "split", "valence", "arousal", "dominance"])
            writer.writerows(self.rows)

    def test_train_only_vocabulary_and_official_splits(self):
        self.write_rows()
        corpus = load_emobank(self.path, max_tokens=2, min_frequency=1)
        self.assertNotIn("unseenonly", corpus.vocabulary)
        self.assertNotIn("testonly", corpus.vocabulary)
        self.assertEqual(corpus.ids["test"], ["test1"])
        inputs, targets, ratings = corpus.datasets["train"].tensors
        torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])
        self.assertTrue(targets[:, -1].eq(3).all().item())
        self.assertEqual(corpus.statistics["train"]["truncated_rows"], 1)
        self.assertEqual(corpus.statistics["dev"]["unknown_tokens"], 1)
        torch.testing.assert_close(ratings[0], torch.tensor([0.5, -0.5, 0.1]))

    def test_invalid_targets_and_duplicate_ids_are_rejected(self):
        for invalid in (float("nan"), float("inf"), 3.0):
            self.rows[0][3] = invalid
            self.write_rows()
            with self.assertRaises(ValueError):
                load_emobank(self.path)
        self.rows[0][3] = 0.5
        self.rows[-1][0] = "train1"
        self.write_rows()
        with self.assertRaises(ValueError):
            load_emobank(self.path)

    def test_missing_split_is_rejected(self):
        self.rows.pop()
        self.write_rows()
        with self.assertRaises(ValueError):
            load_emobank(self.path)


if __name__ == "__main__":
    unittest.main()