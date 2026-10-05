"""Evaluation weighting and checkpoint round-trip checks."""

from io import BytesIO
import unittest

import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset

from .manifold_losses import LossWeights
from .model import EmotionLanguageModel
from .train import evaluate, parse_args, train_epoch


class TrainingTests(unittest.TestCase):
    def test_epoch_updates_weights_and_rejects_empty_loader(self):
        torch.manual_seed(42)
        model = EmotionLanguageModel(12)
        inputs = torch.tensor([[2, 4, 5], [2, 6, 0]])
        targets = torch.tensor([[4, 5, 3], [6, 3, 0]])
        vad = torch.zeros(2, 3)
        dataset = TensorDataset(inputs, targets, vad)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = model.token_head.weight.detach().clone()
        model.eval()
        loss = train_epoch(model, DataLoader(dataset, batch_size=2), optimizer, LossWeights())
        self.assertTrue(model.training)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.token_head.weight))
        empty = TensorDataset(inputs[:0], targets[:0], vad[:0])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            train_epoch(model, DataLoader(empty), optimizer, LossWeights())

    def test_default_paths_resolve_after_package_move(self):
        args = parse_args([])
        self.assertTrue(args.csv.is_file())
        self.assertEqual(args.output_dir.parent.parent.name, "Alexander")
        self.assertEqual(args.variant, "baseline")


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_token_counts_and_mean_posteriors(self):
        torch.manual_seed(42)
        model = EmotionLanguageModel(12)
        model.eval()
        inputs = torch.tensor([[2, 4, 5], [2, 6, 0], [2, 7, 8]])
        targets = torch.tensor([[4, 5, 3], [6, 3, 0], [7, 8, 3]])
        vad = torch.tensor([[0.5, 0.0, 0.1], [-0.5, 0.0, -0.1], [0.0, 0.0, 0.0]])
        dataset = TensorDataset(inputs, targets, vad)
        weights = LossWeights(cross=0, lexicon=0, disentanglement=0)
        first = evaluate(model, DataLoader(dataset, batch_size=2), weights)
        whole = evaluate(model, DataLoader(dataset, batch_size=3), weights)
        self.assertEqual(first["target_tokens"], 8)
        self.assertIsNone(first["vad"]["arousal"]["r2"])
        expected = functional.cross_entropy(
            model(inputs, vad)["logits"].reshape(-1, 12), targets.reshape(-1), ignore_index=0
        ).item()
        self.assertAlmostEqual(first["losses"]["language"], expected, places=5)
        self.assertAlmostEqual(first["losses"]["total"], whole["losses"]["total"], places=5)
        self.assertEqual(first, evaluate(model, DataLoader(dataset, batch_size=2), weights))

    def test_checkpoint_round_trip_and_vad_is_text_only(self):
        model = EmotionLanguageModel(12)
        model.eval()
        inputs = torch.tensor([[2, 4, 5]])
        expected = model(inputs, torch.zeros(1, 3))
        changed_condition = model(inputs, torch.ones(1, 3))
        torch.testing.assert_close(expected["affect"], changed_condition["affect"])
        checkpoint = BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = EmotionLanguageModel(12)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        restored.eval()
        actual = restored(inputs, torch.zeros(1, 3))
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name])


if __name__ == "__main__":
    unittest.main()