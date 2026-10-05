"""Analytic checks of diagnostic extensions and their failure cases."""

import unittest

import torch
from torch import nn

from .geometry import decoder_geometry, graph_distances, neighborhood_loss, segment_lengths
from .model import EmotionLanguageModel, training_loss


class GeometryTests(unittest.TestCase):
    def make_decoder(self, scales):
        decoder = nn.Linear(5, 3, bias=False).double()
        with torch.no_grad():
            decoder.weight.zero_()
            decoder.weight[:, :3] = torch.diag(torch.tensor(scales, dtype=torch.float64))
        return decoder

    def test_affine_metric_and_lengths(self):
        decoder = self.make_decoder([2, 3, 4])
        affect, private = torch.zeros(3, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
        report = decoder_geometry(decoder, affect, private)
        torch.testing.assert_close(report["metric"], torch.diag(torch.tensor([4., 9., 16.], dtype=torch.float64)))
        self.assertEqual(int(report["rank"]), 3)
        lengths = segment_lengths(decoder, affect[None], torch.ones(1, 3, dtype=torch.float64), private)
        self.assertAlmostEqual(float(lengths[0]), 29 ** 0.5)
        self.assertIsNone(decoder.weight.grad)

    def test_collapsed_decoder_is_not_repaired(self):
        decoder = self.make_decoder([1, 1, 0])
        report = decoder_geometry(decoder, torch.zeros(3, dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
        self.assertEqual(int(report["rank"]), 2)
        self.assertEqual(float(torch.linalg.det(report["metric"])), 0)

    def test_graph_lengths_and_disconnected_components(self):
        decoder = self.make_decoder([2, 3, 4])
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=torch.float64)
        distances, _, components = graph_distances(decoder, points, torch.zeros(2, dtype=torch.float64), 1)
        self.assertEqual(components, 1)
        self.assertAlmostEqual(distances[0, 2], 4.0)
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.], [10., 0., 0.], [11., 0., 0.]], dtype=torch.float64)
        distances, _, components = graph_distances(decoder, points, torch.zeros(2, dtype=torch.float64), 1)
        self.assertEqual(components, 2)
        self.assertEqual(distances[0, 3], float("inf"))

    def test_neighborhood_loss_matches_labels_and_has_gradients(self):
        labels = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.assertEqual(float(neighborhood_loss(labels, labels)), 0)
        affect = (labels * 0.5).requires_grad_(True)
        loss = neighborhood_loss(affect, labels)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertGreater(float(affect.grad.abs().sum()), 0)
        self.assertEqual(float(neighborhood_loss(labels[:1], labels[:1])), 0)

    def test_decoder_conditioning_is_causal_and_trains_decoder(self):
        torch.manual_seed(42)
        model = EmotionLanguageModel(12, conditioning_mode="decoder")
        model.eval()
        condition = torch.tensor([[0.6, -0.3, 0.4]])
        first = model(torch.tensor([[2, 4, 5]]), condition)
        future_changed = model(torch.tensor([[2, 4, 9]]), condition)
        torch.testing.assert_close(first["logits"][:, :2], future_changed["logits"][:, :2])
        neutral = model(torch.tensor([[2, 4, 5]]), torch.zeros_like(condition))
        torch.testing.assert_close(first["affect"], neutral["affect"])
        self.assertFalse(torch.equal(first["logits"], neutral["logits"]))
        losses = training_loss(first, torch.tensor([[4, 5, 3]]), condition)
        losses["language"].backward()
        self.assertGreater(float(model.decoder[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(model.emotion_condition.weight.grad)

    def test_optional_extension_changes_only_named_loss(self):
        model = EmotionLanguageModel(12).eval()
        targets = torch.tensor([[4, 5, 3], [6, 7, 3]])
        vad = torch.tensor([[0.7, 0.4, 0.2], [-0.7, -0.4, -0.2]])
        outputs = model(torch.tensor([[2, 4, 5], [2, 6, 7]]), vad)
        baseline = training_loss(outputs, targets, vad)
        extension = training_loss(outputs, targets, vad, neighborhood_weight=0.2)
        torch.testing.assert_close(extension["total"], baseline["total"] + 0.2 * extension["neighborhood"])


if __name__ == "__main__":
    unittest.main()