"""Synthetic tests for decoder-path identity training; no licensed data."""

import copy
import unittest

import torch

from run_psycharchives_geometry_training import identity_loss, independent_window_phi
from run_psycharchives_sequence_manifold import SequenceManifold


class GeometryTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = SequenceManifold(22)
        self.block = {"x": torch.randn(4, 12, 22),
                      "observed": torch.ones(4, 12, 22),
                      "lengths": torch.full((4,), 12, dtype=torch.long)}

    def test_windows_do_not_use_middle_records(self):
        first, last = independent_window_phi(self.model, self.block)
        altered = dict(self.block)
        altered["x"] = self.block["x"].clone()
        altered["x"][:, 5:7] += 100
        first_changed, last_changed = independent_window_phi(self.model, altered)
        torch.testing.assert_close(first, first_changed)
        torch.testing.assert_close(last, last_changed)

    def test_normalized_path_loss_is_scale_invariant(self):
        original = identity_loss(self.model, self.block, "decoder_path_identity")
        scaled = copy.deepcopy(self.model)
        with torch.no_grad():
            scaled.decoder[-1].weight.mul_(5)
            scaled.decoder[-1].bias.mul_(5)
        result = identity_loss(scaled, self.block, "decoder_path_identity")
        torch.testing.assert_close(original, result, atol=1e-6, rtol=1e-6)

    def test_geometry_loss_reaches_decoder_and_slow_encoder(self):
        loss = identity_loss(self.model, self.block, "decoder_path_identity")
        loss.backward()
        self.assertGreater(float(self.model.decoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.slow.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
