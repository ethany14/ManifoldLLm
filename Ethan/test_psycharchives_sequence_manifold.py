"""Synthetic invariance tests; no licensed observations are needed."""

import unittest

import torch

from run_psycharchives_sequence_manifold import SequenceManifold, next_mask


class SequenceManifoldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = SequenceManifold(22).eval()
        self.x = torch.randn(1, 4, 22)
        self.observed = torch.ones_like(self.x)

    def test_future_record_cannot_change_previous_predictions(self):
        before = self.model(self.x, self.observed)
        altered = self.x.clone()
        altered[:, 3] += 10
        after = self.model(altered, self.observed)
        torch.testing.assert_close(before["phi"][:, :3], after["phi"][:, :3])
        torch.testing.assert_close(before["predicted_y"][:, :3], after["predicted_y"][:, :3])

    def test_masked_cue_value_has_no_effect(self):
        observed = self.observed.clone()
        observed[:, 1, 5] = 0
        before = self.model(self.x, observed)
        altered = self.x.clone()
        altered[:, 1, 5] += 1000
        after = self.model(altered, observed)
        torch.testing.assert_close(before["phi"], after["phi"])
        torch.testing.assert_close(before["predicted_y"], after["predicted_y"])

    def test_order_changes_history_encoding(self):
        before = self.model(self.x, self.observed)
        swapped = self.x.clone()
        swapped[:, 0], swapped[:, 1] = self.x[:, 1].clone(), self.x[:, 0].clone()
        after = self.model(swapped, self.observed)
        self.assertFalse(torch.allclose(before["phi"][:, 1], after["phi"][:, 1]))

    def test_next_mask_excludes_final_and_padding(self):
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
        expected = torch.tensor([[True, True, False], [True, False, False]])
        torch.testing.assert_close(next_mask(valid), expected)


if __name__ == "__main__":
    unittest.main()
