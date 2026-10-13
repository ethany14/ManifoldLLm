"""Checks for the shared language-only components."""

import unittest

import torch

from .core import CausalLanguageModel, GRUBackbone


class CoreTests(unittest.TestCase):
    def test_backbone_returns_causal_token_states(self):
        model = GRUBackbone(12)
        first = model.encode_tokens(torch.tensor([[2, 4, 5]]))
        second = model.encode_tokens(torch.tensor([[2, 4, 8]]))
        self.assertEqual(first.shape, (1, 3, 64))
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_plain_model_has_no_manifold_parameters(self):
        model = CausalLanguageModel(12)
        logits = model(torch.tensor([[2, 4, 5]]))
        self.assertEqual(logits.shape, (1, 3, 12))
        logits.square().mean().backward()
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertIsNotNone(model.token_head.weight.grad)
        self.assertEqual(set(model._modules), {"embedding", "recurrent", "token_head"})


if __name__ == "__main__":
    unittest.main()