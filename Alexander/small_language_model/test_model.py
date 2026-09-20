"""CPU checks for scratch initialization, causality and trainable emotion weights."""

import unittest

import torch

from .model import EmotionLanguageModel, training_loss


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = EmotionLanguageModel(12)

    def test_training_updates_language_and_emotion_weights(self):
        tokens = torch.tensor([[2, 4, 5], [2, 6, 7], [2, 8, 9]])
        targets = torch.tensor([[4, 5, 3], [6, 7, 3], [8, 9, 3]])
        vad = torch.tensor([[0.8, 0.7, 0.5], [-0.8, -0.5, -0.6], [0.4, -0.7, 0.3]])
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        losses = training_loss(self.model(tokens, vad), targets, vad)
        self.assertTrue(torch.isfinite(losses["total"]).item())
        losses["total"].backward()
        optimizer.step()
        for name in ("embedding.weight", "emotion_condition.weight", "affect_head.weight"):
            self.assertFalse(torch.equal(before[name], self.model.state_dict()[name]))

    def test_predictions_are_causal(self):
        vad = torch.zeros(1, 3)
        first = self.model(torch.tensor([[2, 4, 5]]), vad)["logits"]
        second = self.model(torch.tensor([[2, 4, 9]]), vad)["logits"]
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_emotion_changes_token_scores(self):
        tokens = torch.tensor([[2, 4]])
        positive = self.model(tokens, torch.ones(1, 3))["logits"]
        negative = self.model(tokens, -torch.ones(1, 3))["logits"]
        self.assertFalse(torch.equal(positive, negative))

    def test_variational_manifold_gradients(self):
        tokens = torch.tensor([[2, 4, 5], [2, 6, 7]])
        targets = torch.tensor([[4, 5, 3], [6, 7, 3]])
        vad = torch.tensor([[0.8, 0.6, 0.4], [-0.8, -0.5, -0.4]])
        outputs = self.model(tokens, vad)
        self.assertFalse(torch.equal(outputs["affect"], outputs["affect_sample"]))
        losses = training_loss(outputs, targets, vad)
        losses["total"].backward()
        for name in ("private_head.weight", "affect_log_variance.weight",
                     "private_log_variance.weight", "decoder.0.weight"):
            gradient = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertEqual(losses["cross"].item(), 0)
        self.assertEqual(losses["lexicon"].item(), 0)

    def test_reconstruction_trains_both_codes(self):
        self.model.eval()
        outputs = self.model(torch.tensor([[2, 4], [2, 6]]), torch.zeros(2, 3))
        self.assertFalse(outputs["features"].requires_grad)
        (outputs["reconstruction"] - outputs["features"]).square().sum().backward()
        for head in (self.model.affect_head, self.model.private_head):
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)

    def test_evaluation_is_deterministic_and_padding_invariant(self):
        self.model.eval()
        tokens = torch.tensor([[2, 4, 5]])
        vad = torch.zeros(1, 3)
        first = self.model(tokens, vad)
        padded = self.model(torch.tensor([[2, 4, 5, 0, 0]]), vad)
        torch.testing.assert_close(first["affect"], first["affect_sample"])
        torch.testing.assert_close(first["affect"], padded["affect"])
        losses = training_loss(first, torch.tensor([[4, 5, 3]]), vad)
        self.assertTrue(torch.isfinite(losses["total"]).item())
        self.assertEqual(losses["decorrelation"].item(), 0)


if __name__ == "__main__":
    unittest.main()