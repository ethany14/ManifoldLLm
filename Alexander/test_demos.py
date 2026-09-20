"""Fast CPU checks for equations and the teaching demonstrations."""

import unittest

import torch

from manifold_losses import (
    LossWeights,
    cross_covariance,
    disentanglement_loss,
    gaussian_kl,
    reparameterize,
    squared_error,
    total_objective,
)


class EquationTests(unittest.TestCase):
    def test_standard_normal_has_zero_kl(self):
        zeros = torch.zeros(4, 3)
        self.assertEqual(gaussian_kl(zeros, zeros).item(), 0.0)
        self.assertAlmostEqual(gaussian_kl(torch.ones(4, 3), zeros).item(), 1.5)

    def test_covariance_is_centered_and_unbiased(self):
        shared = torch.tensor([[1.0], [2.0], [3.0]])
        private = 2 * shared + 100
        self.assertAlmostEqual(cross_covariance(shared, private).item(), 2.0)
        self.assertAlmostEqual(disentanglement_loss(shared, private).item(), 4.0)
        with self.assertRaises(ValueError):
            cross_covariance(shared[:1], private[:1])

    def test_squared_error_sums_features(self):
        self.assertEqual(squared_error(torch.ones(4, 3), torch.zeros(4, 3)).item(), 3.0)

    def test_reparameterization_and_gradients(self):
        torch.manual_seed(42)
        mean = torch.ones(8, 3, requires_grad=True)
        log_variance = torch.zeros(8, 3, requires_grad=True)
        self.assertTrue(torch.equal(reparameterize(mean, log_variance, False), mean))
        reparameterize(mean, log_variance).square().mean().backward()
        self.assertGreater(mean.grad.abs().sum().item(), 0)
        self.assertGreater(log_variance.grad.abs().sum().item(), 0)

    def test_weighting_applies_beta_once(self):
        terms = {name: torch.tensor(1.0) for name in (
            "self", "cross", "ground_truth", "lexicon", "disentanglement", "kl"
        )}
        weights = LossWeights(cross=2, supervision=3, lexicon=4,
                              disentanglement=5, kl=6, beta=7)
        self.assertEqual(total_objective(terms, weights).item(), 65.0)


if __name__ == "__main__":
    unittest.main()