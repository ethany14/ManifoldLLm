"""Differentiable losses from Li et al. (ACL 2026), equations 4-11.

Squared norms sum over features and average over examples. Modality and
ordered-pair sums are performed by the caller. See README.md for attribution.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LossWeights:
    cross: float = 1.0
    supervision: float = 5.0
    lexicon: float = 0.2
    disentanglement: float = 0.1
    kl: float = 0.01
    beta: float = 1.0


def reparameterize(mean: Tensor, log_variance: Tensor, sample: bool = True) -> Tensor:
    """Equation 4; use the posterior mean for deterministic evaluation."""
    if not sample:
        return mean
    return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)


def squared_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Batch-mean squared Euclidean norm for equations 5, 7 and section 3.5."""
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("Expected matching [batch, features] tensors.")
    return (predicted - target).square().sum(dim=1).mean()


def gaussian_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
    """Closed-form KL(diagonal Gaussian || N(0, I)), before beta in eq. 10."""
    return 0.5 * (
        mean.square() + log_variance.exp() - 1.0 - log_variance
    ).sum(dim=1).mean()


def cross_covariance(shared: Tensor, private: Tensor) -> Tensor:
    """Equation 8: centered, unbiased sample cross-covariance."""
    if shared.ndim != 2 or private.ndim != 2 or len(shared) != len(private):
        raise ValueError("Expected two matrices with the same batch size.")
    if len(shared) < 2:
        raise ValueError("Cross-covariance requires at least two examples.")
    shared_centered = shared - shared.mean(dim=0, keepdim=True)
    private_centered = private - private.mean(dim=0, keepdim=True)
    return shared_centered.T @ private_centered / (len(shared) - 1)


def disentanglement_loss(shared: Tensor, private: Tensor) -> Tensor:
    """Equation 9 for one modality; decorrelation is not independence."""
    return cross_covariance(shared, private).square().sum()


def total_objective(terms: dict, weights: LossWeights) -> Tensor:
    """Equations 10-11; terms['kl'] is unweighted, so beta is applied once."""
    supervision = terms["ground_truth"] + weights.lexicon * terms["lexicon"]
    return (
        terms["self"]
        + weights.cross * terms["cross"]
        + weights.supervision * supervision
        + weights.disentanglement * terms["disentanglement"]
        + weights.kl * weights.beta * terms["kl"]
    )