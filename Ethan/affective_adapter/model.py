from __future__ import annotations

import torch
from torch import nn


class ManifoldAdapter(nn.Module):
    """Small nonlinear map from frozen LLM states to manifold coordinates.

    The optional heads are auxiliary measurements. They encourage the learned
    coordinates to retain valence/arousal and categorical emotion information,
    but the LLM itself is never updated.
    """

    def __init__(
        self,
        input_dim: int,
        manifold_dim: int = 2,
        hidden_dim: int = 512,
        num_emotions: int = 0,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, manifold_dim),
        )
        self.affect_head = nn.Linear(manifold_dim, 2)
        self.emotion_head = (
            nn.Linear(manifold_dim, num_emotions) if num_emotions > 0 else None
        )

    def forward(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(hidden_state)
        # EmoBank targets are normalized to [-1, 1]. Bounding the auxiliary
        # head prevents physically invalid affect estimates during early training.
        output = {"z": z, "valence_arousal": torch.tanh(self.affect_head(z))}
        if self.emotion_head is not None:
            output["emotion_logits"] = self.emotion_head(z)
        return output


def local_distance_loss(
    predicted: torch.Tensor, target: torch.Tensor, neighbors: int = 8
) -> torch.Tensor:
    """Preserve target-manifold distances to each sample's local neighbours."""
    if target.shape[0] < 2:
        return predicted.new_zeros(())
    with torch.no_grad():
        target_dist = torch.cdist(target, target)
        k = min(neighbors + 1, target.shape[0])
        indices = target_dist.topk(k, largest=False).indices[:, 1:]
        mask = torch.zeros_like(target_dist, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        expected = target_dist[mask]
    actual = torch.cdist(predicted, predicted)[mask]
    scale = expected.mean().clamp_min(1e-6)
    return torch.mean(((actual - expected) / scale) ** 2)
