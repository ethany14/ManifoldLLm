"""Scratch-trained causal language components, independent of manifold research."""

from torch import Tensor, nn


class GRUBackbone(nn.Module):
    """Embed tokens and compute causal hidden states without research-specific heads."""

    def __init__(self, vocabulary_size: int, embedding_dim: int = 32,
                 hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.recurrent = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def encode_tokens(self, tokens: Tensor) -> Tensor:
        hidden, _ = self.recurrent(self.embedding(tokens))
        return hidden


class CausalLanguageModel(GRUBackbone):
    """Plain next-token model for students who do not need affective geometry."""

    def __init__(self, vocabulary_size: int, embedding_dim: int = 32,
                 hidden_dim: int = 64):
        super().__init__(vocabulary_size, embedding_dim, hidden_dim)
        self.token_head = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.token_head(self.encode_tokens(tokens))