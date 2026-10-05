"""Research extensions: decoder geometry, not equations from the source article."""

import math

import torch
from torch import Tensor, nn


def decode_affect(decoder: nn.Module, affect: Tensor, private: Tensor) -> Tensor:
    """Evaluate f(affect, private) with a fixed private vector or aligned batch."""
    if private.ndim == 1:
        private = private.expand(*affect.shape[:-1], -1)
    return decoder(torch.cat((affect, private), dim=-1))


def decoder_geometry(decoder: nn.Module, affect: Tensor, private: Tensor,
                     absolute_tolerance: float = 1e-6,
                     relative_tolerance: float = 1e-4) -> dict[str, Tensor]:
    """Full-feature-space pullback metric; never add a ridge to hide lost rank."""
    if affect.shape != (3,) or private.ndim != 1:
        raise ValueError("Expected one 3D affect vector and one private vector.")
    if any(not math.isfinite(value) or value < 0
           for value in (absolute_tolerance, relative_tolerance)):
        raise ValueError("Rank tolerances must be finite and nonnegative.")
    with torch.enable_grad():
        jacobian = torch.autograd.functional.jacobian(
            lambda coordinate: decode_affect(decoder, coordinate, private),
            affect.detach().clone().requires_grad_(True), vectorize=True,
        ).detach()
    singular_values = torch.linalg.svdvals(jacobian)
    threshold = torch.maximum(singular_values.new_tensor(absolute_tolerance),
                              relative_tolerance * singular_values[0])
    rank = (singular_values > threshold).sum()
    return {
        "jacobian": jacobian,
        "metric": jacobian.T @ jacobian,
        "singular_values": singular_values,
        "rank": rank,
        "threshold": threshold,
    }


@torch.no_grad()
def segment_lengths(decoder: nn.Module, starts: Tensor, ends: Tensor,
                    private: Tensor, subdivisions: int = 8) -> Tensor:
    """Approximate decoder-induced lengths of straight affect-coordinate segments."""
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("Expected matching [segments, 3] endpoints.")
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive.")
    fraction = torch.linspace(0, 1, subdivisions + 1, device=starts.device, dtype=starts.dtype)
    points = starts[:, None, :] + fraction[None, :, None] * (ends - starts)[:, None, :]
    features = decode_affect(decoder, points, private)
    return torch.linalg.vector_norm(features[:, 1:] - features[:, :-1], dim=-1).sum(dim=1)


def neighborhood_loss(affect: Tensor, labels: Tensor, neighbors: int = 8) -> Tensor:
    """Extension N1: match absolute VAD distances on human-label k-neighbor edges."""
    if len(affect) < 2:
        return affect.new_zeros(())
    with torch.no_grad():
        distances = torch.cdist(labels, labels)
        distances.fill_diagonal_(torch.inf)
        indices = distances.topk(min(neighbors, len(labels) - 1), largest=False).indices
        expected = torch.cdist(labels, labels).gather(1, indices)
    actual = torch.cdist(affect, affect).gather(1, indices)
    return (actual - expected).square().mean()


def graph_distances(decoder: nn.Module, affect: Tensor, private: Tensor,
                    neighbors: int = 8, subdivisions: int = 8):
    """Extension D2: sparse kNN graph approximation, not exact geodesics."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path

    if len(affect) < 2 or neighbors < 1:
        raise ValueError("Graph requires at least two points and positive neighbors.")
    distances = torch.cdist(affect, affect)
    distances.fill_diagonal_(torch.inf)
    nearest = distances.topk(min(neighbors, len(affect) - 1), largest=False).indices
    edges = sorted({tuple(sorted((index, int(other))))
                    for index in range(len(affect)) for other in nearest[index]})
    starts = torch.tensor([edge[0] for edge in edges], device=affect.device)
    ends = torch.tensor([edge[1] for edge in edges], device=affect.device)
    lengths = segment_lengths(decoder, affect[starts], affect[ends], private, subdivisions)
    rows = torch.cat((starts, ends)).cpu().numpy()
    columns = torch.cat((ends, starts)).cpu().numpy()
    values = torch.cat((lengths, lengths)).cpu().numpy()
    graph = csr_matrix((values, (rows, columns)), shape=(len(affect), len(affect)))
    components = connected_components(graph, directed=False, return_labels=False)
    distances, predecessors = shortest_path(graph, directed=False, return_predecessors=True)
    return distances, predecessors, int(components)