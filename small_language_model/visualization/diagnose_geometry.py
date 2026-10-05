"""D1-D3 research diagnostics; no optimization or pretrained models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from ..data import TARGETS, load_emobank
from ..geometry import decode_affect, decoder_geometry, graph_distances, segment_lengths
from ..model import EmotionLanguageModel
from .plot_manifold import extract_points


@torch.no_grad()
def encode(model, dataset):
    batches = []
    for tokens, _, ratings in DataLoader(dataset, batch_size=32):
        result = model(tokens, torch.zeros_like(ratings, dtype=torch.float64))
        batches.append({name: result[name] for name in ("affect", "private", "features")})
    return {name: torch.cat([batch[name] for batch in batches]) for name in batches[0]}


def correlation(first, second):
    first, second = np.asarray(first), np.asarray(second)
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 3 or np.ptp(first[finite]) == 0 or np.ptp(second[finite]) == 0:
        return None
    return float(spearmanr(first[finite], second[finite]).statistic)


def nearest_overlap(first, second, neighbors=8):
    count = min(neighbors, len(first) - 1)
    if count < 1:
        return None
    def nearest(points):
        distances = torch.cdist(points, points)
        distances.fill_diagonal_(torch.inf)
        return distances.topk(count, largest=False).indices
    first_indices, second_indices = nearest(first), nearest(second)
    return float((first_indices[:, :, None] == second_indices[:, None, :]).any(dim=2).double().mean())


def diagnose(checkpoint_path: Path, output: Path, split="dev", csv_path=None,
             max_points=96, grid_size=17, neighbors=8, seed=42):
    if max_points < 3 or grid_size < 3 or neighbors < 1:
        raise ValueError("Require max_points/grid_size >= 3 and neighbors >= 1.")
    _, source_metadata = extract_points(checkpoint_path, split, csv_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    corpus = load_emobank(Path(source_metadata["csv"]), settings["max_tokens"],
                          settings["max_vocabulary"], settings["min_frequency"])
    model = EmotionLanguageModel(**checkpoint["config"]).double().eval()
    model.load_state_dict(checkpoint["state_dict"])
    training = encode(model, corpus.datasets["train"])
    observed = encode(model, corpus.datasets[split])
    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.randperm(len(training["affect"]), generator=generator)[:max_points]
    indices = torch.randperm(len(observed["affect"]), generator=generator)[:max_points]
    affect = observed["affect"][indices]
    labels = corpus.datasets[split].tensors[2][indices].double()
    train_affect = training["affect"][train_indices]
    if min(len(train_affect), len(affect)) < 3:
        raise ValueError("Need at least three training and evaluation points.")
    mean_private = training["private"].mean(dim=0)
    order = torch.linalg.vector_norm(training["private"] - mean_private, dim=1).argsort()
    references = {
        "train_mean": mean_private,
        "train_example_q25": training["private"][order[len(order) // 4]],
        "train_example_q75": training["private"][order[3 * len(order) // 4]],
    }
    local_rows, reference_reports = [], {}
    for name, private in references.items():
        spectra = []
        ranks = []
        for sample, point in zip(indices.tolist(), affect):
            local = decoder_geometry(model.decoder, point, private)
            singular = local["singular_values"].tolist()
            rank = int(local["rank"])
            spectra.append(singular)
            ranks.append(rank)
            row = {"id": corpus.ids[split][sample], "private_reference": name, "rank": rank,
                   "sigma_max": singular[0], "sigma_middle": singular[1], "sigma_min": singular[2],
                   "threshold": float(local["threshold"])}
            row.update({f"metric_{first}_{second}": float(local["metric"][first, second])
                        for first in range(3) for second in range(3)})
            local_rows.append(row)
        reference_reports[name] = {
            "private_code": private.tolist(), "full_rank_fraction": sum(rank == 3 for rank in ranks) / len(ranks),
            "median_singular_values": np.median(spectra, axis=0).tolist(),
        }

    nodes = torch.cat((train_affect, affect))
    graph, predecessors, components = graph_distances(model.decoder, nodes, mean_private, neighbors)
    pair_indices = np.triu_indices(len(affect), 1)
    graph_eval = graph[len(train_affect):, len(train_affect):][pair_indices]
    euclidean = torch.cdist(affect, affect).numpy()[pair_indices]
    human = torch.cdist(labels, labels).numpy()[pair_indices]
    finite = np.isfinite(graph_eval)
    origin = len(train_affect)
    candidate_distances = torch.linalg.vector_norm(affect - affect[0], dim=1).numpy()
    candidate_distances[~np.isfinite(graph[origin, origin:])] = -1
    candidate_distances[0] = -1
    endpoint = origin + int(candidate_distances.argmax())
    route = []
    if endpoint != origin and np.isfinite(graph[origin, endpoint]):
        cursor = endpoint
        route = [cursor]
        while cursor != origin:
            cursor = int(predecessors[origin, cursor])
            if cursor < 0 or len(route) > len(nodes):
                raise RuntimeError("Invalid graph predecessor chain.")
            route.append(cursor)
        route.reverse()
    straight = torch.stack([nodes[origin], nodes[endpoint]])
    fraction = torch.linspace(0, 1, 65, dtype=torch.float64)[:, None]
    straight_dense = straight[0] + fraction * (straight[1] - straight[0])
    with torch.no_grad():
        mean_feature = training["features"].mean(dim=0)
        _, spectrum, basis = torch.linalg.svd(training["features"] - mean_feature, full_matrices=False)
        projection = basis[:3].T
        quantiles = torch.quantile(training["affect"], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64), dim=0)
        valence = torch.linspace(quantiles[0, 0], quantiles[2, 0], grid_size)
        arousal = torch.linspace(quantiles[0, 1], quantiles[2, 1], grid_size)
        grid_valence, grid_arousal = torch.meshgrid(valence, arousal, indexing="ij")
        grid = torch.stack((grid_valence.flatten(), grid_arousal.flatten(),
                            quantiles[1, 2].expand(grid_size ** 2)), dim=1).double()
        slice_features = decode_affect(model.decoder, grid, mean_private)
        projected = (slice_features - mean_feature) @ projection
        distances = torch.cdist(train_affect, train_affect)
        distances.fill_diagonal_(torch.inf)
        support_threshold = float(torch.quantile(distances.min(dim=1).values, 0.95))
        support = torch.cdist(grid, train_affect).min(dim=1).values <= support_threshold
        path_support = torch.cdist(straight_dense, train_affect).min(dim=1).values <= support_threshold
        projected_straight = (decode_affect(model.decoder, straight_dense, mean_private) - mean_feature) @ projection
        projected_route = ((decode_affect(model.decoder, nodes[route], mean_private) - mean_feature) @ projection
                           if route else None)
    slice_sigma = [float(decoder_geometry(model.decoder, point, mean_private)["singular_values"][-1])
                   for point in grid]
    training_label_std = corpus.datasets["train"].tensors[2].double().std(dim=0, unbiased=False)
    observed_std = observed["affect"].std(dim=0, unbiased=False)
    ratios = [float(value / target) if target > 0 else None
              for value, target in zip(observed_std, training_label_std)]
    straight_length = float(segment_lengths(model.decoder, straight[:1], straight[1:], mean_private, 64)[0])
    report = {
        "source": source_metadata, "diagnostic_departures": ["D1", "D2", "D3"],
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "variant": checkpoint.get("provenance", {}).get("variant", "legacy_baseline"),
        "settings": {"max_points": max_points, "grid_size": grid_size, "neighbors": neighbors, "seed": seed},
        "graph_node_ids": [{"split": "train", "id": corpus.ids["train"][index]} for index in train_indices.tolist()]
            + [{"split": split, "id": corpus.ids[split][index]} for index in indices.tolist()],
        "rank_tolerances": {"absolute": 1e-6, "relative": 1e-4},
        "private_references": reference_reports,
        "affect_std": dict(zip(TARGETS, observed_std.tolist())),
        "std_ratio_to_training_labels": dict(zip(TARGETS, ratios)),
        "low_spread_warning": any(value is not None and value < 0.25 for value in ratios),
        "neighborhood_overlap": nearest_overlap(affect, labels, neighbors),
        "pair_distances": {
            "pairs": len(human), "finite_graph_pairs": int(finite.sum()), "components": components,
            "euclidean_human_spearman": correlation(euclidean, human),
            "euclidean_human_spearman_same_finite_pairs": correlation(euclidean[finite], human[finite]),
            "graph_human_spearman": correlation(graph_eval, human),
        },
        "path": {"nodes": route, "graph_length": float(graph[origin, endpoint]) if route else None,
                 "straight_coordinate_path_length": straight_length,
                 "straight_near_training_fraction": float(path_support.double().mean()),
                 "warning": "Graph path is sample-constrained, not an exact geodesic or guaranteed shorter than a continuous straight path"},
        "slice": {"dominance": float(quantiles[1, 2]), "quantiles": quantiles.tolist(),
                  "support_threshold": support_threshold, "near_training_fraction": float(support.double().mean()),
                  "projection": "PCA fitted on training GRU features only; metrics use full feature space",
                  "projection_variance_fraction": float(spectrum[:3].square().sum() / spectrum.square().sum().clamp_min(1e-12))},
        "limits": ["Local rank is not global injectivity or emotional validity",
                   "Support uses distances to a training subsample, not a density guarantee",
                   "Graph uses train and evaluation coordinates but no evaluation labels for edge construction; transductive diagnostic only",
                   "Private codes are not proven non-affective; two empirical references test sensitivity",
                   "Diagnostic thresholds are heuristics, not thresholds from the article"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for filename, records in (
        ("local_geometry.csv", local_rows),
        ("decoder_slice.csv", [dict(zip(TARGETS, point.tolist()), sigma_min=sigma, near_training=bool(supported),
                                    pc1=float(position[0]), pc2=float(position[1]), pc3=float(position[2]))
                               for point, sigma, supported, position in zip(grid, slice_sigma, support, projected)]),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    torch.save({"nodes": nodes, "route": route, "straight": straight_dense,
                "private_reference": mean_private, "projection": projection, "projection_mean": mean_feature},
               output / "geometry_arrays.pt")
    draw(report, projected, grid, support, slice_sigma, grid_size, projected_straight, projected_route, output)
    return report


def draw(report, projected, grid, support, sigma, size, straight, route, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure = plt.figure(figsize=(16, 6), layout="constrained")
    title = f"{report['variant']} | epoch {report['source']['best_epoch']} | decoder geometry diagnostics"
    figure.suptitle(title, fontsize=14)
    axis = figure.add_subplot(1, 3, 1, projection="3d")
    mesh = projected.numpy().reshape(size, size, 3)
    masked = mesh.copy()
    masked[~support.numpy().reshape(size, size)] = np.nan
    axis.plot_wireframe(mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2], color="gray", alpha=0.25)
    axis.plot_surface(masked[:, :, 0], masked[:, :, 1], masked[:, :, 2], color="#078477", alpha=0.8)
    axis.set(title="Fixed-dominance slice (PCA projection)", xlabel="PC1", ylabel="PC2", zlabel="PC3")
    heat = figure.add_subplot(1, 3, 2)
    colors = heat.scatter(grid[:, 0], grid[:, 1], c=sigma, cmap="viridis", marker="s", s=22)
    heat.scatter(grid[~support, 0], grid[~support, 1], marker="x", c="gray", s=12)
    heat.set(title="Smallest Jacobian singular value\n(full feature space)", xlabel="Valence", ylabel="Arousal")
    figure.colorbar(colors, ax=heat, shrink=0.7)
    paths = figure.add_subplot(1, 3, 3, projection="3d")
    paths.plot(*straight.numpy().T, label="Straight coordinate path", color="#b34828")
    if route is not None:
        paths.plot(*route.numpy().T, label="Sample graph path", color="#087e8b", marker=".")
    paths.set(title="Decoded paths (PCA projection)", xlabel="PC1", ylabel="PC2", zlabel="PC3")
    paths.legend(fontsize=8)
    for panel in (axis, paths):
        for coordinate_axis in (panel.xaxis, panel.yaxis, panel.zaxis):
            coordinate_axis.set_major_locator(MaxNLocator(nbins=3))
        panel.tick_params(labelsize=8)
    figure.supxlabel("Gray grid / crosses: weak sampled-data support. A projected slice and local rank do not prove manifold recovery.", fontsize=10)
    figure.savefig(output / "decoder_geometry.png", dpi=160, bbox_inches="tight", pad_inches=0.35)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--max-points", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = diagnose(args.checkpoint, args.output_dir, args.split, args.csv,
                      args.max_points, args.grid_size, args.neighbors, args.seed)
    print(json.dumps({"variant": report["variant"], "private_references": report["private_references"],
                      "pair_distances": report["pair_distances"],
                      "low_spread_warning": report["low_spread_warning"]}, indent=2))


if __name__ == "__main__":
    main()