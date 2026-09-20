"""Inference-only Beck manifold-personality engineering prototype.

Inputs are standardized Beck feature arrays, not raw language. The slow
posterior was trained from first-ten-prompt anchors; a changed anchor is a
re-estimation, not a calibrated online Bayesian update.
"""

from __future__ import annotations

import numpy as np
import torch
from datetime import datetime
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

from run_beck_riemannian_vae import SlowFastVAE


class PersonalityManifold:
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model = SlowFastVAE(checkpoint["state_dim"], checkpoint["time_dim"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.state_dim = int(checkpoint["state_dim"])
        self.time_dim = int(checkpoint["time_dim"])
        self.checkpoint_seed = int(checkpoint["seed"])
        self.training_contract = checkpoint.get("training_contract", "first-ten-prompt-anchor")
        self.preprocessing = checkpoint.get("preprocessing")

    @staticmethod
    def _vector(value, expected, label):
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (expected,) or not np.isfinite(array).all():
            raise ValueError(f"{label} must have {expected} finite standardized values")
        return torch.from_numpy(array)

    def posterior(self, anchor):
        anchor = self._vector(anchor, self.state_dim, "anchor")
        with torch.no_grad():
            mean, logvar = self.model.slow(anchor[None]).chunk(2, dim=1)
        return mean[0].numpy(), logvar[0].clamp(-8, 5).numpy()

    def state_posterior(self, deviation):
        deviation = self._vector(deviation, self.state_dim, "deviation")
        with torch.no_grad():
            mean, logvar = self.model.fast(deviation[None]).chunk(2, dim=1)
        return mean[0].numpy(), logvar[0].clamp(-8, 5).numpy()

    def local_metric(self, phi, ridge=1e-4):
        p = self._vector(phi, 4, "phi").requires_grad_(True)
        jac = torch.autograd.functional.jacobian(
            lambda v: self.model.decode_person(v[None])[0], p,
            vectorize=True, strategy="forward-mode")
        metric = jac.T @ jac + ridge * torch.eye(4)
        return metric.detach().numpy()

    def decoded_path_length(self, phi_a, phi_b, steps=8):
        a = self._vector(phi_a, 4, "phi_a")
        b = self._vector(phi_b, 4, "phi_b")
        fractions = torch.linspace(0, 1, steps + 1)
        path = a[None] * (1 - fractions[:, None]) + b[None] * fractions[:, None]
        with torch.no_grad():
            decoded = self.model.decode_person(path)
        return float(torch.linalg.vector_norm(torch.diff(decoded, dim=0), dim=1).sum())

    def infer(self, anchor, deviation, time, record_label="observation"):
        a = self._vector(anchor, self.state_dim, "anchor")
        d = self._vector(deviation, self.state_dim, "deviation")
        t = self._vector(time, self.time_dim, "time")
        phi, phi_logvar = self.posterior(a.numpy())
        z, z_logvar = self.state_posterior(d.numpy())
        latent = torch.cat((torch.from_numpy(phi), torch.from_numpy(z), t))[None]
        with torch.no_grad():
            probability = torch.sigmoid(self.model.behavior(latent))[0, 0].item()
        eigenvalues = np.linalg.eigvalsh(self.local_metric(phi))
        return {
            "record_label": record_label,
            "checkpoint_seed": self.checkpoint_seed,
            "slow_person_posterior": {"mean": phi.tolist(),
                                      "std": np.exp(0.5 * phi_logvar).tolist()},
            "fast_state_posterior": {"mean": z.tolist(),
                                     "std": np.exp(0.5 * z_logvar).tolist()},
            "local_metric_eigenvalues": eigenvalues.tolist(),
            "predicted_next_self_reported_studying": float(probability),
            "scope": "Beck self-report prototype; not a validated personality identity",
        }

    def update_fast_state(self, previous_snapshot, new_deviation, new_time,
                          record_label="new_observation"):
        """Keep the slow posterior fixed; only re-estimate momentary state."""
        d = self._vector(new_deviation, self.state_dim, "new_deviation")
        t = self._vector(new_time, self.time_dim, "new_time")
        phi = np.asarray(previous_snapshot["slow_person_posterior"]["mean"], dtype=np.float32)
        z, z_logvar = self.state_posterior(d.numpy())
        latent = torch.cat((torch.from_numpy(phi), torch.from_numpy(z), t))[None]
        with torch.no_grad():
            probability = torch.sigmoid(self.model.behavior(latent))[0, 0].item()
        updated = dict(previous_snapshot)
        updated["record_label"] = record_label
        updated["fast_state_posterior"] = {"mean": z.tolist(),
                                            "std": np.exp(0.5 * z_logvar).tolist()}
        updated["predicted_next_self_reported_studying"] = float(probability)
        updated["update_note"] = "Slow posterior held fixed; this is not a learned long-term update."
        return updated

    def _infer_accumulated_history(self, sums, counts, n_prompts, current, timestamp, label):
        if self.training_contract != "variable-prefix-history" or self.preprocessing is None:
            raise ValueError("This checkpoint was not trained on variable-length history prefixes")
        prep = self.preprocessing
        medians = np.asarray(prep["state_imputer_median"], dtype=np.float32)
        center = np.asarray(prep["state_scaler_mean"], dtype=np.float32)
        scale = np.asarray(prep["state_scaler_scale"], dtype=np.float32)
        prefix = np.divide(sums, counts, out=medians.copy(), where=counts > 0)
        current = np.where(np.isfinite(current), current, medians)
        anchor_std = (prefix - center) / scale
        deviation_std = (current - center) / scale - anchor_std
        when = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
        if not isinstance(when, datetime):
            raise ValueError("timestamp must be a datetime or ISO timestamp string")
        studying = prep["state_columns"].index("studying")
        history_rate = float(sums[studying] / counts[studying]) if counts[studying] else 0.0
        # Match build_pairs(): cyclical time uses the recorded integer hour.
        hour = when.hour
        raw_time = np.asarray([history_rate, np.sin(2 * np.pi * hour / 24),
                               np.cos(2 * np.pi * hour / 24), when.weekday()], dtype=np.float32)
        time_std = ((raw_time - np.asarray(prep["time_scaler_mean"], dtype=np.float32))
                    / np.asarray(prep["time_scaler_scale"], dtype=np.float32))
        snapshot = self.infer(anchor_std, deviation_std, time_std, label)
        snapshot["history_observation_count"] = int(n_prompts)
        snapshot["history_accumulator"] = {"sum": sums.tolist(), "count": counts.tolist()}
        snapshot["update_note"] = "Slow and fast posteriors re-estimated from the enlarged observed history."
        return snapshot

    def infer_from_raw_history(self, state_history, timestamp, record_label="history_snapshot"):
        """Use raw Beck STATE values in chronological order, including current prompt."""
        history = np.asarray(state_history, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != self.state_dim or len(history) < 10:
            raise ValueError("state_history must have at least ten rows and all STATE columns")
        if np.isinf(history).any():
            raise ValueError("state_history contains infinity")
        sums = np.nansum(history, axis=0).astype(np.float32)
        counts = np.isfinite(history).sum(axis=0).astype(np.float32)
        return self._infer_accumulated_history(sums, counts, len(history), history[-1],
                                               timestamp, record_label)

    def update_history(self, previous_snapshot, new_state, timestamp,
                       record_label="later_history_snapshot"):
        """Streaming prefix update; both q(phi) and q(z) are re-estimated."""
        if self.training_contract != "variable-prefix-history":
            raise ValueError("This checkpoint was not trained on variable-length history prefixes")
        current = np.asarray(new_state, dtype=np.float32)
        if current.shape != (self.state_dim,) or np.isinf(current).any():
            raise ValueError("new_state must contain all raw STATE features (NaN allowed)")
        accumulator = previous_snapshot["history_accumulator"]
        sums = np.asarray(accumulator["sum"], dtype=np.float32) + np.nan_to_num(current, nan=0.0)
        counts = np.asarray(accumulator["count"], dtype=np.float32) + np.isfinite(current)
        n = int(previous_snapshot["history_observation_count"]) + 1
        return self._infer_accumulated_history(sums, counts, n, current, timestamp, record_label)

    def approximate_geodesic(self, phi_a, phi_b, reference_phi, neighbors=12):
        """Shortest path in a reference graph with decoder-induced edge lengths."""
        reference = np.asarray(reference_phi, dtype=np.float32)
        if reference.ndim != 2 or reference.shape[1] != 4 or len(reference) <= neighbors:
            raise ValueError("reference_phi must be N x 4 with N > neighbors")
        a = np.asarray(phi_a, dtype=np.float32)
        b = np.asarray(phi_b, dtype=np.float32)
        all_phi = np.vstack((reference, a[None], b[None]))
        n = len(reference)
        search = NearestNeighbors(n_neighbors=neighbors + 1).fit(reference)
        edges = set()
        for i, row in enumerate(search.kneighbors(reference, return_distance=False)):
            edges.update(tuple(sorted((i, int(j)))) for j in row if i != j)
        for query_index, query in ((n, a), (n + 1, b)):
            row = search.kneighbors(query[None], n_neighbors=neighbors, return_distance=False)[0]
            edges.update(tuple(sorted((query_index, int(j)))) for j in row)
        rows, cols, lengths = [], [], []
        for i, j in sorted(edges):
            length = self.decoded_path_length(all_phi[i], all_phi[j])
            if not np.isfinite(length) or length <= 0:
                raise ValueError("Invalid edge length")
            rows.extend((i, j)); cols.extend((j, i)); lengths.extend((length, length))
        graph = csr_matrix((lengths, (rows, cols)), shape=(n + 2, n + 2))
        distance = float(shortest_path(graph, directed=False, indices=n)[n + 1])
        if not np.isfinite(distance):
            raise ValueError("Disconnected reference graph")
        return {"approximate_graph_geodesic": distance,
                "straight_decoded_path_length": self.decoded_path_length(a, b),
                "euclidean_coordinate_distance": float(np.linalg.norm(a - b)),
                "reference_points": n, "neighbors": neighbors,
                "units": "standardized decoder-output path length"}

    @staticmethod
    def llm_condition(snapshot):
        """Structured side information only; does not generate text."""
        return {
            "long_term_estimate": snapshot["slow_person_posterior"],
            "current_state_estimate": snapshot["fast_state_posterior"],
            "uncertainty_is_explicit": True,
            "usage_rule": "Treat as uncertain, revisable context; do not claim identity or inner experience.",
        }
