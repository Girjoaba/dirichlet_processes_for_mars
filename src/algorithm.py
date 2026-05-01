"""
Streaming Hierarchical DP-Means.

Single-class, NumPy/SciPy reference implementation of the streaming
Hierarchical Dirichlet-Process Means algorithm used as the clustering
backbone for global-scale Mars CRISM hyperspectral analysis.

The algorithm processes a stream of "tiles" (3-D feature cubes of shape
(H, W, D), with NaN marking invalid pixels). For each tile it runs a
local DP-Means pass to discover tile-specific centers, then merges them
into a persistent set of global centers. After the stream has been
consumed, ``predict`` assigns every pixel of any tile to its nearest
global center.
"""

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import normalize


class HierarchicalDPMeans:
    def __init__(
        self,
        local_percentile: float = 50.0,
        global_percentile: float = 30.0,
        spherical: bool = False,
        sample_size: int = 5000,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.local_percentile = local_percentile
        self.global_percentile = global_percentile
        self.spherical = spherical
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.global_centers_: list[np.ndarray] = []
        self.global_counts_: list[int] = []
        self.global_lambda_: Optional[float] = None
        self._rng = np.random.default_rng(random_state)

    def reset(self) -> None:
        self.global_centers_ = []
        self.global_counts_ = []
        self.global_lambda_ = None
        self._rng = np.random.default_rng(self.random_state)

    def partial_fit(self, tile: np.ndarray) -> np.ndarray:
        """Process one tile, update global state in place, return (H, W) labels."""
        return self._run(tile, update=True)

    def predict(self, tile: np.ndarray) -> np.ndarray:
        """Assign every pixel in `tile` to the nearest current global center."""
        return self._run(tile, update=False)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run(self, data: np.ndarray, update: bool) -> np.ndarray:
        h, w, valid_mask, valid_data = self._preprocess(data)
        if valid_data.size == 0:
            return np.full((h, w), -1, dtype=np.int32)

        if not update:
            if not self.global_centers_:
                raise RuntimeError(
                    "predict() called before any partial_fit(); no global centers fit yet."
                )
            labels = self._assign_to_global(valid_data)
            return self._scatter(h, w, valid_mask, labels)

        local_lambda = self._estimate_lambda(valid_data, self.local_percentile)
        local_centers, local_labels = self._core_dp_means(valid_data, local_lambda)

        if self.global_lambda_ is None:
            if len(local_centers) > 1:
                self.global_lambda_ = float(
                    np.percentile(
                        pdist(local_centers, metric="sqeuclidean"),
                        self.global_percentile,
                    )
                )
            else:
                self.global_lambda_ = local_lambda

        global_labels = self._merge_local_into_global(local_centers, local_labels)
        return self._scatter(h, w, valid_mask, global_labels)

    def _preprocess(self, data: np.ndarray):
        """Flatten (H, W, D) -> (N_valid, D), masking NaN pixels."""
        h, w, d = data.shape
        flat = data.reshape(h * w, d)
        valid_mask = ~np.isnan(flat).any(axis=1)
        valid = flat[valid_mask]
        if valid.size > 0 and self.spherical:
            valid = normalize(valid, norm="l2", axis=1)
        return h, w, valid_mask, valid

    def _estimate_lambda(self, valid: np.ndarray, percentile: float) -> float:
        """Estimate DP-Means λ from a percentile of pairwise sq-distances."""
        if valid.shape[0] > self.sample_size:
            idx = self._rng.choice(valid.shape[0], size=self.sample_size, replace=False)
            sample = valid[idx]
        else:
            sample = valid
        return float(np.percentile(pdist(sample, metric="sqeuclidean"), percentile))

    def _core_dp_means(
        self, data: np.ndarray, penalty: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inner DP-Means loop. Promotes one new center per iteration."""
        seed_idx = int(self._rng.integers(0, data.shape[0]))
        centers = np.array([data[seed_idx]])
        labels = np.zeros(data.shape[0], dtype=np.int32)

        for _ in range(self.max_iter):
            dists = cdist(data, centers, metric="sqeuclidean")
            min_dists = np.min(dists, axis=1)
            labels = np.argmin(dists, axis=1)

            furthest = int(np.argmax(min_dists))
            if min_dists[furthest] > penalty:
                centers = np.vstack([centers, data[furthest]])
                continue

            new_centers = np.zeros_like(centers)
            for i in range(len(centers)):
                mask = labels == i
                if mask.any():
                    new_centers[i] = data[mask].mean(axis=0)
                else:
                    new_centers[i] = centers[i]

            if np.allclose(centers, new_centers, atol=self.tol):
                break
            centers = new_centers

        return centers, labels

    def _merge_local_into_global(
        self, local_centers: np.ndarray, local_labels: np.ndarray
    ) -> np.ndarray:
        """For each local cluster, MERGE into nearest global center or SPAWN a new one."""
        global_labels = np.zeros_like(local_labels)

        for i, center in enumerate(local_centers):
            size = int((local_labels == i).sum())
            if size == 0:
                continue

            if not self.global_centers_:
                self.global_centers_.append(center)
                self.global_counts_.append(size)
                matched = 0
            else:
                centers_arr = np.array(self.global_centers_)
                dists = cdist([center], centers_arr, metric="sqeuclidean")[0]
                nearest = int(np.argmin(dists))

                if dists[nearest] > self.global_lambda_:
                    self.global_centers_.append(center)
                    self.global_counts_.append(size)
                    matched = len(self.global_centers_) - 1
                else:
                    n_old = self.global_counts_[nearest]
                    updated = (
                        self.global_centers_[nearest] * n_old + center * size
                    ) / (n_old + size)
                    if self.spherical:
                        updated /= np.linalg.norm(updated)
                    self.global_centers_[nearest] = updated
                    self.global_counts_[nearest] += size
                    matched = nearest

            global_labels[local_labels == i] = matched

        return global_labels

    def _assign_to_global(
        self, valid: np.ndarray, chunk_size: int = 10_000
    ) -> np.ndarray:
        """Inference path: nearest global center for every valid pixel."""
        centers_arr = np.array(self.global_centers_)
        out = np.empty(valid.shape[0], dtype=np.int32)
        for i in range(0, valid.shape[0], chunk_size):
            chunk = valid[i : i + chunk_size]
            d = cdist(chunk, centers_arr, metric="sqeuclidean")
            out[i : i + chunk_size] = np.argmin(d, axis=1)
        return out

    @staticmethod
    def _scatter(
        h: int, w: int, valid_mask: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """1-D valid-pixel labels -> (H, W) image with -1 for invalid pixels."""
        out = np.full(h * w, -1, dtype=np.int32)
        out[valid_mask] = labels
        return out.reshape(h, w)
