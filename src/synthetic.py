"""
Synthetic Mars-like tile generator.

Produces (H, W, D) feature cubes drawn from a shared mixture-of-Gaussians
in feature space, with a configurable fraction of NaN pixels to mimic the
invalid regions present in real CRISM data. The shared mixture lets the
streaming HDP discover a coherent global cluster set across tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TileSpec:
    height: int = 512
    width: int = 512
    n_features: int = 14


def make_global_mixture(
    n_clusters: int = 6,
    n_features: int = 14,
    spread: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """The set of cluster means shared across all tiles in the stream."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_clusters, n_features)) * spread


def generate_tile(
    spec: TileSpec,
    global_centers: np.ndarray,
    nan_fraction: float = 0.10,
    cluster_std: float = 1.0,
    cluster_weights: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample one (H, W, D) tile from the shared mixture.

    Tiles can use a non-uniform `cluster_weights` to mimic the regional
    composition skew of real Mars surface tiles (a few dominant minerals
    plus a long tail of rare phases).
    """
    rng = rng or np.random.default_rng()
    n_clusters, n_features = global_centers.shape
    assert n_features == spec.n_features

    n_pixels = spec.height * spec.width
    if cluster_weights is None:
        cluster_weights = np.ones(n_clusters) / n_clusters

    labels = rng.choice(n_clusters, size=n_pixels, p=cluster_weights)
    data = (
        global_centers[labels]
        + rng.standard_normal((n_pixels, n_features)) * cluster_std
    )

    # Carve out invalid pixels
    nan_mask = rng.random(n_pixels) < nan_fraction
    data[nan_mask] = np.nan

    return data.reshape(spec.height, spec.width, n_features)


def generate_stream(
    n_tiles: int,
    spec: TileSpec = TileSpec(),
    n_clusters: int = 6,
    nan_fraction: float = 0.10,
    cluster_std: float = 1.0,
    skew: float = 1.5,
    seed: int = 42,
):
    """Yield ``n_tiles`` tiles drawn from a shared global mixture.

    `skew` > 1 makes per-tile cluster weights more uneven, exercising the
    SPAWN/MERGE logic harder when a tile's dominant cluster differs from
    what the running global state has seen so far.
    """
    rng = np.random.default_rng(seed)
    global_centers = make_global_mixture(
        n_clusters=n_clusters, n_features=spec.n_features, seed=seed
    )

    for _ in range(n_tiles):
        weights = rng.dirichlet(np.ones(n_clusters) / skew)
        yield generate_tile(
            spec=spec,
            global_centers=global_centers,
            nan_fraction=nan_fraction,
            cluster_std=cluster_std,
            cluster_weights=weights,
            rng=rng,
        )
