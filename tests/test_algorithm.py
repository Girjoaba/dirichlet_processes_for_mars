"""End-to-end smoke tests for the serial HDP baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from algorithm import HierarchicalDPMeans  # noqa: E402
from synthetic import (  # noqa: E402
    TileSpec,
    generate_stream,
    generate_tile,
    make_global_mixture,
)


def test_streaming_runs_and_promotes_centers():
    """Streaming a few tiles should produce well-shaped (H, W) int32 label
    maps and discover more than one global cluster (otherwise DP-Means is
    silently trapped at K=1, which is the failure mode we care about)."""
    spec = TileSpec(height=128, width=128, n_features=8)
    tiles = list(generate_stream(n_tiles=4, spec=spec, n_clusters=5, seed=0))

    model = HierarchicalDPMeans(
        local_percentile=50.0,
        global_percentile=30.0,
        random_state=0,
    )
    for tile in tiles:
        labels = model.partial_fit(tile)
        assert labels.shape == (spec.height, spec.width)
        assert labels.dtype == np.int32

    assert len(model.global_centers_) >= 2


def test_predict_assigns_every_valid_pixel():
    spec = TileSpec(height=64, width=64, n_features=6)
    centers = make_global_mixture(n_clusters=4, n_features=6, seed=1)
    rng = np.random.default_rng(1)
    train_tile = generate_tile(spec, centers, nan_fraction=0.0, rng=rng)
    test_tile = generate_tile(spec, centers, nan_fraction=0.2, rng=rng)

    model = HierarchicalDPMeans(
        local_percentile=50.0, global_percentile=30.0, random_state=1
    )
    model.partial_fit(train_tile)

    labels = model.predict(test_tile)
    valid_mask = ~np.isnan(test_tile).any(axis=2)

    assert labels.shape == (spec.height, spec.width)
    assert (labels[valid_mask] >= 0).all()
    assert (labels[~valid_mask] == -1).all()


def test_reset_clears_state():
    spec = TileSpec(height=32, width=32, n_features=4)
    tile = next(iter(generate_stream(n_tiles=1, spec=spec, n_clusters=3, seed=7)))

    model = HierarchicalDPMeans(
        local_percentile=50.0, global_percentile=30.0, random_state=7
    )
    model.partial_fit(tile)
    assert model.global_centers_
    assert model.global_lambda_ is not None

    model.reset()
    assert model.global_centers_ == []
    assert model.global_counts_ == []
    assert model.global_lambda_ is None


if __name__ == "__main__":
    test_streaming_runs_and_promotes_centers()
    test_predict_assigns_every_valid_pixel()
    test_reset_clears_state()
    print("ok")
