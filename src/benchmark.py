"""
Serial CPU baseline benchmark for the streaming Hierarchical DP-Means.

Runs the full two-pass pipeline (training stream + inference pass) on
synthetic Mars-like tiles and prints a phase-level timing breakdown.
This is the baseline that the EuroHack26 GPU port should beat.

Run from the repo root:

    python src/benchmark.py
    python src/benchmark.py --tiles 4 --height 256 --width 256
"""

import argparse
import time
from contextlib import contextmanager

from algorithm import HierarchicalDPMeans
from synthetic import TileSpec, generate_stream


@contextmanager
def stopwatch(label: str, registry: dict[str, float]):
    t0 = time.perf_counter()
    yield
    registry[label] = time.perf_counter() - t0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiles", type=int, default=4)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--features", type=int, default=14)
    p.add_argument("--clusters", type=int, default=6)
    p.add_argument("--local-percentile", type=float, default=50.0)
    p.add_argument("--global-percentile", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    spec = TileSpec(height=args.height, width=args.width, n_features=args.features)
    print(
        f"Stream: {args.tiles} tiles x ({spec.height}x{spec.width}x{spec.n_features}) "
        f"= {args.tiles * spec.height * spec.width:,} pixels total"
    )

    timings: dict[str, float] = {}

    with stopwatch("data_generation", timings):
        tiles = list(
            generate_stream(
                n_tiles=args.tiles,
                spec=spec,
                n_clusters=args.clusters,
                seed=args.seed,
            )
        )

    model = HierarchicalDPMeans(
        local_percentile=args.local_percentile,
        global_percentile=args.global_percentile,
        spherical=False,
        random_state=args.seed,
    )

    with stopwatch("training_stream", timings):
        for tile in tiles:
            model.partial_fit(tile)

    with stopwatch("inference_pass", timings):
        for tile in tiles:
            model.predict(tile)

    total = timings["training_stream"] + timings["inference_pass"]
    print()
    print(f"Discovered global clusters: {len(model.global_centers_)}")
    print(f"Global lambda:              {model.global_lambda_:.4f}")
    print()
    print("Phase-level timings:")
    for phase, dur in timings.items():
        share = dur / total * 100 if phase != "data_generation" else 0.0
        suffix = "" if phase == "data_generation" else f"  ({share:5.1f}% of HDP)"
        print(f"  {phase:18}: {dur:8.2f}s{suffix}")
    print(f"  {'HDP total':18}: {total:8.2f}s")


if __name__ == "__main__":
    main()
