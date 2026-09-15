"""
main.py
Entry point for Monte Carlo volume estimation.

Usage:
    python main.py <input_file.dat>

Example:
    python main.py sphere.dat
"""

import sys

from parser    import parse_input, print_config
from shapes    import build_shape
from geometry  import Geometry, build_bounding_box
from samplers  import PointSampler, LineSampler
from statistics import compute_stats
from results   import print_results


def main():
    # ------------------------------------------------------------------
    # 1. Read input file
    # ------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file.dat>")
        sys.exit(1)

    filepath = sys.argv[1]

    print(f"\nReading input from: {filepath}")
    config = parse_input(filepath)
    print_config(config)

    # ------------------------------------------------------------------
    # 2. Build shape and geometry
    # ------------------------------------------------------------------
    shape = build_shape(config)
    bbox  = build_bounding_box(shape)

    geometry = Geometry(bbox)
    geometry.add_shape(shape)

    print(f"\n  Bounding box  : {bbox}")
    print(f"  Bbox volume   : {bbox.volume():.6f}")
    print(f"  Analyt. volume: {shape.analytical_volume():.6f}")
    print()

    # ------------------------------------------------------------------
    # 3. Run Point Sampling algorithm
    # ------------------------------------------------------------------
    print(f"Running Point Sampling  (n = {config['n_points']:,}) ...")
    point_sampler = PointSampler(geometry, config['n_points'], seed=42)
    point_sampler.run()
    print(f"  Done in {point_sampler.cpu_time:.4f} s")

    # ------------------------------------------------------------------
    # 4. Run Line Drawing algorithm
    # ------------------------------------------------------------------
    print(f"Running Line Drawing    (n = {config['n_lines']:,}) ...")
    line_sampler = LineSampler(geometry, config['n_lines'], seed=137)
    line_sampler.run()
    print(f"  Done in {line_sampler.cpu_time:.4f} s")

    # ------------------------------------------------------------------
    # 5. Compute statistics
    # ------------------------------------------------------------------
    point_stats = compute_stats(
        scores      = point_sampler.scores,
        bbox_volume = bbox.volume(),
        cpu_time    = point_sampler.cpu_time,
    )

    line_stats = compute_stats(
        scores      = line_sampler.scores,
        bbox_volume = bbox.volume(),
        cpu_time    = line_sampler.cpu_time,
    )

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print_results(
        shape_name   = shape.name,
        analytical_v = shape.analytical_volume(),
        point_stats  = point_stats,
        line_stats   = line_stats,
    )


if __name__ == "__main__":
    main()