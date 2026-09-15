"""
samplers.py
Two Monte Carlo volume estimators:
    PointSampler  — fraction of random points inside the shape
    LineSampler   — fraction of total line length inside the shape
"""

import numpy as np
import time
from geometry import Geometry
from intersection import intersect


class PointSampler:
    """
    Point sampling (hit-or-miss) Monte Carlo.

    Samples n_samples random points uniformly inside the bounding box.
    Records which points fall inside the shape.

    Volume estimate:
        V ≈ V_bbox * (number of hits) / n_samples

    Per-sample scores stored in self.scores (1.0 = hit, 0.0 = miss)
    so that variance and FOM can be computed from them.
    """

    def __init__(self, geometry: Geometry, n_samples: int, seed: int = None):
        self.geometry  = geometry
        self.n_samples = n_samples
        self.rng       = np.random.default_rng(seed)

        # Filled after run()
        self.points    = None   # (n, 3) array
        self.scores    = None   # (n,) float array — 1.0 or 0.0
        self.cpu_time  = None   # seconds

    def run(self):
        """Sample all points and record hit/miss."""
        bbox = self.geometry.bounding_box
        n    = self.n_samples

        t_start = time.perf_counter()

        # Draw all points at once (vectorised)
        self.points = bbox.sample_random_points(n, self.rng)

        # Test each point — loop is unavoidable since contains() is per-shape
        inside = np.empty(n, dtype=bool)
        for i in range(n):
            inside[i] = self.geometry.contains(self.points[i])

        self.scores   = inside.astype(float)   # 1.0 / 0.0
        self.cpu_time = time.perf_counter() - t_start

    def estimate_volume(self) -> float:
        if self.scores is None:
            raise RuntimeError("Call run() before estimate_volume().")
        return self.geometry.bounding_box.volume() * np.mean(self.scores)


class LineSampler:
    """
    Line (chord) sampling Monte Carlo.

    Draws n_lines random wall-to-wall lines through the bounding box.
    For each line, finds the analytical intersection with the shape
    and records the fraction of the line length that is inside.

    Volume estimate:
        V ≈ V_bbox * mean(length_inside / length_total)

    Per-sample scores stored in self.scores (ratio in [0, 1])
    so that variance and FOM can be computed from them.
    """

    def __init__(self, geometry: Geometry, n_lines: int, seed: int = None):
        self.geometry = geometry
        self.n_lines  = n_lines
        self.rng      = np.random.default_rng(seed)

        # Filled after run()
        self.scores   = None   # (n,) ratio: length_inside / length_total
        self.cpu_time = None

    def run(self):
        """Draw all lines, compute inside-length ratio for each."""
        bbox   = self.geometry.bounding_box
        n      = self.n_lines
        scores = np.zeros(n, dtype=float)

        t_start = time.perf_counter()

        for i in range(n):
            start, end = bbox.sample_random_line(self.rng)
            direction  = end - start
            length_total = float(np.linalg.norm(direction))

            if length_total < 1e-14:
                scores[i] = 0.0
                continue

            # direction is already scaled to the full box length
            # P(t) = start + t * direction,  t in [0,1]
            length_inside = 0.0
            for shape in self.geometry.shapes:
                intervals = intersect(start, direction, shape)
                for (t_enter, t_exit) in intervals:
                    length_inside += (t_exit - t_enter) * length_total

            scores[i] = length_inside / length_total

        self.scores   = scores
        self.cpu_time = time.perf_counter() - t_start

    def estimate_volume(self) -> float:
        if self.scores is None:
            raise RuntimeError("Call run() before estimate_volume().")
        return self.geometry.bounding_box.volume() * np.mean(self.scores)