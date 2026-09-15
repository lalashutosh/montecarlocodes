"""
statistics.py
Compute mean volume, relative statistical error, and Figure of Merit (FOM).

Based on the pi-estimation example from lecture slides.

For N samples with per-sample scores x_i:

    mean_x   = (1/N) * sum(x_i)
    var_x    = (1/(N-1)) * sum((x_i - mean_x)²)    [sample variance]

    volume   = V_bbox * mean_x

    The standard error of the mean:
        sigma_mean = sqrt(var_x / N)

    Relative statistical error (as fraction):
        rel_err = sigma_mean / mean_x = sqrt(var_x / N) / mean_x

    Figure of Merit:
        FOM = 1 / (rel_err² * T)

    where T is the CPU time in seconds.
    Higher FOM = more computationally efficient algorithm.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Stats:
    volume:    float   # estimated volume
    rel_err:   float   # relative statistical error (fraction, e.g. 0.005 = 0.5%)
    fom:       float   # figure of merit
    cpu_time:  float   # wall time in seconds
    n:         int     # number of samples


def compute_stats(scores: np.ndarray,
                  bbox_volume: float,
                  cpu_time: float) -> Stats:
    """
    Compute volume estimate, relative error, and FOM from per-sample scores.

    Parameters
    ----------
    scores      : 1-D array of per-sample values
                  PointSampler  → 0.0 or 1.0  (miss / hit)
                  LineSampler   → ratio in [0, 1]
    bbox_volume : volume of the bounding box
    cpu_time    : elapsed CPU time for the sampling run (seconds)

    Returns
    -------
    Stats dataclass
    """
    n      = len(scores)
    mean_x = np.mean(scores)
    # ddof=1 → unbiased sample variance
    var_x  = np.var(scores, ddof=1)

    volume = bbox_volume * mean_x

    # Relative error on the volume estimate
    # sigma_mean = sqrt(var_x / N), rel_err = sigma_mean / mean_x
    if mean_x < 1e-14:
        rel_err = float('inf')
    else:
        rel_err = np.sqrt(var_x / n) / mean_x

    # FOM = 1 / (rel_err² * T)
    if rel_err < 1e-14 or cpu_time < 1e-14:
        fom = float('inf')
    else:
        fom = 1.0 / (rel_err ** 2 * cpu_time)

    return Stats(
        volume   = float(volume),
        rel_err  = float(rel_err),
        fom      = float(fom),
        cpu_time = float(cpu_time),
        n        = int(n),
    )