"""
intersection.py
Analytical line-shape intersection solvers.

A line is parameterised as:
    P(t) = origin + t * direction,   t in [0, 1]

where origin = wall_start, direction = wall_end - wall_start.

Each solver returns a list of (t_enter, t_exit) intervals where the
line is inside the shape, clipped to [0, 1].
An empty list means no intersection.
"""

import numpy as np
import math
from typing import List, Tuple

Interval = Tuple[float, float]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _solve_quadratic(a: float, b: float, c: float):
    """
    Solve a*t² + b*t + c = 0.
    Returns (t1, t2) with t1 <= t2, or None if no real roots.
    """
    if abs(a) < 1e-14:
        # Degenerate: linear equation b*t + c = 0
        if abs(b) < 1e-14:
            return None   # no solution
        t = -c / b
        return (t, t)

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    sqrt_d = math.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2.0 * a)
    t2 = (-b + sqrt_d) / (2.0 * a)
    return (t1, t2)


def _clip_interval(t1: float, t2: float) -> Interval | None:
    """
    Clip interval [t1, t2] to [0, 1].
    Returns None if the clipped interval is empty.
    """
    lo = max(t1, 0.0)
    hi = min(t2, 1.0)
    if lo >= hi:
        return None
    return (lo, hi)


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

def intersect_sphere(origin: np.ndarray, direction: np.ndarray,
                     r: float) -> List[Interval]:
    """
    Intersect line P(t) = origin + t*direction with sphere of radius r.
    Equation: |P(t)|² = r²
    Expands to: (dir·dir)t² + 2(orig·dir)t + (orig·orig - r²) = 0
    """
    a = float(np.dot(direction, direction))
    b = 2.0 * float(np.dot(origin, direction))
    c = float(np.dot(origin, origin)) - r * r

    roots = _solve_quadratic(a, b, c)
    if roots is None:
        return []

    t1, t2 = roots
    interval = _clip_interval(t1, t2)
    return [interval] if interval else []


# ---------------------------------------------------------------------------
# Spherical Shell
# ---------------------------------------------------------------------------

def intersect_shell(origin: np.ndarray, direction: np.ndarray,
                    r_outer: float, r_inner: float) -> List[Interval]:
    """
    Intersect line with spherical shell between r_inner and r_outer.

    The line is inside the shell in the set difference:
        [t1_outer, t2_outer]  minus  [t1_inner, t2_inner]

    This can yield 0, 1, or 2 sub-intervals.
    """
    # Solve for outer sphere
    a = float(np.dot(direction, direction))
    b = 2.0 * float(np.dot(origin, direction))

    c_outer = float(np.dot(origin, origin)) - r_outer * r_outer
    c_inner = float(np.dot(origin, origin)) - r_inner * r_inner

    roots_outer = _solve_quadratic(a, b, c_outer)
    if roots_outer is None:
        return []   # misses outer sphere entirely

    t1_out, t2_out = roots_outer

    roots_inner = _solve_quadratic(a, b, c_inner)

    intervals = []

    if roots_inner is None:
        # Line does not intersect inner sphere → entire outer interval is shell
        seg = _clip_interval(t1_out, t2_out)
        if seg:
            intervals.append(seg)
    else:
        t1_in, t2_in = roots_inner

        # Left shell segment: [t1_out, t1_in]
        seg1 = _clip_interval(t1_out, t1_in)
        if seg1:
            intervals.append(seg1)

        # Right shell segment: [t2_in, t2_out]
        seg2 = _clip_interval(t2_in, t2_out)
        if seg2:
            intervals.append(seg2)

    return intervals


# ---------------------------------------------------------------------------
# Cone  (and shared lateral-surface solver used by frustum too)
# ---------------------------------------------------------------------------

def _intersect_cone_surface(origin: np.ndarray, direction: np.ndarray,
                             r_base: float, r_top: float,
                             h: float) -> List[Interval]:
    """
    Find the interval(s) of t where the line P(t) = origin + t*direction
    is INSIDE the frustum (or cone if r_top=0).

    The frustum is bounded by:
        - Lateral surface:  x² + y² = R(z)²,  R(z) = r_base - slope*z
        - Bottom cap:       z = 0,  x²+y² <= r_base²
        - Top cap:          z = h,  x²+y² <= r_top²

    We collect ALL candidate t values where the line crosses any boundary,
    sort them, then test the midpoint of each sub-interval to decide inside/outside.
    This correctly handles all entry/exit combinations including through the caps.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction
    slope = (r_base - r_top) / h

    # --- Collect candidate t values from all three boundaries ---
    candidates = [0.0, 1.0]   # always include the endpoints

    # 1. Lateral surface: A t² + B t + C = 0
    R0 = r_base - slope * oz
    dR = -slope * dz

    A = dx*dx + dy*dy - dR*dR
    B = 2.0*(ox*dx + oy*dy - R0*dR)
    C = ox*ox + oy*oy - R0*R0

    roots = _solve_quadratic(A, B, C)
    if roots is not None:
        candidates.extend(roots)

    # 2. Bottom cap plane: z(t) = 0  =>  oz + t*dz = 0
    if abs(dz) > 1e-14:
        t_bot = -oz / dz
        candidates.append(t_bot)

        # 3. Top cap plane: z(t) = h  =>  oz + t*dz = h
        t_top = (h - oz) / dz
        candidates.append(t_top)

    # --- Sort and filter to [0, 1] ---
    candidates = sorted(set(candidates))

    # --- Test midpoint of each sub-interval ---
    intervals = []
    for i in range(len(candidates) - 1):
        t_lo = candidates[i]
        t_hi = candidates[i + 1]

        # Skip segments outside [0, 1]
        lo = max(t_lo, 0.0)
        hi = min(t_hi, 1.0)
        if hi - lo < 1e-14:
            continue

        # Test midpoint of this sub-interval
        t_mid = 0.5 * (lo + hi)
        p = origin + t_mid * direction
        px, py, pz = p

        # Check height bounds
        if pz < 0.0 or pz > h:
            continue

        # Check radial bound
        r_at_z = r_base - slope * pz
        if px*px + py*py <= r_at_z*r_at_z:
            intervals.append((lo, hi))

    return intervals


def intersect_cone(origin: np.ndarray, direction: np.ndarray,
                   r: float, h: float) -> List[Interval]:
    """Intersect line with cone (r_top = 0)."""
    return _intersect_cone_surface(origin, direction,
                                   r_base=r, r_top=0.0, h=h)


def intersect_frustum(origin: np.ndarray, direction: np.ndarray,
                      r: float, r_top: float, h: float) -> List[Interval]:
    """Intersect line with frustum (truncated cone)."""
    return _intersect_cone_surface(origin, direction,
                                   r_base=r, r_top=r_top, h=h)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def intersect(origin: np.ndarray, direction: np.ndarray,
              shape) -> List[Interval]:
    """
    Dispatch to the correct analytical intersection solver based on shape type.
    Returns list of (t_enter, t_exit) intervals in [0, 1].
    """
    from shapes import Sphere, SphericalShell, Cone, TruncatedCylinder

    if isinstance(shape, Sphere):
        return intersect_sphere(origin, direction, shape.r)

    elif isinstance(shape, SphericalShell):
        return intersect_shell(origin, direction, shape.r, shape.r_inner)

    elif isinstance(shape, Cone):
        return intersect_cone(origin, direction, shape.r, shape.h)

    elif isinstance(shape, TruncatedCylinder):
        return intersect_frustum(origin, direction,
                                 shape.r, shape.r_top, shape.h)

    else:
        raise TypeError(f"No intersection solver for shape type: {type(shape)}")