"""
shapes.py
Geometric shape classes. Each implements:
    contains(point) -> bool
    analytical_volume() -> float
    name -> str (for display)
"""

import numpy as np
import math


class Shape:
    """Abstract base class for all shapes."""

    def contains(self, point: np.ndarray) -> bool:
        raise NotImplementedError

    def analytical_volume(self) -> float:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

class Sphere(Shape):
    """
    Solid sphere of radius r centered at origin.
    Inside condition: x² + y² + z² <= r²
    """

    def __init__(self, r: float):
        self.r = r

    def contains(self, point: np.ndarray) -> bool:
        return float(np.dot(point, point)) <= self.r ** 2

    def analytical_volume(self) -> float:
        return (4.0 / 3.0) * math.pi * self.r ** 3

    @property
    def name(self) -> str:
        return f"Sphere (r={self.r})"


# ---------------------------------------------------------------------------
# Spherical Shell
# ---------------------------------------------------------------------------

class SphericalShell(Shape):
    """
    Shell between inner radius r_inner and outer radius r.
    Thickness = r / 50, so r_inner = r - r/50 = 49r/50.
    Inside condition: r_inner² <= x²+y²+z² <= r²
    """

    def __init__(self, r: float):
        self.r       = r
        self.r_inner = r * (49.0 / 50.0)   # thickness = r/50

    def contains(self, point: np.ndarray) -> bool:
        dist2 = float(np.dot(point, point))
        return self.r_inner ** 2 <= dist2 <= self.r ** 2

    def analytical_volume(self) -> float:
        return (4.0 / 3.0) * math.pi * (self.r ** 3 - self.r_inner ** 3)

    @property
    def name(self) -> str:
        return (f"Spherical Shell "
                f"(r={self.r}, r_inner={self.r_inner:.6f}, "
                f"thickness={self.r/50:.6f})")


# ---------------------------------------------------------------------------
# Cone
# ---------------------------------------------------------------------------

class Cone(Shape):
    """
    Cone with base radius r at z=0 and apex at z=h.
    Inside condition:
        0 <= z <= h
        x² + y² <= (r * (1 - z/h))²
    """

    def __init__(self, r: float, h: float):
        self.r = r
        self.h = h

    def contains(self, point: np.ndarray) -> bool:
        x, y, z = point
        if z < 0.0 or z > self.h:
            return False
        r_at_z = self.r * (1.0 - z / self.h)
        return x ** 2 + y ** 2 <= r_at_z ** 2

    def analytical_volume(self) -> float:
        return (1.0 / 3.0) * math.pi * self.r ** 2 * self.h

    @property
    def name(self) -> str:
        return f"Cone (r={self.r}, h={self.h})"


# ---------------------------------------------------------------------------
# Truncated Cylinder (Frustum)
# ---------------------------------------------------------------------------

class TruncatedCylinder(Shape):
    """
    Frustum (truncated cone) with base radius r at z=0,
    top radius r_top at z=h.
    Inside condition:
        0 <= z <= h
        x² + y² <= R(z)²   where R(z) = r - (r - r_top) * z/h
    """

    def __init__(self, r: float, r_top: float, h: float):
        self.r     = r
        self.r_top = r_top
        self.h     = h

    def _radius_at_z(self, z: float) -> float:
        return self.r - (self.r - self.r_top) * z / self.h

    def contains(self, point: np.ndarray) -> bool:
        x, y, z = point
        if z < 0.0 or z > self.h:
            return False
        r_at_z = self._radius_at_z(z)
        return x ** 2 + y ** 2 <= r_at_z ** 2

    def analytical_volume(self) -> float:
        # V = (pi * h / 3) * (r² + r*r_top + r_top²)
        r, r_top, h = self.r, self.r_top, self.h
        return (math.pi * h / 3.0) * (r ** 2 + r * r_top + r_top ** 2)

    @property
    def name(self) -> str:
        return f"Truncated Cylinder / Frustum (r={self.r}, r_top={self.r_top}, h={self.h})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_shape(config: dict) -> Shape:
    """
    Construct the correct Shape subclass from a parsed config dict.
    """
    shape_type = config['shape']

    if shape_type == 'sphere':
        return Sphere(r=config['r'])

    elif shape_type == 'shell':
        return SphericalShell(r=config['r'])

    elif shape_type == 'cone':
        return Cone(r=config['r'], h=config['h'])

    elif shape_type == 'frustum':
        return TruncatedCylinder(
            r=config['r'],
            r_top=config['r_top'],
            h=config['h']
        )

    else:
        raise ValueError(f"Unknown shape type: '{shape_type}'")