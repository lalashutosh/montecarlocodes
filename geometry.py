"""
geometry.py
Bounding box construction and geometry container.

Defines:
    BoundingBox         - axis-aligned box; supports sampling points/lines
    build_bounding_box  - construct the tightest axis-aligned box around a shape
    Geometry            - container binding a bounding box to a set of shapes
"""

import numpy as np

from shapes import Sphere, SphericalShell, Cone, TruncatedCylinder


class BoundingBox:
    """
    Axis-aligned bounding box [xmin,xmax] x [ymin,ymax] x [zmin,zmax].
    """

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax

    def volume(self) -> float:
        return ((self.xmax - self.xmin) *
                (self.ymax - self.ymin) *
                (self.zmax - self.zmin))

    def sample_random_points(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Vectorised sampling of n uniform random points inside the box."""
        x = rng.uniform(self.xmin, self.xmax, size=n)
        y = rng.uniform(self.ymin, self.ymax, size=n)
        z = rng.uniform(self.zmin, self.zmax, size=n)
        return np.column_stack((x, y, z))

    def sample_random_line(self, rng: np.random.Generator):
        """
        Draw one random wall-to-wall line (chord) through the box.

        A uniform random point inside the box is combined with an
        isotropic random direction; the infinite line through that point
        is then clipped to the box walls using the slab method, giving
        the entry/exit points of the chord.

        Returns
        -------
        (start, end) : tuple of (3,) np.ndarray
        """
        # Uniform random point inside the box
        point = np.array([
            rng.uniform(self.xmin, self.xmax),
            rng.uniform(self.ymin, self.ymax),
            rng.uniform(self.zmin, self.zmax),
        ])

        # Isotropic random direction (uniform on the unit sphere)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        # Slab method: find [t_enter, t_exit] where point + t*direction
        # stays within the box, for the infinite line.
        t_enter, t_exit = -np.inf, np.inf
        lo = (self.xmin, self.ymin, self.zmin)
        hi = (self.xmax, self.ymax, self.zmax)

        for axis in range(3):
            d = direction[axis]
            p = point[axis]
            if abs(d) < 1e-14:
                continue   # parallel to this slab; point's coord already inside range
            t1 = (lo[axis] - p) / d
            t2 = (hi[axis] - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit  = min(t_exit, t2)

        start = point + t_enter * direction
        end   = point + t_exit * direction
        return start, end

    def __repr__(self) -> str:
        return (f"[{self.xmin:.4f}, {self.xmax:.4f}] x "
                f"[{self.ymin:.4f}, {self.ymax:.4f}] x "
                f"[{self.zmin:.4f}, {self.zmax:.4f}]")


def build_bounding_box(shape) -> BoundingBox:
    """
    Construct the tightest axis-aligned bounding box around a shape.
    """
    if isinstance(shape, Sphere):
        r = shape.r
        return BoundingBox(-r, r, -r, r, -r, r)

    elif isinstance(shape, SphericalShell):
        r = shape.r
        return BoundingBox(-r, r, -r, r, -r, r)

    elif isinstance(shape, Cone):
        r = shape.r
        h = shape.h
        return BoundingBox(-r, r, -r, r, 0.0, h)

    elif isinstance(shape, TruncatedCylinder):
        r = max(shape.r, shape.r_top)
        h = shape.h
        return BoundingBox(-r, r, -r, r, 0.0, h)

    else:
        raise TypeError(f"No bounding box rule for shape type: {type(shape)}")


class Geometry:
    """
    Container binding a bounding box to one or more shapes.

    Point sampling and line sampling both operate through this container,
    so that (in principle) a scene could hold more than one shape.
    """

    def __init__(self, bounding_box: BoundingBox):
        self.bounding_box = bounding_box
        self.shapes = []

    def add_shape(self, shape):
        self.shapes.append(shape)

    def contains(self, point: np.ndarray) -> bool:
        """A point is inside the geometry if it is inside any of its shapes."""
        return any(shape.contains(point) for shape in self.shapes)
