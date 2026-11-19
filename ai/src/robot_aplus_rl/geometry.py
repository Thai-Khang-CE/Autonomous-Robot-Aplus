"""Planar geometry: angle wrapping, differential-drive kinematics and swept collision tests.

Conventions
-----------
* World frame: x to the right, y up, origin in the lower-left corner, units in metres.
* Heading ``theta`` is measured counter-clockwise from the +x axis, wrapped to [-pi, pi).
* Obstacles are stored as an ``(N, 3)`` float array of ``(cx, cy, radius)``.
* The robot is a disc of radius ``robot_radius``; collisions are tested against the disc,
  not against the centre point, by inflating every obstacle by the robot radius
  (configuration-space test).
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi)."""
    return float((angle + math.pi) % TWO_PI - math.pi)


def unicycle_step(
    x: float, y: float, theta: float, v: float, omega: float, dt: float
) -> tuple[float, float, float]:
    """Forward-Euler differential-drive update (exactly the Week 9 model).

    x' = x + v cos(theta) dt,  y' = y + v sin(theta) dt,  theta' = wrap(theta + omega dt)
    """
    return (
        x + v * math.cos(theta) * dt,
        y + v * math.sin(theta) * dt,
        wrap_angle(theta + omega * dt),
    )


def goal_distance_and_bearing(
    x: float, y: float, theta: float, gx: float, gy: float
) -> tuple[float, float]:
    """Euclidean distance to the goal and goal direction relative to the heading (radians)."""
    dx, dy = gx - x, gy - y
    return math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - theta)


def empty_obstacles() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float64)


def clearance(px: float, py: float, obstacles: np.ndarray, world_size: float) -> float:
    """Signed free distance from a point to the nearest obstacle surface or wall."""
    wall = min(px, py, world_size - px, world_size - py)
    if len(obstacles) == 0:
        return float(wall)
    d = np.hypot(obstacles[:, 0] - px, obstacles[:, 1] - py) - obstacles[:, 2]
    return float(min(wall, d.min()))


def disc_in_collision(
    px: float, py: float, radius: float, obstacles: np.ndarray, world_size: float
) -> bool:
    """True if a disc of ``radius`` at (px, py) overlaps an obstacle or leaves the world."""
    return clearance(px, py, obstacles, world_size) < radius


def first_contact_fraction(
    p0: tuple[float, float],
    p1: tuple[float, float],
    radius: float,
    obstacles: np.ndarray,
    world_size: float,
) -> float | None:
    """Swept-disc test for straight motion from ``p0`` to ``p1``.

    Returns the smallest fraction ``s`` in [0, 1] at which a disc of ``radius`` moving along the
    segment first touches an obstacle or a wall, or ``None`` if the whole sweep is free.
    Because the motion is tested continuously (analytically), the robot cannot tunnel through a
    thin obstacle between two control steps.
    """
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    hits: list[float] = []

    # Walls: the disc centre must stay inside [radius, world_size - radius]^2.
    lo, hi = radius, world_size - radius
    for start, delta in ((x0, dx), (y0, dy)):
        if start < lo or start > hi:
            return 0.0
        if delta < 0.0:
            s = (lo - start) / delta
            if s <= 1.0:
                hits.append(s)
        elif delta > 0.0:
            s = (hi - start) / delta
            if s <= 1.0:
                hits.append(s)

    if len(obstacles):
        cx, cy, r = obstacles[:, 0], obstacles[:, 1], obstacles[:, 2] + radius
        fx, fy = x0 - cx, y0 - cy
        c = fx * fx + fy * fy - r * r
        if np.any(c <= 0.0):
            return 0.0  # already touching / overlapping
        a = dx * dx + dy * dy
        if a > 0.0:
            b = dx * fx + dy * fy
            disc = b * b - a * c
            ok = disc >= 0.0
            if np.any(ok):
                s = (-b[ok] - np.sqrt(disc[ok])) / a
                s = s[(s >= 0.0) & (s <= 1.0)]
                hits.extend(s.tolist())

    return float(min(hits)) if hits else None


def segment_is_free(
    p0: tuple[float, float],
    p1: tuple[float, float],
    radius: float,
    obstacles: np.ndarray,
    world_size: float,
) -> bool:
    return first_contact_fraction(p0, p1, radius, obstacles, world_size) is None
