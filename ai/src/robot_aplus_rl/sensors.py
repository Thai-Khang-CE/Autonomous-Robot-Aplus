"""Simulated range sensor (2D ray casting against circles and the world boundary)."""

from __future__ import annotations

import math

import numpy as np


def ray_angles(num_rays: int) -> np.ndarray:
    """Ray offsets relative to the robot heading: k * 2*pi / num_rays (0 = straight ahead,
    counter-clockwise positive). With 8 rays: 0, 45, 90, ..., 315 deg -> 360 deg field of view."""
    return np.arange(num_rays, dtype=np.float64) * (2.0 * math.pi / num_rays)


def cast_rays(
    x: float,
    y: float,
    theta: float,
    offsets: np.ndarray,
    obstacles: np.ndarray,
    world_size: float,
    max_range: float,
) -> np.ndarray:
    """Distance (metres, measured from the robot centre) along each ray to the first surface.

    Readings are clipped to ``max_range``. If the sensor origin lies inside an obstacle the
    reading is 0.
    """
    ang = theta + offsets
    ux, uy = np.cos(ang), np.sin(ang)

    # World boundary: distance to the box [0, W]^2 from inside.
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(ux > 0, (world_size - x) / ux, np.where(ux < 0, -x / ux, np.inf))
        ty = np.where(uy > 0, (world_size - y) / uy, np.where(uy < 0, -y / uy, np.inf))
    dist = np.minimum(tx, ty)
    dist = np.maximum(dist, 0.0)

    if len(obstacles):
        fx = x - obstacles[:, 0]  # (N,)
        fy = y - obstacles[:, 1]
        c = fx * fx + fy * fy - obstacles[:, 2] ** 2  # (N,)
        b = ux[:, None] * fx[None, :] + uy[:, None] * fy[None, :]  # (R, N)
        disc = b * b - c[None, :]
        with np.errstate(invalid="ignore"):
            t_near = -b - np.sqrt(disc)
        valid = (disc >= 0.0) & (t_near >= 0.0)
        t_obs = np.where(valid, t_near, np.inf).min(axis=1)
        dist = np.minimum(dist, t_obs)
        if np.any(c <= 0.0):
            dist[:] = 0.0

    return np.minimum(dist, max_range)
