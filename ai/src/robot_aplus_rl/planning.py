"""Grid-based navigability check and approximate shortest collision-free path length.

The configuration space is discretised into square cells; a cell is free when its centre has at
least ``robot_radius + margin`` clearance. Reachability uses an 8-connected flood fill. The
reference path length (used only for the approximate SPL metric) is obtained with 8-connected
Dijkstra followed by greedy line-of-sight shortcutting validated with the exact swept-disc test.
Because the grid is coarse and inflated by a small margin, the resulting length approximates
(and is normally slightly longer than) the true continuous shortest collision-free path, so the
derived metric is always labelled ``spl_approx``.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from robot_aplus_rl.geometry import segment_is_free

_NEIGHBOURS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


def free_grid(
    obstacles: np.ndarray, world_size: float, inflate: float, resolution: float
) -> np.ndarray:
    """Boolean grid ``free[i, j]`` for the cell centred at ((i+.5) res, (j+.5) res)."""
    n = int(round(world_size / resolution))
    centres = (np.arange(n) + 0.5) * resolution
    gx, gy = np.meshgrid(centres, centres, indexing="ij")
    clear = np.minimum.reduce([gx, gy, world_size - gx, world_size - gy])
    if len(obstacles):
        d = (
            np.hypot(gx[..., None] - obstacles[:, 0], gy[..., None] - obstacles[:, 1])
            - obstacles[:, 2]
        )
        clear = np.minimum(clear, d.min(axis=-1))
    return clear > inflate


def _cell(p: tuple[float, float], resolution: float, n: int) -> tuple[int, int]:
    return (
        min(max(int(p[0] / resolution), 0), n - 1),
        min(max(int(p[1] / resolution), 0), n - 1),
    )


def _snap(free: np.ndarray, cell: tuple[int, int], max_radius: int = 3) -> tuple[int, int] | None:
    """Nearest free cell within ``max_radius`` cells (start/goal may sit inside the margin)."""
    if free[cell]:
        return cell
    n = free.shape[0]
    best, best_d = None, math.inf
    for di in range(-max_radius, max_radius + 1):
        for dj in range(-max_radius, max_radius + 1):
            i, j = cell[0] + di, cell[1] + dj
            if 0 <= i < n and 0 <= j < n and free[i, j] and di * di + dj * dj < best_d:
                best, best_d = (i, j), di * di + dj * dj
    return best


def is_reachable(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: np.ndarray,
    world_size: float,
    robot_radius: float,
    margin: float,
    resolution: float,
) -> bool:
    if segment_is_free(start, goal, robot_radius + margin, obstacles, world_size):
        return True
    free = free_grid(obstacles, world_size, robot_radius + margin, resolution)
    n = free.shape[0]
    s = _snap(free, _cell(start, resolution, n))
    g = _snap(free, _cell(goal, resolution, n))
    if s is None or g is None:
        return False
    # Vectorised 8-connected flood fill (iterated binary dilation restricted to free cells).
    reached = np.zeros_like(free)
    reached[s] = True
    while True:
        grown = reached.copy()
        grown[1:, :] |= reached[:-1, :]
        grown[:-1, :] |= reached[1:, :]
        rows = grown.copy()
        grown[:, 1:] |= rows[:, :-1]
        grown[:, :-1] |= rows[:, 1:]
        grown &= free
        if grown[g]:
            return True
        if np.array_equal(grown, reached):
            return False
        reached = grown


def reference_path(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: np.ndarray,
    world_size: float,
    robot_radius: float,
    margin: float,
    resolution: float,
) -> list[tuple[float, float]] | None:
    """Approximate shortest collision-free path (list of waypoints) or ``None``."""
    if segment_is_free(start, goal, robot_radius, obstacles, world_size):
        return [tuple(start), tuple(goal)]
    free = free_grid(obstacles, world_size, robot_radius + margin, resolution)
    n = free.shape[0]
    s = _snap(free, _cell(start, resolution, n))
    g = _snap(free, _cell(goal, resolution, n))
    if s is None or g is None:
        return None
    dist = {s: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    heap = [(0.0, s)]
    while heap:
        d, (i, j) = heapq.heappop(heap)
        if (i, j) == g:
            break
        if d > dist[(i, j)]:
            continue
        for di, dj, w in _NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and free[a, b]:
                nd = d + w
                if nd < dist.get((a, b), math.inf):
                    dist[(a, b)] = nd
                    parent[(a, b)] = (i, j)
                    heapq.heappush(heap, (nd, (a, b)))
    if g not in dist:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(parent[cells[-1]])
    cells.reverse()
    pts = [tuple(start)]
    pts += [((i + 0.5) * resolution, (j + 0.5) * resolution) for i, j in cells]
    pts.append(tuple(goal))

    # Greedy line-of-sight shortcutting with the exact swept-disc test.
    out = [pts[0]]
    k = 0
    while k < len(pts) - 1:
        nxt = k + 1
        for m in range(len(pts) - 1, k + 1, -1):
            if segment_is_free(pts[k], pts[m], robot_radius, obstacles, world_size):
                nxt = m
                break
        out.append(pts[nxt])
        k = nxt
    return out


def path_length(points: list[tuple[float, float]]) -> float:
    return float(sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)))
