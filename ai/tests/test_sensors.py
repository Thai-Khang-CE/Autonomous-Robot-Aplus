import math

import numpy as np
import pytest

from robot_aplus_rl.sensors import cast_rays, ray_angles

EMPTY = np.zeros((0, 3))


def test_ray_layout():
    a = ray_angles(8)
    assert a.shape == (8,)
    assert np.allclose(np.degrees(a), [0, 45, 90, 135, 180, 225, 270, 315])


def test_empty_room_wall_distances():
    d = cast_rays(1.0, 2.0, 0.0, ray_angles(8), EMPTY, 6.0, 10.0)
    assert d[0] == pytest.approx(5.0)   # +x wall
    assert d[2] == pytest.approx(4.0)   # +y wall
    assert d[4] == pytest.approx(1.0)   # -x wall
    assert d[6] == pytest.approx(2.0)   # -y wall
    assert d[1] == pytest.approx(4.0 * math.sqrt(2))  # hits top wall at 45 deg


def test_rays_rotate_with_heading():
    d = cast_rays(1.0, 2.0, math.pi / 2, ray_angles(8), EMPTY, 6.0, 10.0)
    assert d[0] == pytest.approx(4.0)  # now looking at +y wall
    assert d[2] == pytest.approx(1.0)  # left is -x


def test_obstacle_hit_and_max_range_clip():
    obs = np.array([[3.0, 2.0, 0.5]])
    d = cast_rays(1.0, 2.0, 0.0, ray_angles(8), obs, 6.0, 3.5)
    assert d[0] == pytest.approx(1.5)   # 3.0 - 0.5 - 1.0
    assert d[4] == pytest.approx(1.0)
    assert d.max() <= 3.5
    d_far = cast_rays(0.5, 3.0, 0.0, ray_angles(8), EMPTY, 6.0, 3.5)
    assert d_far[0] == pytest.approx(3.5)


def test_obstacle_behind_is_ignored():
    obs = np.array([[0.5, 2.0, 0.2]])
    d = cast_rays(2.0, 2.0, 0.0, ray_angles(8), obs, 6.0, 10.0)
    assert d[0] == pytest.approx(4.0)
    assert d[4] == pytest.approx(2.0 - 0.5 - 0.2)


def test_approaching_obstacle_reduces_front_reading():
    obs = np.array([[4.0, 3.0, 0.4]])
    readings = [cast_rays(x, 3.0, 0.0, ray_angles(8), obs, 6.0, 3.5)[0] for x in (1.0, 2.0, 3.0)]
    assert readings[0] > readings[1] > readings[2]
