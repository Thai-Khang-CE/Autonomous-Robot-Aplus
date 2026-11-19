import math

import numpy as np
import pytest

from robot_aplus_rl.geometry import (
    clearance,
    disc_in_collision,
    first_contact_fraction,
    goal_distance_and_bearing,
    unicycle_step,
    wrap_angle,
)


@pytest.mark.parametrize(
    "angle, expected",
    [(0.0, 0.0), (math.pi / 2, math.pi / 2), (3 * math.pi, -math.pi), (-3 * math.pi / 2, math.pi / 2),
     (2 * math.pi + 0.1, 0.1), (-math.pi, -math.pi)],
)
def test_wrap_angle(angle, expected):
    assert wrap_angle(angle) == pytest.approx(expected, abs=1e-12)
    assert -math.pi <= wrap_angle(angle) < math.pi


def test_unicycle_straight_and_turn():
    x, y, th = unicycle_step(1.0, 2.0, 0.0, 0.35, 0.0, 0.2)
    assert (x, y, th) == pytest.approx((1.07, 2.0, 0.0))
    x, y, th = unicycle_step(1.0, 2.0, math.pi / 2, 0.35, 0.0, 0.2)
    assert (x, y) == pytest.approx((1.0, 2.07))
    # pure rotation: position unchanged, heading advances by omega*dt
    x, y, th = unicycle_step(1.0, 2.0, 0.0, 0.0, 1.5, 0.2)
    assert (x, y, th) == pytest.approx((1.0, 2.0, 0.3))
    # heading wraps
    _, _, th = unicycle_step(0.0, 0.0, math.pi - 0.1, 0.0, 1.5, 0.2)
    assert th == pytest.approx(-math.pi + 0.2)


def test_goal_distance_and_bearing():
    d, a = goal_distance_and_bearing(0.0, 0.0, 0.0, 3.0, 4.0)
    assert d == pytest.approx(5.0)
    assert a == pytest.approx(math.atan2(4, 3))
    d, a = goal_distance_and_bearing(1.0, 1.0, math.pi / 2, 1.0, 3.0)  # goal straight ahead
    assert a == pytest.approx(0.0)
    _, a = goal_distance_and_bearing(1.0, 1.0, 0.0, 0.0, 1.0)  # goal behind
    assert abs(a) == pytest.approx(math.pi)
    _, a = goal_distance_and_bearing(1.0, 1.0, 0.0, 1.0, 0.0)  # goal to the right
    assert a == pytest.approx(-math.pi / 2)


def test_clearance_and_disc_collision_use_robot_radius():
    obs = np.array([[3.0, 3.0, 0.5]])
    assert clearance(3.0, 4.0, obs, 6.0) == pytest.approx(0.5)
    assert not disc_in_collision(3.0, 4.0, 0.18, obs, 6.0)
    # centre outside the obstacle but disc overlaps it -> collision (not a point test)
    assert disc_in_collision(3.0, 3.6, 0.18, obs, 6.0)
    assert disc_in_collision(0.1, 3.0, 0.18, np.zeros((0, 3)), 6.0)  # wall overlap


def test_swept_contact_obstacle_and_no_tunnelling():
    obs = np.array([[2.0, 1.0, 0.05]])  # very thin obstacle
    # one large jump straight through the obstacle must still be detected
    s = first_contact_fraction((1.0, 1.0), (3.0, 1.0), 0.18, obs, 6.0)
    assert s is not None
    contact_x = 1.0 + s * 2.0
    assert contact_x == pytest.approx(2.0 - 0.05 - 0.18)
    assert first_contact_fraction((1.0, 2.0), (3.0, 2.0), 0.18, obs, 6.0) is None


def test_swept_contact_walls():
    s = first_contact_fraction((1.0, 1.0), (-1.0, 1.0), 0.18, np.zeros((0, 3)), 6.0)
    assert 1.0 + s * -2.0 == pytest.approx(0.18)
    s = first_contact_fraction((5.0, 5.0), (5.0, 7.0), 0.18, np.zeros((0, 3)), 6.0)
    assert 5.0 + s * 2.0 == pytest.approx(6.0 - 0.18)
    assert first_contact_fraction((0.1, 3.0), (0.2, 3.0), 0.18, np.zeros((0, 3)), 6.0) == 0.0
