import pytest

from robot_aplus_rl.config import RewardConfig
from robot_aplus_rl.rewards import compute_reward

CFG = RewardConfig(progress=10.0, goal=20.0, collision=20.0, step=0.05)


def test_progress_toward_goal_is_positive():
    r, c = compute_reward(2.0, 1.93, False, False, CFG)
    assert r == pytest.approx(10 * 0.07 - 0.05)
    assert r > 0 and c["progress"] > 0


def test_moving_away_is_negative():
    r, _ = compute_reward(2.0, 2.07, False, False, CFG)
    assert r < 0


def test_standing_still_costs_step_penalty():
    r, _ = compute_reward(2.0, 2.0, False, False, CFG)
    assert r == pytest.approx(-0.05)


def test_goal_bonus_and_collision_penalty():
    r_goal, c = compute_reward(0.3, 0.24, True, False, CFG)
    assert c["goal"] == 20.0 and r_goal > 19
    r_col, c = compute_reward(1.0, 1.0, False, True, CFG)
    assert c["collision"] == -20.0 and r_col < -19


def test_success_and_collision_are_exclusive():
    with pytest.raises(ValueError):
        compute_reward(1.0, 0.1, True, True, CFG)
