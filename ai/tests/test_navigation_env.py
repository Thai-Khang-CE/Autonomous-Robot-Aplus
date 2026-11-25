import math
import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from robot_aplus_rl import Navigation2DEnv
from robot_aplus_rl.scenarios import CATEGORIES, custom_scenario, generate_scenario

FORWARD = np.array([1.0, 0.0], dtype=np.float32)  # v = v_max, omega = 0
STOP = np.array([-1.0, 0.0], dtype=np.float32)  # v = 0


def reset_custom(env, start, goal, obstacles=()):
    return env.reset(seed=0, options={"scenario": custom_scenario(start, goal, obstacles)})


def run_until_done(env, action, limit=400):
    for _ in range(limit):
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            return obs, r, term, trunc, info
    raise AssertionError("episode did not end")


def test_gymnasium_env_checker(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any checker warning fails the test
        gym_check_env(Navigation2DEnv(cfg), skip_render_check=True)


def test_sb3_env_checker(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sb3_check_env(Navigation2DEnv(cfg), warn=True)


def test_spaces(env):
    assert env.observation_space.shape == (11,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (2,)
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    assert env.observation_space.low[:9].min() == 0 and env.observation_space.low[9:].min() == -1


def test_observation_bounds_and_dtype_over_random_rollouts(env):
    obs, _ = env.reset(seed=123)
    for _ in range(2000):
        assert obs.dtype == np.float32 and obs.shape == (11,)
        assert env.observation_space.contains(obs)
        assert np.all(np.isfinite(obs))
        assert obs[-2] ** 2 + obs[-1] ** 2 == pytest.approx(1.0, abs=1e-5)
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        if term or trunc:
            obs, _ = env.reset()


def test_action_mapping_and_clipping(env):
    env.reset(seed=0)
    assert env.action_to_velocity([-1, 0]) == pytest.approx((0.0, 0.0))
    assert env.action_to_velocity([1, 1]) == pytest.approx((0.35, 1.5))
    assert env.action_to_velocity([0, -1]) == pytest.approx((0.175, -1.5))
    assert env.action_to_velocity([5, -9]) == pytest.approx((0.35, -1.5))  # clipped
    with pytest.raises(ValueError):
        env.action_to_velocity([np.nan, 0])
    with pytest.raises(ValueError):
        env.action_to_velocity([0.0, 0.0, 0.0])


def test_observation_values_match_geometry(env, cfg):
    obs, _ = reset_custom(env, (1.0, 3.0, 0.0), (4.0, 3.0), [(3.0, 3.0, 0.5)])
    assert obs[0] == pytest.approx(1.5 / cfg.max_range)  # front ray hits the obstacle
    assert obs[4] == pytest.approx(1.0 / cfg.max_range)  # rear ray hits the x=0 wall
    assert obs[8] == pytest.approx(3.0 / cfg.world_diagonal)
    assert (obs[9], obs[10]) == pytest.approx((0.0, 1.0))  # goal straight ahead


def test_kinematics_inside_env(env):
    reset_custom(env, (1.0, 3.0, 0.0), (5.0, 3.0))
    env.step(FORWARD)
    x, y, th = env.pose
    assert (x, y, th) == pytest.approx((1.0 + 0.35 * 0.2, 3.0, 0.0))
    env.step(np.array([-1.0, 1.0]))
    assert env.pose == pytest.approx((x, y, 0.3))


def test_moving_toward_goal_positive_reward_and_away_negative(env):
    reset_custom(env, (1.0, 3.0, 0.0), (5.0, 3.0))
    _, r, *_ = env.step(FORWARD)
    assert r > 0
    reset_custom(env, (3.0, 3.0, math.pi), (5.0, 3.0))
    _, r, *_ = env.step(FORWARD)
    assert r < 0


def test_goal_reached_terminates_with_success(env, cfg):
    reset_custom(env, (1.0, 3.0, 0.0), (2.0, 3.0))
    _, r, term, trunc, info = run_until_done(env, FORWARD)
    assert term and not trunc
    assert info["outcome"] == "success" and info["is_success"]
    assert info["distance_to_goal"] <= cfg.goal_tolerance
    assert info["reward_components"]["goal"] == cfg.reward.goal
    with pytest.raises(RuntimeError):  # terminal bonus cannot be collected twice
        env.step(FORWARD)


def test_boundary_collision(env, cfg):
    reset_custom(env, (5.0, 3.0, 0.0), (1.0, 1.0))
    _, r, term, trunc, info = run_until_done(env, FORWARD)
    assert term and not trunc and info["outcome"] == "collision" and info["is_collision"]
    assert env.pose[0] == pytest.approx(cfg.world_size - cfg.robot_radius)  # stopped at contact
    assert r < -cfg.reward.collision + 1.0
    with pytest.raises(RuntimeError):
        env.step(FORWARD)


def test_obstacle_collision_and_ray_reading_decreases(env, cfg):
    obs, _ = reset_custom(env, (1.0, 3.0, 0.0), (5.0, 1.0), [(3.0, 3.0, 0.4)])
    front = [obs[0]]
    info = {}
    for _ in range(100):
        obs, r, term, trunc, info = env.step(FORWARD)
        front.append(obs[0])
        if term or trunc:
            break
    assert info["outcome"] == "collision"
    assert all(a > b for a, b in zip(front, front[1:]))  # approaching -> monotone decrease
    # the disc touches the obstacle: centre distance == r_obstacle + r_robot (no penetration)
    assert math.hypot(env.pose[0] - 3.0, env.pose[1] - 3.0) == pytest.approx(0.4 + cfg.robot_radius)


def test_timeout_is_truncation(env, cfg):
    reset_custom(env, (1.0, 1.0, 0.0), (5.0, 5.0))
    steps = 0
    while True:
        _, r, term, trunc, info = env.step(STOP)
        steps += 1
        if term or trunc:
            break
    assert trunc and not term
    assert steps == cfg.max_episode_steps
    assert info["outcome"] == "timeout" and info["is_timeout"] and not info["is_success"]
    assert r == pytest.approx(-cfg.reward.step)


def test_episode_summary_tracks_path_and_trajectory(env):
    reset_custom(env, (1.0, 3.0, 0.0), (2.0, 3.0))
    *_, info = run_until_done(env, FORWARD)
    s = info["episode_summary"]
    assert s["outcome"] == "success"
    assert s["episode_length"] == len(s["trajectory"]) - 1
    assert s["path_length"] == pytest.approx(0.07 * s["episode_length"])
    assert s["time_to_goal"] == pytest.approx(0.2 * s["episode_length"])
    assert s["initial_distance"] == pytest.approx(1.0)


@pytest.mark.parametrize("category", CATEGORIES)
def test_generated_scenarios_are_valid(cfg, category):
    from robot_aplus_rl.geometry import clearance
    from robot_aplus_rl.planning import reference_path

    for seed in range(15):
        sc = generate_scenario(category, seed, cfg)
        assert clearance(*sc.start[:2], sc.obstacles, cfg.world_size) >= cfg.robot_radius
        assert clearance(*sc.goal, sc.obstacles, cfg.world_size) >= cfg.robot_radius
        assert math.dist(sc.start[:2], sc.goal) >= cfg.min_start_goal_distance
        ref = reference_path(sc.start[:2], sc.goal, sc.obstacles, cfg.world_size,
                             cfg.robot_radius, cfg.planning_margin, cfg.planning_resolution)
        assert ref is not None


def test_random_policy_full_episodes(env):
    outcomes = []
    env.reset(seed=7)
    env.action_space.seed(7)
    while len(outcomes) < 20:
        *_, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            outcomes.append(info["outcome"])
            env.reset()
    assert set(outcomes) <= {"success", "collision", "timeout"}


def test_render_rgb_array(cfg):
    env = Navigation2DEnv(cfg, render_mode="rgb_array")
    env.reset(seed=1)
    env.step(FORWARD)
    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
