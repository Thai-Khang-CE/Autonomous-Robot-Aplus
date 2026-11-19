import numpy as np

from robot_aplus_rl import Navigation2DEnv
from robot_aplus_rl.scenarios import generate_scenario, scenario_set


def rollout(env, seed, n=300):
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    trace = [obs]
    for _ in range(n):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        trace.append(np.append(obs, r))
        if term or trunc:
            obs, _ = env.reset()
            trace.append(obs)
    return trace


def test_same_seed_same_rollout(cfg):
    a = rollout(Navigation2DEnv(cfg), 42)
    b = rollout(Navigation2DEnv(cfg), 42)
    assert len(a) == len(b)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_different_seed_different_scenarios(cfg):
    e = Navigation2DEnv(cfg)
    e.reset(seed=1)
    s1 = e.scenario.scenario_id
    e.reset(seed=2)
    assert e.scenario.scenario_id != s1


def test_scenario_generation_deterministic(cfg):
    for cat in ("open", "indoor", "narrow"):
        a, b = generate_scenario(cat, 99, cfg), generate_scenario(cat, 99, cfg)
        assert np.array_equal(a.obstacles, b.obstacles)
        assert a.start == b.start and a.goal == b.goal


def test_validation_and_test_sets_are_disjoint_from_training_range(cfg):
    val = scenario_set("validation", 3, cfg)
    test = scenario_set("test", 3, cfg)
    assert all(s.seed >= 1_000_000 for s in val + test)
    assert not {s.scenario_id for s in val} & {s.scenario_id for s in test}


def test_scenario_pool_cycles_in_order(cfg):
    pool = scenario_set("validation", 2, cfg)
    env = Navigation2DEnv(cfg, scenario_pool=pool)
    ids = []
    for _ in range(len(pool) + 1):
        env.reset()
        ids.append(env.scenario.scenario_id)
    assert ids[: len(pool)] == [s.scenario_id for s in pool]
    assert ids[-1] == pool[0].scenario_id
