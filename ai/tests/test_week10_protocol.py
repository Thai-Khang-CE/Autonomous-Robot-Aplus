"""Regression tests for the frozen Week 10 protocol and reference validity."""
from importlib import import_module

import numpy as np
import pytest


@pytest.fixture
def week10(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("week10")


def test_frozen_holdout_detects_tampering(tmp_path, week10):
    week10.freeze(tmp_path)
    first_hash = week10.digest(tmp_path / "manifests/protocol.json")
    week10.freeze(tmp_path)
    assert week10.digest(tmp_path / "manifests/protocol.json") == first_hash
    validation = week10.load_scenarios(tmp_path, "validation")
    test = week10.load_scenarios(tmp_path, "test")
    assert len({sc.scenario_id for sc in validation}) == 30
    assert len({sc.scenario_id for sc in test}) == 150
    assert {sc.seed for sc in validation}.isdisjoint(sc.seed for sc in test)
    path = tmp_path / "configs/test_scenarios.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen file changed"):
        week10.load_scenarios(tmp_path, "test")


def test_budget_and_safety_objective(week10):
    week10.validate_budget("ppo", {"n_steps": 128, "batch_size": 128}, 8, 8192)
    with pytest.raises(ValueError, match="whole rollouts"):
        week10.validate_budget("ppo", {"n_steps": 128, "batch_size": 250}, 8, 8192)
    with pytest.raises(ValueError, match="whole rollouts"):
        week10.validate_budget("ppo", {"n_steps": 128, "batch_size": 128}, 8, 8191)
    with pytest.raises(ValueError, match="learning starts"):
        week10.validate_budget("sac", {"learning_starts": 8192, "buffer_size": 300000}, 1, 8192)
    assert week10.navigation_score(0.6, 0.2, 0.5) == pytest.approx(0.50)
    assert week10.navigation_score(0.6, 0.1, 0.5) > week10.navigation_score(0.6, 0.2, 0.5)


def test_invalid_approximate_path_excluded(week10, monkeypatch):
    from robot_aplus_rl.config import load_env_config
    from robot_aplus_rl.scenarios import custom_scenario
    module = import_module("evaluate_week10")
    sc = custom_scenario((1, 3, 0), (5, 3), [(3, 3, 0.3)])
    monkeypatch.setattr(module, "reference_path", lambda *args: [sc.start[:2], sc.goal])
    assert module.valid_reference(sc, load_env_config()) is None
