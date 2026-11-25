"""Typed environment configuration loaded from ``configs/environment.yaml``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "configs" / "environment.yaml"


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 10.0
    goal: float = 20.0
    collision: float = 20.0
    step: float = 0.05


@dataclass(frozen=True)
class EnvConfig:
    robot_radius: float = 0.18
    max_linear_velocity: float = 0.35
    max_angular_velocity: float = 1.5
    world_size: float = 6.0
    goal_tolerance: float = 0.25
    dt: float = 0.2
    max_episode_steps: int = 250
    num_rays: int = 8
    max_range: float = 3.5
    reward: RewardConfig = field(default_factory=RewardConfig)
    train_category_weights: dict[str, float] = field(
        default_factory=lambda: {"open": 0.3, "indoor": 0.4, "narrow": 0.3}
    )
    min_start_goal_distance: float = 2.0
    start_clearance: float = 0.15
    goal_clearance: float = 0.05
    planning_resolution: float = 0.05
    planning_margin: float = 0.03

    def __post_init__(self) -> None:
        positive = {
            "robot_radius": self.robot_radius,
            "max_linear_velocity": self.max_linear_velocity,
            "max_angular_velocity": self.max_angular_velocity,
            "world_size": self.world_size,
            "goal_tolerance": self.goal_tolerance,
            "dt": self.dt,
            "max_range": self.max_range,
            "planning_resolution": self.planning_resolution,
        }
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.max_episode_steps < 1 or self.num_rays < 1:
            raise ValueError("max_episode_steps and num_rays must be >= 1")
        if 2 * self.robot_radius >= self.world_size:
            raise ValueError("robot does not fit inside the world")
        if not self.train_category_weights or min(self.train_category_weights.values()) < 0:
            raise ValueError("train_category_weights must be non-empty and non-negative")

    @property
    def world_diagonal(self) -> float:
        return float(self.world_size * 2**0.5)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def env_config_from_dict(raw: dict[str, Any]) -> EnvConfig:
    """Build an :class:`EnvConfig` from the nested YAML layout."""
    robot = raw.get("robot", {})
    world = raw.get("world", {})
    task = raw.get("task", {})
    sensor = raw.get("sensor", {})
    scen = raw.get("scenarios", {})
    defaults = EnvConfig()
    return EnvConfig(
        robot_radius=float(robot.get("radius", defaults.robot_radius)),
        max_linear_velocity=float(robot.get("max_linear_velocity", defaults.max_linear_velocity)),
        max_angular_velocity=float(robot.get("max_angular_velocity", defaults.max_angular_velocity)),
        world_size=float(world.get("size", defaults.world_size)),
        goal_tolerance=float(task.get("goal_tolerance", defaults.goal_tolerance)),
        dt=float(task.get("dt", defaults.dt)),
        max_episode_steps=int(task.get("max_episode_steps", defaults.max_episode_steps)),
        num_rays=int(sensor.get("num_rays", defaults.num_rays)),
        max_range=float(sensor.get("max_range", defaults.max_range)),
        reward=RewardConfig(**{k: float(v) for k, v in raw.get("reward", {}).items()}),
        train_category_weights={
            str(k): float(v)
            for k, v in scen.get("train_category_weights", defaults.train_category_weights).items()
        },
        min_start_goal_distance=float(
            scen.get("min_start_goal_distance", defaults.min_start_goal_distance)
        ),
        start_clearance=float(scen.get("start_clearance", defaults.start_clearance)),
        goal_clearance=float(scen.get("goal_clearance", defaults.goal_clearance)),
        planning_resolution=float(scen.get("planning_resolution", defaults.planning_resolution)),
        planning_margin=float(scen.get("planning_margin", defaults.planning_margin)),
    )


def load_env_config(path: str | Path | None = None) -> EnvConfig:
    path = Path(path) if path is not None else DEFAULT_ENV_CONFIG
    with open(path, encoding="utf-8") as fh:
        return env_config_from_dict(yaml.safe_load(fh) or {})


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
