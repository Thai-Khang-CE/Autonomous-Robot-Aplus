"""Reward function r_t = k_p (d_t - d_{t+1}) + k_g I_goal - k_c I_collision - k_t."""

from __future__ import annotations

from robot_aplus_rl.config import RewardConfig


def compute_reward(
    prev_distance: float,
    new_distance: float,
    reached_goal: bool,
    collided: bool,
    cfg: RewardConfig,
) -> tuple[float, dict[str, float]]:
    """Return the scalar reward and its components (for logging / tests).

    The progress term is the difference of a potential (negative goal distance) and therefore
    rewards motion toward the goal and penalises motion away from it. Terminal terms are paid
    only on the step in which the event occurs; the environment refuses further steps after
    termination, so they cannot be paid twice.
    """
    if reached_goal and collided:
        raise ValueError("a step cannot be both a success and a collision")
    components = {
        "progress": cfg.progress * (prev_distance - new_distance),
        "goal": cfg.goal if reached_goal else 0.0,
        "collision": -cfg.collision if collided else 0.0,
        "step": -cfg.step,
    }
    return float(sum(components.values())), components
