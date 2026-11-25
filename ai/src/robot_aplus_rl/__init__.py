"""Robot Aplus Week 9: PPO vs SAC for goal-directed navigation in a simplified 2D world.

This package is a numerical (non-Gazebo) simulator; results obtained with it must not be
reported as Gazebo or physical-robot results.
"""

from robot_aplus_rl.config import EnvConfig, load_env_config
from robot_aplus_rl.navigation_env import Navigation2DEnv

__all__ = ["EnvConfig", "Navigation2DEnv", "load_env_config"]
__version__ = "0.9.0"
