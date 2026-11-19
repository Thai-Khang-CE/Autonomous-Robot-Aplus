import pytest

from robot_aplus_rl import Navigation2DEnv, load_env_config


@pytest.fixture(scope="session")
def cfg():
    return load_env_config()


@pytest.fixture
def env(cfg):
    e = Navigation2DEnv(cfg)
    yield e
    e.close()
