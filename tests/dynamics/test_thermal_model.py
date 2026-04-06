import pytest
import numpy as np

from core.models import Action, Observation
from dynamics.thermal_model import apply_transition
from tasks.task_config import TaskConfig


def test_apply_transition_is_deterministic_without_volatility(task_config, observation):
    action = Action(cooling=[0.2, 0.4, 0.6])
    next_obs = apply_transition(observation, action, task_config, np.random.default_rng(42))
    assert next_obs.time_step == 1
    assert next_obs.cooling == [0.2, 0.4, 0.6]
    assert next_obs.workloads == pytest.approx(observation.workloads)


def test_apply_transition_clips_workloads(monkeypatch, task_config, observation):
    task_config = task_config.model_copy(update={"workload_volatility": 0.5})
    class MockRNG:
        def uniform(self, low, high, size):
            return [1.0, -1.0, 0.0]
    next_obs = apply_transition(observation, Action(cooling=[0.0, 0.0, 0.0]), task_config, MockRNG())
    assert next_obs.workloads == [1.0, 0.0, 0.3]


def test_apply_transition_never_drops_below_ambient(task_config):
    obs = Observation(
        temperatures=[22.0, 22.1, 22.2],
        workloads=[0.0, 0.0, 0.0],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=22.0,
        time_step=0,
    )
    next_obs = apply_transition(obs, Action(cooling=[1.0, 1.0, 1.0]), task_config, np.random.default_rng(42))
    assert min(next_obs.temperatures) >= 22.0


def test_apply_transition_applies_degradation_after_threshold(base_task_dict):
    base_task_dict.update({"degradation_step": 1, "degraded_zone": 1})
    config = TaskConfig.model_validate(base_task_dict)
    obs = Observation(
        temperatures=[45.0, 48.0, 50.0],
        workloads=[0.4, 0.5, 0.3],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=22.0,
        time_step=1,
    )
    cooled = apply_transition(obs, Action(cooling=[0.5, 0.5, 0.5]), config, np.random.default_rng(42))
    assert cooled.temperatures[1] > cooled.temperatures[0]
