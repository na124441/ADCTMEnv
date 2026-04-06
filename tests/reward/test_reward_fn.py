import pytest

from core.models import Action, Observation
from reward.reward_fn import compute_reward


def test_compute_reward_zero_penalty_on_safe_first_step(task_config):
    prev_obs = Observation(
        temperatures=[60.0, 61.0, 62.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=22.0,
        time_step=0,
    )
    curr_obs = prev_obs.model_copy(update={"time_step": 1})
    reward = compute_reward(prev_obs, curr_obs, Action(cooling=[0.0, 0.0, 0.0]), task_config)
    assert reward.value < 0.0
    assert reward.value == pytest.approx(-0.034846938775510206)


def test_compute_reward_penalizes_violations_energy_and_jitter(task_config):
    prev_obs = Observation(
        temperatures=[60.0, 60.0, 60.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.1, 0.1, 0.1],
        ambient_temp=22.0,
        time_step=1,
    )
    curr_obs = Observation(
        temperatures=[86.0, 87.0, 84.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.5, 0.6, 0.7],
        ambient_temp=22.0,
        time_step=2,
    )
    reward = compute_reward(prev_obs, curr_obs, Action(cooling=[0.5, 0.6, 0.7]), task_config)
    assert reward.value < 0.0


def test_compute_reward_bypasses_jitter_near_safe_limit(task_config):
    prev_obs = Observation(
        temperatures=[80.0, 80.0, 80.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[0.0, 0.0, 0.0],
        ambient_temp=22.0,
        time_step=1,
    )
    curr_obs = Observation(
        temperatures=[84.5, 84.0, 83.0],
        workloads=[0.2, 0.2, 0.2],
        cooling=[1.0, 1.0, 1.0],
        ambient_temp=22.0,
        time_step=2,
    )
    reward = compute_reward(prev_obs, curr_obs, Action(cooling=[1.0, 1.0, 1.0]), task_config)
    assert reward.value == pytest.approx(-3.0933673469387757)
