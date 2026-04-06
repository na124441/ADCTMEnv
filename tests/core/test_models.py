import pytest
from pydantic import ValidationError

from core.models import Action, InfoDict, Observation, Reward


def test_observation_accepts_equal_length_lists():
    obs = Observation(
        temperatures=[1.0, 2.0],
        workloads=[0.1, 0.2],
        cooling=[0.0, 0.3],
        ambient_temp=20.0,
        time_step=0,
    )
    assert len(obs.temperatures) == 2


def test_observation_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        Observation(
            temperatures=[1.0, 2.0],
            workloads=[0.1],
            cooling=[0.0, 0.3],
            ambient_temp=20.0,
            time_step=0,
        )


def test_observation_rejects_empty_lists():
    with pytest.raises(ValidationError):
        Observation(temperatures=[], workloads=[], cooling=[], ambient_temp=20.0, time_step=0)


def test_observation_rejects_negative_time_step():
    with pytest.raises(ValidationError):
        Observation(
            temperatures=[1.0],
            workloads=[0.1],
            cooling=[0.0],
            ambient_temp=20.0,
            time_step=-1,
        )


def test_action_clips_values():
    action = Action(cooling=[-1.0, 0.4, 1.8])
    assert action.cooling == [0.0, 0.4, 1.0]


def test_reward_accepts_scalar():
    reward = Reward(value=-2.5)
    assert reward.value == -2.5


def test_info_dict_requires_step():
    with pytest.raises(ValidationError):
        InfoDict()
