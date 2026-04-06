import pytest

from core.models import Reward
from core.simulator import SimulationSession


def test_simulation_session_initializes_from_config(task_config):
    session = SimulationSession(task_config)
    assert session.step_counter == 0
    assert session.done is False
    assert session.observation.time_step == 0
    assert session.observation.cooling == [0.0] * task_config.num_zones


def test_simulation_session_from_task_name_loads_easy():
    session = SimulationSession.from_task_name("easy.json")
    assert session.config.num_zones == 3


def test_simulation_session_step_advances(monkeypatch, task_config):
    session = SimulationSession(task_config)

    def fake_transition(prev_obs, action, config, rng):
        return prev_obs.model_copy(
            update={
                "temperatures": [value + 1.0 for value in prev_obs.temperatures],
                "cooling": action.cooling,
                "time_step": prev_obs.time_step + 1,
            }
        )

    monkeypatch.setattr("core.simulator.apply_transition", fake_transition)
    monkeypatch.setattr("core.simulator.compute_reward", lambda **_: Reward(value=-1.5))

    result = session.step({"cooling": [0.1, 0.2, 0.3]})
    assert result["done"] is False
    assert result["info"]["step"] == 1
    assert session.step_counter == 1
    assert session.observation.time_step == 1


def test_simulation_session_step_rejects_wrong_action_length(task_config):
    session = SimulationSession(task_config)
    with pytest.raises(ValueError, match="Action length"):
        session.step({"cooling": [0.1]})


def test_simulation_session_rejects_step_after_done(monkeypatch, task_config):
    task_config = task_config.model_copy(update={"max_steps": 1})
    session = SimulationSession(task_config)
    monkeypatch.setattr("core.simulator.apply_transition", lambda prev_obs, action, config, rng: prev_obs.model_copy(update={"cooling": action.cooling, "time_step": 1}))
    monkeypatch.setattr("core.simulator.compute_reward", lambda **_: Reward(value=0.0))
    session.step({"cooling": [0.1, 0.1, 0.1]})
    with pytest.raises(ValueError, match="Episode already finished"):
        session.step({"cooling": [0.1, 0.1, 0.1]})
