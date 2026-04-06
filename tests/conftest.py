import importlib.util
from pathlib import Path

import pytest
from rich.console import Console

from core.dashboard_state import DashboardState
from core.models import Observation
from tasks.task_config import TaskConfig


ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def base_task_dict():
    return {
        "num_zones": 3,
        "initial_temperatures": [45.0, 48.0, 50.0],
        "initial_workloads": [0.4, 0.5, 0.3],
        "ambient_temperature": 22.0,
        "safe_temperature": 85.0,
        "max_steps": 4,
        "target_temperature": 70.0,
        "workload_volatility": 0.0,
        "jitter_bypass_threshold": 2.0,
    }


@pytest.fixture
def task_config(base_task_dict):
    return TaskConfig.model_validate(base_task_dict)


@pytest.fixture
def observation(task_config):
    return Observation(
        temperatures=task_config.initial_temperatures,
        workloads=task_config.initial_workloads,
        cooling=[0.0] * task_config.num_zones,
        ambient_temp=task_config.ambient_temperature,
        time_step=0,
    )


@pytest.fixture
def dashboard_state():
    return DashboardState(
        step=2,
        temperatures=[72.0, 77.5, 81.0],
        cooling=[0.2, 0.5, 0.8],
        reward=-3.2,
        total_reward=-8.7,
        done=False,
        safe_temp=85.0,
        num_zones=3,
        target_temp=70.0,
        max_steps=10,
        overshoot_term=-0.5,
        energy_term=-0.7,
        smoothness_term=-0.2,
        deltas=[0.0, 0.5, -0.3],
        forecasts=["Stable/Cooling", "~4 steps", "~2 steps"],
        action_quality=["Neutral", "Stable (Maintaining)", "Optimal (Reactive)"],
        trajectory_status="TRENDING",
        policy_signature={"avg": 0.5, "var": 0.02},
        temp_history=[[71.0, 76.0, 82.0], [72.0, 77.5, 81.0]],
        action_history=[[0.1, 0.4, 0.9], [0.2, 0.5, 0.8]],
        prompt='{"cooling":[0.1,0.2,0.3]}',
        llm_raw_output='{"cooling":[0.2,0.5,0.8]}',
        parsed_action={"cooling": [0.2, 0.5, 0.8]},
        task_name="easy.json",
        model="test-model",
    )


@pytest.fixture(autouse=True)
def reset_global_sessions():
    import core.env
    import server.app

    core.env.CURRENT_SESSION = None
    server.app.CURRENT_SESSION = None
    yield
    core.env.CURRENT_SESSION = None
    server.app.CURRENT_SESSION = None


def render_text(renderable) -> str:
    console = Console(width=140, record=True)
    console.print(renderable)
    return console.export_text()


@pytest.fixture
def render_rich_text():
    return render_text


@pytest.fixture
def root_inference_module():
    module_path = ROOT_DIR / "inference.py"
    spec = importlib.util.spec_from_file_location("adctm_root_inference_test", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load root inference module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
