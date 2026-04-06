from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as root_app
import server.app as server_app
from core.models import Action, Observation
from core.simulator import SimulationSession
from grader.evaluator import evaluate_trajectory
from tasks.task_config import TaskConfig


ROOT_DIR = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT_DIR / "tasks"


def _read(path: str) -> str:
    return (ROOT_DIR / path).read_text(encoding="utf-8")


def test_required_submission_files_exist():
    for relative_path in ["openenv.yaml", "README.md", "Dockerfile", "inference.py", "app.py"]:
        assert (ROOT_DIR / relative_path).exists(), f"Missing required file: {relative_path}"


def test_root_submission_imports_cleanly():
    assert root_app.app is server_app.app


def test_openenv_yaml_contains_required_spec_fields():
    content = _read("openenv.yaml")
    for snippet in [
        'name: "ADCTM"',
        'version: "1.0.0"',
        "observation:",
        'type: "core.models.Observation"',
        "action:",
        'type: "core.models.Action"',
        "reward:",
        'type: "core.models.Reward"',
        "capabilities:",
        '- "reset"',
        '- "step"',
        '- "state"',
    ]:
        assert snippet in content


def test_readme_covers_submission_flow():
    content = _read("README.md")
    for section in [
        "Overview",
        "Tasks and Difficulty Scaling",
        "Installation",
        "Submission Execution",
    ]:
        assert section in content


def test_root_inference_script_matches_requirement_contract():
    content = _read("inference.py")
    assert "from openai import OpenAI" in content
    assert 'os.getenv("API_BASE_URL"' in content
    assert 'os.getenv("MODEL_NAME"' in content
    assert 'os.getenv("HF_TOKEN")' in content


def test_active_tasks_are_rooted_in_tasks_directory():
    task_files = sorted(TASKS_DIR.glob("*.json"))
    assert [path.name for path in task_files] == ["easy.json", "hard.json", "medium.json"]

    configs = {}
    for path in task_files:
        with path.open(encoding="utf-8") as handle:
            cfg = TaskConfig.model_validate(json.load(handle))
        configs[path.stem] = cfg
        assert cfg.seed is not None

    assert configs["easy"].num_zones < configs["medium"].num_zones < configs["hard"].num_zones
    assert configs["easy"].max_steps < configs["medium"].max_steps < configs["hard"].max_steps


def test_grader_outputs_normalized_deterministic_scores_for_all_tasks():
    for task_name in ["easy", "medium", "hard"]:
        with (TASKS_DIR / f"{task_name}.json").open(encoding="utf-8") as handle:
            cfg = TaskConfig.model_validate(json.load(handle))

        observations = [
            {
                "temperatures": list(cfg.initial_temperatures),
                "workloads": list(cfg.initial_workloads),
                "cooling": [0.0] * cfg.num_zones,
                "ambient_temp": cfg.ambient_temperature,
                "time_step": 0,
            }
        ]
        actions = []
        for step in range(1, min(cfg.max_steps, 3) + 1):
            cooling = [0.25] * cfg.num_zones
            actions.append({"cooling": cooling})
            observations.append(
                {
                    "temperatures": [temp + 0.5 for temp in cfg.initial_temperatures],
                    "workloads": list(cfg.initial_workloads),
                    "cooling": cooling,
                    "ambient_temp": cfg.ambient_temperature,
                    "time_step": step,
                }
            )

        score_one = evaluate_trajectory(observations, actions, cfg)
        score_two = evaluate_trajectory(observations, actions, cfg)
        assert 0.0 <= score_one <= 1.0
        assert score_one == score_two


def test_api_exposes_reset_step_state_for_root_submission(monkeypatch):
    client = TestClient(server_app.app)

    reset_response = client.post("/reset")
    assert reset_response.status_code == 200

    step_response = client.post("/step", json={"cooling": [0.2, 0.2, 0.2]})
    assert step_response.status_code == 200
    payload = step_response.json()
    assert {"observation", "reward", "done", "info"} <= payload.keys()

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state = state_response.json()
    assert {"config", "observation", "step_counter", "done", "seed", "rng_state"} <= state.keys()


def test_dockerfile_matches_server_entrypoint():
    content = _read("Dockerfile")
    for snippet in [
        "FROM python:3.11-slim",
        "WORKDIR /app",
        "COPY requirements.txt .",
        "RUN pip install --no-cache-dir -r requirements.txt",
        "COPY . .",
        "EXPOSE 7860",
        "ENV PORT=7860",
        'CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]',
    ]:
        assert snippet in content


@pytest.mark.parametrize(
    ("task_name", "threshold"),
    [("easy", 0.60), ("medium", 0.50), ("hard", 0.35)],
)
def test_zero_policy_is_weak_on_active_tasks(task_name: str, threshold: float):
    session = SimulationSession.from_task_name(task_name)
    actions = []
    observations = [session.observation.model_dump()]

    while not session.done:
        action = Action(cooling=[0.0] * session.config.num_zones)
        actions.append(action.model_dump())
        step_data = session.step(action.model_dump())
        observations.append(step_data["observation"])

    score = evaluate_trajectory(observations, actions, session.config)
    assert score < threshold
