import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tasks.task_config import TaskConfig


TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"


@pytest.mark.parametrize("task_name", ["easy.json", "medium.json", "hard.json"])
def test_task_json_files_are_valid(task_name):
    with (TASKS_DIR / task_name).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cfg = TaskConfig.model_validate(payload)
    assert cfg.num_zones == len(cfg.initial_temperatures) == len(cfg.initial_workloads)


def test_task_config_rejects_mismatched_temperatures(base_task_dict):
    base_task_dict["initial_temperatures"] = [1.0]
    with pytest.raises(ValidationError):
        TaskConfig.model_validate(base_task_dict)


def test_task_config_rejects_invalid_workload_range(base_task_dict):
    base_task_dict["initial_workloads"] = [0.1, 1.2, 0.2]
    with pytest.raises(ValidationError):
        TaskConfig.model_validate(base_task_dict)


def test_task_config_accepts_extra_keys(base_task_dict):
    base_task_dict["custom_field"] = "ok"
    cfg = TaskConfig.model_validate(base_task_dict)
    # model_extra should handle custom fields because extra="allow" is set in TaskConfig
    assert cfg.model_extra["custom_field"] == "ok"
