"""
FastAPI environment server for ADCTM.

The handlers here provide the HTTP contract used by the local runners and by
submission validation: `/reset`, `/step`, and `/state`.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from core.models import InfoDict, ResetPayload, ResetResponse, StateResponse, StepResponse
from core.simulator import SimulationSession
from grader.evaluator import evaluate_trajectory
from tasks.task_config import TaskConfig


class PrettyJSONResponse(JSONResponse):
    """
    Pretty-print JSON responses to make local debugging easier.
    """

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(", ", ": "),
        ).encode("utf-8")


app = FastAPI(
    title="Multi-Zone Cooling OpenEnv",
    version="0.1.0",
    default_response_class=PrettyJSONResponse,
)

CURRENT_SESSION: Optional[SimulationSession] = None
env_lock = threading.Lock()


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Welcome to the OpenEnv Multi-Zone Cooling API!"}


def _ensure_initialized() -> SimulationSession:
    if CURRENT_SESSION is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return CURRENT_SESSION


def _extract_task_config(
    payload: Optional[ResetPayload],
    task_name: Optional[str],
) -> tuple[str, Optional[Dict[str, Any]]]:
    """
    Normalize `/reset` inputs.

    Supported call patterns:
    - `POST /reset` with `{}` for validator compatibility.
    - `POST /reset?task_name=easy`
    - `POST /reset` with `{"task_name": "easy"}` or `{"task": "easy"}`
    - `POST /reset` with a full task configuration payload.
    """

    raw_payload: Dict[str, Any] = {}
    if payload is not None:
        raw_payload.update(payload.model_dump(exclude_none=True))
        if payload.model_extra:
            raw_payload.update(payload.model_extra)

    selected_task = task_name or raw_payload.pop("task_name", None) or raw_payload.pop("task", None) or "easy"

    # These are allowed reset parameters but are not currently used by the session.
    raw_payload.pop("seed", None)
    raw_payload.pop("episode_id", None)

    config_keys = set(TaskConfig.model_fields.keys())
    config_payload = {key: value for key, value in raw_payload.items() if key in config_keys}

    return str(selected_task), config_payload or None


@app.post("/reset", response_model=ResetResponse)
def reset(
    payload: Optional[ResetPayload] = Body(default=None),
    task_name: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    global CURRENT_SESSION

    try:
        selected_task, config_payload = _extract_task_config(payload, task_name)
        session = (
            SimulationSession.from_dict(config_payload)
            if config_payload is not None
            else SimulationSession.from_task_name(selected_task)
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Default task config not found.")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    with env_lock:
        CURRENT_SESSION = session

    return ResetResponse(
        observation=session.observation,
        reward=None,
        done=session.done,
        info=InfoDict(step=session.step_counter),
    ).model_dump()


@app.post("/step", response_model=StepResponse)
def step(action_dict: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    with env_lock:
        session = _ensure_initialized()
        try:
            result = StepResponse.model_validate(session.step(action_dict))
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return result.model_dump()


@app.get("/state", response_model=StateResponse)
def get_full_state() -> Dict[str, Any]:
    with env_lock:
        session = _ensure_initialized()
        return StateResponse.model_validate(session.model_dump()).model_dump()


@app.post("/simulate")
def simulate(task_name: str = "easy", cooling_level: float = 0.4) -> Dict[str, Any]:
    try:
        reset_payload = reset(task_name=task_name)
        initial_observation = reset_payload["observation"]
        session = _ensure_initialized()
        num_zones = session.config.num_zones
    except HTTPException as exc:
        raise HTTPException(status_code=400, detail=f"Error resetting environment: {exc.detail}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error loading task configuration: {exc}")

    observations = [initial_observation]
    actions = []
    total_reward = 0.0
    done = False

    while not done:
        action = {"cooling": [cooling_level] * num_zones}

        try:
            step_result = step(action)
        except HTTPException as exc:
            raise HTTPException(status_code=400, detail=f"Error stepping environment: {exc.detail}")

        observations.append(step_result["observation"])
        actions.append(action)
        total_reward += float(step_result["reward"]["value"])
        done = bool(step_result["done"])

    score = evaluate_trajectory(observations, actions, session.config)

    return {
        "task": task_name,
        "steps": len(actions),
        "total_reward": total_reward,
        "score": score,
        "status": "completed",
    }


if __name__ == "__main__":
    import uvicorn

    # Keep parity with `server.app.main()` and Hugging Face Spaces defaults.
    port = int(__import__("os").getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
