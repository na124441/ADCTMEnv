#!/usr/bin/env python3

import json
import os
import time
import threading
from typing import Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv
from openai import OpenAI

from grader.evaluator import evaluate_trajectory
from inference.prompt import build_prompt as detailed_build_prompt
from models import Action, Observation, ResetResponse, Reward, StepResponse
from tasks.task_config import TaskConfig

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TEMPERATURE = 0.0
DEFAULT_TASKS = ("easy", "medium", "hard")
MAX_STEPS = 40

def call_llm_with_timeout(prompt: str, timeout_sec=3) -> str:
    result = [""]

    def target():
        try:
            result[0] = call_llm(prompt)
        except Exception:
            result[0] = ""

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_sec)

    return result[0] if not thread.is_alive() else ""

def parse_action(content: str, num_zones: int) -> List[float]:
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in model response")

        data = json.loads(content[start:end])
        action = Action.model_validate({"cooling": data.get("cooling", [])})
        if len(action.cooling) != num_zones:
            raise ValueError(f"Expected {num_zones} cooling values")
        return action.cooling
    except Exception:
        return [0.3] * num_zones


def predict_action(state: Dict[str, object]) -> List[float]:
    """
    Deterministic fallback policy used by lightweight contract tests.
    """

    raw_temperatures = state.get("temperatures", [])
    raw_workloads = state.get("workloads", [])
    target_temperature = float(state.get("target_temperature", 70.0))

    temperatures = [float(value) for value in raw_temperatures] if isinstance(raw_temperatures, list) else []
    workloads = [float(value) for value in raw_workloads] if isinstance(raw_workloads, list) else []

    if not temperatures:
        num_zones = int(state.get("num_zones", 0) or 0)
        return [0.3] * max(num_zones, 0)

    if len(workloads) < len(temperatures):
        workloads = workloads + [0.0] * (len(temperatures) - len(workloads))

    cooling = []
    for temp, workload in zip(temperatures, workloads):
        temp_term = max(0.0, (temp - target_temperature) / max(target_temperature, 1.0))
        workload_term = 0.35 * max(0.0, min(1.0, workload))
        cooling.append(max(0.0, min(1.0, 0.25 + temp_term + workload_term)))

    return cooling


def call_llm(prompt: str) -> str:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is required")
        
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,  # type: ignore[arg-type]
                temperature=TEMPERATURE,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            if attempt == 0:
                raise exc
            time.sleep(2**attempt)
    return ""


def log_start(task_name: str, env: str, model: str) -> None:
    print(f"[START] task={task_name} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def _parse_reset_response(payload: Dict[str, object]) -> ResetResponse:
    if "observation" in payload:
        return ResetResponse.model_validate(payload)
    return ResetResponse(observation=Observation.model_validate(payload))


def _parse_step_response(payload: Dict[str, object], fallback_observation: Observation) -> StepResponse:
    if "observation" in payload:
        normalized = dict(payload)
        reward_payload = normalized.get("reward")
        if isinstance(reward_payload, (int, float)):
            normalized["reward"] = {"value": float(reward_payload)}
        elif reward_payload is None:
            normalized["reward"] = {"value": 0.0}
        return StepResponse.model_validate(normalized)

    reward_payload = payload.get("reward", 0.0)
    reward = Reward(value=float(reward_payload)) if isinstance(reward_payload, (int, float)) else Reward(value=0.0)
    return StepResponse(
        observation=fallback_observation,
        reward=reward,
        done=bool(payload.get("done", False)),
        info={"step": fallback_observation.time_step},
    )


def run_task(task_name: str) -> float:
    log_start(task_name=task_name, env="ADCTM", model=MODEL_NAME)

    steps = 0
    score = 0.0
    rewards: List[float] = []
    success = False

    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, timeout=5)
        resp.raise_for_status()

        reset_result = _parse_reset_response(resp.json())
        current_observation = reset_result.observation

        task_config_path = os.path.join(os.path.dirname(__file__), "tasks", f"{task_name}.json")
        if not os.path.exists(task_config_path):
            task_config_path = os.path.join(os.getcwd(), "tasks", f"{task_name}.json")

        with open(task_config_path, encoding="utf-8") as handle:
            config = TaskConfig(**json.load(handle))

        all_obs = [current_observation.model_dump()]
        all_actions = []
        done = reset_result.done

        while not done and steps < MAX_STEPS:
            steps += 1
            error_msg = None

            try:
                if steps > 2:
                    action_vals = predict_action(
                        {
                            "temperatures": current_observation.temperatures,
                            "workloads": current_observation.workloads,
                            "target_temperature": config.target_temperature,
                        }
                    )
                else:
                    prompt = detailed_build_prompt(current_observation, config.model_dump())
                    raw = call_llm_with_timeout(prompt, timeout_sec=3)
                    action_vals = parse_action(raw, len(current_observation.temperatures)) if raw else predict_action(
                        {
                            "temperatures": current_observation.temperatures,
                            "workloads": current_observation.workloads,
                            "target_temperature": config.target_temperature,
                        }
                    )
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {str(exc).replace('\n', ' ')}"
                action_vals = predict_action(
                    {
                        "temperatures": current_observation.temperatures,
                        "workloads": current_observation.workloads,
                        "target_temperature": config.target_temperature,
                    }
                )

            action_payload = Action(cooling=action_vals).model_dump()
            all_actions.append(action_payload)

            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json=action_payload,
                    timeout=5,
                )
                step_resp.raise_for_status()
                step_result = _parse_step_response(step_resp.json(), current_observation)
            except Exception as exc:
                if not error_msg:
                    error_msg = f"{type(exc).__name__}: {str(exc).replace('\n', ' ')}"
                step_result = StepResponse(
                    observation=current_observation,
                    reward=Reward(value=0.0),
                    done=True,
                    info={"step": steps},
                )

            current_observation = step_result.observation
            all_obs.append(current_observation.model_dump())

            reward = step_result.reward.value
            done = step_result.done
            rewards.append(reward)

            action_str = str(action_vals)
            log_step(step=steps, action=action_str, reward=reward, done=done, error=error_msg)

        try:
            result = evaluate_trajectory(all_obs, all_actions, config, return_details=True)
            score = min(max(float(result.get("score", 0.0)), 0.0), 1.0)
        except Exception:
            score = 0.0

        success = done

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc).replace('\n', ' ')}"
    finally:
        log_end(success=success, steps=steps, rewards=rewards)

    return score


def execute_simulation(task_names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for task_name in task_names or DEFAULT_TASKS:
        try:
            results[task_name] = run_task(task_name)
        except Exception:
            results[task_name] = 0.0
    return results


def main() -> None:
    execute_simulation()


if __name__ == "__main__":
    main()
