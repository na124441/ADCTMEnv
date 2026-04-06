#!/usr/bin/env python3
"""
Complementary OpenEnv Baseline Inference Script.
Runs a trajectory roll-out for the three benchmark tasks (easy, medium, hard)
using an LLM-driven policy and prints **rich-styled** terminal output.

All pretty-printing helpers are defined inside this file – the script no longer
imports the third-party ``printer`` package, so the “cannot import name
‘print_banner’” error disappears.
"""

# ----------------------------------------------------------------------
# 1️⃣  Imports – standard library / third party
# ----------------------------------------------------------------------
import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

from openai import OpenAI
from dotenv import load_dotenv
from models import Action, Observation, ResetResponse, Reward, StepResponse

# ----------------------------------------------------------------------
# 2️⃣  Rich-based printing utilities (copied from printer.py)
# ----------------------------------------------------------------------
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.padding import Padding
from rich import box

# Global Rich console
_console = Console()

# ── Semantic colour aliases ──────────────────────────────────────────
_GREEN  = "green"
_YELLOW = "yellow"
_RED    = "red"
_BLUE   = "dodger_blue1"
_CYAN   = "cyan"
_MUTED  = "grey62"
_WHITE  = "bright_white"

# Task palette – colour + symbol used for easy/medium/hard headings
TASK_PALETTE = {
    "easy":   (_GREEN,  "●", "easy"),
    "medium": (_YELLOW, "●", "medium"),
    "hard":   (_RED,    "●", "hard"),
}

# ── Helper functions used by the printing helpers -------------------------
def _temp_spark(temp: float, critical: float = 55.0) -> Text:
    """Create an inline temperature bar (green → yellow → red)."""
    ratio = min(temp / critical, 1.0)
    filled = int(ratio * 12)
    bar = "█" * filled + "░" * (12 - filled)
    colour = _RED if temp >= critical * 0.9 else (_YELLOW if temp >= critical * 0.72 else _GREEN)
    t = Text()
    t.append(bar, style=colour)
    t.append(f"  {temp:.1f}°C", style=_WHITE)
    return t


def _grade(score: float) -> str:
    """Letter grade used in the final summary."""
    if score >= 0.85: return "S"
    if score >= 0.70: return "A"
    if score >= 0.55: return "B"
    if score >= 0.40: return "C"
    return "F"


# ── Public printing helpers (mirroring the original printer module) ───────
def print_banner(model_name: str, env_url: str) -> None:
    """Shows the ADCTM startup banner with model/environment metadata."""
    logo = Text(justify="center")
    logo.append("  ADCTM\n", style=f"bold {_CYAN}")
    logo.append("  Autonomous Data Centre Thermal Management\n", style=f"bold {_WHITE}")
    logo.append("  OpenEnv Benchmark  ·  LLM / RL Agent Evaluation\n", style=_MUTED)

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style=_MUTED, width=14)
    meta.add_column(style=f"bold {_CYAN}")
    meta.add_row("  model",   model_name)
    meta.add_row("  env url", env_url)
    meta.add_row("  tasks",   "easy  ·  medium  ·  hard")

    # For Rich, grouping two renderables (a Text and a Table) inside a Panel
    # can be done easily via rich.console.Group.
    from rich.console import Group
    
    _console.print()
    _console.print(
        Panel(
            Group(logo, meta),
            border_style=_CYAN,
            box=box.DOUBLE_EDGE,
            expand=False,
            padding=(0, 4),
        )
    )
    _console.print()


def print_task_start(task_name: str) -> None:
    """Visually distinct header shown before a task begins."""
    colour, dot, label = TASK_PALETTE.get(task_name, (_WHITE, "●", task_name))
    _console.print()
    _console.print(
        Rule(
            title=f"  [{colour}]{dot}  task — {label}[/]  ",
            style=colour,
            characters="─",
        )
    )
    _console.print()


def print_step(
    step: int,
    total_steps: int,
    actions: List[float],
    reward: float,
    obs: Dict,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """Renders a single simulation step as a compact Rich table."""
    temps = obs.get("temperatures", [])
    loads = obs.get("workloads", [])
    n_zones = len(temps)

    # ---- Progress bar ----
    pct = step / max(total_steps, 1)
    bar_width = 20
    filled_n = int(pct * bar_width)
    prog = "━" * filled_n + "╍" * (bar_width - filled_n)
    prog_col = _GREEN if done else _BLUE

    # ---- Reward colour ----
    r_colour = _GREEN if reward >= -0.1 else (_YELLOW if reward >= -0.5 else _RED)

    # ---- Header line ----
    header = Text()
    header.append(f"  step {step:>2}/{total_steps}  ", style=f"bold {_WHITE}")
    header.append(f"[{prog}]", style=prog_col)
    header.append("  reward ", style=_MUTED)
    header.append(f"{reward:+.3f}", style=f"bold {r_colour}")
    if done:
        header.append("   done", style=f"bold {_GREEN}")
    if error:
        header.append(f"   ! {error[:50]}", style=_RED)

    _console.print(header)

    # ---- Zone table ----
    grid = Table(
        box=None,
        show_header=(step == 1),
        header_style=_MUTED,
        padding=(0, 1),
        expand=False,
    )
    grid.add_column("zone", style=_MUTED, width=6)
    grid.add_column("temperature", width=22, no_wrap=True)
    grid.add_column("load", width=12, no_wrap=True)
    grid.add_column("cooling", width=14, no_wrap=True)

    for i in range(n_zones):
        t = temps[i]   if i < len(temps)   else 0.0
        w = loads[i]   if i < len(loads)   else 0.0
        c = actions[i] if i < len(actions) else 0.0

        load_bar = "▰" * int(w * 8) + "▱" * (8 - int(w * 8))
        cool_bar = "▰" * int(c * 8) + "▱" * (8 - int(c * 8))
        load_col = _RED if w > 0.8 else (_YELLOW if w > 0.5 else _GREEN)
        cool_col = _BLUE if c > 0.6 else (_YELLOW if c > 0.3 else _MUTED)

        grid.add_row(
            f"z{i+1:02}",
            _temp_spark(t),
            Text(load_bar, style=load_col),
            Text(f"{cool_bar} {c:.2f}", style=cool_col),
        )

    _console.print(Padding(grid, (0, 4)))


def print_task_result(
    task_name: str,
    score: float,
    steps: int,
    rewards: List[float],
    details: Optional[Dict] = None,
) -> None:
    """Shows the end-of-task score card with metrics and a reward sparkline."""
    colour, dot, label = TASK_PALETTE.get(task_name, (_WHITE, "●", task_name))
    score_colour = _GREEN if score >= 0.7 else (_YELLOW if score >= 0.4 else _RED)
    grade = _grade(score)

    # ----- Score headline -----
    headline = Text()
    headline.append("\n  score   ", style=_MUTED)
    headline.append(f"{score:.4f}", style=f"bold {score_colour}")
    headline.append("   grade  ", style=_MUTED)
    headline.append(grade, style=f"bold {score_colour}")
    headline.append("   steps  ", style=_MUTED)
    headline.append(f"{steps}\n", style=_WHITE)

    # ----- Metric bars (if details provided) -----
    metric_rows = Table(box=None, show_header=False, padding=(0, 2))
    metric_rows.add_column(style=_MUTED, width=14)
    metric_rows.add_column(width=8, justify="right")
    metric_rows.add_column(width=18)

    metric_defs = [
        ("safety",     "safety",     _RED),
        ("precision",  "precision",  _BLUE),
        ("efficiency", "efficiency", _GREEN),
        ("smoothness", "smoothness", _YELLOW),
    ]
    if details:
        for display, key, col in metric_defs:
            val = details.get(key, 0.0)
            bar = "█" * int(val * 16) + "░" * (16 - int(val * 16))
            metric_rows.add_row(
                f"  {display}",
                Text(f"{val:.3f}", style=f"bold {col}"),
                Text(bar, style=col),
            )

    # ----- Reward sparkline -----
    spark = Text()
    spark.append("  rewards  ", style=_MUTED)
    if rewards:
        mn, mx = min(rewards), max(rewards)
        rng = mx - mn if mx != mn else 1.0
        levels = "  ▂▃▄▅▆▇█"
        for r in rewards[-36:]:
            idx = int((r - mn) / rng * 8)
            spark.append(levels[max(0, min(8, idx))], style=_BLUE)
        avg = sum(rewards) / len(rewards)
        spark.append(f"  avg {avg:+.3f}", style=_MUTED)

    body = Text.assemble(headline, "\n", spark, "\n")

    _console.print()
    _console.print(
        Panel(
            body,
            title=f"[{colour}]{dot}  {label} complete[/]",
            title_align="left",
            border_style=score_colour,
            box=box.HEAVY,
            expand=False,
            padding=(0, 2),
        )
    )
    if details:
        _console.print(Padding(metric_rows, (0, 2)))
    _console.print()


def print_final_summary(results: Dict[str, float]) -> None:
    """Renders a table summarising the scores for all tasks."""
    _console.print()
    _console.print(Rule("  benchmark results  ", style=_CYAN, characters="═"))
    _console.print()

    table = Table(
        box=box.DOUBLE_EDGE,
        border_style=_CYAN,
        header_style=f"bold {_CYAN}",
        show_header=True,
        expand=False,
        padding=(0, 2),
    )
    table.add_column("task", style=f"bold {_WHITE}", width=12)
    table.add_column("score", justify="right", width=10)
    table.add_column("grade", justify="center", width=8)
    table.add_column("bar", width=24)
    table.add_column("status", justify="center", width=10)

    total = 0.0
    for task_name, score in results.items():
        colour, dot, label = TASK_PALETTE.get(task_name, (_WHITE, "●", task_name))
        score_colour = _GREEN if score >= 0.7 else (_YELLOW if score >= 0.4 else _RED)
        grade = _grade(score)
        bar_fill = int(score * 22)

        bar = Text()
        bar.append("█" * bar_fill, style=score_colour)
        bar.append("░" * (22 - bar_fill), style=_MUTED)

        status = (
            Text("pass", style=f"bold {_GREEN}")
            if score >= 0.4
            else Text("fail", style=f"bold {_RED}")
        )

        table.add_row(
            Text(f"{dot}  {label}", style=colour),
            Text(f"{score:.4f}", style=f"bold {score_colour}"),
            Text(grade, style=f"bold {score_colour}"),
            bar,
            status,
        )
        total += score

    avg = total / max(len(results), 1)
    avg_colour = _GREEN if avg >= 0.7 else (_YELLOW if avg >= 0.4 else _RED)

    table.add_section()
    table.add_row(
        Text("  average", style=f"bold {_WHITE}"),
        Text(f"{avg:.4f}", style=f"bold {avg_colour}"),
        "", "", "",
    )

    _console.print(Padding(table, (0, 2)))
    _console.print()
    _console.print(
        Padding(
            Text.assemble(
                Text("overall  ", style=_MUTED),
                Text(f"{avg:.4f}", style=f"bold {avg_colour}"),
                Text("  / 1.0000", style=_MUTED),
            ),
            (0, 4),
        )
    )
    _console.print()
    _console.print(Rule(style=_MUTED))
    _console.print()


def print_error(message: str) -> None:
    """Render an error panel in red."""
    _console.print(
        Panel(
            Text(f"  {message}", style=_RED),
            title="[bold red]error[/]",
            border_style=_RED,
            box=box.HEAVY,
            expand=False,
        )
    )


def print_info(message: str) -> None:
    """Simple muted line (used for non-critical information)."""
    _console.print(Text(f"  {message}", style=_MUTED))


def print_connecting(env_url: str) -> None:
    """Small line shown while the script contacts the backend."""
    _console.print(Text(f"\n  connecting to {env_url} ...\n", style=_BLUE))


# ----------------------------------------------------------------------
# 3️⃣  Configuration & constants
# ----------------------------------------------------------------------
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME", "none")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "none")

# OpenEnv backend (normally a Hugging-Face Space running locally)
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Zero randomness for reproducibility (baseline)
TEMPERATURE = 0.0


# ----------------------------------------------------------------------
# 4️⃣  Helper functions: prompt building, action parsing, LLM call
# ----------------------------------------------------------------------
def _build_prompt(obs: Dict[str, Any]) -> str:
    """Return a concise instruction prompt for the LLM."""
    num_zones = len(obs["temperatures"])
    return f"""
You are the controller for a multi-zone data-centre cooling system. 
Goal: keep temperatures within safe ranges while minimising energy use.

Current system state:
- Ambient temperature: {obs['ambient_temp']}°C
- Zone temperatures: {obs['temperatures']}
- Zone workloads (0-1 normalised): {obs['workloads']}
- Previous cooling (0-1 normalised): {obs['cooling']}

Output **exactly one** valid JSON object with the cooling levels for each of the {num_zones} zones.
Format: {{"cooling": [v1, v2, …, v{num_zones}]}}
All values must be between 0.0 and 1.0.  
Only the JSON object – no extra text.
"""


def parse_action(content: str, num_zones: int) -> List[float]:
    """Extract a list of floats from the LLM’s JSON reply."""
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])
        action = Action.model_validate({"cooling": data.get("cooling", [])})
        if len(action.cooling) != num_zones:
            raise ValueError("Unexpected number of cooling values")
        return action.cooling
    except Exception:
        # Safe fallback – 30 % cooling for every zone.
        return [0.3] * num_zones


def _parse_reset_response(payload: Dict[str, Any]) -> ResetResponse:
    if "observation" in payload:
        return ResetResponse.model_validate(payload)
    return ResetResponse(observation=Observation.model_validate(payload))


def _parse_step_response(payload: Dict[str, Any], fallback_observation: Observation) -> StepResponse:
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


def call_llm(
    prompt: str,
    model: str,
    api_base: str,
    api_key: str,
    retries: int = 3,
) -> str:
    """
    Send *prompt* to the language model and return its raw reply.
    Two execution paths:
      • Ollama-compatible models (detected by model name)
      • OpenAI-compatible models (default)
    Retries use exponential back-off.
    """
    use_ollama = "gpt-oss" in model.lower() or "ollama" in model.lower()

    if use_ollama:
        # Warn the user if they pointed to an OpenAI endpoint.
        if "openai.com" in api_base.lower() and "ollama" not in api_base.lower():
            print_error(
                f"Model '{model}' looks Ollama-based but API_BASE_URL points at "
                f"OpenAI ('{api_base}').  Ollama models normally run on a local "
                "server (e.g. http://localhost:11434)."
            )
        import ollama  # type: ignore

        client = (
            ollama.Client(host=api_base)
            if api_base and ("localhost" not in api_base or "11434" in api_base)
            else ollama
        )

        for attempt in range(retries):
            try:
                resp = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a thermal control engineer. Always output JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.0},
                )
                return resp["message"]["content"].strip()
            except Exception as exc:
                if attempt == retries - 1:
                    raise exc
                time.sleep(2 ** attempt)

    else:
        # OpenAI path – works with any model hosted on an OpenAI-compatible endpoint.
        # We need to treat tokens differently to avoid validation errors with OpenAI python lib
        api_key_for_client = api_key if api_key and api_key != "none" else "none"
        # ensure no trailing slashes or missing /v1
        if api_base.endswith("/"):
            api_base = api_base[:-1]
        if "localhost" in api_base and "11434" in api_base and not api_base.endswith("/v1"):
            api_base = api_base + "/v1"

        client = OpenAI(base_url=api_base, api_key=api_key_for_client)
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a thermal control engineer. Always output JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                if attempt == retries - 1:
                    raise exc
                time.sleep(2 ** attempt)

    return ""


# ----------------------------------------------------------------------
# 5️⃣  Core task runner – uses the Rich helpers defined above
# ----------------------------------------------------------------------
from tasks.task_config import TaskConfig
from grader.evaluator import evaluate_trajectory

def run_task(task_name: str) -> float:
    """Run a whole episode for *task_name* and return the graded score."""
    # --------------------------------------------------------------
    # Header for this task
    # --------------------------------------------------------------
    print_task_start(task_name)

    # --------------------------------------------------------------
    # Reset the environment and fetch the initial observation
    # --------------------------------------------------------------
    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_name": task_name},
            timeout=30,
        )
        if resp.status_code != 200:
            # Some services expose a GET-style reset
            resp = requests.post(
                f"{ENV_URL}/reset",
                params={"task_name": task_name},
                timeout=30,
            )
        resp.raise_for_status()
        reset_result = _parse_reset_response(resp.json())
        current_observation = reset_result.observation
        obs = current_observation.model_dump()
    except Exception as exc:
        print_error(f"Failed to reset environment for task '{task_name}': {exc}")
        raise

    # --------------------------------------------------------------
    # Load the per-task configuration (contains max_steps, etc.)
    # --------------------------------------------------------------
    task_cfg_path = Path(__file__).parent / "tasks" / f"{task_name}.json"
    if not task_cfg_path.is_file():
        # Fallback – script may have been launched from a different cwd
        task_cfg_path = Path.cwd() / "tasks" / f"{task_name}.json"
        if not task_cfg_path.is_file():
            raise FileNotFoundError(f"Task configuration not found: {task_cfg_path}")

    with open(task_cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    config = TaskConfig(**cfg_dict)

    # Determine max steps for progress-bar rendering (default 0 → hidden bar)
    max_steps = getattr(config, "max_steps", cfg_dict.get("max_steps", 0))
    try:
        max_steps = int(max_steps)
    except Exception:
        max_steps = 0

    # --------------------------------------------------------------
    # Book-keeping containers
    # --------------------------------------------------------------
    all_observations: List[Dict] = [obs]
    all_actions: List[Dict] = []
    done = reset_result.done
    step_idx = 0
    rewards: List[float] = []
    step_error: Optional[str] = None

    # --------------------------------------------------------------
    # Main rollout loop
    # --------------------------------------------------------------
    while not done:
        step_idx += 1
        step_error = None

        # ---- Build prompt & query the LLM ----
        prompt = _build_prompt(obs)

        try:
            raw_reply = call_llm(prompt, MODEL_NAME, API_BASE_URL, HF_TOKEN)
            if not raw_reply:
                action_vals = [0.3] * len(obs.get("temperatures", [0.0]))
            else:
                action_vals = parse_action(raw_reply, len(obs.get("temperatures", [0.0])))
        except Exception as exc:
            step_error = f"{type(exc).__name__}: {exc}"
            action_vals = [0.3] * len(obs.get("temperatures", [0.0]))

        # Record the action for later grading.
        action_payload = Action(cooling=action_vals).model_dump()
        all_actions.append(action_payload)

        # ---- Step the environment ----
        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json=action_payload,
                timeout=30,
            )
            step_resp.raise_for_status()
            step_result = _parse_step_response(step_resp.json(), current_observation)
        except requests.RequestException as exc:
            step_error = f"Env step error: {exc}"
            step_result = StepResponse(
                observation=current_observation,
                reward=Reward(value=0.0),
                done=True,
                info={"step": step_idx},
            )

        # Update state for the next iteration
        current_observation = step_result.observation
        obs = current_observation.model_dump()
        all_observations.append(obs)

        reward_val = step_result.reward.value
        rewards.append(reward_val)
        done = step_result.done

        # ---- Rich-styled step output ----
        print_step(
            step=step_idx,
            total_steps=max_steps,
            actions=action_vals,
            reward=reward_val,
            obs=obs,
            done=done,
            error=step_error,
        )

    # --------------------------------------------------------------
    # Grade the completed trajectory
    # --------------------------------------------------------------
    final_score_details = evaluate_trajectory(
        all_observations,
        all_actions,
        config,
        return_details=True,
    )
    graded_score = float(final_score_details.get("score", 0.0))

    # --------------------------------------------------------------
    # Display the per-task result panel
    # --------------------------------------------------------------
    print_task_result(
        task_name=task_name,
        score=graded_score,
        steps=step_idx,
        rewards=rewards,
        details=final_score_details,
    )

    return graded_score


# ----------------------------------------------------------------------
# 6️⃣  Entry point – run all three tasks and print the final table
# ----------------------------------------------------------------------
def main() -> None:
    """Execute easy / medium / hard tasks and show a coloured summary."""
    # ----- Banner & connection notice -----
    print_banner(MODEL_NAME, ENV_URL)
    print_connecting(ENV_URL)

    # ----- Credential sanity check -----
    if not HF_TOKEN and "gpt-oss" not in MODEL_NAME.lower():
        print_error(
            "Missing credentials (API_BASE_URL / HF_TOKEN). Cannot contact LLM – aborting run."
        )
        return

    # ----- Run the three benchmark tasks -----
    task_names = ["easy", "medium", "hard"]
    results: Dict[str, float] = {}

    for tn in task_names:
        try:
            results[tn] = run_task(tn)
        except Exception as exc:
            print_error(f"Execution error on task '{tn}': {exc}")
            results[tn] = 0.0

    # ----- Final aggregated summary table -----
    print_final_summary(results)


if __name__ == "__main__":
    if sys.platform == "win32":
        # Force utf-8 encoding for stdout on windows
        sys.stdout.reconfigure(encoding='utf-8')
    main()
