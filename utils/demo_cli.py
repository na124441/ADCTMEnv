#!/usr/bin/env python3
"""
Demo CLI App (Not used in evaluation)
=====================================
Top-level inference pipeline for ADCTM.
This script orchestrates the entire simulation: it initializes tasks,
interacts with the LLM via OpenAI API to play the role of the cooling
system controller, evaluates the trajectory, and renders the rich terminal UI dashboard.
"""

import argparse
import json
import os
import queue
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

# numpy is used for thermal gradient analysis and plotting AUC regions
import numpy as np
# HTTP client for interacting with the OpenEnv server over REST
import requests
# Dotenv is used to load environment variables (API keys, endpoints) from a .env file
from dotenv import load_dotenv

# Rich library components for building the complex, interactive terminal UI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.theme import Theme
from rich.table import Table
from rich.columns import Columns

# Importing internal analysis tools
from analysis.policy_analyzer import evaluate_action_quality
from analysis.trend_predictor import predict_thermal_future
# Importing internal core models and states
from core.dashboard_state import DashboardState
from core.models import Action, Observation
from core.paths import TASKS_DIR
from core.simulator import SimulationSession
# Importing grading and task logic
from grader.evaluator import evaluate_trajectory
from inference.parser import parse_llm_response
from inference.prompt import build_prompt

from tasks.task_config import TaskConfig
# UI rendering function
from ui.dashboard import make_dashboard

# Attempt to load matplotlib for post-simulation visual plotting of thermal profiles.
# If it fails, fallback gracefully without breaking the system.
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Load environment variables into the os.environ dictionary
load_dotenv()


def _get_theme() -> Theme:
    """
    Generates a custom stylized theme mapping for the Rich terminal console.
    Returns:
        Theme: the color palette configuration.
    """
    return Theme(
        {
            "cyan": "#8be9fd",
            "magenta": "#bd93f9",
            "red": "#ff5555",
            "yellow": "#ffb86c",
            "green": "#50fa7b",
            "dim": "#6272a4",
        }
    )

# Globally initialize the console using the above theme map for UI outputs
console = Console(theme=_get_theme())

# ----------------------------------------------------------------------
# CLI Argument parsing
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse commandâ€‘line arguments for the demo CLI."""
    parser = argparse.ArgumentParser(
        description="Demo CLI for ADCTM (supports OpenAI, HuggingFace, and Ollama backâ€‘ends)"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        help="Model name to use (e.g., gpt-oss:120b-cloud)",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
        help="Base URL for the LLM API",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY"),
        help="API authentication token",
    )
    parser.add_argument(
        "--env-url",
        default=None,
        help="OpenEnv server URL (e.g., http://localhost:7860)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run simulation locally (no HTTP calls)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick headâ€‘less test on the 'easy' task and exit",
    )
    return parser.parse_args()

# ----------------------------------------------------------------------
# Test mode helper
# ----------------------------------------------------------------------
def run_demo_test(args: argparse.Namespace):
    """Execute a single headâ€‘less task (easy) and print a concise report."""
    # Load the easy task configuration
    task_cfg = _load_task_config("easy.json")

    result = execute_simulation(
        task_cfg=task_cfg,
        task_name="easy",
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        use_http=bool(args.env_url),
        env_url=args.env_url,
        headless=True,
    )

    print("\n--- Demo CLI Test Mode (headâ€‘less) ---")
    print(f"Task          : {result['task_name']}")
    print(f"Model         : {result['model']}")
    print(f"Steps executed: {result['steps']}")
    print(f"Total reward  : {result['total_reward']:.4f}")
    print(f"Trajectory score: {result['trajectory_score']:.4f}")
    if result.get("parse_errors"):
        print(f"Parse errors  : {len(result['parse_errors'])}")
    else:
        print("Parse errors  : None")
    print("--- End of Test ---\n")

def call_llm(prompt: str, model: str, api_base: str, api_key: str, retries: int = 3) -> str:
    """
    Sends a constructed prompt to an LLM via either the Ollama library or the OpenAI client.
    Hybrid approach: Detects 'gpt-oss' or 'ollama' models to use the native Ollama library.
    
    Args:
        prompt (str): The state representation and instructions.
        model (str): Name of the model to use (e.g., gpt-4o-mini).
        api_base (str): Base URL of the API.
        api_key (str): Authentication token.
        retries (int): Number of connection attempts before failing.
        
    Returns:
        str: Raw string output from the LLM.
    """
    # Detect if we should use the Ollama library backend
    use_ollama = "gpt-oss" in model.lower() or "ollama" in model.lower()

    if use_ollama:
        import ollama
        # The Ollama library handles local/cloud connection automatically.
        # It reads from the local environment/config for cloud-offloading.
        client = ollama.Client(host=api_base) if api_base and "localhost" not in api_base else ollama
        
        for attempt in range(retries):
            try:
                response = client.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': 'You are a cooling system controller. Output ONLY a valid JSON object.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    options={'temperature': 0.0}
                )
                return response['message']['content'].strip()
            except Exception as exc:
                if attempt == retries - 1:
                    raise exc
                time.sleep(2**attempt)
    else:
        from openai import OpenAI
        # Standard OpenAI client for GPT models and Hugging Face Router
        client = OpenAI(base_url=api_base, api_key=api_key)
        
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a cooling system controller. Output ONLY a valid JSON object.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0, # Zero temperature ensures highly deterministic, reliable logic
                )
                # Extract and return the stripped text chunk from the response
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                # If we hit the maximum retries, raise the error upwards
                if attempt == retries - 1:
                    raise exc
                # Wait with exponential backoff before the next attempt
                time.sleep(2**attempt) 
    return ""
  # Fallback return (theoretically unreachable due to loop layout)


def _fallback_action(num_zones: int) -> Action:
    """
    Creates a basic, safe fallback action if parsing fails.
    Defaults to 30% cooling power across all zones to prevent overheating during unexpected errors.
    """
    return Action(cooling=[0.3] * num_zones)


def _load_task_config(task_name: str) -> Dict[str, Any]:
    """
    Helper function to load the raw JSON task parameters given a task identifier.
    Reads from the common TASKS_DIR.
    """
    task_file = TASKS_DIR / f"{task_name.replace('.json', '')}.json"
    with task_file.open(encoding="utf-8") as handle:
        return json.load(handle)


def execute_simulation(
    task_cfg: Dict[str, Any],
    task_name: str,
    model: str,
    api_key: str,
    api_base: str,
    use_http: bool = False,
    env_url: Optional[str] = None,
    headless: bool = False,
) -> Dict[str, Any]:
    """
    Core simulation loop for offline inference or HTTP inference.
    Does not use background threading or live UI, effectively operating as a headless trajectory evaluator.
    """
    # Parse the raw dictionary parameters into a structured Pydantic configuration object
    cfg = TaskConfig.model_validate(task_cfg)
    
    # Initialize trackers for analytics
    observations = []
    actions = []
    parse_errors = []
    total_reward = 0.0
    done = False

    # Setting up the simulation session either locally or via REST HTTP integration
    if use_http:
        if not env_url:
            raise ValueError("env_url is required when use_http=True")
            
        # Call the initial reset endpoint
        response = requests.post(f"{env_url.rstrip('/')}/reset", json=cfg.model_dump(), timeout=30)
        response.raise_for_status()
        obs = Observation.model_validate(response.json())
    else:
        # Create a local simulation object instance
        session = SimulationSession(cfg)
        obs = session.observation

    # Log the first observation before starting steps for historical tracking
    observations.append(obs.model_dump())

    # Main inference loop restricted by configured maximum steps
    while obs.time_step < cfg.max_steps:
        # Prompt generation based on the environment observation state
        prompt = build_prompt(obs, cfg.model_dump())
        
        # Consult LLM for next action
        raw_output = call_llm(prompt, model, api_base, api_key)

        try:
            # Attempt to parse into an Action structured dictionary
            action = parse_llm_response(raw_output, cfg.num_zones)
        except ValueError as exc:
            # Upon parse failure, track the error and issue a fallback
            parse_errors.append({"step": obs.time_step, "error": str(exc), "raw_output": raw_output})
            action = _fallback_action(cfg.num_zones)

        # Apply action to environment
        if use_http:
            response = requests.post(
                f"{env_url.rstrip('/')}/step",
                json=action.model_dump(),
                timeout=30,
            )
            response.raise_for_status()
            step_data = response.json()
        else:
            step_data = session.step(action.model_dump())

        # Validate step updates into objects and increment counters
        obs = Observation.model_validate(step_data["observation"])
        observations.append(obs.model_dump())
        actions.append(action.model_dump())
        total_reward += step_data["reward"]["value"]
        done = step_data["done"]

        # Halt environment if termination happens organically
        if done:
            break

    # Return structured simulation metrics report
    return {
        "task_name": task_name,
        "model": model,
        "steps": len(actions),
        "done": done,
        "total_reward": total_reward,
        "trajectory_score": evaluate_trajectory(observations, actions, cfg), # Calculate full grade
        "observations": observations,
        "actions": actions,
        "parse_errors": parse_errors,
    }


def simulation_worker(
    task_cfg: Dict[str, Any],
    task_name: str,
    model: str,
    api_base: str,
    api_key: str,
    q: queue.Queue,
    results: Dict[str, Any],
) -> None:
    """
    Background worker process. This runs in a separate thread, executing
    the simulation step by step. It continuously feeds the current dashboard state
    into the Thread-Safe Queue so the UI thread can flawlessly render frames without blocking.
    """
    # Initialize basic physics simulation mapping
    cfg = TaskConfig.model_validate(task_cfg)
    session = SimulationSession(cfg)

    # UI constants and basic circular queue setup for displaying sliding logs
    u_alpha, u_beta, u_gamma = 2.0, 1.0, 0.5
    messages = deque(maxlen=20)
    messages.append("System online")

    # State variables initialization for continuous dashboard tracking
    obs = session.observation
    total_reward = 0.0
    step_reward = 0.0
    action: Optional[Action] = None
    temp_hist = [obs.temperatures]
    action_hist = []
    
    current_prompt = ""
    current_raw = ""
    current_parsed = None
    current_error = None

    def get_latest_state(log_msg: Optional[str] = None) -> DashboardState:
        """
        Inner helper function building a snapshot of the current local state variables,
        packaging it into a structured DashboardState model tailored for UI consumption.
        """
        # Append debug message string to console trace log
        if log_msg:
            messages.append(log_msg)

        try:
            # Extrapolate calculations: Delta offsets, Reward formulas, Output validation
            max_t = max(obs.temperatures)
            prev_temps = temp_hist[-2] if len(temp_hist) > 1 else obs.temperatures
            deltas = [t - p for t, p in zip(obs.temperatures, prev_temps)]
            cooling = action.cooling if action else [0.0] * cfg.num_zones
            
            # Component scoring logic for metrics visualization
            o_term = -u_alpha * (max(0.0, max_t - cfg.safe_temperature) ** 2)
            e_term = -u_beta * sum(value**2 for value in cooling)
            s_term = -u_gamma * abs(max_t - max(prev_temps))

            # External analytical models prediction 
            forecasts = predict_thermal_future(obs.temperatures, temp_hist, cfg.safe_temperature)
            quality = evaluate_action_quality(obs.temperatures, cooling, deltas, cfg.safe_temperature)

            # Analyze recent thermal trend patterns using short horizon of 10 steps
            recent_max_temps = [max(values) for values in temp_hist[-10:]]
            status = "INITIALIZING"
            if len(recent_max_temps) >= 5:
                trend = recent_max_temps[-1] - recent_max_temps[0]
                # Label status depending on absolute delta over a time-frame window
                if trend > 1.0:
                    status = "DIVERGING"
                elif abs(trend) < 0.2:
                    status = "STABLE"
                else:
                    status = "TRENDING"

            # Profile system action variance properties to evaluate agent traits
            recent_actions = [values for values in action_hist[-10:] if values]
            avg_a = 0.0
            var_a = 0.0
            if recent_actions:
                # Flatten complex lists for statistical variance
                flattened = [value for values in recent_actions for value in values]
                avg_a = sum(flattened) / len(flattened)
                var_a = sum((value - avg_a) ** 2 for value in flattened) / len(flattened)

            # Construct and return the full state dict for UI transmission
            return DashboardState(
                step=obs.time_step,
                temperatures=obs.temperatures,
                cooling=cooling,
                reward=step_reward,
                total_reward=total_reward,
                done=session.done,
                alpha=u_alpha,
                beta=u_beta,
                gamma=u_gamma,
                safe_temp=cfg.safe_temperature,
                num_zones=cfg.num_zones,
                target_temp=cfg.target_temperature,
                max_steps=cfg.max_steps,
                overshoot_term=o_term,
                energy_term=e_term,
                smoothness_term=s_term,
                deltas=deltas,
                forecasts=forecasts,
                action_quality=quality,
                trajectory_status=status,
                policy_signature={"avg": avg_a, "var": var_a},
                temp_history=list(temp_hist),
                action_history=list(action_hist),
                prompt=current_prompt,
                llm_raw_output=current_raw,
                parsed_action=current_parsed,
                parse_error=current_error,
                task_name=task_name,
                model=model,
                log=deque(messages),
            )
        except Exception as exc:
            # In the event capturing system fails mid-render loop, provide basic error banner
            return DashboardState(
                step=0,
                temperatures=[0.0],
                cooling=[0.0],
                reward=0.0,
                total_reward=0.0,
                done=True,
                error_banner=str(exc),
            )

    try:
        # Feed the first static state configuration to the renderer
        q.put(get_latest_state())
        
        while not session.done:
            # Provide initial simulation phase variables, building standard prompt string
            current_prompt = build_prompt(obs, cfg.model_dump())
            q.put(get_latest_state("Analyzing thermal gradients..."))

            try:
                # LLM request over HTTP, waits synchronously blocking thread ONLY. 
                # UI continues running from existing main thread loop
                current_raw = call_llm(current_prompt, model, api_base, api_key)
                q.put(get_latest_state("Logic trace received"))
                
                try:
                    # Parse valid result
                    action = parse_llm_response(current_raw, cfg.num_zones)
                    current_parsed = action.model_dump()
                    current_error = None
                except ValueError as exc:
                    # Handle json or validation issue gracefully
                    current_error = str(exc)
                    action = _fallback_action(cfg.num_zones)
                    current_parsed = action.model_dump()
            except Exception as exc:
                # Terminal failure during inference layer
                messages.append(f"LLM error: {exc}")
                q.put(get_latest_state())
                break

            # Send parsed action matrix back into simulation session step
            step_data = session.step(action.model_dump())
            obs = Observation.model_validate(step_data["observation"])
            
            # Apply tracking logic to historical accumulators
            step_reward = step_data["reward"]["value"]
            total_reward += step_reward
            temp_hist.append(obs.temperatures)
            action_hist.append(action.cooling)

            # Inform UI frame is ready, adding slight tick time delay artificially for visuals sake
            q.put(get_latest_state("Execution cycle complete"))
            time.sleep(0.05)
            
    except Exception as exc:
        messages.append(f"Fatal error: {exc}")
        q.put(get_latest_state())

    q.put(get_latest_state("Episode finished"))
    
    # Store complete session outcome variables upon background task closure
    results.update(
        {
            "score": total_reward,
            "history": temp_hist,
            "action_history": action_hist,
            "num_zones": cfg.num_zones,
            "safe_temp": cfg.safe_temperature,
            "target_temp": cfg.target_temperature,
            "task_name": task_name,
            "observations": [obs.model_dump() for obs in [Observation(temperatures=t, workloads=[0.0]*cfg.num_zones, cooling=[0.0]*cfg.num_zones, ambient_temp=cfg.ambient_temperature, time_step=i) for i, t in enumerate(temp_hist)]], # Approximation for metrics
            "actions": [{"cooling": a} for a in action_hist],
            "cfg": cfg,
        }
    )


def main(args: argparse.Namespace) -> None:
    """
    Primary execution loop. Setup arg parsing, start UI or headless execution 
    based on argument modes, and display final analytical charts when complete.
    """
    # Pull list of json task profiles iteratively from the TASKS directory
    task_files = sorted(TASKS_DIR.glob("*.json"))
    if not task_files:
        console.print("[bold red]No task files found.[/]")
        return

    # Loop sequentially processing each scenario mapped inside `./tasks/`
    for task_file in task_files:
        with task_file.open(encoding="utf-8") as handle:
            task_cfg = json.load(handle)

        # Main graphical dashboard visualization logic path
        if args.local or not args.env_url:
            import threading

            q: queue.Queue = queue.Queue(maxsize=1)
            results: Dict[str, Any] = {}
            
            # Initiate background daemon thread dedicated solely towards processing the heavy physics 
            # and latent HTTP LLM connections so the UI FPS retains perfectly smooth cadence 
            worker_thread = threading.Thread(
                target=simulation_worker,
                args=(task_cfg, task_file.name, args.model, args.api_base, args.api_key, q, results),
                daemon=True,
            )
            worker_thread.start()

            # Main interactive terminal context manager lifecycle
            with Live(
                Panel("Initializing dashboard..."), refresh_per_second=4, console=console, screen=False
            ) as live:
                # Polling mechanism waits continuously whilst the queue emits states
                while worker_thread.is_alive() or not q.empty():
                    current_state = None
                    try:
                        # Clear old stale frames dropping them from pipe, selecting only most recent state layout
                        while not q.empty():
                            current_state = q.get_nowait()
                            
                        if current_state:
                            # Verify state isn't emitting structural application errors
                            if hasattr(current_state, "error_banner") and current_state.error_banner:
                                live.update(
                                    Panel(
                                        f"[bold red]CRASH DETECTED[/]\n\n{current_state.error_banner}",
                                        border_style="red",
                                    )
                                )
                            else:
                                # Delegate Dashboard state into the UI visual renderer generator
                                live.update(make_dashboard(current_state, console.encoding))
                    except queue.Empty:
                        pass
                        
                    # Slow loop polling to conserve main-thread CPU usage
                    time.sleep(0.05)
                    
        # Otherwise execute completely remotely via REST loop with no graphics window
        else:
            results = execute_simulation(
                task_cfg=task_cfg,
                task_name=task_file.name,
                model=args.model,
                api_key=args.api_key,
                api_base=args.api_base,
                use_http=True,
                env_url=args.env_url,
                headless=False,
            )

        # Finally, attempt to execute external diagnostic graph logic showcasing trajectory success criteria
        if HAS_MATPLOTLIB and ("history" in results or "observations" in results):
            try:
                # Prepare data for plotting
                history = results.get("history", [obs["temperatures"] for obs in results.get("observations", [])])
                actions = results.get("action_history", [act["cooling"] for act in results.get("actions", [])])
                safe_temp = results.get("safe_temp", 90.0)
                target_temp = results.get("target_temp", 80.0)
                num_zones = results.get("num_zones", 1)
                
                fig, ax1 = plt.subplots(figsize=(12, 7))
                ax2 = ax1.twinx() # Create a twin axis for cooling actions
                
                steps = range(len(history))
                
                # Plot Temperatures on primary Y-axis
                for zone_idx in range(num_zones):
                    zone_temps = [temps[zone_idx] for temps in history]
                    line, = ax1.plot(steps, zone_temps, label=f"Zone {zone_idx + 1} Temp", linewidth=2)
                    
                    # Area Under the Curve (AUC) fill relative to target_temp
                    ax1.fill_between(steps, zone_temps, target_temp, 
                                     where=(np.array(zone_temps) > target_temp),
                                     interpolate=True, color=line.get_color(), alpha=0.15)
                    ax1.fill_between(steps, zone_temps, target_temp, 
                                     where=(np.array(zone_temps) <= target_temp),
                                     interpolate=True, color=line.get_color(), alpha=0.05)

                # Plot Cooling Actions on secondary Y-axis (stepped)
                action_steps = range(len(actions))
                for zone_idx in range(num_zones):
                    zone_cools = [a[zone_idx] * 100 for a in actions] # Percent
                    ax2.step(action_steps, zone_cools, where='post', 
                             linestyle='--', alpha=0.4, label=f"Zone {zone_idx + 1} Cooling %")

                # Thresholds
                ax1.axhline(y=safe_temp, color="red", linestyle="--", alpha=0.8, label="Safe Limit")
                ax1.axhline(y=target_temp, color="green", linestyle=":", alpha=0.6, label="Target")

                # Formatting
                ax1.set_xlabel("Simulation Step", fontweight='bold')
                ax1.set_ylabel("Temperature (Â°C)", fontweight='bold', color='darkred')
                ax2.set_ylabel("Cooling Command (%)", fontweight='bold', color='darkblue')
                ax2.set_ylim(0, 105)
                
                plt.title(f"Detailed Trajectory Analysis: {results['task_name']}\nModel: {args.model}", fontsize=14, fontweight='bold')
                
                # Combined legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                
                plt.grid(True, which='both', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                # Save plot
                plot_path = Path("trajectory_plot.png")
                plt.savefig(plot_path)
                console.print(f"[bold green]Graph saved to {plot_path}[/]")
                
                plt.show() 
            except Exception as exc:
                console.print(f"[dim yellow]Warning: Graph rendering failed: {exc}[/]")

        # Detailed Score Report Card
        try:
            # Reconstruct data for evaluation if needed
            if "observations" in results and "actions" in results:
                # Handle both local (results from simulation_worker) and remote (from execute_simulation)
                obs_list = results["observations"]
                act_list = results["actions"]
                
                # If we have the cfg object directly, use it, otherwise reconstruct
                if "cfg" in results:
                    cfg = results["cfg"]
                else:
                    # Fallback reconstruction for remote runs
                    cfg_dict = _load_task_config(results["task_name"])
                    cfg = TaskConfig.model_validate(cfg_dict)

                
                details = evaluate_trajectory(obs_list, act_list, cfg, return_details=True)
                
                table = Table(title="Trajectory Performance Metrics", show_header=True, header_style="bold magenta")
                table.add_column("Metric", style="dim")

                table.add_column("Score", justify="right")
                table.add_column("Status", justify="center")
                table.add_column("Raw Value", justify="right")

                def get_status(val: float) -> str:
                    if val >= 0.9: return "[bold green]EXCELLENT[/]"
                    if val >= 0.7: return "[green]GOOD[/]"
                    if val >= 0.5: return "[yellow]FAIR[/]"
                    return "[bold red]POOR[/]"

                table.add_row("Thermal Safety", f"{details['safety']:.2%}", get_status(details['safety']), "N/A")
                table.add_row("Energy Efficiency", f"{details['energy']:.2%}", get_status(details['energy']), f"{details['raw_energy']:.3f} avg")
                table.add_row("Control Smoothness", f"{details['jitter']:.2%}", get_status(details['jitter']), f"{details['raw_jitter']:.3f} jitter")
                table.add_row("Target Tracking", f"{details['target']:.2%}", get_status(details['target']), f"{details['raw_target_error']:.3f} error")
                
                summary_panel = Panel(
                    table,
                    title=f"[bold cyan]Evaluation Report - {results['task_name']}[/]",
                    subtitle=f"Final Normalized Score: [bold yellow]{details['score']:.4f}[/]",
                    border_style="cyan"
                )
                console.print(summary_panel)
            else:
                score = results.get("score", results.get("trajectory_score", 0.0))
                console.print(f"Final score: [bold cyan]{score:.2f}[/]\n")
        except Exception as exc:
            console.print(f"[dim yellow]Warning: Detailed report failed: {exc}[/]")
            score = results.get("score", results.get("trajectory_score", 0.0))
            console.print(f"Final score: [bold cyan]{score:.2f}[/]\n")

        time.sleep(1)


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        run_demo_test(args)
    else:
        # Existing entry point â€“ call the original main logic (presumed to be `main(args)`)
        # If the original script used a different function name, replace `main` accordingly.
        main(args)
