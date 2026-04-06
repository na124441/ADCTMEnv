# ADCTM Project Summary

Generated on March 31, 2026

## 1. Executive Summary

ADCTM stands for Autonomous Data Centre Thermal Management. The repository implements a complete simulation environment for controlling cooling across multiple data-center zones, plus the tooling needed to run an LLM-driven controller, score its behavior, visualize its decisions, and validate the whole system with automated tests.

At a high level, the project answers one operational question:

"Given current temperatures, workloads, and past cooling decisions, how should an agent set the next cooling levels so the racks stay safe without wasting energy or oscillating?"

The codebase is organized around that loop:

1. A task configuration defines the number of thermal zones and operating constraints.
2. A simulation session produces observations and accepts cooling actions.
3. A thermal model advances temperatures and workloads one step at a time.
4. A reward function scores each action locally.
5. A grader evaluates the whole trajectory globally.
6. An inference pipeline prompts an LLM to act as the controller.
7. A Rich-based dashboard visualizes the reasoning and the system state.
8. A FastAPI service exposes the environment over HTTP.
9. A broad pytest suite verifies the implementation.

In practice, the repository is both:

- an OpenEnv-style benchmark environment
- a reference application for running and observing an AI control loop end to end

## 2. Repository At A Glance

| Area | Purpose | Key Files |
| --- | --- | --- |
| Root entry points | Launch server and top-level inference workflow | `app.py`, `inference.py` |
| Configuration | Core physical coefficients | `config/constants.py` |
| Core runtime | Data models, paths, API environment, simulator, state objects | `core/models.py`, `core/env.py`, `core/simulator.py`, `core/dashboard_state.py` |
| Dynamics | One-step thermal transition logic | `dynamics/thermal_model.py` |
| Reward | Per-step reward shaping | `reward/reward_fn.py` |
| Grading | Whole-episode metrics and normalized scoring | `grader/metrics.py`, `grader/evaluator.py` |
| Analysis | Lightweight interpretability heuristics | `analysis/policy_analyzer.py`, `analysis/trend_predictor.py` |
| Inference package | Prompt building, parsing, re-export shim | `inference/prompt.py`, `inference/parser.py`, `inference/inference.py` |
| Server wrapper | Adds a one-call `/run` endpoint on top of the core API | `server/app.py` |
| Tasks | Difficulty presets and task schema | `tasks/*.json`, `tasks/task_config.py` |
| UI | Dashboard composition and panel rendering | `ui/dashboard.py`, `ui/panels/*.py` |
| Packaging and ops | Dependency, deployment, environment metadata | `pyproject.toml`, `requirements.txt`, `Dockerfile`, `openenv.yaml` |
| Quality layer | Test suite, runner, and readiness checks | `tests/`, `run_tests.py`, `TEST_PLAN.md` |

## 3. End-To-End Control Flow

The easiest way to understand the project is to follow one complete episode.

```text
Task JSON
  -> TaskConfig validation
  -> SimulationSession initialization
  -> Observation emitted
  -> Prompt built for the LLM
  -> LLM returns JSON cooling action
  -> Parser validates and clamps action
  -> Thermal model computes next temperatures/workloads
  -> Reward function computes step reward
  -> Session checks done / max_steps
  -> Dashboard renders current state
  -> Grader scores the full trajectory at the end
```

There are two execution modes:

- Local mode: `inference.py --local` keeps everything in-process using `SimulationSession`.
- Remote mode: `inference.py --env-url ...` talks to the FastAPI API using `/reset` and `/step`.

There is also a convenience server endpoint:

- `POST /run` in `server/app.py` loads a task and executes the full LLM-controlled rollout in one request.

## 4. Root-Level Entry Points

### `app.py`

This file is intentionally thin. It imports `app` and `main` from `server.app` and delegates execution there. Its role is mainly convenience: users can run `python app.py` without needing to know the internal package layout.

### Root `inference.py`

This is the operational centerpiece of the project. It contains:

- CLI argument parsing
- OpenAI-compatible model calling logic
- the local and HTTP simulation loops
- the threaded dashboard worker
- optional post-run plotting with matplotlib

Because it owns orchestration rather than a narrow utility, it sits at the repository root instead of inside a single subpackage.

## 5. Core Data Models And State Objects

### `core/models.py`

This file defines the strongly typed contracts that everything else relies on.

#### `Observation`

Represents what the controller is allowed to see at each step:

- `temperatures`: per-zone temperatures
- `workloads`: per-zone utilization in `[0, 1]`
- `cooling`: previous per-zone cooling commands
- `ambient_temp`: external ambient temperature
- `time_step`: current step index

Important validation behavior:

- all three per-zone arrays must have the same length
- `time_step` must be non-negative
- empty lists are rejected

#### `Action`

Represents the agent's decision:

- `cooling`: one float per zone

Important validation behavior:

- values are clamped into `[0.0, 1.0]`
- downstream code still checks that the list length matches the configured number of zones

#### `Reward`

Wraps the scalar reward value returned after each step.

#### `InfoDict`

Currently minimal, but keeps the output structure explicit and extensible.

### `tasks/task_config.py`

`TaskConfig` is the schema for scenario definition. It controls both the initial state and the rules of the environment.

Core fields:

- `num_zones`
- `initial_temperatures`
- `initial_workloads`
- `ambient_temperature`
- `safe_temperature`
- `max_steps`
- `target_temperature`

Advanced and dynamic fields:

- `workload_volatility`
- `degradation_step`
- `degraded_zone`
- `jitter_bypass_threshold`

Important consistency checks:

- initial temperature and workload lengths must match `num_zones`
- `safe_temperature` must be above `target_temperature`
- workloads must stay within `[0, 1]`
- degradation parameters must be non-negative and in range
- extra keys are allowed, which makes the task schema forward-compatible

### `core/state.py`

`EnvState` is a simple Pydantic container that groups:

- the active `TaskConfig`
- the current `Observation`
- the current `step_counter`
- the `done` flag

It is conceptually useful even though the main runtime logic is centered around `SimulationSession`.

### `core/dashboard_state.py`

`DashboardState` is the UI-focused state transfer object. It is an immutable dataclass designed for the Rich dashboard layer rather than for physics or API exchange.

It carries:

- live temperatures, cooling, and reward information
- hyperparameters and thresholds
- reward decomposition terms
- derived analytics like deltas, forecasts, and action assessments
- rolling histories for charts
- prompt / raw LLM output / parsed action data
- runtime identity like task name and model name
- a bounded log buffer for status messages

This is the object that lets the dashboard render a full "command center" view from one frozen snapshot.

### `core/paths.py`

This centralizes path discovery:

- `BASE_DIR` = repository root
- `TASKS_DIR` = `<repo>/tasks`

The value here is not complexity reduction so much as consistency: task loading happens in one way across the codebase.

## 6. Simulation Lifecycle And Environment API

### `core/simulator.py`

`SimulationSession` is the environment runtime wrapper. It combines the task configuration, the current observation, the step counter, and the terminal state.

Its responsibilities are:

- initialize the first observation from a validated task
- load tasks either from raw dicts or from named JSON files
- accept an action, validate it, and apply one transition
- compute the per-step reward
- track `done` based on `max_steps`
- return a normalized step payload

The `step()` method is the most important method in the entire environment layer. Its sequence is:

1. Reject stepping after the episode is already done.
2. Parse the raw action dict into an `Action`.
3. Verify the action length matches `config.num_zones`.
4. Save the previous observation.
5. Produce the next observation with `apply_transition(...)`.
6. Compute reward with `compute_reward(...)`.
7. Update internal state and counters.
8. Mark the episode done when `step_counter >= max_steps`.
9. Return observation, reward, done, and step metadata.

### `core/env.py`

This file exposes the environment as a FastAPI application. It is the shared HTTP base used by `server/app.py`.

Endpoints:

- `GET /`
  Purpose: health check / welcome message.

- `POST /reset`
  Purpose: create a new `SimulationSession`.
  Behavior: if no config is supplied, load the default `easy` task.

- `POST /step`
  Purpose: advance the active session one step using a submitted action.

- `GET /state`
  Purpose: return the full current session state for debugging or inspection.

Important implementation details:

- the active session is stored in a module-level `CURRENT_SESSION`
- a thread lock protects session mutation
- validation and user errors are translated into appropriate HTTP errors

This design makes the environment convenient for external agents and benchmarks that expect a RESTful reset-step-state pattern.

## 7. Thermal Dynamics Model

### `dynamics/thermal_model.py`

This file contains the one-step system dynamics through `apply_transition(...)`.

The model uses three physical coefficients defined in `config/constants.py`:

- `ALPHA = 5.0`
- `BETA = 8.0`
- `GAMMA = 0.1`

The per-step temperature update is:

```text
delta_t = ALPHA * workload
          - cooling_effect * cooling
          + GAMMA * (ambient_temp - temperature)
```

Then:

```text
next_temperature = max(temperature + delta_t, ambient_temp)
```

Interpretation of each term:

- `ALPHA * workload`
  Adds heat as server utilization rises.

- `cooling_effect * cooling`
  Removes heat based on the control action.

- `GAMMA * (ambient_temp - temperature)`
  Pulls the zone toward ambient temperature over time.

Workload evolution:

- workloads are randomly perturbed each step by a uniform noise term
- the magnitude is controlled by `config.workload_volatility`
- results are clipped back into `[0.0, 1.0]`

Degradation behavior:

- if both `degradation_step` and `degraded_zone` are configured
- and the current observation step has reached the degradation threshold
- then the degraded zone's cooling effect is halved

That degradation mechanic is what makes the hard task materially different from a simple scale-up in zone count.

## 8. Reward Design

### `reward/reward_fn.py`

The reward function turns one transition into a scalar objective. It is cost-based, so the returned reward is the negative of combined penalties.

The penalty components are:

1. Temperature violation penalty
   `sum(max(0, temp - safe_temp)^2)`

2. Energy cost
   `sum(cooling)`

3. Jitter penalty
   `sum(abs(current_cooling - previous_cooling))`

The weighted total is:

```text
total_cost = 2.0 * temp_penalty
           + 1.0 * energy_cost
           + 0.5 * jitter

reward = -total_cost
```

Important nuance:

- the jitter term is only considered after the first step
- the implementation bypasses jitter when the system is near the safety threshold
- the bypass threshold is controlled by `config.jitter_bypass_threshold`

This reward structure encodes the intended engineering trade-off very clearly:

- safety first
- energy efficiency second
- control smoothness third

## 9. Trajectory Grading

### `grader/metrics.py`

This module computes three whole-trajectory metrics:

- safety ratio
- average energy use
- average jitter

Definitions:

- safety ratio:
  fraction of post-initial steps where every zone stays at or below `safe_temperature`

- average energy:
  mean of all cooling values across all zones and steps

- average jitter:
  mean absolute change in cooling between consecutive actions

### `grader/evaluator.py`

This combines the metrics into a final normalized score:

```text
score = 0.5 * safety
      + 0.3 * (1 - avg_energy)
      + 0.2 * (1 - avg_jitter)
```

Then the result is clamped into `[0.0, 1.0]`.

This separation between reward and grade is architecturally useful:

- reward drives local, step-by-step optimization pressure
- grader evaluates overall task performance in a benchmark-friendly way

## 10. Task Definitions And Difficulty Progression

The repository ships with three predefined tasks in `tasks/`.

| Task | Zones | Initial Temps | Initial Workloads | Ambient | Safe | Target | Max Steps | Extra Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `easy.json` | 3 | 45, 48, 50 | 0.4, 0.5, 0.3 | 22 | 85 | 70 | 30 | none |
| `medium.json` | 5 | 50, 55, 58, 53, 57 | 0.6, 0.7, 0.5, 0.6, 0.8 | 30 | 80 | 68 | 20 | none |
| `hard.json` | 8 | 60, 62, 58, 61, 63, 59, 64, 60 | 0.9, 0.8, 0.9, 0.7, 0.95, 0.85, 0.9, 0.8 | 35 | 75 | 65 | 15 | volatility 0.08, degradation at step 7 in zone 4 |

The difficulty progression is not just "more zones":

- ambient temperature increases
- safe temperature decreases
- target temperature decreases
- episode length shortens
- workloads get heavier
- the hard task introduces explicit degradation

This creates a sensible curriculum for benchmarking controllers.

## 11. LLM Inference Pipeline

### Prompt construction: `inference/prompt.py`

This module builds the model-facing instruction string.

The prompt explicitly tells the model:

- keep every zone under the safe temperature
- minimize cooling usage
- avoid rapid changes
- output exactly one JSON object
- provide a `cooling` list whose order matches the zone order

It also renders a zone table with:

- current temperature
- current workload
- previous cooling level

This is a good prompt shape for control problems because it is explicit, tabular, and machine-parseable.

### Output parsing: `inference/parser.py`

This module extracts the first JSON object from the model response, validates the `cooling` list, converts each element to float, clamps values into range, and returns a typed `Action`.

If the response does not contain valid JSON or the wrong number of zones, parsing fails fast.

### Package shim: `inference/inference.py`

This file dynamically loads the root `inference.py` and re-exports `execute_simulation`.

Its purpose is namespace convenience:

- package code can import `inference.execute_simulation`
- the real orchestration code can still live at the repository root

### Root orchestration: `inference.py`

This file provides several important functions.

#### `call_llm(...)`

- creates an OpenAI-compatible client
- sends a system prompt plus the task prompt
- uses `temperature=0.0` for deterministic behavior
- retries with exponential backoff

#### `_fallback_action(...)`

If parsing fails, the system falls back to `0.3` cooling in every zone. This keeps the rollout alive instead of failing hard.

#### `execute_simulation(...)`

This is the headless evaluator. It:

- validates the supplied task config
- initializes either local or HTTP execution
- records observations and actions
- prompts the LLM on every step
- applies the resulting action
- stores parse errors without aborting the episode
- accumulates reward
- returns a full result bundle including the final trajectory score

#### `simulation_worker(...)`

This is the dashboard-enabled execution path. It runs in a background thread and continuously emits `DashboardState` snapshots through a queue while the UI thread renders them.

In addition to simulation work, it computes:

- thermal deltas
- short-horizon temperature forecasts
- action quality heuristics
- recent action statistics
- behavior labels for the dashboard

#### `main()`

The CLI entry point:

- finds all task JSON files
- runs each task sequentially
- launches the live dashboard in local mode
- uses headless execution in HTTP mode
- optionally plots thermal traces with matplotlib
- prints the final score

## 12. Analysis And Interpretability Helpers

### `analysis/policy_analyzer.py`

This module adds human-readable interpretation on top of raw control signals.

`evaluate_action_quality(...)` classifies each zone's action using heuristics such as:

- optimal reactive behavior near the safety limit
- passive or under-reactive behavior when hot
- over-reactive behavior when already cool
- stable maintenance behavior

`assess_policy_type(...)` labels the overall controller style from rolling action statistics, for example:

- `AGGRESSIVE-REACTIVE`
- `STEADY-STABLE`
- `STEADY-AGGRESSIVE`
- `PASSIVE-CONSERVATIVE`
- `BALANCED-OPTIMAL`

These are not formal control-theory proofs, but they are useful for operator-facing observability.

### `analysis/trend_predictor.py`

This module estimates near-future thermal risk using a rolling history window.

For each zone it returns labels like:

- `Initializing...`
- `Stable/Cooling`
- `CRITICAL`
- `< 1 step`
- `~N steps`

This gives the dashboard predictive, not just reactive, insight.

## 13. Dashboard UI Architecture

### `ui/dashboard.py`

This is the composition layer that arranges the full Rich layout.

Top-level regions:

- header
- visuals
- intelligence
- agent
- footer

It merges status, charts, intelligence summaries, and the LLM trace into a single terminal interface.

### `ui/panels/header.py`

Renders the top strip with:

- task name
- model name
- current step
- alpha, beta, gamma
- safe temperature

### `ui/panels/metrics.py`

Renders compact live metrics such as:

- average temperature
- maximum temperature
- trajectory status
- confidence
- total reward

### `ui/panels/thermal.py`

Provides the visual thermals:

- normalized heat bars from target-to-safe range
- an action overlay comparing thermal load versus cooling intensity

### `ui/panels/analysis.py`

Provides the analytical side panels:

- delta graph
- reward decomposition chart
- forecast panel
- action quality table
- behavior summary

### `ui/panels/llm.py`

Shows the model's reasoning artifacts in a structured way:

- the prompt sent to the model
- raw JSON output
- parsed per-zone cooling decision
- parse error banner if parsing failed

Taken together, the dashboard makes the project much more than a plain benchmark. It becomes an inspectable control-system demo.

## 14. Server Wrapper And Unified Run Endpoint

### `server/app.py`

This module layers extra functionality on top of `core.env.app`.

Key behavior:

- imports the base FastAPI app from `core.env`
- dynamically loads the root `inference.py`
- exposes `POST /run`

`POST /run` accepts either:

- `task_id`
- or a raw `config`

It also accepts:

- `model`
- `api_key`
- `api_base`

Then it calls `execute_simulation(...)` directly and returns the full result.

This endpoint is useful when a client wants one HTTP request that performs:

- task loading
- LLM control
- full rollout execution
- final scoring

without having to manually drive `/reset` and `/step`.

## 15. Packaging, Metadata, And Deployment Files

### `pyproject.toml`

Defines the package metadata:

- project name: `multi-zone-cooling`
- version: `1.0.0`
- Python requirement: `>=3.10`
- runtime dependencies
- a console script entry point named `server`

### `requirements.txt`

Lists the runtime and test dependencies, including:

- FastAPI
- Uvicorn
- Pydantic
- NumPy
- OpenAI client
- Requests
- Dotenv
- Rich
- Plotext
- Pytest

### `openenv.yaml`

Provides OpenEnv-style environment metadata:

- name
- version
- description
- observation/action/reward types
- supported capabilities
- domain and difficulty metadata

### `Dockerfile`

Packages the project for containerized deployment. Important choices:

- base image: `python:3.11-slim`
- installs dependencies from `requirements.txt`
- copies the full repository
- exposes port `7860`
- launches `uvicorn server.app:app`

This lines up with Hugging Face Space style deployment expectations.

### `README.md`

Documents:

- the environment concept
- action and observation spaces
- the task ladder
- install and run instructions
- Docker usage
- baseline scores

### `TEST_PLAN.md`

This file is a planning artifact describing the intended test inventory by module. It is especially useful for maintainers because it explains the reasoning behind the test suite structure and the important risk areas.

### `run_tests.py`

Provides a clean wrapper around pytest and disables auto-loaded plugins for a more stable local test run.

## 16. Automated Test Suite And Current Health

The repository includes a substantial pytest suite under `tests/` with coverage across:

- root entry points
- config constants
- submission readiness checks
- core schemas and simulator behavior
- task configuration validation
- thermal model and reward math
- grader metrics and evaluator logic
- analysis helpers
- inference prompt, parser, and orchestration
- FastAPI server behavior
- Rich dashboard composition and panels

Repository-level test inventory observed in this workspace:

- 39 files under `tests/`
- broad module-by-module coverage
- helper fixtures in `tests/conftest.py`
- an optional OpenEnv CLI validation path guarded behind a skip condition

Validation run performed for this summary:

```text
Command: python run_tests.py
Result: 81 passed, 1 skipped
Duration: 13.29s reported by pytest
```

That result matters because it shows the project is not just structured well on paper. The current workspace state is also executable and testable.

## 17. Notable Design Strengths

The strongest parts of the repository are architectural rather than cosmetic.

### Clear separation of concerns

The codebase cleanly separates:

- state contracts
- physics
- reward shaping
- episode grading
- agent interaction
- HTTP serving
- terminal visualization
- automated validation

This makes the project easy to reason about and straightforward to extend.

### Strong typed boundaries

Pydantic models are used at the important edges:

- observation payloads
- action payloads
- reward payloads
- task configurations

That helps keep both API behavior and internal simulation behavior predictable.

### Dual execution modes

The project supports:

- in-process simulation for fast development and dashboarding
- HTTP-driven simulation for external integration

This is a valuable design choice because it supports both experimentation and benchmark deployment.

### Good observability

The combination of:

- dashboard state snapshots
- LLM prompt and output tracing
- control-quality heuristics
- temperature forecasting

means the project can be explained and debugged, not just run.

### Mature testing posture

The existence of:

- a detailed test plan
- per-module tests
- readiness checks
- a dedicated test runner

shows that the project is moving from prototype status toward reliable submission-quality software.

## 18. Practical Extension Points

If someone wants to grow the project, the main extension points are clear.

### Add more scenarios

Create new JSON files in `tasks/` with different:

- zone counts
- workload patterns
- ambient conditions
- degradation patterns
- safety margins

### Tune the physics model

Adjust:

- `ALPHA` for workload heating intensity
- `BETA` for cooling effectiveness
- `GAMMA` for ambient coupling

or replace `apply_transition(...)` with a more sophisticated thermal model.

### Change controller behavior

Modify:

- `inference/prompt.py` to change the agent instruction style
- `inference/parser.py` if the action schema changes
- root `inference.py` if a different model backend or orchestration strategy is needed

### Expand interpretability

The analysis helpers are intentionally lightweight and easy to extend. New heuristics or learned predictors could be added without changing the simulator itself.

### Upgrade the dashboard

Because the UI is panelized, new charts or status panels can be added with limited coupling.

## 19. Final Assessment

ADCTM is a well-structured simulation-and-control project that combines:

- an environment definition
- a thermal dynamics engine
- a reward model
- a benchmark-oriented grader
- an LLM control pipeline
- a live dashboard
- an HTTP API
- a healthy test suite

Its main value is that it does not stop at one layer. It connects scenario definition, simulation, agent control, visualization, evaluation, packaging, and automated validation into one coherent system.

If someone needed to explain the repository in one sentence, the best description would be:

"ADCTM is an end-to-end benchmark and demo environment for AI-controlled multi-zone data-center cooling, with typed state models, realistic task presets, a simulation engine, an LLM controller loop, a live operator dashboard, and strong automated test coverage."
