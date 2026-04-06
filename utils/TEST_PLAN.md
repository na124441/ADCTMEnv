# ADCTM Test Plan

## Scope
This document defines the first-pass test inventory for each project module so we can convert the codebase into a reliable `pytest` suite. The goal is to cover:

- schema validation and boundary handling
- deterministic simulation logic
- reward and grading math
- API contract behavior
- inference parsing and orchestration
- dashboard rendering and analytics helpers

## Suggested Test Layout

```text
tests/
  conftest.py
  test_app_entry.py
  core/
  dynamics/
  reward/
  grader/
  tasks/
  analysis/
  inference/
  server/
  ui/
```

## Shared Fixtures

### Base task config fixture
Use a deterministic fixture derived from [`tasks/easy.json`](../tasks/easy.json) with optional overrides:

```python
{
    "num_zones": 3,
    "initial_temperatures": [45.0, 48.0, 50.0],
    "initial_workloads": [0.4, 0.5, 0.3],
    "ambient_temperature": 22.0,
    "safe_temperature": 85.0,
    "max_steps": 30,
    "target_temperature": 70.0,
    "workload_volatility": 0.0,
}
```

### Common testing utilities
- seed `numpy.random` or patch `numpy.random.uniform` where transition outputs must be deterministic
- use `fastapi.testclient.TestClient` for API tests
- patch LLM and HTTP calls in inference tests
- prefer minimal `DashboardState` builders for UI rendering tests

## Module Test Matrix

### [`app.py`](../app.py)
Purpose: top-level server entry shim.

Test cases:
- import exposes `app` from [`server/app.py`](../server/app.py)
- `main()` delegates to server main without altering arguments
- script entry path is import-safe and has no side effects besides module import

### [`config/constants.py`](../config/constants.py)
Purpose: physics constants.

Test cases:
- constants exist and are numeric
- constants remain positive where expected (`ALPHA`, `BETA`, `GAMMA` > 0)
- regression test locks expected values to avoid accidental tuning drift

### [`core/models.py`](../core/models.py)
Purpose: observation, action, reward, and info schemas.

Test cases:
- `Observation` accepts valid equal-length arrays
- `Observation` rejects mismatched `temperatures`, `workloads`, and `cooling`
- `Observation` rejects empty arrays
- `Observation.time_step` rejects negative values
- `Action` accepts valid cooling list
- `Action` clips values below `0.0` to `0.0`
- `Action` clips values above `1.0` to `1.0`
- `Action` preserves in-range floats exactly
- `Reward` accepts scalar float values
- `InfoDict` requires `step`

### [`core/state.py`](../core/state.py)
Purpose: environment state container.

Test cases:
- `EnvState` stores config, observation, counter, and done flag
- default values are `step_counter=0` and `done=False`
- invalid nested objects surface validation errors

### [`core/paths.py`](../core/paths.py)
Purpose: filesystem path definitions.

Test cases:
- `BASE_DIR` resolves to repository root
- `TASKS_DIR` resolves to `<repo>/tasks`
- task directory exists in the working tree

### [`tasks/task_config.py`](../tasks/task_config.py)
Purpose: task schema and consistency validation.

Test cases:
- accepts valid easy, medium, and hard task JSON payloads
- rejects `num_zones < 1`
- rejects `max_steps <= 0`
- rejects mismatched `initial_temperatures` length
- rejects mismatched `initial_workloads` length
- rejects `safe_temperature <= target_temperature`
- rejects negative `workload_volatility`
- rejects workload values outside `[0, 1]`
- rejects negative `degradation_step`
- rejects out-of-range `degraded_zone`
- accepts extra keys because `extra="allow"`

### [`core/simulator.py`](../core/simulator.py)
Purpose: main episode session and task loading.

Test cases:
- constructor initializes observation from `TaskConfig`
- `get_state()` returns config, observation, counter, and done
- `from_dict()` builds a working session from raw dict input
- `from_task_name("easy")` loads [`tasks/easy.json`](../tasks/easy.json)
- `from_task_name("easy.json")` strips extension correctly
- `from_task_name()` raises `FileNotFoundError` for missing task
- `step()` rejects actions after session is done
- `step()` rejects wrong action length
- `step()` validates action schema errors from bad payloads
- `step()` increments `step_counter`
- `step()` updates `observation.time_step`
- `step()` marks `done=True` exactly at `max_steps`
- `model_dump()` mirrors `get_state()`

Integration-style cases:
- patch [`dynamics/thermal_model.py`](../dynamics/thermal_model.py) and [`reward/reward_fn.py`](../reward/reward_fn.py) to verify `step()` passes correct `prev_obs`, `curr_obs`, `act`, and `config`
- run a deterministic no-volatility episode and verify returned `info.step` increments monotonically

### [`dynamics/thermal_model.py`](../dynamics/thermal_model.py)
Purpose: one-step thermal transition.

Test cases:
- deterministic transition with `workload_volatility=0.0`
- workload random walk stays clipped to `[0.0, 1.0]`
- returned `Observation.time_step` increments by one
- returned `cooling` matches action cooling
- temperatures move according to `ALPHA`, `BETA`, `GAMMA`
- temperatures never fall below ambient temperature
- degradation not applied before `degradation_step`
- degradation halves cooling effect for `degraded_zone` at and after threshold
- only degraded zone is affected by degradation logic

Edge cases:
- zero cooling with high workloads increases temperatures
- high cooling with low workloads can cool but not below ambient
- patch RNG to produce both positive and negative workload perturbations

### [`reward/reward_fn.py`](../reward/reward_fn.py)
Purpose: reward calculation from safety, energy, and jitter.

Test cases:
- no violations + zero cooling + first step yields zero penalty reward
- temperature violations create quadratic penalty
- energy cost equals sum of current cooling
- jitter penalty applies from second step onward
- first step bypasses jitter penalty
- jitter bypass activates near safe threshold using `jitter_bypass_threshold`
- return type is `Reward`
- reward is negative total cost

Risk-focused cases:
- verify `safe_temperature` from config is actually used in violation math
- verify function does not raise `NameError` under normal inputs
- verify non-ASCII identifiers do not break import/runtime in CI environment

### [`grader/metrics.py`](../grader/metrics.py)
Purpose: trajectory metrics.

Test cases:
- zero actions returns `(0.0, 0.0, 0.0)`
- safety ratio uses post-initial observations only
- safety ratio is `1.0` when all steps stay below threshold
- safety ratio reflects mixed safe/unsafe steps correctly
- average energy equals mean of all cooling values
- one-step action list yields zero jitter
- multi-step jitter equals mean absolute difference across consecutive actions

### [`grader/evaluator.py`](../grader/evaluator.py)
Purpose: weighted trajectory score.

Test cases:
- combines metrics with weights `0.5`, `0.3`, `0.2`
- perfect safety, zero energy, zero jitter produces `1.0`
- poor metrics produce score clamped at lower bound `0.0`
- overly large negative component combinations still clamp into `[0, 1]`
- patch `compute_metrics()` to verify delegation and weighting only

### [`analysis/policy_analyzer.py`](../analysis/policy_analyzer.py)
Purpose: heuristic assessment of control quality and policy style.

Test cases for `evaluate_action_quality()`:
- near-safe temperature with strong action returns `"Optimal (Reactive)"`
- near-safe temperature with moderate action returns `"Warning (Passive)"`
- near-safe temperature with weak action returns under-reacting label
- low temperature with large action returns over-reacting label
- negative delta beyond threshold returns cooling label
- small delta with hot temp and low action returns idle-danger label
- small delta with hot temp and nontrivial action returns stable-maintaining label
- neutral branch covers remaining combinations

Test cases for `assess_policy_type()`:
- high variance returns aggressive-reactive
- very low variance returns steady-stable
- high average action with moderate variance returns steady-aggressive
- low average action with moderate variance returns passive-conservative
- otherwise returns balanced-optimal

### [`analysis/trend_predictor.py`](../analysis/trend_predictor.py)
Purpose: forecast thermal trend and time-to-violation.

Test cases:
- insufficient history returns `"Initializing..."` for each zone
- cooling or flat trend returns `"Stable/Cooling"`
- already above safe temperature returns `"CRITICAL"`
- positive trend crossing within one step returns short-warning label
- positive trend further out returns approximate step count string
- multiple zones are processed independently

### [`inference/parser.py`](../inference/parser.py)
Purpose: parse model text into `Action`.

Test cases:
- parses plain JSON object
- parses JSON surrounded by extra text
- rejects response with no JSON block
- rejects missing `cooling` key
- rejects non-list `cooling`
- rejects wrong list length
- rejects non-numeric values
- clamps parsed values into `[0, 1]`

### [`inference/prompt.py`](../inference/prompt.py)
Purpose: construct prompt text from observation and config.

Test cases:
- `build_zone_table()` renders one row per zone
- table contains temperature, workload, and previous cooling values
- `build_prompt()` injects safe temp, step, ambient temp, and zone count
- prompt includes exact instruction to output only JSON
- zone ordering in prompt matches observation ordering

### [`inference.py`](../inference.py)
Purpose: root inference workflow, dashboard worker, and CLI.

Test cases for helper functions:
- `_get_theme()` returns expected named styles
- `_fallback_action()` returns `0.3` for every zone
- `_load_task_config()` loads task JSON correctly

Test cases for `call_llm()`:
- successful response returns stripped content
- transient failure retries and eventually succeeds
- repeated failure raises final exception after retry budget

Test cases for `execute_simulation()` in local mode:
- validates task config before execution
- requires `env_url` when `use_http=True`
- appends initial observation before loop
- uses fallback action when parsing fails
- records parse errors with step and raw output
- stops when environment returns `done=True`
- computes `trajectory_score` with collected observations/actions

Test cases for `execute_simulation()` in HTTP mode:
- sends `/reset` and `/step` to provided base URL
- strips trailing slash from `env_url`
- propagates HTTP errors from reset/step calls

Test cases for `simulation_worker()`:
- enqueues initial dashboard state
- updates log/status across loop iterations
- stores final `results` payload with score/history/task metadata
- falls back cleanly when LLM parse fails
- returns crash banner state when internal dashboard construction fails

Test cases for `main()`:
- prints message and exits when no task files exist
- local mode starts worker thread and renders dashboard
- remote mode calls `execute_simulation()`
- plotting branch is skipped safely when matplotlib unavailable

### [`inference/inference.py`](../inference/inference.py)
Purpose: package shim that re-exports root inference execution.

Test cases:
- module exposes `execute_simulation`
- import fails loudly if root inference module cannot be loaded

### [`core/dashboard_state.py`](../core/dashboard_state.py)
Purpose: immutable dashboard DTO.

Test cases:
- dataclass defaults populate optional fields correctly
- `log` default is a bounded deque of length 20
- instance is frozen and rejects mutation
- list and dict defaults are isolated per instance

### [`ui/dashboard.py`](../ui/dashboard.py)
Purpose: compose Rich dashboard layout.

Test cases:
- returns a `Layout`
- creates expected top-level sections: header, visuals, intelligence, agent, footer
- handles normal state with populated logs
- handles empty log by showing standby message
- critical overheat branch promotes red alert styling/message
- optimization-gap branch shows idle warning when hot and low action

### [`ui/panels/header.py`](../ui/panels/header.py)
Purpose: render top header views.

Test cases:
- `render_title()` returns a `Panel`
- `render_header()` includes task, model, and step values
- `render_header()` includes alpha, beta, gamma, and safe temperature values

### [`ui/panels/metrics.py`](../ui/panels/metrics.py)
Purpose: compact metrics summary.

Test cases:
- uses average and max temp from state
- shows `"COMPLETED"` when `done=True`
- maps stable, trending, and fallback statuses to distinct styles
- handles empty temperature list without crashing

### [`ui/panels/thermal.py`](../ui/panels/thermal.py)
Purpose: thermal bars and action overlay.

Test cases for `render_normalized_heatbars()`:
- returns a `Panel`
- uses UTF block char for UTF encodings
- falls back to `#` for non-UTF encodings
- marks above-safe zones with critical styling and `X`
- clamps bar lengths between `0` and `20`

Test cases for `render_action_overlay()`:
- returns a `Panel`
- produces one row per zone
- action bar length scales with action intensity
- negative normalized temperatures do not produce negative bar lengths

### [`ui/panels/analysis.py`](../ui/panels/analysis.py)
Purpose: analytical dashboard panels.

Test cases:
- `render_delta_graph()` maps strong rise, mild rise, mild fall, strong fall, and flat deltas correctly
- `render_reward_chart()` scales bars from absolute penalty terms
- `render_forecast_panel()` colors imminent forecasts more aggressively
- `render_forecast_panel()` tolerates malformed forecast strings
- `render_action_quality_panel()` renders one row per zone
- `render_behavior_summary()` surfaces under-reaction and over-reaction issues
- `render_behavior_summary()` shows stable message when no issues exist

### [`ui/panels/llm.py`](../ui/panels/llm.py)
Purpose: prompt/raw/parsed decision trace.

Test cases:
- returns a `Panel`
- renders prompt fallback placeholder when prompt is empty
- renders raw JSON fallback `{}` when output is empty
- renders one parsed cooling row per zone
- parse error branch includes error panel and parsed table together

### [`server/app.py`](../server/app.py)
Purpose: REST API around shared simulation session.

Test cases for API endpoints:
- `GET /` returns welcome payload
- `POST /reset` with no body loads default easy task
- `POST /reset` with valid config returns initial observation
- `POST /reset` with invalid config returns `422`
- `POST /step` before reset returns `400`
- `POST /step` with invalid action schema returns `422`
- `POST /step` with wrong action length returns `400`
- `GET /state` before reset returns `400`
- `GET /state` after reset returns full session dump

Concurrency/state cases:
- reset replaces previous global session
- `_ensure_initialized()` raises `HTTPException` when session missing
- route layer converts `ValidationError` and `ValueError` to correct HTTP codes

### [`core/env.py`](../core/env.py)
Purpose: this file currently duplicates the FastAPI environment entrypoint and needs tests aligned to its current behavior once its final form stabilizes.

Pre-design test list:
- mirror the same API contract tests as [`server/app.py`](../server/app.py) if it remains a public entrypoint
- add regression tests for any divergences introduced in the edited version of the file
- add import test to confirm `app` object remains available to [`server/app.py`](../server/app.py)

## End-to-End Scenarios

These should become a small integration suite after the unit layer is in place.

### Local simulation flow
- load each of [`tasks/easy.json`](../tasks/easy.json), [`tasks/medium.json`](../tasks/medium.json), and [`tasks/hard.json`](../tasks/hard.json)
- patch `call_llm()` to emit deterministic valid actions
- assert episode terminates at `max_steps`
- assert score stays within `[0.0, 1.0]`

### Parse-failure recovery flow
- force malformed LLM output for one or more steps
- assert fallback action is used
- assert parse error is recorded without aborting the run

### API-driven remote flow
- boot `TestClient(app)`
- run `execute_simulation(..., use_http=True, env_url=test_server_url)` via patched request layer or live test server
- assert reset, step, and final scoring all complete successfully

## Priority Order

### P0
- [`core/models.py`](../core/models.py)
- [`tasks/task_config.py`](../tasks/task_config.py)
- [`core/simulator.py`](../core/simulator.py)
- [`dynamics/thermal_model.py`](../dynamics/thermal_model.py)
- [`reward/reward_fn.py`](../reward/reward_fn.py)
- [`server/app.py`](../server/app.py)
- [`inference/parser.py`](../inference/parser.py)

### P1
- [`grader/metrics.py`](../grader/metrics.py)
- [`grader/evaluator.py`](../grader/evaluator.py)
- [`inference.py`](../inference.py)
- [`analysis/policy_analyzer.py`](../analysis/policy_analyzer.py)
- [`analysis/trend_predictor.py`](../analysis/trend_predictor.py)

### P2
- [`ui/dashboard.py`](../ui/dashboard.py)
- [`ui/panels/header.py`](../ui/panels/header.py)
- [`ui/panels/metrics.py`](../ui/panels/metrics.py)
- [`ui/panels/thermal.py`](../ui/panels/thermal.py)
- [`ui/panels/analysis.py`](../ui/panels/analysis.py)
- [`ui/panels/llm.py`](../ui/panels/llm.py)

## Notes Before Implementation

- Some files contain likely regression risks that tests should expose immediately, especially [`reward/reward_fn.py`](../reward/reward_fn.py).
- UI tests should focus on object creation and key rendered text fragments, not full Rich snapshot exactness at first.
- For transition and reward math, deterministic fixtures plus patched randomness will matter more than large test volume.
