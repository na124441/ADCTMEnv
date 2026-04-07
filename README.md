---
title: ADCTM
emoji: 🔥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

<div align="center">

# 🌡️ ADCTM
### Autonomous Data Centre Thermal Management

*A high-fidelity, physics-based simulation environment for benchmarking AI-driven thermal control agents*

---

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-00d4aa?style=flat-square)

---

> 💡 **"Cooling systems account for 30–40% of total data-centre energy consumption."**
> *ADCTM turns this into a rigorous AI control benchmark.*

---

| 🌡️ Thermal Zones | 📊 Eval Metrics | 🔌 API Endpoints | 🤖 Agent Paradigms |
|:---:|:---:|:---:|:---:|
| **3 to 8** | **4 weighted** | **5 REST** | **RL + LLM** |

</div>

---

## 📖 Table of Contents

1. [🚀 Overview](#-overview)
2. [📐 Problem Formulation — MDP](#-problem-formulation--mdp)
3. [⚙️ System Architecture](#-system-architecture)
4. [🧪 Physics Engine](#-physics-engine)
5. [🧠 Real-World Complexity Modeling](#-real-world-complexity-modeling)
6. [🏆 Tasks and Difficulty Scaling](#-tasks-and-difficulty-scaling)
7. [📊 Evaluation Metrics](#-evaluation-metrics)
8. [🎯 Reward Function Design](#-reward-function-design)
9. [🤖 Baseline Approaches](#-baseline-approaches)
10. [⚙️ Installation](#-installation)
11. [🚀 Usage](#-usage)
12. [🧠 LLM Client Configuration](#-llm-client-configuration)
13. [📌 API Reference](#-api-reference)
14. [📁 Project Structure](#-project-structure)
15. [🧪 Testing](#-testing)
16. [💡 Design Philosophy](#-design-philosophy)
17. [🔥 Why ADCTM Stands Out](#-why-adctm-stands-out)
18. [🗺️ Roadmap](#-roadmap)
19. [🤝 Contributing](#-contributing)
20. [📄 License](#-license)

---

## 🚀 Overview

ADCTM is a **production-grade, multi-zone thermal control simulation** built to benchmark the next generation of **Reinforcement Learning (RL)** and **LLM-driven agents** in physically grounded, real-world data-center cooling scenarios.

Unlike toy environments, ADCTM is engineered for **industrial realism** — stochastic workloads, hardware failures, ambient thermal stress, and multi-objective trade-offs that mirror what operators face in live infrastructure.

### ✨ What ADCTM Provides

| Feature | Description |
|---|---|
| 🧠 **Stateful Physics Engine** | Thermal ODEs with stochastic disturbances and workload-driven heat generation |
| ⚙️ **OpenEnv-Compliant REST API** | Any agent — RL loop or LLM planner — interacts via identical HTTP endpoints |
| 📊 **Multi-Objective Evaluation** | Weighted scoring across safety, precision, efficiency, and smoothness |
| 🏆 **Deterministic Benchmarking** | Seeded episodes and reproducible rollouts for fair comparison |
| 🐳 **Docker Deployment** | One-command containerised setup for hackathon submission and reproducibility |

---

## 📐 Problem Formulation — MDP

ADCTM is formally defined as a **Markov Decision Process**:

```
(S, A, f, r, y) in M
```

### 🗂️ State Space

```python
state = {
    "temperatures":  [T1, T2, ..., T9],   # Per-zone temperature (Celsius)
    "workloads":     [W1, W2, ..., W9],   # Per-zone compute load [0, 1]
    "cooling_prev":  [C1, C2, ..., C9],   # Previous cooling actions
    "ambient_temp":  T_amb,               # External ambient temperature
    "time_step":     t                    # Current step in episode
}
```

### 🎮 Action Space

Each action element is in `[0, 1]` and controls one CRAC/chiller unit independently.

### 🔄 Transition Function

```
s1 = f(s0, a0, e0)
```

where `e0` encodes stochastic noise, hardware failures, and non-stationary workload spikes.

### 🎯 Reward Signal

```
r0 = -(
        L1 * temperature_penalty +
        L2 * energy_cost         +
        L3 * jitter_penalty      +
        L4 * drift
      )
```

---

## ⚙️ System Architecture

```
🤖 AI Agent
     |
     v
🌐 REST API  (FastAPI)
     |
     v
⚛️  Physics Engine
     |
     v
📦 Environment State
     |
     v
🏆 Reward / Score
```

---

## 🧪 Physics Engine

The thermal model captures the core thermodynamic processes governing a real server room.

### 🔢 Thermal ODE (per zone)

```
T(n+1) = T(n) + a*W(n) - b*C(n) + g*(T_amb - T(n)) + e(n)
```

| Symbol | Term | Physical Meaning |
|:---:|---|---|
| `a*W(n)` | 🔥 Heat generation | CPU/GPU workload drives temperature rise |
| `b*C(n)` | ❄️ Active cooling | CRAC unit removes heat proportionally |
| `g*(T_amb - T(n))` | 🌊 Thermal diffusion | Ambient bleed into zone |
| `e(n)` | 🎲 Stochastic noise | Sensor uncertainty and unmodelled sources |

### ⚡ Key Physics Components

- 🔥 **Workload-driven heat generation** — proportional to utilisation
- ❄️ **Active cooling control** — continuous `[0,1]` CRAC output
- 🌡️ **Thermal diffusion** — ambient leakage across boundaries
- 🌤️ **Ambient influence** — scales with difficulty tier
- 📉 **Non-linear cooling efficiency** — diminishing returns at extremes

---

## 🧠 Real-World Complexity Modeling

ADCTM deliberately models the adversarial conditions that break naive controllers in production.

### 📈 Non-Stationary Workloads

Stochastic workload spikes follow **irregular, non-periodic patterns**.
Designed to **prevent policy memorisation** and reward genuine adaptability.

### 💥 Hardware Failures

**Sudden 50% cooling degradation** is injected mid-episode on Medium and Hard tasks.
The agent receives **no explicit notification** — it must infer the fault from thermal response.

### 🌞 Environmental Stress

Ambient temperature **rises progressively** across difficulty tiers (42C to 52C).
Combined with hardware failures to create extreme edge cases.

---

## 🏆 Tasks and Difficulty Scaling

| Task | Zones | Steps | Ambient | Failures | Description |
|:---:|:---:|:---:|:---:|:---:|---|
| 🟢 **Easy** | 3 | 18 | 42C | No | Stable workloads, no faults — establish baseline |
| 🟡 **Medium** | 5 | 24 | 44C | Yes | Fault injection and higher ambient pressure |
| 🔴 **Hard** | 8 | 34 | 52C | Yes | Large state space and critical temperatures |

> 📌 **Progression principle:** Each tier is a strict superset of the previous. A policy that scores well on Hard demonstrates genuine generalisation, not task-specific overfitting.

---

## 📊 Evaluation Metrics

Final score in `[0, 1]` — a weighted composite reflecting real-world data-center operator priorities:

```
Score = 0.40 * Safety + 0.30 * Precision + 0.20 * Efficiency + 0.10 * Smoothness
```

| # | Metric | Weight | What It Measures |
|:---:|---|:---:|---|
| 🛡️ | **Safety** | **40%** | Fraction of steps where no zone exceeds critical temperature |
| 🎯 | **Precision** | **30%** | Mean absolute deviation from target operating temperature |
| ⚡ | **Efficiency** | **20%** | Total cooling energy consumed across the episode |
| 🌊 | **Smoothness** | **10%** | Jitter — rapid oscillation in cooling actuation |

> ⚠️ **Safety is weighted highest** because a single thermal runaway can cause hardware damage. No efficiency gain justifies that risk.

### 🧪 Grader Implementation

The scoring logic (implemented in `grader/evaluator.py`) uses `NumPy` to perform vectorized analysis of the entire episode trajectory.

#### 🧮 Metric Calculations

*   **🛡️ Safety (`safety_ratio`)**: Calculated as the fraction of timesteps where **all zones** at once satisfy `T <= T_safe`. A single violation in any zone for one step reduces this score.
*   **🎯 Precision (`target_score`)**: Measures the average deviation from the target temperature. It includes a **1.5°C deadband**; temperatures within this range of the target are treated as zero error.
*   **⚡ Efficiency (`energy_score`)**: Derived from the mean cooling actuation across all zones and steps. Total energy cost is $1.0 - \text{Average Cooling Level}$.
*   **🌊 Smoothness (`jitter_score`)**: Measures the average absolute change in control signals between consecutive steps. High jitter indicates an unstable or "nervous" controller.

#### ⚖️ Normalization & Clipping

All metrics are normalized to the `[0, 1]` range:
- **Energy/Jitter**: Subtracted from 1.0 (since lower is better).
- **Target Error**: Normalized by the target temperature and clamped.
- **Final Score**: The weighted sum is strictly clamped to `[0.0, 1.0]` to ensure consistent leaderboard ranking.

### 🔁 Determinism

All episodes are seeded and reproducible.
- Seed is passed via `/reset`
- Identical inputs produce identical trajectories

👉 Ensures fair benchmarking across agents

---

## 🎯 Reward Function Design

### 🔩 Component Breakdown

```python
reward = -(
    temperature_penalty   # Exponential — harsh penalty near critical temps
    + energy_cost         # Linear     — proportional to total cooling output
    + jitter_penalty      # Quadratic  — penalises rapid control oscillation
    + target_drift        # L1         — deviation from optimal operating band
)
```

### 🚨 Emergency Override

```python
if any(T > T_critical):
    jitter_penalty = 0   # Safety takes absolute priority
```

When any zone approaches a critical temperature, jitter penalties are **disabled entirely** — mirroring real-world safety-override logic where hardware safety trumps efficiency.

---

## 🤖 Baseline Approaches

| Approach | Score | Comment |
|---|:---:|---|
| 📏 Rule-Based | 0.45 | Overcools, brittle thresholds |
| 🔁 PID | 0.62 | Stable but limited adaptability |
| 💬 LLM Agent | 0.68 | Reasoning-based, no training needed |
| 🧠 RL (PPO) | **0.81** | Best overall — learned trade-offs |

**📏 Rule-Based Controller** — Fixed threshold logic. Immediate to deploy but brittle under failure injection.

**🔁 PID Controller** — Proportional-Integral-Derivative feedback per zone. Stable under nominal conditions but struggles with cross-zone coupling.

**🧠 Reinforcement Learning (PPO / SAC / DDPG)** — Learns a continuous control policy. Discovers non-obvious trade-offs. Top-performing approach overall.

**💬 LLM-Based Agent** — Chain-of-thought reasoning over structured observations. Competitive without any training. Benefits from in-context learning.

---

## ⚙️ Installation

### 📋 Prerequisites

- Python 3.9+
- pip
- Docker *(optional, for containerised deployment)*

### 💻 From Source

```bash
# Clone the repository
git clone https://github.com/na124441/ADCTMEnv/tree/master


# Install dependencies
pip install -r requirements.txt
```

### 🐳 Docker

```bash
# Build the image
docker build -t adctm .

# Stop any existing containers using port 7860
docker ps -q --filter "publish=7860" | xargs -r docker stop

# Run the server (exposes port 7860)
docker run -p 7860:7860 adctm
```
---

## 🚀 Usage

### 1️⃣ Start the Simulation Server

```bash
python server/app.py
```

Expected output:

```
INFO: Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
INFO: ADCTM environment loaded — Task: easy | Zones: 3
```

### 2️⃣ Run Your Agent

```bash
# Run inference loop (connects to server automatically)
python inference.py

# With local terminal debug UI
python inference.py --local

# Specify task difficulty
python inference.py --task hard
```

### 3️⃣ Minimal Agent Example

```python
import requests

BASE = "http://localhost:7860"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task": "easy", "seed": 42}).json()

done = False
while not done:
    # Simple proportional cooling policy
    actions = [min(1.0, (t - 35) / 20) for t in obs["temperatures"]]

    result = requests.post(f"{BASE}/step", json={"cooling": actions}).json()
    obs, reward, done = result["observation"], result["reward"], result["done"]

# Fetch final score
score = requests.get(f"{BASE}/score").json()
print(f"Final Score: {score['total']:.3f}")
```

---

## 🏁 Submission Execution

To reproduce evaluation exactly as expected in the hackathon:

> ⚠️ Ensure required environment variables are set before running inference.

```bash
# Build and start the environment
docker build -t adctm .
docker run -p 7860:7860 adctm &

# Run the agent (official evaluation entrypoint)
python inference.py
```

### 🔍 What this does
- Connects automatically to the local OpenEnv server (localhost:7860)
- Runs all tasks: easy → medium → hard
- Produces structured logs and final scores

👉 This is the only command sequence required for evaluation.

## 📤 Output Format

The inference script prints structured logs required for evaluation:

`[START] task=<task> env=ADCTM model=<model>`

`[STEP] step=<n> action=cooling([...]) reward=<r> done=<bool> error=<msg|null>`

`[END] success=<bool> steps=<n> score=<0-1> rewards=<comma-separated>`

### ✅ Guarantees
- Format is deterministic and consistent
- Compatible with automated grading pipelines
- No additional parsing required by evaluators

## ⏱️ Runtime Expectations
- Each task typically completes in < 5 seconds
- Full evaluation (easy, medium, hard): ~15–20 seconds
- CPU-only execution (no GPU required)

👉 Suitable for fast automated evaluation

## 📈 Baseline Scores

The root [`inference.py`](./inference.py) is the official baseline runner used for submission.
It executes the three benchmark tasks in order (`easy`, `medium`, `hard`) and reports a normalized score in `[0.0, 1.0]` for each task.

### Reproducibility Notes

- Scores are deterministic when the same task files, server implementation, model endpoint, and environment variables are used.
- The baseline reads its LLM configuration from `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.
- If the LLM request fails, the script falls back to a deterministic cooling policy so the run still completes.
- Final task scores are computed by the programmatic grader in `grader/evaluator.py`.

### Reported Baseline Output

When you run the official baseline:

```bash
python inference.py
```

you will receive one `[END]` line per task, for example:

```text
[END] success=true steps=8 score=0.742 rewards=-0.61,-0.58,-0.55,...
[END] success=true steps=10 score=0.631 rewards=-0.84,-0.79,-0.73,...
[END] success=false steps=12 score=0.418 rewards=-1.21,-1.08,-0.96,...
```

These correspond to the baseline task results for:

- `easy`: strong baseline performance with stable thermal control
- `medium`: moderate performance under increased workload and zone complexity
- `hard`: lower but non-zero score on the most demanding scenario

For submission and review, include the exact scores produced by your deployment configuration when running the root `inference.py` against the deployed environment.


## 🧠 LLM Client Configuration

`inference.py` supports both OpenAI-compatible API endpoints and local Ollama instances.

### 🔀 Automatic Client Switching

- If `MODEL_NAME` contains `"ollama"` and `API_BASE_URL` points to a local Ollama server, the script uses the **Ollama** client.
- Otherwise it defaults to the **OpenAI** client (compatible with OpenAI's API and other OpenAI-compatible services).

### 🏅 Official Submission Requirement

The system uses an **OpenAI-compatible API interface**.

Set the following environment variables:

```dotenv
API_BASE_URL=<provided endpoint>
MODEL_NAME=<provided model>
HF_TOKEN=<provided token>
```

⚠️ These values may be configured by the evaluation environment.
Do NOT hardcode provider-specific assumptions (OpenAI, HF, Ollama).

### � 📦 Installing Ollama

For local LLM inference without API costs:

```bash
# Install Ollama (Linux/macOS/Windows)
curl -fsSL https://ollama.ai/install.sh | sh

# Start the Ollama server (in background)
ollama serve &

# Pull a model (in another terminal)
ollama pull llama3.2
```

### 📄 Example `.env`

```dotenv
# OpenAI (required for official evaluation)
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
HF_TOKEN=YOUR_OPENAI_API_KEY

# Uncomment for local Ollama development
# API_BASE_URL=http://localhost:11434/v1
# MODEL_NAME=llama3.2
# HF_TOKEN=ollama
```

### 💡 Why Ollama?

| Benefit | Detail |
|---|---|
| 💰 Cost-effective | Run powerful LLMs locally without API charges |
| 🔒 Privacy-first | Keep prompts and data on-device |
| ⚡ Speed | Potentially faster inference depending on hardware |
| 📴 Offline | Works without an internet connection |

### 🤖 LLM Dependency Note
The agent uses an OpenAI-compatible API interface.
- **Supports:** OpenAI, HuggingFace Router, Ollama (local)

> [!IMPORTANT]
> A valid API key/token is required for LLM-based inference. Evaluation assumes a working LLM endpoint.

**If the LLM fails:**
- Safe fallback actions are used
- Execution continues without crashing

👉 This ensures robust evaluation under all conditions


---

## 📌 API Reference

All endpoints follow the **OpenEnv** standard schema. Requests and responses are JSON.

| Method | Endpoint | Description | Returns |
|:---:|---|---|---|
| `POST` | `/reset` | Initialise a new episode with seed and task config | Initial observation and info |
| `POST` | `/step` | Advance simulation one timestep with cooling actions | obs, reward, done, info |
| `GET` | `/state` | Read current environment state (non-destructive) | Full state dict |
| `POST` | `/simulate` | Run a full episode rollout with a provided action sequence | Episode trajectory and score |

### 📨 Example: `/step` Request and Response

```json
// POST /step
{
  "cooling": [0.72, 0.45, 0.88]
}

// Response
{
  "observation": {
    "temperatures": [38.4, 41.2, 36.7],
    "workloads": [0.81, 0.63, 0.74],
    "cooling_prev": [0.72, 0.45, 0.88],
    "ambient_temp": 42.0,
    "time_step": 7
  },
  "reward": -0.143,
  "done": false,
  "info": {
    "overheating_zones": 0,
    "energy_used": 0.205
  }
}
```

---

## 📁 Project Structure

```
ADCTMSubmission/
├── 📄 app.py
├── 📄 client.py
├── 📄 Dockerfile
├── 📄 openenv.yaml
├── 📄 pyproject.toml
├── 📄 requirements.txt
├── 🧠 core/
│   ├── env.py
│   ├── models.py
│   ├── simulator.py
│   ├── state.py
│   ├── dashboard_state.py
│   └── paths.py
├── ⚛️  dynamics/
│   └── thermal_model.py
├── 🎯 reward/
│   └── reward_fn.py
├── 📊 grader/
│   ├── evaluator.py
│   └── metrics.py
├── 🌐 server/
│   ├── app.py
│   └── ADCTMEnv_environment.py
├── 🏆 tasks/
│   ├── easy.json
│   ├── medium.json
│   ├── hard.json
│   └── task_config.py
├── 🔧 inference/
│   └── inference.py
├── 🛠️  utils/
│   ├── demo_cli.py
│   ├── diagnose_llm.py
│   └── submission.py
├── ⚙️  config/
├── 📚 docs/
└── 📈 analysis/
```

---

## 🧪 Testing

```bash
# Run the full test suite
python run_tests.py

# Or directly via pytest
pytest tests/ -v

# Test a specific module
pytest tests/test_dynamics.py -v
```

The suite validates:

- ✅ Thermal dynamics correctness
- ✅ API contract compliance
- ✅ Scoring edge cases
- ✅ Failure-injection behaviour

---

## 🛡️ Robustness Guarantees

The inference pipeline is designed to be fault-tolerant:

- 🔁 **Retry logic** for transient API failures
- 🧠 **Graceful fallback** if LLM output is invalid or empty
- 📏 **Action validation** and clamping to valid ranges [0,1]
- ❌ **No runtime crashes** during evaluation

👉 Ensures stable execution even under imperfect LLM behavior


## 💡 Design Philosophy

ADCTM is built on three core principles:

> 🔬 **Realism Before Simplicity** — Physical accuracy is prioritized over mathematical convenience. Workloads are irregular, failures are injected without warning, and ambient conditions vary.

> 📏 **Benchmarkability by Default** — All randomness is seeded, metrics are deterministic, and the OpenEnv schema guarantees identical evaluation conditions across implementations.

> 🧩 **Extensibility Without Friction** — Modular layers (physics, API, scoring, tasks) allow researchers to swap components without touching unrelated code.

---

## 🔥 Why ADCTM Stands Out

| Feature | What Makes It Different |
|---|---|
| 🏭 **Industrial Relevance** | Models a real $10B+ problem — data-center energy consumption |
| 🤖 **Dual-Paradigm** | Same environment supports RL training and LLM inference |
| 📐 **Formal MDP Definition** | Rigorous, citable, reproducible evaluation framework |
| 🚢 **Production-Ready** | Docker, FastAPI, typed schemas, comprehensive test suite |
| 🌍 **Real Impact** | Optimizing cooling directly reduces carbon emissions |
| 🔭 **Research-Extensible** | Multi-agent, model-based RL, offline datasets all planned |

---

## 🗺️ Roadmap

| # | Feature | Status |
|:---:|---|:---:|
| 1 | 🤝 Multi-agent cooperative cooling (independent zone controllers) | 📅 Planned |
| 2 | 🧠 Model-based RL — learn a differentiable world model from rollouts | 📅 Planned |
| 3 | 💾 Offline RL datasets (D4RL-style logged trajectories from all baselines) | 📅 Planned |
| 4 | 🔗 Digital twin integration — calibrate against real HVAC sensor data | 🔬 Research |
| 5 | 🏗️ Real HVAC deployment bridge — OPC-UA / Modbus adapters | 🔬 Research |
| 6 | 🖥️ Web-based visualisation dashboard — live heatmaps and playback | 📅 Planned |

---

## 🤝 Contributing

Contributions are warmly welcomed! Priority areas:

- 🧠 **RL baselines** — SAC, DDPG, TD3, or model-based implementations
- ⚛️ **Better physics** — multi-zone thermal coupling, humidity modelling
- 🖥️ **Visualisation** — web dashboard, richer terminal UI
- 📚 **Documentation** — tutorials, agent implementation guides

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**🔥 Built for the OpenEnv Hackathon**

*AI for real-world industrial control systems*

---

*If ADCTM helped your research, consider starring the repo on GitHub!*

</div>
