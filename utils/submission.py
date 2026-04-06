#!/usr/bin/env python3
"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘          OpenEnv Competition Deep Evaluator  v2.0                           â•‘
â•‘                                                                              â•‘
â•‘  A standalone, plug-and-play submission analysis tool.                       â•‘
â•‘  Point it at any OpenEnv project directory and it performs:                  â•‘
â•‘                                                                              â•‘
â•‘   Phase 1 â€” Real submission-gate compliance checks (auto-disqualification)  â•‘
â•‘   Phase 2 â€” Deep LLM analysis via Ollama (gpt-oss:120b-cloud)               â•‘
â•‘   Phase 3 â€” Mock judge panel scoring all 5 official criteria                 â•‘
â•‘   Phase 4 â€” Deep Implementation Plan & Problem List                          â•‘
â•‘   Phase 5 â€” Final composite grade + action plan                              â•‘
â•‘                                                                              â•‘
â•‘  Usage:  python evaluator.py [/path/to/project]                              â•‘
â•‘  Deps:   pip install ollama rich pydantic pyyaml                             â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
"""

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import ollama
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.markdown import Markdown

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# GLOBALS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

MODEL = "qwen3-coder:480b-cloud"
CONSOLE = Console(highlight=True)
SEP = Rule(style="dim")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DATA CLASSES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class FileInspection:
    path: Path
    exists: bool
    size_bytes: int = 0
    lines: int = 0
    syntax_ok: Optional[bool] = None  # Python files only
    syntax_error: str = ""
    ast_classes: List[str] = field(default_factory=list)
    ast_functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    content_preview: str = ""
    full_content: str = ""


@dataclass
class GateCheck:
    id: str
    name: str
    category: str
    passed: bool
    score: float
    max_score: float
    severity: str  # "DISQUALIFY" | "CRITICAL" | "MAJOR" | "MINOR"
    evidence: str = ""
    detail: List[str] = field(default_factory=list)


@dataclass
class PhaseOneReport:
    gates: List[GateCheck] = field(default_factory=list)
    total_score: float = 0
    max_score: float = 0
    disqualified: bool = False
    disq_reasons: List[str] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    project_stats: Dict = field(default_factory=dict)


@dataclass
class LLMSection:
    title: str
    content: str
    score_hint: Optional[float] = None  # 0â€“100 if the section produces a score


@dataclass
class PhaseTwoReport:
    sections: List[LLMSection] = field(default_factory=list)
    raw_evidence: str = ""


@dataclass
class JudgeCriterion:
    name: str
    weight: int
    raw_score: float  # 0â€“100
    weighted: float
    sub_scores: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class PhaseThreeReport:
    criteria: List[JudgeCriterion] = field(default_factory=list)
    total_weighted: float = 0
    phase1_prediction: str = ""
    phase2_prediction: str = ""
    phase3_prediction: str = ""
    head_verdict: str = ""
    action_plan: List[str] = field(default_factory=list)


@dataclass
class PhaseFourReport:
    final_problems: List[str] = field(default_factory=list)
    detailed_plan: str = ""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# UTILITIES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def read_file(path: Path, max_bytes: int = 500_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except Exception:
        return ""


def inspect_python_file(path: Path) -> FileInspection:
    insp = FileInspection(path=path, exists=path.exists())
    if not insp.exists:
        return insp

    content = read_file(path)
    insp.full_content = content
    insp.content_preview = content[:500]
    insp.size_bytes = path.stat().st_size
    insp.lines = content.count("\n") + 1

    # Syntax check via py_compile
    try:
        compile(content, str(path), "exec")
        insp.syntax_ok = True
    except SyntaxError as e:
        insp.syntax_ok = False
        insp.syntax_error = str(e)

    # AST analysis
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                insp.ast_classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                insp.ast_functions.append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    insp.imports += [a.name for a in node.names]
                else:
                    insp.imports.append(node.module or "")
    except Exception:
        pass

    return insp


def run_subprocess(cmd: List[str], cwd: Path, timeout: int = 30) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError:
        return -2, "", f"Command not found: {cmd[0]}"
    except Exception as e:
        return -3, "", str(e)


def llm_call(system: str, user: str, max_tokens: int = 2000) -> str:
    """Single Ollama call with retry. Increased max_tokens to prevent truncation."""
    for attempt in range(3):
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0.2, "num_predict": max_tokens},
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            if attempt == 2:
                return f"[LLM_ERROR: {e}]"
            time.sleep(2)
    return "[LLM_ERROR: max retries]"


def extract_json_from_llm(text: str) -> dict:
    # Strip markdown fences
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # Find first { ... }
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PHASE 1 â€” DEEP SUBMISSION GATE CHECKS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SubmissionGateChecker:
    """
    Performs every check from the official Pre-Submission Checklist
    plus deep code analysis. Checks are grouped by severity.
    """

    def __init__(self, project_dir: Path):
        self.d = project_dir
        self.gates: List[GateCheck] = []
        self.stats: Dict[str, Any] = {}

    # â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _add(self, id, name, cat, passed, score, max_score, severity, evidence="", detail=None):
        self.gates.append(GateCheck(
            id=id, name=name, category=cat, passed=passed,
            score=score, max_score=max_score, severity=severity,
            evidence=evidence, detail=detail or [],
        ))

    def _find_files(self, pattern: str) -> List[Path]:
        return sorted(self.d.rglob(pattern))

    # â”€â”€ CATEGORY A: Mandatory File Presence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_mandatory_files(self):
        CONSOLE.print("  [dim]Checking mandatory files...[/dim]")
        mandatory = [
            ("openenv.yaml", "openenv.yaml manifest", "DISQUALIFY", 15),
            ("Dockerfile", "Dockerfile", "DISQUALIFY", 15),
            ("inference.py", "inference.py at root", "DISQUALIFY", 15),
            ("README.md", "README.md documentation", "CRITICAL", 8),
        ]
        for fname, label, sev, pts in mandatory:
            fp = self.d / fname
            self._add(
                id=f"file_{fname}", name=f"File exists: {fname}",
                cat="Mandatory Files", passed=fp.exists(),
                score=pts if fp.exists() else 0, max_score=pts,
                severity=sev,
                evidence=str(fp) if fp.exists() else f"MISSING: {fp}",
            )

    # â”€â”€ CATEGORY B: openenv.yaml deep validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_openenv_yaml(self):
        CONSOLE.print("  [dim]Parsing openenv.yaml...[/dim]")
        yp = self.d / "openenv.yaml"
        if not yp.exists():
            self._add("yaml_parse", "openenv.yaml parseable", "Manifest",
                      False, 0, 20, "DISQUALIFY", "File not found")
            return

        try:
            data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            self._add("yaml_parse", "openenv.yaml parseable", "Manifest",
                      False, 0, 20, "DISQUALIFY", f"YAML parse error: {e}")
            return
        except Exception as e:
            self._add("yaml_parse", "openenv.yaml parseable", "Manifest",
                      False, 0, 20, "DISQUALIFY", f"File read error: {e}")
            return

        self._add("yaml_parse", "openenv.yaml parseable", "Manifest",
                  True, 5, 5, "CRITICAL", "Parsed successfully")

        required_top = ["name", "version", "description"]
        for key in required_top:
            present = key in data and bool(data[key])
            self._add(f"yaml_{key}", f"yaml has '{key}'", "Manifest",
                      present, 3 if present else 0, 3, "MAJOR",
                      f"value={data.get(key, 'MISSING')!r}")

        # tasks key
        tasks_val = data.get("tasks", None)
        has_tasks = isinstance(tasks_val, (list, dict)) and len(tasks_val) >= 1
        self._add("yaml_tasks", "yaml has 'tasks' with entries", "Manifest",
                  has_tasks, 5 if has_tasks else 0, 5, "DISQUALIFY",
                  f"tasks={tasks_val}")

        # Extra useful keys
        for key in ["author", "tags", "base_url", "env_id"]:
            present = key in data
            self._add(f"yaml_opt_{key}", f"yaml has optional '{key}'", "Manifest",
                      present, 1 if present else 0, 1, "MINOR",
                      f"{'present' if present else 'missing'}")

        self.stats["openenv_yaml"] = data

    # â”€â”€ CATEGORY C: OpenEnv API Compliance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_openenv_api(self):
        CONSOLE.print("  [dim]Checking OpenEnv API (step/reset/state)...[/dim]")
        env_candidates = (
                self._find_files("env.py") +
                self._find_files("environment.py") +
                self._find_files("*_env.py")
        )
        # Also scan server/app.py for the REST endpoints
        server_candidates = self._find_files("app.py")

        def_step = def_reset = def_state = False
        step_returns_obs = step_returns_reward = False
        reset_returns_obs = False
        state_returns = False
        endpoint_step = endpoint_reset = False
        async_support = False

        for fp in env_candidates + server_candidates:
            content = read_file(fp)
            insp = inspect_python_file(fp)

            if "def step" in content or "async def step" in content:
                def_step = True
            if "def reset" in content or "async def reset" in content:
                def_reset = True
            if "def state" in content or "async def state" in content:
                def_state = True
            if "async def" in content:
                async_support = True

            # Check return values in step
            if def_step and ("reward" in content) and ("observation" in content or "obs" in content):
                step_returns_obs = step_returns_reward = True
            if def_reset and ("observation" in content or "obs" in content):
                reset_returns_obs = True
            if def_state:
                state_returns = True

            # REST endpoints
            if "@app.post" in content or "@router.post" in content:
                if "/step" in content:
                    endpoint_step = True
                if "/reset" in content:
                    endpoint_reset = True

        self._add("api_step", "env defines step()", "OpenEnv API", def_step,
                  10 if def_step else 0, 10, "DISQUALIFY",
                  f"Found in {len(env_candidates)} env candidates")
        self._add("api_reset", "env defines reset()", "OpenEnv API", def_reset,
                  10 if def_reset else 0, 10, "DISQUALIFY")
        self._add("api_state", "env defines state()", "OpenEnv API", def_state,
                  5 if def_state else 0, 5, "MAJOR")
        self._add("api_step_returns", "step() returns obs+reward+done+info", "OpenEnv API",
                  step_returns_obs and step_returns_reward,
                  6 if (step_returns_obs and step_returns_reward) else 0, 6, "CRITICAL")
        self._add("api_reset_returns", "reset() returns observation", "OpenEnv API",
                  reset_returns_obs, 4 if reset_returns_obs else 0, 4, "CRITICAL")
        self._add("api_rest_step", "POST /step endpoint", "OpenEnv API",
                  endpoint_step, 4 if endpoint_step else 0, 4, "MAJOR")
        self._add("api_rest_reset", "POST /reset endpoint", "OpenEnv API",
                  endpoint_reset, 4 if endpoint_reset else 0, 4, "MAJOR")
        self._add("api_async", "Async API support", "OpenEnv API",
                  async_support, 2 if async_support else 0, 2, "MINOR")

    # â”€â”€ CATEGORY D: Typed Pydantic Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_typed_models(self):
        CONSOLE.print("  [dim]Checking Pydantic models...[/dim]")
        models_files = self._find_files("models.py")
        all_classes = []
        has_basemodel = has_obs = has_act = has_rew = False
        validator_count = 0
        field_count = 0

        for mp in models_files:
            insp = inspect_python_file(mp)
            all_classes += insp.ast_classes
            content = insp.full_content

            if "BaseModel" in content and "pydantic" in content.lower():
                has_basemodel = True
            if any(c for c in insp.ast_classes if "observation" in c.lower() or "obs" in c.lower()):
                has_obs = True
            if any(c for c in insp.ast_classes if "action" in c.lower() or "act" in c.lower()):
                has_act = True
            if any(c for c in insp.ast_classes if "reward" in c.lower()):
                has_rew = True
            validator_count += content.count("@validator") + content.count("@field_validator")
            field_count += content.count("Field(")

        self.stats["pydantic_classes"] = all_classes

        self._add("models_pydantic", "Uses Pydantic BaseModel", "Typed Models",
                  has_basemodel, 8 if has_basemodel else 0, 8, "DISQUALIFY",
                  f"Files checked: {[str(p.relative_to(self.d)) for p in models_files]}")
        self._add("models_obs", "Observation model defined", "Typed Models",
                  has_obs, 6 if has_obs else 0, 6, "DISQUALIFY",
                  f"Classes found: {all_classes}")
        self._add("models_act", "Action model defined", "Typed Models",
                  has_act, 6 if has_act else 0, 6, "DISQUALIFY")
        self._add("models_rew", "Reward model defined", "Typed Models",
                  has_rew, 4 if has_rew else 0, 4, "MAJOR")
        self._add("models_validators", "Has field validators", "Typed Models",
                  validator_count > 0, min(validator_count, 3), 3, "MINOR",
                  f"Validators found: {validator_count}, Field() calls: {field_count}")

    # â”€â”€ CATEGORY E: Task Files and Graders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_tasks_and_graders(self):
        CONSOLE.print("  [dim]Checking tasks and graders...[/dim]")
        tasks_dir = self.d / "tasks"
        task_files = []
        if tasks_dir.exists():
            task_files = list(tasks_dir.glob("*.json")) + list(tasks_dir.glob("*.yaml"))

        # Also look anywhere
        all_task_files = self._find_files("easy.json") + self._find_files("easy.yaml") + \
                         self._find_files("medium.json") + self._find_files("medium.yaml") + \
                         self._find_files("hard.json") + self._find_files("hard.yaml")

        has_easy = bool(self._find_files("easy.json") + self._find_files("easy.yaml"))
        has_medium = bool(self._find_files("medium.json") + self._find_files("medium.yaml"))
        has_hard = bool(self._find_files("hard.json") + self._find_files("hard.yaml"))
        count = sum([has_easy, has_medium, has_hard])

        self._add("tasks_3plus", "3+ task difficulty levels (easy/medium/hard)", "Tasks",
                  count >= 3, min(count * 5, 15), 15, "DISQUALIFY",
                  f"easy={has_easy}, medium={has_medium}, hard={has_hard}")

        # Inspect task JSON content
        task_details = []
        for tf in all_task_files[:3]:
            try:
                data = json.loads(tf.read_text(encoding="utf-8"))
                keys = list(data.keys()) if isinstance(data, dict) else ["(list)"]
                task_details.append(f"{tf.name}: keys={keys}")
            except Exception as e:
                task_details.append(f"{tf.name}: parse error â€” {e}")

        has_seeds = any("seed" in s for s in task_details)
        has_zones = any("zone" in s or "target" in s for s in task_details)

        self._add("tasks_structure", "Task files have valid structure", "Tasks",
                  len(task_details) >= 2, 4 if len(task_details) >= 2 else 0, 4, "MAJOR",
                  "\n".join(task_details))
        self._add("tasks_seeds", "Task files include random seeds", "Tasks",
                  has_seeds, 3 if has_seeds else 0, 3, "MINOR")

        # Grader checks
        grader_files = self._find_files("evaluator.py") + self._find_files("grader.py") + \
                       self._find_files("metrics.py")
        has_grader = len(grader_files) > 0

        self._add("grader_exists", "Grader module exists", "Tasks",
                  has_grader, 8 if has_grader else 0, 8, "DISQUALIFY",
                  f"Files: {[str(p.relative_to(self.d)) for p in grader_files]}")

        # Check grader can produce varied scores (not always same)
        score_varies = False
        grader_score_range = False
        for gf in grader_files:
            content = read_file(gf)
            # Look for 0.0 and 1.0 or floating point score logic
            if re.search(r"0\.0|0\.5|1\.0|score\s*[+\-\*]|partial|weight", content):
                score_varies = True
            if "0.0" in content and "1.0" in content:
                grader_score_range = True

        self._add("grader_varies", "Grader produces varied scores (not always same)", "Tasks",
                  score_varies, 6 if score_varies else 0, 6, "DISQUALIFY",
                  "Graders that always return same score â†’ disqualified")
        self._add("grader_range", "Grader explicitly uses 0.0â€“1.0 range", "Tasks",
                  grader_score_range, 4 if grader_score_range else 0, 4, "MAJOR")

        self.stats["task_details"] = task_details

    # â”€â”€ CATEGORY F: Inference Script Deep Check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_inference_script(self):
        CONSOLE.print("  [dim]Deep-checking inference.py...[/dim]")
        inf_path = self.d / "inference.py"
        if not inf_path.exists():
            self._add("inf_exists", "inference.py at root", "Inference Script",
                      False, 0, 40, "DISQUALIFY")
            return

        insp = inspect_python_file(inf_path)
        content = insp.full_content

        # Syntax
        self._add("inf_syntax", "inference.py syntax valid", "Inference Script",
                  insp.syntax_ok is not False,
                  5 if insp.syntax_ok is not False else 0, 5, "DISQUALIFY",
                  insp.syntax_error or "OK")

        # Mandatory env vars (must read from os.getenv, not hardcoded)
        for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
            found = f'os.getenv("{var}")' in content or f"os.getenv('{var}')" in content
            hardcoded = re.search(rf'"{var}"\s*=\s*"[^"{{]', content)
            passed = found and not hardcoded
            self._add(f"inf_env_{var}", f"Reads {var} from os.getenv()", "Inference Script",
                      passed, 3 if passed else 0, 3, "CRITICAL",
                      f"Found getenv={found}, hardcoded risk={bool(hardcoded)}")

        # OpenAI client
        uses_openai = "OpenAI(" in content or "from openai" in content
        self._add("inf_openai_client", "Uses OpenAI client class", "Inference Script",
                  uses_openai, 4 if uses_openai else 0, 4, "DISQUALIFY")

        # Exact [START] log format check
        # Must emit: [START] task=<name> env=<benchmark> model=<model>
        start_pattern = re.search(r'\[START\].*task=.*env=.*model=', content)
        self._add("inf_log_start", "Emits [START] task=... env=... model=...", "Inference Script",
                  bool(start_pattern), 4 if start_pattern else 0, 4, "DISQUALIFY",
                  f"Pattern match: {bool(start_pattern)}")

        # [STEP] step=N action=... reward=... done=... error=...
        step_pattern = re.search(r'\[STEP\].*step=.*action=.*reward=.*done=.*error=', content)
        self._add("inf_log_step", "Emits [STEP] step=N action=... reward=... done=... error=...",
                  "Inference Script", bool(step_pattern),
                  4 if step_pattern else 0, 4, "DISQUALIFY")

        # [END] success=... steps=... rewards=...
        end_pattern = re.search(r'\[END\].*success=.*steps=.*reward', content)
        self._add("inf_log_end", "Emits [END] success=... steps=... rewards=...",
                  "Inference Script", bool(end_pattern),
                  4 if end_pattern else 0, 4, "DISQUALIFY")

        # reward formatted to 2 decimal places
        reward_fmt = ":.2f" in content or "{:.2f}" in content or "f'{" in content
        self._add("inf_reward_format", "Reward formatted to 2 decimal places (:.2f)",
                  "Inference Script", reward_fmt, 2 if reward_fmt else 0, 2, "MAJOR")

        # done/success lowercase booleans
        lower_bool = "str(done).lower()" in content or "str(success).lower()" in content
        self._add("inf_lower_bool", "done/success emitted as lowercase (true/false)",
                  "Inference Script", lower_bool, 2 if lower_bool else 0, 2, "MAJOR")

        # env.reset() and env.step() calls
        calls_reset = "env.reset()" in content or "await env.reset()" in content
        calls_step = "env.step(" in content or "await env.step(" in content
        calls_close = "env.close()" in content or "await env.close()" in content
        self._add("inf_env_reset", "Calls env.reset()", "Inference Script",
                  calls_reset, 3 if calls_reset else 0, 3, "DISQUALIFY")
        self._add("inf_env_step", "Calls env.step()", "Inference Script",
                  calls_step, 3 if calls_step else 0, 3, "DISQUALIFY")
        self._add("inf_env_close", "Calls env.close() in finally block", "Inference Script",
                  calls_close, 2 if calls_close else 0, 2, "MAJOR")

        # MAX_STEPS / timeout guard
        has_maxsteps = "MAX_STEPS" in content or "max_steps" in content
        self._add("inf_maxsteps", "Has MAX_STEPS guard (runtime < 20min)", "Inference Script",
                  has_maxsteps, 2 if has_maxsteps else 0, 2, "MAJOR")

        # AST stats
        self._add("inf_functions", f"Has {len(insp.ast_functions)} functions defined",
                  "Inference Script", len(insp.ast_functions) >= 3,
                  min(len(insp.ast_functions), 4), 4, "MINOR",
                  f"Functions: {insp.ast_functions}")

        self.stats["inference_lines"] = insp.lines
        self.stats["inference_classes"] = insp.ast_classes
        self.stats["inference_functions"] = insp.ast_functions

    # â”€â”€ CATEGORY G: Dockerfile Deep Validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_dockerfile(self):
        CONSOLE.print("  [dim]Analyzing Dockerfile...[/dim]")
        df_path = self.d / "Dockerfile"
        if not (df_path).exists():
            df_path = self.d / "server" / "Dockerfile"
        if not df_path.exists():
            self._add("docker_exists", "Dockerfile found", "Docker",
                      False, 0, 25, "DISQUALIFY")
            return

        content = read_file(df_path)
        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]

        # FROM
        from_line = next((l for l in lines if l.startswith("FROM")), None)
        uses_slim = from_line and ("slim" in from_line or "alpine" in from_line or "3.1" in from_line)
        self._add("docker_from", "Dockerfile has FROM instruction", "Docker",
                  bool(from_line), 5 if from_line else 0, 5, "DISQUALIFY",
                  from_line or "MISSING")

        # WORKDIR
        has_workdir = any(l.startswith("WORKDIR") for l in lines)
        self._add("docker_workdir", "Dockerfile has WORKDIR", "Docker",
                  has_workdir, 2 if has_workdir else 0, 2, "MAJOR")

        # COPY / ADD requirements/deps
        copy_lines = [l for l in lines if l.startswith("COPY") or l.startswith("ADD")]
        copies_requirements = any("requirements" in l or "pyproject" in l for l in copy_lines)
        self._add("docker_copy_req", "COPY requirements/pyproject.toml", "Docker",
                  copies_requirements, 3 if copies_requirements else 0, 3, "MAJOR",
                  str(copy_lines[:3]))

        # RUN pip install
        pip_install = any("pip install" in l for l in lines)
        self._add("docker_pip", "RUN pip install", "Docker",
                  pip_install, 3 if pip_install else 0, 3, "MAJOR")

        # EXPOSE port
        expose_lines = [l for l in lines if l.startswith("EXPOSE")]
        self._add("docker_expose", "EXPOSE port declared", "Docker",
                  bool(expose_lines), 2 if expose_lines else 0, 2, "MINOR",
                  str(expose_lines))

        # CMD / ENTRYPOINT
        has_cmd = any(l.startswith("CMD") or l.startswith("ENTRYPOINT") for l in lines)
        self._add("docker_cmd", "CMD or ENTRYPOINT defined", "Docker",
                  has_cmd, 5 if has_cmd else 0, 5, "DISQUALIFY")

        # Try actual docker build if docker is available
        rc, _, err = run_subprocess(["docker", "--version"], self.d, timeout=5)
        docker_available = rc == 0
        if docker_available:
            CONSOLE.print("  [dim]  â†’ Docker available, attempting build (dry-run check)...[/dim]")
            rc2, _, err2 = run_subprocess(
                ["docker", "build", "--no-cache", "--dry-run", "-f", str(df_path), "."],
                self.d, timeout=30,
            )
            # dry-run may not exist on all versions; just check if docker itself responds
            self._add("docker_build_attempt", "docker CLI responds (build attempted)", "Docker",
                      True, 3, 3, "MINOR", "Docker is installed and callable")
        else:
            self._add("docker_build_attempt", "docker CLI available (build check skipped)", "Docker",
                      False, 0, 3, "MINOR", "Docker not installed on evaluator machine")

        self.stats["dockerfile_lines"] = len(lines)
        self.stats["dockerfile_from"] = from_line

    # â”€â”€ CATEGORY H: Reward Function Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_reward_function(self):
        CONSOLE.print("  [dim]Analysing reward function...[/dim]")
        rwd_files = self._find_files("reward_fn.py") + self._find_files("reward*.py")
        if not rwd_files:
            # Try looking in any Python file
            rwd_files = self._find_files("*.py")

        has_reward_module = bool(self._find_files("reward_fn.py") + self._find_files("reward*.py"))
        self._add("reward_module", "Dedicated reward module exists", "Reward",
                  has_reward_module, 6 if has_reward_module else 0, 6, "MAJOR")

        dense_signal = partial_reward = penalty_logic = False
        reward_formula_found = False

        for rf in rwd_files[:5]:
            content = read_file(rf)
            # Dense: reward changes at each step, not just binary end
            if re.search(r"reward\s*[+\-]=|step_reward|partial|incremental|timestep", content, re.I):
                dense_signal = True
            if re.search(r"partial|progress|fraction|0\.\d+|ratio", content, re.I):
                partial_reward = True
            if re.search(r"penalty|punish|negative|clip|clamp|\-\d+\.?\d*\s*\*", content, re.I):
                penalty_logic = True
            if re.search(r"def.*reward|return.*reward|reward\s*=", content):
                reward_formula_found = True

        self._add("reward_dense", "Reward provides dense per-step signal (not just binary)",
                  "Reward", dense_signal, 8 if dense_signal else 0, 8, "MAJOR")
        self._add("reward_partial", "Reward gives partial progress signals", "Reward",
                  partial_reward, 6 if partial_reward else 0, 6, "MAJOR")
        self._add("reward_penalty", "Reward penalizes undesirable behaviour", "Reward",
                  penalty_logic, 5 if penalty_logic else 0, 5, "MAJOR")
        self._add("reward_formula", "Reward function formula defined", "Reward",
                  reward_formula_found, 3 if reward_formula_found else 0, 3, "MINOR")

    # â”€â”€ CATEGORY I: Code Quality & Structure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_code_quality(self):
        CONSOLE.print("  [dim]Auditing code quality and project structure...[/dim]")
        all_py = self._find_files("*.py")

        # Count total lines of Python code
        total_lines = 0
        syntax_errors = []
        for fp in all_py:
            insp = inspect_python_file(fp)
            total_lines += insp.lines
            if insp.syntax_ok is False:
                syntax_errors.append(f"{fp.relative_to(self.d)}: {insp.syntax_error}")

        self.stats["total_python_files"] = len(all_py)
        self.stats["total_python_lines"] = total_lines
        self.stats["syntax_errors"] = syntax_errors

        # Syntax clean
        self._add("code_syntax_clean", "All Python files parse without SyntaxError",
                  "Code Quality", len(syntax_errors) == 0,
                  8 if not syntax_errors else max(0, 8 - len(syntax_errors) * 2), 8, "CRITICAL",
                  "\n".join(syntax_errors) if syntax_errors else "All clean")

        # Test files
        test_files = self._find_files("test_*.py") + self._find_files("*_test.py")
        has_tests = len(test_files) > 0
        self._add("code_tests", f"Test files present ({len(test_files)} found)",
                  "Code Quality", has_tests, min(len(test_files), 5), 5, "MINOR",
                  str([str(t.relative_to(self.d)) for t in test_files[:5]]))

        # Type annotations
        typed_files = 0
        for fp in all_py[:20]:
            content = read_file(fp)
            if ": " in content and "->" in content:
                typed_files += 1
        self._add("code_typed", f"Type annotations used ({typed_files}/{min(len(all_py), 20)} files)",
                  "Code Quality", typed_files >= 3,
                  min(typed_files, 4), 4, "MINOR")

        # requirements.txt or pyproject.toml
        has_reqs = (self.d / "requirements.txt").exists() or (self.d / "pyproject.toml").exists()
        self._add("code_deps", "requirements.txt or pyproject.toml present",
                  "Code Quality", has_reqs, 3 if has_reqs else 0, 3, "MAJOR")

        # Hardcoded API key smell check
        hardcoded_key_smell = False
        for fp in [self.d / "inference.py", self.d / "server" / "app.py"]:
            if fp.exists():
                c = read_file(fp)
                if re.search(r'api_key\s*=\s*"[a-zA-Z0-9_\-]{10,}"', c, re.I):
                    hardcoded_key_smell = True

        self._add("code_no_hardcoded_keys", "No hardcoded API keys detected",
                  "Code Quality", not hardcoded_key_smell,
                  4 if not hardcoded_key_smell else 0, 4, "CRITICAL",
                  "Hardcoded credentials detected!" if hardcoded_key_smell else "Clean")

        # README completeness
        readme = self.d / "README.md"
        if readme.exists():
            rc = readme.read_text(encoding="utf-8", errors="replace").lower()
            sections = {
                "environment description": bool(re.search(r"##.*description|##.*overview|##.*about", rc)),
                "action space": "action" in rc and ("space" in rc or "observation" in rc),
                "observation space": "observation" in rc,
                "task descriptions": "task" in rc and ("easy" in rc or "medium" in rc or "hard" in rc),
                "setup instructions": "install" in rc or "setup" in rc or "docker" in rc,
                "baseline scores": "score" in rc or "baseline" in rc,
            }
            covered = sum(sections.values())
            self._add("readme_sections", f"README covers {covered}/6 required sections",
                      "Code Quality", covered >= 4, min(covered, 6), 6, "MAJOR",
                      str(sections))
        else:
            self._add("readme_sections", "README sections check", "Code Quality",
                      False, 0, 6, "MAJOR", "README.md not found")

    # â”€â”€ CATEGORY J: Real-World Domain Check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_domain_validity(self):
        CONSOLE.print("  [dim]Checking real-world domain validity...[/dim]")
        all_text = ""
        for fp in [self.d / "README.md", self.d / "openenv.yaml"]:
            if fp.exists():
                all_text += read_file(fp).lower()

        # Signs of toy/game domain
        toy_signals = ["game", "minecraft", "atari", "cartpole", "mujoco", "gridworld",
                       "maze", "toy", "test echo", "hello world"]
        real_signals = ["temperature", "thermal", "data center", "scheduling", "triage",
                        "moderation", "customer", "finance", "medical", "supply chain",
                        "manufacturing", "infrastructure", "deployment", "server", "cooling",
                        "workload", "incident", "ticket", "review", "compliance"]

        toy_hits = [s for s in toy_signals if s in all_text]
        real_hits = [s for s in real_signals if s in all_text]

        is_real = len(real_hits) >= 2 and len(toy_hits) == 0
        self._add("domain_real_world", "Domain is a real-world task (not toy/game)",
                  "Domain", is_real, 10 if is_real else 5, 10, "MAJOR",
                  f"Real signals: {real_hits[:5]}, Toy signals: {toy_hits}")

        # Physics / simulation signals
        has_simulation = any(
            kw in all_text for kw in
            ["simulat", "physic", "thermal", "model", "dynamics", "transit"]
        )
        self._add("domain_simulation", "Environment simulates real system dynamics",
                  "Domain", has_simulation, 6 if has_simulation else 0, 6, "MINOR",
                  f"Simulation keywords found: {has_simulation}")

    # â”€â”€ RUN ALL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run_all(self) -> PhaseOneReport:
        checks = [
            self.check_mandatory_files,
            self.check_openenv_yaml,
            self.check_openenv_api,
            self.check_typed_models,
            self.check_tasks_and_graders,
            self.check_inference_script,
            self.check_dockerfile,
            self.check_reward_function,
            self.check_code_quality,
            self.check_domain_validity,
        ]
        for fn in checks:
            fn()

        report = PhaseOneReport(gates=self.gates, project_stats=self.stats)
        report.total_score = sum(g.score for g in self.gates)
        report.max_score = sum(g.max_score for g in self.gates)

        # Disqualification logic (matches competition rules exactly)
        disq_map = {
            "No inference.py": any(g.id == "inf_exists" and not g.passed for g in self.gates),
            "No Dockerfile": any(g.id == "docker_exists" and not g.passed for g in self.gates),
            "No openenv.yaml": any(g.id == "file_openenv.yaml" and not g.passed for g in self.gates),
            "env doesn't define step/reset": any(
                g.id in ("api_step", "api_reset") and not g.passed for g in self.gates),
            "No 3 tasks with graders": any(g.id == "tasks_3plus" and not g.passed for g in self.gates),
            "Graders always return same score": any(g.id == "grader_varies" and not g.passed for g in self.gates),
            "No Pydantic Observation/Action": any(
                g.id in ("models_obs", "models_act") and not g.passed for g in self.gates),
        }

        report.disq_reasons = [reason for reason, triggered in disq_map.items() if triggered]
        report.disqualified = len(report.disq_reasons) > 0
        report.critical_failures = [
            g.name for g in self.gates
            if not g.passed and g.severity in ("DISQUALIFY", "CRITICAL")
        ]
        return report


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PHASE 2 â€” LLM DEEP ANALYSIS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class LLMAnalyzer:
    COMPETITION_RUBRIC = textwrap.dedent("""\
    COMPETITION SCORING RUBRIC:
      Real-world utility (30%): 0-5=toy, 6-15=shallow, 16-25=good, 26-30=excellent gap-filler
      Task & grader quality (25%): 3+ tasks, difficulty range, deterministic, hard task challenges frontier models
      Environment design (20%): clean reset(), action/obs spaces, dense reward, sensible episode boundaries
      Code quality & compliance (15%): openenv validate passes, docker works, HF deploys, baseline reproduces
      Creativity & novelty (10%): novel domain, interesting mechanics, clever reward design
    """)

    def __init__(self, project_dir: Path, phase1: PhaseOneReport):
        self.d = project_dir
        self.p1 = phase1

    def _collect_evidence(self) -> str:
        """Collect the most informative files for the LLM, up to ~10k chars."""
        priority = [
            ("README.md", 4000),
            ("openenv.yaml", 1000),
            ("core/env.py", 2000),
            ("core/simulator.py", 2000),
            ("core/models.py", 1500),
            ("dynamics/thermal_model.py", 2000),
            ("reward/reward_fn.py", 1500),
            ("grader/evaluator.py", 1500),
            ("grader/metrics.py", 1000),
            ("tasks/easy.json", 500),
            ("tasks/medium.json", 500),
            ("tasks/hard.json", 500),
            ("server/app.py", 1500),
            ("inference.py", 2000),
        ]
        parts = []
        total = 0
        for rel, limit in priority:
            fp = self.d / rel
            if fp.exists():
                content = read_file(fp)[:limit]
                block = f"\n\n### {rel}\n```\n{content}\n```"
                parts.append(block)
                total += len(block)
            if total > 14000:
                break
        return "".join(parts)

    def _p1_summary(self) -> str:
        passed = sum(1 for g in self.p1.gates if g.passed)
        total = len(self.p1.gates)
        return (
            f"Phase 1 fit score: {self.p1.total_score:.0f}/{self.p1.max_score:.0f} "
            f"({100 * self.p1.total_score / max(self.p1.max_score, 1):.1f}%). "
            f"{passed}/{total} checks passed. "
            f"Disqualified: {self.p1.disqualified}. "
            f"Disq reasons: {self.p1.disq_reasons or 'none'}. "
            f"Critical failures: {self.p1.critical_failures[:4] or 'none'}."
        )

    def analyze(self) -> PhaseTwoReport:
        evidence = self._collect_evidence()
        p1_summ = self._p1_summary()
        rubric = self.COMPETITION_RUBRIC

        base_ctx = (
            f"PROJECT FILES:\n{evidence}\n\n"
            f"AUTOMATED CHECK SUMMARY:\n{p1_summ}\n\n"
            f"{rubric}"
        )

        analyses = [
            ("Architecture & Modularity",
             "You are a senior RL engineer. Analyse the PROJECT ARCHITECTURE:\n"
             "- How well separated are simulator, env wrapper, server, inference agent?\n"
             "- Is state management clean? Does reset() truly give a fresh episode?\n"
             "- Are there any coupling problems or circular dependencies?\n"
             "- Reference specific files and class names.\n"
             "Be direct, technical, 200 words max."),

            ("Thermal / Domain Simulation Quality",
             "Analyse the DOMAIN SIMULATION quality:\n"
             "- How realistic is the underlying physics/simulation?\n"
             "- Does the environment model something humans actually do in the real world?\n"
             "- How well do the observation fields map to the real domain?\n"
             "- Would this be useful to train or evaluate real agents?\n"
             "Be specific. 200 words max."),

            ("Reward Function Analysis",
             "Analyse the REWARD FUNCTION in depth:\n"
             "- Is it dense (signal every step) or sparse (only at episode end)?\n"
             "- Are there partial progress signals?\n"
             "- Are bad actions penalised?\n"
             "- Could an agent easily exploit or game the reward?\n"
             "- What is the scale/range of reward values?\n"
             "Quote relevant code lines. 200 words max."),

            ("Task & Grader Quality",
             "Analyse TASK QUALITY and GRADERS:\n"
             "- Are the 3 tasks (easy/medium/hard) genuinely different in difficulty?\n"
             "- What makes the hard task hard â€” for a frontier LLM agent?\n"
             "- Are grader scores deterministic and reproducible?\n"
             "- Can graders return the same score regardless of agent behaviour? (DQ risk)\n"
             "- Do graders cover the full 0.0â€“1.0 range?\n"
             "200 words max."),

            ("Inference Script & Spec Compliance",
             "Analyse the INFERENCE SCRIPT and SPEC COMPLIANCE:\n"
             "- Does inference.py emit exactly the required [START]/[STEP]/[END] format?\n"
             "- Are all mandatory env vars read from os.getenv()?\n"
             "- Does it use the OpenAI client as required?\n"
             "- Is the runtime likely to stay under 20 minutes?\n"
             "- Any risk of the baseline not reproducing?\n"
             "Cite specific lines or patterns. 200 words max."),

            ("Code Quality & Security",
             "Analyse CODE QUALITY and SECURITY:\n"
             "- Are Pydantic models typed correctly?\n"
             "- Any hardcoded API keys or credentials?\n"
             "- Is error handling present (try/except in critical paths)?\n"
             "- Are there any obvious bugs, off-by-one errors, infinite loops?\n"
             "- Is the project structure clean and navigable?\n"
             "200 words max."),

            ("Strengths",
             "List the 5 strongest points of this submission as numbered bullets.\n"
             "Be specific, technical, competition-focused. Mention exact components."),

            ("Weaknesses & Risks",
             "List the 5 biggest risks or weaknesses as numbered bullets.\n"
             "Include: disqualification risks, scoring risks, technical fragility.\n"
             "Be blunt. Prioritise by competition impact."),

            ("Pre-Submission Action Plan",
             "Give a prioritised action plan of 6 specific tasks to do BEFORE submitting.\n"
             "Order: most critical first. Each item should name the file to change and what to do.\n"
             "Format: numbered list, one line each."),
        ]

        sections = []
        with Progress(
                SpinnerColumn(), TextColumn("{task.description}"),
                TimeElapsedColumn(), console=CONSOLE, transient=True,
        ) as prog:
            t = prog.add_task("Running LLM analysis...", total=len(analyses))
            for title, system in analyses:
                prog.update(t, description=f"  LLM: {title}...")
                content = llm_call(system, base_ctx, max_tokens=1000)
                sections.append(LLMSection(title=title, content=content))
                prog.advance(t)

        report = PhaseTwoReport(sections=sections, raw_evidence=evidence)
        return report


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PHASE 3 â€” MOCK JUDGE PANEL (Official Rubric)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class JudgePanel:
    """
    Five judges, one per criterion.  Each judge receives:
    - The full competition rubric for their criterion
    - The automated Phase 1 evidence
    - Relevant LLM analysis sections
    - Project file excerpts
    They return: a raw score (0â€“100), a weighted score, and a rationale.
    """

    CRITERIA = [
        (
            "Real-world Utility", 30,
            textwrap.dedent("""\
            JUDGE BRIEF â€” Real-world Utility (weight 30%)

            SCORING GUIDE (map your score to this):
              0â€“16  (raw 0â€“53):  Toy/artificial â€” no practical application. Score: 0-5/30
              17â€“50 (raw 17â€“50): Valid domain but shallow modelling of real task. Score: 6-15/30
              51â€“83 (raw 51â€“83): Good domain modelling, useful for agent evaluation. Score: 16-25/30
              84â€“100(raw 84+):   Excellent â€” fills a real gap, immediate RL/agent community value. Score: 26-30/30

            KEY QUESTIONS:
            - Does the environment simulate something humans actually do in the real world?
            - Would this be useful for training or evaluating real agents?
            - Is it a game, toy, or echo test? (score 0-5)
            - Could a real company deploy this to evaluate AI systems?
            """),
        ),
        (
            "Task & Grader Quality", 25,
            textwrap.dedent("""\
            JUDGE BRIEF â€” Task & Grader Quality (weight 25%)

            SUB-CRITERIA (each worth ~6 points out of 25):
              a) 3+ tasks with difficulty range (easyâ†’mediumâ†’hard)?
              b) Graders produce scores between 0.0â€“1.0?
              c) Graders deterministic and reproducible?
              d) Hard task genuinely challenges frontier models (GPT-4, Claude)?

            DISQUALIFICATION TRIGGER:
            - Graders that always return the same score â†’ score 0
            """),
        ),
        (
            "Environment Design", 20,
            textwrap.dedent("""\
            JUDGE BRIEF â€” Environment Design (weight 20%)

            SUB-CRITERIA:
              a) reset() produces a truly clean state (no leaked state between episodes)?
              b) Action/observation types well-designed and documented?
              c) Reward function provides useful varying signal (not just sparse)?
              d) Episode boundaries sensible (not infinite, not too short)?
            """),
        ),
        (
            "Code Quality & Spec Compliance", 15,
            textwrap.dedent("""\
            JUDGE BRIEF â€” Code Quality & Spec Compliance (weight 15%)

            SUB-CRITERIA:
              a) openenv validate would pass?
              b) docker build && docker run works?
              c) HF Space deploys and responds (POST /reset returns 200)?
              d) Baseline inference.py script runs and reproduces scores?
              e) Clean project structure, typed models, documented?
            """),
        ),
        (
            "Creativity & Novelty", 10,
            textwrap.dedent("""\
            JUDGE BRIEF â€” Creativity & Novelty (weight 10%)

            SUB-CRITERIA:
              a) Domain we haven't seen in OpenEnv before?
              b) Reward design has interesting properties (beyond simple threshold)?
              c) Clever mechanics that make the environment engaging for agent research?
              d) Original approach to the problem?
            """),
        ),
    ]

    def __init__(self, project_dir, phase1: PhaseOneReport, phase2: PhaseTwoReport):
        self.d = project_dir
        self.p1 = phase1
        self.p2 = phase2

    def _build_evidence(self, criterion_name: str) -> str:
        p1_summ = (
            f"Automated fit score: {self.p1.total_score:.0f}/{self.p1.max_score:.0f}. "
            f"Disqualified: {self.p1.disqualified}. "
            f"Disq reasons: {self.p1.disq_reasons or 'none'}.\n"
            f"Critical failures: {self.p1.critical_failures[:5] or 'none'}.\n\n"
            "KEY CHECK RESULTS:\n"
        )
        # Surface checks most relevant to each criterion
        relevant = [g for g in self.p1.gates if not g.passed and g.severity in ("DISQUALIFY", "CRITICAL")][:8]
        passed_notable = [g for g in self.p1.gates if g.passed and g.score >= 5][:8]
        for g in relevant:
            p1_summ += f"  FAIL [{g.severity}] {g.name}: {g.evidence}\n"
        for g in passed_notable:
            p1_summ += f"  PASS {g.name}: {g.evidence}\n"

        # Relevant LLM sections
        llm_relevant = "\n\n".join(
            f"=== {s.title} ===\n{s.content}"
            for s in self.p2.sections
            if any(kw in s.title.lower() for kw in
                   criterion_name.lower().split() + ["strength", "weakness"])
        )[:3000]

        # File excerpt
        excerpts = "\n".join([
            f"\n### {p.name}\n" + read_file(self.d / p)[:800]
            for p in [
                Path("../openenv.yaml"), Path("../core/models.py"), Path("../reward/reward_fn.py"),
                Path("../grader/evaluator.py"), Path("../tasks/hard.json"),
            ]
            if (self.d / p).exists()
        ])[:3000]

        return f"{p1_summ}\n\nLLM ANALYSIS:\n{llm_relevant}\n\nFILE EXCERPTS:\n{excerpts}"

    def score_criterion(self, name: str, weight: int, brief: str) -> JudgeCriterion:
        evidence = self._build_evidence(name)
        prompt = (
            f"{brief}\n\n"
            f"EVIDENCE FROM THIS SUBMISSION:\n{evidence}\n\n"
            f"PROJECT STATS:\n{json.dumps(self.p1.project_stats, default=str, indent=2)[:500]}\n\n"
            "INSTRUCTIONS:\n"
            "You are a competition judge. Score this submission on the criterion above.\n"
            "Respond ONLY with valid JSON (no markdown, no commentary outside JSON):\n"
            "{\n"
            '  "score": <integer 0-100>,\n'
            '  "rationale": "<2-3 sentences, technical and specific>",\n'
            '  "risk_flags": ["<flag1>", "<flag2>"]  // list of specific risks, can be empty\n'
            "}\n"
            "The score is a raw 0-100 number. It will be scaled by the weight to give the "
            "contribution to the 100-point total."
        )
        result = llm_call(
            "You are a strict, experienced competition judge from Meta/Hugging Face. "
            "You respond ONLY in valid JSON, no other text.",
            prompt,
            max_tokens=300,
        )
        data = extract_json_from_llm(result)
        raw = max(0, min(100, int(data.get("score", 50))))
        weighted = round((raw / 100) * weight, 2)
        return JudgeCriterion(
            name=name,
            weight=weight,
            raw_score=raw,
            weighted=weighted,
            rationale=data.get("rationale", result[:300]),
            risk_flags=data.get("risk_flags", []),
        )

    def head_verdict(self, criteria: List[JudgeCriterion], total: float) -> Tuple[str, List[str]]:
        scores_summary = "\n".join(
            f"  {c.name} ({c.weight}%): {c.raw_score}/100 â†’ {c.weighted:.1f} pts"
            for c in criteria
        )
        prompt = (
            f"FINAL SCORES:\n{scores_summary}\n"
            f"TOTAL: {total:.1f}/100\n"
            f"DISQUALIFIED: {self.p1.disqualified}\n"
            f"DISQ REASONS: {self.p1.disq_reasons}\n\n"
            "Write a 4-sentence head judge verdict:\n"
            "  Sentence 1: Does this pass Phase 1 auto-validation and why?\n"
            "  Sentence 2: How will it rank in Phase 2 agentic evaluation?\n"
            "  Sentence 3: Is it likely to reach Phase 3 human review?\n"
            "  Sentence 4: The single most important thing to fix before submission.\n"
            "\nThen on a new line, list 5 specific action items numbered 1-5."
        )
        text = llm_call(
            "You are the head judge. Be direct, technical, specific.",
            prompt, max_tokens=1000
        )
        # Split verdict from action plan
        lines = text.strip().splitlines()
        verdict_lines = []
        action_lines = []
        in_actions = False
        for line in lines:
            if re.match(r"^\s*[1-5][.)]\s", line):
                in_actions = True
            if in_actions:
                action_lines.append(line.strip())
            else:
                verdict_lines.append(line.strip())
        return " ".join(verdict_lines), action_lines

    def run(self) -> PhaseThreeReport:
        criteria = []
        with Progress(
                SpinnerColumn(), TextColumn("{task.description}"),
                TimeElapsedColumn(), console=CONSOLE, transient=True,
        ) as prog:
            t = prog.add_task("Mock judging...", total=len(self.CRITERIA))
            for name, weight, brief in self.CRITERIA:
                prog.update(t, description=f"  Judge: {name}...")
                c = self.score_criterion(name, weight, brief)
                criteria.append(c)
                prog.advance(t)

        total = sum(c.weighted for c in criteria)
        verdict, actions = self.head_verdict(criteria, total)

        phase_predictions = {
            "phase1": ("PASS âœ…" if not self.p1.disqualified else "FAIL âŒ â€” " + "; ".join(self.p1.disq_reasons)),
            "phase2": (
                "Strong contender ðŸ¥‡ (top 25%)" if total >= 78 else
                "Competitive ðŸ¥ˆ (top 50%)" if total >= 62 else
                "Borderline ðŸŸ¡ (middle tier)" if total >= 48 else
                "At risk âš ï¸ (needs work)"
            ),
            "phase3": (
                "Top 10 very likely ðŸ†" if total >= 82 else
                "Top 25 likely ðŸŽ¯" if total >= 68 else
                "Possible ðŸ”§ (polish needed)" if total >= 55 else
                "Unlikely without major work âŒ"
            ),
        }

        return PhaseThreeReport(
            criteria=criteria,
            total_weighted=total,
            phase1_prediction=phase_predictions["phase1"],
            phase2_prediction=phase_predictions["phase2"],
            phase3_prediction=phase_predictions["phase3"],
            head_verdict=verdict,
            action_plan=actions,
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PHASE 4 â€” DEEP IMPLEMENTATION PLAN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class DeepImplementationPlanner:
    def __init__(self, p1: PhaseOneReport, p2: PhaseTwoReport, p3: PhaseThreeReport):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def run(self) -> PhaseFourReport:
        problems = []
        for g in self.p1.gates:
            if not g.passed:
                problems.append(f"[{g.severity}] {g.name}: {g.evidence}")
        for c in self.p3.criteria:
            if c.risk_flags:
                for f in c.risk_flags:
                    problems.append(f"[{c.name} Risk] {f}")
        for s in self.p2.sections:
            if "weakness" in s.title.lower() or "risk" in s.title.lower() or "action plan" in s.title.lower():
                problems.append(f"[LLM Analysis - {s.title}] {s.content[:500]}...")

        if not problems:
            return PhaseFourReport(final_problems=[], detailed_plan="No major problems found! The project is in excellent shape.")

        prompt = (
            "Here are all the issues, risks, and weaknesses found in the submission:\n"
            + "\n".join(f"- {p}" for p in problems) +
            "\n\nBased on the above, please provide:\n"
            "1. A clear, bulleted 'Final List of Problems' that are holding the project back.\n"
            "2. A 'Highly Detailed Implementation Plan' addressing each problem. For each problem, specify exactly what needs to be changed, which files are likely involved, and give concrete step-by-step instructions."
        )

        with Progress(
                SpinnerColumn(), TextColumn("{task.description}"),
                TimeElapsedColumn(), console=CONSOLE, transient=True,
        ) as prog:
            t = prog.add_task("Drafting detailed implementation plan...", total=None)
            content = llm_call(
                system="You are an expert Principal Engineer. Provide a highly detailed, actionable plan. Use markdown.",
                user=prompt,
                max_tokens=4000
            )

        return PhaseFourReport(final_problems=problems, detailed_plan=content)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# RENDERING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def render_phase1(r: PhaseOneReport):
    CONSOLE.print(Rule("[bold cyan]PHASE 1 â€” SUBMISSION GATE CHECKS[/bold cyan]"))

    # Group by category
    cats: Dict[str, List[GateCheck]] = {}
    for g in r.gates:
        cats.setdefault(g.category, []).append(g)

    for cat, gates in cats.items():
        t = Table(title=cat, box=box.SIMPLE_HEAD, border_style="blue",
                  show_lines=False, expand=True)
        t.add_column("Check", style="white", width=42)
        t.add_column("Status", justify="center", width=9)
        t.add_column("Score", justify="right", width=10)
        t.add_column("Severity", justify="center", width=11)
        t.add_column("Evidence / Detail", style="dim", width=48)
        for g in gates:
            status = "[green]âœ“[/green]" if g.passed else "[red]âœ—[/red]"
            sev_color = {
                "DISQUALIFY": "bold red",
                "CRITICAL": "red",
                "MAJOR": "yellow",
                "MINOR": "dim",
            }.get(g.severity, "white")
            score_str = (
                f"[green]{g.score:.0f}[/green]/{g.max_score:.0f}" if g.passed
                else f"[red]{g.score:.0f}[/red]/{g.max_score:.0f}"
            )
            evidence = g.evidence[:60].replace("\n", " ") if g.evidence else ""
            t.add_row(g.name, status, score_str,
                      f"[{sev_color}]{g.severity}[/{sev_color}]", evidence)
        CONSOLE.print(t)
        CONSOLE.print()

    # Stats bar
    pct = 100 * r.total_score / max(r.max_score, 1)
    bar_color = "green" if pct >= 75 else "yellow" if pct >= 50 else "red"
    CONSOLE.print(Panel(
        f"[bold {bar_color}]Fit Score: {r.total_score:.0f} / {r.max_score:.0f}  ({pct:.1f}%)[/bold {bar_color}]\n\n"
        + (
            "[bold red]â›” DISQUALIFICATION TRIGGERS FOUND:\n[/bold red]"
            + "\n".join(f"  â€¢ {r_}" for r_ in r.disq_reasons)
            if r.disqualified else
            "[bold green]âœ… No disqualification triggers detected[/bold green]"
        )
        + (
            f"\n\n[red]Critical failures ({len(r.critical_failures)}):[/red]\n"
            + "\n".join(f"  â€“ {f}" for f in r.critical_failures[:8])
            if r.critical_failures else ""
        ),
        title="Phase 1 Summary",
        border_style=bar_color,
    ))

    # Project stats
    stats = r.project_stats
    st = Table(title="Project Stats", box=box.MINIMAL, border_style="dim")
    st.add_column("Metric", style="bold")
    st.add_column("Value")
    stat_items = [
        ("Python files", stats.get("total_python_files", "?")),
        ("Python LOC", stats.get("total_python_lines", "?")),
        ("Pydantic classes", ", ".join(stats.get("pydantic_classes", [])[:6]) or "none found"),
        ("Syntax errors", len(stats.get("syntax_errors", []))),
        ("Inference LOC", stats.get("inference_lines", "?")),
        ("Inference funcs", ", ".join(stats.get("inference_functions", [])[:5]) or "none"),
        ("Dockerfile FROM", stats.get("dockerfile_from", "not found")),
        ("openenv name", stats.get("openenv_yaml", {}).get("name", "?")),
        ("openenv version", stats.get("openenv_yaml", {}).get("version", "?")),
    ]
    for k, v in stat_items:
        st.add_row(k, str(v))
    CONSOLE.print(st)


def render_phase2(r: PhaseTwoReport):
    CONSOLE.print(Rule("[bold magenta]PHASE 2 â€” LLM DEEP ANALYSIS[/bold magenta]"))

    for section in r.sections:
        color = {
            "Architecture": "cyan",
            "Thermal": "blue",
            "Reward": "magenta",
            "Task": "yellow",
            "Inference": "green",
            "Code Quality": "orange3",
            "Strengths": "bright_green",
            "Weaknesses": "bright_red",
            "Pre-Submission": "gold1",
        }.get(section.title.split()[0], "white")

        CONSOLE.print(f"[bold {color}]=== {section.title} ===[/bold {color}]")
        CONSOLE.print(Markdown(section.content))
        CONSOLE.print()


def render_phase3(r: PhaseThreeReport):
    CONSOLE.print(Rule("[bold yellow]PHASE 3 â€” MOCK JUDGE PANEL[/bold yellow]"))

    t = Table(title="Official Competition Scorecard", box=box.DOUBLE_EDGE,
              border_style="yellow", show_lines=True, expand=True)
    t.add_column("Criterion", style="bold white", width=28)
    t.add_column("Weight", justify="center", width=8)
    t.add_column("Raw (0â€“100)", justify="right", width=12)
    t.add_column("Weighted Score", justify="right", width=14)
    t.add_column("Judge Rationale", style="dim", width=50)

    for c in r.criteria:
        color = "green" if c.raw_score >= 75 else "yellow" if c.raw_score >= 50 else "red"
        t.add_row(
            c.name,
            f"{c.weight}%",
            f"[{color}]{c.raw_score}[/{color}]",
            f"[{color}]{c.weighted:.1f} / {c.weight}[/{color}]",
            c.rationale[:120],
        )

    total_color = "green" if r.total_weighted >= 75 else "yellow" if r.total_weighted >= 55 else "red"
    t.add_row(
        "[bold]TOTAL[/bold]", "[bold]100%[/bold]", "â€”",
        f"[bold {total_color}]{r.total_weighted:.1f} / 100[/bold {total_color}]", ""
    )
    CONSOLE.print(t)
    CONSOLE.print()

    # Risk flags
    all_risks = []
    for c in r.criteria:
        for flag in c.risk_flags:
            all_risks.append(f"[{c.name}] {flag}")
    if all_risks:
        CONSOLE.print(Panel(
            "\n".join(f"  âš  {r_}" for r_ in all_risks),
            title="âš ï¸  Risk Flags Raised by Judges",
            border_style="red",
        ))
        CONSOLE.print()

    # Phase predictions
    pred_t = Table(box=box.SIMPLE, border_style="dim", expand=False)
    pred_t.add_column("Evaluation Phase", style="bold", width=32)
    pred_t.add_column("Prediction", width=40)
    pred_t.add_row("Phase 1 â€” Auto Validation", r.phase1_prediction)
    pred_t.add_row("Phase 2 â€” Agentic Eval", r.phase2_prediction)
    pred_t.add_row("Phase 3 â€” Human Review", r.phase3_prediction)
    CONSOLE.print(Panel(pred_t, title="Competition Phase Predictions", border_style="blue"))
    CONSOLE.print()

    # Verdict
    CONSOLE.print(Panel(
        r.head_verdict or "No verdict generated.",
        title="âš–ï¸  Head Judge Verdict",
        border_style="yellow",
        expand=True,
    ))
    CONSOLE.print()

    # Action plan
    if r.action_plan:
        ap_text = "\n".join(r.action_plan)
        CONSOLE.print(Panel(
            ap_text,
            title="ðŸ—‚  Pre-Submission Action Plan",
            border_style="gold1",
            expand=True,
        ))


def render_phase4(r: PhaseFourReport):
    CONSOLE.print(Rule("[bold bright_cyan]PHASE 4 â€” DEEP IMPLEMENTATION PLAN & PROBLEM LIST[/bold bright_cyan]"))

    problems_text = "\n".join(f"â€¢ {p}" for p in r.final_problems)
    if not problems_text:
        problems_text = "No problems found!"

    CONSOLE.print(Panel(
        problems_text,
        title="[bold red]Final List of Problems Holding the Project Back[/bold red]",
        border_style="red",
        expand=True,
    ))
    CONSOLE.print()

    CONSOLE.print("[bold bright_cyan]Highly Detailed Implementation Plan[/bold bright_cyan]")
    CONSOLE.print(Markdown(r.detailed_plan))
    CONSOLE.print()


def render_final(p1: PhaseOneReport, p2: PhaseTwoReport, p3: PhaseThreeReport):
    CONSOLE.print()
    CONSOLE.print(Rule("[bold white]â•â•â•  COMPOSITE FINAL EVALUATION  â•â•â•[/bold white]"))

    fit_pct = 100 * p1.total_score / max(p1.max_score, 1)
    judge_pts = p3.total_weighted
    combined = fit_pct * 0.25 + judge_pts * 0.75

    grade, grade_color = (
        ("A+  ðŸ†  Top 5% â€” Competition Winner territory", "bright_green") if combined >= 88 else
        ("A   ðŸ¥‡  Top 15% â€” Strong contender", "green") if combined >= 80 else
        ("B+  ðŸ¥ˆ  Top 30% â€” Competitive submission", "green") if combined >= 72 else
        ("B   ðŸŽ¯  Mid-field â€” Good but needs polish", "yellow") if combined >= 64 else
        ("C+  ðŸ”§  Below average â€” Several issues to fix", "yellow") if combined >= 55 else
        ("C   âš ï¸   Weak submission â€” Major work needed", "red") if combined >= 45 else
        ("D   âŒ  At serious risk of disqualification", "bold red")
    )

    CONSOLE.print(Panel(
        f"[bold]Submission Fit Score:[/bold]  {p1.total_score:.0f}/{p1.max_score:.0f}  ({fit_pct:.1f}%)\n"
        f"[bold]Judge Panel Score:[/bold]     {judge_pts:.1f} / 100\n"
        f"[bold]Combined Score:[/bold]        {combined:.1f} / 100\n\n"
        f"[bold {grade_color}]GRADE:  {grade}[/bold {grade_color}]\n\n"
        f"[bold]Disqualified:[/bold]  {'[bold red]YES â€” FIX IMMEDIATELY[/bold red]' if p1.disqualified else '[green]NO[/green]'}\n"
        + (("\n[red]" + "\n".join(f"  â€¢ {r}" for r in p1.disq_reasons) + "[/red]") if p1.disqualified else ""),
        title="ðŸ“Š Final Composite Evaluation",
        border_style=grade_color,
        expand=False,
    ))


def save_report(project_dir: Path, p1: PhaseOneReport, p2: PhaseTwoReport, p3: PhaseThreeReport, p4: PhaseFourReport):
    fit_pct = 100 * p1.total_score / max(p1.max_score, 1)
    combined = fit_pct * 0.25 + p3.total_weighted * 0.75
    out = {
        "project_dir": str(project_dir),
        "combined_score": round(combined, 2),
        "phase1": {
            "total": p1.total_score,
            "max": p1.max_score,
            "pct": round(fit_pct, 1),
            "disqualified": p1.disqualified,
            "disq_reasons": p1.disq_reasons,
            "critical_failures": p1.critical_failures,
            "project_stats": {k: str(v) for k, v in p1.project_stats.items()},
            "gates": [
                {"id": g.id, "name": g.name, "cat": g.category,
                 "passed": g.passed, "score": g.score, "max": g.max_score,
                 "severity": g.severity, "evidence": g.evidence}
                for g in p1.gates
            ],
        },
        "phase2_llm": {s.title: s.content for s in p2.sections},
        "phase3_judge": {
            "total_weighted": p3.total_weighted,
            "phase1_pred": p3.phase1_prediction,
            "phase2_pred": p3.phase2_prediction,
            "phase3_pred": p3.phase3_prediction,
            "verdict": p3.head_verdict,
            "action_plan": p3.action_plan,
            "criteria": [
                {"name": c.name, "weight": c.weight, "raw": c.raw_score,
                 "weighted": c.weighted, "rationale": c.rationale,
                 "risks": c.risk_flags}
                for c in p3.criteria
            ],
        },
        "phase4_plan": {
            "final_problems": p4.final_problems,
            "detailed_plan": p4.detailed_plan,
        },
    }
    out_path = project_dir / "evaluation_report.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    CONSOLE.print(f"\n[dim]ðŸ“„ Full report â†’ {out_path}[/dim]")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    project_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()

    CONSOLE.print()
    CONSOLE.print(Panel.fit(
        "[bold white]OpenEnv Competition Deep Evaluator  v2.0[/bold white]\n"
        "[dim]Submission Gate  â€¢  LLM Analysis  â€¢  Mock Judge Panel  â€¢  Implementation Plan  â€¢  Composite Grade[/dim]",
        border_style="bright_blue", padding=(1, 6),
    ))
    CONSOLE.print(f"[dim]  Project : {project_dir}[/dim]")
    CONSOLE.print(f"[dim]  Model   : {MODEL}[/dim]")
    CONSOLE.print(f"[dim]  Time    : {time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    CONSOLE.print()

    if not project_dir.exists():
        CONSOLE.print(f"[bold red]ERROR: Directory not found: {project_dir}[/bold red]")
        sys.exit(1)

    # â”€â”€ Phase 1 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    CONSOLE.print("[bold cyan]Running Phase 1 â€” Deep Submission Gate Checks...[/bold cyan]")
    checker = SubmissionGateChecker(project_dir)
    p1 = checker.run_all()
    CONSOLE.print()
    render_phase1(p1)
    CONSOLE.print()

    # â”€â”€ Phase 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    CONSOLE.print("[bold magenta]Running Phase 2 â€” LLM Deep Analysis...[/bold magenta]")
    CONSOLE.print("[dim](This makes multiple Ollama calls â€” may take 2â€“5 minutes)[/dim]")
    analyzer = LLMAnalyzer(project_dir, p1)
    p2 = analyzer.analyze()
    CONSOLE.print()
    render_phase2(p2)
    CONSOLE.print()

    # â”€â”€ Phase 3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    CONSOLE.print("[bold yellow]Running Phase 3 â€” Mock Judge Panel...[/bold yellow]")
    CONSOLE.print("[dim](5 judge calls + verdict â€” may take 2â€“4 minutes)[/dim]")
    panel = JudgePanel(project_dir, p1, p2)
    p3 = panel.run()
    CONSOLE.print()
    render_phase3(p3)
    CONSOLE.print()

    # â”€â”€ Phase 4 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    CONSOLE.print("[bold bright_cyan]Running Phase 4 â€” Deep Implementation Plan...[/bold bright_cyan]")
    CONSOLE.print("[dim](Synthesizing all analysis into a detailed plan â€” may take 1â€“2 minutes)[/dim]")
    planner = DeepImplementationPlanner(p1, p2, p3)
    p4 = planner.run()
    CONSOLE.print()
    render_phase4(p4)
    CONSOLE.print()

    # â”€â”€ Final â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    render_final(p1, p2, p3)
    save_report(project_dir, p1, p2, p3, p4)


if __name__ == "__main__":
    main()
