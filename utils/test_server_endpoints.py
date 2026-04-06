#!/usr/bin/env python3
"""Test all ADCTM server endpoints and display results in a Rich table.

Usage:
    python test_server_endpoints.py

The script assumes the FastAPI server is already running on http://localhost:7860.
If you need to start it automatically, uncomment the `start_server()` call at the end.
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("ADCTM_API_URL", "http://localhost:7860")
TASKS_DIR = Path(__file__).parent.parent / "tasks"
EASY_TASK_FILE = TASKS_DIR / "easy.json"

console = Console()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _request(method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
    """Perform an HTTP request and return a dict with status and payload.

    Returns a dictionary with the keys:
        - ``method``   : HTTP method used
        - ``url``      : Full URL
        - ``status``   : HTTP status code (int) or ``None`` on exception
        - ``ok``       : ``True`` if 200 <= status < 300
        - ``payload``  : JSON response (dict) or error message (str)
    """
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.request(method, url, timeout=10, **kwargs)
        status = resp.status_code
        ok = resp.ok
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
    except Exception as exc:  # pragma: no cover â€“ network errors
        status = None
        ok = False
        payload = str(exc)
    return {
        "method": method.upper(),
        "url": url,
        "status": status,
        "ok": ok,
        "payload": payload,
    }

def _print_table(results: List[Dict[str, Any]]) -> None:
    """Render a Rich table from a list of result dictionaries, showing detailed response data."""
    table = Table(title="ADCTM Server Endpoint Test Results", box=box.SIMPLE_HEAVY)
    table.add_column("Endpoint", style="cyan", no_wrap=True)
    table.add_column("Method", style="magenta", justify="center")
    table.add_column("Status", style="green", justify="center")
    table.add_column("OK?", style="bright_green", justify="center")
    table.add_column("Response", style="yellow")

    for r in results:
        status = str(r["status"]) if r["status"] is not None else "ERR"
        ok = "âœ“" if r["ok"] else "âœ—"
        payload = r["payload"]
        if isinstance(payload, dict):
            # prettyâ€‘print JSON, then truncate to keep table readable
            pretty = json.dumps(payload, indent=2)
            # replace newlines with spaces for table cell, then truncate
            pretty_one_line = pretty.replace("\n", " ")
            summary = (pretty_one_line[:200] + "â€¦") if len(pretty_one_line) > 200 else pretty_one_line
        else:
            summary = str(payload)[:200]
        table.add_row(r["url"].replace(BASE_URL, ""), r["method"], status, ok, summary)

    console.print(table)

# ---------------------------------------------------------------------------
# Individual endpoint tests
# ---------------------------------------------------------------------------
def test_reset() -> Dict[str, Any]:
    """POST /reset with the full ``easy.json`` payload."""
    if not EASY_TASK_FILE.is_file():
        raise FileNotFoundError(f"Task file not found: {EASY_TASK_FILE}")
    with open(EASY_TASK_FILE, "r", encoding="utf-8") as f:
        task_cfg = json.load(f)
    return _request("POST", "/reset", json=task_cfg)

def test_step(initial_obs: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step using a simple cooling action.

    The action is a list of ``0.3`` for each zone â€“ this works for any
    ``num_zones`` value present in the observation.
    """
    num_zones = len(initial_obs.get("temperatures", []))
    action = {"cooling": [0.3] * num_zones}
    return _request("POST", "/step", json=action)

def test_run() -> Dict[str, Any]:
    """POST /run with ``task_id":"easy.json"`` â€“ the highâ€‘level oneâ€‘shot call."""
    return _request("POST", "/run", json={"task_id": "easy.json"})

def test_state() -> Dict[str, Any]:
    """GET /state â€“ returns the internal server state (if any)."""
    return _request("GET", "/state")

# ---------------------------------------------------------------------------
# Optional helper to start the server automatically (commented out by default)
# ---------------------------------------------------------------------------
def start_server() -> subprocess.Popen:
    """Spawn ``uvicorn`` as a background process.

    Returns the ``Popen`` object so the caller can terminate it later.
    """
    cmd = ["python", "-m", "server.app"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Give the server a moment to bind the port
    time.sleep(2)
    return proc

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main() -> None:
    # Uncomment the next two lines if you want the script to launch the server
    # server_proc = start_server()
    # try:
    results: List[Dict[str, Any]] = []

    # 1ï¸âƒ£ Reset â€“ we need the initial observation for the step test
    reset_res = test_reset()
    results.append(reset_res)
    init_obs = reset_res.get("payload", {}) if reset_res["ok"] else {}

    # 2ï¸âƒ£ Step â€“ only if reset succeeded
    if init_obs:
        step_res = test_step(init_obs)
        results.append(step_res)
    else:
        console.print("[red]Skipping /step test â€“ reset failed.[/red]")

    # 3ï¸âƒ£ Run â€“ the highâ€‘level endpoint
    results.append(test_run())

    # 4ï¸âƒ£ State â€“ optional, may be empty depending on server implementation
    results.append(test_state())

    # Render the table
    _print_table(results)
    # finally:
    #     server_proc.terminate()
    # except KeyboardInterrupt:
    #     server_proc.terminate()

if __name__ == "__main__":
    main()
