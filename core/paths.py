"""
Path resolution centralized definitions module.
Anchors relative internal directory resolutions dynamically handling standard OS directory abstractions.
"""
from pathlib import Path

# Core absolute application root resolved inherently through the module path structure.
BASE_DIR = Path(__file__).resolve().parent.parent

# Centralized location resolving towards JSON payload configurations defining the agent challenge logic scenarios.
TASKS_DIR = BASE_DIR / "tasks"
