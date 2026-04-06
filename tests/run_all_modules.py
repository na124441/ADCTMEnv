from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT_DIR / "tests"

# Ordered to match the current project structure.
TEST_TARGETS = [
    "tests/test_app_entry.py",
    "tests/test_config_constants.py",
    "tests/test_submission_readiness.py",
    "tests/core",
    "tests/tasks",
    "tests/dynamics",
    "tests/reward",
    "tests/grader",
    "tests/analysis",
    "tests/inference",
    "tests/server",
    "tests/ui",
]


def run_target(target: str) -> int:
    print(f"\n=== Running {target} ===")
    return pytest.main(["-q", target])


def main() -> int:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.chdir(ROOT_DIR)
    tests_dir_str = str(TESTS_DIR)
    sys.path[:] = [entry for entry in sys.path if entry not in ("", tests_dir_str)]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    results: list[tuple[str, int]] = []
    for target in TEST_TARGETS:
        exit_code = run_target(target)
        results.append((target, exit_code))

    print("\n=== Module Summary ===")
    for target, exit_code in results:
        status = "PASS" if exit_code == 0 else "FAIL"
        print(f"{status:4}  {target}")

    failed = [target for target, exit_code in results if exit_code != 0]
    if failed:
        print("\nFailing targets:")
        for target in failed:
            print(f"- {target}")
        return 1

    print("\nAll module test groups passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
