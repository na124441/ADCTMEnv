# grader/metrics.py
import numpy as np
from typing import List, Dict
from tasks.task_config import TaskConfig


def compute_metrics(
    observations: List[Dict],
    actions: List[Dict],
    config: TaskConfig
) -> tuple[float, float, float, float]:
    """
    Compute safety, energy usage, and jitter metrics from the full trajectory.

    Args:
        observations: List of observation dicts including initial state
        actions: List of action dicts (length = steps taken)
        config: Task configuration defining thresholds

    Returns:
        Tuple of:
            - safety_ratio: fraction of steps where all zones were safe
            - avg_energy: average cooling level across zones/steps
            - avg_jitter: average absolute difference in cooling between steps
            - avg_target_error: average normalized deviation from target
    """
    # Convert to NumPy arrays
    temps = np.array([obs["temperatures"] for obs in observations])  # Shape: (steps+1, N)
    cools = np.array([act["cooling"] for act in actions])           # Shape: (steps, N)

    num_steps = len(actions)
    if num_steps == 0:
        return 0.0, 0.0, 0.0, 0.0  # No steps taken

    # 1ï¸âƒ£ Safety metric: % of steps where all zones are below safe_temp
    safe_mask = (temps[1:] <= config.safe_temperature).all(axis=1)  # Skip initial state
    safety_ratio = safe_mask.sum() / num_steps

    # 2ï¸âƒ£ Energy metric: average cooling level
    avg_energy = cools.mean()  # Since cooling âˆˆ [0,1], mean is normalized

    # 3ï¸âƒ£ Jitter metric: average absolute change in cooling per zone
    if num_steps > 1:
        diffs = np.abs(np.diff(cools, axis=0))  # Differences along time axis
        avg_jitter = diffs.mean()
    else:
        avg_jitter = 0.0

    # 4ï¸âƒ£ Target metric: average normalized error from target (with deadband)
    DEADBAND = 1.5
    diffs_target = np.abs(temps[1:] - config.target_temperature)
    errors_target = np.maximum(0.0, diffs_target - DEADBAND)
    normalized_errors = errors_target / config.target_temperature
    avg_target_error = float(normalized_errors.mean())

    return safety_ratio, avg_energy, avg_jitter, avg_target_error
