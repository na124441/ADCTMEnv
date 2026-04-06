import numpy as np

from core.models import Action, Observation, Reward
from tasks.task_config import TaskConfig


def compute_reward(
    prev_obs: Observation,
    curr_obs: Observation,
    act: Action,
    config: TaskConfig,
) -> Reward:
    """
    Compute scalar reward based on current and previous states.

    Components:
        - Temperature violation penalty (quadratic over-limit)
        - Energy cost (sum of cooling levels)
        - Jitter penalty (absolute change in cooling levels)
        - Target tracking penalty (normalized squared error from target)

    Returns:
        Reward(value=float): Negative total cost.
    """
    safe_temp = config.safe_temperature

    temps = np.array(curr_obs.temperatures, dtype=float)
    violations = np.maximum(0.0, temps - safe_temp) ** 2
    temp_penalty = float(violations.sum())

    cool = np.array(act.cooling, dtype=float)
    energy_cost = float(cool.sum())

    if prev_obs.time_step > 0:
        prev_cool = np.array(prev_obs.cooling, dtype=float)
        jitter = float(np.abs(cool - prev_cool).sum())
        if np.max(temps) >= config.safe_temperature - config.jitter_bypass_threshold:
            jitter = 0.0
    else:
        jitter = 0.0

    # NEW: Target tracking term (LQR-style normalized squared error with deadband)
    DEADBAND = 1.5
    
    def compute_normalized_error(t, target):
        diff = abs(t - target)
        return max(0.0, diff - DEADBAND) / target

    target_penalty = sum(
        compute_normalized_error(t, config.target_temperature)**2
        for t in temps
    )
    target_penalty /= len(temps)  # Normalize across zones

    lambda_1 = 2.0
    lambda_2 = 1.0
    lambda_3 = 0.5
    lambda_4 = 3.0  # target_tracking_weight

    total_cost = (
        lambda_1 * temp_penalty
        + lambda_2 * energy_cost
        + lambda_3 * jitter
        + lambda_4 * target_penalty
    )
    return Reward(value=-total_cost)
