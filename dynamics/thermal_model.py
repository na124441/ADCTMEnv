import numpy as np

from config.constants import ALPHA, BETA, GAMMA
from core.models import Action, Observation
from tasks.task_config import TaskConfig


def apply_transition(
    obs: Observation, 
    act: Action, 
    config: TaskConfig, 
    rng: np.random.Generator
) -> Observation:
    """
    Apply one simulation step and return the next observation.
    Uses localized deterministic random generation.
    """
    temperatures = np.array(obs.temperatures)
    workloads = np.array(obs.workloads)
    cooling = np.array(act.cooling)
    ambient_temp = obs.ambient_temp

    workloads = workloads + rng.uniform(
        -config.workload_volatility,
        config.workload_volatility,
        size=len(workloads),
    )
    workloads = np.clip(workloads, 0.0, 1.0)

    cooling_effect = np.full(len(workloads), BETA)
    if config.degradation_step is not None and config.degraded_zone is not None:
        if obs.time_step >= config.degradation_step:
            cooling_effect[config.degraded_zone] = BETA * 0.5

    delta_t = ALPHA * workloads - cooling_effect * cooling + GAMMA * (ambient_temp - temperatures)
    
    # Use a small epsilon for numerical stability and ensure we don't dip below ambient
    EPSILON = 1e-4
    next_temperatures = np.maximum(temperatures + delta_t, ambient_temp - EPSILON)
    next_temperatures = np.maximum(next_temperatures, ambient_temp)

    return Observation(
        temperatures=next_temperatures.tolist(),
        workloads=workloads.tolist(),
        cooling=cooling.tolist(),
        ambient_temp=ambient_temp,
        time_step=obs.time_step + 1,
    )
