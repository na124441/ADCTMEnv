# grader/evaluator.py
from typing import List, Dict
from tasks.task_config import TaskConfig
from .metrics import compute_metrics


def evaluate_trajectory(
    observations: List[Dict],
    actions: List[Dict],
    config: TaskConfig,
    return_details: bool = False
) -> float | Dict[str, float]:
    """
    Evaluate the full trajectory and assign a normalized score in [0.0, 1.0].
    
    If return_details is True, returns a dictionary containing the overall 
    score and its individual components.
    """
    safety, energy, jitter, target_error = compute_metrics(observations, actions, config)

    # Normalize components to [0,1]
    energy_score = 1.0 - energy
    jitter_score = 1.0 - jitter
    target_score = max(0.0, min(1.0, 1.0 - target_error))

    # Weighted sum
    w_safety = 0.4
    w_target = 0.3
    w_energy = 0.2
    w_jitter = 0.1

    final_score = (
        w_safety * safety +
        w_target * target_score +
        w_energy * energy_score +
        w_jitter * jitter_score
    )

    # Clamp to [0.0, 1.0]
    final_score = max(0.0, min(1.0, final_score))

    if return_details:
        return {
            "score": final_score,
            "safety": safety,
            "energy": energy_score,
            "jitter": jitter_score,
            "target": target_score,
            "raw_energy": energy,
            "raw_jitter": jitter,
            "raw_target_error": target_error
        }

    return final_score

