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
    Evaluate the full trajectory and assign a normalized score in (0.0, 1.0).
    
    If return_details is True, returns a dictionary containing the overall 
    score and its individual components.
    """
    safety, energy, jitter, target_error = compute_metrics(observations, actions, config)

    # Define a small epsilon to ensure scores are strictly between 0 and 1
    epsilon = 1e-6

    # Normalize components to (0,1)
    # Clamp safety to be strictly between epsilon and 1 - epsilon
    safety_score = max(epsilon, min(1.0 - epsilon, safety))
    
    # Clamp energy_score to be strictly between epsilon and 1 - epsilon
    energy_score = max(epsilon, min(1.0 - epsilon, 1.0 - energy))
    
    # Clamp jitter_score to be strictly between epsilon and 1 - epsilon
    jitter_score = max(epsilon, min(1.0 - epsilon, 1.0 - jitter))
    
    # Clamp target_score to be strictly between epsilon and 1 - epsilon
    target_score = max(epsilon, min(1.0 - epsilon, 1.0 - target_error))

    # Weighted sum
    w_safety = 0.4
    w_target = 0.3
    w_energy = 0.2
    w_jitter = 0.1

    final_score = (
        w_safety * safety_score +
        w_target * target_score +
        w_energy * energy_score +
        w_jitter * jitter_score
    )

    # Ensure final_score is strictly between 0 and 1
    final_score = max(epsilon, min(1.0 - epsilon, final_score))

    if return_details:
        return {
            "score": final_score,
            "safety": safety_score,
            "energy": energy_score,
            "jitter": jitter_score,
            "target": target_score,
            "raw_safety": safety, # Added for debugging/details
            "raw_energy": energy,
            "raw_jitter": jitter,
            "raw_target_error": target_error
        }

    return final_score
