from typing import List, Dict

def evaluate_action_quality(temps: List[float], actions: List[float], deltas: List[float], safe_temp: float) -> List[str]:
    """Assesses the quality of agent cooling actions relative to thermal urgency."""
    num_zones = len(temps)
    assessments = []
    
    for i in range(num_zones):
        temp = temps[i]
        action = actions[i]
        delta = deltas[i]
        
        # Heuristics for control-system validation
        if temp > safe_temp - 5.0: # Close to safety limit
            if action > 0.8: assessments.append("Optimal (Reactive)")
            elif action > 0.5: assessments.append("Warning (Passive)")
            else: assessments.append("Under-reacting âš ")
        elif temp < 70 and action > 0.5:
             assessments.append("Over-reacting (Wasteful)")
        elif delta < -0.5:
             assessments.append("Optimal (Cooling)")
        elif abs(delta) < 0.1 and temp > 75:
             if action < 0.2: assessments.append("Idle (Dangerous)")
             else: assessments.append("Stable (Maintaining)")
        else:
            assessments.append("Neutral")
            
    return assessments

def assess_policy_type(avg_action: float, action_variance: float) -> str:
    """Classifies the agent's behavioral persona based on rolling statistics."""
    if action_variance > 0.05:
        return "AGGRESSIVE-REACTIVE"
    elif action_variance < 0.005:
        return "STEADY-STABLE"
    elif avg_action > 0.7:
        return "STEADY-AGGRESSIVE"
    elif avg_action < 0.2:
        return "PASSIVE-CONSERVATIVE"
    else:
        return "BALANCED-OPTIMAL"
