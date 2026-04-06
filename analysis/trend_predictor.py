from typing import List, Optional

def predict_thermal_future(temps: List[float], history: List[List[float]], safe_temp: float, window: int = 5) -> List[str]:
    """Predicts time-to-violation for each zone based on rolling temperature trends."""
    num_zones = len(temps)
    if len(history) < window:
        return ["Initializing..." for _ in temps]
        
    forecasts = []
    for z_idx in range(num_zones):
        current_temp = temps[z_idx]
        
        # Calculate average velocity over the window
        deltas = []
        for i in range(1, window):
            prev = history[-(i+1)][z_idx]
            curr = history[-i][z_idx]
            deltas.append(curr - prev)
            
        avg_velocity = sum(deltas) / len(deltas)
        
        if avg_velocity <= 0:
            forecasts.append("Stable/Cooling")
        else:
            # How many steps until it reaches safe_temp?
            steps_to_critical = (safe_temp - current_temp) / avg_velocity
            if steps_to_critical < 0:
                forecasts.append("CRITICAL")
            elif steps_to_critical < 1.0:
                forecasts.append("< 1 step âš ")
            else:
                forecasts.append(f"~{int(steps_to_critical)} steps")
                
    return forecasts
