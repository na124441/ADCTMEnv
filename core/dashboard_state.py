"""
Data structures for managing complex UI state representation.
Provides a comprehensive dataclass container bridging physics outputs
and Rich Terminal visualization tools. 
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Deque
from collections import deque

@dataclass(frozen=True)
class DashboardState:
    """
    A frozen configuration block holding all relevant dashboard UI data.
    Designed comprehensively without modification functions, effectively acting 
    like a single immutable frame rendered by the frontend engine.
    """
    # Core simulation properties
    step: int
    temperatures: List[float]
    cooling: List[float]
    reward: float
    total_reward: float
    done: bool

    # Domain hyper-parameters mapping directly to evaluation logic
    alpha: float = 2.0
    beta: float = 1.0
    gamma: float = 0.5
    safe_temp: float = 85.0
    num_zones: int = 1
    target_temp: float = 65.0
    max_steps: int = 50

    # Physics internal breakdowns mapped for visual reporting panels
    overshoot_term: float = 0.0
    energy_term: float = 0.0
    smoothness_term: float = 0.0
    deltas: List[float] = field(default_factory=list)
    confidence: float = 100.0
    trajectory_status: str = "INITIALIZING"
    policy_signature: Dict[str, float] = field(default_factory=lambda: {"avg": 0.0, "var": 0.0})

    # Phase 2 Intelligence metadata outputs produced by external heuristic scripts
    forecasts: List[str] = field(default_factory=list)      # Time-to-violation predictions mapped forward
    action_quality: List[str] = field(default_factory=list) # Control action evaluation strings mapped
    events: List[str] = field(default_factory=list)          # Significant logged events mapping to flags

    # Deep memory tracking history allowing graph visualization of the full rollout natively
    temp_history: List[List[float]] = field(default_factory=list)
    action_history: List[List[float]] = field(default_factory=list)

    # Contextual awareness regarding what the Agent produced inside the last iteration cycle
    prompt: str = ""
    llm_raw_output: str = "" # Full unstructured string 
    parsed_action: Optional[Dict[str, Any]] = None # Clean interpreted logic form 
    parse_error: Optional[str] = None # System syntax log of exception during parse validation 

    # Runtime parameters
    task_name: str = "unknown"
    model: str = "unknown"
    
    # Internal ring buffer preserving brief activity debug tracing inside a standard UI footer UI length
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=20))
    error_banner: Optional[str] = None
    final_figure_path: Optional[str] = None
