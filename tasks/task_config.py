# tasks/task_config.py
from pydantic import BaseModel, Field, model_validator
from pydantic import ConfigDict
from typing import List, Optional


class TaskConfig(BaseModel):
    """
    Defines the parameters of a specific task/environment scenario.
    Loaded at reset() to initialize the environment.
    """
    model_config = ConfigDict(extra="allow")

    num_zones: int = Field(..., ge=1, description="Number of cooling zones")
    initial_temperatures: List[float] = Field(..., min_length=1)
    initial_workloads: List[float] = Field(..., min_length=1)
    ambient_temperature: float = Field(..., description="Constant ambient temperature (Â°C)")
    safe_temperature: float = Field(..., description="Maximum allowed temperature per zone")
    max_steps: int = Field(..., gt=0, description="Episode horizon (steps)")
    target_temperature: float = Field(..., description="Target temp goal (used in grading)")
    seed: int = Field(42, description="Random seed dictating simulation reproducibility")

    # Dynamic elements
    workload_volatility: float = Field(0.05, description="Volatility for workload random walk")
    degradation_step: Optional[int] = Field(None, description="Step at which fan degradation begins")
    degraded_zone: Optional[int] = Field(None, description="Index of the zone suffering degradation")
    jitter_bypass_threshold: float = Field(2.0, ge=0.0, description="Temperature delta below safe limits where jitter penalty is bypassed.")

    @model_validator(mode="after")
    def validate_zone_consistency(self) -> "TaskConfig":
        """Ensure task parameters are internally consistent."""
        if len(self.initial_temperatures) != self.num_zones:
            raise ValueError(
                f"initial_temperatures length ({len(self.initial_temperatures)}) "
                f"must match num_zones ({self.num_zones})"
            )
        if len(self.initial_workloads) != self.num_zones:
            raise ValueError(
                f"initial_workloads length ({len(self.initial_workloads)}) "
                f"must match num_zones ({self.num_zones})"
            )
        if self.safe_temperature <= self.target_temperature:
            raise ValueError("safe_temperature must be greater than target_temperature")
        if self.workload_volatility < 0:
            raise ValueError("workload_volatility must be non-negative")
        if any(not 0.0 <= workload <= 1.0 for workload in self.initial_workloads):
            raise ValueError("initial_workloads values must stay within [0.0, 1.0]")
        if self.degradation_step is not None and self.degradation_step < 0:
            raise ValueError("degradation_step must be non-negative")
        if self.degraded_zone is not None and not 0 <= self.degraded_zone < self.num_zones:
            raise ValueError("degraded_zone must be a valid zone index")
        return self
