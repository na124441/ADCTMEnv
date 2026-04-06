"""
Shared Pydantic models for the ADCTM environment.

These schemas are used by the simulator, FastAPI server, and local runners so
the HTTP contract stays consistent across the project.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tasks.task_config import TaskConfig


class Observation(BaseModel):
    """
    Observable environment state returned to an agent after reset/step.
    """

    model_config = ConfigDict(extra="forbid")

    temperatures: List[float] = Field(..., min_length=1)
    workloads: List[float] = Field(..., min_length=1)
    cooling: List[float] = Field(..., min_length=1)
    ambient_temp: float = Field(..., description="Ambient temperature in Celsius")
    time_step: int = Field(..., ge=0)

    @model_validator(mode="after")
    def same_length(self) -> "Observation":
        n_temps = len(self.temperatures)
        if len(self.workloads) != n_temps or len(self.cooling) != n_temps:
            raise ValueError("All lists (temperatures, workloads, cooling) must have equal length.")
        return self


class Action(BaseModel):
    """
    Cooling command produced by an agent.
    """

    model_config = ConfigDict(extra="forbid")

    cooling: List[float] = Field(..., min_length=1)

    @field_validator("cooling")
    @classmethod
    def clip_values(cls, values: List[float]) -> List[float]:
        return [max(0.0, min(1.0, float(value))) for value in values]


class Reward(BaseModel):
    """
    Scalar reward emitted after each step.
    """

    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., description="Scalar reward value")


class InfoDict(BaseModel):
    """
    Auxiliary metadata returned by the HTTP API.
    """

    model_config = ConfigDict(extra="allow")

    step: int = Field(..., ge=0)


class ResetPayload(BaseModel):
    """
    Supported `/reset` payload fields.

    Additional task configuration fields are allowed so callers can either send a
    full `TaskConfig` payload or a lightweight task selector.
    """

    model_config = ConfigDict(extra="allow")

    task_name: Optional[str] = None
    task: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0)
    episode_id: Optional[str] = None


class ResetResponse(BaseModel):
    """
    Canonical response returned by `/reset`.
    """

    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: Optional[Reward] = None
    done: bool = False
    info: InfoDict = Field(default_factory=lambda: InfoDict(step=0))


class StepResponse(BaseModel):
    """
    Canonical response returned by `/step`.
    """

    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: Reward
    done: bool
    info: InfoDict


class StateResponse(BaseModel):
    """
    Canonical response returned by `/state`.
    """

    model_config = ConfigDict(extra="allow")

    config: TaskConfig
    observation: Observation
    step_counter: int = Field(..., ge=0)
    done: bool
    seed: int
    rng_state: Dict[str, Any]
