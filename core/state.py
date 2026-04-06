"""
Simplified state abstractions resolving internal variable representations locally.
"""
from pydantic import BaseModel
from typing import Optional
from .models import Observation
from tasks.task_config import TaskConfig


class EnvState(BaseModel):
    """
    Stores the internal state of the environment during a single episode.
    Maps high-level components seamlessly allowing fast configuration references internally.
    """
    config: TaskConfig               # immutable task definition loaded once per run natively
    observation: Observation         # current observed physical environment snapshot
    step_counter: int = 0            # local increment mapped tracking sequence executions securely
    done: bool = False               # termination condition evaluation flag
