"""
Core physics simulation orchestration logic module.
Encapsulates environment lifecycles internally managing transitions mapped continuously 
by discrete time steps tracking iteration bounds safely.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

from core.models import Action, InfoDict, Observation, StepResponse
from core.paths import TASKS_DIR
from dynamics.thermal_model import apply_transition
from reward.reward_fn import compute_reward
from tasks.task_config import TaskConfig


class SimulationSession:
    """
    A unified wrapper containing the full environment state logic. 
    Maintains all observation iterations, physical constraints, and termination rules locally.
    """
    def __init__(self, config: TaskConfig):
        """
        Initialization assigning the rigid task environment structure globally.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.observation = Observation(
            temperatures=config.initial_temperatures,
            workloads=config.initial_workloads,
            cooling=[0.0] * config.num_zones,
            ambient_temp=config.ambient_temperature,
            time_step=0,
        )
        self.step_counter = 0
        self.done = False

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the full internal session snapshot mapped functionally.
        Used extensively for API consumption payloads seamlessly.
        """
        return {
            "config": self.config.model_dump(),
            "observation": self.observation.model_dump(),
            "step_counter": self.step_counter,
            "done": self.done,
            "seed": self.config.seed,
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_dict(cls, task_config: Dict[str, Any]) -> "SimulationSession":
        """
        Alternate constructor passing raw dictionary definitions towards TaskConfig validations natively.
        """
        return cls(TaskConfig.model_validate(task_config))

    @classmethod
    def from_task_name(cls, task_name: str = "easy") -> "SimulationSession":
        """
        Alternate syntactic constructor simplifying task injection loading dynamically resolving filenames securely.
        """
        task_file = TASKS_DIR / f"{task_name.replace('.json', '')}.json"
        with task_file.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main transition function pushing physics simulations forward.
        Resolves input actions, manages state loops calculating reward variables incrementally.
        
        Args:
            action_dict: Target cooling commands map.
        
        Returns:
            Dict mapping next physical observations, resulting rewards structures and termination statuses dynamically.
        """
        # Short-circuit logic check blocking attempts to actuate against terminated scenarios securely.
        if self.done:
            raise ValueError("Episode already finished. Reset before stepping again.")

        # Cast raw inputs securely parsing Pydantic typing definitions strictly mapping parameters locally constraint wise.
        action = Action.model_validate(action_dict)
        if len(action.cooling) != self.config.num_zones:
            raise ValueError(
                f"Action length ({len(action.cooling)}) does not match num_zones ({self.config.num_zones})"
            )

        # Retain history
        prev_obs = self.observation
        
        # Extrapolate physical heat generation formula mathematically.
        new_obs = apply_transition(prev_obs, action, self.config, self.rng)
        
        # Calculate grade scoring mechanics for current action iteration mapped sequentially.
        reward_obj = compute_reward(
            prev_obs=prev_obs,
            curr_obs=new_obs,
            act=action,
            config=self.config,
        )

        # Mutate internal state
        self.observation = new_obs
        self.step_counter += 1
        
        # Compare iteration lengths defining termination boundaries securely
        self.done = self.step_counter >= self.config.max_steps

        # Transmit standard OpenAI integration output format dictionary natively
        return StepResponse(
            observation=new_obs,
            reward=reward_obj,
            done=self.done,
            info=InfoDict(step=self.step_counter),
        ).model_dump()

    def model_dump(self) -> Dict[str, Any]:
        """
        Utility mapping redirect abstracting class instance states safely.
        """
        return self.get_state()
