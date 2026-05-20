from __future__ import annotations

from typing import Protocol, runtime_checkable

from rllm.agents.agent import Action
from rllm.rewards.reward_types import RewardConfig, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn


@runtime_checkable
class RewardFunction(Protocol):
    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        ...


def zero_reward(task_info: dict, action: str) -> RewardOutput:
    return RewardOutput(reward=0.0, metadata={})


def search_reward_fn(task_info: dict, action: str) -> RewardOutput:
    reward_fn = RewardSearchFn(RewardConfig())
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)
