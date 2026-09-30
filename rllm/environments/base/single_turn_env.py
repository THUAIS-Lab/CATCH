import warnings
from typing import Any

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
from rllm.data.utils import attach_livecodebench_prompt_fields
from rllm.rewards.reward_fn import RewardFunction, zero_reward


class SingleTurnEnvironment(MultiTurnEnvironment):
    """
    A simple environment for single-turn interactions with LLMs.
    This is a special case of MultiTurnEnvironment where max_turns=1.
    The environment provides a question/prompt and evaluates the response using a custom reward function.
    """

    def __init__(self, task: dict | None = None, reward_fn: RewardFunction | None = None, get_full_reward_output=False, **kwargs):
        """
        Initialize the single turn environment.

        Args:
            task: Dictionary containing the task information, including at least a "question" field
        """
        super().__init__(task=task, max_turns=1, **kwargs)
        if reward_fn is None:
            warnings.warn("No reward function provided, using zero reward", stacklevel=2)
        self.reward_fn = reward_fn or zero_reward
        self.get_full_reward_output = get_full_reward_output

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action: Response string from the LLM

        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # Store the action in history
        self.history.append(action)

        # Calculate reward for the current turn using the abstract method
        assert self.task is not None, "Task is not set"
        reward, next_obs = self.get_reward_and_next_obs(self.task, action, self.get_full_reward_output)

        # Increment turn counter
        self.current_turn += 1

        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    def get_reward_and_next_obs(self, task: dict, action: Any, get_full_reward_output=False) -> tuple[float, dict]:
        """
        Compute the reward based on the task and action.

        Args:
            task: The task dictionary containing relevant information
            action: The action taken by the agent

        Returns:
            Tuple of (reward: float, next_observation: Dict)
        """
        reward_output = self.reward_fn(task_info=task, action=action)

        if get_full_reward_output:
            return reward_output, {}
        return reward_output.reward, {}

    @staticmethod
    def from_dict(env_args: dict) -> "SingleTurnEnvironment":
        env_args = dict(env_args)
        reward_fn = env_args.pop("reward_fn", None)
        get_full_reward_output = env_args.pop("get_full_reward_output", False)
        if "task" in env_args:
            task = attach_livecodebench_prompt_fields(dict(env_args["task"]), env_args)
        else:
            task = attach_livecodebench_prompt_fields(env_args, env_args)
        return SingleTurnEnvironment(task=task, reward_fn=reward_fn, get_full_reward_output=get_full_reward_output)
