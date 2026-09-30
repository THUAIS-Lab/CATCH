import re
import string
from collections import Counter
from typing import Protocol, runtime_checkable

from rllm.agents.agent import Action
from rllm.rewards.code_reward import RewardCodeFn
from rllm.rewards.math_reward import RewardMathFn
from rllm.rewards.pc_swe_reward import RewardPCSWEFn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions"""

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for an agent's action.

        Args:
            task_info: The task dictionary containing question, answer, and other metadata
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward value, either as a float or RewardOutput object
        """
        ...


# Simple example implementation
def zero_reward(task_info: dict, action: str) -> RewardOutput:
    """
    A simple reward function that always returns zero.
    Useful as a placeholder when no specific reward logic is needed.

    Args:
        task: The task dictionary
        action: The agent's response

    Returns:
        float: Always returns 0.0
    """
    return RewardOutput(reward=0.0, metadata={})


def math_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for math tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on math evaluation
    """
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)


def search_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for search tasks that implements the RewardFunction protocol.

    Args:
        task_info: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        RewardOutput: The calculated reward value based on search evaluation
    """
    reward_config = RewardConfig()
    reward_fn = RewardSearchFn(reward_config)
    if isinstance(action, Action):
        action = action.action

    # Create RewardInput from task_info and action
    reward_input = RewardInput(task_info=task_info, action=action)

    return reward_fn(reward_input)


def _get_llm_monitor_config(config) -> dict | None:
    """Extract llm_monitor config dict from the config object.

    Handles OmegaConf DictConfig, plain dict, argparse Namespace, or objects
    with attribute access.
    """
    # OmegaConf DictConfig
    if hasattr(config, "rllm"):
        rllm = config.rllm
        if hasattr(rllm, "llm_monitor"):
            llm_cfg = rllm.llm_monitor
            if hasattr(llm_cfg, "model"):
                return {
                    k: getattr(llm_cfg, k)
                    for k in (
                        "model", "api_key", "base_url", "api_key_env",
                        "timeout", "max_retries", "max_tokens", "temperature",
                        "concurrency", "extra_body",
                        "penalize_is_hack", "penalize_reward",
                    )
                    if hasattr(llm_cfg, k)
                }
    # Plain dict
    if isinstance(config, dict):
        llm_cfg = config.get("rllm", {}).get("llm_monitor")
        if llm_cfg and llm_cfg.get("model"):
            return llm_cfg
    return None


def code_reward_fn(config, task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for code tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on code execution results
    """
    reward_config = RewardConfig()
    # Allow Hydra overrides for RewardConfig fields via +rllm.env.env_args.reward_config.*
    _reward_overrides = (
        config.rllm.env.get("env_args", {}).get("reward_config", {})
        if hasattr(config, "rllm")
        else {}
    )
    for _k, _v in _reward_overrides.items():
        if hasattr(reward_config, _k):
            setattr(reward_config, _k, _v)
    use_pc_swe_reward = "repo_files" in task_info

    # Construct LLM monitor if configured (only for pc_swe reward path)
    llm_monitor = None
    llm_monitor_penalize_is_hack = True
    llm_monitor_penalize_reward = -0.3
    if use_pc_swe_reward:
        llm_cfg = _get_llm_monitor_config(config)
        if llm_cfg and llm_cfg.get("model"):
            from rllm.rewards.llm_monitor import LLMMonitor

            llm_monitor = LLMMonitor(
                model=llm_cfg["model"],
                api_key=llm_cfg.get("api_key", "EMPTY"),
                base_url=llm_cfg.get("base_url", ""),
                api_key_env=llm_cfg.get("api_key_env", ""),
                timeout=float(llm_cfg.get("timeout", 120.0)),
                max_retries=int(llm_cfg.get("max_retries", 3)),
                max_tokens=int(llm_cfg.get("max_tokens", 256)),
                temperature=float(llm_cfg.get("temperature", 0.6)),
                concurrency=int(llm_cfg.get("concurrency", 16)),
                extra_body=llm_cfg.get("extra_body", None),
            )
            llm_monitor_penalize_is_hack = bool(
                llm_cfg.get("penalize_is_hack", True)
            )
            llm_monitor_penalize_reward = float(
                llm_cfg.get("penalize_reward", -0.3)
            )

    reward_fn = (
        RewardPCSWEFn(
            reward_config,
            exp_config=config,
            llm_monitor=llm_monitor,
            llm_monitor_penalize_is_hack=llm_monitor_penalize_is_hack,
            llm_monitor_penalize_reward=llm_monitor_penalize_reward,
        )
        if use_pc_swe_reward
        else RewardCodeFn(reward_config, exp_config=config)
    )
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)


def f1_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function that computes F1 score between predicted text and gold text.

    This function normalizes both texts (lowercase, remove punctuation, remove articles,
    fix whitespace) before tokenizing and computing F1 score based on token overlap.

    Args:
        task_info: The task dictionary containing ground_truth (gold text)
        action: The agent's predicted response/solution

    Returns:
        RewardOutput: The calculated reward value (F1 score)

    Example:
        >>> task_info = {"ground_truth": "Hello, world!"}
        >>> action = "hello there world"
        >>> output = f1_reward_fn(task_info, action)
        >>> print(output.reward)  # F1 score between the texts
    """
    if isinstance(action, Action):
        action = action.action

    # Extract gold text from task_info
    gold_text = task_info.get("ground_truth", "")
    if gold_text is None:
        gold_text = ""

    def normalize_text(s: str) -> str:
        """Normalize text for evaluation (following HotpotQA/SQuAD standards)"""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    # Normalize and tokenize both texts
    predicted_normalized = normalize_text(str(action))
    gold_normalized = normalize_text(str(gold_text))
    predicted_tokens = predicted_normalized.split()
    gold_tokens = gold_normalized.split()

    # Handle empty cases - if neither predicted nor gold are passed, return 0
    if not predicted_tokens and not gold_tokens:
        f1_score = 0.0
    elif not predicted_tokens or not gold_tokens:
        f1_score = 0.0
    else:
        # Compute token overlap using Counter intersection
        predicted_counter = Counter(predicted_tokens)
        gold_counter = Counter(gold_tokens)
        common = predicted_counter & gold_counter
        num_same = sum(common.values())

        if num_same == 0:
            f1_score = 0.0
        else:
            precision = num_same / len(predicted_tokens)
            recall = num_same / len(gold_tokens)
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return RewardOutput(reward=f1_score, metadata={})
