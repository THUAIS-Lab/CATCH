# Environment settings must be available before third-party imports.
# ruff: noqa: E402
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

import hydra

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer
from functools import partial


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # train_dataset = DatasetRegistry.load_dataset("deepcoder", "train")
    # test_dataset = DatasetRegistry.load_dataset("deepcoder", "test")
    # we load dataset by setting data.train_files and data.val_files in the config

    env_args = {"reward_fn": partial(code_reward_fn, config=config), "get_full_reward_output": True}

    trainer = AgentTrainer(
        agent_class=CompetitionCodingAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=None,
        val_dataset=None,
    )
    trainer.train()


if __name__ == "__main__":
    main()
