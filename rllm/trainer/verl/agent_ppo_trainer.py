import asyncio
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread
from collections import defaultdict
from typing import Dict, Optional, Iterator

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    compute_advantage,
    compute_response_mask,
)
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine


def _summarize_reward_hack_metrics(reward_outputs, *, prefix: str) -> dict[str, float]:
    reward_items = list(reward_outputs)
    if not reward_items:
        return {}

    metrics = {}
    split_name = {"critic": "train", "val": "val"}.get(prefix, prefix)
    binary_mean_fields = {
        "all_passed_easy": "all_passed_easy",
        "all_passed_hard": "all_passed_hard",
        "all_passed": "all_passed",
        "reward_cache_bonus": "with_cache_bonus",
        "reward_format": "with_format_reward",
    }

    reward_wo_hack_values = []
    reward_w_hack_time_s_values = []
    reward_wo_hack_time_s_values = []
    hack_indicators = []
    trivial_hack_indicators = []
    nontrivial_hack_indicators = []
    hack_reward_values = []
    trivial_reward_values = []
    nontrivial_reward_values = []
    binary_field_sums = defaultdict(float)
    binary_field_seen = {field_name: False for field_name in binary_mean_fields}
    for item in reward_items:
        metadata = item.get("metadata", {})

        for field_name in binary_mean_fields:
            field_value = metadata.get(field_name)
            if field_value is None:
                continue
            binary_field_seen[field_name] = True
            if field_name == "reward_cache_bonus":
                binary_field_sums[field_name] += float(float(metadata.get("reward_cache_bonus", 0.0)) > 0.0)
            elif field_name == "reward_format":
                binary_field_sums[field_name] += float(float(metadata.get("reward_format", 0.0)) > 0.0)
            else:
                binary_field_sums[field_name] += float(field_value != 0)

        reward_wo_hack = metadata.get("reward_wo_hack")
        if reward_wo_hack is None:
            continue

        reward_wo_hack_float = float(reward_wo_hack)
        reward_wo_hack_values.append(reward_wo_hack_float)

        reward_w_hack_time_s = metadata.get("reward_w_hack_time_s")
        if reward_w_hack_time_s is not None:
            reward_w_hack_time_s_values.append(float(reward_w_hack_time_s))

        reward_wo_hack_time_s = metadata.get("reward_wo_hack_time_s")
        if reward_wo_hack_time_s is not None:
            reward_wo_hack_time_s_values.append(float(reward_wo_hack_time_s))

        reward_w_hack = float(metadata.get("reward_w_hack", item.get("reward", 0.0)))
        is_hack = metadata.get("is_hack")
        if is_hack is None:
            is_hack = reward_w_hack != 0.0 and reward_wo_hack_float == 0.0
        trivial_hack = bool(metadata.get("trivial_hack", False))
        nontrivial_hack = bool(metadata.get("nontrivial_hack", False))

        hack_indicators.append(float(bool(is_hack)))
        trivial_hack_indicators.append(float(trivial_hack))
        nontrivial_hack_indicators.append(float(nontrivial_hack))

        if is_hack:
            hack_reward_values.append(reward_w_hack)
        if trivial_hack:
            trivial_reward_values.append(reward_w_hack)
        if nontrivial_hack:
            nontrivial_reward_values.append(reward_w_hack)

    if reward_wo_hack_values:
        metrics[f"hack/{split_name}/reward_wo_hack"] = float(np.mean(reward_wo_hack_values))
        metrics[f"hack/{split_name}/hack_rate"] = float(np.mean(hack_indicators))
        metrics[f"hack/{split_name}/trivial_hack_rate"] = float(np.mean(trivial_hack_indicators))
        metrics[f"hack/{split_name}/nontrivial_hack_rate"] = float(np.mean(nontrivial_hack_indicators))
    if hack_reward_values:
        metrics[f"hack/{split_name}/hack_samples_reward/mean"] = float(np.mean(hack_reward_values))
    if trivial_reward_values:
        metrics[f"hack/{split_name}/trivial_samples_reward/mean"] = float(np.mean(trivial_reward_values))
    if nontrivial_reward_values:
        metrics[f"hack/{split_name}/nontrivial_samples_reward/mean"] = float(np.mean(nontrivial_reward_values))
    if prefix == "critic":
        traj_num = len(reward_items)
        for field_name, metric_suffix in binary_mean_fields.items():
            if not binary_field_seen[field_name]:
                continue
            metrics[f"critic/{metric_suffix}/mean"] = float(binary_field_sums[field_name] / traj_num)
    if prefix == "critic" and reward_w_hack_time_s_values:
        metrics["timing_s/reward_w_hack"] = float(np.mean(reward_w_hack_time_s_values))
        metrics["timing_s/reward_w_hack_min"] = float(np.min(reward_w_hack_time_s_values))
        metrics["timing_s/reward_w_hack_max"] = float(np.max(reward_w_hack_time_s_values))
    if prefix == "critic" and reward_wo_hack_time_s_values:
        metrics["timing_s/reward_wo_hack"] = float(np.mean(reward_wo_hack_time_s_values))
        metrics["timing_s/reward_wo_hack_min"] = float(np.min(reward_wo_hack_time_s_values))
        metrics["timing_s/reward_wo_hack_max"] = float(np.max(reward_wo_hack_time_s_values))
    return metrics


def _summarize_llm_monitor_metrics(reward_outputs, *, prefix: str) -> dict[str, float]:
    """Compute LLM monitor metrics: is_hack_rate, acc, precision, recall, f1.

    Only emits metrics when llm_monitor_is_hack / llm_monitor_confusion_tag
    are present in reward output metadata.
    """
    split_name = {"critic": "train", "val": "val"}.get(prefix, prefix)

    is_hack_values = []
    confusion_tags = []

    for item in reward_outputs:
        meta = item.get("metadata", {})
        is_hack = meta.get("llm_monitor_is_hack")
        if is_hack is not None:
            is_hack_values.append(float(bool(is_hack)))
        tag = meta.get("llm_monitor_confusion_tag")
        if tag is not None:
            confusion_tags.append(tag)

    metrics = {}
    if is_hack_values:
        metrics[f"monitor/{split_name}/is_hack_rate"] = float(np.mean(is_hack_values))

    if confusion_tags:
        tp = confusion_tags.count("TP")
        fp = confusion_tags.count("FP")
        tn = confusion_tags.count("TN")
        fn = confusion_tags.count("FN")
        total = tp + fp + tn + fn
        if total > 0:
            metrics[f"monitor/{split_name}/llm_acc"] = (tp + tn) / total
            metrics[f"monitor/{split_name}/llm_precision"] = (
                tp / (tp + fp) if (tp + fp) > 0 else 0.0
            )
            metrics[f"monitor/{split_name}/llm_recall"] = (
                tp / (tp + fn) if (tp + fn) > 0 else 0.0
            )
            prec = metrics[f"monitor/{split_name}/llm_precision"]
            rec = metrics[f"monitor/{split_name}/llm_recall"]
            metrics[f"monitor/{split_name}/llm_f1"] = (
                2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            )

    return metrics


def _summarize_rule_monitor_metrics(reward_outputs, *, prefix: str) -> dict[str, float]:
    """Compute rule-based monitor metrics: hack_method rates among nontrivial
    hack samples and normal_* rates among clean samples. Multi-label hack_method
    strings are split by "+", so a single sample can contribute to multiple
    rates (sum may exceed 1.0). Trivial hacks are excluded from both sides.
    """
    split_name = {"critic": "train", "val": "val"}.get(prefix, prefix)

    normal_methods: list[str] = []
    hack_methods: list[str] = []
    nontrivial_count = 0
    normal_count = 0
    for item in reward_outputs:
        meta = item.get("metadata", {})
        method_str = meta.get("hack_method", "unknown")
        methods = method_str.split("+")
        if method_str.startswith("normal"):
            normal_methods.extend(methods)
            normal_count += 1
        elif meta.get("nontrivial_hack", False):
            hack_methods.extend(methods)
            nontrivial_count += 1

    hack_types = [
        "exit0", "eq", "sys", "calls.json", "xfail",
        "fixture_helper", "expected_output", "unknown",
    ]
    normal_types = [
        "normal", "normal_eq", "normal_xfail", "normal_exit0",
        "normal_sys", "normal_calls.json", "normal_fixture_helper",
        "normal_expected_output",
    ]

    metrics: dict[str, float] = {}

    if hack_methods:
        for method in hack_types:
            count = hack_methods.count(method)
            key = method.replace(".", "_")
            metrics[f"monitor/{split_name}/rule_{key}_rate"] = count / nontrivial_count

    if normal_methods:
        for method in normal_types:
            count = normal_methods.count(method)
            key = method.replace(".", "_")
            metrics[f"monitor/{split_name}/rule_{key}_rate"] = count / normal_count

    return metrics


def _is_prompt_within_limit(doc, *, env_class, agent_class, base_env_args, full_agent_args,
                            max_steps, tokenizer, chat_parser, max_prompt_length) -> bool:
    from rllm.agents.utils import convert_messages_to_tokens_and_masks
    try:
        extra_info = doc.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        env = env_class.from_dict({**extra_info, **base_env_args})
        observation, info = env.reset()

        agent = agent_class(**full_agent_args)
        agent.reset()
        info["max_steps"] = max_steps
        agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)

        messages = agent.chat_completions
        tokens, _ = convert_messages_to_tokens_and_masks(
            messages, tokenizer=tokenizer, parser=chat_parser,
            contains_first_msg=True, contains_generation_msg=True,
        )
        return len(tokens) <= max_prompt_length
    except Exception as e:
        print(f"Warning: error computing initial prompt length, keeping sample: {e}")
        return True


class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}

        assert self.config.actor_rollout_ref.hybrid_engine, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"

        if self.config.rllm.stepwise_advantage.enable:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

        self.train_template_pool = None
        self.val_template_pool = None
        if self.config.rllm.env.get("env_args", {}).get("backend", "") in ["e2b", "ppio"]:
            train_batch_size = self.config.data.train_batch_size
            train_max_templates = self.config.rllm.env.get("env_args", {}).get("train_template_pool_size", 3 * train_batch_size)
            train_max_build_workers = self.config.rllm.env.get("env_args", {}).get("train_template_max_build_workers", train_batch_size)
            val_batch_size = self.config.data.val_batch_size
            val_max_templates = self.config.rllm.env.get("env_args", {}).get("val_template_pool_size", 2 * val_batch_size)
            val_max_build_workers = self.config.rllm.env.get("env_args", {}).get("val_template_max_build_workers", val_batch_size)
            try:
                from rllm.environments.swe.utils.template_pool import TemplatePool
                self.train_template_pool = TemplatePool(
                    max_templates=train_max_templates,
                    max_build_workers=train_max_build_workers,
                    backend=self.config.rllm.env.get("env_args", {}).get("backend", ""),
                    debug=True,
                )
                self.val_template_pool = TemplatePool(
                    max_templates=val_max_templates,
                    max_build_workers=val_max_build_workers,
                    backend=self.config.rllm.env.get("env_args", {}).get("backend", ""),
                )
                print(f"Initialized {self.config.rllm.env.get('env_args', {}).get('backend', '')} template pool with train_max_templates={train_max_templates}, val_max_templates={val_max_templates}")
            except Exception as e:
                print(f"Warning: Failed to initialize {self.config.rllm.env.get('env_args', {}).get('backend', '')} template pool: {e}")
                self.train_template_pool = None
                self.val_template_pool = None

    def init_workers(self):
        super().init_workers()

        if self.config.data.get("filter_overlong_initial_prompts", False):
            self._filter_overlong_initial_prompts()

        engine_args = OmegaConf.to_container(self.config.rllm.agent.get("engine_args", {})) or {}
        n_parallel_agents = engine_args.pop("n_parallel_agents", None) or self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        print(f"n_parallel_agents: {n_parallel_agents}")

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=self.async_rollout_manager,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.rllm.stepwise_advantage.enable,
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
            **engine_args,
        )

    def _filter_overlong_initial_prompts(self):
        """Pre-filter train/val samples whose initial prompt already exceeds max_prompt_length.

        Simulates env.reset() + agent.update_from_env() for each sample to compute
        the actual initial prompt token count (including system prompt, chat template, etc.),
        then removes samples that would immediately fail the length check during rollout.

        Skipped for backends requiring external services (e2b, ppio).
        """
        from rllm.parser import ChatTemplateParser
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        backend = self.config.rllm.env.get("env_args", {}).get("backend", "")
        if backend in ["e2b", "ppio"]:
            print("Skipping initial prompt length filtering for external backend environments")
            return

        max_prompt_length = self.config.data.max_prompt_length
        chat_parser = ChatTemplateParser.get_parser(
            self.tokenizer,
            disable_thinking=self.config.rllm.get("disable_thinking", False),
        )
        full_agent_args = dict(self.config.rllm.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.rllm.env.get("env_args", {})) | self.env_args

        from functools import partial
        filter_fn = partial(
            _is_prompt_within_limit,
            env_class=self.env_class,
            agent_class=self.agent_class,
            base_env_args=base_env_args,
            full_agent_args=full_agent_args,
            max_steps=self.config.rllm.agent.max_steps,
            tokenizer=self.tokenizer,
            chat_parser=chat_parser,
            max_prompt_length=max_prompt_length,
        )

        # Filter train dataset
        original_len = len(self.train_dataset)
        self.train_dataset.dataframe = self.train_dataset.dataframe.filter(
            filter_fn,
            num_proc=128,
            desc=f"Filtering train initial prompts > {max_prompt_length} tokens",
        )
        filtered_len = len(self.train_dataset)
        print(f"Filtered overlong train prompts: {original_len} -> {filtered_len} ({original_len - filtered_len} removed)")

        if original_len != filtered_len:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
            collate_fn = default_collate_fn
            num_workers = self.config.data["dataloader_num_workers"]
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=train_sampler,
            )
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            if self.config.trainer.total_training_steps is not None:
                self.total_training_steps = self.config.trainer.total_training_steps
            print(f"Rebuilt train dataloader: {len(self.train_dataloader)} batches, {self.total_training_steps} total steps")

        # Filter val dataset
        val_original_len = len(self.val_dataset)
        self.val_dataset.dataframe = self.val_dataset.dataframe.filter(
            filter_fn,
            num_proc=128,
            desc=f"Filtering val initial prompts > {max_prompt_length} tokens",
        )
        val_filtered_len = len(self.val_dataset)
        print(f"Filtered overlong val prompts: {val_original_len} -> {val_filtered_len} ({val_original_len - val_filtered_len} removed)")

        if val_original_len != val_filtered_len:
            collate_fn = default_collate_fn
            num_workers = self.config.data["dataloader_num_workers"]
            val_batch_size = self.config.data.val_batch_size
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
            print(f"Rebuilt val dataloader: {len(self.val_dataloader)} batches")

    def init_envs_and_agents(self, batch: DataProto):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        assert self.agent_class is not None and self.env_class is not None, "Agent and environment classes must be provided"
        env_args: list[dict] = batch.non_tensor_batch["extra_info"].tolist()
        if self.config.rllm.env.get("env_args", {}).get("backend", "") in ["e2b", "ppio"]:
            template_ids: list[str|None] = batch.non_tensor_batch['template_id'].tolist()
            for template_id, env_arg in zip(template_ids, env_args):
                env_arg.update({
                    'template_id': template_id
                })


        full_agent_args = dict(self.config.rllm.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.rllm.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            if self.config.rllm.env.get("env_args", {}).get("backend", "") in ["e2b", "ppio"]:
                if template_ids[i] is None:
                    return i, None
            # backend=e2b is already in base_env_args
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            if self.config.rllm.env.get("env_args", {}).get("backend", "") in ["e2b", "ppio"]:
                if template_ids[i] is None:
                    return i, None
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def fill_batch_queue(self, batch_queue: Queue[DataProto], batch_dict_iter: Iterator[torch.utils.data.DataLoader], template_pool=None):
        try:
            if self.config.rllm.env.get("env_args", {}).get("backend", "") in ["e2b", "ppio"]:
                assert template_pool is not None
                while not template_pool.is_build_queue_full():
                    batch_dict = next(batch_dict_iter)
                    single_batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch_queue.put(single_batch)
                    template_pool.build_template_async(single_batch)
            else:
                batch_dict = next(batch_dict_iter)
                single_batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch_queue.put(single_batch)
        except StopIteration as e:
            raise e
 
    def _truncate_samples(self, samples: list[str], max_size: int=35000):
        for idx, s in enumerate(samples):
            if len(s) > max_size:
                samples[idx] = s[:max_size] + "...[TOO LONG, TRUNCATED] The remaining length is " + str(len(s) - max_size) + " characters."
        return samples

    def _reshape_rows_to_lark_max_size(self, rows):
        max_size = 40000
        new_rows = []
        for row in rows:
            rest_lengths = [len(str(cell)) for cell in row]
            offsets = [0] * len(row)
            while any([l > 0 for l in rest_lengths]):
                new_row = []
                for i, (rest_length, offset) in enumerate(zip(rest_lengths, offsets)):
                    if not isinstance(row[i], str) and offsets[i] == 0:
                        new_row.append(row[i])
                        rest_lengths[i] = 0
                        offsets[i] = -1
                        continue
                    elif rest_length > 0:
                        new_row.append(row[i][offset:offset+max_size])
                        offsets[i] += max_size
                        rest_lengths[i] = len(row[i]) - offsets[i]
                    else:
                        new_row.append('')
                new_rows.append(new_row)
        return new_rows

    def _maybe_log_val_generations_to_wandb(self, inputs, scores, passed_tests_num, total_tests_num, outputs, model_codes, tests,
                                            detailed_results, data_sources, table_attr_name, table_name):
        generations_to_log = self.config.trainer.get("val_generations_to_log_to_wandb", 0)

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found.'
            )
            return

        import wandb
        import numpy as np

        tests = self._truncate_samples(tests)
        detailed_results = self._truncate_samples(detailed_results)

        # Create tuples of (input, output, score) and sort by input text
        # samples = list(zip(inputs, outputs, scores))
        samples = list(
            zip(
                inputs,
                scores,
                passed_tests_num,
                total_tests_num,
                outputs,
                model_codes,
                tests,
                detailed_results,
                data_sources,
            )
        )
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum(
            [
                [
                    f"{i+1}_inputs",
                    f"{i+1}_scores",
                    f"{i+1}_passed_tests_num",
                    f"{i+1}_total_tests_num",
                    f"{i+1}_outputs",
                    f"{i+1}_model_codes",
                    f"{i+1}_tests",
                    f"{i+1}_detailed_results",
                    f"{i+1}_data_sources",
                ]
                for i in range(len(samples))
            ], []
        )

        # if not hasattr(self, 'validation_table'):
        #     # Initialize the table on first call
        #     self.validation_table = wandb.Table(columns=columns)

        if not hasattr(self, table_attr_name):
            # Initialize the table on first call
            setattr(self, table_attr_name, wandb.Table(columns=columns))

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        # new_table = wandb.Table(columns=columns, data=self.validation_table.data)
        new_table = wandb.Table(columns=columns, data=getattr(self, table_attr_name).data)


        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        # wandb.log({"val/generations": new_table}, step=self.global_steps)
        wandb.log({f"val/{table_name}": new_table}, step=self.global_steps)
        # self.validation_table = new_table
        setattr(self, table_attr_name, new_table)

    def _maybe_log_val_generations_to_lark(self, inputs, scores, passed_tests_num, total_tests_num, outputs, model_codes, tests,
                                            detailed_results, data_sources, is_hack, is_trivial_hack, hack_methods,
                                            llm_monitor_is_hack=None, llm_monitor_confusion_tag=None,
                                            llm_monitor_raw_response=None,
                                            table_attr_name=None, table_name=None, mode="sampling"):
        generations_to_log = self.config.trainer.get("val_generations_to_log_to_lark", 0)

        if generations_to_log == 0:
            return

        import numpy as np

        import rllm.utils.lark as lark

        llm_monitor_is_hack = llm_monitor_is_hack or []
        llm_monitor_confusion_tag = llm_monitor_confusion_tag or []
        llm_monitor_raw_response = llm_monitor_raw_response or []

        tests = self._truncate_samples(tests)
        detailed_results = self._truncate_samples(detailed_results)

        # Create tuples of (input, output, score) and sort by input text
        samples = list(
            zip(
                inputs,
                scores,
                passed_tests_num,
                total_tests_num,
                outputs,
                model_codes,
                tests,
                detailed_results,
                data_sources,
                is_hack,
                is_trivial_hack,
                hack_methods,
                llm_monitor_is_hack,
                llm_monitor_confusion_tag,
                llm_monitor_raw_response,
            )
        )
        samples.sort(key=lambda x: x[0])  # Sort by input text

        if mode == "sampling":
            # Use fixed random seed for deterministic shuffling
            rng = np.random.RandomState(42)
            rng.shuffle(samples)

            # Take first N samples after shuffling
            samples = samples[:generations_to_log]
        elif mode == "same_sampling":
            grouped_samples = defaultdict(list)
            for sample in samples:
                grouped_samples[sample[0]].append(sample)

            unique_inputs = sorted(grouped_samples.keys())
            rng = np.random.RandomState(42)
            rng.shuffle(unique_inputs)

            selected_samples = []
            for key in unique_inputs:
                if len(selected_samples) >= generations_to_log:
                    break
                selected_samples.extend(grouped_samples[key])
            samples = selected_samples
        elif mode == "all":
            pass
        else:
            raise ValueError(f"Invalid mode {mode} for logging val generations to lark. Valid modes are 'sampling', 'same_sampling' and 'all'.")

        # Create column names for all samples
        columns = ["index"] + [
                    "inputs",
                    "scores",
                    "passed_tests_num",
                    "total_tests_num",
                    "outputs",
                    "model_codes",
                    "tests",
                    "detailed_results",
                    "data_sources",
                    "is_hack",
                    "is_trivial_hack",
                    "hack_methods",
                    "llm_monitor_is_hack",
                    "llm_monitor_confusion_tag",
                    "llm_monitor_raw_response",
                ]

        if not hasattr(self, table_attr_name):
            setattr(self, table_attr_name, lark.create_spreadsheet(title=f'val/{table_name}', folder_token=self.lark_run_folder_token))
            lark.auth_full_access(token=getattr(self, table_attr_name).spreadsheet_token, repo_type='sheet')
        
        sheet_id = lark.add_sheet(spreadsheet_token=getattr(self, table_attr_name).spreadsheet_token, sheet_title=f'step-{self.global_steps}')

        rows = [columns] + np.concatenate(
            [
                np.array(list(range(1, len(samples)+1)), dtype=object).reshape(-1, 1),
                np.array(samples, dtype=object)
            ],
            axis=1
        ).tolist()
        reshaped_rows = self._reshape_rows_to_lark_max_size(rows)

        lark.append_rows(spreadsheet_token=getattr(self, table_attr_name).spreadsheet_token,
                         sheet_id=sheet_id, rows=reshaped_rows)
        lark.set_row_height(spreadsheet_token=getattr(self, table_attr_name).spreadsheet_token,
                            sheet_id=sheet_id, row_height=27, start_index=1, end_index=len(reshaped_rows))

    def _maybe_log_train_generations_to_wandb(self, inputs, scores, passed_tests_num, total_tests_num,
                                              entropies, advantages, outputs, model_codes, tests, detailed_results, data_sources,
                                              table_attr_name, table_name, mode="varied_instruction"):
        generations_to_log = self.config.trainer.get("train_generations_to_log_to_wandb", 0)

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `train_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found.'
            )
            return

        import wandb
        import numpy as np

        tests = self._truncate_samples(tests)
        detailed_results = self._truncate_samples(detailed_results)

        # Create tuples of (input, output, score) and sort by input text
        samples = list(
            zip(
                inputs,
                scores,
                passed_tests_num,
                total_tests_num,
                entropies,
                advantages,
                outputs,
                model_codes,
                tests,
                detailed_results,
                data_sources,
            )
        )
        samples.sort(key=lambda x: x[0])  # Sort by input text

        max_samples = generations_to_log

        if mode == "varied_instruction":
            # Use fixed random seed for deterministic shuffling
            rng = np.random.RandomState(42)
            rng.shuffle(samples)
            # Take first N samples after shuffling
            samples = samples[:max_samples]
        elif mode == "same_instruction":
            # Group samples by instruction
            grouped_samples = defaultdict(list)
            for sample in samples:
                grouped_samples[sample[0]].append(sample)
            rng = np.random.RandomState(42)
            grouped_samples_values = list(grouped_samples.values())
            rng.shuffle(grouped_samples_values)
            # Take first N samples from each instruction
            samples = [
                sample
                for group in grouped_samples_values
                for sample in group
            ][:max_samples]
        else:
            raise ValueError(f"Invalid mode {mode} for logging train generations to wandb. Valid modes are 'varied_instruction' and 'same_instruction'.")

        # Create column names for all samples
        columns = ["step"] + sum(
            [
                [
                    f"{i+1}_inputs",
                    f"{i+1}_scores",
                    f"{i+1}_passed_tests_num",
                    f"{i+1}_total_tests_num",
                    f"{i+1}_entropies",
                    f"{i+1}_advantages",
                    f"{i+1}_outputs",
                    f"{i+1}_model_codes",
                    f"{i+1}_tests",
                    f"{i+1}_detailed_results",
                    f"{i+1}_data_sources",
                ]
                for i in range(max_samples)
            ],
            [],
        )

        # if not hasattr(self, 'train_table'):
        if not hasattr(self, table_attr_name):
            # Initialize the table on first call
            # getattr(self, table_attr_name) = 
            setattr(self, table_attr_name, wandb.Table(columns=columns))
            # self.train_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        # new_table = wandb.Table(columns=columns, data=self.train_table.data)
        new_table = wandb.Table(
            columns=columns, data=getattr(self, table_attr_name).data
        )

        # Add new row with all data
        row_data = [self.global_steps]

        num_fields_per_sample = 11
        for i in range(max_samples):
            if i < len(samples):
                row_data.extend(samples[i])
            else:
                row_data.extend([None] * num_fields_per_sample)

        new_table.add_data(*row_data)

        # Update reference and log
        # wandb.log({"train/generations": new_table}, step=self.global_steps)
        # self.train_table = new_table
        wandb.log({f"train/{table_name}": new_table}, step=self.global_steps)
        # getattr(self, table_attr_name) = new_table
        setattr(self, table_attr_name, new_table)

    def _maybe_log_train_generations_to_lark(self, inputs, scores, passed_tests_num, total_tests_num,
                                              entropies, advantages, outputs, model_codes, tests, detailed_results, data_sources,
                                              is_hack, is_trivial_hack, hack_methods,
                                              llm_monitor_is_hack=None, llm_monitor_confusion_tag=None,
                                              llm_monitor_raw_response=None,
                                              table_attr_name=None, table_name=None, mode="varied_instruction"):
        generations_to_log = self.config.trainer.get("train_generations_to_log_to_lark", 0)

        if generations_to_log == 0:
            return

        import numpy as np

        import rllm.utils.lark as lark

        tests = self._truncate_samples(tests)
        detailed_results = self._truncate_samples(detailed_results)

        llm_monitor_is_hack = llm_monitor_is_hack or []
        llm_monitor_confusion_tag = llm_monitor_confusion_tag or []
        llm_monitor_raw_response = llm_monitor_raw_response or []

        # Create tuples of (input, output, score) and sort by input text
        samples = list(
            zip(
                inputs,
                scores,
                passed_tests_num,
                total_tests_num,
                entropies,
                advantages,
                outputs,
                model_codes,
                tests,
                detailed_results,
                data_sources,
                is_hack,
                is_trivial_hack,
                hack_methods,
                llm_monitor_is_hack,
                llm_monitor_confusion_tag,
                llm_monitor_raw_response,
            )
        )
        samples.sort(key=lambda x: x[0])  # Sort by input text

        max_samples = generations_to_log

        if mode == "varied_instruction":
            # Use fixed random seed for deterministic shuffling
            rng = np.random.RandomState(42)
            rng.shuffle(samples)
            # Take first N samples after shuffling
            samples = samples[:max_samples]
        elif mode == "same_instruction":
            # Group samples by instruction
            grouped_samples = defaultdict(list)
            for sample in samples:
                grouped_samples[sample[0]].append(sample)
            rng = np.random.RandomState(42)
            grouped_samples_values = list(grouped_samples.values())
            rng.shuffle(grouped_samples_values)
            # Take first N samples from each instruction
            samples = [
                sample
                for group in grouped_samples_values
                for sample in group
            ][:max_samples]
        elif mode == "all_instruction":
            pass
        else:
            raise ValueError(f"Invalid mode {mode} for logging train generations to lark. Valid modes are 'varied_instruction', 'same_instruction', and 'all_instruction'.")

        # Create column names for all samples
        columns = ["index"] + [
                    "inputs",
                    "scores",
                    "passed_tests_num",
                    "total_tests_num",
                    "entropies",
                    "advantages",
                    "outputs",
                    "model_codes",
                    "tests",
                    "detailed_results",
                    "data_sources",
                    "is_hack",
                    "is_trivial_hack",
                    "hack_methods",
                    "llm_monitor_is_hack",
                    "llm_monitor_confusion_tag",
                    "llm_monitor_raw_response",
                ]

        if not hasattr(self, table_attr_name):
            setattr(self, table_attr_name, lark.create_spreadsheet(title=f'train/{table_name}', folder_token=self.lark_run_folder_token))
            lark.auth_full_access(token=getattr(self, table_attr_name).spreadsheet_token, repo_type='sheet')
        
        sheet_id = lark.add_sheet(spreadsheet_token=getattr(self, table_attr_name).spreadsheet_token, sheet_title=f'step-{self.global_steps}')

        rows = [columns] + np.concatenate(
            [
                np.array(list(range(1, len(samples)+1)), dtype=object).reshape(-1, 1),
                np.array(samples, dtype=object)
            ],
            axis=1
        ).tolist()
        reshaped_rows = self._reshape_rows_to_lark_max_size(rows)

        lark.append_rows(spreadsheet_token=getattr(self, table_attr_name).spreadsheet_token,
                         sheet_id=sheet_id, rows=reshaped_rows)
        lark.set_row_height(spreadsheet_token=getattr(self, table_attr_name).spreadsheet_token,
                            sheet_id=sheet_id, row_height=27, start_index=1, end_index=len(reshaped_rows))

    def _maybe_log_train_nontrivial_hack_samples_to_lark(self, inputs, outputs, entropies, advantages,
                                                          model_codes, hack_methods, reward_w_hacks,
                                                          reward_wo_hacks, detailed_results, tests,
                                                          llm_monitor_is_hack=None, llm_monitor_confusion_tag=None,
                                                          llm_monitor_raw_response=None):
        if self.lark_run_folder_token is None:
            return

        import rllm.utils.lark as lark

        llm_monitor_is_hack = llm_monitor_is_hack or []
        llm_monitor_confusion_tag = llm_monitor_confusion_tag or []
        llm_monitor_raw_response = llm_monitor_raw_response or []

        tests = self._truncate_samples(tests)

        # Filter to only non-trivial hack samples
        samples = list(zip(
            inputs, outputs, entropies, advantages, model_codes,
            hack_methods, reward_w_hacks, reward_wo_hacks, detailed_results, tests,
            llm_monitor_is_hack, llm_monitor_confusion_tag, llm_monitor_raw_response,
        ))

        if not samples:
            return

        columns = [
            "index", "inputs", "outputs", "entropies", "advantages",
            "model_codes", "hack_method", "reward_w_hack", "reward_wo_hack",
            "detailed_results", "tests",
            "llm_monitor_is_hack", "llm_monitor_confusion_tag", "llm_monitor_raw_response",
        ]

        table_attr_name = "lark_train_nontrivial_hack_table"
        if not hasattr(self, table_attr_name):
            setattr(self, table_attr_name, lark.create_spreadsheet(
                title="train/non_trivial_hack_samples",
                folder_token=self.lark_run_folder_token,
            ))
            lark.auth_full_access(
                token=getattr(self, table_attr_name).spreadsheet_token,
                repo_type="sheet",
            )

        sheet_id = lark.add_sheet(
            spreadsheet_token=getattr(self, table_attr_name).spreadsheet_token,
            sheet_title=f"step-{self.global_steps}",
        )

        rows = [columns] + np.concatenate(
            [
                np.array(list(range(1, len(samples) + 1)), dtype=object).reshape(-1, 1),
                np.array(samples, dtype=object),
            ],
            axis=1,
        ).tolist()
        reshaped_rows = self._reshape_rows_to_lark_max_size(rows)

        lark.append_rows(
            spreadsheet_token=getattr(self, table_attr_name).spreadsheet_token,
            sheet_id=sheet_id, rows=reshaped_rows,
        )
        lark.set_row_height(
            spreadsheet_token=getattr(self, table_attr_name).spreadsheet_token,
            sheet_id=sheet_id, row_height=27,
            start_index=1, end_index=len(reshaped_rows),
        )

    def _compute_reward_detail_metrics(self, batch: DataProto):
        reward_outputs = batch.non_tensor_batch['reward_output']
        metrics = _summarize_reward_hack_metrics(reward_outputs, prefix="critic")

        # LLM monitor metrics (only populated when llm_monitor ran)
        metrics.update(_summarize_llm_monitor_metrics(reward_outputs, prefix="critic"))

        # Rule monitor metrics (hack_method distribution among nontrivial hacks)
        metrics.update(_summarize_rule_monitor_metrics(reward_outputs, prefix="critic"))

        if self.config.rllm.get("code_reward_type", "normal") == "example_only":
            passed_tests_nums_original = [
                batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('passed_tests_original', 0)
                for i in range(len(batch.non_tensor_batch['reward_output']))
            ]
            total_tests_nums_original = [
                batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('total_tests_original', 0)
                for i in range(len(batch.non_tensor_batch['reward_output']))
            ]
            score_original = [
                batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('all_passed_original', 0)
                for i in range(len(batch.non_tensor_batch['reward_output']))
            ]
            metrics['critic/score_original/mean'] = np.mean(score_original)
            metrics['critic/score_original/max'] = np.max(score_original)
            metrics['critic/score_original/min'] = np.min(score_original)
        return metrics

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking
        import debugpy
        # debugpy.listen(4242)

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        debug = True
        import logging

        class LoggingFilter(logging.Filter):
            def filter(slf, record):
                need_to_filter = self.config.trainer.get("log_filter", None)
                if need_to_filter is None:
                    return True
                for filter_name in need_to_filter:
                    if record.name.startswith(filter_name):
                        return False

        logging.basicConfig(level=logging.INFO)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.addFilter(LoggingFilter())

        self.lark_run_folder_token = None
        if self.config.trainer.get("train_generations_to_log_to_lark", 0) > 0 or self.config.trainer.get("val_generations_to_log_to_lark", 0) > 0:
            import sys

            import rllm.utils.lark as lark

            # An inactive W&B logger or missing run ID must not prevent Lark logging.
            wandb_run = getattr(sys.modules.get("wandb"), "run", None)
            run_name = getattr(wandb_run, "name", None) or self.config.trainer.get("experiment_name") or "experiment"
            run_id = getattr(wandb_run, "id", None)
            folder_name = f"{run_name}-{run_id}" if run_id else run_name
            self.lark_run_folder_token = lark.create_folder(name=folder_name).token
            lark.auth_full_access(token=self.lark_run_folder_token, repo_type="folder")

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1
        init_global_steps = self.global_steps

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            batch_queue = Queue[DataProto]()
            batch_dict_iter = iter(self.train_dataloader)
            try:
                self.fill_batch_queue(batch_queue, batch_dict_iter, self.train_template_pool)
            except StopIteration:
                continue

            while not batch_queue.empty():
                batch = batch_queue.get_nowait()
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                if self.config.rllm.env.get("env_args", {}).get("backend", "") in ["e2b", "ppio"]:
                    if debug:
                        print(f"[fit_agent] step: {self.global_steps} getting template ids")
                    template_ids = self.train_template_pool.get_template_ids(batch)
                    batch.non_tensor_batch["template_id"] = template_ids.non_tensor_batch["template_id"] # now batch has a non_tensor_batch key named "template_id" for env init
                    self.train_template_pool.mark_batch_template_to_clean(batch)

                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                with marked_timer("step", timing_raw):
                    self.init_envs_and_agents(batch)

                    # False
                    if self.config.rllm.stepwise_advantage.enable:
                        final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.sample_level_repeat(repeat_counts)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                        # batch needs to be padded to divisor of world size, we will pad with everything masked out
                        batch = batch.union(final_gen_batch_output)
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                    else:
                        if debug:
                            print(f"[fit_agent] step: {self.global_steps} generating agent trajectory")
                        final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                        batch = batch.union(final_gen_batch_output)
                        metrics.update(generate_metrics)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.rllm.rejection_sample.enable:
                            # log the actual complete training rewards before rejection sampling
                            token_level_rewards = None  # for metrics calculation
                            if self.config.rllm.stepwise_advantage.enable:
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                non_pad_steps = batch.select_idxs(non_pad_step_indices)
                                is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                token_level_rewards = last_step_batch.batch["token_level_scores"]
                            else:
                                token_level_rewards = batch.batch["token_level_scores"]
                            full_sequence_score = token_level_rewards.sum(-1)
                            metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()
                            if self.config.rllm.get("code_reward_type", "normal") == "example_only":
                                full_sequence_score_original = [
                                    batch.non_tensor_batch["reward_output"][i].get('metadata', {}).get('all_passed_original', 0)
                                    for i in range(len(batch.non_tensor_batch["reward_output"]))
                                ]
                                metrics["critic/full-score_original/mean"] = np.mean(full_sequence_score_original)
                                metrics["critic/full-score_original/max"] = np.max(full_sequence_score_original)
                                metrics["critic/full-score_original/min"] = np.min(full_sequence_score_original)

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            # old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.rllm.stepwise_advantage.enable:
                            if self.config.rllm.stepwise_advantage.mode == "per_step":
                                batch.batch["token_level_rewards"] = batch.batch["mc_returns"]
                                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                            elif self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                other_step_indices = np.where(is_last_step == False)[0]
                                other_step_batch = batch.select_idxs(other_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            else:
                                raise ValueError(f"Stepwise advantage mode {self.config.rllm.stepwise_advantage.mode} not supported")

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                            # remove the padded last steps
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                            # batch = batch.merge(other_step_batch)
                            batch = DataProto.concat([batch, other_step_batch])

                    mask = batch.batch["attention_mask"][:, -1] == 1
                    truncated_num = mask.sum().item()
                    if self.config.rllm.mask_truncated_samples:
                        batch = batch[~mask]
                    metrics["response_length/truncated_num"] = truncated_num
                    metrics["batch/training_group_num"] = len(set(batch.non_tensor_batch["uid"]))

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # batch.batch: ['input_ids', 'attention_mask', 'position_ids', 'responses', 'prompts', 'token_level_scores', 'response_mask', 'old_log_probs', 'entropys', 'token_level_rewards', 'advantages', 'returns']
                    # batch.non_tensor_batch: ['reward_model', 'extra_info', 'raw_prompt_ids', 'index', 'tools_kwargs', 'interaction_kwargs', 'uid']
                    # batch.meta_info: ['temperature', 'global_token_num']
                    # breakpoint()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # if False:
                    # NOTE: len(batch) maybe != train_batch_size
                    input_ids = batch.batch['input_ids']
                    response_mask = batch.batch['response_mask']
                    attention_mask = batch.batch['attention_mask']
                    reward_tensor = batch.batch['token_level_scores']
                    response_length = response_mask.shape[-1]
                    entropys = batch.batch['entropys']
                    advantages = batch.batch['advantages']
                    sample_inputs = [
                        self.tokenizer.decode(input_ids[i][:-response_length][attention_mask[i][:-response_length] == 1], skip_special_tokens=False)
                        for i in range(len(input_ids))
                    ]
                    sample_outputs = [
                        self.tokenizer.decode(input_ids[i][-response_length:][response_mask[i] == 1], skip_special_tokens=False)
                        for i in range(len(input_ids))
                    ]
                    sample_model_codes = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('model_code', None)
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_tests = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('tests', None)
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_detailed_results = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('detailed_results', None)
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_passed_tests_num = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('passed_tests', 'nan')
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_total_tests_num = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('total_tests', 'nan')
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_is_hack = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('is_hack', False)
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_is_trivial_hack = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('trivial_hack', False)
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_hack_methods = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('hack_method', 'none')
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_llm_monitor_is_hack = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('llm_monitor_is_hack', 'N/A')
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_llm_monitor_confusion_tag = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('llm_monitor_confusion_tag', 'N/A')
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    sample_llm_monitor_raw_response = [
                        str(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('llm_monitor_raw_response', 'N/A')
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    batch_data_sources = batch.non_tensor_batch.get("data_source", [])
                    if not batch_data_sources:
                        batch_data_sources = [
                            batch.non_tensor_batch['extra_info'][i].get('data_source', "unknown")
                            for i in range(reward_tensor.shape[0])
                        ]
                    if self.global_steps % self.config.trainer.get("log_train_generations_to_wandb_freq", 10) == 0 or self.global_steps - init_global_steps < 3:
                        # breakpoint()
                        self._maybe_log_train_generations_to_wandb(inputs=sample_inputs, scores=reward_tensor.sum(-1).tolist(),
                                                                   passed_tests_num=sample_passed_tests_num, total_tests_num=sample_total_tests_num,
                                                                   entropies=verl_F.masked_mean(entropys, response_mask, axis=-1).tolist(),
                                                                   advantages=verl_F.masked_mean(advantages, response_mask, axis=-1).tolist(),
                                                                   outputs=sample_outputs, model_codes=sample_model_codes,
                                                                   tests=sample_tests, detailed_results=sample_detailed_results,
                                                                   data_sources=batch_data_sources,
                                                                   table_attr_name='train_table', table_name='generations_same_instruction', mode="same_instruction")
                        self._maybe_log_train_generations_to_wandb(inputs=sample_inputs, scores=reward_tensor.sum(-1).tolist(),
                                                                   passed_tests_num=sample_passed_tests_num, total_tests_num=sample_total_tests_num,
                                                                   entropies=verl_F.masked_mean(entropys, response_mask, axis=-1).tolist(),
                                                                   advantages=verl_F.masked_mean(advantages, response_mask, axis=-1).tolist(),
                                                                   outputs=sample_outputs, model_codes=sample_model_codes,
                                                                   tests=sample_tests, detailed_results=sample_detailed_results,
                                                                   data_sources=batch_data_sources,
                                                                   table_attr_name='train_table_2', table_name='generations_varied_instruction', mode="varied_instruction")
                        self._maybe_log_train_generations_to_lark(inputs=sample_inputs, scores=reward_tensor.sum(-1).tolist(),
                                                                   passed_tests_num=sample_passed_tests_num, total_tests_num=sample_total_tests_num,
                                                                   entropies=verl_F.masked_mean(entropys, response_mask, axis=-1).tolist(),
                                                                   advantages=verl_F.masked_mean(advantages, response_mask, axis=-1).tolist(),
                                                                   outputs=sample_outputs, model_codes=sample_model_codes,
                                                                   tests=sample_tests, detailed_results=sample_detailed_results,
                                                                   data_sources=batch_data_sources,
                                                                   is_hack=sample_is_hack,
                                                                   is_trivial_hack=sample_is_trivial_hack,
                                                                   hack_methods=sample_hack_methods,
                                                                   llm_monitor_is_hack=sample_llm_monitor_is_hack,
                                                                   llm_monitor_confusion_tag=sample_llm_monitor_confusion_tag,
                                                                   llm_monitor_raw_response=sample_llm_monitor_raw_response,
                                                                   table_attr_name='lark_train_table', table_name='generations_same_instruction', mode="same_instruction")
                        self._maybe_log_train_generations_to_lark(inputs=sample_inputs, scores=reward_tensor.sum(-1).tolist(),
                                                                   passed_tests_num=sample_passed_tests_num, total_tests_num=sample_total_tests_num,
                                                                   entropies=verl_F.masked_mean(entropys, response_mask, axis=-1).tolist(),
                                                                   advantages=verl_F.masked_mean(advantages, response_mask, axis=-1).tolist(),
                                                                   outputs=sample_outputs, model_codes=sample_model_codes,
                                                                   tests=sample_tests, detailed_results=sample_detailed_results,
                                                                   data_sources=batch_data_sources,
                                                                   is_hack=sample_is_hack,
                                                                   is_trivial_hack=sample_is_trivial_hack,
                                                                   hack_methods=sample_hack_methods,
                                                                   llm_monitor_is_hack=sample_llm_monitor_is_hack,
                                                                   llm_monitor_confusion_tag=sample_llm_monitor_confusion_tag,
                                                                   llm_monitor_raw_response=sample_llm_monitor_raw_response,
                                                                   table_attr_name='lark_train_table_2', table_name='generations_varied_instruction', mode="varied_instruction")
                        # self._maybe_log_train_generations_to_lark(inputs=sample_inputs, scores=reward_tensor.sum(-1).tolist(),
                        #                                            passed_tests_num=sample_passed_tests_num, total_tests_num=sample_total_tests_num,
                        #                                            entropies=verl_F.masked_mean(entropys, response_mask, axis=-1).tolist(),
                        #                                            advantages=verl_F.masked_mean(advantages, response_mask, axis=-1).tolist(),
                        #                                            outputs=sample_outputs, model_codes=sample_model_codes,
                        #                                            tests=sample_tests, detailed_results=sample_detailed_results,
                        #                                            data_sources=batch_data_sources,
                        #                                            table_attr_name='lark_train_table_3', table_name='generations_all_instruction', mode="all_instruction")

                    # --- Log non-trivial hack samples to lark ---
                    sample_nontrivial_hack = [
                        bool(
                            batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('nontrivial_hack', False)
                        )
                        for i in range(len(batch.non_tensor_batch['reward_output']))
                    ]
                    nontrivial_hack_indices = [i for i, v in enumerate(sample_nontrivial_hack) if v]
                    if nontrivial_hack_indices:
                        sample_hack_methods = [
                            str(
                                batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('hack_method', 'none')
                            )
                            for i in nontrivial_hack_indices
                        ]
                        sample_reward_w_hack = [
                            str(
                                batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('reward_w_hack', 'nan')
                            )
                            for i in nontrivial_hack_indices
                        ]
                        sample_reward_wo_hack = [
                            str(
                                batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('reward_wo_hack', 'nan')
                            )
                            for i in nontrivial_hack_indices
                        ]
                        self._maybe_log_train_nontrivial_hack_samples_to_lark(
                            inputs=[sample_inputs[i] for i in nontrivial_hack_indices],
                            outputs=[sample_outputs[i] for i in nontrivial_hack_indices],
                            entropies=[verl_F.masked_mean(entropys, response_mask, axis=-1).tolist()[i] for i in nontrivial_hack_indices],
                            advantages=[verl_F.masked_mean(advantages, response_mask, axis=-1).tolist()[i] for i in nontrivial_hack_indices],
                            model_codes=[sample_model_codes[i] for i in nontrivial_hack_indices],
                            hack_methods=sample_hack_methods,
                            reward_w_hacks=sample_reward_w_hack,
                            reward_wo_hacks=sample_reward_wo_hack,
                            detailed_results=[sample_detailed_results[i] for i in nontrivial_hack_indices],
                            tests=[sample_tests[i] for i in nontrivial_hack_indices],
                            llm_monitor_is_hack=[sample_llm_monitor_is_hack[i] for i in nontrivial_hack_indices],
                            llm_monitor_confusion_tag=[sample_llm_monitor_confusion_tag[i] for i in nontrivial_hack_indices],
                            llm_monitor_raw_response=[sample_llm_monitor_raw_response[i] for i in nontrivial_hack_indices],
                        )

                    # collect metrics
                    metrics.update(self._compute_reward_detail_metrics(batch=batch))
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                self._save_batch_to_disk(batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

                if self.config.rllm.env.get("env_args", {}).get("backend", "") in ["e2b", "ppio"]:
                    self.train_template_pool.clean_marked_template_async()

                try:
                    self.fill_batch_queue(batch_queue, batch_dict_iter, self.train_template_pool)
                except StopIteration:
                    continue

    def _validate_agent(self):
        rewards_lst = []
        reward_output_lst = []
        data_source_lst = []
        uid_lst = []
        test_batch_queue = Queue[DataProto]()
        test_batch_dict_iter = iter(self.val_dataloader)
        try:
            self.fill_batch_queue(test_batch_queue, test_batch_dict_iter, self.val_template_pool)
        except StopIteration:
            pass

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_model_codes = []
        sample_tests = []
        sample_detailed_results = []
        sample_passed_tests_num = []
        sample_total_tests_num = []
        sample_is_hack = []
        sample_is_trivial_hack = []
        sample_hack_methods_val = []
        sample_llm_monitor_is_hack_val = []
        sample_llm_monitor_confusion_tag_val = []
        sample_llm_monitor_raw_response_val = []

        while not test_batch_queue.empty():
            test_batch = test_batch_queue.get_nowait()
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)

            if self.config.rllm.env.get("env_args", {}).get("backend", "") in ["e2b", "ppio"]:
                template_ids = self.val_template_pool.get_template_ids(test_batch) # template_ids may contain None if the template building is failed
                test_batch.non_tensor_batch["template_id"] = template_ids.non_tensor_batch["template_id"] # now batch has a non_tensor_batch key named "template_id" for env init
                self.val_template_pool.mark_batch_template_to_clean(test_batch)

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            # Disable LLM monitor penalization during validation
            import json
            extra_infos = test_batch.non_tensor_batch.get("extra_info", [])
            for i in range(len(extra_infos)):
                if isinstance(extra_infos[i], str):
                    extra_infos[i] = json.loads(extra_infos[i])
                extra_infos[i]["_disable_llm_penalize"] = True
            self.init_envs_and_agents(test_batch)

            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            test_batch = test_batch.union(test_output_gen_batch)
            # test_batch.batch: ['input_ids', 'attention_mask', 'position_ids', 'responses', 'prompts', 'token_level_scores', 'response_mask']
            # test_batch.non_tensor_batch: ['reward_model', 'extra_info', 'raw_prompt_ids', 'index', 'tools_kwargs', 'interaction_kwargs', 'uid']
            # breakpoint()

            reward_tensor = test_batch.batch["token_level_scores"]
            reward_output_lst.extend(list(test_batch.non_tensor_batch['reward_output']))

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            test_batch_data_sources = test_batch.non_tensor_batch.get("data_source", [])
            if not test_batch_data_sources:
                test_batch_data_sources = [
                    test_batch.non_tensor_batch['extra_info'][i].get('data_source', "unknown")
                    for i in range(reward_tensor.shape[0])
                ]
            data_source_lst.append(test_batch_data_sources)
            uid_lst.append(test_batch.non_tensor_batch["uid"])

            input_ids = test_batch.batch['input_ids']
            response_mask = test_batch.batch['response_mask']
            attention_mask = test_batch.batch['attention_mask']
            response_length = response_mask.shape[-1]
            sample_inputs.extend([
                self.tokenizer.decode(input_ids[i][:-response_length][attention_mask[i][:-response_length] == 1], skip_special_tokens=False)
                for i in range(len(input_ids))
            ])
            sample_outputs.extend([
                self.tokenizer.decode(input_ids[i][-response_length:][response_mask[i] == 1], skip_special_tokens=False)
                for i in range(len(input_ids))
            ])
            sample_model_codes.extend([
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('model_code', None)
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ])
            sample_tests.extend([
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('tests', None)
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ])
            sample_detailed_results.extend([
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('detailed_results', None)
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ])
            passed_tests_nums = [
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('passed_tests', 'nan')
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ]
            total_tests_nums = [
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('total_tests', 'nan')
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ]
            sample_passed_tests_num.extend(passed_tests_nums)
            sample_total_tests_num.extend(total_tests_nums)
            sample_is_hack.extend([
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('is_hack', False)
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ])
            sample_is_trivial_hack.extend([
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('trivial_hack', False)
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ])
            sample_hack_methods_val.extend([
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('hack_method', 'none')
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ])
            sample_llm_monitor_is_hack_val.extend([
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('llm_monitor_is_hack', 'N/A')
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ])
            sample_llm_monitor_confusion_tag_val.extend([
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('llm_monitor_confusion_tag', 'N/A')
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ])
            sample_llm_monitor_raw_response_val.extend([
                str(
                    test_batch.non_tensor_batch['reward_output'][i].get('metadata', {}).get('llm_monitor_raw_response', 'N/A')
                )
                for i in range(len(test_batch.non_tensor_batch['reward_output']))
            ])

            if self.config.rllm.env.get("env_args", {}).get("backend", "") in ["e2b", "ppio"]:
                self.val_template_pool.clean_marked_template_async()

            try:
                self.fill_batch_queue(test_batch_queue, test_batch_dict_iter, self.val_template_pool)
            except StopIteration:
                pass

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, scores=reward_tensor.tolist(),
                                                 passed_tests_num=sample_passed_tests_num, total_tests_num=sample_total_tests_num,
                                                 outputs=sample_outputs, model_codes=sample_model_codes,
                                                 tests=sample_tests, detailed_results=sample_detailed_results, data_sources=data_sources.tolist(),
                                                 table_attr_name='val_table', table_name='val_generations')
        self._maybe_log_val_generations_to_lark(inputs=sample_inputs, scores=reward_tensor.tolist(),
                                                 passed_tests_num=sample_passed_tests_num, total_tests_num=sample_total_tests_num,
                                                 outputs=sample_outputs, model_codes=sample_model_codes,
                                                 tests=sample_tests, detailed_results=sample_detailed_results, data_sources=data_sources.tolist(),
                                                 is_hack=sample_is_hack,
                                                 is_trivial_hack=sample_is_trivial_hack,
                                                 hack_methods=sample_hack_methods_val,
                                                 llm_monitor_is_hack=sample_llm_monitor_is_hack_val,
                                                 llm_monitor_confusion_tag=sample_llm_monitor_confusion_tag_val,
                                                 llm_monitor_raw_response=sample_llm_monitor_raw_response_val,
                                                 table_attr_name='lark_val_table', table_name='val_generations_sampling', mode='sampling')
        self._maybe_log_val_generations_to_lark(inputs=sample_inputs, scores=reward_tensor.tolist(),
                                                 passed_tests_num=sample_passed_tests_num, total_tests_num=sample_total_tests_num,
                                                 outputs=sample_outputs, model_codes=sample_model_codes,
                                                 tests=sample_tests, detailed_results=sample_detailed_results, data_sources=data_sources.tolist(),
                                                 is_hack=sample_is_hack,
                                                 is_trivial_hack=sample_is_trivial_hack,
                                                 hack_methods=sample_hack_methods_val,
                                                 llm_monitor_is_hack=sample_llm_monitor_is_hack_val,
                                                 llm_monitor_confusion_tag=sample_llm_monitor_confusion_tag_val,
                                                 llm_monitor_raw_response=sample_llm_monitor_raw_response_val,
                                                 table_attr_name='lark_val_table_2', table_name='val_generations_same_sampling', mode='same_sampling')
        # self._maybe_log_val_generations_to_lark(inputs=sample_inputs, scores=reward_tensor.tolist(),
        #                                          passed_tests_num=sample_passed_tests_num, total_tests_num=sample_total_tests_num,
        #                                          outputs=sample_outputs, model_codes=sample_model_codes,
        #                                          tests=sample_tests, detailed_results=sample_detailed_results, data_sources=data_sources.tolist(), 
        #                                          table_attr_name='lark_val_table_2', table_name='val_generations_all', mode='all')

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        metric_dict = {}
        metric_dict.update(_summarize_reward_hack_metrics(reward_output_lst, prefix="val"))

        # LLM monitor metrics (only populated when llm_monitor ran)
        metric_dict.update(_summarize_llm_monitor_metrics(reward_output_lst, prefix="val"))

        # Rule monitor metrics (hack_method distribution among nontrivial hacks)
        metric_dict.update(_summarize_rule_monitor_metrics(reward_output_lst, prefix="val"))

        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)

        return metric_dict

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}
        with marked_timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.async_rollout_mode:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                raise ValueError("Only async rollout mode is supported")
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with marked_timer("collect_trajectory", timing_raw):
            steps = []
            gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
            for _, trajectory in enumerate(gen_seq_generator):
                steps.append(trajectory)
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(steps, uids=uids)
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict]):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        traj_metrics = []
        metrics = {}

        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["trajectory_reward"])
            chat_completions.append(traj["chat_completions"])
            traj_metrics.append(traj["metrics"])

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            v_list = [v for v in v_list if v is not None and v >= 0]
            if not v_list:
                continue
            v_list = np.array(v_list)
            metrics.update(
                {
                    f"traj/{k}_mean": v_list.mean(),
                    f"traj/{k}_min": v_list.min(),
                    f"traj/{k}_max": v_list.max(),
                }
            )

        # Save chat completions to a file
        save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
        os.makedirs(save_dir, exist_ok=True)
        # Save it into a jsonl files (self.global_steps)
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), "a+") as f:
            for chat_completion in chat_completions:
                f.write(json.dumps(chat_completion) + "\n")

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_initial_tokens_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_response_tokens_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token (e.g., eos token)
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, score in enumerate(traj_scores):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= score_batch.shape[1]:
                score_batch[i, resp_len - 1] = score

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "response_mask": traj_mask,
        }
        non_tensor_batch = None
        if any(['reward_output' in traj for traj in trajectories]):
            non_tensor_batch = {
                "reward_output": [traj.get('reward_output', None) for traj in trajectories],
            }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch), metrics
        # return DataProto.from_dict(tensors=tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="response_mask"):
        """
        Visualize the trajectory from tensor_batch using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key=mask_key,
            reward_key="token_level_scores",
            show_workflow_metadata=False,
        )

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray):
        from verl.utils.torch_functional import pad_sequence_to_length

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_prompts_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_responses_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = attention_mask[:, max_prompt_length:]

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token of each step
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                resp_len = response_lengths[step_index]
                if resp_len > 0 and resp_len <= score_batch.shape[1]:
                    score_batch[step_index, resp_len - 1] = traj_score
                    mc_return_batch[step_index, resp_len - 1] = all_mc_returns[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "response_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            if "is_last_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_last_step"][idx] = False
            if "is_pad_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch

    def _save_batch_to_disk(self, batch: DataProto):
        batchs_dir = os.path.join(self.config.trainer.default_local_dir, "batchs")
        os.makedirs(batchs_dir, exist_ok=True)
        filepath = os.path.join(batchs_dir, f"{self.global_steps}.jsonl")

        input_ids = batch.batch['input_ids']
        response_mask = batch.batch['response_mask']
        attention_mask = batch.batch['attention_mask']
        response_length = response_mask.shape[-1]
        entropys = batch.batch['entropys']
        advantages = batch.batch['advantages']

        with open(filepath, "a+") as f:
            for i in range(len(input_ids)):
                inp = self.tokenizer.decode(
                    input_ids[i][:-response_length][attention_mask[i][:-response_length] == 1],
                    skip_special_tokens=False,
                )
                out = self.tokenizer.decode(
                    input_ids[i][-response_length:][response_mask[i] == 1],
                    skip_special_tokens=False,
                )
                score = batch.batch['token_level_scores'][i].sum().item()
                entropy = verl_F.masked_mean(entropys[i:i+1], response_mask[i:i+1], axis=-1).item()
                adv = verl_F.masked_mean(advantages[i:i+1], response_mask[i:i+1], axis=-1).item()
                meta = batch.non_tensor_batch['reward_output'][i].get('metadata', {})
                extra_info = batch.non_tensor_batch['extra_info'][i]
                if isinstance(extra_info, str):
                    extra_info = json.loads(extra_info)
                data_source = extra_info.get('data_source', 'unknown')
                row = {
                    "input": inp,
                    "output": out,
                    "score": score,
                    "entropy": entropy,
                    "advantage": adv,
                    "is_hack": meta.get("is_hack", False),
                    "is_trivial_hack": meta.get("trivial_hack", False),
                    "hack_method": meta.get("hack_method", "none"),
                    "llm_monitor_is_hack": meta.get("llm_monitor_is_hack", "N/A"),
                    "llm_monitor_confusion_tag": meta.get("llm_monitor_confusion_tag", "N/A"),
                    "llm_monitor_raw_response": meta.get("llm_monitor_raw_response", "N/A"),
                    "detailed_result": str(meta.get("detailed_results", None)),
                    "tests": str(meta.get("tests", None)),
                    "total_tests_num": str(meta.get("total_tests", "nan")),
                    "passed_tests_num": str(meta.get("passed_tests", "nan")),
                    "data_source": data_source,
                }
                f.write(json.dumps(row) + "\n")

    def shutdown(self):
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
