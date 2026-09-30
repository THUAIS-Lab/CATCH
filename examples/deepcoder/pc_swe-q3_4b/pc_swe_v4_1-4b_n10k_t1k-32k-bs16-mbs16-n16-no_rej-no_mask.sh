#!/usr/bin/env bash

# Run from the repository root; Slurm preserves the submission directory.
set +x
cd -- "${SLURM_SUBMIT_DIR:-$PWD}" || exit

# Load .env defaults without overriding exported variables or logging credentials.
_env_exports="$(python3 - <<'PYENV'
import os
import shlex
from dotenv import load_dotenv

existing = set(os.environ)
load_dotenv(".env", override=False)
for key in sorted(set(os.environ) - existing):
    print("export " + shlex.quote(key + "=" + os.environ[key]))
PYENV
)" || exit
eval "$_env_exports" || exit
unset _env_exports

# === Script-specific configuration ===
# Optional per-run setting; never share a run ID between experiments.
# export WANDB_RUN_ID=xxxxx
exp_name=$(basename "$0" .sh)
export CUDA_VISIBLE_DEVICES=0,1,2,3

train_files="$DEEPCODER_SWE_V4_1_TRAIN_FILE"
val_files="$DEEPCODER_SWE_V4_1_VAL_FILE"
model_path="$SFT_N10K_T1K_MODEL_PATH"
output_dir="${CHECKPOINT_ROOT}/$proj_name/$exp_name"

# === Runtime setup ===
wandb login "$WANDB_API_KEY"
set -x
ulimit -n 1048576

python3 -m examples.deepcoder.train_deepcoder \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=16 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=28672 \
    +data.filter_overlong_initial_prompts=True \
    data.seed=42 \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.log_filter=['e2b','httpx'] \
    trainer.project_name=$proj_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    +trainer.train_generations_to_log_to_wandb=0 \
    +trainer.val_generations_to_log_to_wandb=0 \
    +trainer.train_generations_to_log_to_lark="$TRAIN_LARK_SAMPLES" \
    +trainer.val_generations_to_log_to_lark="$VAL_LARK_SAMPLES" \
    rllm.agent.max_steps=1 \
    rllm.stepwise_advantage.enable=False \
    rllm.rejection_sample.enable=False \
    +rllm.env.env_args.get_full_reward_output=True \
    trainer.total_epochs=100 \
    trainer.default_local_dir="$output_dir"
