set -x

source /data/wangsl/envs/rllm/bin/activate
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export E2B_DOMAIN=sandbox.ppio.cn
# Open-source sanitization: hardcoded E2B_API_KEY export removed.

export WANDB_ENTITY="wangshouli0801-Harbin Institute of Technology"
# Open-source sanitization: hardcoded W&B login token and login command removed.

proj_name=reward_hacking_mre
exp_name=$(basename "$0" .sh)-no_rejection_sample-n4-ppio-bs16

# train_files=/root/aicloud-fs/wangshouli/datasets/R2E_Gym_Lite/train_verl.parquet
# val_files=/root/aicloud-fs/wangshouli/datasets/SWE_Bench_Verified/test_verl.parquet
# model_path=/root/aicloud-fs/wangshouli/models/Qwen3-8B
train_files=/data/wangsl/datasets/R2E_Gym_Subset/train_verl.parquet
val_files=/data/wangsl/datasets/SWE_Bench_Verified/test_verl.parquet
model_path=/data/MODEL/Qwen3-8B

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    rllm.rejection_sample.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.log_filter=['e2b','httpx'] \
    trainer.project_name=$proj_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    rllm.env.name=swe \
    +rllm.env.env_args.backend=ppio \
    +rllm.agent.engine_args.n_parallel_agents=32 \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=20 \
    rllm.agent.overlong_filter=False \
    rllm.agent.trajectory_timeout=5400 \
    trainer.total_epochs=1000 \
    trainer.log_episodes=1 \
    trainer.default_local_dir=/data/wangsl/checkpoints/${proj_name}/${exp_name}
