# catch_rl: training and data preparation

Run the Qwen3-4B RL experiments in `pc_swe-q3_4b/` with your prepared
RPC/SWE coding datasets and SFT checkpoints.

**Run all commands below from the repository root**, including `sbatch`.
Activate your Python environment before starting.

## 1. Install dependencies

```bash
pip install -e '.[verl]'
```

Use a CUDA/PyTorch environment compatible with your GPUs and the project's
training dependencies.

## 2. Configure shared settings

Create a local configuration file if you do not already have one:

```bash
test -f .env || cp .env.example .env
```

Edit `.env` and replace the relevant `xxxxx` values:

| Settings | What to provide |
| --- | --- |
| `WANDB_ENTITY`, `WANDB_API_KEY` | Your W&B user/team and API key; training logs are sent to W&B. |
| `DEEPCODER_SWE_V4_1_TRAIN_FILE`, `DEEPCODER_SWE_V4_1_VAL_FILE` | Prepared training and validation parquet files. |
| `CHECKPOINT_ROOT` | Directory for checkpoints and run outputs. |
| `SFT_N10K_T1K_MODEL_PATH`, `SFT_N8K_T3K_MODEL_PATH` | SFT checkpoints matching the selected initialization-data mixture. |
| `DATA_ROOT` | Dataset root used by the preparation scripts. |
| `API_MODEL`, `API_BASE_URL`, `API_KEY` | Model, endpoint, and key for data conversion. Only needed when preparing data. |
| `MONITOR_BASE_URL`, `MONITOR_API_KEY` | Endpoint and key for monitor-enabled RL. Only needed for those recipes. |

The default project name is `catch_rl`. Already-exported environment variables
override values in `.env`. Keep real credentials in your local `.env`, which is
ignored by Git; do not put them in `.env.example`.

## 3. Choose a training recipe

Select a script in `examples/deepcoder/pc_swe-q3_4b/` and check its
**Script-specific configuration** block before running it:

- `model_path` must point to the checkpoint for that recipe's SFT mixture.
- `exp_name` identifies the experiment.
- `CUDA_VISIBLE_DEVICES`, `trainer.n_gpus_per_node`, and any Slurm GPU request
  must match your GPU allocation.
- Set a separate `WANDB_RUN_ID` only when intentionally selecting your own run;
  do not reuse one ID across different experiments.

For the n8k/t3k baseline:

```bash
bash examples/deepcoder/pc_swe-q3_4b/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16.sh
```

For monitor-enabled training:

```bash
bash examples/deepcoder/pc_swe-q3_4b/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16-w_monitor.sh
```

The `-gr1e-2.sh` and `-chi2.sh` recipes select the corresponding regularization
methods. The other filenames distinguish SFT mixtures and reward-weight
settings. Keep the recipe's hyperparameters unless you intend to run a different
experiment.

### Submit with Slurm

Submit from the repository root and use a partition available on your cluster:

```bash
sbatch --partition=YOUR_PARTITION \
    examples/deepcoder/pc_swe-q3_4b/pc_swe_v4_1-4b_n8k_t3k-32k-bs16-mbs16-n16.sh
```

Review the script's `#SBATCH` resource requests first. The job reads `.env` from
the submission directory even when Slurm runs a spool copy of the script.

## 4. Prepare data when needed

Configure the conversion API settings in `.env`, then check `INPUT_PATH` and
`OUTPUT_PATH` at the top of the preparation script you need:

```bash
# Prepare DeepCoder training data.
bash examples/deepcoder/prepare_deepcoder_swe_train_data.sh

# Prepare DeepCoder validation/test data.
bash examples/deepcoder/prepare_deepcoder_swe_test_data.sh

# Prepare Skywork data.
bash examples/deepcoder/prepare_skywork_swe_data.sh
```

Run only the commands needed for your dataset. Conversion calls an external
model API and may incur charges. **Check the output path before running:** an
existing parquet file can be replaced, and a script may use the same input and
output path. Back up any data you need to retain.

The Python preparation scripts also read the repository's `.env` before using
their defaults; explicit CLI arguments override those defaults.

## Optional: Feishu/Lark sample logs

Lark logging is off by default:

```dotenv
TRAIN_LARK_SAMPLES=0
VAL_LARK_SAMPLES=0
```

To enable it, fill in `LARK_APP_ID`, `LARK_APP_SECRET`, and `LARK_OPEN_ID`, then
set the sample counts you want in `.env`. This setting is separate from the LLM
monitor used by monitor-enabled experiments.

## Outputs

By default, each training run writes to:

```text
<CHECKPOINT_ROOT>/catch_rl/<exp_name>/
```

Checkpoints and per-step batch files are saved under that run directory. The
configured W&B project receives training and validation metrics. Choose distinct
experiment names/output directories when you want to keep runs separate.
