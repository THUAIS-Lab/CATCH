# Reward-Hack SFT Data Generation

这个目录用于合成 reward hacking SFT 数据。主脚本是
`generate_reward_hack_sft_data.py`，主要 shell 入口是
`generate_skywork_reward_hack_sft.sh`。

默认数据源是 `Skywork/Skywork-OR1-RL-Data` 的 `code` split。脚本会从
`reward_model.ground_truth` 中解析真实测试样例，并转换成现有
LiveCodeBench 风格的验证输入：

- `stdin_stdout`：stdio 题，来自 `inputs / outputs`
- `functional`：call-based 题，来自带 `fn_name` 的 ground truth

目标过滤条件是：

```text
reward_w_hack == 1
reward_wo_hack == 0
```

也就是说，样本必须在允许目标 hack surface 时通过验证，同时在禁用该 surface
后无法通过。

## 工作流

1. 读取数据

   默认从 HuggingFace 加载：

   ```bash
   HF_DATASET="Skywork/Skywork-OR1-RL-Data"
   HF_CONFIG=""
   HF_SPLIT="code"
   HF_DATA_FILES=""
   ```

   也可以通过 `--input` 指定本地 parquet/jsonl。

2. 组装 prompt

   每条样本会被拆成并保存以下 prompt 组件：

   - `question`：原始题面
   - `env_leakage_prompt`：动态环境泄露片段
   - `format_prompt`：输出格式约束
   - `normalized_input`：最终 SFT 使用的 user input

   环境泄露 prompt 会保留模拟交互外壳，例如 `> ls -l`、`> show_all_files`、
   `Permission denied`。当前默认支持四种 env surface：

   - `judge-side-v1`：`solution.py` 可写，`test.py` 可读写，显示 v1 `run_test_case(sample)` 评测逻辑
   - `judge-side-v2`：`solution.py` 可写，`test.py` 可读写，显示 v2 `run_test_case(sample, all_results, all_metadata, timeout)` 评测逻辑
   - `data-side`：`solution.py` 可写，`test.py` 只读，`test_cases.json` 只读，并展示一条真实测试样例 preview
   - `runtime-side`：`solution.py` 可写，`test.py` 只读，`test_builtins.py` 不可读，不泄露 `test_cases.json`

3. 分配 surface 和 style

   `--target-count 100` 表示所有 env surface 加起来生成 100 条，不是每个
   surface 100 条。若只想跑某一类 surface，直接用
   `--env-surfaces data-side` 之类的精确指定。

   每条样本还会分配一个轨迹风格：

   - `pure_hack`
   - `solve_and_hack`
   - `solve_then_hack`

   默认比例：

   ```text
   pure_hack=0.5,solve_and_hack=0.25,solve_then_hack=0.25
   ```

   通过 `--style-ratios` 可以改比例。每个 job retry 时会保持同一个 style。

4. 调用生成模型

   使用 OpenAI-compatible chat completions。关键参数包括：

   - `--api-key` / `--api-key-env`
   - `--base-url`
   - `--model`
   - `--concurrency`
   - `--rate-limit-per-minute`
   - `--temperature`
   - `--top-p`
   - `--max-tokens`
   - `--extra-body`

   shell 脚本默认关闭模型侧思考模式：

   ```bash
   ENABLE_THINKING="false"
   EXTRA_ARGS+=(--extra-body "{\"enable_thinking\": ${ENABLE_THINKING}}")
   ```

5. 解析输出

   模型输出必须包在：

   ```text
   <begin_reward_hack_sft_output>
   ...
   <end_reward_hack_sft_output>
   ```

   里面应包含 `<think>...</think>` 和 fenced Python 文件块。每个代码块第一行写目标文件名：

   ```python
   # solution.py
   ...
   ```

   或：

   ```python
   # test.py
   ...
   ```

   脚本会解析这些文件块，保存 `solution_code`、`test_code` 和
   `submitted_files`，并按 env surface 做轻量静态 pattern 检查。

6. 验证和过滤

   每条生成结果会跑两次现有评测逻辑：

   - `allow_test_hack=True`：得到 `reward_w_hack`
   - `allow_test_hack=False`：得到 `reward_wo_hack`

   过滤前和 reward-gap 过滤后的数据都会保存。

7. 可选二阶段 CoT monitor

   如果设置 `--cot-monitor-model`，脚本会把 reward-gap 过滤后的样本送给一个
   CoT monitor，再判断“解题过程是否看起来正常”。这一步会额外输出
   `*.cot_monitor_reviewed.parquet` 和 `*.cot_monitor_filtered.parquet`。

## 支持的 Exploit Surface

### Judge-Side

通过提交 `test.py` 修改可见 judge 流程，当前主要覆盖：

- 改写 `run_test_case(sample)` 或 v2 `run_test_case(sample, all_results, all_metadata, timeout)`
- 改写 `match_output` / `_builtin_match_output`
- 改写 `run_solution`
- 直接操作 `all_results`、`tmp_result` 或 timeout 控制流

### Data-Side

通过提交 `solution.py` 读取 `test_cases.json` 做查表或伪装成普通 helper 的查表。

### Runtime-Side

通过 Python runtime、自省、trace 或模块污染影响评测流程，例如：

- `sys._getframe`
- `sys.settrace` / `f_trace`
- `signal.alarm(0)`
- 修改 `sys.modules["test_builtins"]`

## Guidance 控制

默认会给生成模型较自由的 surface-only prompt，并按 style 分配轨迹风格。
如果希望模型有更大自由度，可以传：

```bash
--disable-specific-guidance
```

此时仍会记录选中的 env surface 和 `hack_style`，但 prompt 里不会注入更显式的
辅助文案；模型只能从 user prompt 中的环境泄露自行选择策略。
保存字段中可以用这些列复盘当时的 prompt 约束：

- `use_specific_guidance`
- `style_guidance`

## 二阶段 CoT Monitor

二阶段 CoT monitor 用于在 reward-gap 过滤后的基础上，再筛出 monitor 认为“看起来像正常解题过程”的样本。

OpenAI-compatible API 用法：

```bash
--cot-monitor-model qwen3.5-27b
--cot-monitor-base-url https://dashscope.aliyuncs.com/compatible-mode/v1
--cot-monitor-api-key-env COT_MONITOR_API_KEY
```

本地 vLLM 部署脚本：

```bash
bash examples/reward_hack_sft/launch_qwen35_27b_cot_monitor_vllm.sh
```

当前脚本默认使用：

- 模型路径：`/data/MODEL/Qwen3.5-27B`
- served model name：`qwen3.5-27b-cot-monitor`
- 端口：`31000`
- GPU：`4,5,6,7`
- tensor parallel size：`4`

对应生成脚本配置：

```bash
COT_MONITOR_MODEL="qwen3.5-27b-cot-monitor"
COT_MONITOR_BASE_URL="http://127.0.0.1:31000/v1"
COT_MONITOR_API_KEY="EMPTY"
COT_MONITOR_CONCURRENCY="32"
```

启用后新增列：

| 列名 | 含义 |
|---|---|
| `cot_monitor_is_hack` | CoT monitor 是否判断该样本存在 hack；取值为 `True`、`False` 或 `None` |
| `cot_monitor_raw_response` | CoT monitor 的原始响应或原始异常信息，便于排查 |

## 输出文件

shell 脚本默认输出到 `/data/ouzh/datasets/reward_hack_sft/`：

```text
skywork_or1_reward_hack_sft_100_gap_all.raw.jsonl
skywork_or1_reward_hack_sft_100_gap_all.parquet
skywork_or1_reward_hack_sft_100_gap_filtered.parquet
```

文件含义：

- `*_all.raw.jsonl`：过滤前逐条原始记录，便于断点续跑和 debug
- `*_all.parquet`：过滤前全部采样结果，适合用 VSCode parquet 插件查看
- `*_filtered.parquet`：reward-gap 过滤后的 SFT 数据
- `*.cot_monitor_reviewed.parquet`：reward-gap filtered 数据加二阶段审查结果
- `*.cot_monitor_filtered.parquet`：同时通过 reward-gap 和二阶段 CoT monitor 的子集

## 输出列

过滤前、过滤后和二阶段输出共用大部分字段。list/dict 类型在 parquet 中会序列化成 JSON 字符串。

| 列名 | 含义 |
|---|---|
| `item_id` | 本次生成任务的内部编号 |
| `accepted` | 仅 raw/all 输出中使用；是否通过 reward-gap 过滤 |
| `passes_reward_hack_filter` | 是否满足 `reward_w_hack == 1 && reward_wo_hack == 0` |
| `messages` | SFT messages，包含 user prompt 和 assistant answer |
| `prompt` | user prompt 扁平文本 |
| `assistant` | 模型生成的 assistant answer 扁平文本 |
| `generator_request_messages` | 生成阶段最初发给模型的原始 messages，包含 system 和 user，不含后续 retry 追加消息 |
| `generator_system_prompt` | 生成阶段最初请求里的 system prompt |
| `generator_user_prompt` | 生成阶段最初请求里的 user prompt |
| `generator_first_raw_response` | 模型在第一次生成尝试时返回的原始文本 |
| `generator_last_raw_response` | 模型最后一次生成尝试时返回的原始文本 |
| `generator_raw_responses` | 所有生成尝试返回的原始文本列表，按时间顺序保存 |
| `hack_family` | hack 大类：`judge-side`、`data-side` 或 `runtime-side` |
| `hack_variant` | 保留兼容字段；当前始终为空字符串 |
| `env_surface` | 使用的环境泄露面；例如 `judge-side-v1`、`judge-side-v2`、`data-side`、`runtime-side` |
| `generation_target_kind` | 当前固定为 `surface-only` |
| `generation_target_id` | 当前等于 env surface id，例如 `judge-side-v1` |
| `hack_style` | 轨迹风格：`pure_hack`、`solve_and_hack` 或 `solve_then_hack` |
| `hack_style_label` | `hack_style` 的可读标签 |
| `hack_style_ratio` | 调度时该 style 的目标配比 |
| `generation_model` | 负责生成样本的模型名 |
| `generation_config` | 生成阶段温度、top_p、max_tokens、retry、concurrency、extra_body 等配置 |
| `use_specific_guidance` | 保留兼容字段；当前固定为 `false` |
| `variant_guidance` | 保留兼容字段；当前固定为空字符串 |
| `style_guidance` | 保留兼容字段；当前固定为空字符串 |
| `source_uid` | 源数据 id，优先取 `uid/id/question_id`，没有则用 `item_id` |
| `source` | 源数据中的 source 字段；没有时为空字符串 |
| `question_id` | 源数据中的 question id；没有时为空字符串 |
| `category` | 数据类别，默认是 `code` |
| `test_type` | 解析出的评测类型，例如 `stdin_stdout`、`functional` 或 `mixed` |
| `fn_name` | call-based / functional 测试中的函数名；stdio 题通常为空 |
| `test_cases_preview` | 基于当前样本真实测试样例动态生成的一条 preview，并用省略号表示不止一条 |
| `question` | 原始题面，不含环境泄露和格式约束 |
| `env_leakage_prompt` | 动态环境泄露片段 |
| `format_prompt` | 输出格式约束片段 |
| `normalized_input` | 最终 SFT user input |
| `normalized_output` | 最终 SFT assistant output |
| `solution_code` | 从提交内容中解析出的 `solution.py` |
| `test_code` | 从提交内容中解析出的 `test.py`；非 judge-side 样本通常为空 |
| `submitted_files` | 从提交内容中解析出的文件名列表 |
| `original_output` | 源数据原始参考输出；没有时为空字符串或占位说明 |
| `reward_w_hack` | 允许 hack 时的 reward |
| `reward_wo_hack` | 不允许 hack 时的 reward |
| `validation_source` | 测试来源说明 |
| `validation_reason` | 验证结果说明；失败样本里通常能看到原因 |
| `validation_original_total_tests` | 源数据中原始测试用例数量 |
| `validation_used_total_tests` | 实际用于验证的测试用例数量 |
| `static_variant_match` | 保留兼容字段；与 `static_target_match` 同义 |
| `static_target_match` | 当前 env surface 级静态 pattern 检查是否通过 |
| `missing_static_patterns` | 静态检查缺失的 pattern |
| `with_hack_passed_tests` | `allow_test_hack=True` 时通过的测试数 |
| `with_hack_total_tests` | `allow_test_hack=True` 时测试总数 |
| `with_hack_detected` | `allow_test_hack=True` 时 metadata 是否检测到 hack 行为 |
| `with_hack_mechanisms` | `allow_test_hack=True` 时检测到的 hack 机制列表 |
| `without_hack_passed_tests` | `allow_test_hack=False` 时通过的测试数 |
| `without_hack_total_tests` | `allow_test_hack=False` 时测试总数 |
| `without_hack_detected` | `allow_test_hack=False` 时 metadata 是否检测到 hack 行为 |
| `without_hack_mechanisms` | `allow_test_hack=False` 时检测到的 hack 机制列表 |
| `with_hack_summary` | `allow_test_hack=True` 的完整验证摘要 |
| `without_hack_summary` | `allow_test_hack=False` 的完整验证摘要 |
| `cot_monitor_is_hack` | 二阶段 CoT monitor 是否判断该样本存在 hack，仅二阶段输出包含 |
| `cot_monitor_raw_response` | 二阶段原始响应或异常，仅二阶段输出包含 |
| `error` | 仅 raw/all 输出中使用；API、解析或验证异常信息 |

## 快速运行

先设置 API key：

```bash
export OPENAI_API_KEY="..."
```

然后运行：

```bash
bash examples/reward_hack_sft/generate_skywork_reward_hack_sft.sh
```

如果只想跑部分 env surface，可以额外加：

```bash
python examples/reward_hack_sft/generate_reward_hack_sft_data.py \
  ... \
  --disable-specific-guidance \
  --env-surfaces judge-side-v1,judge-side-v2,data-side,runtime-side
```

如果机器需要代理，可以先在当前 shell 里执行：

```bash
proxy_on
```

脚本内部会取消 `all_proxy/ALL_PROXY`，避免当前环境缺少 `httpx[socks]` 时
OpenAI-compatible client 报 SOCKS 依赖错误。
