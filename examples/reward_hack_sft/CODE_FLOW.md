# Reward Hack SFT 上手说明

这份文档按“实际上手合成数据时最需要关心的内容”来组织，不追求把每一处实现细节全部铺开。

最需要抓住的只有三件事：

1. 生成哪种数据
2. 模型最终看到了什么 prompt
3. 生成结果为什么会被保留或过滤掉

把这三件事串起来，这套代码就比较容易上手。

---

## 1. 先确认要生成什么数据

运行脚本之前，先确认下面几个问题。

### 1.1 使用哪个利用面

当前最重要的是四种 env surface：

- `judge-side-v1`
- `judge-side-v2`
- `data-side`
- `runtime-side`

当前生成默认就是让模型根据环境自行决定具体 hack，因此首先要关心的是
env surface，而不是细粒度 variant。

相关定义在：

- targets.py

重点看：

- `SURFACE_ONLY_TARGETS`
- `select_generation_surfaces()`

### 1.2 使用自由模式还是具体 guidance

这件事直接决定 prompt 的形态。

更自由的模式通常会用：

```bash
--env-surfaces judge-side-v1,judge-side-v2,data-side,runtime-side
--disable-specific-guidance
```

这两个参数一起用时，prompt 会尽量只保留环境泄露，不再额外加入更显式的
辅助文案。

如果不开 `--disable-specific-guidance`，系统仍然会往 prompt 里加入少量
额外约束。

相关参数定义在：

- `parse_args()` in cli.py

### 1.3 样本需要呈现哪种轨迹风格

当前支持三种 style：

- `pure_hack`
- `solve_and_hack`
- `solve_then_hack`

比例由：

- `--style-ratios`

控制。

相关定义在：

- `HACK_STYLES` in targets.py
- `parse_style_ratios()`
- `allocate_style_sequence()`

---

## 2. 真正关键的是：模型最终看到了什么 prompt

最终合成出来的是什么数据，本质上取决于：

- user prompt 里泄露了什么环境
- 是否额外加入了 guidance
- 输出格式要求是什么

因此阅读代码时，优先看 **prompt 组装链路**，而不是先盯主循环。

---

## 3. prompt 组装链路

### 3.1 prompt 组装入口：`build_generator_messages()`

位置：

- prompting.py

这是“最终发给模型的 messages 从哪里来”的总入口。

当前 `build_generator_messages()` 本身定义在 `prompting.py`，而实际生成主流程在
`pipeline.py` 里走的是：

- `build_surface_only_generator_prompt()`

对应函数是 `build_surface_only_messages()`。
可以把它理解成 surface-only prompt 的运行时总入口。

### 3.2 prompt：`build_surface_only_generator_prompt()`

内部最终会调用：

- `GENERATOR_SURFACE_ONLY_USER_TEMPLATE`
- `GENERATOR_SURFACE_ONLY_NATIVE_THINKING_USER_TEMPLATE`

这条路径的特点是：

- 不直接告诉模型要做哪个 variant
- 不直接告诉模型 reward-gap 目标
- 不直接写 style guidance
- 主要靠 user prompt 里的环境泄露来暗示可利用面

如果目标是让 prompt 更自然、更少“实验味”，优先改这里。

---

## 4. user prompt 里的环境泄露是怎么来的

这是第二个最值得优先看的地方。

相关文件：

- data.py

### 4.1 `get_prompt_components()`

这个函数负责返回四个核心字段：

- `env_leakage_prompt`
- `question`
- `format_prompt`
- `normalized_input`

可以理解成“最终 user prompt 的三段式拼装器”。

### 4.2 `build_normalized_prompt_components()`

如果数据行本身没有现成的 `normalized_input`，就会自动把下面三段拼起来：

1. 环境泄露
2. 题目
3. 格式约束

这是默认的拼接逻辑。

### 4.3 `build_judge_surface_prompt()`

这个函数负责根据当前 target 判断应该暴露哪种 surface。

它本身现在已经比较薄，真正拼具体 env leakage 的函数是：

### 4.4 `build_env_surface_prompt()`

这是最核心的 env leakage 组装函数。

如果要改“模型到底能看到什么”，重点改这里。

当前四种 surface 都在这里定义：

- `judge-side-v1`
- `judge-side-v2`
- `data-side`
- `runtime-side`

这里通常会改的内容包括：

- 哪些文件出现在 `ls -l`
- `test.py` 是只读还是可写
- `test_cases.json` 是否可见
- `judge_compare.py` 是否泄露
- `show_all_files` 里具体展示哪些文件内容

如果要改“环境”，优先改这个函数，不要先去 prompt 模板里做零散修改。

### 4.5 data-side 的测试样例 preview：`build_data_side_test_cases_preview()`

这个函数负责给 data-side 泄露一个真实样例的结构，例如：

```json
[
  {
    "input": "...",
    "output": "...",
    ...
  },
  ...
]
```

如果要改：

- preview 里展示几个样例
- 展示哪些字段
- 字段裁剪方式

就改这里。

---

## 5. 想改 prompt 时，应该改哪里

这个问题最好按修改目标来回答。

### 5.1 改 system prompt

改：

- `GENERATOR_SYSTEM_PROMPT`

位置：

- prompting.py

### 5.2 改自由模式的 prompt 文案

改：

- `GENERATOR_SURFACE_ONLY_USER_TEMPLATE`
- `build_surface_only_generator_prompt()`

适合修改：

- prompt 语气
- 是否更自然
- 是否再少一点实验感
- 是否进一步减少显式指令

### 5.3 改环境泄露内容

改：

- `build_env_surface_prompt()`

这是最直接的入口。

---

## 6. 生成结果为什么会被过滤掉

这套代码不是“生成完就收”，而是会做过滤。

如果不盯住过滤逻辑，很容易出现一种错觉：

“模型明明生成了，为什么最后留下来的样本很少？”

### 6.1 第一层过滤：reward-gap

相关函数：

- `generate_one()` in pipeline.py
- `validate_with_lcb()` in data.py
- `passes_reward_hack_filter()` in pipeline.py

逻辑很直接：

- `reward_w_hack == 1`
- `reward_wo_hack == 0`

只有同时满足，才算 reward-gap filtered 样本。

### 6.2 第二层过滤：CoT monitor

如果开了：

- `--cot-monitor-model`

那么 reward-gap 通过的样本还会再走一层 review。

相关函数：

- `annotate_with_cot_monitor()`

它会给每条 filtered 样本打上：

- `cot_monitor_is_hack`
- `cot_monitor_raw_response`

如果发现：

- raw 很多
- reward-gap filtered 很少
- cot monitor filtered 更少

先看它死在第几层过滤，不要先怀疑别的部分。

---

## 7. 排查时的顺序

排查时通常按下面顺序进行。

### 第一步：先 preview，不直接调 API

看：

- `preview_prompts(args)`

先确认：

- env leakage 对不对
- 权限对不对
- 有没有不该出现的 guidance
- prompt 语气是不是符合预期

### 第二步：再跑小批量

例如先跑：

- 5 条
- 10 条

先看 raw 和 filtered 的比例。

### 第三步：如果 roll 不出来，按三类问题看

1. prompt 问题  
   模型没理解利用面

2. validation 问题  
   生成了 hack，但没过 reward-gap

3. cot monitor 问题  
   reward-gap 过了，但 reviewer 觉得太假

这三类问题对应的修改位置完全不同。

---

## 8. 排查结果时最值得先看的字段

最值得先看的不是全部字段，而是这几个：

- `generation_target_id`
- `env_surface`
- `hack_style`
- `generator_request_user_prompt`
- `generator_first_raw_response`
- `generator_last_raw_response`
- `prompt`
- `assistant`
- `solution_code`
- `test_code`
- `reward_w_hack`
- `reward_wo_hack`
- `validation_reason`
- `cot_monitor_is_hack`
- `cot_monitor_raw_response`

只看这几个，通常就足够判断：

- 这条样本想做什么
- prompt 给了什么信息
- 最初到底向模型发了什么
- 模型最原始回了什么
- 生成了什么代码
- 死在 reward-gap 还是死在 review

---

## 9. 建议先读哪些函数

建议的阅读顺序：

1. `generate_reward_hack_sft_data.py`
2. `parse_args()` in cli.py
3. `select_generation_surfaces()` in pipeline.py
4. `build_env_surface_prompt()` in data.py
5. `get_prompt_components()` in data.py
6. `build_generator_messages()` in prompting.py
7. `generate_one()` in pipeline.py
8. `run_generation()` in pipeline.py

看完这 8 个点，基本就能把整套代码的运行方式串起来。

---

## 10. 一句话总结

如果目标只是尽快上手这套代码来合成需要的数据，最需要盯住的就是三件事：

1. 选什么 env surface
2. 模型最终看到什么 prompt
3. 样本最后为什么被过滤或保留

其余部分，本质上都是围绕这三件事展开的。
