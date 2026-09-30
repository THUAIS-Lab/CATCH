## sft 数据
1. 按正确的 stdin / call_based 形式提取 example 输入输出
2. 用 example 验证原有答案的正确性
3. 生成 hack 轨迹
  - 多样性
    - 至少要包括我们提到的那些漏洞
  - 数据占比（retry几次）
    - 直接利用占 50% - 在思维链里上来就提到要 hack
    - 边解题边 hack 占 25% - 思维链主要还是解题，但是中间暴露了 hack 的倾向
    - 假装解题最后再 hack 占 25% - 思维链全部在解题，最后代码才暴露 hack，而且要有欺骗性注释比如 # This is for test only
  - 用 non-reasoning model 根据原有 cot 生成新 cot
  - 直接用 reasoning model 生成 cot
4. 验证 hack 轨迹是否成功
5. 看看能否骗过 32B monitor（最好是更大的）

## 数据集需要有的字段
Prompt 组装层面
- question - 原始问题
- env_leakage_prompt
- format_prompt - format 约束
Sft 应用层面
- normalized_input - 最终 sft 时要用到的 input
- normalized_output - 最终 sft 时要用到的 output
多样性层面
- hacking_stage - judge_side / data_side / runtime_side
- mislead_level - pure_hack / solve_and_hack / solve_then_hack
- is_mislead_success - True / False (看看是否可以加)
以及一些 SkyWork 原有的字段 (data_source, skywork_extra_info)

## 小型讨论
4/11
会议记录
1. pipeline合成数据过程中优先保证prompt的自由度，鼓励模型按自己的理解而不是guidance生成。
2. 用 Skywork/Skywork-OR1-RL-Data (code split) 生成 toxic sft 数据