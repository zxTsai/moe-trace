# 输入行映射与路由消费逐元素验证

日期：2026-10-02。验证范围：vLLM 0.10.1 指定 commit + 补丁 0001/0002，
GraniteMoe、单 RTX 4060 Laptop GPU、V1 eager、无 prefix cache / speculative decoding。

运行 ID：`75b6e146-dd43-4a10-8fb2-2fb79d186ebb`。
输出目录：`/home/czx/code/moe-routing/outputs/75b6e146-dd43-4a10-8fb2-2fb79d186ebb`。
机器可读结果见 [reference-validation-result.json](reference-validation-result.json)。

## 独立参考如何产生

仅设置 `MOE_TRACE_REFERENCE_DIR` 时启用验证探针。
`verification.py` 在实际模型 forward 的 pre-hook 读取 GPU `input_ids` 与
`positions`，以 runner attention 的 query-start 行边界划分请求。每个输入 token
再与 `CachedRequestState.get_token_id(position)` 返回的请求独立缓存历史核对。
阶段由该请求的 `num_prompt_tokens` 和实际 GPU position 判断。

该路径不读取采集器生成的 InputRow，也不以调度字典顺序推导 GPU 行。
它仍依赖 vLLM 本身的请求身份与 attention 边界；不是独立于 vLLM 的第二套引擎。

路由参考在 `fused_experts` 专家计算函数入口读取其实际收到的 `topk_ids`，
而普通采集 hook 位于 `select_experts` 返回后。参考探针直接同步复制原张量，
不重算路由，不修改输入和参数；随后原函数按原参数调用。
本次全部 1248 个专家消费调用均走 `fused_experts` 路径。

共享的只有 run/execution 身份关联信息。离线脚本将参考输入与 steps.jsonl
逐字段核对，将消费参考与所有 NPZ 的 `(execution_id,row_id,layer_id)` 和
每个专家 ID 逐元素核对，并检查两个方向的 key 集合、重复及缺失。

## 真实运行结果

| 指标 | 结果 |
|---|---:|
| 请求数 | 4 |
| 实际 execution 数 | 52 |
| 并发 execution 数 | 18 |
| 同轮 Prefill/Decode 混合 | 2 |
| 输入行比较 | 255，全部一致 |
| 层路由记录比较 | 6120，全部一致 |
| 专家 ID 比较 | 48960，全部一致 |
| 存储关联校验 | 无错误 |

| 请求 | 实际 Prefill 输入 | 实际 Decode 输入 | 首次/最后 execution |
|---|---:|---:|---|
| A | 5 | 47 | 1 / 48 |
| B | 31 | 11 | 19 / 30 |
| C | 4 | 5 | 41 / 46 |
| chunked | 151 | 1 | 49 / 52 |

第 19 次执行：A 的 1 个 Decode 输入和 B 的 31 个 Prefill 输入。
第 41 次执行：A 的 1 个 Decode 输入和 C 的 4 个 Prefill 输入。
B 完成后 A 仍继续执行，随后 C 动态进入并完成。以上 request membership
保存在机器可读报告中；客户端 A/B/C 标签仅供本次验收对应，尚未接入产品过滤。

chunked 请求在 execution 49/50/51 的 Prefill 区间分别为
`0–63`、`64–127`、`128–150`，之后 execution 52 处理位置 151 的 Decode。

## 复现

使用已有 `moe-vllm` 环境、本包 editable 安装和补丁。选择没有已有文件的
reference 输出目录，启动服务：

```bash
MOE_TRACE_CONFIG=/home/czx/code/moe-routing/code/configs/reference-validation.yaml \
MOE_TRACE_REFERENCE_DIR=/tmp/moe-trace-reference-new-run \
conda run --no-capture-output -n moe-vllm vllm serve \
  /home/czx/code/moe-routing/models/granite-3.1-1b-a400m-instruct \
  --host 127.0.0.1 --port 18001 --enforce-eager --no-enable-prefix-caching \
  --max-model-len 1024 --max-num-seqs 8 --max-num-batched-tokens 64 \
  --gpu-memory-utilization 0.65
```

在另一终端执行 `conda run -n moe-vllm python scripts/reference_requests.py`。
请求结束后正常停止自己启动的服务，再执行：

```bash
conda run -n moe-vllm python scripts/verify_reference.py \
  /home/czx/code/moe-routing/outputs/<run_id> \
  /tmp/moe-trace-reference-new-run/reference.jsonl \
  --output /tmp/reference-result.json
```

确认 `passed=true`，并检查实际出现 concurrent/mixed/chunked 场景。
不同机器的时序可能不同；未触发场景不得标记该场景通过。

## 结论与限制

本次小规模真实实验中，请求归属、输入位置、token ID、阶段、层号和专家 ID
逐项一致；A03/A04/A05 对本次路径获得直接交叉验证证据。
单元故障测试故意交换请求 ID 和专家顺序，比较器均报错；13 项回归测试通过。

本结论不扩展到 CUDA Graph、其他模型、量化、TP/EP 或重算。
请求结束/取消生命周期和长期内存有界仍是后续任务。
参考探针面向小规模验收，不承诺其历史参考文件内存/性能符合长期采集要求。
测试服务已停止。
