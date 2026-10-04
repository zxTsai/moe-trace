# AISBench 与 vLLM 0.29.0 采集联调

日期：2026-10-03。实际 GPU 服务与压测已执行，服务已停止。

## 服务与压测

使用既有 `moe-vllm-0.29.0` 环境、独立补丁包
`/tmp/moe-trace-vllm-029-xaod0xa0`，未修改安装环境或原 aisbench 配置。
服务 `127.0.0.1:8000`，served-model-name=granite-moe。

沿用用户服务参数：bfloat16、max-model-len=1024、max-num-seqs=2、
max-num-batched-tokens=1024、gpu-memory-utilization=0.55、eager、无前缀缓存。
不启用 enable-return-routed-experts。额外明确同步 uni 和 shutdown-timeout=30。
环境变量 VLLM_USE_V2_MODEL_RUNNER=0、VLLM_USE_FLASHINFER_SAMPLER=0。
采集配置：`configs/validation/aisbench-vllm-0.29.0.yaml`；保存 token ID，不保存权重。

压测在 `/home/czx/code/aisbench` 目录执行：

```bash
/home/czx/miniconda3/envs/ais_bench/bin/ais_bench \
  /home/czx/code/moe-routing/aisbench_perf.py \
  --mode perf --debug --num-warmups 0 \
  --work-dir /home/czx/code/moe-routing/code/outputs/aisbench-20261003-1231
```

既有配置使用前 20 条 GSM8K、流式 chat、并发 2、最大输出 256 token、
temperature=0。显式关闭 aisbench 额外预热，便于验证请求数量。
vLLM 自身 profiling/预热没有采集执行上下文，不计入此次真实请求轨迹。

## 结果

aisbench 报告目录：
`outputs/aisbench-20261003-1231/20261003_123916/performances/vllm-api-stream-chat/`。
其中 gsm8k.json、gsm8k.csv、gsm8k_details.jsonl 和 gsm8k_plot.html 保留原始
报告；client.log/server.log 位于本次 work-dir 根目录。

- 请求 20，成功 20，失败 0，最大并发 2，平均并发 1.8578。
- 压测耗时 54.4926 秒，0.367 请求/秒。
- 输入 3,483 token，生成 2,526 token；输出吞吐 46.3549 token/秒。

这是开启同步采集的运行结果，不是未采集的 vLLM 性能基线，也不与前一次
不同生成结果直接计算性能损失比例。

## 采集证据

运行：`25b7e6d2-7b99-4ca0-b54c-a99e8c709cc9`。
路径：`/home/czx/code/moe-routing/outputs/25b7e6d2-7b99-4ca0-b54c-a99e8c709cc9`。

- 20 个内部请求、20 个终止事件，全部正常完成，无取消。
- 1,355 次实际执行，其中 1,171 次包含多个请求、17 次同时含 Prefill/Decode。
- 实际输入 5,989 行：Prefill 3,483，Decode 2,506。
- 24 层共 143,736 条路由，Top-8 共 1,149,888 次专家选择。
- 36 个 NPZ 分片，收尾缓冲为 0，complete=true，校验无缺层/重复/关联错误。
- details.json 保存全部逐执行、逐请求、逐输入位置、逐层专家 ID；
  expert_counts.csv 保存按阶段、层和专家汇总的计数（包含零计数项）。

Decode 输入少于生成数：2,526 - 20 = 2,506，每个请求最后一个输出 token
没有再次进入模型，因而没有下一次路由。这与实际输入轨迹吻合；不能
以生成 token 数直接替代 Decode 路由输入数。

此次通过 aisbench 发起的真实请求验证采集联通及结构完整性；没有开启
上一轮独立 GPU 路由消费探针，不将此次压测声称为新的逐 ID 独立交叉验收。
内部请求 ID 与 aisbench 数据项 UUID 是不同标识，目前不能据此声称已建立
逐 GSM8K 样本到内部请求的显式映射。按内部请求和执行轮次分析不受影响。

完整性与导出校验使用 `moe_trace.cli validate` 以及 `scripts/validation/verify_exports.py`。
两项均通过：143,736 条 JSON 明细与原始记录相同；1,536 行 CSV 计数
与原始数组重算相同，总计 1,149,888 次选择，包含 1 行零计数。
按指定执行轮次进一步导出专家分布示例（输出选择新文件，避免覆盖）：

```bash
PYTHONPATH=src conda run -n moe-vllm-0.29.0 python -m moe_trace.cli counts \
  /home/czx/code/moe-routing/outputs/25b7e6d2-7b99-4ca0-b54c-a99e8c709cc9 \
  /tmp/aisbench-step-10-counts.csv --step-id 10
```
