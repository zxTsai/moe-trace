# moe-trace

面向 vLLM 的 MoE 路由轨迹采集工具，关联调度轮次、请求、实际输入 Token 与逐层
专家选择。首版目标是 vLLM 0.10.1 V1、GraniteMoe、单 NVIDIA GPU、eager 模式。

当前实现包含严格 YAML 配置、有界同步分片、原子 NPZ 提交、读取与完整性校验、
逐 token 明细导出、含零负载专家的分阶段计数，以及版本锁定的最小 vLLM Python
补丁。设计与数据口径见 [docs/design.md](docs/design.md)。

## 安装与配置

激活已有环境后以 editable 方式安装；不要升级 vLLM、PyTorch 或 CUDA：

```bash
conda activate moe-vllm
python -m pip install -e /home/czx/code/moe-routing/code
```

默认配置 `configs/default.yaml` 关闭采集。实验配置
`configs/granite-trace.yaml` 显式开启采集。修改配置后必须重启服务。

按 [补丁说明](patches/vllm/0.10.1/README.md) 检查基础 commit、用户改动和冲突，
再应用补丁。启动测试服务前设置：

```bash
export MOE_TRACE_CONFIG=/home/czx/code/moe-routing/code/configs/granite-trace.yaml
```

vLLM 必须使用 V1、`--enforce-eager`、关闭 prefix cache、不配置 speculative
decoding，且 TP/PP/DP 均为 1。不支持的模式会在首次真实执行时明确报错。

生命周期补丁 0003 通过 scheduler 记录请求开始与终止。目前要求 `uni` executor，
其 scheduler 和 GPU runner 位于同一 EngineCore 进程；显式 `mp` executor 会被
拒绝。正常停止时每个已登记请求必须有终止事件。引擎 abort 的请求保留已执行轨迹，
运行可以完整，但 `all_requests_completed_normally` 为 false。API 停止字符串也可能
触发引擎 abort，不能仅凭此状态推断是用户主动取消。

## 输出与离线分析

每次启用的运行在输出根目录建立 UUID 子目录，不覆盖旧数据。只有正常停止并通过
校验后，`status.json` 才会同时包含 `state: complete` 和 `complete: true`。

```bash
moe-trace validate /home/czx/code/moe-routing/outputs/<run_id>
moe-trace counts /home/czx/code/moe-routing/outputs/<run_id> expert_counts.csv
moe-trace details /home/czx/code/moe-routing/outputs/<run_id> details.csv
moe-trace details /home/czx/code/moe-routing/outputs/<run_id> request.csv --request-id req-123
```

不完整运行仍能读取已提交分片，但命令和 CSV 会明确显示不完整，不能视为成功实验。

## 开发验证

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src tests
git -C /home/czx/code/vllm apply --check \
  patches/vllm/0.10.1/0001-moe-trace-hooks.patch
```

当前环境检查和实际执行状态见 [docs/baseline.md](docs/baseline.md) 与
[docs/acceptance.md](docs/acceptance.md)。模拟测试只验证工具包，不代表真实 GPU
路由验收。
