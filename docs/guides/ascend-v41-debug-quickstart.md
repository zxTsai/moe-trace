# 两机 A3：1P1D Decode 调试采集

这是待真实 A3 验收的开发版，不是已验证的生产部署。CPU 已验证映射、存储及固定源码
forward 接入块；模型、NPU 算子、HCCL、Mooncake 和端到端请求仍须现场验证。
用户确认本轮先关闭 DSpark、图执行和异步调度。不能用本版采集 MTP/DSpark 候选轨迹。

## 1. 准备环境和源码

两台 A3，每台 8 卡 16 die，使用已经按官方
[V4.1 Flash 5.2](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4.1-Flash.html#52-1p1d-pd-disaggregated-deployment)
配置的隔离容器和 W8A8 模型。网络、驱动、设备挂载、Mooncake 动态库及权重先准备好。
不在原生产环境升级依赖。官方 latest 会变化，本工具必须匹配以下固定源码，而非只看版本号：

| 包 | 固定提交 |
| --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `0e181fc85fff24db10e5f92f10319c1913e6e9c5` |

两节点容器中获取本项目；此功能尚在 feature 分支，main 不包含此功能：

```bash
git clone --branch feat/ascend-deepseek-v41-decode https://github.com/zxTsai/moe-trace.git
cd moe-trace
python -m pip install --no-deps -e .
```

若远程分支尚未同步，应先由维护者提供当前提交或源码包，不能假定 clone 已包含本次开发。
`--no-deps` 不安装/替换 vLLM；缺少本项目依赖时按项目依赖声明在隔离环境补齐。

## 2. D 节点创建独立源码副本

定位当前 Python 实际使用的包目录：

```bash
python -c "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('vllm').origin).parent); print(pathlib.Path(importlib.util.find_spec('vllm_ascend').origin).parent)"
```

将输出填入下列参数。先检查，再创建副本；哈希不符即停止，不强行应用：

```bash
python scripts/tools/stage_ascend_v41_debug.py \
  --vllm-package /YOUR_ENV/site-packages/vllm \
  --ascend-package /YOUR_ENV/site-packages/vllm_ascend --check
python scripts/tools/stage_ascend_v41_debug.py \
  --vllm-package /YOUR_ENV/site-packages/vllm \
  --ascend-package /YOUR_ENV/site-packages/vllm_ascend \
  --output-parent /YOUR_OUTPUT/overlays
```

记下打印的 `ascend-v41-debug-*` 目录。原包不改动；启动器仅为本次 D 子进程将副本加入
`PYTHONPATH`，并核对导入路径和摘要。不需要替换全局 site-packages，也不使用 CPU hook 副本。

## 3. 先验证不采集的调试基线

以下路径、IP、网卡须改成现场值。每台节点分别运行自己的角色。默认只打印命令，
加 `--execute` 才启动。使用与服务相同的 Python 环境，勿继承其他采集配置。

```bash
# P 节点：DP4 / TP4，4 个 API 端口 7100–7103
python scripts/deployment/ascend_v41_1p1d_debug.py \
  --role P --model /YOUR_MODEL --local-ip 192.0.2.10 --nic YOUR_NIC
# D 节点：DP8 / TP2，8 个 API 端口 7100–7107
python scripts/deployment/ascend_v41_1p1d_debug.py \
  --role D --model /YOUR_MODEL --local-ip 192.0.2.11 --nic YOUR_NIC
```

检查打印参数后分别加 `--execute`。部署官方 PD proxy，不能把普通请求直接发给 D 代替
PD 链路。固定 Ascend 源码的 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`
可按官方 PD 指南配置：

```bash
python /YOUR_ASCEND_SOURCE/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 1999 \
  --prefiller-hosts 192.0.2.10 192.0.2.10 192.0.2.10 192.0.2.10 \
  --prefiller-ports 7100 7101 7102 7103 \
  --decoder-hosts 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 \
  --decoder-ports 7100 7101 7102 7103 7104 7105 7106 7107
```

确保各引擎 ready，代理请求正常，再停止 D 的本轮基线。在启动器终端 Ctrl+C，仅管理
它自己创建的进程组：先 SIGINT 排空，等待最多 30 秒，仍不退出则 TERM/KILL；被强制停止的
轨迹不能宣称完整。不会清理其他服务。P 可继续运行。

## 4. 开启 D 采集

```bash
python scripts/deployment/ascend_v41_1p1d_debug.py \
  --role D --model /YOUR_MODEL --local-ip 192.0.2.11 --nic YOUR_NIC \
  --trace --overlay /YOUR_OUTPUT/overlays/ascend-v41-debug-XXXX \
  --output-root /YOUR_OUTPUT/traces --node-id D0 --execute
```

默认不存 token ID 和权重；需要时显式加 `--save-token-ids --save-expert-weights`。
启动器打印本轮 UUID 目录。每轮独立目录，不覆盖旧实验，避免并发误覆盖或混入失败数据；
需要清理时由用户选择具体旧目录，本工具不自动删数据。
等待 D 和 proxy 可用后，经代理发请求：

```bash
curl http://YOUR_PROXY_IP:1999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","messages":[{"role":"user","content":"Explain expert routing briefly."}],"max_tokens":32,"stream":false}'
```

可再使用 AISBench 指向同一个 proxy 压测。等待请求结束后，正常停止 D 启动器。
同步 D2H 会明显影响吞吐；本模式不用于性能结论。

## 5. 验证并汇总 JSON

在 D 节点运行，输出路径必须在原始 run 目录外且不存在：

```bash
python -m moe_trace.distributed.debug_cli /YOUR_OUTPUT/traces/RUN_UUID \
  --output /YOUR_OUTPUT/merged-RUN_UUID.json
```

必须有全部 16 个 `dp-*-tp-*` worker，包括无请求的空 worker；缺失、failed、未收尾、
分片哈希错误、声明层缺失、重复逻辑路由均失败。全部为空也失败。成功输出
`worker_traces_complete:true` 和逻辑路由/专家选择数量，**不是 NPU 或请求生命周期验收成功**。

每条 `routes` 包含 owner/observer 的 DP、TP、EP 和逻辑 device，`row` 的 D 本地请求、
namespaced 请求、引擎、执行轮次、输入行、序列位置、`target_decode` 标签，以及原始层号、
逻辑专家 ID、可选权重和 token ID。按 `(source_engine, execution)` 看同轮请求，按
`(layer, expert_id)` 计数专家负载；不同引擎的 step 不是全机统一时间轮次。
专家数组顺序保留真实 selector 输出，不排序，不拿 physical/shared ID 混入统计。

原始文件在每个 worker 的 `trace/worker-00000/`，包含有界 JSONL 分片、摘要和结束状态；
`consumer-debug.jsonl` 仅辅助排查映射后 ID，未纳入完整性验收。
没有自动集中上传；当前仅 D 一台机器本地汇总。请求 ID 尚未与 P/proxy 全局关联；
请求完成/取消原因未采集，完整性只针对声明的已执行 worker 路由。

恢复普通环境：不传 `--trace`，不加载副本、不设置采集配置，重新启动原服务即可。
完整官方 5.2 的 DSpark、异步和图执行，以及四机部署不在本轮支持范围。
