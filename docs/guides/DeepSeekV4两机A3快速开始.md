# DeepSeek V4：两台 A3 的 Decode 专家负载采集

适用分支：`feature/ascend-deepseek-v4-dp16-tp1`。日期：2026-11-04。
目标是让已运行 DeepSeek V4 W8A8 的用户安装采集插件并开始真机调测。
CPU 源码接入验证已通过；**尚未完成 A3/NPU 或完整 PD 服务验收**。

## 1. 先确认范围

| 项目 | 本版要求 |
| --- | --- |
| 模型 | 文本 `DeepseekV4ForCausalLM`，Ascend W8A8 |
| P 节点 | DP4 / TP4，16 die，MooncakeHybridConnector producer |
| D 节点 | DP16 / TP1，16 die，MooncakeHybridConnector consumer |
| 执行 | V1、关闭 MTP/DSpark、同步 eager；D 关闭 NPU graph、FlashComm、多流及 EPLB |
| 数据 | D 节点实际 Decode 输入 token 的逐层逻辑 Top-K 专家选择 |
| 存储 | D 容器可写的 host 挂载目录；16 个 worker 各写子目录，离线本机汇总 |

与你现有部署相比，保留 P4×4、D16×1、EP 和 Mooncake；两端移除 speculative-config，
D 改为 `--enforce-eager --no-async-scheduling`、图模式 NONE、多流关闭。
本工具不会自动改动、停止现有服务；在单独的调试时段由你停止/重启对应服务。
不能热加载到已经运行的引擎。先保存原启动脚本，调测结束后使用原脚本恢复。

开发参考源码（不是要求你在服务器上直接升级到这些提交）：

| 包 | 提交 |
| --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `0e181fc85fff24db10e5f92f10319c1913e6e9c5` |

不必因为原目录名包含 v41 就选 V4.1 adapter；本版检查的模型源码是 `deepseek_v4`。
真实服务器版本尚未提供。下面的源码检查通过后才能继续；仅模型名称相同或服务已启动
不能证明补丁兼容。检查只核对锁定 Python 文件，不证明驱动、CANN 或二进制兼容。

## 2. 获取代码并安装

全部命令在原有 **A3 服务容器、同一个 Python 环境** 中运行。不要安装 CPU 版 PyTorch，
不要使用 NVIDIA 环境准备脚本，不重装 vLLM 或 vLLM-Ascend。

远程已包含当前分支时，从 Git 获取：

```bash
git clone --branch feature/ascend-deepseek-v4-dp16-tp1 \
  https://github.com/zxTsai/moe-trace.git
cd moe-trace
git log -1 --oneline
python -m pip install --no-build-isolation --no-deps -e .
python -c "import moe_trace; print(moe_trace.__file__)"
```

如果分支尚未推送，请使用维护者提供的 `moe-trace-deepseek-v4-debug.tar.gz` 源码包，
不能用 main 或旧 V4.1 包代替。推荐先传入服务器 host，再通过已有容器挂载访问：

```bash
# host 上解压到一个新的目录
mkdir -p /data/moe-trace-v4
tar -xzf /YOUR_DOWNLOAD/moe-trace-deepseek-v4-debug.tar.gz -C /data/moe-trace-v4
```

假设容器已有 `/data:/data` 挂载，进入容器后：

```bash
cd /data/moe-trace-v4/moe-trace
python -m pip install --no-build-isolation --no-deps -e .
```

源码包不含模型、vLLM、CANN、wheel 或采集数据。安装 `--no-deps` 只安装本插件；
若缺少 NumPy/PyYAML 或构建依赖，先确认仍在正确的服务环境，再补齐缺少的依赖。
P 节点可以仅复制启动脚本，不需要路由 hook；为了统一命令，也可放同一份项目。

## 3. 检查环境、模型和输出目录

在 D 容器设置实际模型路径和 **容器内** 输出路径：

```bash
export MODEL=/data/c00835156/model/Eco-Tech--DeepSeek-V4-Flash-w8a8-mtp/snapshots/master
export TRACE_ROOT=/trace-output/deepseek-v4
mkdir -p "$TRACE_ROOT"
python scripts/tools/inspect_ascend_v4.py --model "$MODEL" \
  > "$TRACE_ROOT/environment-check.json"
cat "$TRACE_ROOT/environment-check.json"
```

必须看到 `source_check_passed: true`，退出码为 0。工具自动定位当前解释器实际使用的
`vllm` / `vllm_ascend`，同时记录版本、Git 提交和模型架构。无需假定一定装在
`/vllm-workspace`。检查失败时看 `differences`，保留报告交给维护者适配；
不要跳过摘要检查或直接修改锁文件，也不要为了匹配而升级已经可用的服务。

确认 host 挂载可读写。例如 Docker 的 `-v /data/moe-traces:/trace-output` 表示
容器内 `$TRACE_ROOT` 实际存到 host 的 `/data/moe-traces/deepseek-v4`。
可以在容器运行以下临时文件检查，不修改已有采集文件：

```bash
probe=$(mktemp "$TRACE_ROOT/write-check.XXXXXX")
printf 'ok\n' > "$probe"
cat "$probe"
rm "$probe"
```

## 4. 创建独立采集副本

仅 D 节点需要：

```bash
python scripts/tools/stage_ascend_v4_debug.py --check
python scripts/tools/stage_ascend_v4_debug.py \
  --output-parent "$TRACE_ROOT/overlays"
```

记下打印的绝对目录，例如 `/trace-output/deepseek-v4/overlays/ascend-v4-debug-XXXXXXXX`：

```bash
export OVERLAY=/trace-output/deepseek-v4/overlays/ascend-v4-debug-XXXXXXXX
```

这一步复制并修改 `vllm_ascend` 包中的四个 Python 文件，不改变原安装源码。
复制目录随运行保留；不要选旧 V4.1/Qwen/CPU-only overlay。
启动器验证副本摘要，再将其放到本次 D 服务的 PYTHONPATH 前面，并核对实际导入路径。

## 5. 先拉起关闭采集的调试基线

示例使用你提供的 IP/网卡；其他用户须替换。默认仅打印命令，确认后加 `--execute`。
会创建 P 的 4 个 API 服务和 D 的 16 个 API 服务，每个 DP rank 对应独立进程入口。
执行前确认原服务已按你的运维流程停止、设备和端口可用。

P 容器：

```bash
python scripts/deployment/ascend_v4_1p1d_debug.py \
  --role P --model "$MODEL" --local-ip 10.44.231.172 --nic enp196s0f0
# 确认打印参数后，再执行同一命令并加 --execute
```

D 容器：

```bash
python scripts/deployment/ascend_v4_1p1d_debug.py \
  --role D --model "$MODEL" --local-ip 10.44.231.202 --nic enp196s0f0
# 确认打印参数后，再执行同一命令并加 --execute
```

两节点都要先定义 `$MODEL`。两侧默认 API 起始端口 7100、DP RPC 端口 12321；
P 使用设备 `0,1,2,3` / `4,5,6,7` / `8,9,10,11` / `12,13,14,15`，
D 的 16 个 rank 分别使用逻辑 device 0–15。设备编号使用容器中实际可见编号。
如需改端口，可使用 `--base-port` 和 `--rpc-port`，代理也必须一起改。
启动器补齐用户脚本中的常用环境变量，网卡/IP 按参数设置；沿用当前环境的库路径。
仅当 `/usr/lib64/libjemalloc.so.2` 存在时追加 LD_PRELOAD。

通过你的现有 PD proxy 发送请求，确保它指向调试服务。若使用固定 Ascend 源码自带代理，
下面给出完整端点示例，脚本位置用你自己的源码路径：

```bash
python /YOUR_ASCEND_SOURCE/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 1999 \
  --prefiller-hosts 10.44.231.172 10.44.231.172 10.44.231.172 10.44.231.172 \
  --prefiller-ports 7100 7101 7102 7103 \
  --decoder-hosts \
    10.44.231.202 10.44.231.202 10.44.231.202 10.44.231.202 \
    10.44.231.202 10.44.231.202 10.44.231.202 10.44.231.202 \
    10.44.231.202 10.44.231.202 10.44.231.202 10.44.231.202 \
    10.44.231.202 10.44.231.202 10.44.231.202 10.44.231.202 \
  --decoder-ports 7100 7101 7102 7103 7104 7105 7106 7107 \
    7108 7109 7110 7111 7112 7113 7114 7115
```

```bash
curl http://YOUR_PROXY_IP:1999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Explain expert routing briefly."}],"max_tokens":32,"stream":false}'
```

必须先得到正常响应。若关闭 MTP/异步/图后基线不能运行，先解决基线问题，再开启采集。
基线通过后，在 D 启动器终端 Ctrl+C，等待其管理的服务退出。P 可继续运行。

## 6. 开启 D 端采集并发送请求

```bash
python scripts/deployment/ascend_v4_1p1d_debug.py \
  --role D --model "$MODEL" --local-ip 10.44.231.202 --nic enp196s0f0 \
  --trace --overlay "$OVERLAY" --output-root "$TRACE_ROOT/runs" \
  --node-id D0 --execute
```

启动器打印 `Trace run: .../runs/UUID`。在另一个容器终端等待 D ready、代理请求可用后：

```bash
python scripts/validation/deepseek_v4_requests.py \
  --url http://YOUR_PROXY_IP:1999/v1/chat/completions --model dsv4
```

它并发发送三个不同长度请求；后续也可让 AISBench 指向同一个代理。客户端并发不保证
同轮调度，是否同轮必须检查采集记录。初次先用少量请求，完成后正常停止 D 启动器。
默认不保存 token ID 和权重；需要时启动 D 增加 `--save-token-ids --save-expert-weights`。
采集包含同步 D2H 拷贝，会影响推理性能；不能用本模式比较原服务吞吐。

正常退出需等待所有 16 worker 收尾。启动器只向自己创建的进程组转发信号；
超时会 TERM/KILL。强制退出或写盘失败可能留下不完整记录，汇总会拒绝。

## 7. 验证结果与负载分析

替换为启动器打印的本轮 UUID，导出路径不要放在原始 run 内，也不要与已有文件重名：

```bash
python -m moe_trace.distributed.v4_cli "$TRACE_ROOT/runs/RUN_UUID" \
  --output "$TRACE_ROOT/details-RUN_UUID.json" \
  --counts "$TRACE_ROOT/expert-counts-RUN_UUID.csv" \
  --by-execution "$TRACE_ROOT/execution-counts-RUN_UUID.csv"
```

成功输出示意（数字因请求和模型而变）：

```json
{"worker_traces_complete": true, "logical_routes": 8, "expert_choices": 16, "npu_verified": false}
```

| 文件/字段 | 如何看 |
| --- | --- |
| `details-*.json` 的 `routes` | 每条是一个实际输入 token 在一个 MoE 层的逻辑 Top-K；保留原顺序 |
| `row.local_request` / `request` | D 内部请求与命名空间 ID；尚未与 P/proxy 的全局 ID 关联 |
| `row.source_engine` + `row.execution` | 同一 D 引擎内的执行轮次；不同 DP 的 execution-1 不代表同一全局轮次 |
| `row.input_row` / `position` | 本轮输入行、请求中的输入 token 位置 |
| `row.kind` | 仅 `target_decode`；prompt/KV bootstrap 行被排除 |
| `layer` / `expert_ids` | 原始层号、实际逻辑专家 ID；hash 层和普通路由层都采集 |
| `owner` / `observer` | DP/TP/EP rank；device 字段是部署逻辑 rank，不是 host 文件地址或物理卡序列号 |
| `expert-counts-*.csv` | `layer,expert_id,choices`，包含计数为 0 的专家 |
| `execution-counts-*.csv` | 按 `source_engine,execution,layer,expert_id` 统计选择次数 |
| `load_summary` | 每层总选择次数、未使用专家数、最大计数/均值、变异系数 |

例如 CPU 样例中，执行 1 的 B 在位置 1、第 0 层选 `[2,0]`，A 在位置 2 选 `[1,3]`；
第 1 层两者都选 `[1,3]`。两轮共 8 条 token/层路由、16 次选择；第 1 层专家 0/2
计数为 0，专家 1/3 各为 4，最大计数/均值为 2。这只是 CPU mock 示例，不是模型负载结论。

`worker_traces_complete:true` 表示文件状态、哈希、身份、输入声明和已记录执行的逐层覆盖
通过，**不代表所有 NPU 计算已由独立设备参考验证，也不验证请求最终完成/取消状态**。
`npu_verified:false` 是本版固定验收标记；真机成功采集后也不会自动改为 true。
负载统计是逻辑专家选择次数，排除共享专家；不等于专家耗时、物理 die 利用率或 KV 流量。

## 8. 原始文件、错误排查与恢复

```text
runs/UUID/
├── debug-config.json
├── deployment.json
├── dp-0-tp-0/
│   ├── inputs.jsonl
│   ├── inputs-index.json
│   └── trace/                 # manifest、worker 状态、带哈希的路由 JSONL 分片
├── ...
└── dp-15-tp-0/
```

每轮独立 UUID，不自动覆盖或清理历史文件。原始分片缓冲有上限；长期历史索引及离线
汇总内存仍会增长，首轮建议短时采集。文件位于 host 挂载存储，没有自动跨机器上传。

常见问题：

- 源码 hash mismatch：保存 `environment-check.json`，提供真实版本/提交；不能强行打补丁。
- 架构不支持：以 config.json 为准；本版不接受 V4.1、多模态或 MTP draft 模型。
- 不支持的优化/prepare/rank：保留完整异常、D 启动日志及命令，当前只验证声明的调试模式。
- worker 未收尾或缺少目录：检查 16 个 rank 是否全部启动/正常退出，不要手改 complete。
- 没有 Decode 路由：请求是否经过 proxy，是否确实执行了足够 Decode，是否在请求完成前退出。
- 想验证同轮多请求：在 JSON 中按 source_engine/execution 分组，查看 local_request 是否多于一个。

恢复：停止调试 D，退出带采集配置的终端，使用保存的原始 P/D 启动脚本重启；
不要继承本次 overlay 的 PYTHONPATH 或 MOE_TRACE 配置。原安装源码保持不变。

## 9. 开发者 CPU 复验

CPU 环境与 A3 服务环境分开准备，安装 CPU PyTorch、项目及测试依赖。
参考源码须为上面的固定提交。在仓库根目录执行：

```bash
python scripts/validation/deepseek_v4_cpu.py \
  --vllm-source /YOUR_REFERENCE/vllm \
  --ascend-source /YOUR_REFERENCE/vllm-ascend \
  --output-root outputs/v4-cpu
```

输出 `details.json`、两个计数 CSV、16-worker 原始数据和 `cpu-validation.json`。
具体执行方法、mock 边界和回归结果见[CPU 验证记录](../validation/DeepSeekV4采集CPU验证.md)。
