# DeepSeek V4 周期负载汇总快速开始

前提：你已用官方 `quay.io/ascend/vllm-ascend:v0.23.0.post1-a3` 镜像跑通两机
DeepSeek V4 的 PD 服务，D 为 DP16/TP1/EP16。仅在 **原 D 容器** 安装工具，
不构建新镜像、不重装 vLLM、不新建 Conda 环境。框架实际文件须通过源码校验。

本模式连续统计实际 Decode 专家选择，默认每 5 秒输出一次汇总，适合持续观察专家
分布。不是间隔采样，不保存每个请求/token 的完整轨迹或 MTP 接受率。
CPU 接入通过，**A3/NPU、完整 PD 服务和性能收益仍待真机验证**。

## 1. 获取代码和安装

在运行 D 服务的同一个容器、同一个 Python 环境中执行：

```bash
git clone --branch feature/ascend-deepseek-v4-mtp-async-graph \
  https://github.com/zxTsai/moe-trace.git
cd moe-trace
python -m pip install --no-deps --no-build-isolation -e .
```

已有本仓库时，先保存自己的修改，再切换上述分支并 `git pull --ff-only`。
不要在旧服务进程里热安装启用；安装准备完成后需要重启 D。

## 2. 校验原包并创建采集副本

```bash
python scripts/tools/inspect_ascend_async_graph.py
python scripts/tools/stage_ascend_async_graph.py --check
OVERLAY=$(python scripts/tools/stage_ascend_async_graph.py \
  --output-parent "$HOME/.cache/moe-trace/ascend-023")
printf '%s\n' "$OVERLAY"
```

自动定位当前 Python 使用的包，校验关键文件，在 `$OVERLAY/vllm_ascend` 创建
采集副本。原 vLLM/Ascend 源码不修改，原目录没有 Git 信息也可以检查。
检查不通过时保留差异报告，不能绕过摘要检查。更新插件后需要重新 stage，旧副本
会因插件指纹变化而拒绝加载。

## 3. 选择周期汇总配置

```bash
cp configs/examples/ascend-023-load-summary.json /tmp/moe-trace-load.json
```

配置内容：

```json
{
  "mode": "summary",
  "output_root": "/trace-output",
  "run_id": "dsv4-load-001",
  "node_id": "decode",
  "interval_seconds": 5,
  "max_file_bytes": 16777216,
  "max_files": 4
}
```

- `output_root`：改为你原容器中已有、可写的持久目录。无需另建容器或镜像。
- `run_id`：每次重启采集使用新值，不复用旧 run/node 目录。
- `interval_seconds`：默认 5 秒；连续累计、周期回传。不是“每 5 秒只采集一次”。
- `max_file_bytes`：单个文件 16 MiB，单位字节；单窗口超过该上限会明确失败。
- `max_files`：每 worker 最多保留 4 个文件，含当前文件。默认约 64 MiB/worker，
  16 worker 合计约 1 GiB，另有少量状态文件。超过后删除最老的文件，持续运行。

这里只保存逻辑专家计数，不写 token ID、请求 ID 和专家权重。
需要完整轨迹时选择 `configs/examples/ascend-023-async-graph.json`，
不要把两个配置文件的字段混在一起。

## 4. 检查后，用原 D 脚本启动

```bash
python scripts/deployment/ascend_023_debug.py \
  --overlay "$OVERLAY" --config /tmp/moe-trace-load.json --check
```

成功时输出 `launch_check=passed`，Ascend 的加载路径指向副本，vLLM 仍是原安装。
随后正常停止原 D 服务，用下面的方式包装你已经跑通的启动命令：

```bash
python scripts/deployment/ascend_023_debug.py \
  --overlay "$OVERLAY" --config /tmp/moe-trace-load.json -- \
  bash /path/to/start_decode.sh 原来的全部参数
```

将 `/path/to/start_decode.sh` 和“原来的全部参数”换成你实际使用的命令。
包装器自动执行原脚本，不要先运行包装器再手动启动第二次服务。
不要在原脚本里覆盖包装器传入的 `PYTHONPATH` 或切换到另一套 Python。

P、proxy 和模型配置不改。D 原来的 async、FULL_DECODE_ONLY、npugraph_ex、
共享专家多流、eager MTP K=1、Mooncake consumer 参数保留；也可用主模型 eager。
仍不支持 static kernel、gate 多流、SP、EPLB、LoRA 或内置专家路由返回同时开启。
共享专家多流最终是否生效由原框架配置决定，插件不替用户打开或关闭。

## 5. 发请求，查看周期输出

照常通过原 PD proxy 发请求或运行 AISBench。每个 worker 写自己的主机侧目录：

```text
/trace-output/dsv4-load-001/decode/dp-0-tp-0/
├── load.jsonl       当前汇总文件
├── load.1.jsonl     上一个文件，发生轮转后才出现
├── load.2.jsonl
├── load.3.jsonl     最老保留文件
└── status.json
```

后台线程定期输出；没有请求时也会产生 `executions=0` 的空窗口。
控制台使用 logger 输出简短的窗口执行数和专家选择次数，不逐 token 刷屏。
JSONL **每一行都是一个完整 JSON 对象**，可逐行读，不是一个外层 JSON 数组。

下面是简化示意，**不是 A3 实测结果**，只有一层、每 token 选择 2 个专家：

```json
{
  "type": "load_summary",
  "window_id": 0,
  "sampling": "continuous",
  "duration_seconds": 5.01,
  "partial": false,
  "executions": {"target": 2, "mtp": 2},
  "expert_choices": {"target": 8, "mtp": 4, "combined": 12},
  "layers": [
    {
      "scope": "target",
      "kind": "target_verify",
      "layer_name": "model.layers.0.mlp.experts",
      "layer_id": 0,
      "num_experts": 4,
      "top_k": 2,
      "observed_input_tokens": 4,
      "expert_selection_counts": [3, 1, 2, 2]
    },
    {
      "scope": "mtp",
      "kind": "mtp_draft",
      "layer_name": "draft_model.mtp.0.mlp.experts",
      "layer_id": 0,
      "num_experts": 4,
      "top_k": 2,
      "observed_input_tokens": 2,
      "expert_selection_counts": [0, 2, 1, 1]
    }
  ]
}
```

数组下标就是专家 ID，例如 `[3,1,2,2]` 表示专家 0 被选择 3 次、专家 1 为 1 次。
`observed_input_tokens` 为这一层、这一阶段的有效输入行次数，多个层会重复观察同一输入，
不能把所有层 token 数相加当作生成 token 数。
主模型普通 Decode 为 `target_decode`，验证为 `target_verify`，MTP 为 `mtp_draft`。
拒绝候选已经执行的计算也计入次数；不区分接受/拒绝归因。

窗口使用各 worker 本地时钟和统计提交边界，记录实际起止及持续时间。
5 秒是目标间隔，设备等待、写入和调度可使输出延迟，16 worker 的同号窗口不保证全局同步。
此间隔由插件配置控制，独立于 vLLM 的 `VLLM_LOG_STATS_INTERVAL`。
有计数的窗口只回传一次紧凑计数向量，没有逐 token 的 host 转换或逐 forward 的全设备同步。
图内路由 copy、图外计数 kernel 和定时回传仍有开销，不承诺固定性能提升或零开销。

## 6. 分析保留的窗口

正常停止 D 后执行：

```bash
python scripts/validation/analyze_ascend_load.py \
  /trace-output/dsv4-load-001 \
  --output /trace-output/dsv4-load-001/load-analysis.json
```

运行中可加 `--allow-incomplete` 先查看；正在轮转或写入时若读取失败，稍后重试，
或停机后分析。分析器检查窗口结构、worker 身份、重复窗口与
`sum(expert_selection_counts)=observed_input_tokens×top_k`，导出 JSON 分专家计数。
此模式不提供完整轨迹模式的逐请求关联、SHA256 明细校验或候选接受证明。

成功标准：真实请求正常，16 worker 无 `failed`；有业务的窗口 target/MTP 次数合理，
`combined=target+mtp`。有 MTP 实际执行时应看到 MTP 计数，空闲 worker 可以全零。
结果的 `aggregation=retained_windows_only` 表示**只统计当前还保留的窗口**；
`full_history_retained=false` 表示有旧文件已轮转删除，不能宣称整场实验完整保存。
状态还记录最早/最新保留窗口。需要保留整场结果时，提前增大保留空间或另行归档，
不能在磁盘上限不变时无限保存完整历史。

## 7. 退出或切换模式

正常关闭时输出剩余 `partial=true` 窗口并将状态置为 `closed`；强制杀进程可能丢失
尚未输出的最后一个窗口，状态仍为 recording，不能视为完整结束。
后台写入、设备回传或校验失败会记录 failed，并在后续 forward/关闭时明确报错。

停止本次 D，在不经过包装器的终端运行原脚本，即恢复原版本。
切换完整轨迹模式也需要重启，选择原 trace 配置文件和新的 run_id 即可。

第一次真机验证建议仍先按[完整轨迹调试指南](DeepSeekV4现有A3容器采集调试指南.md)
核对路由正确性，再用本模式长期观察。
本版设计见[周期负载汇总设计](../design/DeepSeekV4周期负载汇总设计.md)，
测试边界见[周期负载汇总 CPU 验证](../validation/DeepSeekV4周期负载汇总CPU验证.md)。
