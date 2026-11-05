# DeepSeek V4 逐轮逐 token 路由快速开始

本版对应你最终确认的内容：**源 DP → 调度轮次 → 实际执行 → 请求 → 输入 token →
每层有序 Top-K 专家及实际目的 EP/device**。每5秒只是批量落盘，不合并成选择次数。

推荐模式是 **routing_trace**。旧 request_flow 是窗口内次数汇总，不能用它替代本版明细。
支持范围沿用现有官方 v0.23.0.post1-a3 镜像、D DP16/TP1/EP16、async/图/共享专家多流/
eager MTP；CPU 接入已验证，**真实 A3/NPU/PD 和性能仍待验收**。

## 1. 在原 Decode 容器获取并安装

```bash
git clone --branch feature/ascend-deepseek-v4-mtp-async-graph \
  https://github.com/zxTsai/moe-trace.git
cd moe-trace
python -m pip install --no-deps --no-build-isolation -e .
```

已有仓库时先保存自己的修改，再 git pull --ff-only。继续使用原服务 Python，
不构建镜像、不重装 vLLM、不修改原安装源码。更新后必须重新生成副本并重启 D。

## 2. 校验并生成采集副本

```bash
python scripts/tools/inspect_ascend_async_graph.py
OVERLAY=$(python scripts/tools/stage_ascend_async_graph.py \
  --output-parent "$HOME/.cache/moe-trace/ascend-023")
```

工具自动定位实际包，校验20个关键文件。匹配失败保留差异，不修改摘要绕过检查。
目的专家拥有者读取实际映射，并与本地 expert map 校验；不按 expert ID 猜 device。

## 3. 使用新的完整轨迹配置

```bash
cp configs/examples/ascend-023-routing-trace.json /tmp/moe-trace-routes.json
```

```json
{
  "mode": "routing_trace",
  "output_root": "/trace-output",
  "run_id": "dsv4-routes-001",
  "node_id": "decode",
  "interval_seconds": 5,
  "max_buffer_bytes": 16777216,
  "max_file_bytes": 67108864,
  "max_files": 4,
  "save_token_ids": false,
  "save_expert_weights": false
}
```

修改 output_root 为容器已有可写持久目录；每次启动换新 run_id。
默认每 worker 缓冲最多16MiB序列化事件，满前提前输出一个 partial 窗口，不丢记录；
单次执行本身超过上限则明确失败，需要增大缓冲。该限制不是整个 Python 进程 RAM 上限，
序列化/解析时还有临时对象内存。默认每 worker 文件64MiB、保留4个，16个worker
约4GiB主机存储，旧窗口轮转删除并明确标记历史不完整。

默认保存实际输入位置，不保存词表 token ID 和权重；需要核对词表 ID 时设
save_token_ids=true。同一个词表 ID 出现在不同输入位置仍分别保存，不能按 ID 去重。

## 4. 包装原 D 启动命令

```bash
python scripts/deployment/ascend_023_debug.py \
  --overlay "$OVERLAY" --config /tmp/moe-trace-routes.json --check

python scripts/deployment/ascend_023_debug.py \
  --overlay "$OVERLAY" --config /tmp/moe-trace-routes.json -- \
  bash /path/to/start_decode.sh 原来的全部参数
```

替换为已经跑通的原脚本与参数，P、proxy、AISBench 用法不变。
原脚本不要覆盖包装器传入的 PYTHONPATH。原 async/图/MTP 参数保留，也允许主模型 eager。
不同时启用内置专家路由返回、旧采集 profile、EPLB、SP 或 static kernel。

插件只由 DP0 输出一次启停 INFO，不逐 DP 打印周期 INFO。各 worker 异常仍报错。
vLLM 自己的日志不因此关闭；若仍有原框架统计刷屏，可在原脚本中按需要使用
`--disable-log-stats` 关闭原吞吐统计。也可显式使用下面的日志级别环境变量，保留 WARNING/ERROR：

```bash
VLLM_LOGGING_LEVEL=WARNING python scripts/deployment/ascend_023_debug.py \
  --overlay "$OVERLAY" --config /tmp/moe-trace-routes.json -- \
  bash /path/to/start_decode.sh 原来的全部参数
```

## 5. 输出按源 DP 存放

```text
/trace-output/dsv4-routes-001/decode/
├── source-dp-000/
│   ├── routes.jsonl
│   ├── routes.1.jsonl    轮转后出现
│   ├── status.json
│   ├── worker.json
│   └── placements.json
├── source-dp-001/
└── ... source-dp-015/
```

一个源 worker 只写自己的文件，所有 D worker 应能访问同一个输出目录，便于解析宿身份。
worker.json 保存实际 DP/TP/EP、hostname、服务进程看到的 device_index 和 visible_devices，
不要把进程逻辑编号直接当作物理卡/Die。placements.json 保存每层实际专家归属。

没有有效 Decode/MTP 数据时不写空窗口，也不打印周期日志；元数据和空的 routes.jsonl
可在初始化时存在。没有新请求但旧请求仍在生成时继续采集。空闲 DP 仍可能需要参加
原模型通信或 dummy 图重放，插件不停止这部分服务行为；图中的路由 copy 不等于完全移除。

## 6. 看一个窗口的原始 Top-K

routes.jsonl 每行是一个批量窗口，嵌套结构是：

```text
rounds[]
  executions[]
    requests[]
      tokens[]
        layers[] → expert_ids[] 与 destination_ep_ranks[] 逐项对齐
```

下面是简化字段示意，目的映射仅为演示，**不是 A3 实测结果**：

```json
{
  "window_id": 0,
  "source": {"dp_rank": 0, "device_index": 0},
  "rounds": [
    {
      "schedule_seq": 20,
      "executions": [
        {
          "execution_id": "dp-0:execution-35",
          "scope": "target",
          "mtp_step": null,
          "requests": [
            {
              "request_id": "A",
              "tokens": [
                {
                  "input_row": 0,
                  "token_position": 101,
                  "kind": "target_verify",
                  "layers": [
                    {
                      "layer_id": 1,
                      "expert_ids": [0, 3, 5, 6, 12, 34],
                      "destination_ep_ranks": [7, 1, 10, 3, 4, 2]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

原始结果包含全部有效 token 和 MoE 层，以及 device_table：按 EP rank 查询实际宿 device。
layer_id 沿用模型原始编号，不把第一个MoE层人工改成1。
同一宿可能在目的数组出现多次，分别对应不同专家，不去重，也不变成累计次数。
每个请求在本轮可以有多个 token，每个 token 有自己的位置、输入行和逐层 Top-K。

调度与执行语义：

- schedule_seq 是当前源 DP 引擎观测到的 target forward 序号；不是 vLLM 提供的全局ID，
  不同 DP 同号轮次不代表同步的一轮。CPU 独立 MTP 方法测试可能从未观测主模型开始，序号为0。
- 同轮主模型与 MTP 用不同 execution_id，MTP 第几次预测由 mtp_step=0/1/... 区分。
- 主模型位置是实际序列位置；MTP token_position 是 kernel 位置，input_sequence_position
  另存为 kernel+1。词表 token ID 可选，不能混淆为位置或请求 ID。
- 5秒边界或缓冲上限可能把同轮的不同执行分到两个窗口，按源+schedule_seq/execution_id 关联。
- 窗口按主机侧已采集完成的事件到达切分，不是严格按GPU执行时间切片。
- sampler_events/proposal_events 保留候选生产和接受/拒绝辅助信息，也可能跨窗口关联。
  拒绝候选已经执行的路由保留。

## 7. 导出可直接打开的 JSON

正常停止 D 后，验证全体源 worker：

```bash
python scripts/validation/inspect_ascend_routing_trace.py \
  /trace-output/dsv4-routes-001 \
  --output /trace-output/dsv4-routes-001/trace-check.json
```

导出源 DP0 的第0个写入窗口，保留原始轮次/执行/请求/token/层结构：

```bash
python scripts/validation/inspect_ascend_routing_trace.py \
  /trace-output/dsv4-routes-001 --source-dp 0 --window 0 \
  --output /trace-output/dsv4-routes-001/source-dp-000-window-0.json
```

可用 --allow-incomplete 查看运行中或不完整退出的数据，并发写入/轮转时读取失败请重试。
检查源身份、Top-K顺序与实际宿映射、逐token层覆盖、执行ID以及辅助事件引用。
full_history_retained=false 说明旧文件已删除，只能验证/查看当前保留窗口。
正常结束输出剩余 partial 数据；强杀可能丢未落盘缓冲，recording/incomplete 不能当完整结束。

**该模式比次数汇总数据量更大，且仍需要实际输入和路由的设备同步/CPU拷贝。**
定时写盘不意味着消除采集开销，真实 A3 的吞吐、TPOT、P99 必须对照测量。
这些是 token 原属设备到专家拥有者的逻辑路由，不是 HCCL/KV 字节或逐跳通信流量。

恢复原版本：停止本次 D，在未经过包装器的终端直接运行原脚本即可，无需重装框架。
