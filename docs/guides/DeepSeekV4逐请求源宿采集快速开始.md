# DeepSeek V4 逐请求源宿采集快速开始

你要的统计口径是：**实际输入 token 所属 DP worker/device → 被选专家部署的 EP worker/device**，
按请求分别累计，每个时间窗口输出一次。一个请求有多条目的关系，跨窗口只输出新增计数。
不是客户端 IP、P→D KV 传输、网络流量或消息数。

适用原官方 v0.23.0.post1-a3 镜像，D DP16/TP1/EP16，单 D 主机，async/图/共享专家多流/
eager MTP。CPU 接入已测试，真实 A3 与性能未验收。安装及原脚本包装方式不变。

## 1. 更新插件和源码副本

首次下载：

```bash
git clone --branch feature/ascend-deepseek-v4-mtp-async-graph \
  https://github.com/zxTsai/moe-trace.git
cd moe-trace
python -m pip install --no-deps --no-build-isolation -e .
```

已有本分支时保存自己的修改，再 `git pull --ff-only`，然后重新 stage。
不重装框架、不构建新镜像、不修改原服务源码：

```bash
python scripts/tools/inspect_ascend_async_graph.py
OVERLAY=$(python scripts/tools/stage_ascend_async_graph.py \
  --output-parent "$HOME/.cache/moe-trace/ascend-023")
```

当前校验 20 个参考文件，包含框架实际使用的专家放置实现。
初始化读取每层实际 global_expert_map；默认没有全局表时调用框架同版本映射提供函数，
并校验其结果与模块实际 local expert map 一致。拒绝缺失、重复拥有者、EPLB 和不明确副本，
不会直接按 expert ID 猜目标 device。

## 2. 使用逐请求源宿配置

```bash
cp configs/examples/ascend-023-request-flow.json /tmp/moe-trace-flow.json
```

```json
{
  "mode": "request_flow",
  "output_root": "/trace-output",
  "run_id": "dsv4-flow-001",
  "node_id": "decode",
  "interval_seconds": 5,
  "max_file_bytes": 16777216,
  "max_files": 4,
  "max_requests_per_window": 128
}
```

把 output_root 改为原容器已有可写持久目录。每次启动换新的 run_id。
默认每5秒输出、每 worker 保留4个16MiB文件，旧文件自动轮转删除。
每窗口最多128个不同请求；超过前提前输出 partial 窗口再继续，不丢请求。
单个 batch 请求数超过上限会明确失败，因此该参数应至少覆盖原 max-num-seqs。

逐请求设备计数占用比全请求合并 summary 更多的内存和统计算子；第一版不承诺开销比例，
不能假定两种模式性能相同。它仍避免逐 token 的 host 转换，定期只回传计数。

## 3. 包装原 Decode 脚本

```bash
python scripts/deployment/ascend_023_debug.py \
  --overlay "$OVERLAY" --config /tmp/moe-trace-flow.json --check

python scripts/deployment/ascend_023_debug.py \
  --overlay "$OVERLAY" --config /tmp/moe-trace-flow.json -- \
  bash /path/to/start_decode.sh 原来的全部参数
```

换成已经跑通的原脚本与参数，重启 D；P、proxy、客户端/AISBench 的用法不变。
原脚本不要覆盖包装器的 PYTHONPATH。停止本次 D 后直接运行原命令，可以恢复原框架路径。

## 4. 看逐请求结果

每 worker 目录除了 load.jsonl/轮转文件/status.json，还增加：

- worker.json：本进程的 node、hostname、DP/TP/EP、device_index 和可见设备配置。
- placements.json：每层逻辑专家的唯一 EP owner，供核对与分析。

load.jsonl 每一行是一个窗口，每个窗口的 requests 数组中，**每个元素是一个请求**。
下面是简化示意，不是 A3 实测结果：

```json
{
  "type": "request_flow_summary",
  "window_id": 10,
  "duration_seconds": 5.01,
  "expert_choices": {"target": 4, "mtp": 0, "combined": 4},
  "requests": [
    {
      "request_id": "A",
      "source": {"dp_rank": 0, "ep_rank": 0, "hostname": "A3-D", "device_index": 0},
      "layers": [
        {
          "scope": "target",
          "kind": "target_decode",
          "layer_id": 0,
          "observed_input_tokens": 2,
          "top_k": 2,
          "destinations": [
            {
              "destination": {"ep_rank": 1, "device_index": 1, "identity_status": "resolved"},
              "expert_selection_count": 2,
              "expert_counts": [{"expert_id": 1, "count": 2}]
            },
            {
              "destination": {"ep_rank": 3, "device_index": 3, "identity_status": "resolved"},
              "expert_selection_count": 2,
              "expert_counts": [{"expert_id": 2, "count": 2}]
            }
          ]
        }
      ]
    }
  ]
}
```

意思是：该窗口中，请求 A 的源是 DP0/device0，第0层观察到2次有效输入，
分别对宿 EP1、EP3 各产生2次专家选择。示意映射故意不是 expert ID 等于 device ID。

实际结果还包含 layer_name、源宿 node/hostname/可见设备、窗口时间及映射摘要。
device_index 是该进程看到的逻辑 NPU 编号，需结合 hostname 和 visible_devices 定位，
不直接冒充物理卡/Die 编号。启动时尚未找到宿 worker.json 时，目的 EP rank 仍记录，
设备身份标 unresolved；需收齐各 worker 的元数据核对，不能把它当作已解析物理设备。

计数是 expert_selection_count：一个 token 选择同一 device 上两个专家，计2次选择，
不表示2个独立 token，也不表示2条网络消息。被拒绝候选已执行的选择也计入。
target_decode、target_verify、mtp_draft 分开；不记录候选接受率或每个 token 的位置明细。
要完整 token 轨迹仍使用 trace 模式。

同一个请求可以出现在多个窗口，当前窗口只记录增量。没有实际 Decode 执行的窗口
requests=[]；旧请求还在生成时继续记录，即使没有新请求进入。
request_id 为引擎内部请求标识，不保证直接等于 HTTP 响应 ID；客户端关联需要另行核对。
窗口由 worker 本地时钟切分，不把16个 worker 同号窗口称为一次全局调度。

## 5. 分析与检查

正常停止 D 后：

```bash
python scripts/validation/analyze_ascend_request_flow.py \
  /trace-output/dsv4-flow-001 \
  --output /trace-output/dsv4-flow-001/request-flow-analysis.json
```

运行中临时查看可加 --allow-incomplete；并发写入或轮转时读取失败可稍后重试。
分析校验 source、实际专家 owner、计数和 Top-K 总数，按源 EP 和请求 ID 分别汇总保留窗口。
检查 worker_coverage_complete、unresolved_destination_ranks 和各 worker 状态。
full_history_retained=false 表示旧窗口已轮转删除，只能统计仍保留的窗口。

旧 summary 模式合并所有请求，只看专家分布；本指南的 request_flow 才是逐请求源宿。
如果源/宿都是同一 device，这代表本地专家选择，不一定发生跨设备传输。
本结果不能用于推算 HCCL 字节数、通信耗时或每卡利用率。
AllGather/MC2 等实际通信路径可能经过其他设备，本功能描述 token 原属设备到专家拥有者的
逻辑选择关系，并非逐跳网络发送方/接收方探针。
