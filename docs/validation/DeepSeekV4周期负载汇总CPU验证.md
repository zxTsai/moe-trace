# DeepSeek V4 周期负载汇总 CPU 验证

日期：2026-11-05。分支：feature/ascend-deepseek-v4-mtp-async-graph。

## 当前结论

第一版 summary 连续设备计数、后台定时回传、JSONL 轮转与 JSON 分析已实现。
保留 trace 模式和原官方镜像/启动参数，通过既有隔离副本入口选择配置。
本报告的软件验证使用真实 CPU 张量和参考源码 Python 方法；
**真实 A3、NPU scatter/event、多流、图捕获和性能收益尚未验收**。

## 验证内容

- CPU 独立 oracle 对照逐专家计数，排除 Prefill、padding 和其他 DP 的输入。
- target_decode、target_verify、mtp_draft 分开，保存零计数专家。
- 对相同参考 MTP K=1/2/3 场景，summary 与 trace 总计一致：

| K | target 次数 | MTP 次数 | 合计 |
| --- | --- | --- | --- |
| 1 | 8 | 4 | 12 |
| 2 | 12 | 8 | 20 |
| 3 | 16 | 12 | 28 |

场景只有一层、每 token 选择两个专家；数值不是 DeepSeek 真机专家分布。
通过原 reference `_run_merged_draft` 循环和 `set_inputs_first_pass` 执行 MTP，
不是只生成符合格式的伪造 JSON。模型、采样输出、路由算子和通信仍由 stub 提供。

- 实际 `ACLGraphWrapper.__call__` 在 CPU 模拟 capture/replay，路由 hook 首次只执行一次，
  重放后新的计数随路由变化，预热不计入负载。
- 窗口内多次 forward 没有调用插件的逐 token host 读取或全设备同步；
  用 CPU Tensor.cpu 观测验证每个有数据窗口仅一次紧凑向量转换。
- 定时线程在无请求时继续产生空窗口，正常退出输出 partial 窗口并停止线程。
- NPU event/同 device 依赖使用 stub 检查调用顺序；graph guard 覆盖预热及重放，
  防止后台设备拷贝与全局图捕获并发。这不能替代 NPU 事件实际安全性测试。
- 文件轮转有界，标明最早/最新保留窗口与历史删除；分析仅统计当前保留窗口。
- 计数越界、窗口过大、后台写失败、非法配置、损坏结果均明确报错。
- 完整轨迹、安装加载和历史 adapter 回归保留。

## 复验

沿用[异步图执行 CPU 验证报告](DeepSeekV4异步图执行采集CPU验证.md)的精确源码基线：
vLLM 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665，
Ascend 1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8。

```bash
python scripts/validation/ascend_023_source_cpu.py \
  --vllm-source /path/to/vllm-v0.23.0 \
  --ascend-source /path/to/vllm-ascend-v0.23.0.post1 \
  --output-root ./outputs/ascend-023-summary-cpu
```

此入口包含 summary、trace、图桥接和参考源码接入测试，保存 pytest 日志、
cpu-validation.json、源码副本和各场景 JSONL。

本轮专项 **58 passed**，提供所有固定历史源码后的全仓回归 **408 passed，无跳过**。
快速开始的 6 段 Bash 通过 `bash -n`、2 段 JSON 通过解析；安装/分析入口标准库
`--help` 及仓库文档链接检查通过。NPU API 的事件/传输 stream 行为仍为模拟检查，
上述数量不表示真实 A3/服务或性能测试通过。

本机复验产物（仅用于证据定位，不是用户安装路径）：
`/home/czx/code/moe-routing/outputs/ascend-023-summary-cpu/ascend-023-source-cpu-zsjk_ewb/`。
K=3 summary 的 CLI 分析结果保存在 `load-analysis-k3.json`，包含 16 个 CPU worker
的保留窗口（1 个活跃、15 个空闲），target=16、mtp=12、combined=28。
这些 worker 为模拟进程身份，没有启动 16 个 NPU 模型进程。

## 性能验收

当前没有 A3 性能测量数据，不报告开销百分比。汇总减少 host 明细转换、全设备等待及
高频 fsync，但仍有路由 clone、设备统计 kernel、NPU event、窗口回传与锁等待。
每层的统计算子可能产生额外 launch 开销，不能仅凭减少 JSON 大小就认定总性能更好。
请保持相同模型、并行、图/eager、MTP、输入输出和 AISBench 并发，分别比较
无插件、trace、summary 的输出吞吐、TPOT、TTFT 与 P99。每组预热并重复至少三次。
trace 必须保证正式测量期间未触顶停止写入，summary 必须无 failed。

第一版只做连续覆盖，不做间隔采样；窗口以各 worker 本地统计提交边界划分，
不声称多个 DP worker 同号窗口是同一全局调度轮次。
