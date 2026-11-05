# DeepSeek V4 逐请求源宿 CPU 验证

范围：request_flow，每窗口逐请求统计源 DP worker/device 到逻辑专家实际宿 EP rank/device。
不是网络字节、KV 传输或通信消息测量；真实 A3 与性能未验收。

验证包括：

- 多请求并发且顺序倒置，保持请求与源宿选择分别计数，不合并到全请求总量。
- 使用故意打乱的专家归属，证明目的端来自映射而不是 expert_id 或简单整除猜测。
- 一个 token 选择同一宿的两个专家，输出2次专家选择而非伪造唯一 token/消息数。
- 同请求跨窗口增量、未知宿元数据 unresolved、请求容量超限提前输出与超大batch拒绝。
- 原框架 determine_expert_map 与 Ascend init_eplb_config 的真实 CPU 方法执行，
  并与运行模块 local expert map 对照；重复/缺失 owner、实际映射不匹配明确拒绝。
- 执行真实参考 MTP Python 循环 K=1/2/3，三种模式（trace/summary/request_flow）
  的 target/MTP 总量一致，并汇总16个CPU worker身份（1活跃、15空闲）。
- 原命令包装器预检和参数透传覆盖 request_flow；关键源码指纹增加专家映射实现到20个文件。

CPU 使用真实 torch 张量，模型、router、专家 kernel、采样、NPU graph/event 和通信仍是 stub。
16个worker为文件与元数据身份模拟，不是16个真实 NPU 服务进程。

复验：

```bash
python scripts/validation/ascend_023_source_cpu.py \
  --vllm-source /path/to/vllm-v0.23.0 \
  --ascend-source /path/to/vllm-ascend-v0.23.0.post1 \
  --output-root ./outputs/ascend-request-flow-cpu
```

源版本沿用[异步图接入报告](DeepSeekV4异步图执行采集CPU验证.md)的 release 基线。
本轮专项 **72 passed**；提供固定历史源码后的全仓 **422 passed，无跳过**。
快速开始的5段Bash通过 bash -n，2段JSON通过解析，仓库文档链接检查通过。

本机复验产物（仅证据定位，不是用户安装路径）：
`/home/czx/code/moe-routing/outputs/ascend-request-flow-cpu/ascend-023-source-cpu-hymxt756/`。
其中 request-flow-analysis-k3.json 来自真实 CLI 读取 CPU 场景，16个worker身份、
2个请求、target=16、mtp=12、combined=28，未解析目的端为空。
上述 device/rank 是 CPU fixture 的元数据，不是真实 A3 device 验收结果。
