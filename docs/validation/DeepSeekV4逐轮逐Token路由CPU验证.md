# DeepSeek V4 逐轮逐 token 路由 CPU 验证

模式 routing_trace，区别于旧 request_flow 的次数汇总。当前保存源、观测调度序号、
实际执行、请求、输入行/位置、逐层有序 Top-K、逐项目的 EP 和设备表。
CPU 软件验证不代表真实 A3/NPU/PD 服务或性能验收。

## 覆盖

- 请求内多个 token、跨轮同请求及重复词表 token ID 在不同位置保留，不按值去重。
- 每层 Top-K 顺序与故意非线性的专家归属逐项对照，不仅比较总计数。
- 主模型与 MTP 同一观测调度序号但不同 execution_id，MTP step 单独记录。
- 主模型验证和 MTP 输入，包含拒绝候选已执行路由；sampler 可跨输出窗口引用执行。
- 源目录 source-dp-000、宿设备表、位置/可选词表ID/权重隐私。
- 实际参考源码 runner/W8A8/采样/MTP Python 循环 K1/2/3 和模拟图重放接入。
- 后台定时批量写，不产生空窗口，不刷周期INFO，容量满前提前输出、单执行超限失败。
- 完整窗口文件轮转、正常关闭残余partial、缺少验证结果 incomplete、损坏映射拒绝。
- 16个CPU worker身份（1活跃15空闲）读取验证、指定源/窗口完整JSON导出。

源版本与20项文件摘要沿用当前 v0.23.0.post1-A3 参考。模型、router/专家kernel、
采样输出、NPU graph/event 和通信使用 stub，未运行真实16个NPU服务进程。

## 复验

```bash
python scripts/validation/ascend_023_source_cpu.py \
  --vllm-source /path/to/vllm-v0.23.0 \
  --ascend-source /path/to/vllm-ascend-v0.23.0.post1 \
  --output-root ./outputs/ascend-routing-trace-cpu
```

会保存 pytest 日志、CPU报告、参考源码副本和每个测试场景的原始 routes.jsonl。
本轮专项 **92 passed**，全仓 **442 passed，无跳过**。
新指南7段Bash通过bash -n，2段JSON通过解析，标准库help及文档链接检查通过。

本机证据目录（不是用户安装路径）：
`/home/czx/code/moe-routing/outputs/ascend-routing-trace-cpu/ascend-023-source-cpu-3i_dxf3y/`。
实际CLI生成 trace-check-k3.json 和 source-dp-000-window-0.json，验证16个CPU worker身份，
其中1个活跃、15个空闲；K=3保留主模型8个输入实例和MTP6个输入实例，共14条逐层路由，
每条Top-K=2，计数自检target=16/mtp=12/合计28。原始明细没有合并成这些计数。
独立的多层测试另使用6个有序专家、非线性目的rank，检查逐项数组、位置和源目录。

## 边界

schedule_seq 是当前源DP观测到的主模型 forward 序号，不是全局scheduler ID。
独立CPU MTP方法场景可能使用0（此前未调用主模型入口）；真实服务中MTP跟随其主模型序号。
完整轨迹仍有输入/路由D2H和设备等待。5秒批量落盘只减少高频写入，不能声称等同summary性能。
硬件正确性、吞吐/TPOT/P99、多流/图和长时间资源使用必须实际A3验证。

操作见[完整路由快速开始](../guides/DeepSeekV4逐轮逐Token路由快速开始.md)。
