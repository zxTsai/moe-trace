# 验证索引

## 当前结论

Qwen3.6 单机 A3 DP8/TP2 的实验采集已完成 CPU 源码接入测试；
见[采集功能 CPU 验证](Qwen3.6采集功能CPU验证.md)，A3 真机未验收。
实验版，尚未完成全部可靠性验收。最新完整矩阵见
[扩展验证记录](full-validation-2026-10-03.md)：两版本真实采集及独立参考、
AISBench、关闭采集、过滤、小分片和受控写失败测试。
长期历史索引增长、真实设备/进程故障、AISBench 样本映射仍未完全覆盖；
全新环境在线安装已实际尝试，但因网络超时失败，不能标记完成；缓存及临时环境保留。

结构重构后的回归见[重构验证记录](repository-refactor-validation.md)。
Ascend feature 的[CPU 验证报告](ascend-cpu-validation.md)只覆盖 CPU 模拟及固定源码
Python 接入点；A3/NPU 验收待完成，与 NVIDIA 真实服务报告分开。
V4.1 分布式开发阶段见[Decode CPU 契约验证](distributed-decode-cpu-validation.md)，
不能据此宣称 PD、EP 或 ACL 图执行已经接入。
两机调试版见[V4.1 Debug 开发验证](ascend-v41-debug-validation.md)：启动工具、D-side 接入、
16-worker 汇总及 274 项 CPU 回归已完成；真实 A3 PD 服务与设备验收仍待执行。

## 按版本与场景查找

| 场景 | 报告 |
| --- | --- |
| 0.10.1 初始环境 | [baseline](baseline.md) |
| 0.10.1 初始 GPU | [GPU 验证](gpu-validation-2026-10-02.md) |
| 0.10.1 独立参考 | [输入和路由核验](reference-validation.md) |
| 0.10.1 生命周期 | [请求生命周期](lifecycle-validation.md) |
| 0.29.0 路由与权重 | [GPU 验证](vllm-0.29.0-validation.md) |
| 0.29.0 原 AISBench 实验 | [20 请求、2 并发](aisbench-vllm-0.29.0-validation.md) |
| main 整合 | [整合回归](main-integration-validation.md) |
| main 真实服务 | [8 请求真实执行](main-live-validation-2026-10-03.md) |
| Ascend 0.23.0 CPU | [CPU 采集与固定源码接入](ascend-cpu-validation.md) |
| 扩展验证 | [双版本、4 并发 AISBench、过滤等](full-validation-2026-10-03.md) |

JSON 小规模摘要与报告放在同一目录。原始轨迹、NPZ、大型 JSON 和压测报告
不提交 Git；本地临时证据可能被系统清理。
[早期验收快照](acceptance.md)保留历史，不能用其旧表推断当前实现状态。
