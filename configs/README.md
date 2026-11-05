# 配置

| 目录/文件 | 用途 |
| --- | --- |
| `examples/default.yaml` | 完整参数模板，默认关闭采集 |
| `examples/vllm-0.10.1.yaml` | 0.10.1 用户示例 |
| `examples/vllm-0.29.0.yaml` | 0.29.0 用户示例 |
| `examples/ascend-vllm-0.23.0.yaml` | Ascend 实验配置，默认关闭，CPU 验证不等于 A3 验收 |
| `examples/ascend-023-async-graph.json` | V4 现有 A3 容器调试：D16/TP1、异步/图/eager MTP；显式包装启动时启用，绝对输出路径和采集上限 |
| `examples/ascend-023-load-summary.json` | 同一环境的连续负载汇总：默认每 5 秒回传，16 MiB/文件、保留 4 文件/worker |
| `examples/ascend-023-request-flow.json` | 逐请求 DP 源到专家 EP 宿，默认每5秒，窗口最多128个请求（超过前提前输出） |
| `examples/ascend-023-routing-trace.json` | 当前推荐：调度/执行/请求/token/层的原始有序Top-K源宿，5秒批量落盘、空闲静默 |
| `validation/` | 历史实验配置，可能包含维护者机器路径，不用于通用快速开始 |

用户示例输出到 `./outputs`，相对于服务启动目录。可复制到仓库根 `trace.yaml`
并编辑；该个人配置被 Git 忽略。YAML 路径不展开环境变量或 `~`。
环境向导另行生成绝对路径配置，默认输出到 `outputs/traces`。
`vllm` 配置段只是约束，不会自动代替服务启动参数。
字段和限制见[使用指南](../docs/guides/usage.md)。

上述相对输出路径约定用于通用 YAML。A3 异步图调试 JSON 使用绝对路径 `/trace-output`，
必须按容器实际挂载/权限修改，使用方法见[现有 A3 容器指南](../docs/guides/DeepSeekV4现有A3容器采集调试指南.md)。
