# 配置

| 目录/文件 | 用途 |
| --- | --- |
| `examples/default.yaml` | 完整参数模板，默认关闭采集 |
| `examples/vllm-0.10.1.yaml` | 0.10.1 用户示例 |
| `examples/vllm-0.29.0.yaml` | 0.29.0 用户示例 |
| `examples/ascend-vllm-0.23.0.yaml` | Ascend 实验配置，默认关闭，CPU 验证不等于 A3 验收 |
| `validation/` | 历史实验配置，可能包含维护者机器路径，不用于通用快速开始 |

用户示例输出到 `./outputs`，相对于服务启动目录。可复制到仓库根 `trace.yaml`
并编辑；该个人配置被 Git 忽略。YAML 路径不展开环境变量或 `~`。
环境向导另行生成绝对路径配置，默认输出到 `outputs/traces`。
`vllm` 配置段只是约束，不会自动代替服务启动参数。
字段和限制见[使用指南](../docs/guides/usage.md)。
