# 脚本入口

以下路径相对于仓库根目录，先用 `--help` 查看参数。

| 入口 | 职责 |
| --- | --- |
| `prepare_environment.py` | 标准库环境准备入口，不必安装工具 |
| `tools/stage_vllm_029.py` | 手动创建 SHA256/冲突检查后的隔离补丁包 |
| `tools/stage_ascend_023.py` | 标准库检查/复制两个固定版本的 Ascend 接入包 |
| `tools/check_ascend_v41_sources.py` | V4.1 调查源码 commit/hash/AST 只读检查，不安装、不打 patch |
| `dev/build_ascend_023_patch.py` | 从固定干净 Git 源码机械生成 Ascend 补丁及哈希 |
| `validation/ascend_cpu.py` | CPU 模拟采集及固定源码 hook 与独立 consumer 对照 |
| `dev/build_vllm_029_patch.py` | 从未修改的安装包生成补丁到 stdout，不应用补丁 |
| `validation/reference_requests.py` | 错峰并发和长提示词请求 |
| `validation/lifecycle_requests.py` | stop/length/流式取消/取消后请求 |
| `validation/verify_reference.py` | 对照独立 GPU 输入及路由消费探针 |
| `validation/verify_lifecycle.py` | 对照客户端结果和内部生命周期 |
| `validation/verify_exports.py` | JSON/CSV 与原始轨迹重算比较 |

请求脚本接受 `--url`、`--model`、`--output`；默认地址
`http://127.0.0.1:8000/v1/completions`，模型名 `granite-moe`。
工作负载和 stop token 用例针对已验证的 Granite 模型；不是通用模型压测器。
请求脚本不启动服务，核验脚本不安装依赖；独立探针必须在启动服务前显式启用。

`stage_vllm_029.py` 旧根目录入口保留为转发，但新文档使用 `tools/`。
详细步骤见[开发指南](../docs/guides/development.md)。
