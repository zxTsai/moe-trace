# 仓库结构重构设计

目标：按职责组织仓库，而不是按机器或开发阶段组织；不改变路由语义、
持久化 schema、公开采集 CLI 或 vLLM 补丁的 adapter 导入路径。

## 目录边界

- `src/moe_trace/` 保留采集、读写、分析、配置和 CLI 公共模块。
- `src/moe_trace/setup/` 负责标准库环境检测、安装计划、隔离补丁与加载文件。
- `src/moe_trace/validation/` 为可选独立参考探针，不混入版本 adapter。
- `scripts/prepare_environment.py` 保留面向用户的无依赖入口；
  `scripts/tools/` 为手动隔离补丁工具，`scripts/dev/` 为补丁维护，
  `scripts/validation/` 为真实请求工作负载和离线核验。
- `configs/examples/` 为用户可复制配置，`configs/validation/` 为历史实验配置。
- `docs/guides/` 为使用指南，`docs/design/` 为设计，`docs/validation/`
  为按日期/版本保留的验证证据；`docs/requirements.md` 保持位置。
- README 只放概览、获取源码、推荐快速开始、当前限制和导航。

## 兼容与安全

保留 `moe-trace`、`moe-trace-env`、`python -m moe_trace.environment`、
`scripts/prepare_environment.py`、`scripts/stage_vllm_029.py` 的旧入口。
保留旧独立环境探针脚本的转发入口，使正在下载的安装任务完成后仍能检测。
底层采集模块和两个 adapter 路径不变。移走的开发/验证脚本、配置及文档路径
统一更新引用，并提供迁移清单。历史机器路径只作为证据，不用于新用户步骤。
所有已有未提交修复保留；不重启下载，不清理输出，不修改现有推理依赖。

## 验证和同步

双环境 pytest、源码/脚本编译、标准库入口检测、所有工具 help、文档内部链接、
配置解析、已保存 GPU 轨迹重验与补丁隔离导入检查。必要时启动小规模真实服务
验证移动后的独立探针。先提交本地 main，再普通 push origin main；不强推。
若远程已前进则保留双方提交并处理，不重置未知工作。
