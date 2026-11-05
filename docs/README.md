# 文档导航

首次使用从[仓库 README](../README.md)开始。所有用户命令默认在仓库根目录运行。

[当前分支更新说明](更新说明.md)记录功能变化、提交检查点、支持边界及 CPU/A3 验收状态。

## 使用指南

- [DeepSeek V4 逐请求源宿快速开始](guides/DeepSeekV4逐请求源宿采集快速开始.md)：每窗口按请求记录源 DP/device 到实际专家宿 EP/device 的选择次数。

- [DeepSeek V4 周期负载汇总快速开始](guides/DeepSeekV4周期负载汇总快速开始.md)：现有官方 D 容器安装，连续计数、默认 5 秒输出 JSONL 与文件轮转，CPU 验证通过，A3 待验收。

- [DeepSeek V4 现有 A3 容器采集调试](guides/DeepSeekV4现有A3容器采集调试指南.md)：保留原异步/图/多流/MTP 配置，只在 D 安装插件与副本；CPU 通过，真机待验收。

- [DeepSeek V4 + MTP 从零部署](guides/DeepSeekV4开启MTP采集指南.md)：宿主机、镜像构建/复用、权重、P/D/proxy、插件、动态 K 路由和结果分析；A3 待验收。

- [DeepSeek V4 两机 A3 快速开始](guides/DeepSeekV4两机A3快速开始.md)：P4×4、D16×1，同步 eager、无 MTP，安装、采集与负载导出；A3 待验收。

- [Qwen3.6 单机 A3 快速开始](guides/Qwen3.6/README.md)：获取代码、环境检查、隔离采集接入、JSON/CSV 汇总与 CPU 验证；A3 待验收。
- [A3 新手快速使用指南](guides/A3新手快速使用指南.md)：两机 V4.1 1P1D 环境、部署、采集和 JSON 查看（待 A3 真机验收）。
- [环境准备](guides/environment-setup.md)：版本检测、Conda 隔离、安装确认、失败处理。
- [完整使用](guides/usage.md)：手动接入、配置、字段、分析、AISBench 与当前限制。
- [开发指南](guides/development.md)：目录边界、测试、脚本参数和旧路径迁移。
- [Ascend CPU 开发环境](guides/ascend-cpu-development.md)：独立测试环境及 CPU / A3 验证边界。
- [Ascend 完整快速开始](guides/ascend-quickstart.md)：官方环境、原服务验证、安装接入、重启采集与结果分析（待 A3 验收）。
- [分布式 Decode CPU 验证](guides/distributed-decode-cpu.md)：V4.1 事件契约与分片汇总原型，不是服务安装指南。
- [配置说明](../configs/README.md)与[脚本说明](../scripts/README.md)。

## 需求与设计

- [逐请求源宿汇总设计](design/DeepSeekV4逐请求源宿汇总设计.md)：实际专家映射、设备身份、请求上限和增量窗口。

- [周期负载汇总第一版设计](design/DeepSeekV4周期负载汇总设计.md)：设备计数、定时线程、事件依赖、滚动保留与 trace 边界。

- [保留异步、图执行和多流的适配设计](design/DeepSeekV4保留异步图执行多流的采集设计.md)：目标镜像 v0.23.0.post1-a3，真机调试入口及 CPU 接入已实现。

- [DeepSeek V4 的 MTP 采集设计](design/DeepSeekV4的MTP采集设计.md)：候选产物、实际预测输入、sampler 决策及 scope 完整性。

- [DeepSeek V4 两机采集适配设计](design/DeepSeekV4两机采集适配设计.md)：独立 patch、hash 路由、16 worker 和输入覆盖校验。

- [Qwen3.6 单机 A3 采集适配设计](design/Qwen3.6单机A3采集适配设计.md)：DP8/TP2、0.23.0.post1 接入与支持边界。
- [需求和操作边界](requirements.md)：原始开发约定。
- [公共采集及数据格式](design/design.md)：输入行、执行关联、分片和校验。
- [0.29.0 版本接入](design/vllm-0.29.0-design.md)。
- [Ascend 0.23.0 CPU 适配](design/ascend-vllm-0.23.0-design.md)：源码基线、hook、限制和验收边界。
- [DeepSeek V4.1 Decode 设计](design/ascend-deepseek-v41-decode-design.md)：PD、DSpark、EP 和图执行的开发边界。
- [环境准备设计](design/environment-setup-design.md)。
- [仓库结构重构设计](design/repository-refactor-design.md)。
- 补丁：[0.10.1](../patches/vllm/0.10.1/README.md)、[0.29.0](../patches/vllm/0.29.0/README.md)。

## 验证

[验证索引](validation/README.md)区分当前结论和历史证据。
报告中的绝对机器路径仅用于定位当时实验，不是安装步骤。自动化或模拟测试
不替代真实 GPU 核验，不把历史版本报告当作所有版本均已验收。
