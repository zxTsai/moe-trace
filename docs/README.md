# 文档导航

首次使用从[仓库 README](../README.md)开始。所有用户命令默认在仓库根目录运行。

## 使用指南

- [环境准备](guides/environment-setup.md)：版本检测、Conda 隔离、安装确认、失败处理。
- [完整使用](guides/usage.md)：手动接入、配置、字段、分析、AISBench 与当前限制。
- [开发指南](guides/development.md)：目录边界、测试、脚本参数和旧路径迁移。
- [配置说明](../configs/README.md)与[脚本说明](../scripts/README.md)。

## 需求与设计

- [需求和操作边界](requirements.md)：原始开发约定。
- [公共采集及数据格式](design/design.md)：输入行、执行关联、分片和校验。
- [0.29.0 版本接入](design/vllm-0.29.0-design.md)。
- [环境准备设计](design/environment-setup-design.md)。
- [仓库结构重构设计](design/repository-refactor-design.md)。
- 补丁：[0.10.1](../patches/vllm/0.10.1/README.md)、[0.29.0](../patches/vllm/0.29.0/README.md)。

## 验证

[验证索引](validation/README.md)区分当前结论和历史证据。
报告中的绝对机器路径仅用于定位当时实验，不是安装步骤。自动化或模拟测试
不替代真实 GPU 核验，不把历史版本报告当作所有版本均已验收。
