# Qwen3.6 0.23.0.post1 采集补丁

只应用到隔离副本，不原地修改用户安装。固定源码摘要在 `source-lock.json`，
补丁在 `0001-qwen-hooks.patch`，修改文件摘要在 `overlay.json`。

使用仓库 `scripts/tools/stage_qwen36_ascend.py` 创建副本；脚本检查原文件摘要、唯一锚点及
Python AST。启动器再次核对副本摘要和实际导入路径，默认不执行服务。

修改三个文件：runner 实际 forward 区块、W8A8 MoE.apply 的逻辑 selector/consumer、
worker load_model 后初始化与 shutdown 前排空。
CPU 源码接入验证通过，A3 待验收；支持范围及命令见
[Qwen 快速开始](../../../docs/guides/Qwen3.6/README.md)。

停用：停止本次采集服务，使用原包启动服务，不传采集配置或副本 PYTHONPATH。
原包没有被改写，不需要反向打补丁，不自动删除用户的实验副本或结果。
