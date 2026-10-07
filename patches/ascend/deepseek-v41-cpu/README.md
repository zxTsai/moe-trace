# DeepSeek V4.1 固定源码 CPU hook

这是 CPU 接入验证补丁，不是 A3/PD 服务部署补丁。
基线 commit 见 [source-hooks.json](source-hooks.json) 与
[源码锁](../../../docs/design/ascend-v41-source-lock.json)。

修改 3 个 Ascend 文件：

1. model_runner_v1.py 的 `_model_forward`：核对真实 forward 的 token ID/位置。
2. routed_experts.py 的 `_select_experts`：router 选择后、log2phy/shared 追加前采集。
3. w8a8_dynamic.py 的 `apply`：实际专家 consumer 调用前保存其输入。

用 [stage 工具](../../../scripts/tools/stage_ascend_v41_cpu.py) 创建唯一隔离副本，
校验原始 commit、10 文件哈希、唯一源码锚点与 AST。它不改 vLLM/Ascend 原始源码，
也不修改 PYTHONPATH、不安装包。生成 patch 与已提交 patch 必须逐字一致。
不要向生产源码 git apply；撤销时停止使用副本即可，原始文件无需回滚。

hooks 默认没有上下文则 no-op；CPU harness 显式提供请求映射与 execution。
`MOE_TRACE_V41_SOURCE_HOOKS=1` 不会自动生成映射，没有上下文会报错。
活动采集拒绝 NPU、graph、SP、V2、profiling/force balancing；只证明固定源码
的 Python 调用链。真实 KV/DSpark、异步、多 rank 输入和 ACL 图仍待接入。

详见[CPU 指南](../../../docs/guides/distributed-decode-cpu.md)。
