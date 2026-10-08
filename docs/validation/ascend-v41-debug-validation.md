# 两机 V4.1 Debug 开发验证

日期：2026-10-08。分支：`feat/ascend-deepseek-v41-decode`。
代码检查点：`5341c10`。使用说明见[调试版快速开始](../guides/ascend-v41-debug-quickstart.md)。

## 已执行

- 固定 vLLM / Ascend 文件摘要检查通过；创建独立 Ascend 包副本，原包不修改。
- debug patch 对固定 Ascend 源码 `git apply --check` 通过。
- staging 重建补丁与仓库补丁字节一致，副本摘要与可信 manifest 一致。
- P/D dry-run 和参数单测：P4×4、D8×2 均覆盖逻辑设备 0–15；无 DSpark，eager、同步。
- 实际 patched runner 的 `with trace_target(...)` forward 接入块由 AST 提取后在 CPU 执行。
  模型回调、路由张量和 consumer 是 mock，未运行整个 GPU execute_model 或加载真实权重。
- 输入批次 B,A 与调度字典 A,B 顺序不同；B 的实际输入 token 21、position 1 正确关联。
  A 的 prompt 输入行被排除；TP0 保存 B 的第 20 层逻辑专家 `[3,1]`；映射后的
  consumer 对照 `[2,3,4]` 只写诊断，不混入逻辑统计。
- 16 个 worker 文件构造，15 个 idle worker 可正常收尾；跨 worker 汇总得到 1 条路由。
  删除末 worker 的 status 后校验失败，不伪装完整。
- 既有单测覆盖不同 prepare 类型、DP 不等长、SP padding、未知布局拒绝、实际输入核对、
  token ID 隐私、省略预热、配置保护。新增验证 idle-worker 初始化。

全量回归命令（环境路径仅是本次证据，不是用户安装要求）：

```bash
MOE_V41_VLLM_SOURCE=/tmp/moe-trace-vllm-v41-source \
MOE_V41_ASCEND_SOURCE=/tmp/moe-trace-ascend-v41-source \
MOE_ASCEND_SOURCE_OVERLAY=/tmp/moe-ascend-023-1zggohdj \
conda run --no-capture-output -n moe-ascend-cpu \
  python -m pytest -o addopts='' -q
```

结果：**274 passed in 6.23s**，无 skip。`git diff --check` 通过。
固定 vLLM 参数源码核实 `--async-scheduling` 使用 BooleanOptionalAction，允许显式
`--no-async-scheduling`。启动器实际用当前 Python 的 CLI 模块，避免 PATH 中另一个 vllm。

## 仍需现场验收

1. 固定镜像/源码和模型在两机调试配置中不采集时正常启动、PD proxy 请求返回成功。
2. 开启 D 采集后真实模型模块发现、prepare 类型与各 rank 行布局符合实现假设。
3. 16 worker 初始化/收尾，真实空闲 rank 与 EP group 排序，原始层号及路由/consumer 对照。
4. 多个不同长度请求并发、动态进入退出、取消及异常退出；同步拷贝的性能影响。
5. 与独立设备参考数据交叉比较，关闭采集恢复基线功能。

本次未启动 A3/NPU 服务，未验证 TorchNPU、HCCL、Mooncake、真实权重或完整请求。
输出 `a3_unverified_debug` 和 `npu_verified:false` 是刻意保留的限制标识。
`worker_traces_complete` 只验证保存的已声明执行记录，不验证请求终止状态。
启动器信号转发/超时清理逻辑已实现，但没有在真实 A3 服务上测试。
不支持 DSpark/MTP、异步/图模式或四机；不提供 P/proxy 到 D 的全局请求关联和自动上传。
诊断 consumer 文件未纳入分片哈希完整性校验。
