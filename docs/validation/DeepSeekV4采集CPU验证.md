# DeepSeek V4 采集功能 CPU 验证

日期：2026-11-04。分支：`feature/ascend-deepseek-v4-dp16-tp1`。
设计：[两机采集适配](../design/DeepSeekV4两机采集适配设计.md)。
使用：[两机 A3 快速开始](../guides/DeepSeekV4两机A3快速开始.md)。

## 结论和范围

V4 专用 CPU 测试 **20 项通过，无跳过**。已执行固定官方源码中真实 Python 方法的
采集接入、D16/TP1 行映射、存储、16-worker 完整性校验和 JSON/CSV 负载导出。
未运行真实 V4 权重、NPU 算子、HCCL、Mooncake 或完整 PD 服务。
适合进入源码检查通过后的 A3 调测，不能据此宣称服务器已经兼容或采集正确性已验收。

功能及测试检查点：`e57bdd3`。完整仓库回归 **315 passed in 9.59s，无跳过**；
包括既有 NVIDIA/Ascend CPU 测试，不代表 315 项都是 V4 测试。
`git diff --check`、固定源码 `git apply --check`、source check、P/D dry-run 均通过。

本地环境 `moe-ascend-cpu`：Python 3.12、CPU PyTorch、pytest 和 editable moe-trace。
参考源码为 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、Ascend
`0e181fc85fff24db10e5f92f10319c1913e6e9c5`；用户服务器实际提交仍待确认。

## 实际执行与 mock 边界

CPU 从生成的独立副本通过 AST 提取并执行：

- runner 中新增 `with trace_target(...)` 的完整 forward 接入块。
- `PrepareAndFinalizeWithAll2All.prepare` 和 `pad_and_split_input_ids`。
- `PrepareAndFinalizeWithMC2.prepare`、`pad_and_split_input_ids` 和真实 padding helper；核对 TP1 padding 排除。
- `AscendFusedTopKRouter._compute_routing`，分别经过 hash 与普通 sqrtsoftplus 路由分支。
- `AscendRoutedExperts._select_experts`、`forward_impl`。
- `AscendW8A8DynamicFusedMoEMethod.apply`，实际调用 consumer hook。

模型 forward 回调、NPU gating/expert kernel、分布式 group 使用 mock；不是运行整个
execute_model、真实 torch_npu 包或完整推理。采集器不执行自己的 Top-K，读取模型
Python 路径调用 mock NPU kernel 后的实际返回。核对 consumer 使用静态 log2phy
映射后的值，保存统计仍为映射前的逻辑专家。

## 可核对样例

DP3 活跃，其他 15 个 worker 正常初始化且空闲收尾。模型两层 0/1，Top-K=2。
首轮输入实际为 B,A，而 scheduler 字典为 A,B；A 的 prompt 行被排除。

| 执行 | 请求 | 输入位置 | token ID | 第 0 层 hash 专家 | 第 1 层普通专家 |
| --- | --- | --- | --- | --- | --- |
| execution-1 | B | 1 | 21 | `[2,0]` | `[1,3]` |
| execution-1 | A | 2 | 12 | `[1,3]` | `[1,3]` |
| execution-2 | A | 3 | 13 | `[2,0]` | `[1,3]` |
| execution-2 | B | 2 | 22 | `[3,1]` | `[1,3]` |

共 **8 条 token/层路由，16 次专家选择**；逐条与独立期望比较，consumer 输入也逐条
核对。第 0 层 4 位专家各选择 2 次；第 1 层计数 `[0,4,0,4]`，包含零负载专家，
最大/均值=2、变异系数=1。数据来自 mock kernel，不能代表真实模型的专家偏好。

## 已覆盖的拒绝与保护

MTP、V2 runner、异步、图、V4.1 架构、DP/TP 不匹配、多流拒绝；未知布局沿用
公共 selector_rows 的严格检查。验证 TP1 AllGather DP 不等长，仅保存本地有效行。
覆盖丢层、重复层、selector 丢行、consumer 被修改、真实 token 对应错误、隐私开关。
覆盖实际 group rank 检查、关闭采集无动作、来源包不匹配和副本被修改时拒绝。
覆盖缺 worker、未收尾、输入 hash 被修改、输入有声明而路由缺失、全空轨迹拒绝。
worker hook 导入名称由生成源码 AST 对照 adapter 检查，避免仅测试 forward 而遗漏启动 hook。

## 复现和证据

在仓库根目录：

```bash
MOE_V4_VLLM_SOURCE=/tmp/moe-trace-vllm-v41-source \
MOE_V4_ASCEND_SOURCE=/tmp/moe-trace-ascend-v41-source \
conda run --no-capture-output -n moe-ascend-cpu \
  python -m pytest -o addopts='' -q tests/test_v4_debug.py

conda run --no-capture-output -n moe-ascend-cpu \
  python scripts/validation/deepseek_v4_cpu.py \
    --vllm-source /tmp/moe-trace-vllm-v41-source \
    --ascend-source /tmp/moe-trace-ascend-v41-source \
    --output-root outputs/v4-cpu
```

完整仓库回归另设置 `MOE_QWEN_ASCEND_SOURCE`、`MOE_V41_VLLM_SOURCE`、
`MOE_V41_ASCEND_SOURCE`、`MOE_ASCEND_SOURCE_OVERLAY` 指向对应旧适配的固定源码，
再执行 `python -m pytest -o addopts='' -q`；这些参考源码不随插件源码包分发。

已生成本地证据：`outputs/v4-cpu/deepseek-v4-cpu-7kti3aqj/`，包含
`cpu-validation.json`、`details.json`、`expert-counts.csv`、`execution-counts.csv`
和 pytest 目录中的 16-worker 原始数据。输出忽略于 Git，可通过脚本重新生成。
`npu_verified:false`、`real_model_loaded:false`、`full_service_verified:false`。

## 仍待 A3 验收

真实服务源码匹配；关闭 MTP/图/异步后的不采集基线；实际模型 MoE 层发现及 hash 层；
16 die EP rank 排序；空闲 rank 生命周期；Mooncake PD 请求；多个请求动态进入/退出；
与设备独立参考路由交叉比较；正常/异常停止；恢复原部署。
没有 P/proxy 全局请求关联、请求终止状态或 MTP 候选接受/拒绝轨迹。
