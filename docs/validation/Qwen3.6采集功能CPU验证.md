# Qwen3.6 采集功能 CPU 验证

日期：2026-10-10。分支：`feat/ascend-qwen36-035b-dp8-tp2`。
这是固定源码接入验证，**不是 A3/NPU、完整 vLLM 服务或真实 Qwen 权重验收**。

## 源码与实现

官方 Ascend `v0.23.0.post1` commit：`1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。
六文件指纹匹配后只修改独立副本，原源码测试前后摘要一致。
涉及 runner forward 上下文、W8A8 selector/consumer、worker 初始化和关闭。
生成补丁与仓库补丁字节一致；启动器校验副本 manifest、修改文件及未修改锁定文件。

实际在 CPU 执行：

- 完整的 `PrepareAndFinalizeWithAll2All.prepare`。
- 完整的 patched `AscendW8A8DynamicFusedMoEMethod.apply`。
- 实际 patched runner 中的 forward `with trace_target` 区块，而非完整 execute_model。

模型 forward 回调、selector、专家 kernel、分布式组使用 mock。selector 用 CPU Top-K 构造
模拟推理结果，采集器仅读取它，不重算路由。没有加载 Qwen 权重或运行 TorchNPU。

## 独立期望对照

DP0 两个 TP 处理两个请求，其他 14 个 worker 空闲并收尾。
第一轮 batch 为 B,A，调度字典为 A,B；第二轮换为 A,B。第一轮含 Prefill 和 padding。
MoE 原始层号是 2、3，每个 token 的两个层均执行并保存。

| 执行 | 请求 | 输入位置 | token ID | 每层专家 | owner |
| --- | --- | --- | --- | --- | --- |
| 1 | B | 1 | 21 | `[3,1]` | DP0/TP0 |
| 1 | A | 2 | 12 | `[2,0]` | DP0/TP1 |
| 2 | A | 3 | 13 | `[1,0]` | DP0/TP0 |
| 2 | B | 2 | 22 | `[2,1]` | DP0/TP1 |

共 **8 条 token/layer 路由、16 次专家选择**。kernel 接收的 CPU 引用张量也独立检查，
采集前后 selector 输出保持原值；Prefill 和 padding 未进入输出。

## 执行命令与产物

使用现有独立 `moe-ascend-cpu` 环境（Python 3.12、Torch 2.10 CPU、pytest、插件）：

```bash
conda run --no-capture-output -n moe-ascend-cpu \
  python scripts/validation/qwen36_cpu.py \
  --ascend-source /tmp/moe-trace-ascend-023post1-qwen-source \
  --output-root outputs/qwen36-cpu
```

一次已执行成功的证据目录：
`outputs/qwen36-cpu/qwen36-cpu-zzycokz7/`（被 Git 忽略，不提交大型轨迹）。
包含 `details.json`、`expert-counts.csv`、`cpu-validation.json` 及 pytest 下的 16 worker 原始文件。
报告显示 `cpu_patched_source_hooks`、独立期望比较通过，
`npu_verified=false`、`full_service_verified=false`、`real_model_loaded=false`。

## 自动测试覆盖

- 实际源码接入、不同批次顺序、有效文本三轴 positions、Prompt/Decode/padding 过滤。
- non-SP All2All 切分、DP 不等长 AllGather 的本 DP 行选择与 TP0 单 owner。
- 空闲 worker 收尾、消费者 IDs 变化、缺层、重复 selector、输入 ID 不符、隐私省略。
- 跨 TP 声明不一致、声明哈希篡改、有效 token 缺路由、worker 未收尾和全空输出失败。
- 关闭采集/预热 no-op、多模态 positions 与异步配置拒绝。
- 副本摘要校验与修改副本拒绝，已有环境检查和旧版本回归。

完整回归使用已有三个 Ascend/V4.1 源码环境变量，再增加：

最终全量结果：**295 passed in 7.74s**，无 skip。
补丁 `git apply --check`、启动器 dry-run 和 `git diff --check` 均通过。

```bash
MOE_QWEN_ASCEND_SOURCE=/tmp/moe-trace-ascend-023post1-qwen-source \
MOE_V41_VLLM_SOURCE=/tmp/moe-trace-vllm-v41-source \
MOE_V41_ASCEND_SOURCE=/tmp/moe-trace-ascend-v41-source \
MOE_ASCEND_SOURCE_OVERLAY=/tmp/moe-ascend-023-1zggohdj \
conda run --no-capture-output -n moe-ascend-cpu python -m pytest -o addopts='' -q
```

## 仍需实机验收

实际 editable vLLM HEAD、完整启动命令与架构需现场核对；版本字符串不是二进制兼容保证。
真实 MoE 模块发现、EP rank 排序、量化 kernel、同步 D2H、混合注意力状态处理、
NPU 服务启动和 Ctrl+C 全 worker 收尾，以及请求并发/取消/设备故障仍需 A3 实测。
不支持图/异步/前缀缓存/多模态/MTP/FlashComm/SP/多流/EPLB；没有请求终止生命周期验收。
不得用此报告声称官方优化配置或所有 Qwen 模型已经可采集。
