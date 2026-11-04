# DeepSeek V4 的 MTP 采集 CPU 验证

日期：2026-11-04。分支：`feature/ascend-deepseek-v4-mtp`。
功能检查点：`37afd67`。基于非 MTP V4 `9a527d5`，原版本保留。

## 动态 K 扩展验证

动态 K 功能检查点：`aad5a0e`。不再写死 K=1；启动器支持 `--num-speculative-tokens K`，
CPU 复验支持 `--k K`。从真实补丁源码提取并执行完整 autoregressive for 循环，
覆盖 K=1/2/3 的每种接受前缀、K=7 的部分接受、K=0 不预测、裁短验证和不同请求接受长度。
额外任意正整数 K 可通过 CPU 脚本的 `--k` 参数复验，不安装/升级服务依赖。

最新动态 K 测试新增 **18 项**，MTP 相关合计 **35 项通过**；全仓库回归
**350 passed in 21.60s，无跳过**。补丁应用检查、K=3 启动 dry-run、源码摘要和
`git diff --check` 通过。全部是 CPU/软件接入验证，不是 A3 或完整 PD 验收。

K=3 的本地持久化证据：`outputs/v4-mtp-cpu/deepseek-v4-mtp-cpu-l5wnawd3/`。
两个请求各提出 3 个候选、各接受第一个，第二个拒绝，第三个未采用；
共 26 条路由、target=40、mtp=12、combined=52 次专家选择。
candidate_node 为 0/1/2，三个预测步骤均有独立执行和 producer；后缀路由仍保留。
输出包含 accepted/rejected/not_adopted/unverified 四种状态，CSV 不把后缀说成独立拒绝。

动态测试完整执行真实 Python for 循环和 buffer/position 更新，模型/路由/采样输出仍 mock。
缺少后续 step、前后 token/输出矩阵不一致、位置钳制到 dummy 坐标均拒绝。
一个物理 MTP 层重复执行 K 次；多个物理 MTP 层、图、异步、CP 等边界没有因此开放。

```bash
conda run --no-capture-output -n moe-ascend-cpu \
  python scripts/validation/deepseek_v4_mtp_cpu.py \
    --vllm-source /tmp/moe-trace-vllm-v41-source \
    --ascend-source /tmp/moe-trace-ascend-v41-source \
    --output-root outputs/v4-mtp-cpu --k 3
```

## 结论与证据范围

动态 K MTP 的真实 Python 接入、三类轨迹、候选生产/接受前缀关联、16-worker
汇总和分 scope 负载导出已在 CPU 执行。未加载真实模型、torch_npu、HCCL、Mooncake，
未运行两台 A3 PD 服务。服务器实际提交仍待确认；不能据此宣称真实硬件验收。

初始 K=1 开发检查点曾有 **17 项通过**、完整仓库 **332 passed in 14.76s，无跳过**。
这些保留为初始历史证据，不是动态 K 的最新测试数量。
补丁 `git apply --check`、固定源码 inventory、P/D dry-run 和 `git diff --check` 通过。

固定源码与非 MTP 参考提交相同，额外核对 MTP 模型、proposer、sampler 和 metadata。
执行的方法/块：

- 补丁后的 runner forward 上下文块。
- All2All.prepare、AscendRoutedExperts._select_experts、forward_impl、W8A8 apply。
- proposer 真实 set_inputs_first_pass：shift 输入 ID、保持 kernel position、设置 sample index。
- 补丁后 MTP 首次 forward 上下文块、真实返回表达式。
- runner 完整 `_sample` Python 方法，rejection sampler 返回后执行结果 hook。
- 源码 AST 确认真实 `_propose` 方法有 wrapper 装饰器、worker 导入的生命周期函数正确。

真实 `_propose` 的整个执行流程、attention/KV 和真实 rejection 算法没有在 CPU 运行。
模型 forward 回调、router/NPU expert kernel、sampler 返回、distributed group 使用 mock。
测试以真实 Python 接入位置读取 mock 算子结果，未在采集器重算路由/采样。

## 可核对场景

DP0 活跃，另外 15 worker 空闲初始化和收尾。target 两层 0/1，mtp 单层 0，Top-K=2。
所有 mock selector 的逻辑专家 `[0,2]`，静态 log2phy 后实际 consumer `[1,0]`；逐调用核对。

| execution | scope | 实际输入 | 保存路由数 | 专家选择次数 |
| --- | --- | --- | --- | --- |
| 1 | target_decode | B:21@1、A:12@2 | 4 | 8 |
| 2 | mtp_draft | B:22@kernel1、A:13@kernel2 | 2 | 4 |
| 3 | target_verify | A:13@3、候选14@4；B:22@2、候选23@3 | 8 | 16 |
| 4 | mtp_draft | A:15@kernel3；B:23@kernel2、24@kernel3 | 3 | 6 |

请求顺序 B,A→A,B，scheduler 字典故意反序。执行 2 产出 B 候选23@3、A 候选14@4。
真实 `_sample` 接入读取 mock sampler：A `[15,-1]` 表示拒绝14，B `[23,24]` 表示接受23。
执行 3 被拒绝的 A 候选两层路由仍保存，贡献 4 次专家选择，不按最终输出删除。
执行 4 复用相同候选 ID A14/B23，但新的 round 和位置为5，未再验证，明确为 unverified。
拒绝后 A 的无效 draft padding 行被 sample index 排除；MTP 全局 padding 不计入有效路由。

合计：**17 条路由，target 24、mtp 10、combined 34 次专家选择**。
4 个产物中 1 accepted、1 rejected、2 unverified；JSON 和两个 CSV 重算一致。
隐私关闭 token/权重后同样通过，seed/candidate/draft 与候选 ledger 均不泄露 token ID。
这些是 mock 数据，不代表 DeepSeek 真实专家偏好。

## 异常和完整性

其他预测方法拒绝；launcher 读取 K，保持同步 eager 和两机拓扑。
错误候选 token、错误 sampler draft ID/接受前缀、sample index 跨请求均标 worker failed。
缺预测输出 hook、缺验证结果、重复 sampler 决策拒绝；不会把缺结果当拒绝。
覆盖带/不带候选混合 batch，关闭采集和 dummy 不触发采集。
候选 ledger hash、伪造请求关联/位置（即使重算 ledger hash）、缺 worker 索引拒绝。
CSV 包含零负载专家，拒绝候选的计数仍存在。

## 复现

在已有 CPU 环境：

```bash
MOE_V4_VLLM_SOURCE=/tmp/moe-trace-vllm-v41-source \
MOE_V4_ASCEND_SOURCE=/tmp/moe-trace-ascend-v41-source \
conda run --no-capture-output -n moe-ascend-cpu \
  python -m pytest -o addopts='' -q tests/test_v4_mtp.py

conda run --no-capture-output -n moe-ascend-cpu \
  python scripts/validation/deepseek_v4_mtp_cpu.py \
    --vllm-source /tmp/moe-trace-vllm-v41-source \
    --ascend-source /tmp/moe-trace-ascend-v41-source \
    --output-root outputs/v4-mtp-cpu
```

全量回归额外设置旧适配的 `MOE_QWEN_ASCEND_SOURCE`、`MOE_V41_VLLM_SOURCE`、
`MOE_V41_ASCEND_SOURCE`、`MOE_ASCEND_SOURCE_OVERLAY`，再执行 `python -m pytest -o addopts='' -q`。
参考框架源码与 CPU 环境不随插件源码包分发。

本地持久化证据：`outputs/v4-mtp-cpu/deepseek-v4-mtp-cpu-_bsjg6x8/`，
包含 details.json、expert-counts.csv、execution-counts.csv、cpu-validation.json
及 16-worker 原始文件。输出忽略于 Git；脚本可重建。
始终标记 npu_verified=false、full_service_verified=false、real_model_loaded=false。

真机前先按[MTP 使用指南](../guides/DeepSeekV4开启MTP采集指南.md)运行源码检查和无采集
基线。尚待：真实 MTP 权重/模型层发现、PD KV 共享、sample/query 行布局、请求动态进入/
退出、NPU 独立路由对照、性能影响、异常与恢复。没有 P/proxy 全局 ID 及客户端最终接受状态。
