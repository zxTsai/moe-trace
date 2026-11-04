# DeepSeek V4 异步、图执行与 MTP 采集 CPU 验证

日期：2026-11-05。分支：`feature/ascend-deepseek-v4-mtp-async-graph`。
结论：提供可安装到现有 A3 D 容器的 **调试候选版**；参考源码 CPU 接入已验证，
**没有在真实 A3/NPU 或完整 PD 服务执行本版采集，未声明硬件验收通过**。

## 参考版本与执行范围

| 项目 | 基线 |
| --- | --- |
| 目标镜像 | quay.io/ascend/vllm-ascend:v0.23.0.post1-a3 |
| 用户提供包版本 | vllm 0.23.0+empty / vllm_ascend 0.23.0.post1 |
| vLLM 参考 | 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 |
| Ascend 参考 | 1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8 |
| 调试目标 | D DP16/TP1/EP16，async、FULL_DECODE_ONLY、npugraph_ex、共享专家多流、eager MTP K=1 |
| 本地执行器 | 独立 moe-ascend-cpu 环境，Python 3.12、CPU PyTorch |

版本/路径由用户提供；本地参考文件按 18 个关键文件摘要校验。没有读取用户服务器文件，
同名镜像实际构建是否一致，必须由服务器的 inspect/stage 检查确认。

CPU 实际执行了补丁后源码中的 runner forward 接入块、W8A8 dynamic apply、
runner `_sample`、MTP `set_inputs_first_pass`、完整 `_run_merged_draft` K=1/2/3 循环，
以及 `ACLGraphWrapper.__call__` 的 capture/replay 控制流。
真实 CPU 张量用于输入、路由与样本；模型、路由选择/专家算子、采样输出、
NPU 图/流和通信由 stub 提供。图重放测试记录并重放 CPU clone 操作，Python 路由钩子
只执行首次捕获一次，之后改变输入专家 ID 仍能得到新路由，旧执行快照不被覆盖。
**这验证 Python 接入与保存契约，不能证明 CANN 图捕获或 NPU 算子的行为。**

## 已覆盖的检查

- 预热时建立图缓冲，不写真实请求事件；实际重放时才提交执行快照。
- 原模型和 sampler 返回对象保持；关闭采集不做采集拷贝或写文件。
- CPU 请求表无需已更新 token；使用设备实际输入，不把 scheduler 的 `-1` 当真实 token。
- 请求顺序在两轮间倒置，采样仍关联冻结的执行；不同请求分段不串行。
- 主模型 Decode/验证、MTP 输入和候选输出位置分开；MTP 移位和后续输入与输出校验。
- 接受、首个拒绝、后续未采用及未验证状态分开；拒绝计算路由不删。
- 实际 W8A8 hook 的逻辑 ID 与独立 mock kernel 消费参数逐行对照。
- 16 worker 的文件覆盖，包括 15 个空闲 worker；摘要校验、零计数专家、缺失 worker 拒绝。
- token/权重隐私开关、缺层报错、样本不匹配、执行/字节上限、失败状态及缺失验证结果的 incomplete 收尾。
- 原始源码不改；stage 摘要检查和语法检查；启动器原命令参数透传，副本篡改拒绝。

K=1/2/3 的两请求 CPU 场景分别得到：

| K | target 专家选择次数 | MTP 次数 | 合计 | 接受 | 首个拒绝 | 后续未采用 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 4 | 12 | 0 | 2 | 0 |
| 2 | 12 | 8 | 20 | 2 | 2 | 0 |
| 3 | 16 | 12 | 28 | 2 | 2 | 2 |

每个 mock token 选择 2 个专家、每 scope 一个 MoE 层；这些是 CPU 软件测试数值，
不是 DeepSeek 真正层数、真实专家分布或 A3 跑模型结果。
三个场景均覆盖 16 worker、`unresolved_candidate_links=0`、末尾 unverified=0。

## 复验命令和产物

在开发 CPU 环境安装工具与测试依赖，准备上述精确版本源码后：

```bash
python scripts/validation/ascend_023_source_cpu.py \
  --vllm-source /path/to/vllm-v0.23.0 \
  --ascend-source /path/to/vllm-ascend-v0.23.0.post1 \
  --output-root ./outputs/ascend-023-source-cpu
```

产物包含 `cpu-validation.json`、`pytest.log`、实际 patch 副本，以及各测试目录内的
`cpu/D/dp-*/events.jsonl` 与 `status.json`。原始轨迹不提交 Git。
没有提供源码路径时，全仓 pytest 中这些源码依赖用例会跳过；应使用本入口验证，
不能把跳过当通过。

本轮专项 **34 passed，无跳过**；提供全部历史固定源码后的全仓回归
**384 passed，无跳过**。完整 patch 通过参考 Ascend 源码 `git apply --check`。
指南中 10 段 Bash 通过 `bash -n`，示意 JSON 通过解析；全仓文档链接检查通过。
源码副本和原命令包装器的实际 CLI 预检通过，未启动本地或远程模型服务。

本机复验产物：
`/home/czx/code/moe-routing/outputs/ascend-023-source-cpu/ascend-023-source-cpu-xq_hdt6p/`。
其中 `cpu-validation.json` 和 `pytest.log` 记录实际执行情况；
`pytest/test_actual_mtp_first_pass_and2/cpu` 是 K=3 的 16-worker CPU 场景。
此绝对路径仅用于定位本次证据，不是用户安装路径。

## 真机仍需确认

1. 目标容器源码摘要、插件加载路径和 16 个实际 worker。
2. CANN 是否允许在当前 npugraph_ex/fused 路径记录 route clone，重放是否实际更新。
3. 共享专家多流与 device-wide 同步在真机的依赖/生命周期、图 bucket 切换。
4. 真实模型/量化、MTP 接受率、异步取消/进入退出、长序列及反复并发。
5. 经原 Mooncake PD proxy 的真实请求结果与每层路由独立对照。
6. A3 内存余量与采集开销；原配置 0.9 显存利用率不能由 CPU 测试保证仍有足够余量。

无法匹配生产轮次的候选保留 `unresolved`；发生缺层/布局或实际输入不匹配会明确失败，
不会伪造完整性。达到采集上限为部分窗口，需 `--allow-incomplete` 分析。
安装与恢复原服务见[现有容器调试指南](../guides/DeepSeekV4现有A3容器采集调试指南.md)。
