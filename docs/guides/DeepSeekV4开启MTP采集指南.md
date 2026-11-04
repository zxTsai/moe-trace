# DeepSeek V4：开启 MTP 的两机 A3 采集

分支：`feature/ascend-deepseek-v4-mtp`。日期：2026-11-04。
基于不带 MTP 的 V4 版本新建，原版本保留。**CPU 源码接入已验证，A3/NPU 待验收。**

## 1. 适用配置

- P 节点 DP4/TP4、D 节点 DP16/TP1，每节点 16 die，EP 和 MooncakeHybridConnector。
- 文本 DeepseekV4ForCausalLM，Ascend W8A8，单个 MTP 层。
- MTP `num_speculative_tokens=K`，按实际执行采集；两端同步 eager，D 关闭图、异步、FlashComm、多流和 EPLB。
- 仅 D 采集；不支持 DSpark、图/异步模式、PP/CP 或多模态。

本版增加 MTP，不恢复原生产启动命令中的 `--async-scheduling`、FULL_DECODE_ONLY
和 shared-expert 多流。需要重启 D 加载独立源码副本；不能附加到已运行的服务。
P 使用本页带 MTP 的调试启动脚本，先验证关闭采集的 PD 基线。

参考提交仍为 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、Ascend
`0e181fc85fff24db10e5f92f10319c1913e6e9c5`，额外锁定 MTP/proposer/sampler 源码。
真实服务器提交尚未提供，源码检查不通过时停止，不跳过摘要校验或直接升级原环境。

## 2. 从获取代码开始

在已装好 vLLM Ascend 的服务容器内，使用同一个 Python：

```bash
git clone --branch feature/ascend-deepseek-v4-mtp https://github.com/zxTsai/moe-trace.git
cd moe-trace
python -m pip install --no-build-isolation --no-deps -e .
```

若当前分支尚未推送，请先使用维护者提供的 `moe-trace-deepseek-v4-mtp.tar.gz`，
不能拿 main、V4 非 MTP 或 V4.1 版本代替：

```bash
mkdir -p /data/moe-trace-v4-mtp
tar -xzf /YOUR_DOWNLOAD/moe-trace-deepseek-v4-mtp.tar.gz -C /data/moe-trace-v4-mtp
cd /data/moe-trace-v4-mtp/moe-trace
python -m pip install --no-build-isolation --no-deps -e .
```

不安装 CPU PyTorch、不替换 vLLM、不使用 NVIDIA 环境向导。容器挂载及输出读写检查
可参考[非 MTP 版指南](DeepSeekV4两机A3快速开始.md)，下面使用独立 MTP 输出目录。

## 3. 环境检查和采集副本

D 容器：

```bash
export MODEL=/data/c00835156/model/Eco-Tech--DeepSeek-V4-Flash-w8a8-mtp/snapshots/master
export TRACE_ROOT=/trace-output/deepseek-v4-mtp
export MTP_K=3  # 设为模型/框架允许的正整数；两节点分别设置
mkdir -p "$TRACE_ROOT"
python scripts/tools/inspect_ascend_v4_mtp.py --model "$MODEL" \
  > "$TRACE_ROOT/environment-check.json"
cat "$TRACE_ROOT/environment-check.json"
```

必须 `source_check_passed:true`，退出码 0。自动查找当前 Python 的真实包位置，
不假定一定在 `/vllm-workspace`；报告含版本、Git 提交和文件差异。

```bash
python scripts/tools/stage_ascend_v4_mtp.py --check
python scripts/tools/stage_ascend_v4_mtp.py --output-parent "$TRACE_ROOT/overlays"
export OVERLAY=/trace-output/deepseek-v4-mtp/overlays/ascend-v4-mtp-XXXXXXXX
```

OVERLAY 填打印的实际目录。副本包括 runner、selector、W8A8、worker、proposer 的补丁，
原安装包保持不变。不要与非 MTP/Qwen/V4.1 的 overlay 或 MOE_TRACE 配置混用。

## 4. 先拉起不采集的 MTP 基线

保存原脚本，按你的运维流程停止原服务，确认设备和端口可用。两端各定义自己的 MODEL。
以下默认仅打印命令；检查后分别加 `--execute` 启动。

```bash
# P：IP、网卡和模型路径改为实际值
python scripts/deployment/ascend_v4_mtp_1p1d.py \
  --role P --model "$MODEL" --local-ip 10.44.231.172 --nic enp196s0f0 --num-speculative-tokens "$MTP_K"

# D：在 D 容器运行
python scripts/deployment/ascend_v4_mtp_1p1d.py \
  --role D --model "$MODEL" --local-ip 10.44.231.202 --nic enp196s0f0 --num-speculative-tokens "$MTP_K"
```

启动器显式添加（以 MTP_K=3 为例，实际使用所设 K）：

```text
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
--enforce-eager --no-async-scheduling
--compilation-config '{"cudagraph_mode":"NONE"}'
```

P 的 API 端口默认 7100–7103，D 7100–7115；DP RPC 12321，设备 0–15。
PD proxy 的端点配置与非 MTP 指南一致。请求必须经过 proxy，不能直接请求 D 代替 PD。
先正常返回文本，再在 D 启动器终端 Ctrl+C 并等待退出。P 可继续运行。
关闭采集时启动器不加载采集副本，默认不产生路由数据。

## 5. 开启 MTP 采集

```bash
python scripts/deployment/ascend_v4_mtp_1p1d.py \
  --role D --model "$MODEL" --local-ip 10.44.231.202 --nic enp196s0f0 --num-speculative-tokens "$MTP_K" \
  --trace --overlay "$OVERLAY" --output-root "$TRACE_ROOT/runs" \
  --node-id D0 --execute
```

记下打印的本轮 UUID。等待 D ready，经代理发送多个请求：

```bash
python scripts/validation/deepseek_v4_requests.py \
  --url http://YOUR_PROXY_IP:1999/v1/chat/completions --model dsv4
```

需要保存 token ID/权重时，D 启动增加 `--save-token-ids --save-expert-weights`；默认关闭。
请求完成后正常停止 D，等待全部 16 worker 收尾，再汇总：

```bash
python -m moe_trace.distributed.v4_mtp_cli "$TRACE_ROOT/runs/RUN_UUID" \
  --output "$TRACE_ROOT/details-RUN_UUID.json" \
  --counts "$TRACE_ROOT/expert-counts-RUN_UUID.csv" \
  --by-execution "$TRACE_ROOT/execution-counts-RUN_UUID.csv"
```

输出须在原始 run 外且不存在。关闭采集后，使用原脚本和干净终端可恢复原服务；
不继承采集 overlay 的 PYTHONPATH 和 MOE_TRACE 环境变量。

## 6. 三类路由与候选的关系

| 字段 | 语义 |
| --- | --- |
| `row.kind=target_decode` | 无候选时主模型的普通 Decode 输入 |
| `row.kind=target_verify`，无 candidate ID | 主模型验证调用中的 seed 输入；隐私开启时 input_role 为 redacted |
| `row.kind=target_verify`，有 candidate ID | 实际被主模型验证的候选输入，关联接受/拒绝 |
| `row.kind=mtp_draft` | MTP 模型实际计算的输入，可含多行上下文；不是该次预测出的 token |
| `model_scope` | `target` 或 `mtp`，同号层不会混合 |
| `source_engine, execution, input_row` | D 引擎、真实执行、实际输入行；不同 DP 的 execution 不代表同一全局时间 |
| `candidate_round, candidate_node` | 候选轮次和节点；每轮每请求 node=0..实际K-1 |
| `candidate_accepted` | 候选验证路由的 sampler 决定；false 的路由仍完整保留 |

MTP 的实际 input_ids 比 target 左移一位，但送入 kernel 的 positions 沿用原 target
坐标。**MTP 路由的 position 保存实际 kernel 坐标**；其输入 token 的序列位置为
position+1，预测候选位置为被采样输入 position+2。输入声明和 candidates 明确记录此关系。
不要把 MTP 输入 token ID 和预测输出 token ID 当成同一个值。

JSON 的 `candidates` 单独保存 request/source_engine、round/node、producer_execution、
producer_input_row、producer_kernel_position、候选位置、可选 token ID、验证 execution 和状态。
状态为 accepted/rejected/not_adopted/unverified；未被调度验证的末轮候选、取消后的候选不被误标为拒绝。
同 token ID 在不同轮次出现时仍按候选轮次和位置关联。

决定直接来自 rejection sampler 的实际输出及 SpecDecodeMetadata，**不是根据客户端输出
长度猜测**。实际有效 sampler 输出数减一是接受前缀长度，还会逐个核对前缀 token ID。
首个未接受候选标 rejected，其后标 not_adopted（后续未采用），已经发生的路由都保留。
`candidate_accepted:false` 表示未接受，细分原因查看 `candidate_decision`。它不等于经过 EOS/stop/长度裁剪后最终返回客户端的状态。

## 7. 负载统计怎么看

JSON 的 `expert_choices` 给出：

```json
{"target":24,"mtp":10,"combined":34}
```

上面是原 K=1 CPU mock 样例：17 条 token/层路由，主模型 24 次选择、MTP 10 次，合计 34 次；
1 个候选接受、1 个拒绝、2 个末轮未验证。被拒绝候选仍贡献主模型的 4 次专家选择。
它不是 DeepSeek 真模型的负载结论，真实数字取决于请求与执行。

专家 CSV 按 scope、执行类型、层、决定、专家计数，包含零负载专家；按执行 CSV 保留
引擎和执行轮次。合计是两模型执行选择次数相加，不把两模型第 0 层合并成同一个层，
也不推断物理设备利用率、共享专家耗时或 KV 流量。

原始目录每 worker 包含：

```text
dp-N-tp-0/
├── trace/                 # target/mtp 声明、路由和 CandidateResult 分片、SHA256、状态
├── inputs.jsonl           # 实际输入声明，带 scope/kind/候选关联
├── inputs-index.json
├── candidates.jsonl       # 真实预测输出与 producer 输入关联
└── candidates-index.json
```

汇总检查全部 16 worker、身份、哈希、每输入×层、候选生产→验证→结果。
已验证候选缺结果会失败；未验证产物导出 accepted=null/state=unverified。
`worker_traces_complete:true` 仅表示保存记录满足声明的完整性约束。
`npu_verified:false` 保留，真实设备路由正确性和完整 PD 服务仍待现场验收。
没有 P/proxy 全局请求映射或请求完成/取消原因采集。

K=3 CPU 样例另得到 26 条路由，target=40、mtp=12、combined=52；
两个请求各接受第 0 个候选，第 1 个拒绝，第 2 个未采用。单个物理 MTP 层实际执行三次，
每次独立执行 ID。不是按 K 虚增计数，而是读取三次执行的真实返回。

## 8. 常见失败与 CPU 复验

- 源码不匹配：保留 MTP environment-check.json，提供真实提交适配，不要强行修改摘要。
- 候选与 producer 不一致：保留 D 日志、原始 run、请求与调度配置；不按 token ID 强行凑关联。
- MTP query/sample index 不匹配：当前源码布局不符合首版假设，检查是否有图、异步、CP、额外 slot 或不同提交。
- 验证结果缺失/异常退出：不会补一个“拒绝”来凑完整；保留失败数据排查。
- 开销：同步拷贝和写盘用于调测，不据此比较原 MTP 服务吞吐；短时、小请求先行。

CPU 开发环境中可重复执行：

```bash
python scripts/validation/deepseek_v4_mtp_cpu.py \
  --vllm-source /YOUR_REFERENCE/vllm \
  --ascend-source /YOUR_REFERENCE/vllm-ascend \
  --output-root outputs/v4-mtp-cpu --k 3
```

不要在 A3 服务环境安装 CPU PyTorch。具体方法、mock 边界与结果见
[MTP CPU 验证记录](../validation/DeepSeekV4的MTP采集CPU验证.md)。
