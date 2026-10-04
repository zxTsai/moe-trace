# moe-trace v1 设计

状态：实现基线（2026-10-02）  
目标适配器：vLLM `0.10.1`，基础 commit
`aab549870df50edf0512f0a59b574f692f546465`，V1 引擎，GraniteMoe，
单 GPU，eager，关闭前缀缓存和投机解码。

## 1. 约束与基线结论

`moe-trace` 是独立 Python 包。vLLM 仅增加两个很小的 Python 调用点：模型
执行前后建立/关闭执行上下文，以及 `FusedMoE` 已经选出并即将消费 Top-K
结果的位置上报结果。公共存储、校验和分析代码不依赖 vLLM。

当前模型配置为 24 个原始模型层、每层 32 个逻辑专家、Top-K 为 8；
`GraniteMoeForCausalLM` 的每个模型层均包含 `block_sparse_moe`，没有共享专家。
这些值在启动时仍从模型配置和实际注册的层读取，不写死到采集代码。

不支持的模式必须在启动时拒绝：非 V1、非单 rank、CUDA graph、投机解码、
前缀缓存、非 GraniteMoe、路由模拟、EPLB，以及无法识别的重算。关闭采集时
补丁调用为常量级空操作，不复制张量、不创建输出目录。

## 2. 采集位置与真实路由

### 2.1 输入行映射

`GPUModelRunner._prepare_inputs` 是实际 GPU 输入排列的权威来源。它按
`self.input_batch.req_ids` 顺序取每个请求本轮的 `num_scheduled_tokens`，请求内
连续排列。对请求索引 `i`：

- 行区间为累计调度 token 数 `[offset[i], offset[i+1])`；
- 位置为 `num_computed_tokens_cpu[i] + arange(num_scheduled_tokens[i])`；
- token ID 是 runner 已填入的 `input_ids_cpu` 对应行；
- `position < len(prompt_token_ids)` 为 `prefill`，否则为 `decode`。

适配补丁在 `_prepare_inputs` 完成后复制这些 CPU 元数据，绝不从调度字典迭代
顺序猜测行序。有效行数严格使用 `total_num_scheduled_tokens`，因此排除 CUDA
graph/DP padding（首版仍拒绝这些模式）。profiling/warmup 不经过正常的带请求
执行上下文；若路由回调在上下文外发生则忽略。

### 2.2 路由结果

Granite 的 `GraniteMoeMoE` 调用 `FusedMoE`。适配点位于
`FusedMoEMethodBase.forward_cuda` 中 `FusedMoE.select_experts` 返回之后、专家
kernel 消费之前。保存的正是该次执行的 `topk_ids`，不重新计算 softmax/topk。
在首版禁用 EPLB/EP 时，这些 global physical ID 与 global logical ID 等价。
权重保存的是 `select_experts` 返回、传给专家 kernel 的已归一化权重。

层号从 `FusedMoE.layer_name`（例如
`model.layers.7.block_sparse_moe.experts`）解析原始模型层编号。每次执行要求配置
声明的所有 MoE 层各上报一次、行数相同、Top-K 相同且专家 ID 在范围内。

## 3. 执行、调度与进程关联

实现更新：补丁 0003 在 scheduler 的 add_request/_free_request 记录生命周期。
当前明确限制为 uni executor，其 scheduler 与 runner 在同一 EngineCore 进程，
其他 executor 被拒绝。新 metadata 声明 request_lifecycle_required，终止事件
缺失将导致收尾失败。详见 lifecycle-validation.md。

worker 进程内维护一个 `contextvars.ContextVar`，其作用域只覆盖一次
`GPUModelRunner.execute_model` 的真实 model forward。adapter 为每次调用分配
单调递增的 `execution_id`；首版一次 scheduler 输出对应一次 forward，因而
`step_id` 同步递增，但两者作为独立字段永久保存，不把相等作为读取假设。

执行开始时 adapter 收到完整的行映射；逐层回调通过当前上下文关联到相同
execution。执行成功返回后提交该 execution；异常时记录错误并将运行标为失败。
API 进程与 worker 不共享全局变量：首版写入完全发生在持有实际输入与路由张量
的 worker。客户端标签到内部请求 ID 的映射预留为显式请求事件接口；没有经过
验证的 API ID 不自动视为内部 ID。

请求开始事件在某个内部 request ID 首次出现在执行行中时产生。请求结束/取消
需要 scheduler 生命周期补丁提供明确原因；若缺少该事件，正常收尾校验失败，
不会猜测结束状态。

## 4. 物理布局与关联规则

schema 版本为 `1.0.0`。运行目录使用独占创建的 UUID `run_id`，不覆盖旧目录。

### `metadata.json`

包含 schema/run/engine/rank、模型与框架版本、vLLM 基础和工作树 commit、有效
配置、原始 MoE 层列表、逐层专家数、Top-K，以及权重语义。

### `steps.jsonl`

每个 execution 一行，至少包含 `run_id, engine_id, rank_id, step_id,
execution_id, row_count` 和 `rows`。每个 row 显式保存 `row_id, request_id,
token_position, phase`，可选 `token_id`。`row_id` 是 execution 内从 0 开始的
实际输入行号。

### `routes-NNNNNN.npz`

每个已提交分片包含：

| key | 维度 | dtype | 含义 |
|---|---:|---|---|
| `execution_ids` | `[R]` | `<U64` | 每条路由行所属执行 |
| `row_ids` | `[R]` | `int32` | execution 内输入行 |
| `layer_ids` | `[R]` | `int32` | 原始模型层号 |
| `expert_ids` | `[R,K]` | `int32` | 实际逻辑专家 ID |
| `expert_weights` | `[R,K]` | `float32` | 可选，实际消费权重 |

`R` 是“执行输入行 × 已采集层”的记录数。关联键是
`(execution_id, row_id, layer_id)`，不依赖文件顺序。step/request/position/phase
经 `(execution_id,row_id)` 与 `steps.jsonl` 关联。token ID 保存在 step row，避免
逐层重复。

分片先写同目录 `.pending` 文件，flush/fsync 后原子 rename 为 `.npz`，再把 manifest
条目追加到状态。`.pending` 是未提交分片；状态引用但文件缺失、未引用的正式
分片、NPZ 无法读取均为完整性错误。默认每 4096 条逻辑记录分片，内存缓冲硬上限
8192；达到上限同步排空，因此不丢数据且不会无限持有张量。

### 其他文件

- `requests.jsonl`：`started/finished/cancelled` 事件、内部 ID、可选客户端标签和原因；
- `status.json`：`running/complete/failed`、计数、已提交分片及错误；
- `expert_counts.csv`：离线生成，含每个 phase/layer/expert（包括零负载）计数。

## 5. 状态机、错误与收尾

新运行先写 `status=running`。每个 execution 先完整校验再进入有界缓冲；任何长度
不匹配、层缺失/重复、专家越界、Top-K 不符、写失败都会记录错误并转为 failed，
错误继续抛回推理调用。运行失败后拒绝继续提交。

正常 `close()`：提交剩余缓冲，校验所有引用和请求生命周期，生成统计，原子写入
`status=complete`。只有该路径且校验无错误才完整。进程崩溃会留下 running 状态或
`.pending`，读取器报告不完整。取消请求允许其已有 execution 完整，同时请求事件
为 cancelled；“运行完整”与“所有请求正常完成”分开报告。

## 6. 配置和过滤

YAML 顶层分为 `trace` 与 `vllm`。严格拒绝未知键。`trace.enabled` 默认 false；
输出根目录、阶段、层、标签过滤、token/权重开关、strict、buffer/shard 限额均经
类型与范围校验。过滤在 execution 行映射构造时完成；adapter 同时切片每层路由
数组，完整性仅针对 metadata 中记录的有效采集范围。首版标签过滤要求显式标签
映射；映射缺失时不把 ID 当标签。

## 7. 验证策略

纯 Python 单元/集成测试覆盖配置拒绝、单/并发/动态批、分块 prefill 的行映射、
分片往返、零负载计数、筛选、重复/缺层/越界、写失败、pending/missing、取消及
关闭模式。模拟路由只验证存储层，不作为真实路由验收。

真实 GPU 验收脚本使用指定模型、`--enforce-eager`、禁用 prefix cache/spec decode，
分别运行 A01-A14。A05 在路由消费调用点临时保留同一 Top-K CPU 快照并与 NPZ
逐元素核对；A03 保存单个 execution 中多个 request ID 的证据；A04 保存同一请求
连续多个 prefill execution 及无缝位置区间。报告逐项标记 PASS/FAIL/NOT RUN/N/A，
绝不由输出 token 数推导 decode 次数。

## 8. 版本适配边界

所有 vLLM 导入与调用集中在 `moe_trace.adapters.vllm_0_10_1`。补丁带基础 commit
检查、应用/撤销说明。未来 vLLM/model/backend 需要新 adapter 与独立真实性验收；
公共 reader/schema/analyzer 保持不变。首版不修改或编译 C++/CUDA。
