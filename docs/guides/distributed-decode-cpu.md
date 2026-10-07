# DeepSeek V4.1 分布式 Decode CPU 验证（开发原型）

本指南验证多 worker 事件契约、分片与汇总，不启动模型服务，不需要 NPU。
尚无 V4.1 安装 patch；不能将 0.23.0 Granite patch 用于 PD/DSpark。
完整设计见[设计文档](../design/ascend-deepseek-v41-decode-design.md)。

## 运行

在已有独立开发环境安装本仓库（不改生产环境、不安装 vLLM/CANN）：

```bash
python -m pip install -e '.[test]'
python -m moe_trace.validation.distributed_decode_cpu --output ./outputs/distributed-decode-cpu
```

报告应包含 state=passed、comparison=passed、workers=4、observed_routes=20、
logical_routes=10、expert_choices=20，且 **npu_verified=false**。
fixture 模拟 DP2/TP2 的数据身份，不是真实部署，非文档 DP8/TP2 的硬件证明。

从 `cpu-validation.json` 读取本次 run_dir，再执行（不要照抄历史 UUID）：

```bash
RUN_DIR=./outputs/distributed-decode-cpu/runs/本次UUID
python -m moe_trace.distributed.cli validate "$RUN_DIR"
python -m moe_trace.distributed.cli merge "$RUN_DIR" ./outputs/distributed-decode-cpu/merged.json
python -m moe_trace.distributed.cli counts "$RUN_DIR" ./outputs/distributed-decode-cpu/expert-counts.csv
```

新 CLI 与现有 `moe-trace validate/details` 不通用：它使用独立实验 schema。
每次 fixture 保存独立 run；报告、details.json、consumer-reference.json 覆盖更新。
merge 成功时原子覆盖指定 JSON，不在原始 run 内写分析文件。

## 保存什么

- `manifest.json`：预期 workers、各 model_scope 的 MoE 层、CPU 验证等级。
- `worker-00000/` 等：每个 worker 的 events 分片和 status；名称由 manifest 顺序决定。
- execution 事件：owner、有效行与预期 observer 集合，检出整行/副本漏采。
- route 事件：observer/owner、engine/execution、内部/全局请求 ID、位置、输入角色、
  目标/draft 类别、候选身份、层和逻辑专家 ID（可选权重）。
- candidate_result 事件：候选接受/拒绝；由唯一发送方记录，不在 TP 副本重复写。

该原型要求调用者提供已验证的输入 owner 与候选映射，不自动推导部署拓扑。
mask 的 token_id 为 null；selector 为条件路由而替换出的 0 不是原始词表 token。
shared experts 和物理专家映射不在当前逻辑 Top-K schema 内。

## 结果解释

示例：请求 A 的 verify-1，位置 8、输入 token 42，在 target 层 20 选 `[2,0]`，
层 21 选 `[3,1]`。候选 c1 被拒绝，但两条路由仍保留并计入专家计算次数。
之后位置 8 的 token 43 在 recompute-2 执行，不与前一次按位置去重。
TP 副本只合并完全相同、且符合声明 owner/observer 的记录；不硬编码 TP0。
不同 engine 的 execution/本地请求同名也不会合并成同一请求。

合并输出 `routes[].candidate_accepted` 为 true/false/null。null 表示没有对应结果，
不是拒绝；完整 run 的已验证候选必须有结果，普通输入及 mask 可以为 null。
专家次数是逻辑选择次数，不代表设备负载、通信字节或最终输出 token 数。
CSV 按执行类别、模型作用域、层、接受/拒绝与专家编号统计，包含零计数专家。
该 fixture 的总计数为 20，其中被拒绝候选贡献 4 次选择，不应从实际计算量扣除。

## 固定源码调查

准备官方 Git 源码：Ascend `0e181fc85fff24db10e5f92f10319c1913e6e9c5`，
vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。在独立目录检出，不能
为了检查而切换生产源码分支或覆盖用户改动。

```bash
python -S scripts/tools/check_ascend_v41_sources.py \
  --vllm-root /path/to/vllm-git \
  --ascend-root /path/to/vllm-ascend-git
MOE_V41_VLLM_SOURCE=/path/to/vllm-git \
MOE_V41_ASCEND_SOURCE=/path/to/vllm-ascend-git \
python -m pytest tests/test_ascend_v41_sources.py -q
```

只读检查 commit、10 个文件哈希与 AST；真实 selector/W8A8 apply 方法提取后在
CPU mock 对象运行，不导入 NPU 框架、不加载模型，不能证明模型端到端兼容。

## 未完成

真实服务输入/KV/DSpark 接入、分布式物理 dispatch、请求生命周期、部署镜像固定、
ACL 图录制/replay、并发多槽与跨流事件、完整 A3 验收及性能比较仍待开发。
静态缓冲原型仅 CPU 测过，要求前次 drain 才能复用；不是异步生产采集器。
汇总需全量读取内存，不适合无限时长采集。局部 CPU 成功不表示服务采集已成功。
worker 缓冲受条数和字节限制，但分片索引随运行增长；尚不承诺无限运行内存有界。

## 固定源码 hook 接入（CPU）

已新增 [CPU hook patch](../../patches/ascend/deepseek-v41-cpu/README.md)，
不再只是构造路由后测试存储：修改后的真实 `_model_forward`、`_select_experts`、
W8A8 `apply` 都在 CPU harness 中执行，hook 读取实际调用张量。
模型、router 的设备实现及专家 kernel 仍由 mock 替代，没有加载模型权重。

```bash
python -S scripts/tools/stage_ascend_v41_cpu.py \
  --vllm-root /path/to/pinned-vllm \
  --ascend-root /path/to/pinned-vllm-ascend \
  --output-parent ./outputs/v41-source-hooks
# 将上一步打印的实际目录填入：
OVERLAY=./outputs/v41-source-hooks/ascend-v41-cpu-实际后缀
python -m moe_trace.validation.ascend_v41_source_cpu \
  --overlay "$OVERLAY" --output ./outputs/v41-source-cpu
```

报告应为 validation_level=cpu_source_hooks、patched_methods_executed=3、
logical_routes=4、expert_choices=8、comparison=passed，同时
model_loaded=false、full_service_run=false、npu_verified=false。
明细和统计分别为 details.json、expert-counts.csv；consumer-reference.json
保存实际专家 consumer 收到的映射后 IDs/权重，不冒充物理设备负载。

请求 A 输入 41、位置 7，层 20 的实际 logical hook 保存 `[3,1]`；
映射并追加 shared 后，实际 consumer 收到 `[2,3,4]`。这两个数组分开保存。
这证明了源码 hook 的 CPU 接入，不证明 A3 图执行；不要将该副本直接部署 PD 服务。
