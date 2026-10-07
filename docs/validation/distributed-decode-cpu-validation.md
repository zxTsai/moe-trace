# 分布式 Decode CPU 开发验证

日期：2026-10-08。分支：feat/ascend-deepseek-v41-decode。
范围：CPU 事件契约与存储，固定源码方法调查，静态路由缓冲原型。
不包括真实 PD 服务、DeepSeek 推理、DSpark kernel、HCCL/EP、ACL 图或 A3。

## 已运行结果

运行 `python -m moe_trace.validation.distributed_decode_cpu --output outputs/distributed-decode-cpu`：

| 项目 | 结果 |
| --- | --- |
| fixture workers | 4（模拟 DP2/TP2 身份） |
| 包含副本的观测 | 20 |
| owner 去重后的逻辑路由 | 10 |
| 专家选择次数 | 20 |
| 独立 consumer fixture 对照 | passed |
| NPU 已验证 | false |

覆盖接受/拒绝候选、同位置重算、target/draft 同号层、mask/seed、乱序与非 TP0 owner。
报告与明细位于 outputs/distributed-decode-cpu，忽略提交。

## 自动化检查

本次提供全部固定源码/overlay 的最终全量回归：**250 passed in 5.65s**，无 skip。

- Worker 数量、结束状态、分片哈希/计数、未提交文件、声明输入、层与副本缺失。
- 写入失败、序列化缓冲字节限制、旧输出不覆盖、JSON 汇总覆盖写。
- CLI validate/merge/counts 与独立 fixture、失败报告覆盖历史成功；CSV 总选择 20，
  rejected 候选选择 4，包含零计数专家。
- 静态缓冲 CPU copy：有效行重排、padding、完成事件、陈旧/重复 copy、复用限制。
- 固定源码检查 10 个 SHA256 与 AST。
- 真实 `_select_experts` 与 W8A8 `apply` 方法在 CPU mock 上执行，验证映射和 shared
  expert 追加后的真实 consumer 入参。非完整官方 UT，非真实模型执行。

## 复现全量回归

使用独立 CPU 开发环境，提供固定源码及此前 0.23.0 overlay：

```bash
MOE_V41_VLLM_SOURCE=/path/to/pinned-vllm \
MOE_V41_ASCEND_SOURCE=/path/to/pinned-vllm-ascend \
MOE_ASCEND_SOURCE_OVERLAY=/path/to/023-overlay \
python -m pytest -o addopts='' -q
```

不提供这些源码时相应集成测试会 skip，不能把 skip 称为源码接入验收。

## 下一关卡

目前没有 V4.1 生产安装 patch 或生产 adapter。需要将经过 KV 接收、SP/prepare 重排的
有效输入映射到 router，补齐候选结果、请求终止和物理 dispatch；再在 A3 验证
量化 eager、图重放、多流、异步和完整 PD/DSpark。CPU 契约成功仅表示声明数据
一致，不证明声明覆盖了全部真实执行，更不证明 NPU 侧数据正确。

## 固定源码 CPU hook 接入（追加验证）

后续已加入 [CPU hook patch](../../patches/ascend/deepseek-v41-cpu/README.md)。
修改后真实 Runner `_model_forward`、selector `_select_experts` 和 W8A8 `apply`
在 CPU mock 上全部执行，不再只把预设 Route 送入存储。
输入核对来自实际 forward 张量；hook 读取 selector 输出与 consumer 输入。
模型/router 设备实现和专家计算仍为 mock，没有加载 DeepSeek 权重或运行服务。

- `cpu_source_hooks`：3 个修改后的真实方法，4 条逻辑路由、8 次选择，对照 passed。
- 实际 logical `[3,1]`；consumer 在映射与 shared 追加后收到 `[2,3,4]`。
- 原始固定源码保持不变，patch 再生成逐字一致；副本 manifest 与固定哈希核对。
- 图、无上下文启用、非法 token、缺 consumer、异常上下文清理等测试。
- 最终全量回归 **258 passed in 6.78s**，无 skip；原有 250 项阶段结果保留。

报告：outputs/v41-source-cpu/source-validation.json；明细 details.json 与
expert-counts.csv；实际 consumer 入参 consumer-reference.json。全部忽略提交。
报告显式保存 model_loaded=false、full_service_run=false、npu_verified=false。
