# vLLM 0.29.0 适配设计

状态：独立适配器、补丁和隔离加载已实现，基本采集链路已通过真实 GPU
验证；范围和证据见 [新版验证报告](vllm-0.29.0-validation.md)，不沿用
0.10.1 的通过结论。

## 基线与隔离

- 原有 0.10.1 基线提交：`9898449`，保留在 `main`。
- 新版开发分支：`feat/vllm-0.29.0`。
- 已安装环境：`moe-vllm-0.29.0`，Python 3.12。
- 新版源码位置：该环境的 `lib/python3.12/site-packages/vllm`。
- 不覆盖原有 0.10.1 适配器、补丁、环境或验证结果。
- 不安装或升级依赖，不编译 CUDA/C++ 扩展，不推送远端。

## 适配范围

公共配置、存储、读取器和 JSON 导出继续复用。新增独立的
`vllm_0_29_0` 适配器及 `patches/vllm/0.29.0/` 补丁。
这不是仅替换版本号：新版执行缓冲、路由入口和 executor 路径均有变化。

首轮仍限定 Granite MoE、单 GPU、uni executor、eager、无前缀缓存、
无投机解码；额外显式关闭异步调度。其他模型、并行模式、微批执行、
流式续接请求和 monolithic 路由后端，未建立可靠映射前明确拒绝。
使用 V1 model runner，并要求 shutdown timeout 大于 0，推荐 30 秒；
默认 0 秒的 API 停机路径会立即强杀 EngineCore，无法保证采集收尾。

## 已检查的集成位置

1. `v1/core/sched/scheduler.py`：`add_request` 新请求登记分支与
   `_free_request`，分别记录请求开始和终止。不能把续接当成新请求。
2. `v1/worker/gpu_model_runner.py`：实际 `_model_forward` 周围建立执行
   上下文；排除预热，按 `input_batch.req_ids` 和每请求实际输入数量
   建立有效行映射。新版输入缓冲使用 `input_ids.cpu`，不能直接沿用
   旧版 `input_ids_cpu`。校验真实 forward 的 token ID、position 和行数。
3. `model_executor/layers/fused_moe/runner/moe_runner.py`：在
   `router.select_experts` 返回后、`routed_experts.forward_modular` 消费前，
   捕获原始 Top-K 张量。monolithic 路径没有这个可见结果，不能静默漏采。
4. `v1/engine/core.py`：无工作、退出忙循环前排空并校验采集结果；
   `v1/executor/uniproc_executor.py` 保留 shutdown 兜底。必须在重置信号
   处理前完成主要收尾，不能仅依赖晚期资源拆除。

阶段仍按实际输入位置与提示词长度判定。只记录实际送入模型的 token，
不能用生成 token 数反推 Decode 记录数。跨进程关联必须重新核实，
不能只凭旧版 uni 行为假设 Python 全局状态共享。

## 补丁应用策略

优先在临时独立 Python 包副本应用补丁，通过 PYTHONPATH 加载，复用
安装环境的预编译扩展，不修改已安装包。补丁记录发行版本、可获取的
上游 commit 标识和目标文件摘要；应用前核验原文件并执行冲突检查。
不得编造安装 wheel 未提供的完整 Git commit。

## 实现与验证顺序

1. 新适配器与最小 Python 补丁；验证关闭采集时 hooks 无副作用。
2. 原有测试回归、新版不支持模式拒绝测试及文件往返测试。
3. 新环境 GPU 基础推理与单请求采集，验证真实导入路径。
4. 并发与分块 Prefill；独立检查实际模型输入和实际专家消费张量，
   对比保存结果，不能只检查文件存在或尺寸。
5. 请求完成、取消与正常停机，导出 JSON 和独立验证报告。

所有报告单独标明 vLLM 版本。0.10.1 的既有通过结果不能转记为
0.29.0 已通过；新版尚未完成的场景保持未执行状态。
