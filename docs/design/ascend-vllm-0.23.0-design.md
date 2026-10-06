# Ascend 0.23.0 CPU 适配设计

## 基线与验收边界

复用 main 的 schema 1.0.0、TraceSession、Reader、分析和 JSON 覆盖导出。
新增 adapter，不修改 NVIDIA adapter，也不把 CPU 模拟冒充 NPU 路由。

- vLLM tag v0.23.0：`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`。
- vLLM Ascend tag v0.23.0：`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。
- CPU 环境 Python 3.12、PyTorch 2.10.0+cpu，不安装设备后端。
- 首个模型限定 GraniteMoeForCausalLM；其真实 Ascend 可运行性仍需 A3 检验。
- 单进程 uni、TP/PP/DP/PCP/DCP=1，非量化、eager、同步调度。
- 拒绝 EP/EPLB、前缀缓存、投机、输入 embedding、LoRA、恢复/抢占重算、图模式、
  FlashComm、多流 gate、融合 MC2 和 PD 分离。模型/执行路径不支持时必须报错。

## 接入位置

1. 上游 scheduler 新请求/释放请求：记录完整生命周期。
2. Ascend NPUModelRunner.execute_model 的真实 `_model_forward`：建立 execution 上下文。
   从已经准备和重排后的 input_batch 获取 counts；逐行核对真实 input_ids 和 positions。
   与 request.get_token_id(position) 比对；position < num_prompt_tokens 为 Prefill，
   其余为 Decode。只接受无 padding 的一维整数 token 输入，不用“一轮一个 token”猜测阶段。
3. AscendUnquantizedFusedMoEMethod.apply：在实际 fused_experts 参数构建之前拷贝
   topk_ids/topk_weights；即权重 dtype 转换之后，而不是重跑 selector。
   验证 zero experts、force balancing、动态 EPLB 等不支持分支未开启。
4. scheduler.shutdown：要求已采集请求有终止事件；异常保持 failed/incomplete。

warmup/dummy_run 不进入 execute_model 的采集上下文，因此不产生路由记录。
step_id 为本 adapter 的真实 execute_model forward 顺序；当前范围一轮一个执行，
保留独立 execution_id，不声称覆盖异步/微批/多 rank。

## CPU 证明链

使用真实 PyTorch CPU 张量和实际 tag 源码：

- adapter 单测覆盖批次重排、并发、混合阶段、分块输入、可选字段、过滤、复制隔离、失败状态。
- 独立模拟专家 consumer 保存其实际入参，adapter 保存输出，两者逐 token/layer 对照；
  consumer 不调用 adapter 的映射函数，避免同源自校验。
- 版本化补丁带原文件 SHA256；检查原始源码、补丁应用/撤销和修改后 Python AST。
- 从原始 tag 源码提取 patched 方法，在 mock 对象上执行真实接入点，验证 context 和生命周期。
- 所有既有测试在无 vLLM/NPU 的独立环境回归。

CPU 验收只证明采集和接入 Python 逻辑。NPU kernel、驱动/CANN、设备同步、
真实模型、A3 并发/AISBench 和完整官方 Ascend UT 均不在此验收内。

## 输出、环境与维护

新增 CPU 验证脚本，输出到指定目录，JSON 明细使用既有覆盖导出。
验证报告必须标识 `cpu_simulation`，给出独立对照数及源码版本，禁止写成硬件验收。
使用独立 Conda 环境；补丁只应用在复制的 Python overlay，不覆盖原安装。
补丁工具遇到版本/哈希/冲突不匹配直接退出。保留现有工具入口。
当前 TraceSession 的全局去重/历史索引仍随运行增长，不承诺长期运行内存有界。
