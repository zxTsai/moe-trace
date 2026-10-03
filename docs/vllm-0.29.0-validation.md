# vLLM 0.29.0 采集验证

日期：2026-10-02。分支：`feat/vllm-0.29.0`。

新版基本采集链路已通过真实 GPU 验证，0.10.1 适配器未修改。
这不是对需求全部验收项或所有 vLLM 执行模式的兼容声明。

## 环境与参数

- Conda `moe-vllm-0.29.0`：Python 3.12、vLLM 0.29.0（上游短标识
  `g98dff2a81`）、PyTorch 2.13.0+cu130，RTX 4060 Laptop GPU。
- 模型：本地 Granite 3.1 1B A400M Instruct，24 层、每层 32 专家、Top-8。
- 在 `/tmp/moe-trace-vllm-029-2pc4ypyw` 独立复制 Python 包，链接安装环境
  已有 native 扩展；实际导入路径记录在 metadata.json。不升级或重新安装
  任何依赖，不修改安装环境和原 0.10.1 vLLM 源码。
- `VLLM_USE_V2_MODEL_RUNNER=0`、`VLLM_USE_FLASHINFER_SAMPLER=0`。
  默认 FlashInfer 输出采样器曾因缺少 nvcc 预热失败；使用官方替代路径
  后启动成功。实际 MoE 后端为 TritonExperts modular 路径。
- `--enforce-eager --no-enable-prefix-caching --no-async-scheduling`
  `--distributed-executor-backend uni --shutdown-timeout 30`
  `--max-model-len 1024 --max-num-seqs 8 --max-num-batched-tokens 64`
  `--gpu-memory-utilization 0.65`，测试端口 18001。
- 开启 token ID 与路由权重。配置：`configs/vllm-0.29.0-validation.yaml`。

## 完整运行证据

运行：`8ca31e33-ccff-4434-8f03-8985c87ca9ef`。

持久目录：
`/home/czx/code/moe-routing/outputs/8ca31e33-ccff-4434-8f03-8985c87ca9ef`。
保留 metadata、requests、steps、两个 NPZ 分片、status、details.json、
expert_counts.csv、客户端映射以及独立探针和验证结果。
小规模报告另见 [validation-result.json](vllm-0.29.0-validation-result.json)。

| 校验项 | 实际结果 |
|---|---|
| 生命周期 | 8 个请求开始、8 个终止，其中 1 个取消 |
| 实际执行 | 63 次 |
| 模型输入 | 278 行，208 Prefill / 70 Decode |
| 逐层记录 | 6,672 条，全部 24 层有记录 |
| 专家 ID 与权重 | 53,376 个 ID 及对应权重与独立消费探针一致 |
| 并发 | 17 次实际执行包含多个请求 |
| 混合阶段 | 2 次执行同时包含 Prefill 和 Decode |
| 分块 Prefill | 151-token 提示词分为位置 0–63、64–127、128–150 |
| 分片收尾 | 4,096 + 2,576 条，缓冲为 0，complete=true |
| JSON 往返 | 6,672 条导出明细与读取器逐条相同 |
| CSV 计数 | 1,536 行，总计 53,376 次，含 103 行零计数，与原始记录重算相同 |

独立核对方法：模型 forward pre-hook 读取真实 GPU input_ids、positions，
并通过 GPU query_start_loc 边界和请求 token 历史建立参考输入行；
另在 RoutedExperts.forward_modular 入口读取实际被消费的 Top-K IDs 和
权重。只共享 run/execution 身份，不读取采集器的 InputRow 或 route 缓冲。
`verify_reference.py` 检查输入字段、执行与路由键集合、ID 和 float32 权重，
没有重新计算 softmax/topk。

取消用例收到两段非空 SSE 后断连，内部状态 FINISHED_ABORTED/abort，
保留 8 个实际输入行、192 条路由；后续请求不再出现该取消请求。
运行完整与全部请求正常完成分别记录：complete=true，
all_requests_completed_normally=false。

新版内部 ID 是客户端 completion ID 的 `-0` 后再加八位随机十六进制
后缀。来源已检查 `input_processor.assign_request_id` 和 completion serving。
生命周期报告保存实际 API/internal 对应关系，要求唯一匹配；这只适用于
本用例的单 completion，不推广到所有 API 或关闭随机化的设置。

## 停机问题及修复

最初两次运行分别为 `823dcf22-e55d-49fd-9512-676e8a47b5dd` 与
`36e26267-dab7-41c3-a0d1-eae8bba16dfe`，保留在 `/tmp/moe-trace-029-outputs`。
两次都未完成排空，只有已提交的分片可读；状态仍是不完整，没有修改成成功。

默认 shutdown timeout=0 会立即强杀 EngineCore；即使设置 30 秒，只把
钩子放到 executor.shutdown，仍可能在退出忙循环、重置信号处理之后
被后续 SIGTERM 打断。最终补丁在 `_handle_shutdown` 无工作分支提前
调用 close_trace，并保留 executor 兜底。最终完整运行证明缓冲排空、
生命周期和层完整性校验均成功。强杀或超时仍不保证完整。

## 关闭采集与回归

关闭采集时使用用户启动参数中的 served-model-name=granite-moe、
enable-return-routed-experts、bfloat16、2 个槽、1024-token 批预算、0.55
显存比例，额外明确同步 uni 和 30 秒停机等待。请求 HTTP 200，3 prompt
token / 4 output token；比较运行目录集合，未新增采集目录。
此用例只证明关闭采集时原生路由开关不受影响；不代表同时开启两套
路由采集已验收。测试服务均已停止。

停机日志有 vLLM/multiprocessing 的一个 leaked semaphore 警告；不影响
本次响应和无采集目录结论，但尚未单独定位，不将其隐藏为无警告运行。

两个 Conda 环境分别执行 29 项工具测试全部通过，Python 编译检查通过。
包括真实输入错位/错 token/padding 拒绝、异步/V2/零等待拒绝、disabled
hooks 无操作、权重参考损坏检测和既有公共存储/生命周期回归。

## 尚未验收的范围

- 新版长时间资源有界、磁盘写失败注入、所有筛选组合与权重关闭 GPU 用例。
- 同时启用原生 enable-return-routed-experts 和本工具的采集。
- 其他模型、monolithic 路由、V2 runner、异步调度、重算/抢占、
  speculative、prefix cache、多卡、微批和流式续接；当前明确拒绝或不支持。

复现启动、补丁校验及撤销步骤见 [新版补丁说明](../patches/vllm/0.29.0/README.md)。
