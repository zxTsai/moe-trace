# main 真实服务验证

2026-10-03，验证提交 `5083a25`，实际启动 vLLM 0.29.0，而非重放旧数据。
环境：`moe-vllm-0.29.0`、RTX 4060 Laptop、Granite 3.1 1B A400M。
保留原环境，隔离补丁目录 `/tmp/moe-trace-vllm-029-jjh0alqj`。

## 服务和请求

测试端口 18001 启动前为空闲。采用 V1 runner、关闭 FlashInfer sampler、
eager、无前缀缓存、同步 uni、停机超时 30 秒；最大长度 1024、最大并发 8、
批 token 预算 64、显存比例 0.55。未开启 `--enable-return-routed-experts`。
健康检查 HTTP 200，MoE 实际使用 TritonExperts modular。

执行 `scripts/validation/reference_requests.py`：3 个错峰不同长度请求及 151-token 长提示词。
执行 `scripts/validation/lifecycle_requests.py`：stop、length、两段输出后断连取消、取消后请求。
8 个 API 请求返回 HTTP 200，取消请求按预期保留中止状态。
Ctrl+C 触发 drain 停机，最后分片已排空。Conda 外层因 KeyboardInterrupt
返回 1，不作为采集成功依据；下列存储状态、独立探针核验和进程退出为依据。

## 本次结果

运行 ID：`80505e24-abbb-4a34-91cc-73c3e670ed9c`。

| 项目 | 结果 |
| --- | --- |
| 实际执行 | 62 次 |
| 多请求同轮执行 | 18 次 |
| 同轮 Prefill / Decode | 2 次 |
| 模型输入行 | 278 |
| 逐层路由 | 6,672 |
| 专家 ID | 53,376，与独立消费探针一致；权重也核验通过 |
| 请求生命周期 | 8 开始、8 终止，包含 1 取消，核验通过 |
| 持久化 | 4,096 + 2,576，缓冲 0，complete=true，无错误 |
| JSON / CSV | 6,672 明细、1,536 计数行、53,376 专家选择，核验通过 |
| JSON 覆盖写 | 对同一路径再次导出成功，不报文件已存在 |

原始轨迹和导出：
`/tmp/moe-main-live-20261003/traces/80505e24-abbb-4a34-91cc-73c3e670ed9c`。
独立探针：`/tmp/moe-main-live-20261003/reference/reference.jsonl`。
参考核验及生命周期报告：`/tmp/moe-main-live-20261003/{reference,lifecycle}-result.json`。
这些是临时测试证据，不提交大型轨迹；临时目录可能被系统清理。

## 警告及边界

启动报告 optional DeepGEMM 因缺少 CUDA_HOME 不可用，实际 Triton 后端正常。
推理首次形状触发 Triton JIT 警告，不是采集错误。未编译 vLLM C++/CUDA 扩展。
服务和引擎进程均已退出，18001 端口释放，未停止任何已有服务。
此验证覆盖当前 0.29.0 基本链路，不代表本次验证了 0.10.1 在线服务、
AISBench 压测、完整新环境下载安装或全部长期可靠性验收。
