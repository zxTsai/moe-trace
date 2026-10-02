# 请求生命周期真实 GPU 验证

日期：2026-10-02。范围：vLLM 0.10.1 V1、GraniteMoe、单 GPU、uni executor、
eager、无前缀缓存和投机解码。新增版本化补丁 `0003-request-lifecycle.patch`。

运行 ID：`f707f9e8-87c1-415e-8c42-ef0ad5f1c5dd`。
完整机器可读证据：[lifecycle-validation-result.json](lifecycle-validation-result.json)。

## 实现与进程边界

scheduler 的 `add_request` 产生 started 事件；`_free_request` 在释放请求前保存
终止事件，直接使用 RequestStatus、get_finished_reason、stop_reason、提示词与
已生成 token 数。采集 session 在首个请求加入时初始化，因此也能保存等待阶段
取消而没有执行路由的请求。后者通过单元测试验证。

默认单 GPU uni executor 的 scheduler 与 GPU runner 在同一 EngineCore 进程，
API server 位于另一进程，取消经 vLLM 本身的消息路径送至 scheduler。
显式 mp executor 的 scheduler 和 worker 分离，当前生命周期关联尚未通过 RPC
传递，因此明确拒绝该 executor，不依赖跨进程共享 Python 全局变量。

新运行的 metadata 包含 `request_lifecycle_required=true`。close 和 reader
校验 started/finished/cancelled 事件序列、终止原因与路由请求引用；缺少终止事件
会失败，取消后继续提交该请求的执行也会失败。status 保存请求数、终止请求数、
引擎 abort 数及 `all_requests_completed_normally`。旧运行未声明此字段，保留旧版
存储校验语义，不将它们作为新版生命周期验收证据。

## 真实实验

服务在独立端口 18001 启动，使用 reference-validation.yaml（保存 token ID），
max-model-len=1024、max-num-seqs=8、max-num-batched-tokens=64、
gpu-memory-utilization=0.65。运行包含调试期间的全部 11 个请求，没有删除失败
触发尝试；共 151 execution、6288 路由记录、2 分片，11 个请求都有终止事件。

修正后的 `scripts/lifecycle_requests.py` 四个用例：

| 用例 | 实际 scheduler 状态 | 终止原因 | 保存输入行/层记录 |
|---|---|---|---:|
| 停止 token 203 | FINISHED_STOPPED | stop | 4 / 96 |
| max_tokens=4 | FINISHED_LENGTH_CAPPED | length | 7 / 168 |
| 收到 2 个 SSE 文本块后关闭连接 | FINISHED_ABORTED | abort | 8 / 192 |
| 取消后再次发送请求 | FINISHED_LENGTH_CAPPED | length | 4 / 96 |

取消请求原要求 512 输出 token，实际 scheduler 已生成 3 个时处理 abort。
客户端收到两块与引擎已产生三 token 不矛盾：消息与计算存在在途数据。
保存的是 6 个 Prefill 和 2 个 Decode 实际输入，未从输出数推导记录。
其最后一次执行为 149；后续请求只出现在 150/151，取消请求没有继续参与执行。
写入器也在每次执行时拒绝已收到终止事件的请求产生新记录。

正常停止后：`complete=true`、`all_requests_completed_normally=false`、
buffered_record_count=0、terminal_request_count=request_count=11；reader validate
无错误。存在取消请求不会把完整采集误判为失败，也不会声称所有请求正常完成。

## API stop 与引擎 abort 的区别

初次用 API stop 字符串 `\n` 触发 stop 时，API 返回 stop，但输出处理器向
引擎发送 abort，因此原始事件为 cancelled/abort。没有将它改写为 scheduler stop。
另一个 EOS 提示尝试未触发 EOS，真实结果是 length。组合换行 token 478 也没有
被该次逐个换行生成触发，真实结果仍为 length。

最后使用实际生成的单换行 token 203，通过 scheduler 的 stop_token_ids 分支
验证 FINISHED_STOPPED。`stop_reason=203` 已保存。
cancelled 在当前文件中表示引擎 abort；仅凭这个事件不能断言是用户主动关闭连接。
本次取消用例有独立客户端主动关闭 SSE 的证据。API 原因与 scheduler 原因的完整
跨进程关联仍属于客户端映射扩展。

## 复现与补充验证

应用补丁 0001、0002、0003，按 reference-validation.md 的服务命令启动，去掉
MOE_TRACE_REFERENCE_DIR；执行：

```bash
conda run -n moe-vllm python scripts/lifecycle_requests.py
# 请求完成后正常停止自己启动的测试服务
conda run -n moe-vllm python scripts/verify_lifecycle.py \
  /home/czx/code/moe-routing/outputs/<run_id> \
  /tmp/moe-trace-lifecycle-client.json \
  --output /tmp/lifecycle-result.json
```

客户端脚本绕开本地 HTTP 的代理，主动关闭 SSE 后等待两秒再发后续请求。
verify_lifecycle 验证实际 API ID 到本测试内部 `-0` ID 的对应存在，但不把这个
后缀规则视为所有 API/引擎的通用规则。

18 项测试通过：包含缺终止事件拒绝、移除已保存终止事件后 reader 检出不完整、
取消轨迹保留、取消后执行拒绝、无执行取消，以及关闭配置缓存后的 hook 空操作。
测试服务已经停止。长时间内存控制与受控写失败仍待后续验证。
