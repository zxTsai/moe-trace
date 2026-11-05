# DeepSeek V4 逐轮逐 token 路由轨迹

用户最终确认的粒度：源 DP → 本引擎观测调度序号 → 实际执行 ID → 请求 → 输入 token →
每个 MoE 层的有序 Top-K 专家及对应目的 EP/device。5秒是批量落盘间隔，不是次数合并。

新增 routing_trace 作为推荐入口，旧 trace/summary/request_flow 保留兼容。
复用已经校验的实际输入、路由、MTP producer/verification 接入与静态专家放置映射。
完整轨迹仍包含图外设备等待与实际路由/位置 D2H，不能承诺与周期计数相同开销。

schedule_seq 是源 DP 引擎观测到的 target forward 序号，不能冒充全局调度ID。
同轮主模型与多次 MTP 共用序号，以不同 execution_id 与 mtp_step 区分。
token_position 为真实 kernel 位置；MTP 另存 input_sequence_position=position+1。
Top-K 保留顺序，目的 EP 数组逐项对齐；worker 元数据表提供实际 device 身份。

目录为 output_root/run/node/source-dp-000/routes.jsonl。每行一个批量窗口，含
rounds[].executions[].requests[].tokens[].layers[]，以及 sampler/proposal 辅助事件。
同一轮可以跨落盘窗口，使用 source+schedule_seq/execution_id 关联，不丢原始顺序。
不创建空负载窗口，不打印每worker周期INFO；只有DP0启停INFO和各worker异常。
本地没有请求时不创建采集执行上下文，但不改变原模型的DP通信/图重放。

CPU 缓冲以序列化字节计量，默认16MiB；满前提前flush，单执行超限明确失败。
默认文件64MiB、保留4个，轮转删除历史并标明保留范围。定时线程仅处理CPU序列化数据，
不会在后台触碰NPU或图捕获。正常退出flush剩余数据；强杀未落盘窗口不能视为完整。
实际设备身份来自worker.json，未发现的目的端保留EP rank并明确unresolved。

验证覆盖并发/请求换序、重复token不同位置/轮次、有序Top-K非相邻归属、多层、
MTP多执行、源目录、空闲无输出/静默、容量提前flush、轮转、隐私、故障及真实参考源码CPU接入。
真实A3/NPU/完整PD服务与性能验收仍待执行。
