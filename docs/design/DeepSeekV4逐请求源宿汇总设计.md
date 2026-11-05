# DeepSeek V4 逐请求专家路由源宿汇总

本次纠正汇总维度：时间窗口 → 请求 → 模型阶段/层 → 目的 EP worker。
源为实际处理该 token 的 DP worker/device；宿为被选逻辑专家的实际部署 EP rank。
一个请求可有多个目的端，长请求跨窗口时每个窗口仅统计新增选择次数。
不是客户端访问、PD KV 传输、网络字节数或真实通信消息追踪。

保留 trace 与旧 summary，新增 request_flow 模式。复用设备侧逐专家计数，按请求/阶段
分组，默认每5秒输出 load.jsonl 中的 requests 数组。被拒绝的实际计算也保留计数。
expert_selection_count 表示专家选择次数；同一 token 选择同一 device 的两个专家时计两次，
不冒充唯一 token 数或消息数。

初始化读取每层 global_expert_map；默认无全局表时调用框架同版本 determine_expert_map
为各 rank 构造映射，并与模块实际 _expert_map 校验。必须每个逻辑专家有唯一宿，
拒绝不明确的副本/EPLB/未知映射。映射在初始化阶段读取，不在图捕获中 host 转换。
保存 placements.json 供复查，不按 expert_id 直接猜 device。

每 worker 记录 worker.json：node、hostname、DP/TP/EP、当前 NPU logical device index、
ASCEND_RT_VISIBLE_DEVICES。宿通过同一输出 run/node 下 worker 目录定位；尚未发现元数据
时保留 ep_rank 并标 endpoint unresolved，不伪造对方物理卡或 die 编号。
这仍是当前单 D 主机 DP16/TP1 支持范围，不声称支持任意跨主机 EP。

资源随窗口请求数增长，新增 max_requests_per_window 默认128。即将超限时提前输出
partial 窗口并开始新窗口，不丢请求；单次batch超过上限明确报错。
沿用定时输出、计数事件、图保护锁、文件轮转，CPU与A3验收严格分开。
