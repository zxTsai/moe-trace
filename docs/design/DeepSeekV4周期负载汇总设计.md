# DeepSeek V4 周期负载汇总第一版

日期：2026-11-05。基线分支：feature/ascend-deepseek-v4-mtp-async-graph。

## 定稿范围

保留原 trace 完整轨迹入口，新增 summary 配置模式：连续记录实际 Decode 路由，
默认每 5 秒按 worker、scope、kind、原始层号汇总专家选择次数，输出 load.jsonl。
控制台只输出窗口编号、执行数和选择次数。不是每 60 秒采样 5 秒，不推算未观察数据。
保留官方镜像、D16/TP1、async、FULL_DECODE_ONLY、共享专家多流和 eager MTP。

summary 不保存请求/token/候选明细或接受率；这些由 trace 模式提供。
target_decode、target_verify、mtp_draft 分别计数，包含拒绝候选产生的已执行计算。
汇总为每层每专家计数，不等于每卡利用率、共享专家计算或网络流量。

## 执行和同步

沿用固定源码钩子和 graph entry 路由槽位。图外按本次设备 positions 和请求分段构造
有效 Decode mask；MTP 另排除拒绝后无效上下文、discard 与 padding。
在当前设备 stream 上执行定长 int64 scatter_add 计数；不逐 token 转成 CPU，不复制词表 token。
每个窗口只保留规模为层数×专家数×阶段数的设备累计值，图仍有路由快照 copy 成本。

计数更新使用锁与 NPU event 管理：更新前等待先前计数事件（允许调用 stream 改变），
更新后记录事件。后台定时线程设置同一 device，持锁等待该事件完成后，在专用传输 stream
打包计数向量并一次拷贝到 CPU，清空本窗口累计，再写 JSONL。
专用 stream 避免把拷贝排到工作 stream 新一轮推理/通信之后。
主线程不逐 forward 调用 device synchronize。
拷贝期间下一次统计更新可能短暂等待锁，定时汇总不是零开销或性能保证。
同一锁还覆盖 graph wrapper 的调用（包括 dummy/warmup），避免后台设备拷贝与
全局图捕获并发；不在 torch.npu.graph 捕获区域里执行 host 拷贝。
NPU 调用和图/多流安全必须真机验收；CPU 仅验证软件契约与参考 Python 接入。

定时器使用 monotonic 时钟，每 interval 触发一次；记录实际起止时间和窗口时长，
允许推理、设备等待和写入造成延迟，不承诺严格实时。无请求期间也输出空窗口，
正常结束输出剩余 partial 窗口。后台故障记录 failed，后续 forward 或 close 明确报错。

## 存储边界

每 worker load.jsonl 定长轮转，保留 max_files（含当前文件），总量不超过
max_file_bytes×max_files 加小型 status.json。单个窗口超过文件上限明确失败；
旧窗口按配置删除，status 标明最早保留窗口与累计轮转数，不能当全历史完整保留。
每个窗口携带 target/MTP/合计次数与完整窗口 payload，包含零计数专家。
单 worker 独立文件，不集中同步16worker，不新增 collective。

## 验证

CPU 检查独立计数 oracle、主模型普通/验证分开、MTP K1/2/3、实际源码块、
padding/Prefill/DP gather 映射、设备计数无逐 token host 转换、定时与空闲窗口、
轮转、关闭残余、异常状态、trace 回归与配置/副本指纹。
另在 A3 分别对比不开插件/trace/summary 的吞吐、TPOT、P99；不预报开销百分比。
