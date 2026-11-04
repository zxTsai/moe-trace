# DeepSeek V4 的 MTP 采集设计

日期：2026-11-04。分支：`feature/ascend-deepseek-v4-mtp`。
基于非 MTP 分支 `9a527d5`，保留原 adapter、patch、指南和测试。

## 动态 K 扩展（本轮）

用户要求取消 K=1 限制。按配置读取最大 K，每次真实 `_propose` 读取本轮 K（0 到配置上限），
采集首次上下文 forward 和后续逐候选 forward，不按配置假造执行次数。
同一个物理 MTP 层可以重复执行 K 次；每次独立 execution，候选 node 为 0..K-1，
共享一个 proposal round。每请求、每步核对输入 token、kernel position、producer 输出。
target 可能只调度该轮候选的前缀，按实际调度长度匹配，未验证后缀保持 unverified。

sampler 以实际有效输出长度减 1 得到接受前缀长度；核对整个接受前缀。
第一个未接受节点标 rejected，后续标 not_adopted，已计算的后缀路由全部保留。
共享 CandidateResult 的 accepted=false 表示未接受，导出另提供细分 decision，
不会把后缀说成独立验证拒绝。源模型长度钳制导致的越界预测必须明确拒绝采集，不能
把被重置到 position 0 的 dummy 行绑定到真实请求。图、异步、多物理 MTP 层等边界保留。

CPU 增加 K=2/3、接受 0/部分/全部、K=0、裁短验证、同 ID 不同节点、
后续输入关联错误和漏采 step 拒绝；K=1 原测试继续回归。

## 首阶段支持范围

两机 A3 P4/TP4、D16/TP1、Ascend W8A8、MooncakeHybridConnector；D V1、同步 eager，
`method=mtp,num_speculative_tokens=1`，单个 MTP 层。不同时开放图/异步、多步 MTP、
DSpark、动态 K、CP/PP 或多模态。服务器实际版本仍未知，沿用 V4 已验证的固定源码并
增加 MTP proposer、metadata、sampler 等文件摘要。不同源码必须先适配，不能跳过检查。

## 源码事实与采集语义

主模型 scheduler 的 `scheduled_spec_decode_tokens` 指明候选输入，不能再要求候选
已存在于请求 output_token_ids。按实际 input_batch 顺序和 input_ids/positions 校验：
普通行标 `target_decode`；有候选的请求标 `target_verify`，非候选输入为 seed，
候选行关联上次真实 MTP 产物的 request、round、node、token 和位置。

Ascend `llm_base_proposer._run_merged_draft` 首次 forward 可含多行上下文；不是每请求
只有一行。实际 MTP input_ids 相对 target 左移一位，positions 仍是原 target 坐标。
使用 proposer 真实 query_start_loc 和 token_indices_to_sample 映射请求、排除 padding；
保存模型实际 position，额外在输入声明中说明 token 序列位置为 position+1。
候选输出位置为被采样 MTP 输入 position+2。draft 输入不冒充它预测出的候选 token。
保存 `mtp_draft` 输入路由和单独的 producer→candidate 关联。

`_run_merged_draft` 返回实际 draft IDs 后登记候选；下一次 target 输入必须与登记匹配。
在 rejection_sampler 返回后直接读取实际 sampled_token_ids 与 SpecDecodeMetadata：
K=1 时两枚有效输出表示接受，一枚表示拒绝；校验接受前缀值等于候选，不自行采样。
不依赖后续截断、EOS、请求输出长度；标注这是 sampler 决策，不是客户端最终保留状态。
未被调度验证的候选保持 unverified，不把取消/停机误标成拒绝。

主模型路由和 MTP 路由采用同一 logical selector/consumer hook，由独立上下文区分
model_scope。保存被拒绝候选的已执行 target 路由；不根据结果删除计算。
继续禁止改变 logical ID 的强制负载均衡或 shared mix placement；静态 log2phy 后 consumer
与捕获预期核对，统计仍用 logical ID。

## 数据与完整性

扩展通用 DecodeRow 允许 `mtp_draft`，保持旧版本 schema/事件读取兼容；CandidateResult
沿用既有事件。每 worker manifest 声明 target/mtp 两个 scope 的原始层号。
输入声明、候选生产关联采用 SHA256 和正常收尾索引；离线同时验证全部 16 worker、
每个有效输入×scope 层、候选生产→验证→结果以及 producer 采样输入。
所有实际 target_verify 候选都必须有结果；最终尚未验证的产物单独导出，不伪造结果。

JSON 保存全部轨迹、候选及实际决定；CSV 按 scope/kind/层/专家/决定计数，含零负载专家；
单独给出 target、mtp、combined 总计，并保留 DP/执行轮次。combined 是相加的选择次数，
不同模型的同名层不会混成同一个层。token/权重默认不保存，生产/验证内部仍核对实际 ID。
同步 D2H、有界分片不作性能承诺，离线汇总和长期历史索引的内存限制沿用 debug 版本。

## CPU 验证与 A3 边界

执行真实补丁 runner forward、draft first-pass forward、MTP 返回和 sampler 接入，
实际 Python router/consumer 使用已有 V4 source 方法，NPU kernel、模型和通信仍 mock。
构造接受、拒绝、同 token 不同 round、请求顺序改变、输入上下文与 padding、未验证末轮、
缺结果、跨请求错误关联、隐私开关及损坏文件，核对三类轨迹和合计计数。
CPU 成功不代表 MTP 模型加载、KV 共享、NPU kernel 或 PD 服务通过，保留 npu_verified=false。
