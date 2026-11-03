# DeepSeek V4 两机 1P1D 调试采集设计

日期：2026-11-04。分支：`feature/ascend-deepseek-v4-dp16-tp1`。

## 目标和边界

沿用用户已跑通的 P DP4/TP4、D DP16/TP1、每机 16 die 和
MooncakeHybridConnector 拓扑。第一阶段只保存 D 节点文本 Decode 的逻辑专家路由，
关闭两端 MTP，D 使用 V1、同步 eager，关闭图执行、FlashComm、共享专家多流、
fused MC2、DSA CP 和 EPLB。不修改原安装源码，不升级环境，不停止已有服务。
P 只调整调试部署参数，不安装路由 hook；先通过代理验证关闭采集的调试基线。

用户尚未提供真实 vLLM/Ascend 提交；开发使用已经取得的官方固定源码：
vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，Ascend
`0e181fc85fff24db10e5f92f10319c1913e6e9c5`。该 Ascend 提交同时包含
DeepSeek V4 与 V4.1，V4 接入必须检查 `models/deepseek_v4/model.py` 和 hash router，
不能用 V4.1 架构检查替代。服务器不匹配时报告差异并停止，不提供强行跳过开关。

## 采集接入

1. 独立 `ascend_v4_debug` adapter、`deepseek-v4-debug` patch 目录和启动/汇总入口。
2. runner 的真实 `_model_forward` 接入执行上下文；以 input_batch 的实际请求顺序，
   结合 scheduler token 数、实际 input_ids/positions 和请求 token 数据核对关联。
3. AscendRoutedExperts 的真实 router 返回后、log2phy/shared 拼接前保存逻辑 Top-K。
   V4 的 hash 层与普通 router 均读实际选择结果，绝不重算。
4. W8A8 apply 入口检查 consumer 与捕获结果；第一阶段禁止 mix placement/EPLB，
   允许静态 log2phy 映射，但核对映射后的实际 consumer ID，逻辑统计仍用映射前 ID。
5. TP1 All2All 行布局为本地输入，AllGather 按真实 DP 元数据排除外部 DP 行和 padding；
   未知 prepare 路径直接报错。每层保存全部本地有效 Decode 行，缺失即失败。
6. model load 后初始化全部 16 worker，shutdown 正常收尾。每个 worker 写 host 目录，
   不新增通信 collective，不自动上传。

## 输出和验证

目录为 `output_root/run_id/dp-N-tp-0/`，每次唯一 run_id。JSONL 分片和状态使用既有
WorkerWriter；另存带 SHA256 的真实 Decode 输入声明，以每个输入 × 每层校验保存覆盖。
保存请求 D 本地 ID、命名空间 ID、执行轮次、输入行、位置、阶段、层号、逻辑专家 ID、
worker/rank，token ID 和权重默认不保存。汇总 JSON，导出逐层专家计数及按 DP/执行轮次
计数 CSV；计数是专家选择次数，不是设备利用率、KV 流量或全局同步调度时间。

staging 自动定位当前解释器的包路径，核对固定文件摘要，复制 Ascend 包后补丁；
启动前验证副本内容、实际导入路径、原 vLLM 摘要及模型架构。保存启动参数与基线信息。
仅生成命令时不拉起服务；执行时只管理本工具创建的子进程组。

CPU 验证覆盖实际补丁 forward 块、All2All prepare、router 选择调用、W8A8 consumer
接入；NPU kernel、模型权重、分布式通信用 mock。覆盖并发请求顺序错位、prompt 过滤、
多个执行、hash/普通两类层、静态映射、DP 不等长、全部 16 worker、空闲 rank、异常与
篡改拒绝。提供可重复运行脚本和小样例。CPU 成功明确标为 `npu_verified:false`。

真实 A3 的模型加载、EP rank 排序、Mooncake PD 链路、吞吐影响和完整请求验收仍待现场
验证；源码摘要不一致时需要用户提供实际版本/源码继续适配，不能承诺任意安装可用。
