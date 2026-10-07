# DeepSeek V4.1 Decode 分布式采集设计（开发中）

## 范围与状态

目标是官方 5.2 的 A3 1P1D：P 为 DP4/TP4，D 为 DP8/TP2，开启 EP，
MooncakeHybridConnector、Ascend 量化、DSpark 5 候选、异步调度；D 目标模型
FULL_DECODE_ONLY，draft eager。仅采集 D 侧，不修改 P 路由或服务算法。
参考：https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4.1-Flash.html

本设计不是硬件支持声明。0.23.0 Granite adapter 保持原有限制。
新功能使用独立 schema，不将 DSpark 候选硬塞入现有 InputRow/TraceSession。
官方 HEAD 调查快照：`0e181fc85fff24db10e5f92f10319c1913e6e9c5`。
该快照是否含目标模型、对应 vLLM commit、部署镜像 digest 尚待源码确认；
不得仅凭文档或 HEAD 将它登记为可运行基线。

## 关联与计数

- deployment/run/node/worker 标识隔离输出；DP/TP/EP rank 和逻辑设备单独保存。
- global_request_id 与 local_request_id 分离，来自已验证的代理/连接器映射。
- scheduler_step 是引擎本地轮次；execution_id 是每次实际执行，不是全局时钟。
- 输入行显式携带 token ID、绝对位置、来源 engine、模型作用域、候选轮次和节点。
- target_decode / target_verify / dspark_draft 分开；不假定 DSpark 是线性 MTP。
- 重算保留新的 execution_id；同一位置不同候选不去重。
- 每个有效输入行指定唯一逻辑 owner；TP 副本必须匹配 owner 内容，否则失败。
  owner 必须由实际输入切分/通信路径提供，不硬编码 TP0。
- 被拒绝候选保留实际计算记录；接受结果是后续事件，不改写历史路由。
- 逻辑专家统计与物理 dispatch 统计分开，shared expert 不计入 Top-K。

## 图与异步执行

输入准备完成时冻结 CPU 身份快照，设备侧路由写入每次执行独占的缓冲槽。
ACL 图中必须包含采集拷贝操作；图捕获不产生用户记录。图重放之后等待对应完成
事件才读取有效行，剔除 padding。槽释放前不得复用；队列满时背压，失败标记不完整。
多流需确保 copy 等待实际 selector/dispatch 的完成事件。Python ContextVar 不能
跨 worker/图重放关联。若融合算子不暴露最终 IDs，先报告编译扩展需求及替代方案，
不擅自修改设备内核（遵守 requirements 操作边界）。

## 存储与汇总

worker 独立有界分片，完成标记附记录数/哈希；汇总检查所有预期 worker、结束标记、
重复身份、owner/副本一致性、候选结果引用、层覆盖。JSON 汇总允许原子覆盖，
worker 分片不覆盖。schema 及 CLI 将在存储实现时另行确认。
首阶段 CPU 契约模型仅处理小型测试数据，不宣称有界生产采集器。

## 验收关卡

1. 文档提交；官方源码 commit 与镜像基线确认。
2. CPU 事件契约：重排、TP 副本、跨请求同名、候选拒绝、重算、错误输入。
3. 分片/汇总及独立 fixture 对照，缺 worker/丢记录必须失败。
4. 固定源码 hook 测试，量化 Eager A3 校验。
5. 图重放、异步、多流、DP8/TP2/EP/PD/DSpark 完整 A3 验收。
6. 开关采集前后生成结果、吞吐、延迟、内存比较。

每个关卡独立提交。不发布未经硬件验收的安装 patch，不合 main、不推远程。
