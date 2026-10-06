# Ascend 0.23.0 CPU 验证报告

## 结论

Ascend adapter 的 CPU 专家路由采集、输出及固定源码 Python 接入验证通过。
**这不是 A3 硬件验收，不代表真实 NPU 推理已支持。** 未执行官方 Ascend 全量 UT。

开发分支：`feat/ascend-vllm-0.23.0`；独立 Conda 环境：`moe-ascend-cpu`。
Python 3.12.14、PyTorch 2.10.0+cpu。没有安装 vLLM、TorchNPU、CANN 或 vLLM Ascend，
没有改动既有 NVIDIA 环境、源码或服务。

固定源码及 hash manifest 见[补丁说明](../../patches/ascend/0.23.0/README.md)。
vLLM tag commit 为 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`；
Ascend tag commit 为 `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。

## 证据矩阵

| 项目 | 结果 |
| --- | --- |
| 既有公共采集/配置/存储/导出及 NVIDIA adapter 单测 | 保持通过 |
| CPU 全套，显式包含固定源码 integration | 182 passed，无 skip |
| 未设置源码 overlay 的普通回归 | 181 passed，1 项源码 integration 明确 skip |
| 实际 tagged Python 源码接入点 | patched apply、_model_forward、forward context、scheduler add/free/shutdown 执行通过 |
| batch 重排及 scheduler 字典顺序不同 | 请求关联正确 |
| chunked Prefill、Decode、混合阶段 | 逐 token 位置/阶段核验通过 |
| 可选 token ID/权重、层过滤、阶段过滤、请求标签 | 通过 |
| BF16 consumer 权重、专家顺序及复制隔离 | 逐元素一致；不重新归一化/排序 |
| 缺层、重复层、错误形状、越界/溢出 ID、NaN/Inf/负权重 | 拒绝；运行不能标记 complete |
| 不支持模式及抢占/恢复/嵌套 forward | 拒绝 |
| 关闭采集及 warmup 外部调用 | 无路由数据产生，不访问设备依赖 |
| 三请求完成、一个未执行请求取消、关闭排空 | 生命周期完整，无缺失终止事件 |
| 原文件哈希、Python AST、补丁应用/撤销 | 通过；原源码未被修改 |
| 从固定干净 commit 重生成补丁/manifest | 与仓库文件一致 |
| 人为错误请求映射/专家排列 | 独立 consumer 校验检测出错误 |
| JSON 再次导出 | 覆盖而不是累加；失败重验证不会遗留成功摘要 |
| compileall、pip check、文档本地链接 | 通过 |

## 独立场景结果

三个不同长度的执行请求 A/B/C，另有请求 D 在没有 forward 时取消：

| 轮次 | 实际输入布局 |
| --- | --- |
| 1 | A Prefill 位置 0–1 |
| 2 | B Prefill 位置 0–1；A Prefill 位置 2–3 |
| 3 | A Decode 位置 4；C Prefill 位置 0；B Decode 位置 2 |
| 4 | C Prefill 位置 1–2；B Decode 位置 3；A Decode 位置 5 |
| 5 | B Decode 位置 4；C Decode 位置 3 |

合计：5 次执行、15 个输入 token 行、2 个 MoE 层、Top-K=2，保存 30 条路由记录，
60 个专家选择及对应 consumer 权重；4 次多请求执行、2 次混合 Prefill/Decode。
所有记录按独立显式场景和 consumer 入参核验；不是只比记录数量。

结果位于本机仓库下 `outputs/ascend-cpu/`（被 Git 忽略）：
`details.json`、`consumer_reference.json`、`cpu-validation.json` 和 `traces/<run_id>/`。
临时固定源码与 overlay 位于 `/tmp/`，可能被系统清理；不提交大型数据或源码副本。
运行 ID 不是固定值，最新摘要指向当前 run；JSON 覆盖，原始 UUID 目录保留。

## 重现与剩余范围

按[CPU 开发指南](../guides/ascend-cpu-development.md)准备独立环境、下载固定源码、
创建 overlay；显式设置 `MOE_ASCEND_SOURCE_OVERLAY` 后运行完整测试。
未准备源码时相关测试明确 skip，不能把该次单测当成源码 integration 已通过。

尚待 A3：验证 Granite 可运行性、真实输入准备与路由行序、NPU 同步/拷贝、实际专家
kernel 入参独立对照、并发/AISBench、退出及硬件异常。其他模型、TP/EP/EPLB、
量化、图模式、PD 分离、缓存复用、投机均未验收。
长期 TraceSession 历史索引仍会增长，本工作不解决生产级长期内存上界。
