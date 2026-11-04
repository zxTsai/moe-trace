# DeepSeek V4：保留异步、图执行和多流的采集适配设计

日期：2026-11-04。分支：`feature/ascend-deepseek-v4-mtp-async-graph`。
基于已提交的同步 eager MTP 版本 `0dc03e8`；原 MTP 分支保持不变。
状态：源码调查和设计阶段，**本分支尚未实现当前原配置的完整采集接入**。

## 用户要求和不变的启动配置

复用已正常运行的两台 A3、原容器及原 Python 环境，只在 D 安装插件和加载采集副本。
P 保持原服务；不构建新镜像，不替换原 vLLM/Ascend，不要求用户关闭以下配置：

| 配置 | 目标值 |
| --- | --- |
| D 拓扑 | DP16 / TP1，EP |
| 调度 | `--async-scheduling` |
| 主模型图执行 | `FULL_DECODE_ONLY`，`enable_npugraph_ex=true` |
| 静态 kernel | `enable_static_kernel=false` |
| 共享专家多流 | `multistream_overlap_shared_expert=true` |
| MTP | `method=mtp`，当前 K=1，`enforce_eager=true` |
| KV | MooncakeHybridConnector consumer，P4×4 / D16×1 |
| 其他 | CPU binding、recompute scheduler、原内存和 batch 参数保留 |

注意：MTP 的 enforce_eager 不等于主模型 enforce_eager；本次配置是主模型图执行、
MTP eager、异步调度和共享专家多流组合，不能套用旧 adapter 的同步输入检查。
仍采集实际 logical routed expert choices，不把共享专家或物理设备利用率混入原计数。

## 已确认的源码差异

用户已确认运行镜像：`quay.io/ascend/vllm-ascend:v0.23.0.post1-a3`。
用户随后提供 pip show，确认服务环境为 Python 3.12.13、vLLM `0.23.0+empty`、
Ascend `0.23.0.post1`，editable 源码分别在 `/vllm-workspace/vllm` 和
`/vllm-workspace/vllm-ascend`。`+empty` 是构建标识，不表示缺少源码。
新适配使用该 release 源码，而非旧 MTP 分支的较新参考源码。release 参考为：
Ascend `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`；
其 Dockerfile 默认 vLLM v0.23.0，参考提交 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`。
构建可以覆盖 VLLM_COMMIT，镜像 tag 也不等同于内容摘要，因此仍要核对实际文件。
不要求用户安装 Git、重拉源码或替换已有镜像。

该 release 的 V4 模型为 `models/deepseek_v4.py`、MTP 为 `models/deepseek_v4_mtp.py`；
路由在 `quantization/methods/w8a8_dynamic.py` 的 select_experts 之后，
不存在旧补丁依赖的 `ops/fused_moe/routed_experts.py` 接入结构。
版本化参考清单保存在 `patches/ascend/deepseek-v4-023post1-async-graph/reference-files.json`。
只读检查入口为 `scripts/tools/inspect_ascend_async_graph.py`，以实际 import spec 定位为先、
默认源码目录为后备。不导入 NPU 框架、不执行 Git、不改变服务；
`reference_source_match=true` 仅是源文件匹配，`adapter_ready=false` 明确表示采集实现未完成。
该检查器已通过 4 项 CPU 测试，在本地 release 参考源码上 18 个文件摘要全部匹配；
没有读取 A3 容器的实际文件，不能把本地结果当作服务器文件匹配或采集通过。

以下图/异步/多流接入原则最初从旧参考源码调查而来，需在 release 基线上逐项重新核实：

- `compilation/acl_graph.py`：首次捕获调用 Python runnable，重放调用 graph.replay 并返回
  缓存 output。Python selector hook 不会随每次图重放运行。
- 旧参考有 `compilation/breakable_aclgraph.py`；0.23.0.post1 的图扩展入口需按其实际
  `enable_npugraph_ex` 分支调查，不能直接照搬旧 wrapper 名称。
- `worker/model_runner_v1.py`：异步请求状态可能保留 PLACEHOLDER_TOKEN_ID，CPU 请求
  token 数据和采样结果会延后更新，不能直接复用 request.get_token_id 的同步断言。
- 旧参考的 `shared_experts.py`/milestones 在新 release 中结构不同；该 release 的
  fused_moe.py、fused_moe_0_23_0.py 与 W8A8 路径必须一起核对流依赖和返回类型。

## 新采集路径

### 图内记录、图外搬运

在图捕获前按 scope、图 entry/bucket、层、行数和 Top-K 建立稳定的 NPU 路由缓冲。
在真实 selector 输出后、log2phy/shared 映射前加入 device copy，使它成为捕获图的一部分。
Python hook 仅负责登记静态关联；不在图捕获中进行 .cpu/.tolist、写盘或 Python 值判断。
每次实际 replay 结束，在图外按 execution 提取这次缓冲，排除预热/捕获样本和 padding。
eager fallback 与 replay 共用输出契约，但分别记录执行路径。

必须为每个实际调用保存独立快照，或者在可证明安全的 ring slot 复用前等待完成，
防止下一次 replay 覆盖上一轮还未搬到 host 的路由。device、pinned host、pending execution
都要有限额；满队列可以背压，但不能丢记录或假标 complete。

### 异步输入和采样关联

在执行入口冻结实际 req_ids/输入分段/执行 ID，不引用后续可变 input_batch。
对实际 device input_ids、positions、有效行 metadata 和候选 mask 建立同轮快照。
不能因为 CPU 侧 token 尚未更新而把占位符当真实输入，也不能将下一轮请求表绑定到上一轮。

候选产物、target 验证和 sampler 结果通过稳定执行/候选标识延迟关联。
结果未就绪保持 pending；收到结果后补充 accepted/rejected/not_adopted，实际未调度验证的
产物保留 unverified。终止或退出时未完成的有效验证不能被当作拒绝或完整实验。

### 多流依赖

记录 routed tensor 的生产 stream/event，使拷贝 stream 在数据真正就绪后读取；
共享专家原有 milestones、wait 和返回值保持。跨 stream 引用的 tensor 生命周期持续到
拷贝完成。不得以仅在主 stream 记录一个 event 代替对实际生产 stream 的确认。

### 安装和启用

继续使用独立源码副本和版本摘要检查，原安装源码不改动。
提供“原 D 启动脚本外加采集加载配置”的入口，不用强制 eager 的旧统一启动器。
加载器不重写原 async/compilation/additional/speculative 参数。
采集有额外拷贝和内存开销，不承诺维持原吞吐或消除全部同步等待。

## 版本确认和实施顺序

默认源码目录按用户确认使用 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend`。
这是 A3 容器内路径，本开发机没有该目录，不能据此推断实际版本。
必须读取运行 D 服务的解释器和源码提交，确定实际 graph wrapper、runner、sampler 和 stream
实现后，再生成可加载补丁；路径相同不代表源码相同。

1. 获取真实 D 端版本/提交，扩展 graph/async/multistream 关键源码摘要清单。
2. 实现图 entry/bucket 路由缓冲与 replay 调用边界，不放开旧 adapter 的保护条件。
3. 实现不可变输入快照、异步结果 join、多流事件及有限额队列。
4. 接入主模型图、eager MTP 与必要的 eager fallback；输出原三类路由和候选状态。
5. CPU 执行固定源码 Python 接入块，并注入重放、延迟/乱序结果、覆盖、padding、队列满、
   错误和退出场景；mock 图/stream 测试只作为软件契约证据。
6. 真实 A3 验证 capture/replay 实际更新、不同 bucket、空闲 rank、并发与结果对照。
   CPU 无法证明 NPU graph 允许新增 copy/event，也无法证明设备多流同步正确。

本设计不等于上述实现已完成。现有同步 eager MTP 版本不能直接用于原始异步/图/多流配置，
也不能通过删除 ConfigError 或只安装插件来获得这些支持。
