# DeepSeek V4：在现有 A3 Decode 容器中安装和调试采集

本指南对应 `trace` 完整轨迹调试模式。需要长期观察专家分布时，使用
[周期负载汇总快速开始](DeepSeekV4周期负载汇总快速开始.md)：相同安装与副本入口，
选择 summary 配置，每 5 秒输出计数、文件轮转，不保存逐请求明细。

适用于你已经按官方指导跑通的两机 1P1D 服务。镜像为
`quay.io/ascend/vllm-ascend:v0.23.0.post1-a3`，vLLM `0.23.0+empty`，
vLLM Ascend `0.23.0.post1`，D 为 DP16/TP1/EP16。

本分支已完成参考源码的 CPU 接入验证，可以拿到 A3 **安装、启动和调试**；
**尚未完成真实 NPU、模型权重、图执行、多流、Mooncake 与完整 PD 服务验收**。
主模型异步调度、`FULL_DECODE_ONLY`、`enable_npugraph_ex=true`、共享专家多流和
eager MTP K=1 保留。插件会在图外增加设备等待与同步拷贝，吞吐会下降，不能用本次
采集评估原服务性能。CPU 还覆盖了 K=2/3 的 Python 循环，但本指南的真机首测使用 K=1。

**不用构建新镜像，不用重新安装 vLLM、不用新建 Conda 环境。只在 D 容器操作。**
安装本身不会把钩子注入已运行的进程，完成准备后需要重启 D 服务。P 及 proxy
继续使用已经跑通的配置。请保留原 D 启动脚本，作为随时恢复的入口。

## 1. 获取本分支并安装工具

进入实际运行 Decode 的原容器，用运行服务的同一个 Python 执行以下命令。
代码目录只需可读写，不要求是 `/vllm-workspace`。

```bash
git clone --branch feature/ascend-deepseek-v4-mtp-async-graph \
  https://github.com/zxTsai/moe-trace.git
cd moe-trace
python -m pip show vllm vllm-ascend
python -m pip install --no-deps --no-build-isolation -e .
```

`--no-deps` 保留原环境的依赖版本，不下载或升级框架。
如果已经克隆本仓库，先保存自己的修改，再切换到上述 feature 分支、
执行 `git pull --ff-only`。镜像中的框架源码没有 Git 元数据也可以使用。

## 2. 检查实际源码，创建采集副本

```bash
python scripts/tools/inspect_ascend_async_graph.py
python scripts/tools/stage_ascend_async_graph.py --check

OVERLAY=$(python scripts/tools/stage_ascend_async_graph.py \
  --output-parent "$HOME/.cache/moe-trace/ascend-023")
printf '%s\n' "$OVERLAY"
```

成功标准：只读检查 `reference_source_match=true`；stage 输出一个新目录。
副本位于上面的 `$OVERLAY/vllm_ascend`。原 `/vllm-workspace/vllm` 和
`/vllm-workspace/vllm-ascend` 的文件不修改。副本内 `debug-hooks.patch`
列出全部修改，`overlay.json` 保存加载文件 SHA256。

工具优先通过当前 Python 的 import spec 确定包位置，不靠猜测安装路径。
若需要显式指定，可给 stage 增加：

```bash
python scripts/tools/stage_ascend_async_graph.py --check \
  --vllm-package /vllm-workspace/vllm/vllm \
  --ascend-package /vllm-workspace/vllm-ascend/vllm_ascend
```

如果出现 `source mismatch`，说明镜像中的实际文件与本补丁参考不同。
**不要修改摘要或把补丁强贴进去**。保存 inspect 输出和第一个差异文件，供继续适配；
检查失败不改动原包。`adapter_ready=false` 表示尚未完成真实 A3 验收，
`debug_entry_ready=true` 表示本分支提供了真机调试安装入口。

## 3. 设置输出目录和有限采集窗口

```bash
cp configs/examples/ascend-023-async-graph.json /tmp/moe-trace-023.json
python -m json.tool /tmp/moe-trace-023.json
```

修改 `/tmp/moe-trace-023.json` 的 `output_root` 为 **D 容器已有的可写持久目录**。
示例默认 `/trace-output`；没有挂载这个路径时，可以改成你已有挂载目录里的子目录，
不用为了插件重新创建容器。检查该路径：

```bash
python - <<'PY'
import json, pathlib, tempfile
c = json.load(open('/tmp/moe-trace-023.json'))
p = pathlib.Path(c['output_root'])
p.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryFile(dir=p):
    pass
print('输出目录可读写:', p)
PY
```

默认每 worker 最多采集 **200 次实际模型执行**，包含 target 和 MTP，
每 worker 文件上限 **256 MiB**。到达任一上限后，停止新增采集，模型继续运行，
状态为 `limit_reached`。16 个 worker 的主机存储总上限约 4 GiB，另有少量索引。
已捕获图中的路由 copy 仍随重放执行；停止写入不等于恢复原性能，退出调试须按第 7 节重启原服务。
需要更长窗口可增加 `max_executions`，同时保留合理的 `max_file_bytes`。

默认不保存 token ID、专家权重，仍保存请求、输入位置和专家 ID。
需要核对 token 时将 `save_token_ids` 改为 `true`；实际输入 token 会写进结果。
每次调试修改 `run_id`，如 `dsv4-debug-002`；`node_id` 用 `decode`。
启动器拒绝复用已有 run/node 目录，避免覆盖前一次结果或混合两轮采集。
分析报告 `analysis.json` 可以覆盖重写；原始事件采用有上限的追加写，保留调试证据。

## 4. 检查加载路径，包装原 D 脚本启动

仍在仓库根目录、同一终端执行：

```bash
python scripts/deployment/ascend_023_debug.py \
  --overlay "$OVERLAY" --config /tmp/moe-trace-023.json --check
```

成功标准：`launch_check=passed`，`origins.vllm_ascend` 指向 `$OVERLAY/vllm_ascend`，
`origins.vllm` 仍指向原安装源码，`origins.moe_trace` 指向本仓库。
这一步不拉模型、不启动服务。

随后正常停止原 D 服务，使用相同参数，通过包装器启动原脚本。例如原命令是：

```bash
bash /path/to/start_decode.sh 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 8000 16 0 10.0.0.2 13345 1
```

对应改为：

```bash
python scripts/deployment/ascend_023_debug.py \
  --overlay "$OVERLAY" --config /tmp/moe-trace-023.json -- \
  bash /path/to/start_decode.sh 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 8000 16 0 10.0.0.2 13345 1
```

**脚本路径、设备、端口、DP rank、地址及 RPC 端口必须换成你已跑通的原值**。
这几个数字仅用于展示脚本参数位置，不是本插件要求的地址或端口。
如果你原本按 DP rank 分别启动多个 D 进程，对每条原启动命令使用同一 overlay/config。
每个进程仍须具有不同的实际 DP rank；一个 worker 写一个目录。

包装器仅设置本次进程树的 `PYTHONPATH` 和采集配置，不改写原脚本的模型参数。
原脚本中的以下配置保留：

```text
--async-scheduling
--speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}'
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
ascend_compilation_config.enable_npugraph_ex=true
ascend_compilation_config.enable_static_kernel=false
multistream_overlap_shared_expert=true
recompute_scheduler_enable=true
MooncakeHybridConnector / kv_consumer
```

原脚本不要再覆盖 `PYTHONPATH`、切到其他 Python 环境或经不继承环境变量的启动器
创建 worker；否则需要在对应 worker 启动入口传递同一环境。不要同时开启
`--enable-return-routed-experts` 或其他旧采集 profile。
共享专家多流的最终有效值仍由原 Ascend 配置决定，例如框架自身可能因 fused MC2
冲突关闭该优化；插件不替用户改写配置。

## 5. 发送请求，确认实际采集

等 D 模型和原 PD proxy 正常就绪后，先从 **原 proxy 地址** 发送一条短请求。
以下地址和模型名换成你实际配置：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"请简短介绍专家混合模型。"}],"max_tokens":32,"stream":false}'
```

随后可以使用已经跑通的 AISBench 流程经 proxy 发起并发请求。
此版本不改变客户端请求格式，也不承担 P/D 之间的 KV 传输采集。

主机侧结果保存在你配置的目录，而非 NPU device 的存储中：

```text
/trace-output/dsv4-debug-001/decode/
├── dp-0-tp-0/
│   ├── events.jsonl
│   └── status.json
├── dp-1-tp-0/
│   ├── events.jsonl
│   └── status.json
└── ... dp-15-tp-0/
```

共享同一主机目录的 16 个进程分别写自己的文件，不会同时写一个 JSONL。
status 包含 DP/TP/EP、实际进程 device index、可见设备设置、主机名和 PID。
多个 DP worker 的 execution 编号各自独立，不把它们称为同一个全局调度轮次。

`events.jsonl` 一行是一个 JSON 对象，主要有三类：

| type | 内容 |
| --- | --- |
| execution | 执行 ID、本 worker 的调度序号、target/mtp、请求、实际输入位置、层与专家；剔除 padding 和 Prefill |
| sampler | 对应 target 执行的接受、首个拒绝、后续未采用决策，候选生产关联是否成功 |
| proposal | 候选轮次、K、请求、候选序号、实际 producer 执行/输入行、候选输出位置 |

例如主模型某个实际输入的路由，下面只是字段示意，**不是 A3 测试结果**：

```json
{
  "request_id": "request-A",
  "input_row": 1,
  "token_position": 101,
  "phase": "decode",
  "kind": "target_verify",
  "candidate_index": 0,
  "layer_name": "model.layers.0.mlp.experts",
  "layer_id": 0,
  "num_experts": 8,
  "expert_ids": [2, 5]
}
```

主模型 `token_position` 是实际序列位置；MTP 的 `token_position` 是 kernel 位置，
其实际输入序列位置另存为 `input_sequence_position=token_position+1`；
预测候选的位置是该 producer 的 `token_position+2`。采集输入路由，不给输出 token
伪造还没执行过的输入路由。共享专家不计入 routed expert 次数，逻辑专家编号不等于 device。

## 6. 分析与成功标准

服务运行时或达到采集窗口上限后，允许先分析部分结果：

```bash
python scripts/validation/analyze_ascend_023_debug.py \
  /trace-output/dsv4-debug-001 --allow-incomplete \
  --output /trace-output/dsv4-debug-001/analysis.json
```

正常停止 D 服务、所有 worker `status.state=closed` 后，可去掉 `--allow-incomplete`
进行完整 worker 覆盖检查。达到上限为 `limit_reached`，仍需保留该选项，结果代表有限窗口。
分析时若写入正在推进，可重试；分析器按 status 已索引的文件前缀及 SHA256 验证。
`failed` 状态即使加该选项也不接受为可分析的成功结果，应先排查错误。

真机初次调测检查：

1. 原 PD proxy 请求返回正常结果，D 日志没有采集异常。
2. 16 个 worker 都生成 status；无请求的 worker 可以没有 execution。
3. 报告 `capture_observed=true`，`executions>0`，`expert_choices.target>0`；
   请求有 MTP 预测执行时，`expert_choices.mtp>0`。
4. `combined=target+mtp`；被拒绝候选的已执行路由仍包含在选择次数中。
5. `unresolved_candidate_links=0` 且 `pending_verification_executions=0` 才表示观察窗口内的验证输入均已取得结果并成功关联；
   否则只用原始路由观察与调试，不能宣称候选归因已验收。

报告按 target_decode、target_verify、mtp_draft、层、专家统计，包含零选择的专家。
末尾产生但尚未被主模型验证的候选记 `unverified`，不当作拒绝。
这些计数反映逻辑专家被选择的次数，不能直接解释为每卡利用率、计算耗时或 KV 网络流量。

## 7. 出错时带哪些资料，如何恢复

保留 `inspect_ascend_async_graph.py` 输出、`overlay.json`、D 的完整异常栈、
失败 worker 的 `status.json` 及对应 `events.jsonl`。不要仅截最后一行错误。

| 情况 | 处理 |
| --- | --- |
| source mismatch | 实际镜像源码存在差异，先核对差异，不绕过摘要检查 |
| graph was captured without route hooks | 需要从进程启动时启用包装器，不能对已预热服务热安装 |
| graph capture recorded no routes / incomplete routing coverage | 实际编译或 fused 路径未经过预期钩子；保留异常与配置继续适配 |
| unsupported debug layout | 超出了 DP16/TP1、无 SP/EPLB/static kernel 等调试范围 |
| unresolved_candidate_links 非零 | 查看 proposal、execution 与 sampler 事件，保留全部记录排查异步候选关联 |
| status 为 incomplete | 主模型已经执行验证，但关闭时没有对应采样结果；按部分窗口调试，不当作完整验证 |
| 没有文件 | 检查是否重启了 D、实际加载路径、启动脚本是否覆盖了环境、输出目录权限 |
| 有 status，没有 execution | 可能尚未有真实 Decode 请求，或请求没有经过这组 D worker |

恢复时停止这次 D 进程，**在未经过包装器的终端**运行原 D 启动脚本。
包装器的环境只在它启动的进程树中生效，原框架文件未改动。
无需卸载或重装 vLLM，无需删除采集数据。更新插件代码后重新 stage，使用新的 run_id。

CPU 开发复验和证据边界见[本版验证报告](../validation/DeepSeekV4异步图执行采集CPU验证.md)。
