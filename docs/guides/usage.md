# 完整使用指南

推荐先按[仓库 README](../../README.md)运行环境向导。以下为已有兼容环境的
手动接入方式及完整配置、输出和分析说明。所有命令在仓库根执行。

## 手动接入 vLLM 0.29.0

### 1. 手动准备配置（不使用向导时）

按上方说明获取源码，激活你自己的 vLLM 环境，在仓库根目录执行以下命令。
必须使用包含 0.29.0 适配器、补丁和下列示例配置的源码版本。

```bash
TRACE_REPO=$(pwd)
TRACE_MODEL=/absolute/path/to/granite-3.1-1b-a400m-instruct
# 把 TRACE_MODEL 替换为你已下载的模型目录。
cp configs/examples/vllm-0.29.0.yaml trace.yaml
```

编辑 `trace.yaml`，将 `output_root` 设为你自己的可写输出目录。
示例使用 `./outputs`，相对于启动服务时的工作目录；以下流程在仓库根启动。
YAML 不会自动展开 `$变量` 或 `~`，绝对路径请填写实际值。
`trace.yaml` 由用户自行创建，不应提交包含个人路径的配置。

### 2. 自动定位安装包，加载隔离补丁

```bash
TRACE_VLLM_PACKAGE=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
TRACE_VLLM_OVERLAY=$(python scripts/tools/stage_vllm_029.py "$TRACE_VLLM_PACKAGE")
# 任何一步报错都停止，不要继续启动服务或绕过摘要检查。
export PYTHONPATH="$TRACE_VLLM_OVERLAY:$TRACE_REPO/src"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export MOE_TRACE_CONFIG="$TRACE_REPO/trace.yaml"
python -c 'import vllm, moe_trace; print(vllm.__version__, vllm.__file__); print(moe_trace.__file__)'
```

脚本核对版本文件和目标 Python 文件 SHA256，创建新的临时包副本、复用
已有预编译扩展，并检查和应用补丁。不改动原安装包、不编译 C++/CUDA。
版本号相同但基础文件不匹配时，也会拒绝应用，不保证兼容任意 wheel 变体。

### 3. 启动采集服务

确认 GPU 和 8000 端口空闲，在同一终端执行：

```bash
vllm serve "$TRACE_MODEL" \
  --served-model-name granite-moe --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 1024 --max-num-seqs 2 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.55 \
  --enforce-eager --no-enable-prefix-caching --no-async-scheduling \
  --distributed-executor-backend uni --shutdown-timeout 30
```

这是一组小规模测试参数，显存充足程度不同可调整并发、token 预算和显存
比例，但不能去掉 eager、同步 uni、关闭前缀缓存和正常停机等待等适配约束。
不需要 `--enable-return-routed-experts`。关闭 FlashInfer 输出采样器用于
避免该采样器可能要求额外 CUDA JIT 编译，不会关闭 MoE 路由。

### 4. 发送请求

等待服务启动成功，在另一个终端执行：

```bash
curl --noproxy '*' --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"granite-moe","prompt":"Explain expert routing.","max_tokens":16,"temperature":0}'
```

也可使用 AISBench 等 API 客户端；自定义端口和 served-model-name 后，
客户端必须同步修改。采集的是服务端轨迹，不要求客户端安装 moe-trace。

### 5. 正常停机、校验并导出

请求结束后，在服务终端按一次 Ctrl+C，等待服务退出。
仍在仓库根目录和有工具依赖的环境中执行；按你的配置填写输出目录：

```bash
TRACE_OUTPUT_ROOT=./outputs
# 把“实际运行ID”替换为上述目录下本次生成的 UUID 子目录名。
TRACE_RUN="$TRACE_OUTPUT_ROOT/实际运行ID"
PYTHONPATH=src python -m moe_trace.cli validate "$TRACE_RUN"
PYTHONPATH=src python -m moe_trace.cli details "$TRACE_RUN" \
  "$TRACE_OUTPUT_ROOT/details.json" --format json
PYTHONPATH=src python -m moe_trace.cli counts "$TRACE_RUN" "$TRACE_RUN/expert_counts.csv"
```

预期结果：校验输出 `complete: true`；固定路径 `details.json` 每次覆盖写，
原始轨迹和专家计数保留在运行目录。若校验失败，不应当作完整实验。
配置关闭采集并重启即可停用；撤销隔离包需移除本次 PYTHONPATH 中的临时包
目录并重启，原安装环境无需恢复。

## 支持范围

| 项目 | 当前范围 |
|---|---|
| vLLM | 0.10.1、0.29.0，分别使用独立适配器和补丁 |
| 模型 | `GraniteMoeForCausalLM`；实际验证 Granite 3.1 1B A400M Instruct |
| GPU | 单 NVIDIA GPU；实际验证 RTX 4060 Laptop / WSL2 |
| 执行模式 | V1 引擎、uni executor、eager，TP/PP/DP=1 |
| 缓存与解码 | 关闭前缀缓存，不使用投机解码 |
| 0.29.0 额外要求 | V1 model runner、关闭异步调度、正数 shutdown timeout |
| 数据 | 实际输入位置、阶段、逐层 Top-K ID；token ID 和权重可选 |
| 客户端 | 服务端采集，不依赖客户端；已有 AISBench 流式 chat 验证 |

其他模型、其他版本、Ascend、多卡、CUDA Graph、异步调度、微批、monolithic
路由、抢占/重算与流式续接不在当前验证范围内；新版适配器对已识别的不支持
模式明确拒绝。不能仅因为模型使用 FusedMoE 就声称兼容。

## 环境与安装

先激活你自己已准备好的 vLLM 环境（Conda 或 virtualenv 均可），在仓库根
执行。不要为了安装本工具升级既有 vLLM、PyTorch 或 CUDA。

工具要求 Python >=3.11、NumPy >=1.26、PyYAML >=6；开发测试需要 pytest。
已有依赖和 setuptools >=68 时，可安装命令行入口：

```bash
python -m pip install --no-deps --no-build-isolation -e .
moe-trace --help
```

也可以不安装，直接使用 `PYTHONPATH=src python -m moe_trace.cli`。
上方快速开始采用此方式，无需绑定特定 Conda 环境名。
0.10.1 应使用对应适配器及基础 commit，不能应用 0.29.0 补丁；集成原理见
[0.10.1 补丁说明](../../patches/vllm/0.10.1/README.md)。该文档中的历史机器路径
需要替换为你自己的工具仓库和 vLLM 源码目录。

## AISBench 压测

AISBench 不是本工具必需依赖。先在你自己的压测环境准备 AISBench、数据集
及配置文件，配置模型 `granite-moe`、服务 `127.0.0.1:8000`、输出长度与
并发；可从小规模 20 条请求、并发 2 开始。

在已安装 AISBench 的终端执行，并把示例配置路径替换为你自己的文件：

```bash
ais_bench /absolute/path/to/aisbench_config.py \
  --mode perf --debug --num-warmups 0 --work-dir ./aisbench-results
```

参数以你安装的 AISBench 版本 `ais_bench --help` 为准；已验证版本支持上述
命令。远程或容器客户端不能用自己的 localhost 指向服务端，需配置可达的
服务地址及网络，不要为压测盲目暴露服务端口。
AISBench 自行发送的预热也是真实 API 请求，不会自动被本工具排除。
结果与证据见 [联调报告](../validation/aisbench-vllm-0.29.0-validation.md)，报告中的
本机路径仅用于保留历史证据，不是其他用户必须使用的安装路径。

## 配置

采集参数在 YAML 的 `trace` 段，vLLM 约束在 `vllm` 段。
`vllm` 段不会自动替代服务启动参数；未知字段和无效参数会报错。

| 参数 | 默认值 / 说明 |
|---|---|
| `enabled` | false，实验显式开启 |
| `output_root` | 请显式设置；通用示例使用 `./outputs`，代码旧默认仍为维护者路径 |
| `phases` | Prefill、Decode 都采集 |
| `layers` | null，全部声明的 MoE 层 |
| `request_labels` | null，全部请求；标签传递尚未完成 AISBench 联调 |
| `save_token_ids` | false |
| `save_expert_weights` | false，开启时保存实际消费权重并转存 float32 |
| `strict` | true，严格实验模式 |
| `buffer_records` | 8192，分片路由缓冲上限 |
| `shard_records` | 4096，必须不大于 buffer_records |
| `engine_id` | engine-0 |

层和阶段筛选已有实现，但不把所有组合声称为已完成真实 GPU 验收。
Granite 路由使用 Top-K 后重新归一化；采集器不再次归一化或缩放。
原始配置和 schema 版本随每次运行保存。

## 输出与字段含义

每次采集使用独立 UUID 目录，不覆盖原始轨迹：

```text
outputs/<run_id>/
├── metadata.json       # 模型、框架、schema、生效配置
├── requests.jsonl      # 请求开始、完成/取消及终止原因
├── steps.jsonl         # 实际执行、输入行和请求位置的关联
├── routes-000001.npz   # 原始路由分片，后续分片依次编号
└── status.json         # 状态、分片清单、计数和错误
```

JSON/CSV 明细与计数是离线导出文件，不会因启动采集自动生成。
NPZ 键、维度、类型和关联规则详见 [数据设计](../design/design.md)。

| 明细字段 | 含义 |
|---|---|
| `run_id`、`engine_id`、`rank_id` | 运行、引擎及 rank 身份 |
| `step_id`、`execution_id` | 当前适配器的轮次和实际模型执行身份 |
| `request_id` | vLLM 内部请求 ID |
| `phase`、`token_position` | Prefill/Decode、原序列中的零起始输入位置 |
| `layer_id`、`expert_ids` | 原始层编号及实际 Top-K 逻辑专家 ID 数组 |
| `token_id`、`expert_weights` | 按配置保存的可选字段 |

同一 execution 可以包含多个请求；按 request_id 和 token_position 区分输入。
当前轮次编号在实际 forward 时分配，不能当作 AISBench 的发送批次，
也不覆盖无 forward 的所有调度事件。

记录的是**输入当前 forward 的 token**，不是刚预测出的 token。
最后一个输出若没有再进入模型，就没有下一次路由。相同 token ID 不代表
相同上下文；不同层同号专家也不是同一参数实体。expert_ids 保留算子返回
顺序，不作为跨后端通用的专家权重排名承诺。

API ID、AISBench 数据项 UUID 和内部 request_id 不是同一标识。
0.29.0 默认会给内部 ID 加随机后缀；本次 AISBench 联调尚未建立逐数据项
到内部请求的显式映射，不应仅靠文件顺序推断对应关系。

## 校验与离线分析

在安装了工具的环境执行；将变量替换为要读取的实际运行目录：

```bash
TRACE_OUTPUT_ROOT=./outputs
TRACE_RUN="$TRACE_OUTPUT_ROOT/实际运行ID"
moe-trace validate "$TRACE_RUN"
moe-trace counts "$TRACE_RUN" "$TRACE_RUN/expert_counts.csv"
moe-trace details "$TRACE_RUN" "$TRACE_OUTPUT_ROOT/details.json" --format json
moe-trace details "$TRACE_RUN" "$TRACE_RUN/details.csv"
moe-trace counts "$TRACE_RUN" /tmp/step-10-counts.csv --step-id 10
moe-trace details "$TRACE_RUN" /tmp/request-step-10.json --format json \
  --request-id 实际内部请求ID --step-id 10
```

未安装入口时，将 `moe-trace` 替换为
`PYTHONPATH=src python -m moe_trace.cli`，并在仓库根目录执行。

测试阶段 JSON 采用覆盖写：反复导出到固定的 `outputs/details.json`，
只保留该文件的最新内容。不会自动删除历史 UUID 目录、原始分片或过去
导出的其他文件。CSV 也覆盖指定文件；请选择专用输出路径。

只有成功收尾且校验通过，status.json 才有 `state=complete`、
`complete=true`。没有结束标记的运行不能推断完整；未提交或缺失分片、
缺层、重复和专家越界会被检查。validate 失败返回非零退出码。

用户取消请求可以拥有完整的已执行轨迹，但
`all_requests_completed_normally=false`；engine abort 也不一定意味着
用户断连。不完整数据仍可分析，导出会显式标明不完整。
基础 CSV 按阶段、层和专家计数，包含零负载专家；统计不受数组排列变化影响。


## 后续验证

当前状态以[验证索引](../validation/README.md)和最新矩阵为准；层/阶段筛选、
关闭 token ID/权重已有部分真实 GPU 用例，不能推广到所有组合。
受控写入失败测试已完成，真实设备/进程故障与长期历史索引有界仍待完善。
开发与维护步骤见[开发指南](development.md)。
