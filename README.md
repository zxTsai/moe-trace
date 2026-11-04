# moe-trace

> 当前 `feature/ascend-deepseek-v4-mtp-async-graph` 分支正在适配用户镜像
> `quay.io/ascend/vllm-ascend:v0.23.0.post1-a3` 的异步、图执行和共享专家多流配置。
> **现已提供保留原 D 启动配置的采集调试入口，CPU 接入验证通过，真实 A3 待验收。**
> 从[现有 A3 容器安装和调试](docs/guides/DeepSeekV4现有A3容器采集调试指南.md)开始：
> 下载代码、安装插件、检查源码、创建副本、包装原脚本启动、查看 JSON 和专家计数。
> 不构建新镜像、不升级框架；采集增加同步等待，会降低吞吐。
> [适配设计](docs/design/DeepSeekV4保留异步图执行多流的采集设计.md) ·
> [CPU 验证与真机待验收项](docs/validation/DeepSeekV4异步图执行采集CPU验证.md)。
> 下文同步 eager MTP 功能属于已有实现，不代表本次原配置适配已经完成。

**长时间观察专家负载：** [周期负载汇总快速开始](docs/guides/DeepSeekV4周期负载汇总快速开始.md)。
第一版已提供 `summary` 模式：连续在设备侧累计，默认每 5 秒输出一条 `load.jsonl`，
按主模型普通 Decode/验证和 MTP 分层统计，文件按大小轮转。无需新镜像，复用原 D 脚本。
`trace` 完整轨迹模式保留；两种模式启动时选择。CPU 已验证，真实 A3 与性能收益待验收。

面向 vLLM 的 MoE 专家路由采集工具，将实际模型执行、内部请求、输入 token
位置、模型层和 Top-K 专家关联起来，支持按执行轮次分析并发请求。

实现为独立工具包、版本化 adapter 和最小 Python 补丁。读取模型实际消费的
路由结果，不重新计算 softmax/Top-K，不依赖 `--enable-return-routed-experts`。

## 支持与状态

| 本分支场景 | 启动配置 | 当前状态 |
| --- | --- | --- |
| v0.23.0.post1-a3，DeepSeek V4，D DP16/TP1/EP16 | 原 async、FULL_DECODE_ONLY、npugraph_ex、共享专家多流、eager MTP K=1 | 可安装真机调试；CPU 源码接入通过，A3 未验收 |
| 新调试输出 | execution/sampler/proposal JSONL、SHA256、JSON 专家计数 | 保留拒绝计算；无法关联的候选明确 unresolved |
| 周期负载 summary | 连续设备计数、默认 5 秒窗口、JSONL 轮转、JSON 汇总 | 无逐请求轨迹/接受率；CPU 接入通过，A3 性能待测 |
| 旧同步 eager MTP profile | 下文原指南 | 独立 adapter/patch，不与新 profile 混装 |

**DeepSeek V4 + MTP 调测入口：** [从零部署快速开始](docs/guides/DeepSeekV4开启MTP采集指南.md)
（宿主机、镜像、模型、PD 服务、插件安装、采集与分析）· [更新说明](docs/更新说明.md)。
独立分支 `feature/ascend-deepseek-v4-mtp`，D DP16/TP1、动态 K 候选 MTP、同步 eager；
区分主模型 Decode/验证和 MTP 预测，保留接受/拒绝及未验证候选，分别汇总专家选择次数。
CPU 接入验证通过，A3 待验收；按实际 K 逐步采集，不同时支持图或异步。

**DeepSeek V4 两机 A3 调测入口：**
[从获取代码、安装到采集与负载分析](docs/guides/DeepSeekV4两机A3快速开始.md)。
分支 `feature/ascend-deepseek-v4-dp16-tp1`，P DP4/TP4、D DP16/TP1、Mooncake PD，
D 仅同步 eager、无 MTP 的 Decode；hash/普通路由接入已完成 CPU 验证，**A3 待验收**。
源码兼容检查通过后才能进入真机调试，不支持原样保留 MTP/异步/图执行的部署。

`main` 统一维护 **vLLM 0.10.1 和 0.29.0**，不必按版本切换 Git 分支；
两个版本仍需各自的环境、adapter 和补丁。

目前验证范围：Linux/WSL2 x86_64、单 NVIDIA GPU、Granite 3.1 1B A400M
Instruct、V1/uni/eager、TP/PP/DP=1、无前缀缓存和投机解码。
0.29.0 还要求 V1 model runner、关闭异步调度和正数停机超时。
其他模型、多卡、Ascend、CUDA Graph 等不在支持承诺内。

Ascend 开发分支新增 **vLLM + vLLM Ascend 0.23.0 CPU 验证功能**：真实 CPU 张量、
模拟并发请求、固定源码 Python hook 和独立 consumer 对照。
**尚未完成 A3/NPU 验收，不能用于声称 Ascend 推理已支持。**
CPU 操作见[开发与验证指南](docs/guides/ascend-cpu-development.md)，
接入边界见[适配设计](docs/design/ascend-vllm-0.23.0-design.md)。

两机 A3 / DeepSeek V4.1 的 Decode 调试开发版另见
[**新手快速使用指南：环境、部署、安装采集、查看 JSON**](docs/guides/A3新手快速使用指南.md)。
首次使用请从该指南开始，不要套用 NVIDIA 环境向导或旧 0.23.0 补丁。
本版要求固定源码，关闭 DSpark/MTP、图执行和异步调度；**真实 A3 验收仍待完成**。

Qwen3.6 单机 A3 DP8/TP2 适配在 `feat/ascend-qwen36-035b-dp8-tp2` 分支进行，
已实现 W8A8 Decode 接入、独立补丁副本、调试启动与 JSON/CSV 汇总，CPU 源码接入验证通过；
**A3 真机尚未验收，不能沿用 DeepSeek 启动器**。
准备与接入计划见[Qwen3.6 适配设计](docs/design/Qwen3.6单机A3采集适配设计.md)。
首次使用见[Qwen3.6 专用快速开始 README](docs/guides/Qwen3.6/README.md)：下载代码、检查环境、加载采集、导出和 CPU 复验。

**实验版，尚未完成全部可靠性验收。** 双版本真实采集、AISBench 并发、
独立消费探针对照、取消、关闭采集、层/阶段过滤和导出已有验证。
整体长期内存仍会随历史索引增长；全新环境在线安装因网络中断未通过，不能据此用于生产。
当前结论见[验证索引](docs/validation/README.md)，历史报告不是最新验收状态。

## NVIDIA 快速开始

### 1. 获取源码

先安装 Git、Conda 和兼容 NVIDIA 驱动，自行下载模型；本工具不安装驱动或模型。

```bash
git clone https://github.com/zxTsai/moe-trace.git
cd moe-trace
```

已有克隆更新前先保存自己的修改，再执行 `git switch main` 和
`git pull --ff-only`。以下命令均在仓库根目录执行。

### 2. 准备专用环境

入口只依赖 Python 标准库，不要求提前安装工具、NumPy 或 vLLM：

```bash
python scripts/prepare_environment.py --detect
python scripts/prepare_environment.py --mode new --version 0.29.0 \
  --name moe-trace-collection
```

安装前显示计划并询问确认；自动化可显式加 `--yes`，只查看计划用
`--dry-run`。兼容的已有环境可选择 `--mode existing`，不自动升级依赖。
新环境默认带 PySocks；默认继承网络代理，允许直连时可显式用 `--no-proxy`。

向导检查依赖、补丁 SHA256、CUDA/扩展及实际导入路径，生成独立补丁包、
配置和加载脚本，**不会启动服务或改写原 vLLM 安装包**。
Conda 只隔离依赖，不隔离 GPU、显存或端口。
完整参数、失败处理及 0.10.1 使用方式见[环境准备指南](docs/guides/environment-setup.md)。

### 3. 加载并启动测试服务

将 `TRACE_SETUP` 替换为向导实际输出目录，`TRACE_MODEL` 替换为你下载的模型：

```bash
conda activate moe-trace-collection
TRACE_SETUP=/absolute/path/to/prepared-directory
TRACE_MODEL=/absolute/path/to/granite-3.1-1b-a400m-instruct
source "$TRACE_SETUP/activate-trace.sh"

vllm serve "$TRACE_MODEL" \
  --served-model-name granite-moe --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 1024 --max-num-seqs 2 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.55 \
  --enforce-eager --no-enable-prefix-caching --no-async-scheduling \
  --distributed-executor-backend uni --shutdown-timeout 30
```

以上是 **0.29.0** 小规模测试参数。先确认 GPU 和端口空闲，依显存调整预算，
不要取消适配约束。已有环境的手动配置、隔离补丁和旧版命令见
[完整使用指南](docs/guides/usage.md)及[0.10.1 补丁说明](patches/vllm/0.10.1/README.md)。

### 4. 请求、停机和导出

服务就绪后，在另一个终端发送请求：

```bash
curl --noproxy '*' --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"granite-moe","prompt":"Explain expert routing.","max_tokens":16,"temperature":0}'
```

请求结束后，在服务终端按一次 Ctrl+C，等待服务正常退出。默认向导配置的
原始轨迹位于 `outputs/traces/<run_id>`；若改过输出目录，请相应替换：

```bash
TRACE_RUN=./outputs/traces/实际运行ID
moe-trace validate "$TRACE_RUN"
moe-trace details "$TRACE_RUN" ./outputs/details.json --format json
moe-trace counts "$TRACE_RUN" "$TRACE_RUN/expert_counts.csv"
```

JSON/CSV 覆盖指定导出文件；原始 UUID 目录不覆盖、不自动清理。
校验失败不能当作完整实验。采集记录的是实际输入 forward 的 token；
最后生成的 token 未再次输入模型时没有下一条路由。

## DeepSeek V4.1 PD Decode 开发状态

`feat/ascend-deepseek-v41-decode` 正在开发多机 A3、PD、EP、DSpark 的 D 侧采集。
目前包含 CPU 事件契约、worker 分片/汇总、静态缓冲原型和固定源码 CPU hook；
**没有可用于该部署的服务安装 patch，尚未接入真实 ACL 图重放或完成 A3 验收。**
不要使用下面的 0.23.0 Granite 安装步骤部署 V4.1。
操作与结果示例见[分布式 Decode CPU 验证指南](docs/guides/distributed-decode-cpu.md)。

## Ascend 快速开始（实验功能）

当前功能在 `feat/ascend-vllm-0.23.0` 分支，尚未合并 `main`。
**仅完成 CPU 模拟和固定源码 Python hook 验证，真实 A3/NPU 验收待完成。**
下列步骤说明独立采集环境的接入方式，不是已经验收的 A3 部署承诺。
当前 NVIDIA 环境向导不安装 Ascend 环境。

**完整步骤见 [Ascend 快速开始：环境 → 服务 → 采集 → 分析](docs/guides/ascend-quickstart.md)**，
包含官方 A3 镜像安装、原服务请求验证、工具安装检查、双包加载路径断言、
重启采集服务、并发请求、正常关闭、轨迹校验及按轮次/请求查看专家分布。
以下是接入摘要；先跑通原服务，安装后必须重启专用服务，不能热注入正在运行的进程。

### 1. 先准备官方推理环境

按 [vLLM Ascend 官方指南](https://docs.vllm.ai/projects/ascend/zh-cn/latest/getting_started/installation.html)
在专用容器或隔离环境安装匹配的 `vLLM 0.23.0 + vLLM Ascend 0.23.0` 及配套软件栈，
先确认未接入采集工具时能够正常推理。不要修改现有生产环境；`latest` 文档会变化，
应固定具体版本。CPU 开发不需要这套 NPU 环境，见本节末尾。
当前仅适配 Granite，其 A3 可运行性未验证；原服务不能跑通时先停止，不用其他模型替代后继续。

### 2. 安装 moe-trace 工具包

获取包含 Ascend 功能的本仓库代码，在上述专用环境中、仓库根目录执行：

```bash
python -m pip install .
python -m pip check
moe-trace --help
python -c 'from moe_trace.adapters import ascend_vllm_0_23_0; print(ascend_vllm_0_23_0.__file__)'
```

这一步安装我们的 adapter、存储、校验、导出模块及 `moe-trace` 命令，
运行依赖为 NumPy、PyYAML；**不安装或升级 vLLM、vLLM Ascend、TorchNPU、CANN，
也不修改它们的源码**。仅安装工具包不会自动启用采集。

### 3. 生成带 hook 的隔离副本

脚本位置：[scripts/tools/stage_ascend_023.py](scripts/tools/stage_ascend_023.py)。
以下路径是占位符，替换为当前环境的原始 Python 包目录：

```bash
python scripts/tools/stage_ascend_023.py \
  --vllm-package /absolute/path/to/site-packages/vllm \
  --ascend-package /absolute/path/to/site-packages/vllm_ascend \
  --output-parent ./outputs/ascend-overlays
```

若不知道安装路径，可在未加载副本的环境中查看：

```bash
python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm").origin).parent)'
python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm_ascend").origin).parent)'
```

参数含义：`--vllm-package` 和 `--ascend-package` 指向原包，不是 Git 仓库根目录；
`--output-parent` 指定副本的父目录。脚本核对接入文件 SHA256、检查补丁锚点和 Python
语法，然后复制两个包并只修改副本。遇到源码差异或已打补丁时拒绝操作，不覆盖原安装。
成功后打印新建的唯一目录，例如：

```text
outputs/ascend-overlays/moe-ascend-023-随机后缀/
├── vllm/
└── vllm_ascend/
```

省略 `--output-parent` 时使用系统临时目录，可能被清理；正式使用建议显式指定持久目录。
这一步不启动服务，也不产生轨迹。

### 4. 仅让专用采集服务加载副本

将脚本打印的目录填入变量。`PYTHONPATH` 指向**包含两个包的父目录**，
不是其中的 `vllm/` 或 `vllm_ascend/`：

```bash
ASCEND_TRACE_OVERLAY=/absolute/path/to/generated/moe-ascend-023-directory
cp configs/examples/ascend-vllm-0.23.0.yaml ./outputs/trace-ascend.yaml
# 编辑 trace 配置：enabled: true；output_root: ./outputs/ascend-traces。
# 测试时可将 save_token_ids、save_expert_weights 都设为 true。
export PYTHONPATH="$ASCEND_TRACE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export MOE_TRACE_CONFIG="$PWD/outputs/trace-ascend.yaml"
export VLLM_USE_V2_MODEL_RUNNER=0
```

随后在这个专用终端/容器中启动符合 adapter 约束的测试服务：Granite、非量化、
uni/eager、同步调度、TP/PP/DP/PCP/DCP=1，关闭前缀缓存、投机、EP/EPLB、FlashComm、
多流及融合 MC2，不同时开启 `--enable-return-routed-experts`。
这些约束必须由真实服务配置满足，不能只写在采集 YAML 里。
完整限制见[设计](docs/design/ascend-vllm-0.23.0-design.md)和
[补丁说明](patches/ascend/0.23.0/README.md)；Granite 在 A3 上的可运行性仍待核验。

此过程不会切换或重启已经运行的服务。不加载副本的服务继续使用原安装；
恢复原方式时，在未设置这些采集变量的新终端启动服务即可，原代码无须撤销补丁。
当前没有自动化 Ascend 安装/启动向导，不能将此方式描述为安装后自动生效的纯插件。

### 5. 请求、停机、确认采集生效

加载副本后按[完整快速开始](docs/guides/ascend-quickstart.md#5-重启采集服务并发送请求)
启动专用服务。在另一个终端发请求，等待成功后在服务终端按一次 Ctrl+C，等待排空：

```bash
curl --noproxy '*' --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"granite-moe","prompt":"Explain expert routing.","max_tokens":16,"temperature":0}'
```

选择**本次**原始轨迹目录，不能拿旧成功运行来验证新安装：

```bash
TRACE_RUN=./outputs/ascend-traces/本次运行ID
moe-trace validate "$TRACE_RUN"
moe-trace details "$TRACE_RUN" ./outputs/ascend-details.json --format json
moe-trace counts "$TRACE_RUN" ./outputs/ascend-expert-counts.csv
```

成功标准：adapter 为 `ascend_vllm_0_23_0`、记录数大于零、请求有终止事件，
校验返回 `complete: true` 且 issues 为空。
“包可导入”“HTTP 200”“轨迹完整”“路由准确”是不同层次，前一项不自动证明后一项；
真实 NPU 路由准确性仍需硬件独立对照验收。

### 6. 查看开启采集后的结果

`ascend-details.json` 按输入 token / 层记录请求 ID、执行轮次、Prefill/Decode、
原始 token 位置和 Top-K 专家；启用配置时包含 token ID、权重。
`ascend-expert-counts.csv` 按阶段、层和专家统计被选择次数，保留零计数。

按轨迹中的实际轮次筛选（不保证第 3 轮必然是并发）：

```bash
moe-trace details "$TRACE_RUN" ./outputs/ascend-step.json --format json --step-id 3
moe-trace counts "$TRACE_RUN" ./outputs/ascend-step-counts.csv --step-id 3
```

同一次执行出现多个内部 request_id 才证明实际并发批处理；包含两种 phase 才证明混合阶段。
在同层/同阶段内比较各专家 count 占比，不能把选择次数当成耗时或 NPU 使用率，
也不能把逻辑专家 ID 当成物理设备号。输出字段、独立校验例子和分析口径见
[完整快速开始的结果分析](docs/guides/ascend-quickstart.md#7-导出结果并分析专家分布)。

### 本机 CPU 验证

无需昇腾卡、Docker、CANN 或模型权重。在已准备的独立 CPU 环境中运行：

```bash
conda activate moe-ascend-cpu
python scripts/validation/ascend_cpu.py --output-root ./outputs/ascend-cpu
```

输出 `details.json`、`consumer_reference.json`、`cpu-validation.json` 及原始
`traces/<run_id>/`；JSON 覆盖写，原始 UUID 目录保留。
环境创建和固定源码接入验证见[CPU 开发指南](docs/guides/ascend-cpu-development.md)，
已有证据见[CPU 验证报告](docs/validation/ascend-cpu-validation.md)。
普通 CPU 模拟不能代替真实 A3 路由核验或官方 Ascend 端到端测试。

## 仓库结构

```text
src/moe_trace/
├── adapters/       # 版本接入；补丁导入路径稳定
├── setup/          # 标准库检测、安装、隔离补丁
├── validation/     # 独立 GPU 参考探针与 Ascend CPU 验证
└── *.py            # 公共采集、存储、读取、分析和 CLI
scripts/
├── prepare_environment.py  # 用户无依赖入口
├── tools/          # 手动隔离补丁工具
├── dev/            # 补丁生成等维护脚本
└── validation/     # 真实请求和离线交叉核验
configs/
├── examples/       # 通用用户配置
└── validation/     # 历史实验配置
docs/
├── guides/         # 使用指南
├── design/         # 架构和数据格式
└── validation/     # 历史证据和当前验证索引
patches/vllm/       # 按版本组织补丁及基础摘要
patches/ascend/     # Ascend 实验补丁，CPU 已验证，A3 待验收
tests/              # 自动化回归，不代替真实 GPU 验收
```

[开发指南](docs/guides/development.md)说明模块边界、验证命令及旧路径迁移。
模型、wheel、环境、大型日志和轨迹不提交，输出保存在被忽略的 `outputs/` 或用户目录。

## 文档

- [文档导航](docs/README.md)
- [配置、字段、分析与 AISBench 使用](docs/guides/usage.md)
- [环境准备](docs/guides/environment-setup.md)
- [Ascend 快速开始：安装、服务、采集、验证和分析](docs/guides/ascend-quickstart.md)
- [两机 A3 V4.1 1P1D Decode 调试版快速开始（A3 待验收）](docs/guides/ascend-v41-debug-quickstart.md)
- [A3 新手快速使用指南：从准备环境到获取 JSON](docs/guides/A3新手快速使用指南.md)
- [公共采集与数据设计](docs/design/design.md)
- [需求和操作边界](docs/requirements.md)
- [验证结果与未完成项](docs/validation/README.md)

尚未单独声明项目许可证；vLLM、AISBench 和模型的许可由各自项目提供。
