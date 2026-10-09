# Qwen3.6 单机 A3 快速开始

适用目标：Qwen3.6-35B-A3B，单台 A3，8 卡 16 die，DP8/TP2，Prefill/Decode 混部，只采集 Decode。

**当前已实现 Qwen 专用采集 adapter、独立源码副本、调试启动器和 JSON/CSV 汇总，CPU 源码接入测试通过。**
真实 A3 尚未验收；支持文本、Ascend W8A8、V1/eager/同步、EP、无前缀缓存、无投机解码、
无 FlashComm/SP/多流/EPLB。不能直接在官方图/异步优化配置上采集，也不能用 DeepSeek 启动器代替。

## 1. 获取采集工具代码

本功能在 `feat/ascend-qwen36-035b-dp8-tp2` 分支，不能下载 main 代替。
本文编写时本次开发提交尚未推送远程。请先向维护者确认分支同步，或使用下面的源码包方式。

### 方式 A：维护者确认远程同步后，用 Git 下载

在 A3 容器的可写目录执行，例如 `/workspace`：

```bash
cd /workspace
git clone --branch feat/ascend-qwen36-035b-dp8-tp2 \
  https://github.com/zxTsai/moe-trace.git
cd moe-trace
git branch --show-current
git log -1 --oneline
```

应位于上述 feature 分支，并包含维护者提供的适配提交。
已有克隆不要覆盖；先保存本地改动，再 fetch 和切换对应分支。不执行强制重置。

### 方式 B：远程尚未同步，由维护者提供源码包

在有最新适配代码的开发机器执行：

```bash
cd /实际插件仓库
git branch --show-current
git log -1 --oneline
git archive --format=tar --output=/tmp/moe-trace-qwen36.tar HEAD
scp /tmp/moe-trace-qwen36.tar 用户名@A3机器IP:/实际接收目录/
```

确认打包的是 Qwen 适配分支。archive 仅包含已提交文件。
接收目录需要通过 Docker 挂载让容器可见。在容器的专用空目录解包：

```bash
mkdir -p /workspace/moe-trace-qwen36
cd /workspace/moe-trace-qwen36
tar -xf /容器可见的接收目录/moe-trace-qwen36.tar
```

不要解包到已有源码目录，以免覆盖用户文件。
后续所有命令均在所获取的**插件仓库根目录**运行。

## 2. 保留已跑通的 A3 推理环境

你已经运行成功的环境优先保留，不重新下载模型，不在生产环境升级依赖。
若尚未准备环境，先按[官方 Qwen3.6 教程](https://docs.vllm.ai/projects/ascend/zh-cn/v0.23.0/tutorials/models/Qwen3.6-35B-A3B.html)
准备 A3 驱动、容器、网络、模型及 vLLM-Ascend；安装官方环境不等于安装采集功能。

用户提供的当前环境基线：

| 项目 | 当前基线 |
| --- | --- |
| vLLM | `0.23.0+empty`，editable 根 `/vllm-workspace/vllm` |
| vLLM-Ascend | `0.23.0.post1`，editable 根 `/vllm-workspace/vllm-ascend` |
| Python | 环境路径 `/usr/local/python3.12.13` |
| 并行 | 单机 DP8/TP2，实际 EP、图和异步选项仍需启动命令确认 |

这是用户的环境信息，不是所有用户必须使用的绝对路径。
版本号不能证明 editable 源码未变更，脚本会查询实际导入位置和文件指纹。
不要在这里安装 CPU 版 PyTorch，也不要用 NVIDIA 环境向导或旧 Granite/DeepSeek 补丁。

在**实际服务所在的容器与 Python 环境**执行：

```bash
python -c "import sys; print(sys.executable)"
python -m pip show vllm vllm-ascend
npu-smi info
```

本轮环境检查不要求停止现有服务，也不会加载模型或修改路由。

## 3. 准备可持久化输出目录

推荐只挂载项目和结果目录，不挂载整个可写 `/home`。例如创建容器时增加：

```bash
-v /data/moe-trace:/workspace/moe-trace \
-v /data/moe-output:/trace-output \
```

以上是 Docker 参数片段，应放在完整 `docker run` 命令的镜像名之前。
前面是宿主机路径，后面是容器路径，两者访问同一份文件，不是复制。
现有容器若已有可写持久化目录，直接复用，不必为了检查报告重建或停掉服务。

例如容器写入 `/trace-output/report.json`，宿主机可在 `/data/moe-output/report.json` 找到。
下文使用 `/trace-output`，请按你的实际挂载路径替换。

## 4. 安装采集工具包

在仓库根目录执行，使用服务所用的 Python：

```bash
python -m pip install --no-deps -e .
python -m pip check
```

`--no-deps` 不自动安装或升级 vLLM/Ascend 依赖。缺少本项目的 NumPy/PyYAML 时，在隔离环境
按项目依赖声明补齐，不盲目升级整套软件栈。
本阶段的检查脚本只用 Python 标准库，也可先不安装工具包直接运行它。
安装工具包不代表采集 hook 已生效，还需要下面的源码检查、副本和重启步骤。

## 5. 自动查询实际包位置并生成报告

将模型目录填入变量，必须是含 `config.json` 的实际本地目录：

```bash
QWEN_MODEL=/实际Qwen3.6模型目录
QWEN_REPORT_DIR=$(mktemp -d /trace-output/qwen36-check.XXXXXX)

python scripts/tools/inspect_qwen36_ascend.py \
  --model "$QWEN_MODEL" \
  --output "$QWEN_REPORT_DIR/environment.json"
```

`/trace-output` 必须已存在并且当前用户可写。mktemp 创建本轮独立目录，不覆盖旧报告。
暂时不方便提供模型路径时，可去掉 `--model`，但报告将缺少模型配置信息。
普通用户不要使用 `--ascend-package` 覆盖自动检测，否则不能证明实际服务加载该路径。

脚本自动收集：

- 当前 Python 路径及 vLLM、vLLM-Ascend、Torch、TorchNPU 版本。
- Python import spec 指向的实际包目录，包括 editable 安装。
- 可识别的源码仓库 HEAD，以及六个 Ascend 关键文件的 SHA256。
- 模型 architectures、text_config 中的专家数、Top-K、层数和量化配置。

它不导入 NPU 软件包、不加载权重、不打补丁、不启动或停止服务。

## 6. 读懂检查结果

报告打印到终端，并保存在刚创建的目录中。查看路径：

```bash
printf '%s\n' "$QWEN_REPORT_DIR/environment.json"
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `packages.*.package_path` | 实际包目录，不一定是 pip show 的 Location |
| `ascend_sources.git_head` | 可识别的 Ascend 源码提交；没有 Git 信息时可能为 null |
| `ascend_sources.all_reference_files_match` | 是否匹配本次调查的官方 post1 文件 |
| `vllm_git_head` | 可识别的实际 vLLM 源码提交 |
| `model.architectures` | 实际模型架构，不能只凭目录名判断 |
| `runtime_supported:false` | inventory 脚本不判定运行时支持；需要实际启动和验证，不是报告出错 |
| `npu_verified:false` | 未验收真实 NPU 采集 |

参考 Ascend commit：`1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。
即使六个文件都匹配，也不证明整个安装、编译扩展、图执行或采集功能可用。
若不匹配，保留报告交给维护者，不重写锁文件来强行通过检查。

## 7. 将环境信息交给维护者

提供：

1. `environment.json` 报告。
2. 实际完整服务启动命令，包括 export 环境变量和全部 vllm 参数。
3. 若需要，启动错误/日志；分享前删除密钥和敏感数据。

重点确认 EP、量化、前缀缓存、图执行、异步调度、FlashComm、多流和 MTP。
这些会影响真实 token 行排列或 hook 执行，不是只知道 DP8/TP2 就足够。

## 8. 创建采集副本

在服务所用容器/Python 的插件仓库根目录运行；自动查找实际 Ascend 包，不原地修改它：

```bash
python scripts/tools/stage_qwen36_ascend.py --output-parent /trace-output/qwen36-overlays
```

检查固定源码摘要和接入锚点后，会打印 `qwen36-overlay-...` 目录。
源码不匹配就停止，提交环境报告给维护者，不强行忽略 hash。
本工具只检查固定 Ascend 文件和运行时契约，不证明 vLLM 二进制完整兼容。
V4.1/Granite 副本不能使用；必须是本命令生成的 Qwen 副本。

## 9. 用调试配置启动采集

先保留原服务的命令和日志，结束自己的测试请求，再正常停止原测试服务，确保设备/端口空闲。
不要终止其他用户或生产服务。混部不需要额外 P 节点或 Mooncake 代理。
先不加 `--execute` 查看实际命令：

```bash
QWEN_OVERLAY=/trace-output/qwen36-overlays/实际生成的目录
python scripts/deployment/qwen36_capture.py \
  --model "$QWEN_MODEL" --overlay "$QWEN_OVERLAY" \
  --output-root /trace-output/qwen36-traces --node-id A3 \
  --save-token-ids --save-expert-weights
```

检查后在同一条命令末尾加 `--execute` 才启动。
启动器使用当前 Python，设置 V1、DP8/TP2、本地 DP8、EP、Ascend 量化、eager 和同步，
关闭缓存/FlashComm/SP bypass/共享专家多流等优化。使用 4096 上下文、512 batch tokens、
每 DP 8 序列、0.8 显存预算作为小规模调试起点；这组参数尚未在你的 A3 实测。
默认只监听 `127.0.0.1:8000`，远程客户端需要显式指定 `--host` 并做好网络访问控制。
`--port` 可改端口；设备可见性应包含 16 个逻辑 device。

启动器打印 `Trace run: /trace-output/qwen36-traces/UUID`，记下完整路径。
等待服务 ready，若配置/架构/实际通信路径不符会明确报错，不声称采集成功。
仅支持 `Qwen3_5MoeForConditionalGeneration` 实际架构，不能由 Qwen 文件夹名推断兼容。
ID 和权重默认不保存，示例显式开启用于调试；分享轨迹时注意脱敏。

## 10. 发请求、停止、导出

另开同环境终端发送请求，或让 AISBench 直接访问该 API 服务：

```bash
curl --noproxy '*' --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"请解释专家混合模型。"}],"max_tokens":32,"stream":false}'
```

先测一条，再测试多个不同长度并发请求。等待请求完成后，在服务终端按一次 Ctrl+C，
等待服务正常退出并排空 writer。异常退出或 worker 未收尾的结果不能汇总为成功。
填入刚才打印的 UUID：

```bash
QWEN_RUN=/trace-output/qwen36-traces/实际UUID
python -m moe_trace.distributed.qwen_cli "$QWEN_RUN" \
  --output /trace-output/qwen36-details-实际UUID.json \
  --counts /trace-output/qwen36-counts-实际UUID.csv
```

导出文件必须不存在并位于原始 run 外。成功输出 `worker_traces_complete:true` 和路由数。
16 个 worker 都必须收尾，包括 idle worker；同时核对输入声明哈希、跨 TP 一致性、
每个 Decode token 的全部层覆盖、分片摘要、重复和越界。
这只证明已执行范围的轨迹完整，尚未采集请求完成/取消终止原因。

查看第一条：

```bash
python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["routes"])); print(json.dumps(d["routes"][0],ensure_ascii=False,indent=2))' /trace-output/qwen36-details-实际UUID.json
```

每条包含 owner/observer 的 DP/TP/EP 和逻辑 device，以及 row 的请求、引擎、执行、
输入位置、token ID、`target_decode`，原始层号、逻辑专家 ID、可选权重。
`global_request` 是 DP 命名空间内的内部请求标识，不是已验证的 API response ID。
当前 input token 的路由不是该 forward 新预测的 token 路由；最后输出 token 未再输入时没有下一轮路由。

各 worker 在 host 的 `UUID/dp-*-tp-*/trace/` 写分片，旁边有 `inputs.jsonl` 和摘要索引。
容器路径 `/trace-output` 对应前述宿主机 `/data/moe-output`；不自动上传到其他机器。
CSV 按层/逻辑专家计数，不把共享专家或 physical ID 混入 Top-K。
同步 D2H 和落盘会明显影响性能，不将此模式用于生产吞吐结论。

## 11. 在 CPU 复验接入功能

使用独立 CPU 开发环境（PyTorch CPU、pytest、插件源码），不安装 NPU 软件栈或模型权重。
下载固定 Ascend 源码用于提取 Python 接入点，不把它安装进 CPU 环境：

```bash
git clone --depth 1 --branch v0.23.0.post1 \
  https://github.com/vllm-project/vllm-ascend.git /你的开发目录/vllm-ascend-post1

python scripts/validation/qwen36_cpu.py \
  --ascend-source /你的开发目录/vllm-ascend-post1 \
  --output-root ./outputs/qwen36-cpu
```

已有相同源码目录时复用，不覆盖或重复下载。该脚本依赖本仓库测试文件。
验证实际 patched forward 区块、All2All.prepare 和完整 W8A8 MoE.apply；模型回调、
selector 和专家 kernel/分布式组是 CPU mock，不运行完整 vLLM 服务或真实 Qwen 权重。
输出独立目录、8 条路由的 `details.json`、16 次专家选择的 CSV 及 `cpu-validation.json`。
见[采集功能 CPU 验证报告](../../validation/Qwen3.6采集功能CPU验证.md)。

## 常见问题

| 问题 | 处理 |
| --- | --- |
| clone 提示没有该分支 | 确认远程同步，或向维护者索取源码包；不能改用 main |
| 找不到检查脚本 | 检查仓库根目录和提交是否包含 Qwen 适配 |
| 包信息是 null | 确认在服务所用容器/Python 中运行 |
| 没有源码 HEAD | 安装可能没有 Git 元数据；文件指纹仍可报告 |
| Permission denied | 检查挂载及当前用户权限，选专用可写目录 |
| 报告文件已存在 | 使用新目录；脚本拒绝覆盖旧文件 |
| 已安装插件但没有路由输出 | 确认加载 Qwen 副本、显式 --execute、请求实际有 Decode forward；停止后再汇总 |

进一步阅读：[适配设计](../../design/Qwen3.6单机A3采集适配设计.md)、
[环境检查验证记录](../../validation/Qwen3.6环境检查验证记录.md)、
[项目 README](../../../README.md)。
