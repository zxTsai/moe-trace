# Qwen3.6 单机 A3 快速开始

适用目标：Qwen3.6-35B-A3B，单台 A3，8 卡 16 die，DP8/TP2，Prefill/Decode 混部，只采集 Decode。

**当前版本只完成环境检查，Qwen 专用采集 adapter、补丁、启动和汇总尚未完成。**
本文可执行到生成环境报告，不是“安装后即可采集”的教程，也不能使用 DeepSeek 启动器代替。
后续实现完成后会补齐采集操作。CPU 已完成回归，A3 真机采集未验收。

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
**安装工具包不代表采集 hook 已生效；本阶段还没有 Qwen 可用补丁。**

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
| `runtime_supported:false` | 当前尚未完成 Qwen 采集运行时；不是报告出错 |
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

## 8. 后续采集流程：尚未开放执行

计划顺序：源码检查 → 独立 Qwen 补丁副本 → 专用加载启动 → 代理/API 发请求或 AISBench →
各 worker 保存 Decode 路由 → 正常收尾 → 严格校验并导出 JSON。
当前是普通单机 P/D 混部，不要求使用 DeepSeek 两机 Mooncake PD 代理。
具体启动与导出命令将在运行时实现并验证后补充，本文不提供占位命令冒充可运行功能。

未来文件将保存到 host 可见的输出目录，device 路由张量拷贝到 CPU 后再写盘。
预期关联请求、实际输入 token 位置、引擎/执行轮次、原始模型层及逻辑专家 ID；
不能把共享专家或物理设备号混入逻辑 Top-K 统计。

## 常见问题

| 问题 | 处理 |
| --- | --- |
| clone 提示没有该分支 | 确认远程同步，或向维护者索取源码包；不能改用 main |
| 找不到检查脚本 | 检查仓库根目录和提交是否包含 Qwen 适配 |
| 包信息是 null | 确认在服务所用容器/Python 中运行 |
| 没有源码 HEAD | 安装可能没有 Git 元数据；文件指纹仍可报告 |
| Permission denied | 检查挂载及当前用户权限，选专用可写目录 |
| 报告文件已存在 | 使用新目录；脚本拒绝覆盖旧文件 |
| 已安装插件但没有路由输出 | 当前只有检查功能，Qwen 采集尚未实现 |

进一步阅读：[适配设计](../../design/Qwen3.6单机A3采集适配设计.md)、
[环境检查验证记录](../../validation/Qwen3.6环境检查验证记录.md)、
[项目 README](../../../README.md)。
