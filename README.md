# moe-trace

面向 vLLM 的 MoE 专家路由采集工具，将实际模型执行、内部请求、输入 token
位置、模型层和 Top-K 专家关联起来，支持按执行轮次分析并发请求。

实现为独立工具包、版本化 adapter 和最小 Python 补丁。读取模型实际消费的
路由结果，不重新计算 softmax/Top-K，不依赖 `--enable-return-routed-experts`。

## 支持与状态

`main` 统一维护 **vLLM 0.10.1 和 0.29.0**，不必按版本切换 Git 分支；
两个版本仍需各自的环境、adapter 和补丁。

目前验证范围：Linux/WSL2 x86_64、单 NVIDIA GPU、Granite 3.1 1B A400M
Instruct、V1/uni/eager、TP/PP/DP=1、无前缀缓存和投机解码。
0.29.0 还要求 V1 model runner、关闭异步调度和正数停机超时。
其他模型、多卡、Ascend、CUDA Graph 等不在支持承诺内。

**实验版，尚未完成全部可靠性验收。** 双版本真实采集、AISBench 并发、
独立消费探针对照、取消、关闭采集、层/阶段过滤和导出已有验证。
整体长期内存仍会随历史索引增长；全新环境在线安装因网络中断未通过，不能据此用于生产。
当前结论见[验证索引](docs/validation/README.md)，历史报告不是最新验收状态。

## 快速开始

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

## 仓库结构

```text
src/moe_trace/
├── adapters/       # 版本接入；补丁导入路径稳定
├── setup/          # 标准库检测、安装、隔离补丁
├── validation/     # 可选独立 GPU 参考探针
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
tests/              # 自动化回归，不代替真实 GPU 验收
```

[开发指南](docs/guides/development.md)说明模块边界、验证命令及旧路径迁移。
模型、wheel、环境、大型日志和轨迹不提交，输出保存在被忽略的 `outputs/` 或用户目录。

## 文档

- [文档导航](docs/README.md)
- [配置、字段、分析与 AISBench 使用](docs/guides/usage.md)
- [环境准备](docs/guides/environment-setup.md)
- [公共采集与数据设计](docs/design/design.md)
- [需求和操作边界](docs/requirements.md)
- [验证结果与未完成项](docs/validation/README.md)

尚未单独声明项目许可证；vLLM、AISBench 和模型的许可由各自项目提供。
