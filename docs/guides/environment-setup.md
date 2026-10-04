# 安全准备采集环境

从完整源码仓库运行 `python scripts/prepare_environment.py`。安装工具后可用
`moe-trace-env`；若不在源码仓库中，指定 `--repository /path/to/moe-trace`。
需要 Linux/WSL2 x86_64、Conda（新环境模式）、Git、可用的 NVIDIA 驱动。
不会安装 Conda、驱动、CUDA Toolkit 或模型，不进行 C++/CUDA 源码编译。

## 检测与选择

```bash
python scripts/prepare_environment.py --detect
python scripts/prepare_environment.py --conda-env my-vllm --detect --gpu
```

默认检测当前 Python；`--python /path/to/python` 或 `--conda-env NAME` 指定目标。
检测会清除继承的 PYTHONPATH，避免把旧隔离补丁误当作原安装包。
普通检测不加载模型、不申请 GPU；`--gpu` 额外执行小规模 CUDA 运算和原生扩展导入。
JSON 中 `assessment.compatible` 表示版本/依赖/基础源码检查，不等于 GPU 或模型推理验收。

交互运行时，先选择新环境（默认推荐）或已有环境，再确认完整安装计划。
已有环境不兼容时停止，改用新环境，不自动升级或修补原环境。

## 非交互模式

```bash
python scripts/prepare_environment.py --mode new --version 0.29.0 \
  --name moe-trace-collection --dry-run
python scripts/prepare_environment.py --mode new --version 0.29.0 \
  --name moe-trace-collection --yes
python scripts/prepare_environment.py --conda-env my-vllm --mode existing --yes
```

`--dry-run` 只检测和显示计划。非交互安装必须同时提供 `--mode` 与 `--yes`。
默认继承调用终端的网络代理，新环境同时安装 PySocks 以支持 SOCKS 代理。
若已有环境缺少 SOCKS 支持，pip 可能报
`Missing dependencies for SOCKS support`；可以配置可用的 HTTP 代理，或在
允许直连时显式添加 `--no-proxy`，只对向导子进程关闭代理，不修改系统配置。
新环境也可用 `--prefix /new/environment/path`，不能与 `--name` 同用。
已有名称、已存在目录、base/root 均拒绝作为新环境目标，不覆盖已有环境。
已有环境只通过 `pip install --no-deps --no-build-isolation -e` 安装工具本身，
不会升级依赖，但会改变该环境的 moe-trace 安装；仍推荐专用环境。

## 支持配置与检查

| vLLM | 新环境 Python | PyTorch | Transformers |
| --- | --- | --- | --- |
| 0.10.1 | 3.11 | 2.7.1 | 4.55.2 |
| 0.29.0 | 3.12 | 2.13.0 | 5.18.0 |

以上是保守的已验证依赖配置，不代表其他组合一定不可用；不匹配时向导拒绝。
新环境只下载二进制 wheel，并执行依赖一致性检查、补丁基础 SHA256 检查、
CUDA/预编译扩展检查，以及隔离补丁和工具实际导入路径检查。
同版本 wheel 的源码仍可能不同；SHA256 不匹配时停止，不能绕过检查。
已被手动修补过的旧 vLLM 安装也可能不匹配；不会覆盖或撤销用户补丁。

`--skip-gpu-check` 仅用于离线准备，结果标记为 `prepared_gpu_unverified`，
不能据此宣称可推理。安装失败会保留新建的部分环境用于排查，不自动删除。

## 输出与启用

每次成功准备创建独立的 `outputs/environments/prepared-*` 目录，包含：

- `environment.json`：环境、执行计划、验证状态和限制。
- `trace.yaml`：可直接使用的配置，默认采集输出在 `outputs/traces`。
- `overlay/vllm`：补丁 Python 副本，原生扩展链接至目标环境。
- `activate-trace.sh`：当前终端的环境变量，不修改 shell 启动文件。

可用 `--output-dir` 和 `--trace-output` 改变输出位置。先激活目标环境，
再 source 提示的脚本；脚本检查 Python 路径，不允许在错误环境中加载补丁。
环境、源码仓库和输出目录必须保留在原位置，移动后需重新准备。
退出当前终端即不再加载隔离补丁。要在当前终端停止加载，先执行
`unset PYTHONPATH MOE_TRACE_CONFIG VLLM_USE_V2_MODEL_RUNNER VLLM_USE_FLASHINFER_SAMPLER`，
再重新激活原环境；若这些变量原本有值，应恢复原值。

向导不会启动/停止任何服务；请按 README 手动选择空闲 GPU、端口并启动模型，
验证请求、正常停机和采集完整性。Conda 只隔离依赖，不隔离 GPU、显存、端口。

## 本次验证边界

重构后已在现有 0.10.1 / 0.29.0 Python 环境分别通过 82 项自动化测试，其中安装
命令由测试替身模拟；另已通过现有 0.29.0 环境真实 CUDA 和扩展检测，
以及真实安装包的隔离补丁应用和实际导入路径检查，未修改原安装包。
完整新建 0.29.0 环境已实际尝试：首次缺少 SOCKS 支持的问题通过 Conda
安装 PySocks 修复；随后镜像下载因连接超时和重试耗尽失败，未进入完整安装
和 GPU 验证。临时环境及缓存保留，不自动重建或清理。
0.10.1 的新 wheel 在线安装未执行；模拟流程测试不等同于真实安装成功。
