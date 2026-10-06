# Ascend CPU 开发环境

目标分支：`feat/ascend-vllm-0.23.0`。

本环境仅用于 moe-trace 的 CPU 张量、模拟批次和文件校验测试，不是
vLLM Ascend 推理环境，也不代表已经通过 Ascend 官方单元测试或 A3 验收。
不需要模型权重、Docker、CANN 或 NPU。不得在现有生产 vLLM 环境执行以下安装。

## 准备独立环境

在仓库根目录执行：

```bash
conda create -n moe-ascend-cpu python=3.12 pip pysocks -y
conda activate moe-ascend-cpu
python -m pip install -e '.[test]' --no-build-isolation
python -m pip install 'torch==2.10.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip check
python -m pytest -o addopts='' -q
```

PyTorch 使用 CPU 版，不能据此验证 NPU 算子、NPU 同步或 Ascend 专家通信。
该环境刻意不安装 vLLM、vLLM Ascend 或 TorchNPU，避免提前引入设备后端。

若 pip 出现代理兼容错误，应先确认代理和网络；可仅对子进程清除代理变量后
重试，不应修改系统代理或其他环境的配置。

## 快速验证采集与导出

```bash
conda activate moe-ascend-cpu
python scripts/validation/ascend_cpu.py --output-root ./outputs/ascend-cpu
```

这个场景模拟三个不同长度请求及一个无 forward 的取消请求，含批次重排、
分块 Prefill、Decode 和同轮混合阶段。真实 CPU Tensor 的 Top-K 结果既传给
采集 hook，也传给独立模拟 consumer；验证器逐项对照保存结果与 consumer 入参。
场景中 softmax/Top-K 属于模拟模型，不属于采集器；采集器不重新计算路由。

输出包括：

- `traces/<run_id>/`：metadata、steps、requests、分片 NPZ 和完整状态。
- `details.json`：逐 token、逐层明细，每次覆盖该文件。
- `consumer_reference.json`：独立 consumer 入参，每次覆盖。
- `cpu-validation.json`：结果和真实运行目录，每次覆盖，明确标识 `npu_verified=false`。

原始 UUID 目录不覆盖，不会自动删除。无需再下载模型权重。

## 固定源码 Python 接入验证

这一步下载源码但**不安装** vLLM、TorchNPU、vLLM Ascend 或 CANN。
只验证经过哈希检查的真实 Python 接入点，框架/设备依赖使用 mock。

在仓库根目录执行，源码放在被 Git 忽略的 `outputs/`：

```bash
mkdir -p outputs/ascend-sources
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git \
  outputs/ascend-sources/vllm
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm-ascend.git \
  outputs/ascend-sources/vllm-ascend

python -S scripts/tools/stage_ascend_023.py --check \
  --vllm-package outputs/ascend-sources/vllm/vllm \
  --ascend-package outputs/ascend-sources/vllm-ascend/vllm_ascend

python -S scripts/tools/stage_ascend_023.py \
  --vllm-package outputs/ascend-sources/vllm/vllm \
  --ascend-package outputs/ascend-sources/vllm-ascend/vllm_ascend \
  --output-parent outputs/ascend-overlays
```

最后一条打印独立 overlay 路径。将其填入变量，不需要把 overlay 加入 CPU
测试进程的 PYTHONPATH（否则可能触发 Ascend 后端导入）：

```bash
ASCEND_CPU_OVERLAY=./outputs/ascend-overlays/实际生成的目录名
python scripts/validation/ascend_cpu.py --overlay "$ASCEND_CPU_OVERLAY" \
  --output-root ./outputs/ascend-cpu-source
MOE_ASCEND_SOURCE_OVERLAY="$ASCEND_CPU_OVERLAY" python -m pytest -o addopts='' -q
```

通过后报告应有 `source_hooks_executed=true` 和 `independent_comparison=passed`。
未设置环境变量时，完整测试中的固定源码接入测试会明确 skip，不能当作已通过。
验证从固定源码提取完整的 unquantized `apply`、`_model_forward`、scheduler
add/free/shutdown，以及 execute_model 中真实 forward 的 with 区块；不执行完整
输入准备、NPU 图编译或采样流水线，不是 Ascend 官方全量 UT。

如果重复执行 clone，已有目录会报错；复用原目录即可，不要重新下载或覆盖本地改动。
仅源码版本检查失败时调查差异，不能重生成 manifest 来掩盖不匹配。

## 后续 A3 验收边界

当前只提供实验 adapter 和隔离补丁，不自动安装 Ascend 软件栈。
NVIDIA 环境向导不接受 Ascend 模式。未来真实服务应使用匹配的官方 A3 环境，
显式设置 `VLLM_USE_V2_MODEL_RUNNER=0`、uni/eager、同步调度，关闭前缀缓存、
投机解码、EP/EPLB、FlashComm、多流和融合 MC2，不同时开启内置 routed-experts 返回。
初始模型为 GraniteMoeForCausalLM；真实 A3 支持性和 NPU 路由仍未验证。
其他 MoE 模型不能仅因用了 FusedMoE 就视为兼容。

实验配置：[ascend-vllm-0.23.0.yaml](../../configs/examples/ascend-vllm-0.23.0.yaml)，
默认关闭采集。更多边界见[设计文档](../design/ascend-vllm-0.23.0-design.md)和
[补丁说明](../../patches/ascend/0.23.0/README.md)。

## 后续验证层次

1. 本环境：现有工具回归；后续新增 Ascend adapter 的模拟请求、路由张量和输出测试。
2. CANN 开发容器：固定 vLLM 和 vLLM Ascend `v0.23.0` 源码，检查补丁和官方 CPU 单元测试。
3. A3 设备：真实模型推理、独立路由核验、并发请求和 AISBench 验收。

官方流程：<https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/contribution/testing.html>
（`latest` 内容可能变化，应在实际源码集成时重新核对。）
