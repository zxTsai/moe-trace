# Ascend 快速开始：环境、服务、采集和结果分析

## 先确认适用范围

**当前是实验接入流程，不是 A3 已验收的部署教程。** CPU 模拟和固定源码 Python
hook 已通过；NPU 算子、真实模型和真实并发采集尚未验收。当前 adapter 仅接受
`GraniteMoeForCausalLM`、非量化、V1/uni/eager、同步、单逻辑设备。不能直接换成
Qwen/DeepSeek，不能用多卡配置绕过检查。

官方 v0.23.0 模型矩阵没有列出 Granite，因此这里不保证它能在 A3 运行。
若原服务不能跑通该模型，先停止此流程，记录环境和错误，完成模型/后端适配后再继续。
官方参考：[安装指南](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html)、
[模型矩阵](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0/docs/source/user_guide/support_matrix/supported_models.md)。

本功能位于 `feat/ascend-vllm-0.23.0`，尚未合入 main。请使用维护者提供的该分支代码；
若远程还没有这个分支，默认 `git clone` 获得的 main 不包含此功能，不能照下面命令假装已安装。
当前环境向导不支持 Ascend；不要在生产服务环境或 NVIDIA 专用环境中执行本流程。

流程：官方环境 → 原服务请求成功 → 停止原测试服务 → 安装工具/生成副本 →
启动采集服务 → 发请求 → 停机排空 → 校验/导出/分析。

## 1. 安装官方 A3 环境

**以下 Docker 步骤只适用于实际 A3 主机，不适用于只有 NVIDIA GPU 的开发机。**
先按官方要求准备宿主机驱动/固件和 Docker，确认 `npu-smi info` 正常。
本工具不安装或升级驱动。官方固定镜像已包含用户态依赖，优先使用它，不混装其他版本。

在包含本 feature 代码的仓库根目录执行。`MODEL_HOST` 替换为你的模型目录；目录内
应已有模型权重，避免重复下载。设备节点示例需要按主机实际情况核对，A3 使用双 DIE。

```bash
npu-smi info
MODEL_HOST=/absolute/path/to/granite-3.1-1b-a400m-instruct
ASCEND_IMAGE=quay.io/ascend/vllm-ascend:v0.23.0-a3
docker pull "$ASCEND_IMAGE"
docker run --name moe-trace-ascend-023 -it --shm-size=1g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$PWD:/workspace/moe-trace" -v "$MODEL_HOST:/models/granite-moe:ro" \
  -p 127.0.0.1:8000:8000 "$ASCEND_IMAGE" bash
```

确认模型/仓库路径存在，勿因路径错误让 Docker 自动创建一个空模型目录。
已有专用容器可直接复用，不要重复创建同名容器。以下安装、服务、校验和分析命令
都在容器内执行；请求命令可从宿主机的第二个终端执行。

```bash
cd /workspace/moe-trace
npu-smi info
python - <<'PY'
import importlib.metadata as metadata
import torch
import torch_npu
for name in ("vllm", "vllm-ascend", "torch", "torch-npu"):
    print(name, metadata.version(name))
assert metadata.version("vllm") == "0.23.0"
assert metadata.version("vllm-ascend") == "0.23.0"
assert torch.npu.is_available(), "NPU is not available"
print("Ascend environment: OK")
PY
```

版本或设备检查失败就先修复官方环境，不进入插件安装。
已按官网完成环境安装的用户可跳过 Docker 创建，保留版本与设备检查。

## 2. 先验证未安装采集工具的原服务

在容器服务终端设置 Bash 数组，后续重启复用同一套参数，避免两次服务配置不同。
`--gpu-memory-utilization` 在 NPU 平台仍使用这个参数名；按空闲设备预算调整。
下列是**待 A3 实测的 Granite 参数**，不是通过实机验收的命令。

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
TRACE_MODEL=/models/granite-moe
ASCEND_SERVE_ARGS=(
  "$TRACE_MODEL" --served-model-name granite-moe --host 0.0.0.0 --port 8000
  --dtype bfloat16 --max-model-len 1024 --max-num-seqs 4
  --max-num-batched-tokens 64 --gpu-memory-utilization 0.55
  --enforce-eager --no-enable-prefix-caching --no-async-scheduling
  --distributed-executor-backend uni --shutdown-timeout 30
  --additional-config '{"enable_flashcomm1":false,"enable_flashcomm2_parallel_size":0,"enable_fused_mc2":0,"enable_balance_scheduling":false,"multistream_overlap_gate":false,"multistream_overlap_shared_expert":false,"eplb_config":{"dynamic_eplb":false}}'
)
vllm serve "${ASCEND_SERVE_ARGS[@]}"
```

这是干净环境的原服务：没有设置 `MOE_TRACE_CONFIG` 或 overlay。
宿主机第二个终端在服务就绪后发送请求：

```bash
curl --noproxy '*' --fail http://127.0.0.1:8000/v1/models
curl --noproxy '*' --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"granite-moe","prompt":"Explain expert routing.","max_tokens":8,"temperature":0}'
```

成功标准：HTTP 200，返回非空的生成结果，而不只是容器/端口存在。
如果失败，保存服务错误，不继续假定采集工具可用。
成功后在**你自己启动的测试服务终端**按一次 Ctrl+C，等待服务退出并释放端口。
不要停止已有生产服务。采集工具不能热注入原进程，后面必须重新启动。

## 3. 安装工具，验证包安装成功

回到同一个容器的服务终端，仓库根目录执行：

```bash
python -m pip install .
python -m pip check
moe-trace --help
python -c 'import moe_trace; from moe_trace.adapters import ascend_vllm_0_23_0; print(moe_trace.__file__); print(ascend_vllm_0_23_0.__file__)'
```

成功标准：依赖一致、命令可用、Ascend adapter 可导入。这只证明包安装成功，
不证明服务已加载 hook，更不证明真实路由准确。
安装声明依赖 NumPy/PyYAML，不安装 vLLM、TorchNPU、CANN；检查现有依赖约束，
不要为消除错误盲目升级官方软件栈。

## 4. 创建隔离副本并确认加载路径

以下命令根据当前环境自动定位原包。路径定位失败先调查，不手动猜目录。

```bash
VLLM_PACKAGE=$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm").origin).parent)')
ASCEND_PACKAGE=$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm_ascend").origin).parent)')
python scripts/tools/stage_ascend_023.py --check \
  --vllm-package "$VLLM_PACKAGE" --ascend-package "$ASCEND_PACKAGE"
python scripts/tools/stage_ascend_023.py \
  --vllm-package "$VLLM_PACKAGE" --ascend-package "$ASCEND_PACKAGE" \
  --output-parent ./outputs/ascend-overlays
```

脚本创建唯一目录并打印路径，不修改原包。即使版本号正确，厂商修改/不同构建导致
接入文件 SHA256 不一致也会被拒绝；这时先完成对应源码适配，不更新 manifest 来绕过检查。

将打印的实际目录填入变量。在首次加载前复制配置，启用采集、token ID 和权重：

```bash
export ASCEND_TRACE_OVERLAY=/workspace/moe-trace/outputs/ascend-overlays/实际生成的目录
cp configs/examples/ascend-vllm-0.23.0.yaml ./outputs/trace-ascend.yaml
```

编辑 `outputs/trace-ascend.yaml` 的 trace 部分：

```yaml
trace:
  enabled: true
  output_root: ./outputs/ascend-traces
  save_token_ids: true
  save_expert_weights: true
  buffer_records: 8192
  shard_records: 4096
```

保留原有 vllm 约束段。配置路径不会展开 `~` 或环境变量。

```bash
export PYTHONPATH="$ASCEND_TRACE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export MOE_TRACE_CONFIG="$PWD/outputs/trace-ascend.yaml"
python - <<'PY'
import importlib.util
import os
from pathlib import Path
overlay = Path(os.environ["ASCEND_TRACE_OVERLAY"]).resolve()
for name in ("vllm", "vllm_ascend"):
    path = Path(importlib.util.find_spec(name).origin).resolve()
    print(name, path)
    assert path.is_relative_to(overlay), f"{name} did not load the overlay"
PY
```

两个路径必须都在生成的副本下；一个指向副本、另一个指向原包时不得继续。
后续校验还会检查 metadata 的 adapter，不以导入路径作为唯一证据。

## 5. 重启采集服务并发送请求

在保存了 Bash 数组、加载变量和配置的同一终端执行：

```bash
vllm serve "${ASCEND_SERVE_ARGS[@]}"
```

不要同时开启 `--enable-return-routed-experts`。服务仍使用单逻辑设备，不因容器挂载
两个 A3 DIE 节点就自动启用 TP/EP。如果终端已换，应重新设置完整启动参数与变量。

就绪后宿主机另一个终端可以先复用第 2 步 curl，确认 HTTP 200。
若要并发/不同长度请求，在宿主机第二个终端进入容器：

```bash
docker exec -it moe-trace-ascend-023 bash
```

进入容器后：

```bash
cd /workspace/moe-trace
python scripts/validation/reference_requests.py \
  --url http://127.0.0.1:8000/v1/completions --model granite-moe \
  --output ./outputs/ascend-client.json
```

该脚本发送三个错峰请求及一个长提示词请求，不重新计算专家路由。
它不是 Ascend 独立消费探针；并发和混合阶段实际是否发生取决于执行时间，必须检查轨迹。
`ascend-client.json` 的 API ID 不得未经验证就当成内部 request_id。

等待所有请求成功完成，再在服务终端按一次 Ctrl+C，等待正常退出和落盘。
正在运行时的 incomplete 状态不等于安装失败；必须区分未关闭与失败。

## 6. 确认 hook 生效和采集完整

原始轨迹位于 `outputs/ascend-traces/<run_id>/`。从本次运行的目录中选择实际 ID，
不要自动选择目录里的历史成功运行。对原始数据执行：

```bash
TRACE_RUN=./outputs/ascend-traces/本次实际运行ID
moe-trace validate "$TRACE_RUN"
```

成功时返回 `{"complete": true, "issues": []}`，退出码为 0。还应检查：

- metadata：`framework.adapter` 为 `ascend_vllm_0_23_0`，两个包版本为 0.23.0。
- status：complete=true，record_count>0，step_count>0，无 errors，缓冲已排空。
- requests：本次真实内部请求有 started 与 finished/cancelled。
- steps/NPZ：请求、输入位置和每个配置层的专家记录能够关联。

如果没有目录/记录，检查是否启用配置、两个包是否来自 overlay，以及是否发出了真实请求。
进程异常退出、缺层或未终止请求不能手动修改为 complete。
**结构完整性验证不等于真实 NPU 路由准确性验收**；metadata 中
`framework.validation_level = "npu_unverified"` 表明
当前项目尚未完成硬件级独立对照。

## 7. 导出结果并分析专家分布

```bash
moe-trace details "$TRACE_RUN" ./outputs/ascend-details.json --format json
moe-trace counts "$TRACE_RUN" ./outputs/ascend-expert-counts.csv
```

JSON/CSV 覆盖指定导出文件，不覆盖原始 UUID 轨迹目录。
JSON 顶层有 storage_complete、validation_issues 和 records；一条记录是某个
request 的实际输入 token 在某个 MoE 层的 Top-K，包含 step_id/execution_id、
request_id、phase、token_position、layer_id、expert_ids，以及开启的 token_id/权重。

使用**轨迹中的实际值**，而不是 API ID，筛选轮次和请求：

```bash
TRACE_STEP=3
TRACE_REQUEST=从轨迹复制的内部请求ID
moe-trace details "$TRACE_RUN" ./outputs/ascend-step.json --format json --step-id "$TRACE_STEP"
moe-trace counts "$TRACE_RUN" ./outputs/ascend-step-counts.csv --step-id "$TRACE_STEP"
moe-trace details "$TRACE_RUN" ./outputs/ascend-request.json --format json --request-id "$TRACE_REQUEST"
```

分析口径：

- 同一 execution_id 中不同 request_id 表示该次 forward 实际处理多个请求。
- 同次执行含 prefill/decode 两种 phase，才证明混合执行发生。
- counts.csv 按 `(phase, layer_id, expert_id)` 统计选择次数，包含未被选择的专家零计数。
  同层同阶段各专家 count 之和应等于该范围的 token 路由条数乘 Top-K。
- 用同层、同阶段的计数占比比较负载，不把不同层同号专家合成一个物理专家。
- Top-K 选择次数不是执行耗时、通信字节、NPU 使用率；逻辑专家 ID 不是物理设备号。
- Prefill 输入提示 token，Decode 输入之前生成的 token。最后生成但未再次输入模型的
  token 没有下一条路由，不能用输出长度直接推算 Decode 记录数。

CPU 已校验的第 3 轮例子（不是承诺真实请求也会出现同一结果）：

| 请求 | 阶段 / 位置 / token ID | 层 0 Top-K | 层 1 Top-K |
| --- | --- | --- | --- |
| A | Decode / 4 / 15 | [3, 1] | [1, 2] |
| C | Prefill / 0 / 31 | [1, 2] | [2, 0] |
| B | Decode / 2 / 23 | [2, 0] | [3, 1] |

此例共 6 条路由、12 个专家选择；层 0 的 count 为 E0=1、E1=2、E2=2、E3=1，
这是合并两个阶段的演示数，实际 counts.csv 将分别显示 Prefill 和 Decode。

## 无 A3 时如何先获得已验证的结果

按[CPU 环境指南](ascend-cpu-development.md)准备独立环境，运行：

```bash
conda activate moe-ascend-cpu
python scripts/validation/ascend_cpu.py --output-root ./outputs/ascend-cpu
```

成功报告应为 state=passed、independent_comparison=passed、npu_verified=false；
5 次执行、30 条路由、60 个专家选择。结果可用上述 validate/details/counts 工具分析。
没有提供 overlay 时只验证模拟逻辑；固定源码 hook 验证需按 CPU 指南额外准备源码。
该流程不需要服务、Docker、NPU 或模型权重，不能用它给 A3 服务签发验收结论。

更多边界：[设计](../design/ascend-vllm-0.23.0-design.md)、
[CPU 验证报告](../validation/ascend-cpu-validation.md)、
[补丁及恢复原环境](../../patches/ascend/0.23.0/README.md)。
