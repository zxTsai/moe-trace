# DeepSeek V4：两台 A3 从零部署 MTP 专家路由采集

更新：2026-11-04。分支：`feature/ascend-deepseek-v4-mtp`。
[更新说明](../更新说明.md) · [设计](../design/DeepSeekV4的MTP采集设计.md) · [CPU 验证报告](../validation/DeepSeekV4的MTP采集CPU验证.md)

本指南从两台 A3 Linux 宿主机开始，不要求已经安装 vLLM、启动模型服务或安装插件。
依次完成宿主机确认、框架环境、模型、PD 基线、插件接入、请求和 JSON/CSV 验证。
有可用 Ascend 容器的用户可跳过镜像构建，但仍需源码检查。

**当前是可进入真机调测的开发版：CPU 验证通过，真实 A3、镜像构建和完整 PD 服务尚未验收。**
本文给出从零准备和调测流程，不承诺任意官方镜像安装本插件后即可采集。
驱动/固件需先由硬件环境管理员按官方安装说明准备；本插件不安装或升级驱动。

## 1. 适用范围与准备清单

| 项目 | 本文使用的配置 |
| --- | --- |
| 机器 | 两台 A3，每台 8 卡、16 die，独立调测时段 |
| P | DP4 / TP4，EP，MooncakeHybridConnector producer |
| D | DP16 / TP1，EP，MooncakeHybridConnector consumer |
| 模型 | 文本 DeepseekV4ForCausalLM，W8A8，含一个物理 MTP 层 |
| MTP | 配置候选数 K，按实际执行 K 采集；单物理 MTP 层重复执行 |
| 模式 | V1、同步 eager；D 关闭图、异步、多流、FlashComm、EPLB |
| 保存位置 | D 宿主机挂载存储；16 个 worker 分目录，本地汇总 |
| 当前限制 | 无 DSpark、PP/CP、多模态；位置被钳制为 dummy 时明确拒绝关联 |

两节点应有足够的 host RAM、共享内存和磁盘容纳模型、镜像、构建缓存及采集文件。
镜像与权重很大，先检查容量；CPU 测试机器无需下载这些资源。
IP、网卡、目录均由你填写，不依赖维护者的本机路径。

外部准备参考：[官方安装指南](https://docs.vllm.ai/projects/ascend/zh-cn/latest/getting_started/installation.html)、
[官方 V4 模型部署](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html)、
[Mooncake 多节点环境](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)。
这些 latest 页面会变化，本插件的关键源码参考固定为：

| 包 | 提交 |
| --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `0e181fc85fff24db10e5f92f10319c1913e6e9c5` |

不能将 v0.23.0、V4.1、Qwen 或最新版镜像自动视为匹配。
已有生产服务先保留原启动脚本；本指南不要求在其环境直接替换依赖。

## 2. 宿主机检查和获取代码（两节点都执行）

先由管理员按照宿主机型号/系统安装兼容的 Ascend 驱动、固件、Docker；
版本应匹配选用的 CANN 用户态栈。安装过程以官方指南为准，不在生产机盲目执行升级。
在两台宿主机的 Bash 终端检查：

```bash
npu-smi info
docker version
git --version
ip -br addr
ls /dev/davinci{0..15}
df -h
```

预期：驱动可查询，Docker daemon 可访问，16 个设备节点存在，识别出 P/D 各自的
通信网卡和 IP。宿主机 IP 可互通不等于 NPU 通信已配置，应按官方多节点指南验证
HCCL/设备互联及 Mooncake 所需网络。不要仅开放一个 HTTP 端口就开始两机推理。

准备自己的部署目录；下面只挂载项目目录，不挂载整个 home：

```bash
export DEPLOY_ROOT="$HOME/moe-trace-a3"
mkdir -p "$DEPLOY_ROOT"/{code,models,traces,env,build}
git clone --branch feature/ascend-deepseek-v4-mtp \
  https://github.com/zxTsai/moe-trace.git "$DEPLOY_ROOT/code/moe-trace"
git -C "$DEPLOY_ROOT/code/moe-trace" log -1 --oneline
```

已有克隆更新前先用 `git status --short` 检查并保存自己的修改，随后：

```bash
git -C "$DEPLOY_ROOT/code/moe-trace" fetch origin
git -C "$DEPLOY_ROOT/code/moe-trace" switch feature/ascend-deepseek-v4-mtp
git -C "$DEPLOY_ROOT/code/moe-trace" pull --ff-only
```

GitHub 不可达时，可将维护者提供的源码包解压到一个新的空目录，再使用该目录；
不能用 main 或旧 MTP 文件混搭。本仓库当前远程为 GitHub，未配置 GitCode 同步。

## 3. 准备 vLLM Ascend 环境

### 路径 A：已有可用镜像/容器

保留已有环境，不重装依赖。跳过镜像构建，继续第 4 节准备模型路径和公共参数，
再到第 5 节执行容器内检查，不必重复创建容器。
如果缺少挂载，容器运行后不能通过再次 docker exec 增加挂载，应使用已有共享路径，
或按自己的运维流程创建带正确挂载的新调试容器。

已有镜像可将下文 IMAGE 改为其名称/固定 digest。不要用“能够拉起 V4”替代第 6 节的源码检查。

### 路径 B：全新软件环境，构建固定源码镜像

在 A3 宿主机构建。使用[固定官方 Dockerfile.a3](https://github.com/vllm-project/vllm-ascend/blob/0e181fc85fff24db10e5f92f10319c1913e6e9c5/Dockerfile.a3)
的 `VLLM_COMMIT` 参数，避免只安装 pip latest 导致关键源码漂移：

```bash
export DEPLOY_ROOT="$HOME/moe-trace-a3"
git clone --filter=blob:none --no-checkout \
  https://github.com/vllm-project/vllm-ascend.git "$DEPLOY_ROOT/build/vllm-ascend"
git -C "$DEPLOY_ROOT/build/vllm-ascend" fetch origin 0e181fc85fff24db10e5f92f10319c1913e6e9c5
git -C "$DEPLOY_ROOT/build/vllm-ascend" checkout --detach 0e181fc85fff24db10e5f92f10319c1913e6e9c5
export IMAGE=moe-v4-mtp:ascend-0e181fc-vllm-ced6857

set -o pipefail
docker build -f "$DEPLOY_ROOT/build/vllm-ascend/Dockerfile.a3" \
  --build-arg VLLM_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607 \
  -t "$IMAGE" "$DEPLOY_ROOT/build/vllm-ascend" \
  2>&1 | tee "$DEPLOY_ROOT/env/image-build.log"
docker image inspect "$IMAGE" --format '{{.Id}}'
```

该官方配方的默认栈包含 CANN 9.1.0 A3、Mooncake NPU 包及 Torch/TorchNPU 等依赖，
由配方配套安装。本项目只核对关键 Python 源码，不锁定所有下载依赖的字节；
构建配方的参数已核对，**没有在本机完成 A3 镜像构建**。
若构建失败，保留日志与版本，先解决基础环境问题，不能继续标记部署成功。
不要把本项目的 CPU 验证 Conda 环境装到 A3 服务环境。

两节点必须使用相同构建结果。可以在两端分别构建；也可在 P 上导出后传到 D：

```bash
docker save --output "$DEPLOY_ROOT/env/moe-v4-mtp-image.tar" "$IMAGE"
```

把该 tar 复制到 D 的自有目录后，在 D 宿主机运行：

```bash
docker load --input "$DEPLOY_ROOT/env/moe-v4-mtp-image.tar"
export IMAGE=moe-v4-mtp:ascend-0e181fc-vllm-ced6857
docker image inspect "$IMAGE" --format '{{.Id}}'
```

核对两端 image ID 一致。也可使用经环境管理员确认的官方 A3 镜像，
但没有已认证的官方 tag/digest 可直接替代本插件的源码检查。

## 4. 准备权重和公共参数（两节点）

使用[官方教程给出的 W8A8 MTP 权重](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp)。
已有完整权重直接复用，无需重新下载；可复制到下列 models 目录或调整挂载。
首次下载可复用镜像里的 ModelScope SDK；下载容器不运行推理，不需要 NPU 设备：

```bash
docker run --rm -i --network host \
  -v "$DEPLOY_ROOT/models:/models" "$IMAGE" python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download(
    model_id="Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
    local_dir="/models/DeepSeek-V4-Flash-w8a8-mtp",
))
PY
```

`local_dir` 的含义见 [ModelScope 下载接口](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py)。
两节点可分别下载，或将已下载完整目录复制/共享给另一节点，模型文件应一致。
只有 config.json 不算下载完成；应保留权重分片索引和全部被引用的 safetensors 文件。

在两节点宿主机各保存同一份参数文件。**192.0.2.* 是文档示例地址，YOUR_* 是占位符，必须修改。**
用各自实际网卡名填写 P_NIC/D_NIC；容器网络采用 host 模式。

```bash
cat > "$DEPLOY_ROOT/env/deploy.env" <<'ENV'
MODEL=/models/DeepSeek-V4-Flash-w8a8-mtp
TRACE_ROOT=/trace-output
MTP_K=3
P_IP=192.0.2.10
D_IP=192.0.2.11
P_NIC=YOUR_P_NIC
D_NIC=YOUR_D_NIC
PROXY_IP=192.0.2.11
ENV
```

默认端口：P API 7100–7103，D API 7100–7115，两端 DP RPC 12321，
P/D connector 30000/30100，proxy 1999。还需按所用 HCCL/Mooncake 版本开放其派生/
握手/传输所需网络端口；以上不是完整防火墙白名单。

## 5. 创建可复用容器并验证环境

在两节点宿主机分别执行；路径与前面的 DEPLOY_ROOT/IMAGE 保持一致。
本段 docker run 仅用于新建容器；路径 A 用户进入已有容器后，直接执行下文容器内检查，
并按已有挂载调整项目路径和 deploy.env 位置。
示例按官方 A3 设备挂载方式展开 16 die，驱动路径不同则按实际安装位置修改。
`--shm-size=512g` 是示例共享内存容量上限，应与宿主机资源和官方模型要求匹配。

```bash
docker run -it --name moe-v4-mtp-debug \
  --network host --shm-size=512g --privileged=true \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci8 --device /dev/davinci9 \
  --device /dev/davinci10 --device /dev/davinci11 \
  --device /dev/davinci12 --device /dev/davinci13 \
  --device /dev/davinci14 --device /dev/davinci15 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v "$DEPLOY_ROOT/code/moe-trace:/workspace/moe-trace" \
  -v "$DEPLOY_ROOT/models:/models:ro" \
  -v "$DEPLOY_ROOT/traces:/trace-output" \
  -v "$DEPLOY_ROOT/env:/deploy-env:ro" \
  -w /workspace/moe-trace "$IMAGE" bash
```

挂载中冒号左边是 host，右边是容器；例如 `/trace-output` 写入本节点 host 的
`$DEPLOY_ROOT/traces`。模型/公共参数只读，项目和采集目录可写。
这里没有 `--rm`，退出后容器保留：停止状态使用 `docker start -ai moe-v4-mtp-debug`，
运行状态另开终端使用 `docker exec -it moe-v4-mtp-debug bash`，不要重复 docker run 同名容器。

以下均在容器内执行，每个新终端都先加载公共参数：

以下 `python` 必须指向该镜像的服务解释器；若环境只有 `python3`，统一替换为 `python3`，
不要混用另一个 Conda/Python 环境。

```bash
cd /workspace/moe-trace
set -a
source /deploy-env/deploy.env
set +a
mkdir -p "$TRACE_ROOT/logs"
npu-smi info
python - <<'PY'
import torch
import torch_npu
import vllm
import vllm_ascend
from mooncake.engine import TransferEngine
assert torch.npu.is_available()
assert torch.npu.device_count() == 16, torch.npu.device_count()
value = torch.ones(1, device="npu:0").cpu().item()
assert value == 1
print("NPU/framework/Mooncake imports: OK")
PY
```

再检查模型完整性和输出读写：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["MODEL"])
cfg = json.loads((root / "config.json").read_text())
assert cfg["architectures"] == ["DeepseekV4ForCausalLM"], cfg.get("architectures")
assert cfg.get("num_nextn_predict_layers", 1) == 1
index = root / "model.safetensors.index.json"
if index.is_file():
    shards = set(json.loads(index.read_text())["weight_map"].values())
    missing = [name for name in shards if not (root / name).is_file()]
    assert not missing, missing
else:
    assert (root / "model.safetensors").is_file(), "missing safetensors/index"
print("Model files: OK")
PY
probe=$(mktemp "$TRACE_ROOT/write-check.XXXXXX")
printf 'ok\n' > "$probe"
cat "$probe"
rm "$probe"
```

这些检查不证明 Mooncake 已成功建链；PD 请求成功才是基线验收条件。

## 6. 检查插件兼容性（P/D 都执行）

先运行标准库检查器，不要求提前安装本插件：

```bash
python scripts/tools/inspect_ascend_v4_mtp.py --model "$MODEL" \
  > "$TRACE_ROOT/environment-check.json"
cat "$TRACE_ROOT/environment-check.json"
```

必须退出码为 0、`source_check_passed:true`。检查自动定位当前解释器加载的包，
记录框架版本、Git 提交、模型架构和关键源码 hash，不假定包一定在 /vllm-workspace。
失败时保留 differences 并交给维护者适配；不跳过校验，不为匹配而替换可用生产环境。
P/D 框架也应一致。通过后继续。

## 7. 启动不采集的 MTP PD 基线

本节只运行原框架；还未安装采集 hook。确认本次调测所用设备、端口空闲，
由你按运维流程停止冲突的旧服务，保留原脚本。不要并行启动两套占满 16 die 的 D 服务。

P 容器，前台启动：

```bash
python scripts/deployment/ascend_v4_mtp_1p1d.py \
  --role P --model "$MODEL" --local-ip "$P_IP" --nic "$P_NIC" \
  --num-speculative-tokens "$MTP_K" --execute
```

D 容器另一个终端，前台启动：

```bash
python scripts/deployment/ascend_v4_mtp_1p1d.py \
  --role D --model "$MODEL" --local-ip "$D_IP" --nic "$D_NIC" \
  --num-speculative-tokens "$MTP_K" --execute
```

如需先审阅生成参数，去掉 `--execute` 即 dry-run，不启动服务。
启动器保留 P4×4、D16×1、EP、Mooncake，显式使用同步 eager 与配置 K；
D 关闭图、异步、多流等未支持优化。首次加载需等待全部引擎 ready，加载时间不是插件开销。

在 D 容器另开终端，加载 deploy.env 后检查全部 20 个 API：

```bash
python - <<'PY'
import os
from urllib.request import urlopen
for name, count in (("P_IP", 4), ("D_IP", 16)):
    for port in range(7100, 7100 + count):
        address = f'http://{os.environ[name]}:{port}/health'
        with urlopen(address, timeout=30) as response:
            assert response.status == 200, address
        print("ready", address)
PY
```

全部 ready 后启动 PD proxy。自动定位当前 Ascend 源码旁的官方代理脚本，
不依赖维护者本机路径；镜像仅含 wheel、没有 examples 时会明确报错，
可在容器中先下载同一固定提交的脚本，再重试下面的代理命令：

```bash
# 仅在安装包没有 examples 时使用
curl -fL 'https://raw.githubusercontent.com/vllm-project/vllm-ascend/0e181fc85fff24db10e5f92f10319c1913e6e9c5/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py' \
  -o "$TRACE_ROOT/load_balance_proxy_server_example.py"
```

```bash
python - <<'PY'
import importlib.util, os, pathlib, subprocess, sys
package = pathlib.Path(importlib.util.find_spec("vllm_ascend").origin).resolve().parent
proxy = package.parent / "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
if not proxy.is_file():
    proxy = pathlib.Path(os.environ["TRACE_ROOT"]) / "load_balance_proxy_server_example.py"
assert proxy.is_file(), f"official proxy script missing: {proxy}"
argv = [sys.executable, str(proxy), "--host", "0.0.0.0", "--port", "1999",
        "--prefiller-hosts", *([os.environ["P_IP"]] * 4),
        "--prefiller-ports", *map(str, range(7100, 7104)),
        "--decoder-hosts", *([os.environ["D_IP"]] * 16),
        "--decoder-ports", *map(str, range(7100, 7116))]
print("PD proxy:", proxy, flush=True)
raise SystemExit(subprocess.call(argv))
PY
```

此终端保持运行。另开终端、加载 deploy.env 后经代理发送请求：

```bash
curl --fail-with-body "http://$PROXY_IP:1999/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Explain expert routing briefly."}],"max_tokens":32,"stream":false}'
```

预期 HTTP 200，返回 completion choices；只有端口可访问或文件存在不算基线通过。
若这里失败，先排查权重、框架、网卡、设备通信或 Mooncake，不归因于未加载的采集 hook。

## 8. 安装插件并创建 D 采集副本

基线通过后，在 D 启动器终端 Ctrl+C，等其管理的服务退出。P 与 proxy 可继续运行。
在 D 容器的独立终端加载 deploy.env，使用与服务相同的 Python 安装：

```bash
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -c "import moe_trace; print(moe_trace.__file__)"
python scripts/tools/stage_ascend_v4_mtp.py --check
python scripts/tools/stage_ascend_v4_mtp.py --output-parent "$TRACE_ROOT/overlays"
```

pip 只安装本工具，不升级 vLLM/TorchNPU。构建需要 setuptools>=68，运行需要
NumPy>=1.26/PyYAML>=6；若缺依赖，先确认解释器、再在独立调测环境补齐。
`pip check` 发现基础镜像自身冲突时保留报告，不盲目升级所有依赖。
复制工具只修改独立 vllm_ascend 副本中的五个 Python 文件，不更改原安装源码。

记下最后打印的目录，例如 `/trace-output/overlays/ascend-v4-mtp-XXXXXXXX`。
为后续新终端保存路径（填实际值）：

```bash
printf '%s\n' '/trace-output/overlays/ascend-v4-mtp-XXXXXXXX' > "$TRACE_ROOT/overlay-path.txt"
export OVERLAY=$(cat "$TRACE_ROOT/overlay-path.txt")
```

每次更新插件/补丁后重新 stage，不能沿用旧版 overlay。
overlay 的路径、内容摘要和实际导入位置由启动器再次检查。

## 9. 开启采集、发送并发请求并正常结束

D 容器，在新终端加载 deploy.env，然后：

```bash
export OVERLAY=$(cat "$TRACE_ROOT/overlay-path.txt")
python scripts/deployment/ascend_v4_mtp_1p1d.py \
  --role D --model "$MODEL" --local-ip "$D_IP" --nic "$D_NIC" \
  --num-speculative-tokens "$MTP_K" \
  --trace --overlay "$OVERLAY" --output-root "$TRACE_ROOT/runs" \
  --node-id D0 --execute
```

启动器打印 `Trace run: .../runs/UUID`，记下本轮路径。
等 D 全部 ready、经 proxy 单请求正常返回后，在另一个 D 容器终端运行：

```bash
python scripts/validation/deepseek_v4_requests.py \
  --url "http://$PROXY_IP:1999/v1/chat/completions" --model dsv4
```

预期三个请求均 HTTP 200。它是功能请求器，不是性能压测；
如需 AISBench，也指向同一个 proxy。客户端并发不保证同一个执行 batch，需看结果分组。
请求结束后在 D 启动器终端 Ctrl+C，等待全部 16 worker 正常收尾，随后再离线汇总。
同步 D2H 和写盘会影响性能，本版不用于生产吞吐结论。

要保留启动日志，可在启动命令末尾增加 `> "$TRACE_ROOT/logs/D-trace.log" 2>&1`，
前台仍用 Ctrl+C 停止，在另一终端 `tail -f "$TRACE_ROOT/logs/D-trace.log"` 查看打印的 run 路径。
需要 token ID/权重时，启动增加 `--save-token-ids --save-expert-weights`；默认均不保存。

## 10. 校验和导出结果

D 容器将本次 UUID 填为启动器实际打印的路径，不能用 overlay 目录代替：

```bash
export RUN_DIR="$TRACE_ROOT/runs/实际UUID"
export RUN_ID=$(basename "$RUN_DIR")
python -m moe_trace.distributed.v4_mtp_cli "$RUN_DIR" \
  --output "$TRACE_ROOT/details-$RUN_ID.json" \
  --counts "$TRACE_ROOT/expert-counts-$RUN_ID.csv" \
  --by-execution "$TRACE_ROOT/execution-counts-$RUN_ID.csv"
```

必须退出码 0、`worker_traces_complete:true`，并出现 target/mtp 两类非空路由。
完整性失败时不能手工改 complete，也不能为凑齐候选补一个拒绝结果。
UUID 每轮独立；导出路径必须是原 run 外的新文件，不能覆盖已有文件。

用标准库检查 JSON 与 CSV 的合计一致：

```bash
python - <<'PY'
import csv, json, os
from collections import Counter
from pathlib import Path
root = Path(os.environ["TRACE_ROOT"])
run = os.environ["RUN_ID"]
details = json.loads((root / f"details-{run}.json").read_text())
totals = Counter()
for route in details["routes"]:
    totals[route["model_scope"]] += len(route["expert_ids"])
assert totals["target"] > 0 and totals["mtp"] > 0
assert totals["target"] == details["expert_choices"]["target"]
assert totals["mtp"] == details["expert_choices"]["mtp"]
assert sum(totals.values()) == details["expert_choices"]["combined"]
with (root / f"expert-counts-{run}.csv").open() as stream:
    assert sum(int(row["choices"]) for row in csv.DictReader(stream)) == sum(totals.values())
print("JSON/CSV totals: OK", dict(details["expert_choices"]))
print("candidate states:", dict(Counter(c["state"] for c in details["candidates"])))
print("npu_verified:", details["npu_verified"])
PY
```

这是安装后软件采集闭环通过的检查，不是独立 NPU 路由正确性验收。
样例 K=3 的 CPU 数据为 26 条路由、target=40/mtp=12/combined=52；
真机无需等于这些数字。`npu_verified:false` 是当前版本固定验收标记。

## 11. 三类路由与候选的关系

| 字段 | 语义 |
| --- | --- |
| `row.kind=target_decode` | 无候选时主模型的普通 Decode 输入 |
| `row.kind=target_verify`，无 candidate ID | 主模型验证调用中的 seed 输入；隐私开启时 input_role 为 redacted |
| `row.kind=target_verify`，有 candidate ID | 实际被主模型验证的候选输入，关联接受/拒绝 |
| `row.kind=mtp_draft` | MTP 模型实际计算的输入，可含多行上下文；不是该次预测出的 token |
| `model_scope` | `target` 或 `mtp`，同号层不会混合 |
| `source_engine, execution, input_row` | D 引擎、真实执行、实际输入行；不同 DP 的 execution 不代表同一全局时间 |
| `candidate_round, candidate_node` | 候选轮次和节点；每轮每请求 node=0..实际K-1 |
| `candidate_accepted` | 候选验证路由的 sampler 决定；false 的路由仍完整保留 |

MTP 的实际 input_ids 比 target 左移一位，但送入 kernel 的 positions 沿用原 target
坐标。**MTP 路由的 position 保存实际 kernel 坐标**；其输入 token 的序列位置为
position+1，预测候选位置为被采样输入 position+2。输入声明和 candidates 明确记录此关系。
不要把 MTP 输入 token ID 和预测输出 token ID 当成同一个值。

JSON 的 `candidates` 单独保存 request/source_engine、round/node、producer_execution、
producer_input_row、producer_kernel_position、候选位置、可选 token ID、验证 execution 和状态。
状态为 accepted/rejected/not_adopted/unverified；未被调度验证的末轮候选、取消后的候选不被误标为拒绝。
同 token ID 在不同轮次出现时仍按候选轮次和位置关联。

决定直接来自 rejection sampler 的实际输出及 SpecDecodeMetadata，**不是根据客户端输出
长度猜测**。实际有效 sampler 输出数减一是接受前缀长度，还会逐个核对前缀 token ID。
首个未接受候选标 rejected，其后标 not_adopted（后续未采用），已经发生的路由都保留。
`candidate_accepted:false` 表示未接受，细分原因查看 `candidate_decision`。它不等于经过 EOS/stop/长度裁剪后最终返回客户端的状态。

## 12. 负载统计怎么看

JSON 的 `expert_choices` 给出：

```json
{"target":24,"mtp":10,"combined":34}
```

上面是原 K=1 CPU mock 样例：17 条 token/层路由，主模型 24 次选择、MTP 10 次，合计 34 次；
1 个候选接受、1 个拒绝、2 个末轮未验证。被拒绝候选仍贡献主模型的 4 次专家选择。
它不是 DeepSeek 真模型的负载结论，真实数字取决于请求与执行。

专家 CSV 按 scope、执行类型、层、决定、专家计数，包含零负载专家；按执行 CSV 保留
引擎和执行轮次。合计是两模型执行选择次数相加，不把两模型第 0 层合并成同一个层，
也不推断物理设备利用率、共享专家耗时或 KV 流量。

原始目录每 worker 包含：

```text
dp-N-tp-0/
├── trace/                 # target/mtp 声明、路由和 CandidateResult 分片、SHA256、状态
├── inputs.jsonl           # 实际输入声明，带 scope/kind/候选关联
├── inputs-index.json
├── candidates.jsonl       # 真实预测输出与 producer 输入关联
└── candidates-index.json
```

汇总检查全部 16 worker、身份、哈希、每输入×层、候选生产→验证→结果。
已验证候选缺结果会失败；未验证产物导出 accepted=null/state=unverified。
`worker_traces_complete:true` 仅表示保存记录满足声明的完整性约束。
`npu_verified:false` 保留，真实设备路由正确性和完整 PD 服务仍待现场验收。
没有 P/proxy 全局请求映射或请求完成/取消原因采集。

K=3 CPU 样例另得到 26 条路由，target=40、mtp=12、combined=52；
两个请求各接受第 0 个候选，第 1 个拒绝，第 2 个未采用。单个物理 MTP 层实际执行三次，
每次独立执行 ID。不是按 K 虚增计数，而是读取三次执行的真实返回。

## 13. 正常停止、恢复和再次更新

1. 等待客户端请求结束，先在 D 采集启动器终端 Ctrl+C，等待 16 worker 收尾。
2. 导出校验成功后，停止自己启动的 proxy 和 P；不向其他用户的服务发送停止信号。
3. 恢复已有服务时，用干净终端执行保存的原启动脚本，不继承 overlay 的 PYTHONPATH
   或 MOE_TRACE 配置。源码副本不改变原安装包。
4. 保留本轮 run、environment-check、模型/镜像版本及日志。采集目录不自动覆盖或删除。
5. 更新插件时先停止本次采集服务，再对源码 `git pull --ff-only`，重新运行 source check、
   stage 和基线验证。新补丁不能使用旧 overlay。

只安装但不再采集时，原服务可继续使用原源码；需要卸载时在同一解释器运行
`python -m pip uninstall moe-trace`，先确认没有本插件启动的服务仍在使用它。
无需删除模型、重装 vLLM 或清理其他容器。

可按下表判断做到哪一步：

| 阶段 | 必须看到的证据 | 能说明什么 |
| --- | --- | --- |
| 环境 | 16 die 可见、基础 NPU 运算成功、Mooncake 可导入 | 基础软件/设备入口可用 |
| 源码 | source_check_passed=true | 与锁定 Python 接入点匹配 |
| 不采集 PD | 全部 API ready、代理请求正常返回文本 | 该调试配置的基础服务可用 |
| 开启采集 | 经代理请求成功，16 worker 正常结束 | 服务确实执行采集入口 |
| 文件闭环 | worker_traces_complete=true，target/mtp 非空，JSON/CSV 合计一致 | 软件采集和保存约束满足 |
| 设备正确性 | 与独立真实设备参考路由逐条核对 | 尚需现场完成，不能由 CPU 测试替代 |

## 14. 常见失败与 CPU 复验

- 源码不匹配：保留 MTP environment-check.json，提供真实提交适配，不要强行修改摘要。
- 候选与 producer 不一致：保留 D 日志、原始 run、请求与调度配置；不按 token ID 强行凑关联。
- MTP query/sample index 不匹配：当前源码布局不符合首版假设，检查是否有图、异步、CP、额外 slot 或不同提交。
- 验证结果缺失/异常退出：不会补一个“拒绝”来凑完整；保留失败数据排查。
- 开销：同步拷贝和写盘用于调测，不据此比较原 MTP 服务吞吐；短时、小请求先行。

CPU 开发环境中可重复执行：

```bash
python scripts/validation/deepseek_v4_mtp_cpu.py \
  --vllm-source /YOUR_REFERENCE/vllm \
  --ascend-source /YOUR_REFERENCE/vllm-ascend \
  --output-root outputs/v4-mtp-cpu --k 3
```

不要在 A3 服务环境安装 CPU PyTorch。具体方法、mock 边界与结果见
[MTP CPU 验证记录](../validation/DeepSeekV4的MTP采集CPU验证.md)。
