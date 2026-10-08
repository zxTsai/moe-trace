# A3 新手快速使用指南：两机 1P1D 专家路由采集

本文面向第一次部署的用户，目标是：**普通请求成功 → 开启采集 → 正常停止 → 得到非空 JSON**。
当前是实验开发版，CPU 测试已通过，尚未完成 A3 真机验收，不能保证现场一次启动成功。
本次仅支持 DeepSeek V4.1、两台 A3、D 端 Decode 调试配置；关闭 DSpark/MTP、图执行和异步调度。
不是任意版本 vLLM-Ascend 安装后 `pip install` 即可用的通用插件。

## 先理解在哪里执行

```text
客户端 / AISBench → PD 代理 → P 节点：处理提示词
                         → D 节点：生成 token，采集专家路由
                                       ↓ 同步拷贝到 CPU
                                  D 节点 host 文件系统
```

| 位置 | 用途 | 执行什么 |
| --- | --- | --- |
| 开发机器 | 提供当前插件源码 | 打包、传输源码 |
| P 宿主机 | 管理容器和设备 | 检查驱动、创建容器、准备模型 |
| P 容器 | Prefill 服务，DP4/TP4 | 安装工具、启动 P，另开终端运行代理 |
| D 宿主机 | 管理容器和保存结果 | 检查驱动、挂载输出目录 |
| D 容器 | Decode 服务，DP8/TP2 | 创建采集副本、启动 D、校验和导出 |

每台 A3 暴露 16 个逻辑 device。P/D 各使用本机全部 16 个；文件不保存在 device 显存中。
除特别说明外，后续 Python 命令在**相应容器内的插件仓库根目录**运行。

## 1. 准备两台 A3、网络和模型

需要已有管理员准备好：A3 驱动、Docker、通信网卡和两机网络、可用端口、兼容的
DeepSeek V4.1 Ascend W8A8 模型。模型需放在两机相同绝对路径。
不要将普通 DeepSeek 权重或其他模型当作本版已支持的模型。

按[官方 V4.1 教程](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4.1-Flash.html)
的第 3 节准备权重并验证多节点通信，第 4.1 节选择 **A3** 容器命令。
官方 latest 可能更新；镜像名和软件版本号本身不能证明与本插件兼容。
如果没有驱动/网络配置权限，请管理员先完成这些步骤，不要在生产环境盲目升级依赖。

在两台**宿主机**检查：

```bash
npu-smi info
ip -br addr
```

记下服务 IP 和对应通信网卡；不要使用 Docker bridge 地址或随意选择回环地址。
需要代理能访问 P 的 7100–7103 和 D 的 7100–7107；DP RPC 默认 12321。
Mooncake 及 HCCL 还需要额外端口，不能仅放行 HTTP 端口；按官方网络指南配置。

## 2. 创建容器并挂载源码、结果

先在两台宿主机准备示例目录：

```bash
mkdir -p /data/moe-trace /data/moe-traces
```

执行官方第 4.1 节 A3 Docker 命令，保留其设备、驱动、网络、IPC 和模型挂载。
在镜像名之前额外加上：

```bash
-v /data/moe-trace:/workspace/moe-trace \
-v /data/moe-traces:/trace-output \
```

Docker 创建属于管理员操作。示例使用 host 网络/IPC、设备和特权设置，必须用于受信任的
隔离调试环境，不与生产服务争用 device 或端口。不要只复制上面两行当作完整 Docker 命令。

| 宿主机 | 容器内 | 内容 |
| --- | --- | --- |
| `/data/moe-trace` | `/workspace/moe-trace` | 插件源码 |
| `/data/moe-traces` | `/trace-output` | 副本、原始轨迹、导出 JSON |

路径可更换，但后续命令必须一起调整；宿主机目录要有写权限和足够磁盘空间。
因为做了挂载，退出或删除容器不会自动删除这里的结果。

在两台**容器内**检查推理环境：

```bash
python -c "import torch,torch_npu; print('NPU available:',torch.npu.is_available()); print('Devices:',torch.npu.device_count())"
python -m pip show vllm vllm-ascend torch torch-npu
```

NPU 应可用；没有限制设备可见性时应看到 16 个 device。
**不要安装 CPU 版 PyTorch，也不要使用本项目 CPU 调测 Conda 环境来启动 A3 服务。**

### 必须通过的源码匹配关卡

本版按以下固定源码开发：

| 包 | 提交 |
| --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `0e181fc85fff24db10e5f92f10319c1913e6e9c5` |

官方镜像未必包含相同源码。第 4 步哈希检查失败时停止，请维护者提供匹配环境或适配
现有镜像。不能重写 hash manifest、强行打补丁或只 checkout Python 文件掩盖编译扩展差异。
本文不提供已验证的固定镜像 digest，也不承诺从源码构建在现场已通过。

## 3. 获取并安装当前采集工具

当前开发分支：`feat/ascend-deepseek-v41-decode`，不是 main。
本文编写时该功能尚未推送；在维护者确认同步之前，不要假设 GitHub 的 main/分支包含它。

可在有当前仓库的**开发机器**打包最新提交：

```bash
cd /实际插件仓库
git branch --show-current
git log -1 --oneline
# 确认是包含本功能的提交；archive 只打包已提交文件。
git archive --format=tar --output=/tmp/moe-trace-debug.tar HEAD
scp /tmp/moe-trace-debug.tar 用户名@P节点IP:/data/moe-trace/
scp /tmp/moe-trace-debug.tar 用户名@D节点IP:/data/moe-trace/
```

用户名、IP 是占位符，必须替换。目标源码目录应是专用空目录，避免覆盖已有用户文件。
两台**容器内**分别执行：

```bash
cd /workspace/moe-trace
tar -xf moe-trace-debug.tar
python -m pip install --no-deps -e .
python -m pip check
```

`--no-deps` 不安装/升级 vLLM。若确实缺少本插件的 `numpy` 或 `PyYAML`，请在这个隔离
环境按项目依赖声明补齐；不要一键升级整个推理软件栈。
维护者确认远程同步后，也可 clone 对应 feature 分支，再核对提交与本文版本。

## 4. 只在 D 容器创建采集源码副本

查询实际安装目录：

```bash
python -c "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('vllm').origin).parent); print(pathlib.Path(importlib.util.find_spec('vllm_ascend').origin).parent)"
```

将输出的第一行和第二行分别填到变量中，不能保留下面的占位符：

```bash
VLLM_PACKAGE=/第一行的实际目录
ASCEND_PACKAGE=/第二行的实际目录

python scripts/tools/stage_ascend_v41_debug.py \
  --vllm-package "$VLLM_PACKAGE" \
  --ascend-package "$ASCEND_PACKAGE" --check
```

看到 `source_check: passed` 后，创建副本：

```bash
python scripts/tools/stage_ascend_v41_debug.py \
  --vllm-package "$VLLM_PACKAGE" \
  --ascend-package "$ASCEND_PACKAGE" \
  --output-parent /trace-output/overlays
```

记下打印的 `/trace-output/overlays/ascend-v41-debug-...` 路径。
只修改副本，不修改原安装。不要把旧 0.23 或 CPU hook 副本作为本版 OVERLAY。

## 5. 先跑通不采集的调试基线

打开 P 和 D 各一个容器终端，填写各自模型目录、服务 IP 和通信网卡。
默认只打印计划，不启动；确认后在命令末尾加 `--execute`。

**P 容器终端：**

```bash
cd /workspace/moe-trace
python scripts/deployment/ascend_v41_1p1d_debug.py \
  --role P --model /实际模型目录 \
  --local-ip P节点IP --nic P节点网卡
```

**D 容器终端：**

```bash
cd /workspace/moe-trace
python scripts/deployment/ascend_v41_1p1d_debug.py \
  --role D --model /实际模型目录 \
  --local-ip D节点IP --nic D节点网卡
```

分别加 `--execute` 并运行后，保持两个终端打开，等待引擎 ready。权重加载可能很慢。
使用干净终端，不继承其他采集配置或手动加入的采集副本 PYTHONPATH。
Mooncake 动态库等环境需先按官方指南配置，本启动器不安装这些依赖。

### 在第三个容器终端启动 PD 代理

代理将请求连接到 P/D；可放在 P 节点。在匹配的 Ascend **源码仓库**中定位下面的脚本。
这个 examples 文件不一定随 site-packages 安装，不能把包目录直接当作源码仓库。

```bash
python /实际Ascend源码仓库/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 1999 \
  --prefiller-hosts P节点IP P节点IP P节点IP P节点IP \
  --prefiller-ports 7100 7101 7102 7103 \
  --decoder-hosts D节点IP D节点IP D节点IP D节点IP D节点IP D节点IP D节点IP D节点IP \
  --decoder-ports 7100 7101 7102 7103 7104 7105 7106 7107
```

在能访问代理的第四个终端发请求：

```bash
curl --noproxy '*' --fail http://代理机器IP:1999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","messages":[{"role":"user","content":"请简要介绍专家混合模型。"}],"max_tokens":32,"stream":false}'
```

必须先成功返回生成文本。直接向 D 发普通请求不等于验证 PD。
若失败，先解决模型/环境/网络/代理问题，不继续开启采集。

## 6. 重启 D，开启采集

请求结束后，在 D 启动器终端按一次 Ctrl+C，等待本轮 D 服务完全退出。P 可保持运行。
启动器只管理自己创建的进程组；等待 30 秒后仍不退出会 TERM/KILL，强制停止的轨迹
不能当作完整。不要执行全机 `pkill` 或清理别人的服务。

在 D 容器填写第 4 步生成的副本路径，并启动：

```bash
OVERLAY=/trace-output/overlays/实际生成的副本目录

python scripts/deployment/ascend_v41_1p1d_debug.py \
  --role D --model /实际模型目录 \
  --local-ip D节点IP --nic D节点网卡 \
  --trace --overlay "$OVERLAY" \
  --output-root /trace-output/traces --node-id D0 \
  --save-token-ids --execute
```

`--save-token-ids` 便于第一次核查，但 token ID 可能包含输入信息，分享轨迹前注意隐私。
默认不保存 ID 或权重；需要权重再加 `--save-expert-weights`。
不同时开启内置 `--enable-return-routed-experts` 或其他采集工具。

启动时会打印 `Trace run: /trace-output/traces/本轮UUID`，记下完整路径。
等待 D 就绪后再次通过代理发第 5 步的请求。先测一条，再增加并发或用 AISBench 指向代理。
同步拷贝会影响吞吐，本模式不用于生产性能测量。

## 7. 正常停止并导出 JSON

等待请求完成后，在 D 服务终端按 Ctrl+C，等待服务退出和缓冲排空。
在 D 容器中填写实际 UUID：

```bash
TRACE_RUN=/trace-output/traces/实际UUID
TRACE_JSON=/trace-output/merged-实际UUID.json

python -m moe_trace.distributed.debug_cli "$TRACE_RUN" --output "$TRACE_JSON"
```

输出位置必须在原始 run 目录外且尚不存在。每轮用新文件名，不覆盖之前结果。
成功时显示 `worker_traces_complete:true`、`logical_routes` 和 `expert_choices`。
如果缺少任意 worker、未收尾、failed、哈希错误或全部为空，汇总失败。
成功只说明声明的 worker 轨迹文件通过检查，不代表 NPU 路由正确性或每个请求生命周期已验收。

查看数量和第一条记录：

```bash
python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("Routes:",len(d["routes"])); print(json.dumps(d["routes"][0],ensure_ascii=False,indent=2))' "$TRACE_JSON"
```

JSON 中 `routes` 的每条记录包含：

- `row`：D 本地请求、引擎、执行轮次、输入行、序列位置、`target_decode` 标签、可选 token ID。
- `layer` / `expert_ids` / `weights`：原始模型层、逻辑专家选择和可选权重。
- `owner` / `observer`：node、DP、TP、EP 和逻辑 device 信息。

按 `(row.source_engine, row.execution)` 看同一引擎同次执行的请求。
按 `(layer, expert_id)` 分别计数专家选择；不同引擎的 step 不是全机同步轮次。
保存的是当前 forward 的输入 token 路由，不是刚预测的输出 token 路由。
最后输出 token 未再次输入模型时，不会有下一次 Decode 路由。

### 在宿主机哪里找结果

```text
D 容器：/trace-output/merged-UUID.json
D 宿主机：/data/moe-traces/merged-UUID.json

原始目录：/data/moe-traces/traces/UUID/
├── deployment.json
├── debug-config.json
├── dp-0-tp-0/trace/worker-00000/events-000000.jsonl
├── dp-0-tp-1/trace/...
└── ...直到 dp-7-tp-1/
```

16 个 worker 分别写 host 子目录，包括没有处理请求的空 worker；不是 16 份 device 内存文件。
当前不自动上传到其他主机。`consumer-debug.jsonl` 是映射后 ID 的辅助诊断，未纳入完整性验收。

## 常见问题和下一步

| 现象 | 应先检查 |
| --- | --- |
| NPU 不可用或设备不足 | 宿主机驱动、容器设备挂载、设备可见性 |
| source hash mismatch | 固定源码不匹配；停止，让维护者检查，不能强制忽略 |
| 找不到代理脚本 | 需要匹配的 Ascend 源码仓库 examples，不只是安装包 |
| 普通请求失败 | 服务 ready、模型、Mooncake 库、网卡、端口和代理配置 |
| 插件导入失败 | 服务与安装插件使用的 Python 是否相同，安装是否成功 |
| 缺少 worker / incomplete | D 初始化或退出日志、16 worker 是否创建并正常收尾 |
| JSON 已存在 | 换一个新的导出文件名；不要覆盖或改动原始分片 |
| 想恢复普通服务 | 停止采集服务，不传 `--trace`，不用副本和采集配置，启动原服务 |

保留报错、启动日志、包版本和 `deployment.json`，向维护者反馈；分享数据前脱敏。
当前没有 P/proxy 到 D 的全局请求映射，没有请求取消/完成原因记录，暂不支持四机、
DSpark/MTP、异步或图模式。真机跑通后再做独立路由对照及并发、取消、失败验收。

进一步参考：[精简操作指南](ascend-v41-debug-quickstart.md)、
[设计](../design/ascend-v41-debug-design.md)、
[CPU 验证报告与 A3 待验收项](../validation/ascend-v41-debug-validation.md)。
