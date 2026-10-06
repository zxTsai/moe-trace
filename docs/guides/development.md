# 开发与目录边界

## 模块职责

公共采集接口 `TraceSession`、`TraceReader`、配置、schema、分析及 CLI 保持原位置。
版本 adapter 只负责从真实 runner/路由消费链映射输入与结果；补丁仍导入
`moe_trace.adapters.vllm_0_10_1` 或 `vllm_0_29_0`，不能混用。
`setup/` 是标准库环境管理，bootstrap 不导入需要 NumPy/PyYAML 的包顶层。
`validation/` 放独立参考探针和 CPU 模拟验证器。GPU 探针通过
`MOE_TRACE_REFERENCE_DIR` 显式启用；Ascend CPU 验证使用独立脚本，常规采集不加载。
不要用模拟/探针数据代替真实采集器数据。
Ascend 开发见[CPU 验证指南](ascend-cpu-development.md)，独立 adapter 为
`ascend_vllm_0_23_0`，补丁同时涉及两个 Python 包。

## 本地回归

在已有依赖的 Python >=3.11 环境中，从仓库根执行：

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src tests scripts
python -S scripts/prepare_environment.py --help
python scripts/prepare_environment.py --detect
```

测试还检查文档本地链接、配置解析和无依赖/兼容入口。pytest 是开发依赖，
不要为运行测试升级生产 vLLM、PyTorch 或 CUDA。
完整服务验证必须另外检查空闲 GPU/端口，按版本启用隔离补丁和 eager 等约束。

## 真实请求和核验

先启动你自己的测试服务；用实际 served-model-name 和端口替换下面变量：

```bash
TRACE_URL=http://127.0.0.1:8000/v1/completions
TRACE_MODEL_NAME=granite-moe
python scripts/validation/reference_requests.py \
  --url "$TRACE_URL" --model "$TRACE_MODEL_NAME" --output /tmp/reference-client.json
python scripts/validation/lifecycle_requests.py \
  --url "$TRACE_URL" --model "$TRACE_MODEL_NAME" --output /tmp/lifecycle-client.json
```

如需独立逐元素验证，在启动服务前设置专用空目录
`MOE_TRACE_REFERENCE_DIR=/absolute/path/to/new-reference-directory`。
停止服务、等待排空后执行（用你的实际路径替换变量）：

```bash
PYTHONPATH=src python scripts/validation/verify_reference.py \
  "$TRACE_RUN" "$TRACE_REFERENCE_DIR/reference.jsonl" --output /tmp/reference-result.json
PYTHONPATH=src python scripts/validation/verify_lifecycle.py \
  "$TRACE_RUN" /tmp/lifecycle-client.json --output /tmp/lifecycle-result.json
PYTHONPATH=src python scripts/validation/verify_exports.py "$TRACE_RUN"
```

最后一项要求先通过 `moe-trace details --format json` 和 `moe-trace counts`
生成运行目录中的 `details.json`、`expert_counts.csv`。
只保留原始运行证据，不把未收尾或失败运行改成 complete。

## 旧路径迁移

| 原路径 | 当前路径 |
| --- | --- |
| `src/moe_trace/environment.py` 实现 | `src/moe_trace/setup/manager.py`；原模块保留 CLI 转发 |
| `src/moe_trace/environment_probe.py` | `src/moe_trace/setup/probe.py`；原脚本保留转发 |
| `adapters/verification*.py` | `src/moe_trace/validation/vllm_*.py` |
| `scripts/stage_vllm_029.py` | `scripts/tools/stage_vllm_029.py`；旧入口保留 |
| `scripts/build_vllm_029_patch.py` | `scripts/dev/build_vllm_029_patch.py` |
| 请求与 `verify_*.py` 脚本 | `scripts/validation/` |
| `configs/default.yaml` | `configs/examples/default.yaml` |
| `configs/granite-trace.yaml` | `configs/examples/vllm-0.10.1.yaml` |
| `configs/example-vllm-0.29.0.yaml` | `configs/examples/vllm-0.29.0.yaml` |
| 实验配置 | `configs/validation/` |
| `docs/environment-setup.md` | `docs/guides/environment-setup.md` |
| 设计文档 | `docs/design/`；需求仍在 `docs/requirements.md` |
| 历史报告及 JSON 摘要 | `docs/validation/` |

运行中任务可继续使用旧探针入口；不会重启、清理或重新下载其环境。
历史报告的实验事实不随目录整理重写；命令中的仓库路径引用已更新。
提交前确认没有模型、wheel、私有配置或大型轨迹；普通推送，不强制覆盖远程。
