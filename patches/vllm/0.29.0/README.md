# vLLM 0.29.0 集成

仅支持 GraniteMoeForCausalLM、单 GPU、同步 uni executor、eager、
无 prefix cache、无 speculative decoding。公共文件格式与 0.10.1 相同。
不修改推理输入或重算 Top-K；采集位置是 MoERunner 选择专家后、
RoutedExperts.forward_modular 消费前。
保存权重是该消费位置的实际权重，转存为 float32；Granite 的配置使用
`renormalize=True`，选中 Top-K 后重新归一化。工具不再次归一化或缩放。

## 安全应用（推荐隔离包）

无需重新安装 vLLM 或依赖。激活你自己的兼容 0.29.0 环境，在工具仓库根执行：

```bash
TRACE_REPO=$(pwd)
TRACE_MODEL=/absolute/path/to/granite-3.1-1b-a400m-instruct
cp configs/examples/vllm-0.29.0.yaml trace.yaml
TRACE_VLLM_PACKAGE=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
TRACE_VLLM_OVERLAY=$(python scripts/tools/stage_vllm_029.py "$TRACE_VLLM_PACKAGE")
```

脚本输出一个 `/tmp/moe-trace-vllm-029-...` 路径。它先核对
`base_files.json` 的目标源码与版本文件 SHA256，然后复制 Python 文件、
链接已有预编译扩展并检查和应用补丁。不会修改安装环境。源码有用户改动
或不匹配时拒绝应用。记录的上游标识 `g98dff2a81` 是 wheel 提供的短标识，
不是完整 Git commit；因此使用发行版本加文件摘要锁定基础。

任何检查失败都停止。编辑自己的 `trace.yaml` 指定输出目录，然后加载隔离包：

```bash
export PYTHONPATH="$TRACE_VLLM_OVERLAY:$TRACE_REPO/src"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export MOE_TRACE_CONFIG="$TRACE_REPO/trace.yaml"
python -c 'import vllm; print(vllm.__version__, vllm.__file__)'
vllm serve "$TRACE_MODEL" \
  --served-model-name granite-moe --host 127.0.0.1 --port 18001 \
  --dtype bfloat16 --max-model-len 1024 --max-num-seqs 2 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.55 \
  --enforce-eager --no-enable-prefix-caching --no-async-scheduling \
  --distributed-executor-backend uni --shutdown-timeout 30
```

通用示例默认输出到 `./outputs`，开启 token ID、不保存权重；
请按需要选择持久输出目录并调整可选字段。实验配置另放 `configs/validation/`。
配置里的 vllm 字段是约束声明，不能替代上述实际启动参数。

`--enable-return-routed-experts` 并非本工具必需；其与本工具同时开启的
组合尚需单独验证。采集当前不依赖 API 原生路由返回结果。
分块/混合阶段验证将 `--max-num-batched-tokens` 调成 64，并使用 8 个
并发槽；不代表生产必须采用这些参数。

必须给 EngineCore 留出排空缓冲的时间。默认 shutdown timeout 为 0
会立即强杀引擎；适配器拒绝该设置。30 秒仍不是强制保证，磁盘很慢或
异常强杀时可能不完整，必须检查 status.json。取消请求不等于运行失败。
补丁在 EngineCore 已无工作、退出忙循环前执行收尾，避免等到 finally
重置信号处理后才写文件；UniProcExecutor.shutdown 保留幂等兜底。

## 撤销与重新生成

停止自己启动的服务，移除本次环境中的临时包 PYTHONPATH 配置并重启，
即可回到原安装包。临时目录保留用于诊断，不需要改动安装环境。
若在自行管理的独立源码副本手动应用补丁，先执行 `git apply --check`；
撤销时先检查 `git apply --reverse --check`，再 `git apply --reverse`。
不得覆盖安装目录或用户已改动源码。

`scripts/dev/build_vllm_029_patch.py <原始 vllm 包目录>` 将补丁输出到 stdout，
用于复查与再生成，不自动写入安装包或仓库。
