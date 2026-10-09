# Qwen3.6 单机 A3 DP8/TP2 采集适配设计

分支：`feat/ascend-qwen36-035b-dp8-tp2`，起点为 Ascend 开发分支的 `e65cb08`。
状态：源码调查与环境检查阶段；尚未实现/验证本场景的采集运行时。
不修改 main，不改写用户已安装源码，不将 V4.1 支持声明扩展到 Qwen。

## 用户提供的环境

- 单台 A3，8 卡 16 die，TP2/DP8，同一引擎处理 Prefill/Decode，目标只保存 Decode。
- Python 路径 `/usr/local/python3.12.13`。
- vLLM `0.23.0+empty`，editable 根 `/vllm-workspace/vllm`。
- vLLM-Ascend `0.23.0.post1`，editable 根 `/vllm-workspace/vllm-ascend`。
- 尚缺实际启动命令、两仓库 HEAD 和模型 config 架构，不能把 editable 版本号视为源码保证。

参考[官方 Qwen3.6 教程](https://docs.vllm.ai/projects/ascend/zh-cn/v0.23.0/tutorials/models/Qwen3.6-35B-A3B.html)。
页面 A3 示例开启 EP、前缀缓存、FULL_DECODE_ONLY、FlashComm1、多流和异步。
用户只确认了 DP/TP 修改，不能假定其他选项已经关闭。

## 源码调查结论

独立下载官方 tag `v0.23.0.post1`，实际 commit 为
`1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`；annotated tag 对象 ID 不是源码 commit。
其 `.github/vllm-main-verified.commit` 指向 `ee0da84ab9e04ac7610e28580af62c365e898389`，
这个是官方 main 参考，不代表用户安装的 `0.23.0+empty` 或本次已有 v0.23.0 源码就是它。

- W8A8 调用链在 `quantization/methods/w8a8_dynamic.py` 的 MoE `apply` 内调用
  `select_experts`；不是 V4.1 的 routed_experts selector 接口。
- `fused_moe.py` 先执行 prepare，再把处理后的 hidden_states/router_logits 传给量化方法。
- 路由 ID 需要在 zero/shared/物理映射或专家分发之前读取；共享专家不进入逻辑 Top-K 计数。
- 多流 gate 可复用另一处提前算好的 Top-K；初版必须明确拒绝而非错钩 selector。
- 真实 runner `_model_forward` 位于 KV/Mamba 上下文中，混合注意力不能绕过原有状态处理。
- 旧 Granite 0.23 adapter 只支持 uni/单设备/无量化；不能仅放宽模型白名单复用。

六个相关文件指纹在 `patches/ascend/qwen36-023post1/source-lock.json`。
指纹匹配是选定源码文件相符，不是完整安装、编译扩展或真机兼容证明。

## 接入计划与边界

1. 标准库检查脚本自动定位当前 Python 的包，记录版本、源码 HEAD、指纹及模型 config；
   不导入 TorchNPU，不修改环境，不加载权重，不启动服务。
2. 新独立 adapter 和 patch 目录，原包保持不动，隔离副本通过启动环境加载。
3. execute_model 实际 forward 前冻结 input_batch 顺序、counts、IDs、positions，
   与 computed positions 交叉核对，position 小于 prompt 长度的行不保存。
4. W8A8 selector 后捕获真实结果，不重算 softmax/Top-K。核实 Qwen 模块发现、原始层号、
   逻辑专家数及消费路径；未知架构/路径报错。
5. 按实际 prepare、SP、DP metadata、TP rank 映射有效行；不按 EP16 推断 token 所属。
   DP8/TP2 在 EP 开启且 PP1 的目标配置中形成 EP16，但每个 DP 有自己的请求/执行空间。
6. 复用分片存储、输入声明及有界 writer；每个 DP/TP worker 独立写 host。
   不套用 DeepSeek 的 Mooncake consumer / P-D 代理 / D-local 名称。
7. 动态 worker 清单与汇总，不把旧 DeepSeek profile 目录/身份检查直接当作 Qwen 支持。
8. 初次调试计划使用 V1/eager/同步、无前缀缓存/投机解码/EPLB/FlashComm/多流。
   是否改这些参数需结合实际启动命令和 Qwen 混合注意力约束；不承诺该组合已在 A3 跑通。

## 验证和交付顺序

- 环境自动发现、缺包、哈希变更、已有输出不覆盖的 CPU 测试。
- 固定 patched 源码接入块的 CPU 执行，真实模型/算子依赖 mock，必须明确标记。
- 混合 Prefill/Decode、并发请求重排、DP 不等长、TP/SP padding、重复/缺层/不完整测试。
- 全量现有测试不能退化；逐阶段 Git 提交。
- 提供中文真机指南与隔离加载脚本，现场先普通请求，再采集，再独立结果对照。
- 没有真实 A3 验收前保持 `npu_verified:false`，不以 CPU 测试声明 Qwen 采集已可用。

## 当前可运行的环境检查

在真机服务所用 Python/容器的插件仓库根运行：

```bash
python scripts/tools/inspect_qwen36_ascend.py \
  --model /实际Qwen模型目录 \
  --output /可写目录/qwen36-environment.json
```

报告文件必须尚不存在，父目录需已存在。把报告以及完整启动命令交给维护者。
不用手动指定 `/vllm-workspace`：脚本先查询实际 import spec。
`--ascend-package` 只用于开发者检查下载源码，不证明服务实际加载该路径。
报告始终是 inventory，不把源码匹配等同于采集功能已完成。
