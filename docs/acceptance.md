# Acceptance status

生命周期更新：补丁 0003 和完整性联动已完成真实 GPU 验证，见
[请求生命周期报告](lifecycle-validation.md)。A10 对真实 SSE 断开取消、已执行
轨迹保留和取消后服务继续推理获得通过证据；stop/length 终止与正常收尾通过。
18 项测试通过。旧快照中的“未接入请求生命周期”已被此次结果取代。

最新更新：输入映射和专家消费逐元素核对已完成，见
[独立参考验证](reference-validation.md)：255 输入行、6120 层记录、48960
专家 ID 全部一致，包含 18 次并发执行、2 次阶段混合以及跨三轮 Prefill。
A03/A04/A05 对此指定模型与路径已取得真实交叉验证证据。下文旧快照中相应
“待核对”描述已由本报告取代；请求生命周期与长期资源等仍未完成。

更新：真实 GPU 测试已在授权的沙箱外执行，详见
[真实 GPU 验证记录](gpu-validation-2026-10-02.md)。下方原表是第一阶段快照，
不代表当前全部状态。A01/A12 真实服务检查通过；单请求、并发、分块与分片关联
已有真实证据，A05、请求生命周期与长期资源有界仍未完成，整体验收未通过。

日期：2026-10-02  
状态定义：PASS、FAIL、NOT RUN、N/A。

本轮 Agent 会话无法访问 GPU：NVML 报告访问被操作系统阻止，目标 Conda 环境中
`torch.cuda.is_available()` 为 false。因此真实 GPU 项不得标记通过。用户此前的
HTTP 200 记录仅作为环境历史，不作为本实现验收结果。

| ID | Status | Evidence / remaining work |
|---|---|---|
| A01 基础服务 | NOT RUN | 本会话无 GPU；未启动服务 |
| A02 单请求 | NOT RUN | 存储往返模拟测试通过；真实 forward 待测 |
| A03 多请求并发 | NOT RUN | 行映射按 runner 实际 batch 顺序实现；真实并发证据待测 |
| A04 分块 Prefill | NOT RUN | 连续位置模拟测试通过；真实多轮证据待测 |
| A05 路由真实性 | NOT RUN | hook 位于 `select_experts` 后、专家消费前；逐元素 GPU 核对待测 |
| A06 完整性 | NOT RUN | 缺层/重复/越界与关联校验单测通过；真实执行待测 |
| A07 无效数据排除 | NOT RUN | 无执行上下文的 warmup 回调被忽略；真实启动观察待测 |
| A08 文件往返 | PASS | NPZ/JSONL round-trip、明细 CSV、缺失分片检测单测通过 |
| A09 计数 | PASS | phase/layer/expert 计数与零负载专家单测通过 |
| A10 请求取消 | NOT RUN | 存储层 cancelled 事件通过；vLLM 生命周期原因集成尚待实现/验证 |
| A11 停机/异常 | NOT RUN | 正常 close、failed 状态、missing/pending 规则单测完成；进程级故障待测 |
| A12 关闭采集 | NOT RUN | 无配置环境变量/`enabled:false` 为 hook 空操作；真实推理待测 |
| A13 可选配置 | NOT RUN | token/weight/phase/label 单测通过；真实权重对齐待测 |
| A14 资源有界 | NOT RUN | 硬上限和多分片同步排空单测通过；长时 GPU 测试待测 |

## 本轮自动验证

- `11 passed`：配置严格校验、分片往返、并发请求行、分块位置、过滤、可选数组、
  零负载计数、严格失败、缺失分片。
- `python -m compileall -q src tests`：通过。
- 对 `/home/czx/code/vllm` 执行 `git apply --check`：通过。
- 在 `/tmp` 副本实际应用补丁并编译三个被修改的 vLLM Python 文件：通过。
- `vllm._C`、`vllm._moe_C` 导入：通过；未修改/编译 C++ 或 CUDA。

## 下一阶段真实验收顺序

1. 安装本包并应用已校验补丁，确认两个包的实际导入路径。
2. 以关闭采集启动一次，再以开启采集、eager、无 prefix/speculation 启动测试服务。
3. A02/A05 小请求先核对每个输入行和消费 Top-K；失败则停止扩大场景。
4. 分别构造同一 execution 含多请求、动态加入/退出，以及强制跨轮次 chunked
   prefill，保存 `steps.jsonl` 证据。
5. 测试取消、正常停机和受控写失败；最后执行 reader validate 与离线重算。

不能只根据生成 token 数判断 decode 路由数；验收以 `steps.jsonl` 中实际执行行为准。
