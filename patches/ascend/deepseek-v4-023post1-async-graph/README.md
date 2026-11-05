# v0.23.0.post1 A3 异步/图/多流适配参考

目标镜像：`quay.io/ascend/vllm-ascend:v0.23.0.post1-a3`。

**此目录提供 v0.23.0.post1 的独立真机调试补丁，不是已验收的生产版本。**
`debug_entry_ready:true`；`adapter_ready:false`、`npu_verified:false` 保留真实 A3 未验收状态。

`0001-debug-hooks.patch` 包含 worker 初始化/退出、ACLGraphWrapper、W8A8 路由、
runner 主模型/采样和 MTP proposer 共五个文件的修改。使用
`scripts/tools/stage_ascend_async_graph.py` 自动校验并创建副本，
**不应把 patch 手工应用到原生产源码**。采集在启动预热前启用，图外同步设备后写入主机 JSONL。
完整安装、原脚本包装、输出与撤销见[现有容器调试指南](../../../docs/guides/DeepSeekV4现有A3容器采集调试指南.md)。

CPU 组件验证：

```bash
python scripts/validation/ascend_023_source_cpu.py --help
```

验证范围见[CPU 验证报告](../../../docs/validation/DeepSeekV4异步图执行采集CPU验证.md)。
包含实际参考源码方法与 CPU 张量、模拟图/算子，不证明真实 NPU 图、跨流同步或完整服务成功。

`reference-files.json` 记录 release 参考的 20 个关键 Python 文件摘要（含实际专家部署映射实现）；用户的镜像可能
包含不同构建提交或发行修补，必须比较实际安装文件。检查不依赖 `.git`，不会升级环境。

在包含本分支源码的 D 容器中使用服务解释器执行：

```bash
python scripts/tools/inspect_ascend_async_graph.py
```

优先按实际 Python 包定位，找不到时使用用户确认的 `/vllm-workspace`。
可选 `--vllm-package` 和 `--ascend-package` 指定实际包目录；`--output` 只写不存在的新文件。
匹配时退出码 0，不匹配/缺失为 2；输出包含版本、实际路径、摘要与差异。
匹配只表示开发参考一致，不代表异步、NPU 图或多流采集可用。
