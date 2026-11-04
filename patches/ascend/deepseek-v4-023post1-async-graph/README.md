# v0.23.0.post1 A3 异步/图/多流适配参考

目标镜像：`quay.io/ascend/vllm-ascend:v0.23.0.post1-a3`。

**此目录目前只有源码指纹清单，没有可应用采集补丁。**
`adapter_ready:false`，不能通过修改这个字段或复用旧 MTP overlay 获得支持。

`reference-files.json` 记录 release 参考的 18 个关键 Python 文件摘要；用户的镜像可能
包含不同构建提交或发行修补，必须比较实际安装文件。检查不依赖 `.git`，不会升级环境。

在包含本分支源码的 D 容器中使用服务解释器执行：

```bash
python scripts/tools/inspect_ascend_async_graph.py
```

优先按实际 Python 包定位，找不到时使用用户确认的 `/vllm-workspace`。
可选 `--vllm-package` 和 `--ascend-package` 指定实际包目录；`--output` 只写不存在的新文件。
匹配时退出码 0，不匹配/缺失为 2；输出包含版本、实际路径、摘要与差异。
匹配只表示开发参考一致，不代表异步、NPU 图或多流采集可用。
