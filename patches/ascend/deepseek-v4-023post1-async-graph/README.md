# v0.23.0.post1 A3 异步/图/多流适配参考

目标镜像：`quay.io/ascend/vllm-ascend:v0.23.0.post1-a3`。

**此目录包含源码指纹清单及开发中的图采集钩子，尚无完整可部署适配器。**
`adapter_ready:false`，不能通过修改这个字段或复用旧 MTP overlay 获得支持。

`graph-hooks.patch` 针对参考版本的 `ACLGraphWrapper.__call__` 和 W8A8
逻辑专家选择位置添加钩子，已通过参考源码的 `git apply --check`。
此补丁只提供图路由设备缓冲接入，不是安装入口，请勿直接应用到生产源码。
尚需接入 runner 的启动预热、真实请求执行、异步 MTP 候选关联和落盘。
尤其是图必须在启用采集上下文时捕获；对先前未带采集钩子捕获的图，桥接层会报错，
避免把空缓冲或旧数据当作本次结果。图快照没有执行 CPU 拷贝，后续消费者必须在
正确的流依赖完成后再转换和写盘。

CPU 组件验证：

```bash
python -m pytest tests/test_graph_routes.py tests/test_ascend_graph_bridge.py tests/test_ascend_async_inventory.py
```

11 项通过，覆盖模拟设备图重放、执行快照隔离、上下文异常清理以及未插桩图拒绝。
测试使用模拟图和 CPU 数组，不证明真实 NPU 图、跨流同步或完整模型服务采集成功。

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
