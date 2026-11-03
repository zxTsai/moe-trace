# DeepSeek V4 DP16/TP1 调试补丁

仅用于 D 节点 V1、同步 eager、无 MTP、Ascend W8A8 的文本 V4。
本目录与 V4.1/Qwen 补丁独立；CPU 源码方法验证通过，A3 尚未验收。

- `base_files.json`：参考提交及关键 Python 文件摘要；不是全二进制兼容保证。
- `0001-debug-hooks.patch`：runner forward、logical router、W8A8 consumer、worker 生命周期。
- `debug-overlay.json`：可信副本摘要，启动器与生成副本比较。

使用 `scripts/tools/stage_ascend_v4_debug.py`，自动定位当前 Python 包，检查摘要、唯一
接入锚点和 AST，再生成独立包副本；不直接在原安装包上 git apply。
V4 hash router 和普通 router 均在实际 selector 返回后读取；静态 log2phy 前保存逻辑 ID，
consumer 检查映射后的真实输入。无效布局、缺层或不支持模式直接失败。

撤销：停止本次调试服务，原环境不加载 overlay 和采集配置，重启原启动脚本。
安装、加载、结果与排错见[两机 A3 快速开始](../../../docs/guides/DeepSeekV4两机A3快速开始.md)。
