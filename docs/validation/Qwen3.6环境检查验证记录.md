# Qwen3.6 环境检查验证记录

日期：2026-10-09。分支：`feat/ascend-qwen36-035b-dp8-tp2`。
本记录只证明环境检查与源码调查，**不证明 Qwen 采集运行时完成或 A3 已通过验收**。

## 实际执行

- 下载独立官方 `v0.23.0.post1` 源码，实际 commit 为
  `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。未修改原有 0.23/V4.1 源码。
- 使用新脚本显式指定下载包路径，六个文件指纹全部匹配。
  本机 base Python 没有安装 NPU 软件包，报告如实给出包缺失与 `runtime_supported:false`。
  这是 source-path override 检查，不是实机 import 路径验证。
- 四个新增测试覆盖：缺包不误报可用、现有报告不覆盖、缺失/改动文件识别、嵌套模型
  config 读取、editable 路径由 import spec 定位而不是根据 metadata Location 推断。
- 包括既有固定源码测试的完整 CPU 回归：**278 passed in 6.22s**，没有 skip。
- `git diff --check` 通过。

回归命令复用 [V4.1 验证报告](ascend-v41-debug-validation.md) 中的三个源码环境变量和
`moe-ascend-cpu` Python 环境；本次没有安装官方 0.23.0.post1 的 NPU 依赖或启动服务。

## 尚未完成

真机完整启动命令、两个 editable 仓库的 HEAD、模型 architectures 尚未取得。
W8A8 selector 与消费路径已调查，但新 adapter、隔离补丁和 Qwen 专用启动/汇总仍待实现。
当前工具只做只读 inventory，输出是否匹配参考源码不等于设备、算子或功能可用。
不能使用已有 DeepSeek 启动器或旧 Granite 单设备 adapter 来宣称 Qwen DP8/TP2 支持。
