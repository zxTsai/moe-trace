# 真实 GPU 验证记录

后续更新：本报告中的 A05 与并发逐行核对缺口已由
[独立参考验证](reference-validation.md) 补齐；其余未完成项仍有效。

环境：WSL2、RTX 4060 Laptop 8GB、Python 3.11、PyTorch 2.7.1+cu128、
vLLM 0.10.1 指定基础 commit、Granite 本地权重。GPU 在默认沙箱内不可见，
在已授权的沙箱外可用。本包使用 `--no-deps --no-build-isolation` 本地 editable
安装，未升级运行依赖。vLLM 应用了版本化补丁 0001 和 0002。

服务参数：127.0.0.1:18001，V1、enforce-eager、关闭 prefix caching、
max-model-len=2048、max-num-seqs=8、max-num-batched-tokens=512、
gpu-memory-utilization=0.75。模型为需求指定的本地路径。

## 证据

输出根目录为 `/home/czx/code/moe-routing/outputs`。

| 运行 | 实际结果 |
|---|---|
| 关闭采集 | HTTP 200，9 prompt / 8 completion tokens；采集目录数前后均为 0 |
| `a5eb810e-5818-460f-b05e-676c9790e90d` | 首次开启：9 Prefill、7 Decode、384 层记录；收尾 hook 未覆盖 UniProc，运行不完整，保留原始状态 |
| `f7751b5a-4df0-45ac-af3c-5d1ae9a0514f` | 修复 UniProc 后：4 Prefill、3 Decode、168 记录、1 分片；reader 无错误，计数总和 1344 |
| `7351b786-1988-474b-85b9-7ad186b35e5d` | 4 请求、15 execution，其中 8 次含多个请求；701-token Prefill 为 0–511 和 512–700；20040 记录全部提交，但退出前未完成校验，状态仍不完整 |
| `7180920d-688b-40f8-a9ce-bbc32372666c` | 校验改为线性索引后：输入行数 512、189、1；16848 记录、5 分片、缓冲 0、complete=true；reader 无错误，计数总和 134784 |

短请求的 expert_ids 数组为 `[168,8]`，原始层号 0–23，专家范围 0–31。
统计保留零负载专家（短请求中有 777 个 phase/layer/expert 零计数条目）。
长提示最后一次 Decode 处理位置 701；最后生成 token 未再输入，因此无位置 702 路由。

## 修复与验证范围

增加 0002 补丁覆盖 V1 UniProcExecutor.shutdown。收尾校验的逐行全集扫描
改为一次建立 `(execution_id,row_id) -> layers` 索引；对已提交的 20040 条记录
校验用时约 0.073 秒。buffer 在追加过程中按 shard_records 排空，测试断言
峰值不超过 buffer_records。回归测试共 12 项通过。

所有由 Agent 启动的测试服务均已停止。已有不完整运行保留，没有手工更改结束标记。

## 尚未通过的需求

- 请求结束/取消原因尚未接入 scheduler；当前 complete 标记只证明存储关联
  校验通过，尚未满足需求中的请求生命周期完整性。因此以上 complete 运行
  不能据此宣称全部需求通过。
- A05 尚缺专家 kernel 消费处的独立快照逐元素交叉核对。
- 标签映射尚未从 API 传到内部请求；权重/token ID 开关仅经过工具层测试。
- writer 仍维护历史执行/重复键，adapter 暂存一次执行全部层的 CPU 路由；
  尚不能宣称长时间运行内存完全有界。
- 受控写失败、取消、合法重算识别、严格启动时拒绝和小规模真实样例交付待完成。

本报告的实际执行证据取代旧报告中“GPU 不可用”的当前状态；该描述仅适用于
默认沙箱内检查。
