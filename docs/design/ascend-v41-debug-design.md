# 两机 A3 1P1D 调试采集设计

用户接受调试模式：P DP4/TP4，D DP8/TP2，EP、Ascend 量化、MooncakeHybridConnector
保留；P/D 不配置 DSpark，D 改为 eager、同步、无多流重叠。只采集 D 的 Decode
输入位置。此配置不等同完整官方 5.2；不改变路由算法或增加通信 collective。

固定源码同 V4.1 source-lock，另锁定 worker shutdown、SP 和 prepare 相关文件。
原始源码保持不动；独立副本加载新 debug adapter，不与 CPU hook 副本混用。

## 真实接入

- execute_model 的实际 `_model_forward` 前后建立上下文，位于 KV connector 上下文中。
- 按已经准备后的 input_batch 请求顺序和 scheduled counts 映射实际 input_ids/positions；
  同步模式核对 computed position 和 request.get_token_id。过滤 position<prompt 长度。
- router selector 后、log2phy/shared 前同步拷贝逻辑选择，不重新计算 Top-K。
- 从实际 prepare_finalize 类型及状态确定行映射，禁止按 shape 猜测通信类型：
  All2All：tensor_split；MC2：SP shard + padding 或全量 pad/split；AllGather：
  DP padded block / EP rank chunks。只保存本 DP 的有效行，SP shard 的 owner 是该 TP。
  AllGather 已含完整 DP 输入时只由 TP0 保存，避免多份逻辑副本统计。
- W8A8 consumer 前保存同一组行的映射后参数，shared 列独立，不视为逻辑 Top-K。
- shutdown 排空分片。异常保持 failed；不伪装已验证请求终止生命周期。

## 输出与验证

每个 DP/TP 进程独立 schema run，顶层 deployment.json 声明 16 个预期 worker。
使用同一个 run_id，目录 dp-0-tp-0 至 dp-7-tp-1。层列表从实际模型模块发现；
缺失层、重复层、prepare 后行数、输入位置、consumer 关联不符即失败。
没有来自代理的全局请求映射时明确标注 D-local namespaced，不能宣称 P/D 全局关联。
所有文件标记 a3_unverified_debug；正常收尾表示声明采集范围存储完整，不是 A3 可靠性证明。

CPU 运行修改后的真实方法，覆盖 DP 不等长、SP padding、TP 切分、混合 Prefill/Decode、
独立 consumer 对照和故障。NPU 模型、算子、HCCL 与真实 PD 仍须部署实测。
调试模式同步 D2H 有显著性能开销，不用于生产性能结论。

启动前必须检查模型路径、镜像内源码哈希、实际导入副本和 16 个 worker 枚举。
启动器提供 dry-run，默认不执行服务；显式 --execute 启动自己的子进程并转发停止。
不下载模型、不升级依赖、不清理其他服务、不推送或合 main。
