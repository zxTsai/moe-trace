# DeepSeek V4 单候选 MTP 补丁

独立于非 MTP V4 补丁，基于 `base_files.json` 的固定源码。
使用 `scripts/tools/stage_ascend_v4_mtp.py` 检查并生成副本，不改原安装包。

补丁位置：runner target forward、rejection sampler 返回、proposer `_propose` 的真实调度
边界、MTP 首次 model forward 和输出返回、logical selector、W8A8 consumer、worker 生命周期。
`0001-mtp-hooks.patch` 与生成副本逐字节比较，`debug-overlay.json` 是可信副本摘要。

仅 D DP16/TP1、MTP K=1、单 MTP 层、同步 eager、无多流/图/CP，默认关闭。
撤销：不加载本次 overlay/采集配置，使用原安装源码与原脚本重启服务。
CPU 接入已验证，真实 A3 未验收；见[使用指南](../../../docs/guides/DeepSeekV4开启MTP采集指南.md)。
