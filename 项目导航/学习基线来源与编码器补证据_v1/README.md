# 学习基线来源与编码器补证据 v1

BENCH-020已完成，只读公开来源与本地静态身份核查；新入口不等于强学习基线合格。原GOMAA本地目标资格失败/已收束条件与默认保留。

- [报告与资格矩阵](来源与编码器证据报告_v1.md)
- [未启动的下一步准入方案](下一步证据准入方案_v1.md)
- [冻结范围](元数据/冻结核查范围_v1.json)
- [本地编码器静态身份](元数据/本地编码器静态身份_v1.json)
- [逐项主张与来源绑定](元数据/证据主张_v1.json)
- [另实现复核](核验/来源另实现复核_v2.json)

恢复只读：仓库根默认Python运行`project/src/eval/audit_baseline_provenance_followup_v2.py`不加--save，或`project/src/tools/close_baseline_provenance_followup_v2.py verify`。不能重跑init/fetch/报告/close，不能覆盖封存原件；没有活动模型进程、训练/导航/新SR、默认升级或Git操作。前批源码/数据/权重原址保留。
