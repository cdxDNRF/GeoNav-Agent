# 空间清理与恢复入口

**已完成：90.03 → 55.01 GiB，减少约35.02 GiB。** 原始数据/当前平台/默认保留，研究结论不变。

[清理报告](清理报告.md) / [保留策略与恢复边界](保留策略与恢复说明.md) / [最终验收](核验/清理验收_v2.json) / [当前存储与恢复映射](核验/当前存储与恢复映射_v2.json)。

- 105 TIFF：无损冷归档；全部解压SHA核验，按原路径恢复后复核。
- 17旧检查点：用户授权仅保留完全相同的推理参数，优化器/旧续训状态永久删除，不能精确续训。
- 搬迁ZIP/缓存：已清理，保留重建依据或小型备份。
- Git：经用户单独授权删除1451无引用blob/tree；7593对象和全部提交、分支、reflog、暂存区与工作文件保留，完整性检查通过。

TIFF恢复（只恢复需要的一组/成员）：

```powershell
& 'D:\PYTHON\python.exe' -B -m project.src.tools.cleanup_space_v1 restore --group cold_00
```

可加`--only 原相对路径`；cold_01为原小区域工程TIFF。恢复拒绝覆盖不同内容。PT仅查轻量映射，不运行旧restore或把新文件冒充原完整checkpoint。

当前核验：`python -B -m project.src.eval.audit_cleanup_retention_v2`，只读、零模型forward。不要重跑prepare/archive/compact/删除器或已封存科研生成器；若需恢复后保留热文件，原址存在不再符合“已移除热副本”的清理审计，模型/数据科研SHA另行核验。备份在DATA/processed_data/工程冷归档_v1，Git不携带，不能再按缓存删除。

后续CLEAN-004（2026-10-12）：用户明确选择保留最终3权重、丢弃其余冷存储后，两TIFF恢复归档和14历史权重已永久删除。此页当时17权重/105 TIFF可恢复说明已失效，当前必须先读[永久清理与边界](../归档永久清理_20261012_v1/README.md)，不运行旧restore或旧全保留审计追过线；原清理seal不回写。
