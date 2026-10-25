# GeoExplorer资源验证 v1

BENCH-003，2026-10-07。**合成资源关卡通过；尚未完成导航基线复现，没有新SR。**

| 入口 | 用途 |
|---|---|
| [资源验证报告](资源验证报告.md) | 实测参数、显存、吞吐、环境差异及结论边界 |
| [下一批训练预算与执行条件](下一批训练预算与执行条件.md) | 已选定预算、共同训练来源、待完成的真实执行器与停止条件 |
| [预算登记](核验/下一批训练预算.json) | 机器可读预算，execution_allowed=false，尚无训练执行源码绑定 |
| [冻结资源协议](核验/冻结资源协议.json) / [执行绑定](核验/执行绑定.json) | 本轮首次模型运行前固化的上限和源码/输入SHA |
| [原始小试记录](运行/资源小试_v1/result.json) / [seal](运行/资源小试_v1/seal.json) | 一次8步合成运行；不保存训练模型、不重跑 |
| [另实现复核](核验/另实现复核.json) | 参数形状求和、日志/限制及来源/数据划分重算，同一开发Agent完成 |
| [完整来源绑定](公开来源/完整来源绑定.json) | 固定官方commit、完整tree和选择的科研源码；无项目根许可证 |

隔离运行环境位于根README已登记的`project/.runtime/geoexplorer_probe_v1/`，仅本地保存，未更改原D盘Python环境。公开来源与已冻结小试源码保留原字节。只读复核可执行：

```powershell
& D:/PYTHON/python.exe project/src/eval/audit_geoexplorer_resource_v1.py
& D:/PYTHON/python.exe project/src/tests/test_audit_geoexplorer_resource_v1.py -v
```

禁止再次运行prepare、execute或带`--save`的审计覆盖本批输出。下一批需另认领并登记新目录。旧默认与历史验收、3个备用区域不变。
