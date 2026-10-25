# BENCH-002：共同评测接口（当前入口）

2026-10-07。本批完成自有共同评测契约、GeoExplorer候选的无模型历史桥与合成核验。**不是GeoExplorer神经网络复现，不是训练，不产生新科研SR。**

- [接口验收报告](共同评测接口验收报告.md)
- [下一项资源实测与训练冻结草案](下一项资源实测与训练协议草案.md)
- [交付源码/输入绑定](核验/交付核验.json)
- [合成队列独立方程复核](合成验证/独立方程复核.json)
- [固定合成队列与运行封存](合成验证/固定队列/evaluator_manifest.json) / [汇总](合成验证/固定队列/summary.json) / [seal](合成验证/固定队列/seal.json)

复核只读命令（仓库根）：

```powershell
& 'D:/PYTHON/python.exe' project/src/eval/audit_benchmark_contract_v1.py '项目导航/共同评测接口_v1/合成验证/固定队列'
```

边界测试命令：

```powershell
& 'D:/PYTHON/python.exe' project/src/tests/test_benchmark_contract_v1.py -v
```

测试夹具在本批核验/测试临时_*创建并按TemporaryDirectory清理，父目录解析到仓库内；不读取真实数据。交付生成的--delivery命令和已有输出不能覆盖重跑，新批次另认领/登记。

代码入口：agents/benchmark_contract_v1.py（公开请求/不可变特征）、agents/geoexplorer_contract_v1.py（已观察历史）、eval/benchmark_runner_v1.py（仅合成5/B10评测）、eval/audit_benchmark_contract_v1.py（不导入执行器的另实现核验）。实际权重适配、张量转换、原位置编码/网络前向及训练留到下一资源关卡；本批不引入第三方实现或许可证未确认的代码。

BENCH-001继续作为资源/来源盘点原件保留；其“下一项草案”为当时计划，本批为该计划的接口交付。旧默认和所有历史科研验收不变；当前科学事实仍以[PROJECT_STATE](../../docs/PROJECT_STATE.md)为准。
