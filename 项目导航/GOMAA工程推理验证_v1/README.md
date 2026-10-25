# GOMAA公开权重小型工程推理验证 v1

BENCH-018：工程关卡通过；actor目标方向诊断未过。仅8张已消费开发源、1份公开PPO权重，方向探针不是导航SR或强基线验收。

- [完整报告](工程推理与目标干预报告_v1.md)
- [冻结协议](元数据/冻结工程协议_v2.json)
- [另实现复核](核验/工程另实现复核_v2.json)
- [运行收据](运行/完成记录_v2.json)
- [下一项条件](下一项验证条件_v1.md)

复核可在仓库根运行 `D:\PYTHON\python.exe -B project/src/eval/audit_gomaa_engineering_v2.py`，只读核验已保存结果，0模型调用。首执行入口为冻结协议所绑定的隔离Python与tools/run_gomaa_engineering_v2.py；启动登记存在时禁止重复执行。不重跑prepare、forward或生成器。公开请求与评测标签分离，标签/完整特征银行没有进入推理worker。

本批依赖BENCH-017原权重/原代码及BENCH-008已消费开发特征；均保留原址。没有替换任何默认，旧验收和旧Geo失败因素保留。运行权重来源/实际训练地图/编码器版本未知，不能把8源称地理未见。
