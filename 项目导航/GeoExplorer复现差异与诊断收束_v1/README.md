# BENCH-012：有限预算学习基线诊断收束

[当前报告v3](复现差异与诊断收束报告_v3.md) / [绑定](元数据/冻结收束协议_v1.json) / [另实现复核](核验/另实现复核_v1.json) / [下一项未启动草案](下一项位置编码单因素草案_v1.md)。

v1/v2为只读初稿；最终v3保留切分勘误并说明首次测试入口间接导入Torch，科学汇总不变。[补充文档源码绑定](元数据/补充文档源码绑定_v1.json)保留old/new/SHA。[测试v2](核验/CPU测试_v2.json)完成9项；首次测试入口命名空间失败记录原址保留。

本交付替代BENCH-011的主线进度入口，保留BENCH-007至011原址证据与各自验收；不替代M0默认或原GeoExplorer基线。不训练/forward/导航，无新SR。

在仓库根设定PYTHONPATH为project/src后，使用D:/PYTHON/python.exe执行`-m tools.prepare_geoexplorer_evidence_close_v1 verify`进行只读SHA核验；`-m eval.audit_geoexplorer_evidence_close_v1`另实现复核不加--save。禁止在本批重跑prepare、--run、报告生成或--save；复现算术需另登记新批次并核对输入绑定。

[测试入口偏差说明](核验/测试入口偏差说明_v1.json)：首次误匹配Ultralytics同名tests，不能称全轮零Torch导入；未加载项目模型/执行forward。统计/审计和后续9项测试保持纯算术范围。
