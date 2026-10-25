# BENCH-013：位置编码工程验证

[当前报告](工程验证报告_v1.md) / [协议v2](元数据/冻结工程协议_v2.json) / [复核](核验/工程另实现复核_v1.json)。

v1工程失败原址保留；v2为当前通过的工程worker，v1 train角色仍由下一批依赖，不能将其整文件删除。模型/默认不变，无新SR。只读复核用PYTHONPATH=project/src及D:/PYTHON/python.exe -B -m eval.audit_geoexplorer_position_v2，不加--save。禁止重跑prepare/engineering覆盖原输出。
