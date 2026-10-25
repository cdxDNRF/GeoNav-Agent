# 原训练资产核验与共同验证准入 v1

BENCH-021，2026-10-10。**静态补证据完成，原训练资产尚未补齐；共同验证执行配置尚未冻结。**

- [核验报告](原训练资产核验报告_v1.md)：新增checkpoint训练记录、完整发布树与未解决项。
- [资产接收清单](资产接收清单与核验说明_v1.md) / [机器清单](元数据/资产准入清单_v1.json)：六类原训练关系；文件SHA相符不能替代来源关系核验。
- [共同验证草案](共同验证条件草案_v1.md) / [状态与关卡](元数据/共同验证草案_v1.json)：准入规则已明确，execution_allowed=false。
- [给作者的资料请求草稿](未发送_原训练资产资料请求_v1.md)：尚未发送，不含凭据或本地私人数据。
- [另实现静态复核](核验/另实现复核_v1.json)、[完整HF树核验](核验/历史树另实现核验_v1.json)、[交付核验与阶段封存目录](核验/)。同一Codex另实现，不是独立人员审查。

旧BENCH-017/019/020原件、模型、默认、已收束因素保持。0训练/模型forward/编码器/导航/模型API/新SR/新源或备用消费；未提交推送或联系作者。

复核入口：根目录运行 `D:\PYTHON\python.exe -B project/src/eval/audit_original_training_assets_v1.py` 或 `project/src/tools/original_asset_history_completion_v1.py verify`，仅只读。交付核验入口为 `project/src/eval/audit_original_training_asset_delivery_v1.py`。**collect/fetch及--save已消费，不重跑**；出现中断先读启动登记和请求收据，不覆盖重采集。

后继只能接收新增的实际资产/来源证明，再认领新版本评审；没有新增资料时，停止同一权重的重复补证据/导航。论文强基线G1、新源共同确认G2、干净环境完整复跑G3仍未完成。
