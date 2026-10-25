# 原训练资料接收 v1

BENCH-022，2026-10-10。**已取得维护者指向的上游真实切分文件；原导航训练缓存/run的完整关系仍未建立，正式共同验证尚未冻结。**

- [接收结论与边界](接收结论与边界_v1.md)：新增作者/仓库成员回复与AiRLoc原切分。
- [修订准入状态](核验/准入状态_v1.json)：候选数据清单已取得；不是完整原训练资产恢复。
- [待发布的简明请求](待发布_原训练资产缺项请求_v1.md)：用户尚未明确授权外发，未发布。
- [原切分CSV](公开来源/Airloc_list_eval_partition.csv)、[网格CSV](公开来源/Airloc_list_eval_partition_grid.csv)、[固定评测CSV](公开来源/Airloc_test_eval_partition_grid_fixed_distance.csv)、[原统计](公开来源/Airloc_stats.json)。仅小型元数据，不包含航拍图块或模型。
- [另实现核验目录](核验/)：输入保护、所有收据、范围/CRC与CSV计数、来源主张、测试、源码快照及最终封存。

公开接口前批403已通过现有认证只读入口补齐，0对外写入。公开评论原件`公开来源/geo_issue5_comments.json`含第三方过期签名媒体查询，仅本地保留且排除Git；[共享脱敏文本](公开来源/geo_issue5_comments_redacted.json)与[SHA映射](核验/评论脱敏映射_v1.json)保留其正文证据。

复核：仓库根运行 `D:\PYTHON\python.exe -B project/src/eval/audit_upstream_asset_intake_v2.py`（无参数）或 `project/src/tools/upstream_asset_intake_v1.py verify`。所有collect/--save/封存步骤已消费，禁止覆盖重跑。收据中的停止保护不是未完成下载，原ZIP/影像/标签/权重均未下载；指定4成员另登记后接收。

旧BENCH-021清单/判定与所有历史原件保持。0模型载入/forward/训练/重编码/导航/API模型调用/新SR/新地域或备用消费；原目标失败、默认及S2/S3/S4验收不变。当前状态见根PROJECT_STATE，交接见HANDOFF。
