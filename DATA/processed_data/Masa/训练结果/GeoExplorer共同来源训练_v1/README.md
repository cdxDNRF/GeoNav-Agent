# GeoExplorer共同来源训练 v1

BENCH-005，2026-10-07。**正式训练已启动，完成状态以训练结果汇总.json及各seed的result.json为准。没有新导航SR。** 用户授权执行器验证和绑定后启动；当前方法默认不替换。

| 入口 | 含义 |
|---|---|
| [冻结执行协议](元数据/冻结执行协议.json) | 首训练forward前2777项输入/源码SHA与完整结构/采样/资源预算 |
| [保护默认](元数据/保护默认.json) | 两个原默认SHA，训练不改默认 |
| [种子0日志](核验/seed_0_stdout.log) | 300轮预训练与256次PPO更新的进度；不作新SR |
| `seed_<N>/events.jsonl` | 每个已提交微批/环境步、epoch/更新及中断/恢复的单调计数；训练真值仅训练端使用 |
| `seed_<N>/*.pt.seal.json` | 新checkpoint字节与冻结协议/逻辑计数绑定，不覆盖旧checkpoint |
| `训练结果汇总.json` / `seed_<N>/result.json` | 运行结束才生成；缺文件或incomplete不能说三种子训练通过 |
| [工程前置报告](../../../../../项目导航/GeoExplorer真实执行准备_v1/执行准备报告.md) | 17项CPU边界测试、完整网络回归与原实现差异 |

109张既有拟合源，28张已知开发源不进入训练bank；原val/test及备用区不消费。三种子0/1/2顺序，各300epoch/32700微批/300预训练更新，随后163840环境步/256PPO更新×4轮。固定FP32、最终预算checkpoint，每种子6小时/整批18小时/18GiB。环境交互量与原M0相同，辅助监督/优化次数/总算力不同，不能称同总训练预算或同轮配对训练。

当前driver为`project/src/train/run_geoexplorer_training_v1.py`，其子进程使用已绑定`geoexplorer_training_v1.py`。**已经启动，不直接再运行driver，也不覆盖或改名已有seed目录。** 原始训练数据、源码、默认只读。当前活动进程和会话写入TASKS/HANDOFF；接手先核对进程、日志、result和checkpoint，不盲目重跑。

SIGINT请求在提交边界保存新的recovery checkpoint，含随机状态、模型/optimizer/scheduler、微批梯度/游标或PPO buffer/旧策略/episode。只允许最后日志计数与checkpoint一致的恢复；突然强杀后日志超checkpoint则不能直接回退重复执行。恢复仍受原累计时间/输出预算约束；CPU恢复测试通过不等于长期GPU恢复稳定性已确认。

全部三种子完成后再运行只读独立计数/奖励分解/权重SHA复核：`D:/PYTHON/python.exe project/src/eval/audit_geoexplorer_training_run_v1.py`。只有初次封存时可另用`--save`，不得覆盖。代码本身不预测或导航；训练完成后共同主对照、目标Full/Disabled/Mean/Wrong和新区域确认另按协议推进。论文方法的公平对标和目标有效性尚未通过。
