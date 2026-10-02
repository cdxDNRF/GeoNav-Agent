# 主动视觉地理定位：可复用的研究决策

更新：2026-09-30。目标已由用户确认：给定目标图片，在未知网格中通过连续局部观测找到对应位置。本次只进行研究、只读项目核验和实验设计。

完整报告：[十九节深度调研与实验规格](D:/桌面/XDUCS-courses/大四/大数据工程/调研/DEEP_RESEARCH_active_image_goal_search.md)。主报告包含47项去重研究、15条候选路线、4条研究路线图、完整架构与五项下一轮实验；论文卡、检索记录及关键引用核验保存在“调研”。

## 当前最值得实现的主线

**目标实例证据 → 跨步视觉记忆 → 持久waypoint/option → 预算内规划 → 逐格执行。**

VLM先回答“去哪个可达位置检验哪条目标线索”，程序负责路程和动作。子目标持续2–3步作为初始实验范围；每步仍获取真实画面，只在到达、反证、强匹配或超时后重选高层目标。先试单项机制，不把所有组件同时堆上。

主要依据是实例匹配与导航分离的[Modular InstanceImageNav](https://openaccess.thecvf.com/content/ICCV2023/html/Krantz_Navigating_to_Objects_Specified_by_Images_ICCV_2023_paper.html)、跨节点记忆的[TSGM](https://proceedings.mlr.press/v205/kim23a.html)、目标/场景图匹配的[UniGoal](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.html)，以及高层子目标的[HRNav](https://aclanthology.org/2026.acl-long.1352/)和[SPF](https://proceedings.mlr.press/v305/hu25e.html)。这些是迁移依据；本项目收益尚待实验。

## 建模上的关键区别

本地当前坐标、5×5边界和无障碍四向转移已知；未知的是目标格与未观察视觉地图。应研究**目标条件的预算搜索**，不必先重做SLAM。

四方向动作本身符合任务，问题在于高层每步只输出动作，可能丢失跨步证据和轨迹承诺。BFS/A*只知道怎样去给定格，不知道哪个格值得去。语义标签“住宅区”也不能代替目标地点实例匹配。

没有空间相关线索时，目标图不能凭空确定远处方向：两张地图可在已访问历史完全相同，却把目标放在不同未访问方向。这是不可区分性推论。此时合理行为是预算内覆盖和弃权；连续道路、水系、布局或边缘关系提供可靠证据后，才偏离覆盖。

`VLM confidence`、cosine或softmax先当排序分数。真信息增益需要 `p(未来观测 | 历史,动作,目标位置)`；没有观测模型只能用proxy。重复看同一静态格不能反复乘同一证据。

## 起点论文能复用什么

| 论文 | 真正可复用组件 | 应避免的误读 |
|---|---|---|
| [GOMAA-Geo](https://arxiv.org/abs/2406.01917)，NeurIPS 2024 | 目标条件的观测/动作序列与位置编码 | zero-shot泛化不等于不训练：含表示对齐、历史预训练和PPO |
| [GeoExplorer](https://arxiv.org/abs/2508.00152)，ICCV 2025 | 动作条件的下一特征预测与训练探索信号 | curiosity不是目标位置belief，也不是显式地图规划 |
| [DynCur-Geo](https://arxiv.org/abs/2608.18673)，2026预印本 | 训练期距离门控好奇心+PBRS的动态平衡 | 训练真目标距离不进推理；按初始距离并非处处领先 |

当前已有BC、PBRS、好奇心、动作筛选和邻接/方向线索实验，不作为新的路线重复建议。三篇论文的数据、目标模态与预算和本地协议有差异，不能直接按SR数字排行。

## 2025–2026补充的高价值方法

- [SemNav](https://arxiv.org/abs/2506.03516)：相同高层评分下，LSP与贪心frontier对照支持检验“多步代价规划”。其低层仍是预训练PointNav；网格BFS是我们的替换设计。
- [SPF](https://arxiv.org/abs/2509.22653)：同Gemini 2.0 Flash的文本动作/PIVOT/waypoint消融为7/40/100；这是指令条件UAV任务，不是本项目SR预测。87属于另一模型Flash-Lite。
- [AnyImageNav](https://arxiv.org/abs/2604.05351)：廉价相关性筛选后才触发昂贵几何注册；可借感知预算级联，无须照搬3D重建。
- [AeroBelief](https://arxiv.org/abs/2609.08164)：把弱上下文intuition、强evidence与覆盖引导分开，适合思考探索器/目标线索接口；语言物体目标、预印本，源码尚待发布。
- [VTM-Nav](https://arxiv.org/abs/2607.14514)：层级视觉拓扑记忆只软性重排当前合法候选。跨episode协议应单独研究，不能在当前评测偷偷复用同图历史。
- [Select2Plan](https://arxiv.org/abs/2411.04006)：VQA+少量样例检索/ICL，不更新policy；仍需要示例数据，并非目标照片导航。

## 下一轮五项实验

| 实验 | 只改变什么 | 必要对照 | 能回答什么 |
|---|---|---|---|
| E0 现有目标线索的范围 | 冻结Edge的源图/目标图扰动协议 | 同题NoTarget、ZeroEdge、错误目标 | 连续切片收益是否可复验、能外推到哪些变化 |
| E1 持久waypoint | 子目标持续性/重规划条件 | G、每步重规划、无目标持久子目标 | 高层承诺能否省调用并改善搜索 |
| E2 实例关系记忆 | 多已观测节点的支持与反证 | 等token FIFO、位置打乱、类别摘要 | 是否真正解决跨步目标判断 |
| E3 同scorer短路径beam | h=1/2/3/4决策时域 | greedy、持久waypoint、coverage-only beam | 收益来自规划还是额外感知/覆盖 |
| E4 跨步关系小scorer | 固定探索器上的历史关系融合 | 同容量双图head、从头NoTarget | 可靠线索覆盖是否增加并转化成目标贡献 |

E0属于复验/诊断，不称新方法；E2/E4必须超出已做邻接头与Edge，否则只是换存储或加层。E4后置，先确认可识别的历史关系。各项Hypothesis、Modification、Baseline、Controlled variable、Metrics、Expected observation、Success criterion和Failure interpretation均已写入[实验规格](D:/桌面/XDUCS-courses/大四/大数据工程/调研/06_next_experiments.md)。

核心机制链：**可靠目标证据 → 实际改变决策 → 同预算成功改善 → 冻结方法后新源图成立**。只减少重访或提升NoTarget覆盖，应将贡献定位为探索效率，不自动称目标理解。

## 当前证据与资源边界

本次直接读取环境、manifest和保存的验收记录：只开放目标/当前图、位置/预算/visited；没有邻格图或全图；B=10、自动到达成功、无STOP。不能照搬室内“到达后验证/停步”收益。缓存历史合法观测可用，未访问图像/特征仍禁止。

本地Masa为137/4/10源图；val100只有4张源图且有重复路线。现有强开发默认Small256 NoTarget仍不是已确认的视觉目标方法；最新Edge是开发候选，默认未变、正式视觉S2未通过。此处是已保存材料快照，本次未重算全部历史成绩。具体来源与限制见[本地核验](D:/桌面/XDUCS-courses/大四/大数据工程/调研/04_local_project_context.md)。

本机实测RTX4060 Laptop、8188MiB。优先冻结编码器，训练小scorer/adapter；当前API单价、额度和总预算未知，以请求、图片数、tokens、延迟和错误率比较。世界模型、完整大VLM训练和多机协作保留后续，先完成单Agent目标证据主线。

源图分离比episode数更关键：4个源图簇无法支持稳定跨地区结论；已反复使用的28开发图也不再是独立确认集。test仅用于方法冻结后的最终评测。相同episode、预算、模型与信息权限配对比较；新方法要同时对强NoTarget和Frontier比较。

## 建议的核心研究贡献表述

可以聚焦：“在局部观测、预算受限的视觉地点搜索中，利用可追溯的目标关系证据选择并保持可执行子目标；将目标引导收益与覆盖收益分开验证。”

这是待验证主张。采用已有VLM、memory graph和beam本身不保证新颖性；创新应落实到证据融合/子目标保持或预算效用中的一个核心规则，并通过对照证明。更广的“通用跨视角无人机定位”需要额外采集与任务协议。

核验入口：[关键引用审查](D:/桌面/XDUCS-courses/大四/大数据工程/调研/08_citation_verification.md)｜[综合与红队意见](D:/桌面/XDUCS-courses/大四/大数据工程/调研/09_synthesis_and_critique.md)。
