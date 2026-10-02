# 07｜2025–2026 training-free goal search / memory targeted gap-fill

检索截止与访问日：2026-09-30（Asia/Shanghai）。本文件只记录本次补缺候选和索引渠道核验；R01–R04 是临时编号。证据优先级：A = 论文原文/正式会议或期刊页面、作者官方代码库；B = 作者项目页；C = 学术索引/第三方聚合，只辅助发现或核对，不作核心论据。arXiv 预印本不计作同行评审发表。代码状态以访问日可见的作者官方库为准；搜索不到只写“未核实”。

## 本次结论

新增四篇独立工作，分别补上零样本实例图像目标、跨回合视觉拓扑记忆、最新航拍目标搜索、以及 VQA + 样例检索规划。它们彼此不是同一论文的不同项目页。

- 与本项目最直接的是 R01 UniGoal：其目标类型明确包括 instance-image-goal，且作者仓库公开。它仍依赖比“当前 patch + goal patch + position + visited”更丰富的场景图和场景理解。
- 记忆结构方面最值得参考的是 R02 VTM-Nav：训练免微调，但任务是室内物体类别目标、同一场景跨 episode 经验复用；需要 RGB-D、位置/朝向、VLM 场景摘要和持久视觉记录。它不能直接视作当前单次、离散 5×5 搜索的现成策略。
- 最接近航拍搜索结构的是 R03 AeroBelief：2026 年 9 月的 aerial ObjectNav，分离弱上下文先验与目标证据，并将覆盖引导与语义热点分开。目标是自然语言描述，不是目标图像；用到比题设更丰富的状态。作者官方 repo 已核实，但访问日源码尚未发布。
- R04 Select2Plan 给出训练免微调的 VQA/ICL 检索规划：可借鉴“先从当前图像选择动作候选，再映射动作”的接口；任务是语言/物体目标、需要示例记忆库，且不是 image-goal 方法。
- Google Scholar、Semantic Scholar、OpenReview、Papers with Code 的本次定向查询未能确认目标论文是否已被其收录。web 工具直接打开这些站点的查询页均拒绝/失败；这只能说明本次访问渠道受限，不能推出数据库未收录。

## R01｜UniGoal: Towards Universal Zero-shot Goal-oriented Navigation

**原始来源。** [CVPR 2025 官方论文 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.pdf)；[arXiv:2503.10630](https://arxiv.org/abs/2503.10630)；[作者官方项目页](https://bagh2178.github.io/)；[作者官方 GitHub](https://github.com/bagh2178/UniGoal)。

**作者与发表状态。** Hang Yin、Xiuwei Xu、Linqing Zhao、Ziwei Wang、Jie Zhou、Jiwen Lu。2025，CVPR 正式论文，页码 19057–19066；不是只停留在 arXiv 的预印本。官方 CVF PDF 的检索摘要含论文标题、作者和摘要；本次 web 工具对完整 PDF/HTML 抽取失败，因此下列未直接读到的数字不猜测。

**任务、观察与动作。** 统一处理 Object-goal、Instance-image-goal、Text-goal 三类目标；其中 instance-image-goal 与“给定目标航拍图片”在目标接口上直接相关。智能体把在线环境观察构造成场景图，把类别、图片或文本目标转成统一目标图。可确认论文以观察生成的场景图为输入；具体相机传感器、原始图像分辨率和 primitive action 定义在本次可读原文中未核实。

**训练、记忆与规划。** 作者称方法可 zero-shot 用于多种场景与目标，不训练任务专属导航策略。场景图随探索在线维护。图匹配分为零匹配、部分匹配、完全匹配：零匹配时探索目标子图；部分匹配时用坐标投影和 anchor-pair 对齐推断位置；完全匹配后做场景图校正和目标验证；blacklist 抑制反复切换/冗余探索。该“图关系 → 阶段切换 → 局部化”的逻辑比直接让 LLM 自由猜整条路径更适合作为可移植组件。

**数据与结果。** 原文摘要称在多个 benchmark、三类导航任务上达到领先的 zero-shot 表现；本次未成功取到官方论文表格，具体 benchmark 清单及数值记为未核实，不从二手摘要补写。

**可迁移组件与权限错配。** 可以只从 goal patch 构造目标外观/对象关系表示，再和当前 patch 中实际可见结构做局部比对；“完全匹配才进入成功确认”也可作为成功判定门。若做四方向动作，必须把图匹配输出限制为当前允许的四个动作候选。完整 UniGoal 需要在线场景图、对象识别和关系推理；本题只准当前 patch、goal patch、position、visited，不能读取邻格/整图/未访问 embedding，因此不能照搬其全场景图或位置推断流程。是否允许从已访问 patch 保存视觉特征需由题设状态接口明定，本条不假设允许。

**官方代码。** 已核实公开且作者维护的 [bagh2178/UniGoal](https://github.com/bagh2178/UniGoal)。README 标明无需训练并支持 instance-image-goal（2025-04-06 更新）；代码树含 src、configs、README、依赖等。仓库曾在 2025 年分步补充支持；结论是“公开代码已找到，image-goal 支持有作者 README 证据”，不是“已复现或已验证能直接运行本题”。

## R02｜VTM-Nav: Harnessing Cross-Episode Experience for Object-Goal Navigation with Hierarchical Visual-Topological Memory

**原始来源。** [arXiv HTML 全文](https://arxiv.org/html/2607.14514)；[arXiv:2607.14514 摘要页](https://arxiv.org/abs/2607.14514)。论文 v1 提交 2026-07-16，v2 为 2026-07-24；本次未找到正式会议/期刊接收信息，故截至访问日标记为 arXiv 预印本、同行评审状态未核实。未找到可确认的作者官方代码 URL。

**作者。** Xiaoran Xu、Yupeng Wu、Tianyu Xue、Yifan Xu、Xuanran Dong、Xiaoshan Yang、Changsheng Xu。原文标出 Xu 与 Wu 共同一作、Yang 通讯作者。

**任务、观察与动作。** 室内 ObjectNav：每个 episode 是独立起点、单一物品类别目标；同一场景的后续 episode 可复用此前自采集的场景记忆。标准配置是 egocentric RGB-D 640×480、水平视场 79°、相机下俯 14°；用估计位置和朝向、当前 VLM 语义观察，以及由深度生成的可导航候选。每步候选以相对距离/角度表示，episode 比较设置限制 40 interaction steps。该输入明显比题设丰富。

**训练、记忆与规划。** 固定 Qwen3-VL-Plus 与导航组件，不更新参数。按房间拓扑维护粗层索引，并在房间名下维护视觉对象记录；区分房间内直接看到的物体与从门口/开口远处看到的线索，保存置信度、视角、摘要、搜索结果和成功后的接近方向。每个 episode 中可更新记忆；episode 之间保留同一 scene 的视觉拓扑记忆。用语义、拓扑邻接、位置一致性重新定位，先按目标找候选房间，再查其本地目标记录，必要时在已观测拓扑上做 BFS；记忆仅作为软先验，且只能重排当前 RGB-D 观察生成的动作候选。执行守卫处理停滞、振荡、阻塞、过早停止。

**数据与结果。** HM3D v0.1、HM3D v0.2、MP3D。匹配的 40-step 控制块中，VTM-Nav 的 SR/SPL 分别为：HM3D v0.1 59.6/31.8，HM3D v0.2 72.0/31.5，MP3D 44.3/16.2。对应 memory-reset WMNav* 为 55.0/31.7、70.0/30.0、43.5/15.6。作者另报告与 HM3D 上自由文本持久记忆比较，VTM-Nav 高 3.1 和 5.5 SR 点。论文明确提示，训练预算与数据协议不一致的文献行只作上下文，直接结论限于相匹配的 40-step 比较。

**可迁移组件与权限错配。** 可借鉴“visited/成功轨迹作为带来源和置信度的记忆”“先验证当前可见证据，再允许记忆轻微偏置动作”“记忆建议不得产生当前接口不存在的动作”。不过本题只保证当前 patch、goal patch、position、visited；VTM 需要 RGB-D、朝向、局部动作候选、语义标签和过去视觉摘要，也把物体类别当目标。其跨 episode 记忆设定与单次 10-step 5×5 搜索不等价。若旧 patch embedding 被禁止，不能移植其视觉记录，只能用许可的 visited 状态和当步图像。

## R03｜Dual-Layer Semantic-Spatial Belief Mapping for Aerial Object Goal Navigation（AeroBelief）

**原始来源。** [arXiv HTML 全文](https://arxiv.org/html/2609.08164)；[arXiv:2609.08164 摘要页](https://arxiv.org/abs/2609.08164)；[作者官方 GitHub](https://github.com/Shawnjx/AeroBelief)。论文提交于 2026-09-08。arXiv 页面只列预印本；截至访问日没有从原始论文确认正式发表/接收。第三方页面称其“submitted to IEEE TMM”，但未获原始来源确认，故不把该投稿状态当事实。

**作者。** Jianqiang Xiao、Xiang Deng、Yuexuan Sun、Yanjin Wu、Wenbiao Yan、Liqiang Nie。

**任务、观察与动作。** UAV 在未知户外场景寻找并接近自然语言描述的物体目标；不是 image-goal。原文称依赖 onboard visual observations，并通过 UAV pose、视向、距离/深度可见性等信息把观测投影到空间记忆；规划器接收热点及以 UAV 航向对齐的区域方向/距离提案。完整 primitive action 集与传感器接口在本次原文抽取中未核实，不能擅自假定是四动作离散格点。

**训练、记忆与规划。** 提出 object-conditioned VLM 判断及保守证据筛选；双层空间 belief：intuition 层累积宽松上下文线索、服务探索，并随时间衰减；evidence 层只保留经资格筛选的较强目标证据；按证据量门控融合，再取热点。另有 quadtree coverage 和 egocentric regional guidance，区域探索分数不依赖语义分数，并有跨步 temporal commitment，以免语义热点过早压制覆盖。论文摘要未明确宣称“strictly training-free”；可确认是 VLM/MLLM 推理与记忆/规则组合，是否含任何任务专属训练在本次可读材料中记为未核实。

**数据与结果。** UAV-ON benchmark；摘要报告 overall SR 21.61%、OSR 35.57%、SPL 10.62。论文结论注明当前评估仅在 AirSim simulation，sim-to-real transfer 未验证。摘要与消融文字出现不同配置/数值，本记录只列摘要的 overall 数字，不把消融数字混为同一主表。

**可迁移组件与权限错配。** 这是新增候选里最贴近“航拍目标搜索”的结构参考：可以把 intuition/evidence 分开；把固定的四方向扫描/覆盖收益与 goal-match 分数分别累计；将单步证据作为热点，再让动作评分在覆盖探索和目标接近之间折中。但本题目标是目标图片，需用当前 patch 与 goal patch 的视觉相似度/局部结构，而非目标语义类别；题设未提供 UAV 深度、视角、姿态控制或可查看候选航拍帧。其世界坐标语义空间图不能直接作为当前任务的观测或地图输入。

**官方代码状态。** 官方仓库 [Shawnjx/AeroBelief](https://github.com/Shawnjx/AeroBelief) 已核实确为论文仓库；截至 2026-09-30 仓库仅 README，声明“source code, configuration files, and evaluation scripts will be released upon acceptance”。因此是官方入口存在、源码尚未公开，不应写成“代码已开源”。

## R04｜Select2Plan: Training-Free ICL-Based Planning through VQA and Memory Retrieval

**原始来源。** [作者项目页](https://lambdavi.github.io/select2plan/)；[arXiv:2411.04006](https://arxiv.org/abs/2411.04006)；DOI [10.1109/LRA.2025.3606790](https://doi.org/10.1109/LRA.2025.3606790)。arXiv 首次提交于 2024-11-06；正式发表于 2025 IEEE Robotics and Automation Letters, 10(11), 11267–11274。因此以 2025 期刊发表计，不称作 2024 同行评审成果。

**作者。** Davide Buoso、Luke Robinson、Giuseppe Averta、Philip Torr、Tim Franzmeyer、Daniele De Martini。

**任务、观察与动作。** 训练免微调的高层机器人导航规划，覆盖 FPV 第一人称和 TPV 第三人称摄像头，目标包括新场景中的未见物体。摘要称可使用简单 monocular camera；将 VQA 定位/选取图像空间点，再映射为计划/动作执行。不是以目标照片作条件的 image-goal 导航。低层动作控制细节应以论文定义为准；本次可读摘要未取得精确离散 action set。

**训练、记忆与规划。** 无 task-specific fine-tuning；通过结构化 VQA 将导航改写成 VLM 擅长的视觉问答，再从带标注视觉样例的 memory bank 做 ICL。FPV 与 TPV 使用不同记忆库，可含不同模拟环境/目标物、在线视频或人类演示。需注意“训练免微调”不等于无数据：依然依赖示例或在线采集轨迹。规划由 VLM 输出文字/图像空间计划，再映射并执行动作。

**数据与结果。** 评估 FPV 与 TPV 场景。arXiv 摘要声称 TPV 相对 baseline VLM 约提升 50%，FPV 仅用 20 个示例便达到与训练模型相当表现；作者项目页给出的增幅描述有不同版本/口径，因此不据此作跨论文排行榜。核心启示是少量示例检索和结构化动作查询可能帮忙校正零样本 VLM，但不是证明只凭当前 patch 可到达。

**可迁移组件与权限错配。** 可以把目标 patch 作为查询条件，并让 VLM 在当前 patch 上对四个合法动作分别给出可解释候选分数；检索历史 action-outcome 示例也可能降低回环。前提是题设允许缓存 visited 单元的观测/动作结果。Select2Plan 原法依赖 annotated memory bank、示例和 image-space keypoint；若只能保存 visited 布尔状态，则其视觉经验检索不可用。它用开放式几何点与语言物体目标，也不等价于 5×5 离散转移。

**代码状态。** 作者项目页可访问，并列出 Paper / arXiv / Code 入口；本次没有核实到可确认的官方 GitHub 代码 URL，故记为代码 URL 未核实，不据此断言无代码。

## 独立性与和主清单的重叠

- R01 与 IGL-Nav、AnyImageNav 在“目标是图片”的任务定义上重叠，但技术路线不同：UniGoal 是 zero-shot 统一目标/场景图与阶段式匹配；IGL-Nav 是 3D Gaussian localization；它们是独立论文。UniGoal 仓库已公开，是本轮可确认的图像目标零样本开源候选。
- R02 与 LM-Nav/VLFM/DUET 等共享 VLM/拓扑或记忆主题，但研究问题为同场景跨 episode 的持久经验复用，不能合并为已有论文的另一个项目名。
- R03 与 GOMAA-Geo、GeoExplorer、DynCur-Geo 及已在主清单覆盖的 SeePointFly 同属 aerial target/search 邻域，问题核心是自然语言目标的空间热点与覆盖规划；不是 image-goal 论文。作为新增方法，它强调弱上下文/强证据两类 belief 分层和与语义分数独立的覆盖引导。它和相关 aerial benchmark/任务不应被称作 image-goal 证据。
- R04 与 HRNav 都用高层 VLM 规划，但 Select2Plan 声称无需 task-specific fine-tuning，并依赖样例检索；HRNav 是训练/在线强化学习层级 image-goal 方法。可分开看，但 Select2Plan 不是 image-goal 目标接口。

## 索引渠道可用性审计

截至 2026-09-30，仅对网页公开检索和直接访问能力作了测试，不对索引收录作绝对判断。

| 渠道 | 本次观察 | 可作出的结论 |
|---|---|---|
| arXiv / CVF / GitHub | 可通过公开 web 查询读到 arXiv 元数据和 HTML；CVF 搜索结果返回 UniGoal 官方 PDF 及摘要，但 PDF/HTML 直取失败；GitHub 可读 UniGoal 与 AeroBelief 官方 repo | 当前最可靠的可核验主证据。CVF 直取失败不等于论文不可公开访问；UniGoal GitHub 与 AeroBelief repo 状态能直接核验 |
| Semantic Scholar | 对 VTM-Nav 与 UniGoal 做 site 限定搜索无论文页命中；直接打开公开搜索 URL 被 web 工具拒绝 | 索引/站点状态未知；不能据此说没有收录 |
| OpenReview | 对 VTM-Nav、AeroBelief 做 site 限定搜索未命中；直接搜索 URL 被 web 工具拒绝 | 这两篇原始记录是 arXiv；本次没有发现 OpenReview 同名正式投稿记录，但 OpenReview 索引覆盖与直接访问均未核实 |
| Google Scholar 公开页面 | site:scholar.google.com 论文查询无命中；直接 scholar 查询页不可通过 web 工具访问 | 不能判断 Scholar 是否收录，也未取得引用数/版本合并信息 |
| Papers with Code | site:paperswithcode.com 查询 UniGoal、VTM-Nav、AeroBelief 未获得目标论文页；直接 PWC 搜索页不可访问 | 收录/可用性未核实。不能把无搜索结果表述为无条目或无代码 |
| DBLP（辅助） | UniGoal 结果显示 CVPR 2025 正式会议论文；VTM-Nav 记录归入 CoRR/arXiv | 只作书目交叉校验；不能代替论文原文或 peer-review 状态证明 |

网页内容可能因工具抓取/站点限制而不完整。上述检索记录是“公开网页工具在当日的可观察结果”，不代表相应学术平台的数据库完整性判断。

## 检索日志

### 搜索批次 1｜2026-09-30

- "training-free" image-goal navigation 2025 2026 retrieval memory VLM
- "image-goal" navigation 2025 arXiv training-free goal memory
- "aerial" image-goal navigation target search VLM 2025 2026
- UniGoal arXiv 2503.10630 official code repository

发现 UniGoal、VTM-Nav、HRNav（已有主清单）、AeroBelief、Select2Plan 等候选；后两者/部分结果由通用学术搜索产生，随后只以原文/作者页面核实。未将 OpenVLM-Nav、EvolveNav 等仅凭非官方摘录列作核心新增。

### 搜索批次 2｜2026-09-30

- site:arxiv.org 2609 aerial object goal navigation belief mapping training-free
- site:openreview.net "training-free" visual navigation 2026 memory retrieval
- "VTM-Nav" 2607.14514 GitHub code
- "UniGoal" CVPR 2025 github bagh2178

核实 AeroBelief 与 VTM-Nav 的 arXiv 页面；找到 UniGoal 作者 GitHub。OpenReview 定向检索未出现这两篇论文结果。

### 搜索批次 3｜2026-09-30

- "UniGoal: Towards Universal Zero-shot Goal-oriented Navigation" arxiv authors method instance image goal CVPR 2025
- "Select2Plan" training-free ICL-based planning VQA memory retrieval ICRA 2026 authors benchmark
- "VTM-Nav" GitHub code
- "AeroBelief" 2609.08164 code Github

确认 UniGoal 的 CVPR 官方 PDF 与作者列表；查到 Select2Plan arXiv 摘要及作者项目页，确认实际刊物是 RA-L 2025（其 2026 ICRA 页面/列表出现不能据为 ICRA 论文）；AeroBelief 页面返回摘要。

### 搜索批次 4｜索引与官方代码，2026-09-30

- "VTM-Nav" 2607.14514 GitHub official implementation
- site:semanticscholar.org/paper "VTM-Nav" 2607.14514
- site:openreview.net "VTM-Nav" 2607.14514 OR "AeroBelief" 2609.08164
- site:paperswithcode.com "UniGoal" CVPR 2025 navigation

未确认 VTM-Nav 官方代码；Semantic Scholar、OpenReview、Papers with Code 定向 query 均无目标论文搜索命中。

### 搜索批次 5｜索引与 AeroBelief 代码，2026-09-30

- site:semanticscholar.org "UniGoal: Towards Universal Zero-shot Goal-oriented Navigation"
- site:scholar.google.com "UniGoal" "Towards Universal Zero-shot Goal-oriented Navigation"
- site:paperswithcode.com "VTM-Nav" OR "AeroBelief" navigation
- "Dual-Layer Semantic-Spatial Belief Mapping" GitHub OR code

定向索引查询没有得到目标平台论文结果；第三方聚合页面提及 AeroBelief 代码，但以作者 GitHub 实查纠正为“仅 README，代码待接收后发布”。第三方代码标签不作为开源事实。

### 全文/原始来源抓取批次 1

- [VTM-Nav arXiv HTML](https://arxiv.org/html/2607.14514)：读取标题/作者、问题定义、训练免微调设定、RGB-D/action候选、分层记忆方法、评测数据与匹配控制表。
- [AeroBelief arXiv HTML](https://arxiv.org/html/2609.08164)：读取标题/作者、摘要、目标类型、评测和模拟器限制。
- [UniGoal CVPR 2025 官方 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.pdf)：检索结果可抽取官方摘要；直接全文抓取失败。

### 全文/官方代码状态抓取批次 2

- [AeroBelief arXiv HTML](https://arxiv.org/html/2609.08164)：读取系统摘要、消融/结论中 simulation-only 限制、论文引用列表。
- [VTM-Nav arXiv HTML](https://arxiv.org/html/2607.14514)：读取检索、guard、40-step 控制、benchmark 数字与实验限制。
- UniGoal CVF PDF 重试仍失败；故该论文精确 benchmark/数字保留未知。

### 全文/平台访问抓取批次 3

- [CVF UniGoal HTML](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.html)：403；不是论文不存在的证据。
- [AeroBelief 作者官方 GitHub](https://github.com/Shawnjx/AeroBelief)：公开可读；2 commits，仅 README，明确代码待论文接收后发布。
- 直访 Google Scholar、Semantic Scholar、OpenReview、Papers with Code 查询页：全部被当前 web 工具拒绝，状态标为访问工具受限；未调用 browser-harness，因为均是普通公开页面，不涉及 JS 交互。

## 尚未确认的空缺

1. UniGoal 官方全文表格中的精确 benchmark、数值、传感器和 primitive-action 配置：CVF 直取失败，本轮保持未知。
2. VTM-Nav 是否有作者未索引/未公开到 GitHub 的官方实现：搜索与论文页未找到，状态未知。
3. AeroBelief 截止日之后的代码发布、接收状态及真实无人机结果：本次访问日之前的官方 repo 声明仍是未来发布，sim-to-real 未验证。
4. Select2Plan 的官方代码 URL：作者页有 Code 入口，但本次未跟到可核实仓库；按未核实记录。
5. 各目标索引服务对四篇论文的实际收录状态与引用数：访问渠道受限；不能用 site 搜索无命中推断未收录。

