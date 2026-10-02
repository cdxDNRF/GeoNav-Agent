# 多Agent导航与角色来源补充（Area：同一执行体内协作）

检索日期：2026-10-01（Asia/Shanghai）。本文件只补充导航内多Agent、角色协作、反思/核验与视觉落地的一手研究；不覆盖2026-09-30旧综述，也不执行项目训练或导航评测。

## 三句结论

现有单执行体文献里，最接近“跨步反馈真正进入下一次决策”的是ReflectVLN：执行Agent输出子目标完成/偏航状态，触发意图Agent改写子目标，后续执行调用确实读取该交接；但论文当前核对到的正文未确认偏航监督是否依赖GT路线，而其双模型训练和室内指令式VLN也不符合本项目的权限与任务定义。DiscussNav提供最完整的多专家感知、历史摘要与分歧仲裁证据，但每步用多模型/多次推理、读取全景候选方向视图，且没有等token/调用的单Agent反思对照；其收益不能证明角色本身带来等成本增益。对本项目而言，应先解决可靠cue极少出现时的覆盖/预算失败；只在公开可观测的停滞或偏航事件触发、并确保跨步消息在随后的真实调用中被消费，才值得再检验“探索规划器＋证据维护/反证监督器”是否超过等预算单Agent。

## 本项目边界与迁移判断

项目执行体给定目标航拍patch、当前patch/位置、visited、剩余预算；目标位置和未访问图像不可见。用户交接的本地诊断补充：S4在500题×3次中失败343次，行动前仅11次有真实邻接机会；SwissView失败198次中仅1次有邻接机会。此为评测后机会覆盖诊断，不是因果上界，也不能用于在线触发。当前Edge邻接线索有真阳性也有false accepts；不能因为VLM回unknown就否决强cue。20题云端小试全输出unknown/不改动作，触发3题均为原基线成功题，唯一失败题未触发，交换后的跨轮消息未被后续调用实际使用。

因此，能迁移的是公开状态上的预算风险、重复覆盖/停滞监控、带来源的observed-only证据账本，和保证后续调用消费消息的交接协议；不能迁移从目标真位置/GT路线推算的在线状态、未访问/全景候选图像、室内房间常识作为目标证据、多个实体机器人并行增加总搜索动作、或未做等成本对照的“多轮输出更好”。

## 来源卡

### A01 — Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions

**Long, Yuxing; Li, Xiaoqi; Cai, Wenzhe; Dong, Hao. 2024，IEEE ICRA 2024，pp. 17380–17387（T1）。** [arXiv论文](https://arxiv.org/abs/2309.11382)；[项目页](https://sites.google.com/view/discussnav)；[作者GitHub](https://github.com/LYX0501/DiscussNav)。arXiv初稿2023-09-20；会议归属和页码由论文/会议条目核对，DOI未核实，不猜。仓库公开，README有任务数据；列出Ubuntu 18.04/Python 3.8/PyTorch 1.13.1、Matterport3D Simulator、RAM、InstructBLIP依赖；未见版本化release，未运行代码。

**任务、输入和动作。** R2R室内Matterport3D，给自然语言路线指令；每步通过全景12方向场景/候选视点进行方向选择。在TurtleBot 4 Lite上测试20条指令，每步按预测方向前进0.5m（有障碍则缩短），局部雷达仅避障、不建图。

**角色、消息、触发和仲裁。** 以GPT-4为主Agent和指令分解/地标/完成估计/决策测试等prompt角色；图像侧用InstructBLIP FlanT5-XL和RAM-14M。角色按问题模板接受指令、场景图/对象图或导航历史，输出地标、场景描述、已执行/进行中/待执行动作、或对候选动作的评议，再把结果交主Agent；轨迹摘要与完成估计提供跨步历史，但论文未定义每个专家拥有可独立持久更新、互相盲化的上下文状态，属于角色化子任务调用。指令解析启动时做；视觉感知与完成估计每一步做；GPT-4每步采样5条预测，方向一致就直接执行，分歧才调用thought-fusion与decision-testing专家仲裁。因而专家咨询是常驻工作流，分歧仲裁才是条件触发。

**证据与限制。** R2R val-unseen 783条、11个未见环境：DiscussNav SR=43、SPL=40、NE=5.32；NavGPT对应34、29、6.46。真实TurtleBot 20条指令SR 25%，NavGPT 10%，预训练DuET 0%。按专家消融SR/SPL：移除指令专家38.77/35.04；视觉专家32.46/29.01；完成估计32.65/30.13；决策测试37.50/33.31；完整43/40。消融在72个环境各抽1条路径，并非等成本因子实验。比较未控制token、API调用数或同调用上限的单Agent reflection；每步咨询和n=5采样本身增加推理量。完成估计依据轨迹里的已过地标、方向变化和动作历史，不是任务真值；但它复核的历史可能已经含有感知误差。论文用定性例子展示地标分解纠错，没有给感知幻觉率或独立视觉真值检查。全景12方向/周边候选图比本项目“只给当前patch/位置、目标图和最多两张已访问图”的权限宽，不能直接外推。

### A02 — ReflectVLN: Training Vision-Language Navigation Agents with Reflective Reasoning

**Wang, Jiahang; Yang, Yirong; Zhu, Yanqing; Luo, Minghua; Xie, Shichao; Liu, Fei; Xu, Mu. 2026，arXiv:2607.12680v1（T1P，2026-07-14）。** [arXiv HTML/论文](https://arxiv.org/html/2607.12680)；[作者官方GitHub](https://github.com/AIprogrammer/ReflectVLN)。截至检索日仍按arXiv预印本列出。GitHub公开但检视到的版本只有README/1 commit，README TODO仍列待发布训练/推理代码、数据、权重、环境；不是可直接复现的代码包。

**闭环定义。** R2R-CE/RxR-CE连续室内VLN，instruction、当前及历史单目RGB、当前语言子目标输入；execution Agent输出粗路线、短期waypoint动作序列和状态token；执行Agent每个动作时再次运行。两个独立参数的Qwen2.5-VL-3B，分别初始化/训练。Intention输出语言子目标；执行侧状态token为A=继续执行、R=子目标完成并请求常规新子目标、C=off-track并请求纠正性反思。R/C时Intention接收指令+视觉历史并更新子目标，下一次Execution条件化于新子目标。论文报告平均约每9.63步才调用一次高层意图模型。token定义和在线传递由算法明确；但本次核对的原文未确认C到底按“执行偏离自定子目标/路线”还是GT示范路线距离计算，也未确认是否有simulator reward作为直接触发器。迁移前必须查清标签定义；只可把运行时由当前RGB/历史/子目标预测的状态当作带误差的公开信号，不能迁入依赖GT的位置偏差阈值。

**训练与结果。** 两阶段训练：Execution先在expert trajectory上做Action-CoT预训练；再用expert demonstration与one-round reflective recovery trajectory联合训练意图/执行角色。数据约1.5M专家样本+100k reflection样本；训练用32张H20，约16小时+60小时。R2R-CE val-unseen：SR62.8/SPL58.5；RxR-CE：SR66.0/SPL57.2。相对DualVLN（R2R 64.3/58.5；RxR 61.4/51.8），R2R SR低1.5点而SPL持平，RxR SR高4.6点、SPL高5.4点。其CorrectNav*是另一套训练/数据来源下的单轮自纠训练比较，不是等调用/等token的在线单Agent reflection控制；不能把双模型效应与训练数据、恢复数据、数据量及Action-CoT分离。其子目标完成/偏航与本项目离散地理patch目标、预算覆盖失败不同。

### A03 — Plan Verification for LLM-Based Embodied Task Completion Agents

**Hariharan, Ananth; Dongre, Vardhan; Hakkani-Tür, Dilek; Tur, Gokhan. 2025，NeurIPS 2025 Workshop: Embodied World Models for Decision Making（T1P，不是NeurIPS主会论文）。** [arXiv论文](https://arxiv.org/abs/2509.02761)；[OpenReview论文PDF](https://openreview.net/attachment?id=9AOGlpRXqq&name=pdf)；[论文提及的代码仓库](https://github.com/AnanthHariharan/Task-Agents)。PDF注明workshop和代码地址；仓库当前文件状态未独立审计。

**协议与作用。** TEACh手工标注100 episodes、15类家庭任务。Planner先生成文本动作序列，Judge逐步标记冗余/无关/矛盾/无依据动作并附解释，Planner据此修改；直到无异议或达预设轮数。Judge只看文本计划，没有环境模拟器或视觉输入。四个LLM（GPT o4-mini、DeepSeek-R1、Gemini 2.5、LLaMA 4 Scout）按人工标注错误评precision/recall，最高recall90%、precision100%；96.5%序列在3轮内收敛。它验证的是计划清理和标注，不是在线导航成功恢复；既非动作前门控，也无matched-token/单Agent反思实证。可借鉴“证据来源+理由+可执行修改”格式，不可当作具备视觉反证能力的导航监督器。

### A04 — Think Twice, Act Once: Verifier-Guided Action Selection for Embodied Agents (VeGAS)

**Singhi, Nishad; Bialas, Christian; Jauhri, Snehal; Prasad, Vignesh; Chalvatzaki, Georgia; Rohrbach, Marcus; Rohrbach, Anna. 2026，CVPR 2026 Findings（T1）。** [arXiv论文](https://arxiv.org/abs/2605.12620)；[作者项目页](https://nishadsinghi.github.io/vegas/)；[作者GitHub](https://github.com/nishadsinghi/vegas)。仓库README截至检索日写明“Code Coming Soon”。

**机制与结果。** Habitat和ALFRED具身推理任务。基础MLLM在每一步采样多个候选动作，另一个经专门训练的生成式verifier依据当前观测与候选来选动作；verifier不是可持久维护地图/证据的独立角色，也没有跨步文字对话，作用是每步动作仲裁。论文用自动构造的失败案例课程训练verifier；发现直接拿未训练MLLM作verifier没有收益。最难的多物体、长时程任务报告相对强CoT基线最高36%相对增益。没有核到等token/等调用预算的单Agent reflection对照，候选采样+verifier必有额外推理成本。可迁移的是“对合法动作候选做经错误数据训练的动作核验”假设，而非其增益数字；本项目需先确认cue误接收、动作错选是否主失败，而非目标邻接机会缺失。

### A05 — CA-VLN: Collaborative Agents in MLLM-Powered Visual-Language Navigation

**Zhu, Ruolin; Li, Shaobin; Zhu, Zixing; Jia, Jing; Yang, Min. 2026，Sensors 26(4):1254（T1）。** [MDPI论文与版本页](https://www.mdpi.com/1424-8220/26/4/1254)；[DOI](https://doi.org/10.3390/s26041254)；[PMC全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12944077/)。本次未找到论文指定的官方代码仓库，故不推断已开源或未开源。

**架构。** R2R、REVERIE、SOON，Matterport3D离散候选视点；CLIP ViT-B/16视觉、LLaVA2-7B构成Knowledge Reasoning/Hierarchical History两个部件，指令、当前候选视图和已访问轨迹参与预测。Knowledge部件生成按步指令、实体/关系知识，History部件维护分层视点/路径文本、episodic graph并做记忆检索；知识/历史作为结构化引导和特征传给融合/动作头，History Agent输出最终动作。这是训练过的模块/状态流水线，不是两个独立对话上下文轮流争论；跨步反馈来自累积访问节点与历史记忆，按每步更新，没有偏航事件门控。训练时Knowledge与History用ground-truth轨迹由MLLM产生指导，再LoRA训练（每阶段可训练参数≤10M）；冻结两个agent后端到端训练融合/预测组件（≤10M），模型7B。

**数字和边界。** R2R val-unseen表中SR/SPL=73.31/61.95，对DuET的71.52/60.42；test-unseen=70.27/60.31，对DuET 69.25/58.68。论文报告逐步推理成本比baseline高约15%（History图注意力约4ms、知识检索约12ms）。这是完整训练模型效果和结构消融，不是等预算的独立Agent效果；室内房间知识、ground-truth轨迹监督、全景候选视点和离散自然语言路线与本项目“目标参考图+局部切图、隐藏未访问区域”有权限与任务错配。论文没有无人机实飞验证。

### A06 — Towards a Science of Scaling Agent Systems

**Kim, Yubin; Gu, Ken; Park, Chanwoo; Park, Chunjong; Schmidgall, Samuel; Heydari, A. Ali; Yan, Yao; Zhang, Zhihan; Zhuang, Yuchen; Liu, Yun; Malhotra, Mark; Liang, Paul Pu; Park, Hae Won; Yang, Yuzhe; Xu, Xuhai; Du, Yilun; Patel, Shwetak; Althoff, Tim; McDuff, Daniel; Liu, Xin. 2025–2026，arXiv:2512.08296当前修订版（T1P；官方仓库称仍在Nature Machine Intelligence审稿修订）。** [arXiv论文](https://arxiv.org/abs/2512.08296)；[官方GitHub](https://github.com/ybkim95/agent-scaling)。代码公开且有版本归档；仓库说明稿件仍在审稿修订，使用原arXiv标题。

**实验与可迁移性。** 260个配置，6个agent benchmark、single/independent/centralized/decentralized/hybrid五种架构、三类LLM，控制工具、prompts与compute。不同任务相对single-agent从可分解financial reasoning +80.8%到sequential planning −70.0%；六bench预测模型交叉验证R²=.373，held-out架构选择87%正确。结果支持强single-agent、重工具、严格序列任务不应默认堆agent，也显示带集中核验拓扑较少传播错误。它不是视觉导航研究，数字不能当本项目预期效应；但这是成本和单Agent强度控制最相关的方法学反证来源。

### A07 — Why Do Multi-Agent LLM Systems Fail? (MAST)

**Cemri, Mert; Pan, Melissa Z.; Yang, Shuyi; Agrawal, Lakshya A.; Chopra, Bhavya; Tiwari, Rishabh; Keutzer, Kurt; Parameswaran, Aditya; Klein, Dan; Ramchandran, Kannan; Zaharia, Matei; Gonzalez, Joseph E.; Stoica, Ion. 2025，arXiv:2503.13657（T1P；此处未核实正式会议版本）。** [arXiv论文](https://arxiv.org/abs/2503.13657)；[作者项目页](https://multi-agent-systems-failure-taxonomy.github.io/MAST/)；[代码与标注数据](https://github.com/multi-agent-systems-failure-taxonomy/MAST)。

基于七个开源MAS、1,642条系统轨迹，提出14类失败并用LLM-as-a-Judge扩展标注。三类包括system/specification、inter-agent misalignment、task verification/termination；示例有重复步骤、上下文丢失/忽略Agent输入、reasoning-action mismatch、无/不完整验证。两项prompt/topology干预最多改善15.6%，但效果依任务/模型不同；论文记载GPT-4下新topology小增益Wilcoxon p=.4不显著，GPT-4o情形p=.03。非导航数据、未做本项目同预算比较。最可用的是要求每条消息被实际下游使用、能改变动作且对失败/成功影响分别归因的审计清单，而非照搬其平均成功率。

### A08 — FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation

**Zuo, Jing; Mu, Lingzhou; Jiang, Fan; Ma, Chengcheng; Xu, Mu; Qi, Yonggang. 2026，CVPR 2026（T1）。** [CVPR论文PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zuo_FantasyVLN_Unified_Multimodal_Chain-of-Thought_Reasoning_for_Vision-and-Language_Navigation_CVPR_2026_paper.pdf)；[arXiv:2601.13976](https://arxiv.org/abs/2601.13976)。官方代码/项目链接本轮未核实。

它是在LH-VLN连续长时程室内语言导航上的单策略训练路线：训练时用text/visual/multimodal CoT对齐与Visual Autoregressive latent压缩，把推理内化；推理时直接instruction-to-action，无多角色消息/额外反思token。论文摘要报告相较显式CoT方法推理延迟降低一个数量级并改善成功率/效率。作为对照，它提醒协作可替代方案是把稳定的历史/推理表征训练进单Agent；本次不引用未从CVF正文表格核实的精确数字。任务仍与航拍目标定位不同。

### A09 — Enhancing Multi-Robot Semantic Navigation Through Multimodal Chain-of-Thought Score Collaboration (MCoCoNav)

**Shen, Zhixuan; Luo, Haonan; Chen, Kexun; Lv, Fengmao; Li, Tianrui. 2025，AAAI 39(14):14664–14672（T1）。** [AAAI论文页/DOI](https://ojs.aaai.org/index.php/AAAI/article/view/33607)；[arXiv:2412.18292](https://arxiv.org/abs/2412.18292)；[作者GitHub](https://github.com/FrankZxShen/MCoCoNav)。

HM3D/MP3D household ObjectNav，两个及以上物理机器人各看本体RGB-D，各自规划；共享global semantic map、frontier/history node分数以及跨视图语义是通信桥。其结果证明的是多实体并行、共享地图和动作分区，不是一个导航执行体中的Agent协作；即使借鉴共享visited/候选证据状态，也必须按本项目同总动作数比较，不能按每机同预算直接外推。故仅作为scope guard，不进入同执行体收益排序。

## 可移植模块与下一轮证据要求

1. **跨步证据维护**：复用当前可见的目标cue、counterevidence、patch/位置、图像哈希、观测来源和实际动作结果，独立记录“观测”和“推断”。屏蔽目标隐藏区；同一图不重复计证据。CA-VLN说明累积visited history可影响后续导航，但其训练/全景权限不同，本项目应先做单Agent结构化记忆基准。
2. **受限探索规划**：以visible visited/剩余预算/停滞和可达性计划新区域，候选分配不能用真实目标距离或隐藏图。失败主因若是没有邻接cue，第二角色应有覆盖增益可测，而非只对cue作文本复核。
3. **偏航监督与反证核验**：仅允许由执行时可见状态触发，例如重复访问、预算风险、计划步数内没有新增覆盖、目标cue与路径证据冲突；执行监督器需给可复核reason/evidence_id/action，且不能因VLM unknown否决已接受强cue。ReflectVLN的A/R/C状态交接可借鉴字段设计；在厘清GT路线监督前不可照抄C的触发定义。
4. **真实消息使用门槛**：每条handoff记录发送时间、公开字段、接收Agent输入hash、触发事件、是否改动作、后续结果；如果消息只能由后续事件触发，就必须确保该事件在episode内真正引发模型调用并读取该版本。按来源图/题分配，分别统计修复baseline失败、损伤baseline成功、有效改变动作、总动作、调用/tokens、SR/SG/Q。
5. **同预算基线**：强单Agent加完全相同的结构化记忆；单Agent两次反思；双角色（状态监控/证据管理+探索规划）；固定调用版本与事件调用版本。除每边相同最大调用数外，报告同实耗token/调用的敏感性对照；避免DiscussNav和VeGAS式每步额外采样被解释为角色贡献。

## 独立核验后的更正（交付版优先于上面的首轮提取）

2026-10-01，按 `15_多Agent专项引用核验_v1.md` 与主Agent补检限缩以下事实：

- A01：ICRA2024归属由作者和索引定位，DOI为 `10.1109/ICRA57147.2024.10611565`；正式IEEE页本轮未成功读取，venue核验保留PARTIAL，不把作者/索引当作官方会刊已核。
- A02：ReflectVLN在线off-track C由执行模型基于在线图像历史预测；训练C/reflection使用参考GT轨迹、waypoint以及能访问环境图的特权专家恢复。此前“GT监督不明”已由独立核验补齐；不能把在线非oracle误写成训练无特权信息。
- A04：VeGAS §6.2/Fig.6有匹配LLM调用数的Self-Consistency消融；该零样本verifier实验无改善/略差，fine-tuned verifier有效。此前未确认同调用对照的描述已更新；仍没有与单体reflection全面等实耗总成本的证据。CVPR2026 Findings只按作者声明，官方会刊未闭环。
- A06：v1为180配置/4基准，v3为260配置/6基准；版本数不能混用，旧版约45%等结论不能作导航SR规则。期刊under revision不表示已发表。
- A07：[NeurIPS2025 Datasets and Benchmarks正式记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html)已核。150专家分析与1600+扩展trace分开；正式摘要未核精确1642，不在最终报告补精确值。

最终决策与可执行协议见 `12_DEEP_RESEARCH_多Agent失败介入与同预算协作_v1.md`、`14_多Agent候选冻结实验设计_v1.md`；首轮来源卡保留以显示更正过程。

## 检索日志与引用边界

Search共8批，每批≤4条：DiscussNav/ICRA与同执行体VLN；agent scaling/角色失败；CA-VLN/VeGAS/计划核验；多机器人语义导航边界；FantasyVLN/MAST；ReflectVLN与作者仓库；ReflectVLN/CA-VLN/VeGAS/scale代码状态；CA-VLN期刊元数据。Fetch共5批：首批多arXiv并发open连接失败；之后读取DiscussNav arXiv全文、CA-VLN全文片段、Plan Verification OpenReview PDF、VeGAS/MAST arXiv、ReflectVLN arXiv HTML、Scaling arXiv摘要与作者代码状态、CA-VLN MDPI元数据/版本页。没有浏览器自动化、JS抓取、外部博客/非primary摘要数字。Search result snippets仅作定位；关键数值以论文/出版社页面或论文表为准。文献数字不与本项目SR直接相减：观测、任务、动作预算、训练、环境和成功判据均不一致。
