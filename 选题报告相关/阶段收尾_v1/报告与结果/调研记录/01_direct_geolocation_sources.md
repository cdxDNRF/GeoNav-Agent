# 直接相关文献与来源卡（G01–G12）

截至 2026-09-30。问题对齐：给定同一鸟瞰场景中的目标航片，策略在离散未知网格内只接收局部连续观测，靠动作历史逐步命中。AGL（active geo-localization）与此最接近。需区分“策略看不到整图”与“环境没有地图”：论文通常由环境持有完整航片/单元坐标，用于生成局部观测、训练奖励和合法动作；策略只见当前 patch、目标 cue、历史与格位编码，并不看全域底图。若项目目标图就是同一数据源的航片，跨模态对齐（地面图/文本）不是当前刚需。

## 核心 AGL 论文

### G01 [GOMAA-Geo: GOal Modality Agnostic Active Geo-localization](https://arxiv.org/html/2406.01917)
Anindya Sarkar, Srikumar Sastry, Aleksis Pirinen, Chongjie Zhang, Nathan Jacobs, Yevgeniy Vorobeychik；NeurIPS 2024。论文：[NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2024/file/bd8b52c2fefdb37e3b3953a37408e9dc-Paper-Conference.pdf)；项目/代码：[GitHub](https://github.com/mvrl/GOMAA-Geo)，代码已发布；README 的模型权重仍标为 Coming Soon。数据：Masa 训练，xBD 灾前/灾后迁移，作者 MM-GAG（航空/地面图像、文本目标）。任务/观测：5×5 等非重叠格；只看当前 aerial patch，目标位置隐藏；目标 cue 可为航空图、地面图、文本；四邻接动作；完整全域不可见。状态另含格位编码，历史是因果序列模型隐式记忆；无显式地图、图搜索或在线重规划。架构/训练：Sat2Cap/CLIP 跨模态对比对齐；Falcon-7B 作 GASP 历史编码器，在随机轨迹上监督“哪些动作使曼哈顿距离变小”；冻结编码器/LLM 后用 PPO actor-critic 直接选一步。预算 B=10，SR=预算内到达目标；Masa 测试 C=4…8 SR 为 .409/.506/.717/.803/.785。迁移：本项目可借其“目标图+完整动作/patch 历史+格位位置编码”；同源航空目标时保留图像匹配头、跳过不必要的地面/文本对齐。限制：训练奖励用真目标格；仍逐步四向决策，非训练免疫。证据 Tier-1（同行评审主论文）；2026-09-30；arXiv 全文 §3–5、表1、附录 L–M，NeurIPS PDF 交叉核验；GitHub 仅用于代码状态。

### G02 [GeoExplorer: Active Geo-localization with Curiosity-Driven Exploration](https://arxiv.org/html/2508.00152)
Li Mi, Manon Béchaz, Zeming Chen, Antoine Bosselut, Devis Tuia；ICCV 2025，页 6122–6131。论文：[CVF 官方页/PDF](https://openaccess.thecvf.com/content/ICCV2025/html/Mi_GeoExplorer_Active_Geo-localization_with_Curiosity-Driven_Exploration_ICCV_2025_paper.html)；[官方实现](https://github.com/limirs/GeoExplorer)，代码公开，依赖 GOMAA/AirLoc 预处理；权重单独发布状态未核实。数据：Masa 训练，MM-GAG、xBD、SwissView100/Monuments 迁移/未见目标。观测/动作与 G01 同型（当前 patch、目标多模态、四方向、5×5/B=10）；输入含格位相对位置编码与历史，非整幅地图。方法：Falcon-7B 因果 Transformer 在随机轨迹上同时预测目标导向动作与动作条件下的下一 patch 特征；冻结该序列模型后以 PPO 训练 actor-critic。总奖励=目标距离进展奖励+预测误差好奇心奖励（MSE 或 cosine）；并非外部规划器，也未显式建图。Masa SR（C=4…8）=.4324/.5318/.8156/.9229/.9497；MM-GAG 跨域/模态也提升，论文注明大距离表现较高可能受有限配置与边界目标分布影响。可移植：把“下一观测预测误差”作为训练期 novelty 信号；需检验其是否改善长程覆盖。限制：静态 curiosity 可能在接近目标时继续诱发绕行。证据 Tier-1（ICCV 原文）；2026-09-30；arXiv HTML §3–5、表1–5及补充 S1/S5，CVF 页面核对作者/venue；GitHub 核验发布情况。

### G03 [DynCur-Geo: Dynamic Curiosity Reward Shaping for Multimodal Active Geo-Localization](https://arxiv.org/html/2608.18673)
Yiming Sun, Yang Zhang, Pengfei Zhu；2026-08-19 arXiv v1，24 页；截至检索日未见正式 venue，属未同行评审预印本。paper：[arXiv](https://arxiv.org/abs/2608.18673)；项目页/GitHub：未核实，论文页未给官方代码链接。数据：Masa 137 张 + MM-GAG 47 张训练池，480k PPO 环境步；测试 Masa、MM-GAG 三模态、SwissView、xBD；另测 8×8/10×10/25×25。观测/目标/四动作/隐地图同 G02；Sat2Cap 航片、CLIP ViT-B/32 地面图/文本编码至 512d；格位编码+目标/patch/动作序列进入冻结 GeoExplorer 因果 Transformer，PPO actor-critic 直接动作，没外部搜索。新增只在训练奖励：冻结 dynamics 预测误差做 intrinsic bonus，以训练可知目标距离比例线性衰减其权重（λmin=.405），再加 PBRS（β=.10）；目标距离/奖励不在推理时输入。5×5、B=10 的 MM-GAG 航空目标 SR，C=4…8 为 .268/.370/.681/.843/.923（均值 .617）；C=4 低于 GOMAA .357、GeoExplorer .349，C=6–8 优势明显。可移植的新差异是 distance-gated curiosity，不是泛称 curiosity/PBRS。局限：预印本；摘要称一致提升但按距离并非处处领先；训练仍依赖目标格 oracle。证据 Tier-1P（arXiv 原文）；2026-09-30；全文 §Method、§Experiments 表1–7、补充 §Evidence Boundaries；检索未找到官方仓库，不把 arXiv 发现器当作代码证据。

## 直接前驱与相邻方法

### G04 [Aerial View Localization with Reinforcement Learning: Towards Emulating Search-and-Rescue](https://arxiv.org/abs/2209.03694)
Aleksis Pirinen, Anton Samuelsson, John Backsund, Kalle Åström；arXiv 2022，ML4RS/ICLR 2023 workshop 版本。作者代码：[AirLoc](https://github.com/aleksispi/airloc)。Masa、Dubai、xBD；同场景目标航片，逐格只观当前 patch，四向移动、到达即成功，地图全域不可见。建筑 U-Net 语义分割+当前/目标双路 patch embedder，学习相邻随机 crop 位移，并输出 exploitation prior；加位置编码、LSTM 历史与 RL policy，显式拆远距离探索/近距离利用。5×5/B=10，另测 7×7/B=14；对 Masa、Dubai、灾区优于 heuristic/learnable baselines，且 5×5 训练能迁移至 7×7。可借鉴相似度 prior 与 LSTM 低成本记忆；GOMAA 是其多模态扩展。证据 Tier-1（SAIS 原文 PDF）；2026-09-30；§III、表 I–III；论文全文和官方 GitHub README。

### G05 [SIGN: Safety-Aware Image-Goal Navigation for Autonomous Drones via Reinforcement Learning](https://arxiv.org/abs/2508.12394)
Zichen Yan, Rui Huang, Lei He, Shao Guo, Lin Zhao；IEEE RA-L 2025（DOI 10.1109/LRA.2025.3645668）。[官方代码/检查点](https://github.com/Zichen-Yan/SIGN) 已发布。目标图 ImageNav；Habitat Gibson 72 train/14 test，并测 MP3D/HM3D；不建全局图。当前+目标 RGB 经 ResNet、GRU 历史与 PPO；连续速度动作，另有 depth collision shield；有图像扰动/未来状态预测辅助训练。Gibson SR/SPL=86.3/53.3%，MP3D/HM3D SR=65.4/64.5%。迁移：未来真实 UAV 连续控制、安全过滤；不能直接作为未知 5×5 网格策略。证据 Tier-1（RA-L/arXiv 原文）；2026-09-30；arXiv §III–V/表 I–II、官方 GitHub README。

### G06 [Think before Go: Hierarchical Reasoning for Image-goal Navigation](https://aclanthology.org/2026.acl-long.1352/)
Pengna Li, Kangyi Wu, Shaoqing Xu, Fang Li, Lin Zhao, Long Chen, Zhi-Xin Yang, Nanning Zheng（ACL 元数据名；官方 PDF 文本抽取处另作 “Nanning Zhen”，暂存作者名核验差异，不自行裁定）；ACL 2026。论文：[官方 ACL PDF](https://aclanthology.org/2026.acl-long.1352.pdf)；公开代码未核实。Gibson 训练，MP3D/HM3D 零微调；目标图+当前及历史 egocentric RGB，无全局地图。VILA 高层 VLM 经 SFT 从 767k 分段轨迹生成可执行短期子任务；每约 15 步调用；低层 2 层 GRU+PPO 执行离散动作，回访与过长路径惩罚抑制游荡。SR/SPL：Gibson 94.0/71.2，MP3D 81.4/56.3，HM3D 80.0/49.3。最直接支持“VLM 产 waypoint/subgoal，低层确定执行”，但依赖专门标注与训练，室内路径语义不能原样照搬鸟瞰四格。证据 Tier-1（ACL 主会全文）；2026-09-30；ACL PDF §3–5、表1–5，ACL 元数据核验作者/venue。

### G07 [IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](https://openaccess.thecvf.com/content/ICCV2025/papers/Guo_IGL-Nav_Incremental_3D_Gaussian_Localization_for_Image-goal_Navigation_ICCV_2025_paper.pdf)
Wenxuan Guo, Xiuwei Xu, Hang Yin, Ziwei Wang, Jianjiang Feng, Jie Zhou, Jiwen Lu；ICCV 2025。项目页/仓库：[GitHub](https://github.com/GWxuan/IGL-Nav)，demo 已发，README 明示代码 coming soon。目标是任意视角照片；在线 posed RGB-D stream；四种局部动作（前进/转向/停），探索时增量构建 3D Gaussian 场景表示，不要求初始全图可见；先 3D 特征粗定位、邻近时 differentiable rendering 精定位。离线被动 RGB-D 训练，不是 RL。可借“先粗后精视觉确认”解决跨视角/终点误认；对同源正射航片与纯网格过重。证据 Tier-1（ICCV 原文）；2026-09-30；§3.1–3.3、代码状态取自官方 GitHub。

### G08 [120 Minutes and a Laptop: Minimalist Image-goal Navigation via Unsupervised Exploration and Offline RL](https://arxiv.org/html/2603.26441)
Xiaoming Liu, Borong Zhang, Qingbiao Li, Steven Morad；2026 arXiv 预印本，称已投稿 RA-L，未核到接收。真实项目页链接仍是模板，Code 按钮落到 YOUR REPO HERE/404；不可记为开源。真实轮式/四足室内 ImageNav，不用全局地图；RGB，DINOv3 冻结编码，堆叠四帧；Pink Uniform Noise 无监督采集，HER 目标重标注，TD3+BC 离线 goal-conditioned RL、FQE 选模型。连续速度动作。用不到 2 小时/laptop 完成采集训练部署；真实环境数据量增大时 SR 改善。迁移到网格可做随机游走轨迹+HER+小型离线 policy；项目已有 BC 后，价值在轨迹覆盖/离线重标注，非“再加 BC”。证据 Tier-1P（arXiv 全文）+ Tier-2 作者项目页（仅检查代码按钮）；2026-09-30；全文 §4–5、项目页实测 404。

### G09 [AnyImageNav: Any-View Geometry for Precise Last-Meter Image-Goal Navigation](https://arxiv.org/html/2604.05351)
Yijie Deng, Shuaihang Yuan, Yi Fang；2026 arXiv 预印本。作者项目页：[AnyImageNav](https://yijie21.github.io/ain/)；公开代码未核实。未知室内环境，目标可任意视角；DINOv2 语义相关度构建 BEV relevance/frontier，平时作探索导航，仅在相似区域调用 3D 多视角基础模型对历史观测与目标作几何注册，恢复精确 6DoF pose；免任务微调、免预建 scene DB。Gibson SR 93.1%、HM3D 82.6%，位置/朝向误差分别 0.27m/3.41°、0.21m/1.23°。可移植“廉价相似度全程筛选 + 昂贵几何核验仅在近目标触发”；同视角网格不需要重建其 3D 子系统。证据 Tier-1P；2026-09-30；全文 §3–4、表2；作者页用于项目状态。

### G10 [Latent World Models with Monotone Planning Costs for Image-Goal Navigation](https://arxiv.org/html/2608.09073)
Amirhosein Chahe, Siwei Cai, Lifeng Zhou；2026-08 arXiv 预印本；官方代码未核实。GNM 60+小时、6 类机器人，1316 held-out trajectories；RGB/history+目标图，无地图/深度/GPS；连续 SE(2) 动作。冻结 DINOv2/v3 encoder，训练 action-conditioned latent world model，以自回归 rollout + monotone cost ranking 学规划可用的 latent 距离，再用 CEM MPC 搜动作序列。较 DINO-WM 朝向误差降低 2.7×，在若干路径/方向指标胜过模型与反应式基线。迁移：小格子可用 learned action-transition + beam search；但当前状态转移确定且仅 25 格，连续世界模型/大数据成本不划算。证据 Tier-1P（全文）；2026-09-30；§3–4、表1；未找到官方代码。

### G11 [What does really matter in image goal navigation?](https://arxiv.org/html/2507.01667)
Gianluca Monaci, Philippe Weinzaepfel, Christian Wolf；3DV 2026 oral（arXiv 2025）。官方作者页：[NAVER Labs Europe](https://europe.naverlabs.com/research/publications/what-does-really-matter-in-image-goal-navigation/)；代码状态未核实。系统对比端到端 RL 的视觉融合、通道堆叠、cross-attention、预训练与几何匹配；重点图像/位姿相对定位能力。发现某些 channel-stack RL 的成功严重依赖 Habitat 允许沿墙“滑动”的模拟器设置；更真实设置下预训练与早期跨注意力融合重要。对本项目意味着仿真无障碍 toy-grid 的成功不证明可迁移到真实 UAV；需做相似 patch、动作历史、转移和地图分布扰动。证据 Tier-1（3DV论文/作者页）；2026-09-30；全文摘要、§3–5、讨论；原始 arXiv 全文。

### G12 [LSVL: Large-scale season-invariant visual localization for UAVs](https://doi.org/10.1016/j.robot.2023.104497)
Jouko Kinnari, Riccardo Renzulli, Francesco Verdoja, Ville Kyrki；Robotics and Autonomous Systems 2023。任务是无初始位姿 UAV 全局定位：必须预备已地理配准卫星图/离线位置朝向描述子库；UAV 正射当前图像结合 VIO，point-mass filter belief 在地图候选上更新，不是目标搜索。100km² 初始不确定度下 23–44 次观测达到 12.6–18.7m 平移误差，季节变化仍工作。可借“地图可用时 belief 与多假设更新”，但地图假设违反本项目约束。此为 known satellite-map 方法的鲜明对照，不能把它的检索库当作 active unknown-grid baseline。证据 Tier-1（同行评审原文/DOI）；2026-09-30；全文摘要、方法图1/§4–6；出版社全文。

## 对项目的决策含义

1. G01–G03 是任务同构基线：完整 grid 在环境侧已知，policy 侧是部分可观测。报告任何“无地图”结论时要写清楚 policy 不看整图，但训练器/环境有目标坐标、格位、距离奖赏及合法动作边界信息。
2. G02 的静态 prediction-error curiosity 是可复用源；G03 的边际是按剩余目标距离门控 curiosity 与 PBRS 同时训练。现项目已有 curiosity/PBRS/合法 mask/BC（据本轮本地 README 只读核验），故不能把这些已实现件作为新建议；首先对照复现 G03 的动态权重公式和按 C 分层结果。其预印本 B=10 的 C=4 反而退化，避免只报平均值。
3. 当前任务同模态同场景：直接比较 target/current patch embedding、显式 belief/候选格评分，比引入 CLIP 全模态 alignment 更简洁。G06 的 VLM 高层子目标值得抽象为“输出目标格/宏动作+置信度”，低层交给已有确定性 governor；但需单独算训练/推理成本，不要把区域排序重命名当贡献。
4. 已知卫星地图定位（G12）前提不同；在线局部视觉建图/3D几何（G07/G09）则适合未来跨视角真实 UAV 图像，但不适合现有正射 5×5 网格的第一条主线。WORLD model + MPC（G10）应留作高成本路线，先测简单格位 belief / action-conditioned transition planner。

## 检索与证据记录

检索按 9 批关键词覆盖 exact-title 三篇、AiRLoc 前驱、UAV ImageNav、HRNav/IGL-Nav/AnyImageNav、离线 RL 与 world model、已知卫星图定位及开源状态。全文抓取 12 篇：arXiv HTML（G01–G03、G08–G11）、CVF/NeurIPS/ACL/SAIS 主会论文及其 PDF（G01/G02/G04/G05/G06/G07）、出版社页面/全文（G12）。论文与作者项目页只作为同一工作的“论文证据/代码状态”两类信息，不算两项独立研究。检索词重点包括 active geo-localization 2025/2026、image-goal unknown grid、UAV image-goal RL、curiosity/prediction-error、hierarchical VLM subgoal、offline goal-conditioned RL、incremental scene localization、known-map UAV localization。未找到官方代码或不确定接收状态均明确标为未核实，不以第三方博客/榜单补空缺。

