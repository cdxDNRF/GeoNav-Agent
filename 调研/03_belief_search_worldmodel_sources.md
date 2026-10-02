# 主动视觉地理定位与图像目标搜索：belief、规划与 world model 证据包

检索日期：2026-09-30（Asia/Shanghai）。范围：主动感知/图像目标 Bayesian search、图搜索与短视野规划、轻训练 scorer/IL/RL 的跨场景风险、world model 的可预测边界，以及未来多智能体扩展。优先一手论文全文、会议论文集、作者项目页和官方代码。以下“迁移判断”是针对本项目协议的推导，不是原论文实验证明。

## 本地协议下的结论

已核验的当前条件是：当前位置、5×5 网格几何与无障碍四邻接转移已知；隐藏的是目标位置 G 与各格视觉地图 M；每步抵达一格后才收到该格图像；系统自动到达成功，没有 stop 动作；未访问图像和 embedding 都不可访问。因而这是“已知有限图上的视觉目标搜索”，不是未知位姿 SLAM，也不是需要重建障碍图的 frontier mapping。Manhattan/BFS 只给行走成本，不提供目标方向。

当目标是在网格中唯一一张目标图片对应的格、而未访问格内容互相独立且没有可见的空间相关线索时，初始后验应是对候选格的先验分布。任何两张地图若在已观察历史上完全一样、但目标分别位于不同未访问方向，智能体输入不可区分，故不可能仅凭当前位置、目标图和历史产生总是正确的远程方向。这是可识别性推论。方向性策略必须依赖明确的相关结构/先验，或转成预算内覆盖搜索。

若先验均匀且每个新格的观察模型相同，探访任意一个未观察格的期望单步信息增益相同：看见匹配目标就成功；没命中就排除该格并重归一化其余候选。因而此时信息价值来自“新候选格”本身，不来自方向；路线规划只需在确定几何和预算下安排覆盖代价。若没有 p(o | G, M, history, action) 观测似然，就不能把 VLM 自报 confidence、CLIP cosine、softmax、frontier 数量或 ensemble disagreement 叫作 Bayesian posterior / 真正 entropy gain。公式 IG(a)=H(b_t)−E_o[H(b_{t+1})] 的期望需要可核验的观测模型。

本项目已有 visited set、防重复 coarse ranking、冻结 action dynamics curiosity、BC/PBRS 与有效的 edge 连续性匹配。它们应作为固定强基线或已知模块，不重新包装成新贡献。真正可新增的部分只能是：目标视觉证据让动作决策在同预算、同信息权限下变好，并在独立图片/区域上成立。Edge continuity 可作为观测线索或匹配假设，仍需与随机/错误边缘、打乱方向和纯覆盖对照。

**ImageNav 与 instance/category search 必须分协议。** 标准 ImageNav 目标是图像所拍摄的位置；InstanceImageNav 的图像突出特定对象实例；ObjectNav 通常给类别标签。三者的目标匹配似然、成功判据与视觉歧义完全不同。若用户的目标图是完整地物/位置图，别用“找到相似类别物体”指标冒充精确地点识别；建议显式标记 exact-image、same-place/different-view、instance、category 四种任务设定。

## 方法建议与边界

1. **Belief 必须由先验与似然组成。** 对已访问格 c，可用验证集校准过的 match likelihood 更新候选 G；对未访问格保留未观察质量，不能将“没有 embedding”误写为负样本，也不能只在 visited set 上 softmax 后把总概率归一成 1。CLIP/VLM 分数首先是排序分数。要转成概率，需在独立地理区域上配对真实匹配、相似但错地点、类别相同实例不同、无目标样本，做温度/Platt/isotonic 校准并报告 Brier/ECE 与阈值覆盖率。单次环境自动成功是特殊观测，可排除目标格；现实无人机仍需检测器与终止协议。

2. **IG 目前最多只能做代理量。** 可测试 2–4 个明确空间假设（例如某个连续边缘/地物边界朝向），计算候选观测点对这些假设的预测分歧；只有不同假设确实预测不同图像/匹配结果时，访问该点才可能分辨假设。若假设对任何未观察格都没有可检验预测，分歧没有根据，应退回覆盖/路线代价。MAX 提供 ensemble disagreement 的方法类比，但它学习 action-conditioned forward model，分歧是 model novelty，不等于目标位置后验。纯 VLM 输出“我很不确定”也不是校准的不确定性。

3. **短路径 beam 与 MCTS 的合理边界。** 5×5 确定网格上，短视野 beam 可排序由已知格分数组成的候选路线，优化已知访问收益、路径/预算、覆盖和去重，并且只执行第一步后用真观测重规划；不要模拟具体未见图像。POMCP/MCTS 需要一个可调用的状态转移/观测模拟器，或可靠的学到的观测模型；对“任意隐藏图像”做 rollout 会把生成先验误当事实。未知图像本身无结构时，MCTS也没有可利用的信息分支；纯 coverage path 或短 beam 更可解释。Yamauchi frontier 原法解决“地图边界的未知空间”，此处几何不未知，frontier 不能直接转成新增信息。

4. **小 scorer / IL/RL 的评估要隔离场景与信息权限。** 保持图像编码器、VLM调用数、训练样本量和动作预算固定，比较无学习、校准线性头/小MLP、BC、RL/探索奖励。按原始源图/地理区域/轨迹划分 train/val/test，禁止同一地点的裁剪图或重叠 patch 泄漏到测试。以新网格 seed、布局/目标位置/起点与拍摄时段为测试分布；另设跨区域测试。DAgger指出序贯策略的动作会改变未来状态分布，因此离线 expert-only BC 容易在自身访问的状态分布外累积误差；要跑闭环策略并统计状态偏移。Monaci et al. 对 ImageNav 的大规模消融提示，仿真设置可能让端到端 agent 走 shortcut；需改变外观、运动/滑动、起点和地图，而不是只随机 episode index。

5. **world model 能预测的是经验分布中的动作后果。** Navigation World Models 通过大量人类/机器人第一人称视频，学习“当前观察+动作序列→未来视图”的条件分布；BEINGS 在已重建的 3D Gaussian Splatting 场景上渲染未来视图并 MPC；NavWAM 在仿真预训练和机器人适配中联合未来观测、动作和目标进度。它们证明 world model 适用于有连贯视觉动力学或场景重建时的控制/规划，不证明模型可恢复当前项目任意独立未访问图块的真实像素。若网格图像是独立抽样、相邻格无视觉连续性，则不同的隐藏地图与相同历史都兼容，具体下一格图像不可识别；生成画面只能表示先验想象，必须标为 model_inferred，不能作为 observed。当前动作转移确定、G unknown、地图M隐式时，重建 SLAM 不必要。若将来建立有跨格真实地理/影像关系的世界先验，先做小型可识别性与校准实验，再谈 imagination planning。

6. **消融与关键负对照。** 同一 episode 配对相同起点/地图和 budget：真实目标图、难度匹配错误目标图、无目标图、打乱图像—位置关联、只用当前位置/已访问数、不用目标图。错误目标应包含同类别/相似地物但不同地点，避免 easy negative。至少报告 Success@budget、找到目标前步数、预算下覆盖率、每预算新位置数、错误命中/假阳性、路径长度，并给按目标类型/区域分层的 bootstrap 区间。报告目标模块机会链：被接受证据的精度、coverage/弃权率、动作改变率、最终 success uplift；只报高精度阈值很容易让几乎恒弃权的模块看起来优秀。

7. **多智能体是以后单独的问题。** Ghods et al. 的分散式多机 active search 研究稀疏连续信号与局部连续感知，目标是多机独立采样和协调；这与当前单机图像实例匹配不同。未来可把单智能体校准后的格级证据、并行分区、通信量、重复访问和冲突处理作为扩展，先固定单智能体目标相关收益，不将 Dec-POMDP/通信贡献混入本轮主张。

## 重要论文卡（8项，按可迁移部件组织）

### S09 — Image-goal 拓扑语义记忆：TSGM
- **Title / authors / venue:** *Topological Semantic Graph Memory for Image-Goal Navigation*, Nuri Kim, Obin Kwon, Hwiyeon Yoo, Yunho Choi, Jeongho Park, Songhwai Oh；CoRL 2022（PMLR 205, pp.393–402）。
- **论文/代码:** [PMLR论文页](https://proceedings.mlr.press/v205/kim23a.html)；[arXiv全文](https://arxiv.org/abs/2209.08274)；[作者项目页](https://rllab-snu.github.io/projects/TSGM/doc.html)；[官方GitHub](https://github.com/rllab-snu/TopologicalSemanticGraphMemory)。项目页标明 CoRL 2022 accepted，仓库公开代码/数据；这是实际发布，不只是计划。
- **数据/任务/观测/动作:** Habitat Gibson 的 ImageNav；给目标图，在连续室内场景到达目标图拍摄位置。论文实验用 panoramic RGB-D/检测到的对象语义；离散 move-forward、turn-left、turn-right、stop（0.25m、10°），与本地按格自动成功协议不同。
- **架构/训练/记忆/规划:** 按运行过程增量建立 image-node/object-node 拓扑图；graph builder、cross-graph mixer、memory decoder 选择动作；训练时用 Gibson 真值对象/检测对象，测试用预训练 detector；记忆只包含行走看到的节点，没有任意未观察节点的图像。不是全图 oracle，也不是 MCTS。
- **指标/结果:** Gibson overall Success/SPL 为 81.1/67.2，VGM 为 76.1/64.5；论文另报对若干竞争记忆基线 success +5–9 个百分点、SPL +7–23.5 个百分点。结果适用于其连续运动和图像距离成功准则。
- **可迁移模块:** 在访问后累积目标相关节点记忆、利用连续观测关系、同 observed-only graph 上做路线选择。
- **风险:** 相机、RGB-D/检测器与场景连续性强于本项目；语义物体图与精确航拍地点图不同。不能直接把其图神经网络增益解释成未知网格中的远程方位判断。

### S11 — 目标实例图像搜索与模块化证据链：Navigating to Objects Specified by Images
- **Title / authors / venue:** *Navigating to Objects Specified by Images*, Jacob Krantz, Théophile Gervet, Karmesh Yadav, Austin Wang, Chris Paxton, Roozbeh Mottaghi, Dhruv Batra, Jitendra Malik, Stefan Lee, Devendra Singh Chaplot；ICCV 2023, pp.10916–10925。
- **论文:** [ICCV Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Krantz_Navigating_to_Objects_Specified_by_Images_ICCV_2023_paper.html)。本轮未确认作者官方代码仓库，故代码标为**未核实**，不能把 paper/supplement 视为代码发布。
- **数据/任务:** HM3D InstanceImageNav 和真实住宅/办公室；目标图指向物体实例，非完整地物图像的 exact place localization。
- **观测/动作/架构/训练:** 机器人 egocentric 视觉、局部 map/导航传感；模块化串接探索、实例重识别、目标定位、局部导航。用特征匹配找回实例并投影到地图，各子模块主要用现成组件，无 task-specific fine-tuning。连续室内导航且有近距离物体定位，不是任意 tile 跳转。
- **记忆/探索/指标/结果:** 探索新环境并保留局部地图/实例 evidence；系统在 HM3D benchmark 的 success 是 end-to-end RL 的 7 倍、ImageNav 模型的 2.3 倍（56% vs 25%），论文也报家/办公室机器人合计 88% success。只比较该论文协议中的基线。
- **可迁移模块:** 把目标图识别、是否属于同实例、空间定位分成可诊断子任务；实地定位用实例特征而不是只用类别标签。
- **风险:** 目标物体、相机连续运动、地图/深度、可停止验证和导航传感均不同；不能把此 56% 与其他协议直接比较，也不能直接套到目标是整幅航拍场景的地图。

### S15 — InstanceImageNav 的探索—验证—利用：IEVE
- **Title / authors / venue:** *Instance-aware Exploration-Verification-Exploitation for Instance ImageGoal Navigation*, Xiaohan Lei, Min Wang, Wengang Zhou, Li Li, Houqiang Li；CVPR 2024, pp.16329–16339。
- **论文/代码:** [CVPR论文PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lei_Instance-aware_Exploration-Verification-Exploitation_for_Instance_ImageGoal_Navigation_CVPR_2024_paper.pdf)；[arXiv](https://arxiv.org/abs/2402.17587)；[官方项目代码](https://github.com/XiaohanLei/IEVE)。仓库明确标“CVPR 2024 implementation”，公开代码；依赖 Habitat、LightGlue、Detectron2 和外部预训练权重，不能等同于零依赖。
- **数据/任务/观测/动作:** HM3D-SEM，目标图含特定 object instance；RGB/检测/分割和局部运动信息；官方 repo 说明 action 与论文对齐为 velocity control。与本地离散格自动终止不同。
- **架构/训练/记忆/规划:** 目标匹配不只看当前图与目标图，还利用距离；把搜索决策分成 confirmed target、potential target、no-target/exploration，并主动切换 explore/verify/exploit。重点是“靠近一点确认”，缓解远处相似物误认；不是未知格未来图像预测器。
- **指标/结果:** HM3D-SEM 上 paper abstract 报经典 segmentation 下 SR 0.684 对 0.561；较稳健 segmentation 下 0.702 对 0.561。来自相同基准对比，受检测质量影响明显。
- **可迁移模块:** 用目标相似证据与检测可靠性分离“可能匹配”和“确认匹配”；做相似错误目标、无目标验证集。
- **风险:** object-centric、室内 RGB-D/语义检测和连续速度控制；本项目自动到达成功，verify/stop 模块不会直接贡献同一个终点。

### S13 — POMDP/MCTS 与 active search 的严格边界：GPOMCP
- **Title / authors / venue:** *POMDP Planning for Object Search in Partially Unknown Environment*, Yongbo Chen, Hanna Kurniawati；NeurIPS 2023 主会。
- **论文/代码:** [NeurIPS论文页与补充材料](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html)，[OpenReview PDF](https://openreview.net/pdf/d0714366b290d7120f7aa2f32c1cb1e6ba99ef8a.pdf)。本轮未核实公开代码，不把相关联论文算成代码发布。
- **任务/状态/观测/动作:** Fetch 在有家具的部分未知 3D 区域找目标物体；POMDP 状态空间随探索扩张，面对定位误差、有限视野、遮挡。论文包含已建 point-cloud 环境/家具空间信息；观测似然和物体语义是任务构造的一部分。不是“只有目标图片和当前任意 tile”的 ImageNav。
- **方法/训练/规划:** 感知模块 + Growing POMCP（在线 Monte Carlo tree search），belief-tree reuse 与新 UCB；另用 guessed target 和更新网格处理未发现物体的无信息/无奖励状态。
- **指标/结果:** Gazebo/Fetch 室内 4 个目标查找场景；相较 POMCP-based baselines 报更快、更高 success、相同计算需求，论文摘要未给可跨任务比较的单个数字。
- **可迁移模块:** 只有当能提供合法 state-transition/observation simulator 或已拟合的校准似然时，belief-tree/MCTS 才有扎实依据；可用于对照短 beam。
- **风险:** 已知3D点云/家具、目标物类别、遮挡与移动观测模型都不同。它证明“对象搜索可有POMDP解”，不证明它是本地网格同任务 evidence。

### S03 — Disagreement 可作为可检验 proxy，但不是 posterior：MAX
- **Title / authors / venue:** *Model-Based Active Exploration*, Pranav Shyam, Wojciech Jaśkowski, Faustino Gomez；ICML 2019, PMLR 97:5779–5788。
- **论文/代码:** [PMLR论文页/全文](https://proceedings.mlr.press/v97/shyam19a.html)，代码链接本轮未核实。
- **任务/观测/动作/训练:** 半随机离散环境与高维连续环境的 task-agnostic RL 探索；学习 ensemble forward models，针对动作未来预测的分歧规划探索路线。并非 goal image location posterior，也不依赖预先知道目标格。
- **记忆/规划/奖励:** 模型集合储存对转移的不同预测，基于 Bayesian exploration novelty 估计分歧并优化动作序列。
- **结果:** 半随机离散任务中比强探索基线至少高一个数量级效率；也展示模型可扩展至高维连续环境的 task-agnostic 表征。此数字只属于论文任务。
- **迁移/风险:** 可支持在多个独立 scorer/world hypotheses 上测试“观测哪个 waypoint 最能区分目标相关预测”。纯模型分歧倾向发现新奇动力学，并不保证找到目标；当前格图像未提供时，不能对实际未见图像计算视觉 embedding disagreement。

### S21 — 反证与公平性：What does really matter in image goal navigation?
- **Title / authors / venue:** *What does really matter in image goal navigation?*, Gianluca Monaci, Philippe Weinzaepfel, Christian Wolf；arXiv:2507.01667（2025，v3 2026-02-03）；作者出版目录将其列为 3DV 2026 oral，故把arXiv全文作为论据、venue以作者目录标注。
- **论文/项目:** [arXiv全文](https://arxiv.org/abs/2507.01667)；[作者发表目录](https://philippeweinzaepfel.github.io/)。未核实官方代码。
- **任务/方法:** ImageNav 中拆分核心导航表征与“由当前观察和目标图计算方向信息”；系统研究 end-to-end RL 能否从 navigation reward 学出相对位姿估计，比较 late fusion、channel stacking、space-to-depth、cross-attention等结构。
- **观测/训练/指标:** 仿真 ImageNav 和更现实的测试；论文摘要是定性结果，没有在此卡中转述未核实数字。
- **主要失败/反证:** success 对 simulator settings 敏感，存在 simulation shortcuts；相关视觉技能能迁移到更现实环境，但程度有限。作者目录进一步概括，在不启用 Habitat sliding 的 realistic setting，early/cross-attention fusion 与预训练很重要。
- **可迁移模块/风险:** 本项目需用无目标、错目标、改变grid采样、held-out地域和可能改变移动机理来排除“固定覆盖路线/环境标签”捷径；仿真中SR变高不等于掌握目标地理关系。

### S17 — 世界模型想象：Navigation World Models
- **Title / authors / venue:** *Navigation World Models*, Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun；arXiv:2412.03572（2024预印本，2026-03修订）。
- **论文/项目:** [arXiv全文](https://arxiv.org/abs/2412.03572)；[作者项目页](https://amirbar.net/nwm)。本轮未核实可复现官方代码发布。
- **数据/任务/观测/动作:** 用多样 human/robot egocentric video 学导航；给过去视觉帧和 navigation action，扩散式 Conditional Diffusion Transformer（约1B参数）产生未来视觉轨迹。熟悉环境模拟轨迹做约束规划；也展示从单帧在未见环境生成可能路径。
- **方法/训练/记忆/规划:** video generative world model 预测 action-conditioned future observations；MPC/trajectory ranking 用这些 imagined frames 衡量目标进展。
- **结果:** 论文报告熟悉环境下可从头规划或重排外部策略候选轨迹；在未见环境生成 plausible trajectory。此处“plausible”不表示生成了真实地图内特定隐藏格的正确像素。
- **迁移/风险:** 若未来无人机有已配准影像、连续飞行视频、影像纹理/道路连通先验，world model 可形成有条件预测；若当前隐藏网格各图块独立随机赋图，则具体真实未见图不可识别，生成画面只有 prior。严格拆分 observed 与 model_inferred，并验证预测准确度再让它影响动作。

### S22 + S23 — World model 依赖场景先验：BEINGS 与 NavWAM（并读）
- **BEINGS title/authors/venue:** *BEINGS: Bayesian Embodied Image-goal Navigation with Gaussian Splatting*, Wugang Meng, Tianfu Wu, Huan Yin, Fumin Zhang；ICRA 2025。 [作者项目页](https://www.mwg.ink/BEINGS-web/)；[代码仓库](https://github.com/guaMass/BEINGS)。代码已公开但 README 仍列有修 bug、集成 ROS、Docker、GPU acceleration 等 TODO，复现成熟度不等同于论文页面齐全。
- **BEINGS 数据/方法:** 用3D Gaussian Splatting 作为 environment prior，在模型预测控制/最优控制下渲染候选位置未来图像，再 Bayesian 更新与规划；论文报告 simulation 与 physical experiments。可迁移思想是 model-predictive visual decision；关键限制是需要场景3DGS地图及当前位姿/可渲染几何，属于当前未访问图像权限之外的额外资源。
- **NavWAM title/authors/status:** *NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation*, Daichi Azuma, Taiki Miyanishi, Koya Sakamoto, Shuhei Kurita, Yaonan Zhu, Petr Khrapchenkov, Motoaki Kawanabe, Yusuke Iwasawa, Yutaka Matsuo；arXiv:2606.13494（2026-06，预印本）。[论文](https://arxiv.org/abs/2606.13494)；[项目页](https://dachii-azm.github.io/navwam/)；截至检索日未核实代码。
- **NavWAM 数据/方法/结果:** 仿真预训练与真实机器人适配；diffusion-transformer 把未来观测、goal-progress value 与 action chunks 编成统一序列，直接产出闭环动作。摘要报告在离线基准和真实机器人部署上优于其规划式 world-model baselines，默认 policy mode 不用 CEM action search；摘要未给数字。
- **并读结论/风险:** BEINGS 的信息来自地图渲染；NavWAM 来自动作—观测训练经验。二者都不是不带图像/场景先验推断任意隐藏地图真值的方法。当前环境已知网格转移，world-model 的数据与算力投入要先证明能提升独立新地图/新区域的目标命中。

## 一手研究来源矩阵（同一论文的项目页/代码页不另计）

| ID | 论文与出处 | 对本任务可用的证据 / 必须保留的差异 |
|---|---|---|
| S01 | Brian Yamauchi, [A Frontier-Based Approach for Autonomous Exploration](https://www.robotfrontier.com/papers/cira97.pdf), CIRA 1997 | 经典真实机器人 frontier，访问已知/未知边界用于建图；本项目网格几何已知，frontier新地图论证不能直接迁移。 |
| S02 | Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS/PMLR 2011（arXiv 2010） | DAgger说明序贯预测动作改变未来观察分布；支持闭环检验BC，而不是只在expert轨迹上测分数。 |
| S03 | Pranav Shyam, Wojciech Jaśkowski, Faustino Gomez, [Model-Based Active Exploration](https://proceedings.mlr.press/v97/shyam19a.html), ICML 2019 | ensemble forward disagreement 是探索novelty的proxy；不是目标概率。详论文卡。 |
| S04 | Devendra Singh Chaplot et al., [Learning to Explore using Active Neural SLAM](https://arxiv.org/abs/2004.05155), ICLR 2020 | 在未知3D环境中学习建图、全局探索和局部规划；算法解决G/地图，不需要用于已知无障碍grid。 |
| S05 | Devendra Singh Chaplot, Dhiraj Gandhi, Abhinav Gupta, Ruslan Salakhutdinov, [Object Goal Navigation using Goal-Oriented Semantic Exploration](https://arxiv.org/abs/2007.00643), NeurIPS 2020 | SemExp用episodic semantic map和目标类别语义先验克服端到端探索/长计划弱点；目标是bed/chair一类，不是精确图像grounding。 |
| S06 | Devendra Singh Chaplot, Helen Jiang, Saurabh Gupta, Abhinav Gupta, [Semantic Curiosity for Active Visual Learning](https://arxiv.org/abs/2006.09367), ECCV 2020 | curiosity为训练检测器选择有用的标注视图；新场景中优于random、prediction-error curiosity、coverage。其奖励与对象识别质量对齐，不是任意目标图定位。 |
| S07 | Lina Mezghani et al., [Memory-Augmented Reinforcement Learning for Image-Goal Navigation](https://arxiv.org/abs/2101.05181), IROS 2022 | attention-based episodic memory，RGB-only，不要求pose/depth；作者称在Gibson达到当时SOTA。Habitat train/eval数据公开于[官方数据仓库](https://github.com/facebookresearch/image-goal-nav-dataset)，该库2024-05归档；本轮未核实官方模型代码。 |
| S08 | Santhosh Kumar Ramakrishnan et al., [PONI: Potential Functions for ObjectGoal Navigation with Interaction-free Learning](https://arxiv.org/abs/2201.10029), CVPR 2022 | 从被动top-down semantic maps训练“where to look”潜势，再与导航器组合；利用地图/类别共现先验。所需语义图训练资源对独立图块不一定存在。 |
| S09 | Nuri Kim et al., [Topological Semantic Graph Memory for Image-Goal Navigation](https://proceedings.mlr.press/v205/kim23a.html), CoRL 2022 | 连续观测的visited graph+object landmarks；详论文卡。 |
| S10 | Jacob Krantz et al., [Instance-Specific Image Goal Navigation: Training Embodied Agents to Find Object Instances](https://arxiv.org/abs/2211.15876), arXiv 2022 | 指出旧ImageNav目标图常随机拍摄而含墙面等歧义，提出HM3D InstanceImageNav：聚焦特定对象实例、goal camera与agent camera可不同。提供grounding instance vs place/category的任务划分。 |
| S11 | Jacob Krantz et al., [Navigating to Objects Specified by Images](https://openaccess.thecvf.com/content/ICCV2023/html/Krantz_Navigating_to_Objects_Specified_by_Images_ICCV_2023_paper.html), ICCV 2023 | 实例重识别、定位投影和局部导航用off-the-shelf模块，非端到端训练；详论文卡。 |
| S12 | Xinyu Sun et al., [FGPrompt: Fine-grained Goal Prompting for Image-goal Navigation](https://papers.nips.cc/paper_files/paper/2023/hash/27c4e15d9af120d7fef04432c7db577f-Abstract-Conference.html), NeurIPS 2023 | 目标图高分辨率细粒度特征作为prompt，引导observation encoder关注相关区域；Gibson/MP3D/HM3D改进。公开[官方代码](https://github.com/XinyuSun/FGPrompt)。适合作为固定matcher/goal encoder的消融先例，不是未来地图预测。 |
| S13 | Yongbo Chen, Hanna Kurniawati, [POMDP Planning for Object Search in Partially Unknown Environment](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html), NeurIPS 2023 | 目标物体+3D家具/点云环境，GPOMCP有 growing state、belief-tree reuse、感知/观测模型；详论文卡。 |
| S14 | Junting Chen et al., [How To Not Train Your Dragon: Training-free Embodied Object Goal Navigation with Semantic Frontiers](https://arxiv.org/abs/2305.16925), 2023 | 语义frontier结合视觉类别prior，评估训练free物体导航并检查segmentation噪声；类目标先验与整幅目标图不等价。 |
| S15 | Xiaohan Lei et al., [Instance-aware Exploration-Verification-Exploitation for Instance ImageGoal Navigation](https://openaccess.thecvf.com/content/CVPR2024/papers/Lei_Instance-aware_Exploration-Verification-Exploitation_for_Instance_ImageGoal_Navigation_CVPR_2024_paper.pdf), CVPR 2024 | 目标实例渐近确认、阶段切换；详论文卡。 |
| S16 | Hongxin Li et al., [MemoNav: Working Memory Model for Visual Navigation](https://arxiv.org/abs/2402.19161), arXiv 2024 | image-goal memory分短期节点、信息筛选/遗忘、长期场景聚合，再用图注意力形成目标相关working memory；可借“保留有用证据而非全部帧”，但记忆复杂度需匹配5×5规模。 |
| S17 | Amir Bar et al., [Navigation World Models](https://arxiv.org/abs/2412.03572), arXiv 2024（2026修订） | action-conditioned未来视频与trajectory imagination；详论文卡。 |
| S18 | Jianhao Jiao et al., [LiteVLoc: Map-Lite Visual Localization for Image Goal Navigation](https://arxiv.org/abs/2410.04419), ICRA 2025 | lightweight topo-metric map，coarse-to-fine camera localization；强调VPR/局部几何匹配需要已有map references。不是map-free预测任意未访问格。 |
| S19 | Zhicheng Feng et al., [Image-Goal Navigation Using Refined Feature Guidance and Scene Graph Enhancement](https://arxiv.org/abs/2503.10986), arXiv 2025 | goal/obs spatial-channel attention、自蒸馏与image/object scene graph；Gibson/HM3D cross-scene，作者报RTX3080最高53.5 FPS；[作者仓库](https://github.com/nubot-nudt/RFSG)公开实现/模型。室内cross-scene不证明航拍独立grid跨区域泛化。 |
| S20 | Jiansong Wan et al., [PIG-Nav: Key Insights for Pretrained Image Goal Navigation Models](https://arxiv.org/abs/2507.17220), arXiv 2025 | ViT goal/obs early fusion、导航辅助任务和大规模游戏视频预训练；摘要报相对视觉导航foundation model，两个仿真+一个真实环境中zero-shot平均+22.6%、fine-tune平均+37.5%，fine-tune数据更少；数据源仍是其pretrain mixture。应用应防止训练数据/地理位置泄漏。 |
| S21 | Gianluca Monaci et al., [What does really matter in image goal navigation?](https://arxiv.org/abs/2507.01667), arXiv 2025 / 3DV 2026 oral（作者目录） | ImageNav架构与RL shortcut负证据；详论文卡。 |
| S22 | Wugang Meng et al., [BEINGS: Bayesian Embodied Image-goal Navigation with Gaussian Splatting](https://www.mwg.ink/BEINGS-web/), ICRA 2025 | Bayesian MPC/视觉预测依赖可渲染3DGS地图与机器人位姿；不是当前未知M免费想象。 |
| S23 | Daichi Azuma et al., [NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation](https://arxiv.org/abs/2606.13494), arXiv 2026-06 | future observation + action + goal progress统一的world-action policy；详论文卡。 |
| S24 | Ramina Ghods, Arundhati Banerjee, Jeff Schneider, [Decentralized multi-agent active search for sparse signals](https://proceedings.mlr.press/v161/ghods21a.html), UAI/PMLR 2021 | 多空中机器人、稀疏连续信号、局部连续感知、分散式采样；提供未来通信/分区方向，不支持当前单机图像目标方法。 |

## 来源层级与证据检查

- **Tier 1 — 论文原文/会议论文集:** CVF Open Access、NeurIPS、PMLR、arXiv作者全文。主要任务定义、系统结构、结果只据论文摘要/方法/实验表，不用新闻稿替代。
- **Tier 2 — 作者项目/官方代码/数据:** TSGM、FGPrompt、IEVE、BEINGS、RFSG、ImageGoal数据集。只核实release形态、依赖和实现状态；不把项目页自称state-of-the-art写成跨论文总排名。
- **Tier 3 — 作者出版目录:** 仅用于核验Monaci et al. 的 3DV 2026 oral 状态及对 no-sliding setting 的简短描述；主要科学结论仍按arXiv论文摘要，明确此元数据来源。
- **未发现/未核实:** Memory-Augmented RL官方代码、GPOMCP代码、MAX代码、NWM代码、Monaci代码、NavWAM代码均未在本次检索中证实。预印本、项目页与数据仓库分开标注。

### 检索日志

日期 2026-09-30。共进行十余组公开检索并直接打开/核对超过八篇原论文/会议全文或摘要。主题查询包括：image-goal navigation / ImageNav benchmark；instance image goal HM3D；Bayesian visual search / POMDP object search；PONI；frontier exploration Yamauchi；active neural SLAM 与 SemExp；goal conditioned graph memory；MCTS/POMCP；ensemble disagreement / MAX；world model future observation prediction；3DGS Bayesian ImageNav；2025/2026 image-goal navigation；跨场景预训练及仿真shortcut；UAI 多机active search。检索优先会议官网、PMLR、arXiv与作者仓库；搜索结果里非一手博客、二手摘要和未来日期项目没有作为核心证据。原文核对位置包括摘要、任务定义、方法、实验表格（特别是TSGM §4/Table 1–2）和官方README。访问日为 2026-09-30；后续venue/repository状态如有更新须再验证。


### 多智能体补充一手证据

- **S25:** Haoyu Zhang, Sandor Veres, Andreas Kolling, *Simultaneous search and monitoring by multiple aerial robots*, Robotics and Autonomous Systems 170 (December 2023), Article 104544, [DOI / publisher page](https://doi.org/10.1016/j.robot.2023.104544)。论文把单机 simultaneous search-and-monitoring 写成 POMDP，多机版本建模为 semi-Dec-POMDP，以启发式 reactive planning 和 game-theoretic 方法分配任务；实验是 UAV 搜索未知地面目标并同时监控已知移动目标。它与 S24 的稀疏信号分散采样是两条独立研究线，但目标表示、协同通信与本地单机图像格搜索均不同，只用于后续扩展背景。
