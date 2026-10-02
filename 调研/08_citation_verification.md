# 引文与关键事实核验

核验日期：2026-09-30。范围为调研稿 `01_direct_geolocation_sources.md`、`02_navigation_memory_hierarchy_sources.md`、`03_belief_search_worldmodel_sources.md` 中对本任务影响较大的十组事实。已对照论文原文、正式论文集页面或作者官方仓库；没有进行数值复现实验。

## 核验结果

| # | 状态 | 核验结论、原文位置与建议改写 |
|---|---|---|
| 1 | **SUPPORTED** | **GOMAA-Geo 的 zero-shot 不等于无训练。** 摘要/§1 的 zero-shot 指跨目标模态、数据域的泛化；§3 明确说训练时目标内容始终是已知位置的航拍图；§4 的 GASP 用随机轨迹监督“使目标距离变近的动作”，随后冻结 LLM，用 PPO 训练 actor-critic。§3 的任务定义每步只观测所在格的 aerial patch，全域不可见；因此可写“部分观测 + 当前图像/目标图/历史序列”，不能写成无训练或整图输入。证据：[arXiv 原文，Abstract、§3–4](https://arxiv.org/html/2406.01917)；同行评审论文入口：[NeurIPS 2024 PDF](https://papers.nips.cc/paper_files/paper/2024/file/bd8b52c2fefdb37e3b3953a37408e9dc-Paper-Conference.pdf)。 |
| 2 | **SUPPORTED** | **GeoExplorer 联合预测动作与动作条件的后续状态特征；PPO 时冻结序列模型。** §3.3 以当前序列和目标表示预测目标导向动作，并用前一动作条件化状态特征预测；§3.4 写明预训练 causal Transformer 冻结，PPO 更新 actor-critic；§3.5 描述推理。§3.1 说明输入是当前 patch，任务全区由非重叠网格构成，不是读取整图。证据：[arXiv 原文，§3.1–3.5](https://arxiv.org/html/2508.00152)；正式会议记录：[CVF ICCV 2025 论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Mi_GeoExplorer_Active_Geo-localization_with_Curiosity-Driven_Exploration_ICCV_2025_paper.html)。 |
| 3 | **SUPPORTED** | **DynCur-Geo 是 2026-08 的 arXiv v1 预印本；距离门控奖励只用于训练，测试时不读目标距离。** §3 的训练奖励把距离进展、门控内在奖励和 PBRS 组合；§4 Table 1 的 aerial 目标行，C=4 时 DynCur SR 0.2681，低于 GOMAA 0.3574 和 GeoExplorer 0.3489；C=5 也略低于 GeoExplorer（0.3702 vs 0.3745），C=6–8 才在该行领先。故不要概括为“各距离都提升”。Table 1 按距离分层，其他模态也呈现并非处处领先的格子。证据：[arXiv 原文，版本信息、§3、§4 Table 1](https://arxiv.org/html/2608.18673)。 |
| 4 | **SUPPORTED** | **HRNav 是训练过的高低层系统，不是 training-free。** ACL 官方记录将论文收录为 ACL 2026 Long Paper，作者元数据拼作 **Nanning Zheng**；若 PDF 文本抽取出现 “Zhen”，应以 ACL 书目元数据及 PDF 作者页拼写为准。论文 §3.3 用 VILA 做高层短期计划；Implementation Details 说明高层经 SFT 训练后冻结用于推理，低层用 DD-PPO 训练。建议写“高层 VLM 经 SFT，低层导航策略经 DD-PPO”，避免笼统称端到端 PPO 或零训练。证据：[ACL Anthology 元数据](https://aclanthology.org/2026.acl-long.1352/)；[正式 PDF，§3.3–3.4、Implementation Details](https://aclanthology.org/2026.acl-long.1352.pdf)。 |
| 5 | **SUPPORTED** | **TSGM 有图记忆，也需要学习训练。** 论文 §3.1–3.5 构造并更新 image/object 图及动作策略；§3.6 明确先 imitation learning 预训练，再用 PPO 微调。不能把它描述成纯训练免疫的图搜索器。最终任务表现见 CoRL 论文 Table 1–2。证据：[CoRL 2022 / PMLR 论文页](https://proceedings.mlr.press/v205/kim23a.html)；[论文全文，§3.5–3.6](https://arxiv.org/html/2209.08274)。 |
| 6 | **PARTIAL** | **SemNav 的同表数字正确，但“非确定性”细节没有被论文文本确认。** §5 Table 1 的列顺序是 SPL、SR：SemNav-Greedy 为 34.3/54.6，SemNav（加 LSP）为 35.9/54.9；这是同一 HM3D ObjectNav 评测中的小幅差异。§3.4 说明两者都用在 HM3D train split 上预训练的 local point-goal policy。它是学习得到的 PointNav 执行器，不是论文所提的确定性 BFS；但论文未说明推理时使用随机采样还是贪心动作，故“非确定性”作为运行属性应删去或注明实现配置未核实。证据：[arXiv 原文，§3.4、§5 Table 1](https://arxiv.org/html/2506.03516)；[OpenReview PDF](https://openreview.net/pdf?id=EZlJLnyI8L)。 |
| 7 | **SUPPORTED** | **VLFM 的高层语义探索可以称 zero-shot，但局部执行仍使用训练过的 PointNav。** §IV 说明 PointNav 以 HM3D train split 训练 2.5B steps；输入为 egocentric depth、目标点相对距离与朝向，不用 RGB。VLFM 自身是 frontier/value map 的语义选择层，整套导航并非完全没有任务相关学习组件。证据：[VLFM 原文，§IV PointNav、§V 实验设置](https://arxiv.org/html/2312.03275)；出版记录：[IEEE ICRA 2024](https://doi.org/10.1109/ICRA57147.2024.10610712)。 |
| 8 | **PARTIAL** | **SPF 的 Table 2 支持“结构化 waypoint 输出在其 UAV 导航任务中优于文本动作”，但需修正匹配 backbone 的数字与任务边界。** Table 2 / §4.2：Gemini 2.0 Flash 下 Plain VLM（text generation）=7%，PIVOT（visual prompting）=40%，SPF（2D waypoint labeling）=100%；87% 是 SPF 使用 Gemini 2.0 Flash-Lite 的结果，并非同 backbone 对照。§3 的任务输入是当前图像加自然语言指令，输出 3D motion/waypoint；不是“给定一张目标图，在未知 5×5 网格中定位对应图块”。不能将这些 SR 与本项目的定位指标直接比较。证据：[CoRL 2025 / PMLR 论文页](https://proceedings.mlr.press/v305/hu25e.html)；[原文，§3、§4.2 Table 2](https://arxiv.org/html/2509.22653)。 |
| 9 | **PARTIAL** | **NWM 会根据历史帧和动作预测未来视觉表示，并用想象轨迹做 goal-conditioned planning；“对任意未见网格格子的真实像素有保证”不在论文结论内。** §3.2 介绍动作条件的 Conditional Diffusion Transformer；§4.4 使用 CEM 比较 imagined final frame 与 goal image 的 LPIPS；§4.3/Table 4 评估未来预测，包括一个 unknown environment 设置。由此推断，生成未来画面可作策略模型的预测/候选评分，但不能等同于目标格真实观测或经验证的真值图像。这是对任务差异的边界推论，不应写成 NWM 作者自己的否定结论。证据：[arXiv 原文，§3.2、§4.3–4.4、Table 4](https://arxiv.org/html/2412.03572)。 |
| 10 | **SUPPORTED** | **三项代码状态区分准确。** GOMAA 官方 GitHub 含数据处理、预训练、训练及推理流程；README 的 Model Zoo 标注模型权重 “Coming Soon”，所以应写“代码公开，权重待发布”。IGL-Nav 官方 GitHub README 写明 “Code is coming soon”，可见仓库当前主要是 README/assets，故不称实现已开源。UniGoal 作者仓库在 2025-04-06 标记发布代码并支持 instance-image-goal 与 text-goal，README 也提供 `ins-image` 运行入口。项目/仓库只用于**可用性**，不能替代论文方法或数值证据。证据：[GOMAA 官方仓库](https://github.com/mvrl/GOMAA-Geo)；[IGL-Nav 官方仓库](https://github.com/GWxuan/IGL-Nav)；[UniGoal 官方仓库](https://github.com/bagh2178/UniGoal)。 |

## 证据等级修订

- **T1：同行评审的一手出版源。** 例如 NeurIPS、CVF、ACL Anthology、PMLR、IEEE/ICRA 记录及论文。若正文抽取用 arXiv HTML 辅助，venue 页面只确认书目信息时，应把两种用途分别标注。
- **T1P：arXiv 预印本原文。** 是一手研究材料，但未因此获得同行评审状态；DynCur-Geo、SemNav、NWM 在本次核验中按此等级。不要把“未检索到接收信息”写成永久未发表，只限定为本次检索日期和已查记录。
- **项目页/代码仓库：只证明发布或可用性。** 可核验“代码公开”“checkpoint 待发布”“README 声明 coming soon”等状态，不作科学主张或论文数字的独立佐证。代码状态会变化，须附核验日期。

## 对主报告的优先编辑

1. 把 GOMAA 和 VLFM 的 zero-shot 解释限定到跨域/跨模态或高层语义组件，显式写出其训练阶段与学习过的低层策略。
2. 把 DynCur-Geo 的收益写成按距离/目标模态分层的结果，并指出 C=4 的 aerial 行低于两个基线；推理时不使用训练奖励或目标距离。
3. 删除 SemNav “非确定性”作为已核实事实，改成“低层使用 HM3D 预训练 PointNav；论文未明确推理采样配置”。
4. SPF 的 matched-backbone Table 2 数字改为 7/40/100（Gemini 2.0 Flash）；另外注明 87% 属于 Flash-Lite，并说明任务为指令条件 UAV 导航，不是未知网格的图像目标定位。
5. 描述 NWM 时区分论文事实与报告推论：动作条件未来帧与图像目标规划是原文事实；“想象帧不等于任意隐藏格真值”是迁移到当前网格题后的证据边界。
