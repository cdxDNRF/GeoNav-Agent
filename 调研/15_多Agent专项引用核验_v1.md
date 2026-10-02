# 多Agent专项引用核验（v1）

核验日期：2026-10-01（Asia/Shanghai）  
范围：仅依据任务书与 `11a`、`11b`、`11c` 中列出的论文/正式元数据来源，复核高影响引用；未运行模型 API、训练、推理评测或本地项目实验。正文提取到的内容有限，以下只报告能定位到的摘要、方法描述、表/协议注释及版本元数据；没有据此猜测未读到的章节、精确数字或正式状态。

状态含义：**SUPPORTED** 表示原文/正式记录直接支持所述核心；**PARTIAL** 表示核心大体成立但范围、预算定义、版本或正式状态须收窄；**UNSUPPORTED** 表示当前指定证据不支持核心说法或与之冲突。

| # | Claim–source pair | 判定 | 原文/正式记录核对与纠错 |
|---|---|---|---|
| 1 | **DiscussNav：ICRA 2024 vs. 2023 preprint；专家逐步咨询、n=5采样与分歧仲裁；缺少同成本reflection控制。** | **PARTIAL** | [arXiv 原文](https://arxiv.org/html/2309.11382)确认预印本首发于2023-09-20；论文描述每步咨询专家、采样五个thought–prediction pair，意见分歧时交decision-testing专家仲裁。所给原文页本身未提供ICRA会议元数据；本次没有取得可直接引用的IEEE/会议正式记录，因此“ICRA 2024, pp. 17380–17387”暂保留为待补正式书目证据，不能把2023与2024写成二选一：首发年和发表年是两个字段。论文报告与NavGPT比较及专家消融，没有检出同模型、同实际token/调用量的单Agent reflection配对控制；不可把性能差值归因于角色分工本身。 |
| 2 | **ReflectVLN：两套训练的3B模型；C偏航在线判据及GT监督；GitHub是否可复现。** | **PARTIAL** | [ReflectVLN 原文](https://arxiv.org/html/2607.12680)的实现细节明确为两个独立初始化的Qwen2.5-VL-3B模型，Table 2称约1.5M expert + 100k reflection samples。§3.5的闭环算法中，执行模型根据当前指令、子目标和图像历史输出状态token；C触发意图模型作纠正，推理期不需oracle trigger。**但训练信号确有特权/GT来源**：Figure 3文字写明reference/oracle轨迹提供ground-truth轨迹与waypoint监督，偏离由相对reference trajectory检测；严重偏离/失败时还调用可访问环境图的privileged expert生成恢复轨迹。因此应改为“在线C由执行模型预测；C/反思训练使用GT/reference与特权恢复数据”，不可笼统称GT使用与否未核实，也不可暗示在线读取GT。原文没有给可直接移植的、公开状态上的确定阈值判据。`11a`所记作者仓库仅README/TODO、代码未发布，是当日仓库快照；本轮未单独重审仓库历史，写作时应加检查日期。 |
| 3 | **VeGAS：CVPR 2026 Findings是否正式发表；verifier训练必要性；是否有同成本反思控制。** | **PARTIAL** | [作者项目页](https://nishadsinghi.github.io/vegas/)与[作者仓库](https://github.com/nishadsinghi/vegas)标注“CVPR 2026 Findings”，但本次未取得该论文在CVF/IEEE正式 proceedings 的独立条目；故可写“作者项目页标注为CVPR 2026 Findings”，暂不写成已由正式会刊页核实。论文[原文](https://arxiv.org/html/2605.12620)§6.2、Table 1–2支持：同一Qwen2.5-VL-3B-Instruct作zero-shot verifier没有收益或略降，fine-tuned verifier带来提升。需要纠正`11a`的过宽表述：VeGAS**确实**在候选数扩展消融中与Self-Consistency配平总LLM calls（§6.2 / Figure 6），但这不等于全篇存在与单Agent reflection的同token/同调用比较，也不等于匹配实耗token。主张“naive verifier无收益”应指zero-shot verifier相对CoT基线；不要概括为所有未训练verifier在所有基准上严格零收益。 |
| 4 | **Scaling Agent Systems：180配置/4基准与260配置/6基准；串行−39–70%；45% saturation；正式稿状态。** | **PARTIAL** | [arXiv版本页](https://arxiv.org/abs/2512.08296)记录v1于2025-12-09、v3于2026-04-08。v1摘要为4个benchmark、180 configurations；当前v3摘要已改为6个benchmark、260 configurations。这是**版本演进**，不是同一版内的来源互相矛盾；引用数字必须注明版本，当前综述应优先采用v3。v1报告的39–70%串行任务下降及约45%单体能力后的边际收益饱和，不得与旧版180/4拆开拼成“当前v3精确数字”，本轮未对v3正文结果表重核这两个数字。45%是其跨工具/规划benchmark的能力饱和趋势，**不是航拍导航SR的触发阈值**。arXiv当前记录只是预印本及版本日期，没有同行评审状态；`11b`转述作者页面称改题期刊稿处于revision，应标为“作者页面报告under revision”，不能写成已发表或录用。 |
| 5 | **MAST：NeurIPS 2025 D&B正式venue；1,642条trace、14类、150条专家样本与发布数据量。** | **PARTIAL** | [NeurIPS正式论文记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html)确认NeurIPS 2025 Datasets and Benchmarks Track，150条trace经专家人工分析、κ=.88、14种失效模式分3类，并公开MAST与MAST-Data。正式摘要称数据集为**1600+ annotated traces**；本次官方页没有核到精确“1,642”，故主报告暂写“1600+”；除非直接引用对应数据快照/论文正文并注明版本，不补精确数。150是taxonomy形成/验证的人工分析样本，不等于公开数据只含150条，也不能把它称作1,600+条均为专家手标。GitHub issue #13只是一条用户报告，本核验未将其当作事实或反证。 |
| 6 | **Reasoning in Token Economies（EMNLP 2024）与Should We Be Going MAD?（ICML 2024）：是否严格equal actual tokens。** | **PARTIAL** | [ACL正式记录与摘要](https://aclanthology.org/2024.emnlp-main.1112/)确认EMNLP 2024，并直接支持“按compute budget比较；可比compute资源下CoT self-consistency常胜过复杂策略；MAD/Reflexion可随compute增加变差”。[PMLR正式记录](https://proceedings.mlr.press/v235/smit24a.html)确认ICML 2024；摘要支持比较cost/time/accuracy，MAD并不稳定胜过self-consistency/ensemble，结果对agreement等超参数敏感。**两篇都不能据摘要概括为所有实验条件均严格匹配实际发生的总token数**；尤其PMLR摘要没有宣称exact-token matching。建议写成“预算感知/成本权衡证据”，把“same actual tokens”留给明确逐条件核对的论文或实验，不从调用数、配置预算推导实耗token相等。 |
| 7 | **Single-Agent LLMs Outperform Multi-Agent Systems…（arXiv:2604.02460）：严格同预算与结论边界。** | **PARTIAL** | [arXiv原文](https://arxiv.org/abs/2604.02460)§1定义controlled resource为中间reasoning/thinking tokens，**排除prompt及最终答案**；摘要与§5称在Qwen3、DeepSeek-R1-Distill-Llama、Gemini 2.5、multi-hop文本QA上按该定义配平后，单体匹配或超过多Agent。它是当前预印本研究，不是普遍“总系统成本严格相等”：正文明确报告Gemini API计量与可见thought tokens存在偏差。因此判定为对其指定benchmark/指定thinking-token口径有力支持，但不能把结论迁移成视觉导航SR规律，不能改写成所有输入、输出、调用、时延都严格相同。 |
| 8 | **MapGPT（ACL 2024）与Learning When to Plan（arXiv:2509.03581）：自适应触发能否training-free自学。** | **SUPPORTED（限缩表述）** | [MapGPT ACL正式记录](https://aclanthology.org/2024.acl-long.529/)确认ACL 2024，并支持其在线拓扑语言地图、基于提示的adaptive multi-step path planning及zero-shot R2R/REVERIE结果。它是单一navigation expert加地图/提示，不是多Agent协作，也不是经在线学习得到的trigger。[Learning When to Plan原文](https://arxiv.org/html/2509.03581)§4.3、§4.4、§5.1写明：零样本下复杂prompt无法可靠诱发dynamic planning；方法用SFT priming后接PPO学习规划时机。故可支持“该文的**learned dynamic-compute trigger**需要SFT+RL，prompt-only不稳定”；不能外推为所有简单规则触发都必须训练。项目若主张training-free自适应学习，则缺该来源支持。 |
| 9 | **Plan Verification（arXiv:2509.02761）：Judge是否只审文字计划，能否证明视觉在线失败恢复。** | **SUPPORTED（仅支持文字计划校验）** | [arXiv原文](https://arxiv.org/abs/2509.02761)摘要支持Planner生成动作序列、Judge迭代批评、Planner修订，并在TEACh手工标注动作/计划上评估。`11a`所核论文协议为Judge读取文本动作计划，无环境视觉输入；因此它只能作“文本计划清理/辅助训练数据质量”的先例，**不支持视觉证据核验、动作前在线门控或导航失败后的实际恢复效果**。venue按`11a`所核为NeurIPS 2025 Workshop，不应写成NeurIPS主会结果。 |
| 10 | **GoalSwarm（arXiv:2603.12908）：每UAV 500步是否等于公平的总预算比较；样本protocol。** | **SUPPORTED（预算限制明确）** | [原文§IV评测协议](https://arxiv.org/html/2603.12908)将每个object subtask设为500 steps；方法在多UAV间执行。原文结果注释又明确baseline为10 subtasks（2 episodes），协调方法为20 subtasks（4 episodes）。因此每个subtask/per-UAV局部上限不能表述成全系统相同总动作数；不同样本数也限制差值的因果解释。它是多个无人机共享地图的系统，不是同一执行体内的多个决策Agent。只能作为“按fleet总步数、任务数和并行/通信成本配平”的评测提醒。 |

## 建议直接替换的关键措辞

- DiscussNav写为“2023 arXiv首发、论文卡所列版本为ICRA 2024；讨论工作流每步运行、五个候选预测在分歧时仲裁；未提供与单体反思的实际token/调用配平证据”。正式会议记录补齐后再把venue升为已核。
- ReflectVLN写为“两套独立3B模型；C由执行模型在在线图像历史上预测，但训练偏航/恢复监督使用参考轨迹与可访问环境图的特权专家”。不要把在线无oracle trigger误写成无GT训练信息。
- VeGAS写为“作者项目页标注CVPR 2026 Findings；zero-shot verifier不增益、fine-tuned verifier有效；存在与Self-Consistency的同调用数消融，尚不能据此声称已与单Agent reflection全面等实耗成本”。
- Scaling写清v1/v3：v1=180/4，v3=260/6；如引用−39–70或45%，同时标版本和其benchmark范围。45%不得写成导航SR阈值。期刊revision注明是作者页面状态。
- MAST使用正式venue和“150人工分析样本 / 14种失效 / 1600+发布trace”三层分开的表述；精确1,642及GitHub issue待直接核数据版本后再写。
- 成本证据使用“budget-aware / matched thinking-token budget（明确口径）/ same-call-count ablation”这些原文各自实际支持的术语，不把它们混成“equal actual total tokens”。

## 主Agent交付前的补核（与上述独立抽查分开记录）

公开仓库raw README已补读：[IGL-Nav](https://raw.githubusercontent.com/GWxuan/IGL-Nav/main/README.md)当前声明代码coming soon；[AeroBelief](https://raw.githubusercontent.com/Shawnjx/AeroBelief/main/README.md)当前声明接收后发布源码/配置/评测。它们只核发布声明，不核复现能力或论文录用。DiscussNav由索引定位DOI `10.1109/ICRA57147.2024.10611565`，正式DOI页面工具访问失败，故第1项venue仍PARTIAL。最终报告对这三项采用限缩状态，没有将查不到写成已公开。

## 核验边界

本次以不超过10个论文/会刊页面的直接页面核读为限，并用少量定位检索补齐官方摘要与元数据。对arXiv HTML的提取不足或无法在正式会刊记录上闭环的地方已标PARTIAL；没有将论文主页、作者GitHub或用户issue当成独立论文证据。以上仅是引用与协议纠错，不构成本项目协作收益的实证结论。
