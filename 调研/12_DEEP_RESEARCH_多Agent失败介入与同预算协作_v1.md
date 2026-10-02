# 多Agent专项补充：失败介入、证据交接与同预算验证

研究截止：2026-10-01（Asia/Shanghai）。对象：**同一导航执行体内的多个决策角色**，给定目标航拍图片，在未知图像网格中主动找到对应位置。本文补充2026-09-30综述，保留旧报告；没有新增训练、项目模型/API推理、导航评测或复现实验。论文结果、本地既有实测与本次设计建议分开陈述。

## 1. 决策：只保留一个优先小试和一个条件候选

**值得继续验证的是“公开状态触发＋证据账本→预算探索规划”，不是扩大常驻Agent或辩论轮数。** 它可能补足有限预算下的覆盖记忆与路线组织。第二项“盲独立反证→同事件计划修正”仅在核验者具备可复核反证能力后保留。本轮没有证据支持扩大云端批次、多无人机或重复目标cue投票。

这一排序来自三层证据。第一，当前默认已经是训练过的Edge模块系统，S2/S3、10×10与SwissView阶段均通过，原综述中的“默认NoTarget、S2尚未通过”等是历史状态。第二，旧多Agent小试没有改变任何动作，触发只落在原成功题，交换消息没有被后续模型请求消费，尚未测试失败修复。第三，文献支持记忆、规划、反馈接口的价值，也反复显示额外编排成本、相关错误以及单体结构化推理的竞争力；缺少与本项目信息权限接近、同时做同记忆/同成本/消息消融的直接证据。[本地勘误与证据](D:/桌面/XDUCS-courses/大四/大数据工程/调研/16_原综述状态勘误与本地证据_v1.md)；[R01 DiscussNav](https://arxiv.org/abs/2309.11382)、[R06 Scaling](https://arxiv.org/abs/2512.08296)、[R10 Token Economies](https://aclanthology.org/2024.emnlp-main.1112/)、[R23 rover架构比较](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1877762/full)。

**保留标准是相对强单Agent的可归因增量。** 若完整已访问账本＋确定性规划，或同模型、同账本、同两次调用的单Agent追平，就采用单Agent实现，把贡献定位于状态或规划。若合法触发仍覆盖不到可执行的失败前状态、输出全unknown、消息不影响计划，收束协作支线；不靠改阈值、加轮数或换更强模型补过门槛。这是本次工程建议，不是论文已经证明的航拍导航结论。

可先读[一页摘要](D:/桌面/XDUCS-courses/大四/大数据工程/调研/13_多Agent专项一页决策摘要_v1.md)，执行前规格见[冻结实验设计](D:/桌面/XDUCS-courses/大四/大数据工程/调研/14_多Agent候选冻结实验设计_v1.md)。以下说明为什么选它、怎样证伪，以及当前不能声称什么。

## 2. 当前证据支持什么问题

| 既有阶段 | 已报告SR / SG | 能支持的范围 |
|---|---|---|
| S2 | 89.33% / 0.320 | 4张已见验证源图、val100×3 |
| S3 | 86.80% / 0.300 | 10张训练/开发未用源文件、test250×3；文件隔离成立，地理不重叠未证明 |
| S4 | 77.13% / 1.201 | 10×10/B20，500题×3，20张已知源图；同区域切图密度与紧预算扩展 |
| SwissView | 86.80% / 0.381 | 20张新源文件、500题×3、5×5/B10；同模态连续目标，不等于异时/任意视角泛化 |

这些成绩来自对应正式验收，不跨协议给论文排名；三份已训练权重与API重复也不是同一种重复。默认冻结Sat2Cap、Small256 GRU探索器与训练目标邻接头，接受阈值0.50。课程允许小模型训练；无训练协作层不能把整个系统改称training-free。[逐项本地出处](D:/桌面/XDUCS-courses/大四/大数据工程/调研/16_原综述状态勘误与本地证据_v1.md)。

旧协作20题/20已见源图固定checkpoint0，三轮API重复。原Edge与三种云端方案均SR95%、SG0.150；45次导航请求全部unknown，动作变化0。只有3道原成功题触发，唯一失败题未触发；双角色60条episode中51条零触发、9条仅一轮，没有下一请求读取交换消息。因此它提供了“本协议没有有效介入”的负结果，不能证明一般多Agent无效，也不能把继承原策略的95%称为协作成绩。6次像素身份正确只证明该诊断任务，不证明目标方向或计划核验能力。[协作报告](D:/桌面/XDUCS-courses/大四/大数据工程/DATA/processed_data/Masa/评测结果/多Agent同预算小验证_v1/多Agent小验证报告.md)、[触发覆盖诊断](D:/桌面/XDUCS-courses/大四/大数据工程/DATA/processed_data/Masa/评测结果/多Agent同预算小验证_v1/协作触发覆盖诊断.json)。

静态读取已有失败诊断后，S4跨权重343条失败只有11条记录到行动前真实目标邻接机会；SwissView198条失败只有1条。这使**探索覆盖/预算组织**比“再核验一次已有局部目标cue”更值得先检查。但这些是评测后真值标签，既不是在线可检测率，也不是可修复数上界；较早改变动作可能改变后续邻接机会。[诊断汇总与原JSON链接](D:/桌面/XDUCS-courses/大四/大数据工程/调研/16_原综述状态勘误与本地证据_v1.md)。

S4还有具体的状态缺口：原探索器25格visited输入以2×2汇总适配100格，丢失粗槽内部访问细节；真实100格visited仍用于线索筛选。这是把“完整已访问账本”作为首项的工程理由，尚不能证明它导致了失败。全账本须同样给确定性规划和单Agent臂，否则记忆增量会被误算成协作增量。[S4报告：推理适配与距离分层](D:/桌面/XDUCS-courses/大四/大数据工程/DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1/十乘十扩展报告.md)。

## 3. 本轮“多Agent”的操作定义

| 类型 | 可核查定义 | 归因边界 |
|---|---|---|
| 单Agent工具/模块系统 | 一个决策流程调用编码器、cue、记忆、BFS或两阶段模型提示 | 模块多不等于决策Agent多；当前Edge属于模块系统 |
| 多建议/采样集成 | 多份独立输出，由程序投票或一致性过滤；无人读取其他意见再修正 | 可测采样多样性，不声称反馈协作 |
| 同执行体多角色决策 | 分离角色上下文；消息进入接收者后续请求；唯一物理执行体按仲裁行动 | 至少审计状态、消息、最终提案和实际动作；同事件交接可以避免下次触发缺失 |
| 多物理机器人/UAV | 各自移动、获得不同局部观测、共享地图或任务 | 新增传感与并行覆盖，要按全队总动作/通信另评 |

角色名不是能力或独立证据来源。相同模型、相同合法观测的两个角色不会自动增加环境信息；同一调用流水线可被单Agent编排器精确模拟。若两个实现发送完全相同的API载荷，Agent标签不存在可识别处理差异；若改了提示、上下文裁剪或私有状态，就把那些差异登记为工作流因素。最终贡献更可能是**证据约束和预算规划**，而非“Agent个数”。这一判断是实验可识别性分析；通用协作失效分类也支持检查忽略同伴输入、规格失配与验证不足。[R07 MAST](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html)。

## 4. 导航/具身原始研究：可借接口，不能搬运SR

下表综合原论文、正式会刊与作者项目页。来源卡保存更细输入/训练/消息记录；“有消息”不等于已证明消息因果贡献。[导航来源卡](D:/桌面/XDUCS-courses/大四/大数据工程/调研/11a_多Agent导航与角色来源.md)、[独立引用核验](D:/桌面/XDUCS-courses/大四/大数据工程/调研/15_多Agent专项引用核验_v1.md)。

| 研究 | 本轮核实的事实与支持位置 | 可迁移部分 / 主要限制 |
|---|---|---|
| [R01 DiscussNav](https://arxiv.org/abs/2309.11382) | 方法段：感知、指令/进度、预测与讨论专家；多预测分歧时仲裁。R2R语言指令和全景候选视野；每步多次推理 | 结构化职责与仲裁可借；没有与同反思/同实际tokens单体配平。作者列ICRA2024，正式IEEE页本轮未成功读取 |
| [R02 ReflectVLN](https://arxiv.org/abs/2607.12680) | 方法/反思数据构造：意图与执行两套独立3B模型；执行模型在线预测continue/complete/off-track，反馈更新后续子目标。训练偏航/恢复监督用参考轨迹及特权专家 | 真实被消费的反馈接口有价值；在线无oracle不等于训练无特权。需要专门训练，且跨基准并非一致领先；当前代码未发布 |
| [R03 Plan Verification](https://arxiv.org/abs/2509.02761) | Judge读取文字动作计划、Planner修订；TEACh计划人工标注评价，来源卡所列NeurIPS2025 Workshop | 支持计划清理接口；没有视觉Judge或视觉在线失败恢复证据，不能写成主会导航结果 |
| [R04 VeGAS](https://arxiv.org/abs/2605.12620) | §6.2/Fig.6有相同LLM调用数的Self-Consistency消融；该实验零样本verifier无改善/略差，专门训练verifier有效 | 候选级核验可借；不等于与单Agent反思全面等总成本。作者标CVPR2026 Findings，官方会刊未闭环 |
| [R05 CA-VLN](https://www.mdpi.com/1424-8220/26/4/1254) | 方法/训练段：MLLM知识/历史角色、LoRA与融合模块 | 角色状态接口可借；是训练模块，不证明复制通用提示角色即可增益 |
| [R18 MapGPT](https://aclanthology.org/2024.acl-long.529/) | 方法：在线拓扑语言地图＋单navigation expert，自适应多步规划 | 提供强单Agent记忆/规划对照；观测图、语言路线与当前航拍实例目标不同 |
| [R21 SpaceVLN](https://arxiv.org/abs/2606.08992) | 方法：在线空间认知记忆、阶段规划与执行，RGB-D/pose等信息 | 支持阶段检查点/持久状态；不可给本网格额外深度、全景、pose或未访问图像，不能由模块数推断协作收益 |
| [R24 Planner Matters!](https://arxiv.org/abs/2605.02168) | 长程GUI角色交接与外部记忆；附录消融显示额外verifier可能降低成功率。补充表来自作者UCSD论文稿，非额外独立研究 | 同一执行体角色接口的正例及反例；单体未获同记忆/同成本，部分配置另有训练。预印本与学位论文都不能充当正式录用证明 |

这些研究使“明确职责、来源约束、同事件反馈”成为合理设计方向，但没有消除能力门槛。当前45次unknown可能来自没有方向信息、通用VLM不具备细粒度关系判别，或旧协议将探索动作也绑定于目标支持。不能由像素身份成功推出核验可用；也不能把所有unknown都理解为坏提示。P1先分开`exploration_plan`和`target_claim`，P2先诊断反证能力，是由这些限制推导的方案。

## 5. 成本与失败反证：强单Agent必须进入主比较

| 证据 | 原论文实际支持的结论 | 不支持的外推 |
|---|---|---|
| [R06 Scaling Agent Systems](https://arxiv.org/abs/2512.08296) | 协作表现取决于任务结构、基线与协调成本；v1为180配置/4基准，v3为260配置/6基准 | v1顺序任务退化范围或约45%基线饱和值不能作本项目SR门槛；作者期刊稿under revision不是已发表 |
| [R07 MAST](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html) | 正式NeurIPS D&B：150条专家分析用于分类、14种模式分3类；公开扩展1600+trace | 不是本项目失败率或协作正效应估计；150专家样本不能写成全部1600+专家手标 |
| [R10 Token Economies](https://aclanthology.org/2024.emnlp-main.1112/)、[R11 Should we be going MAD?](https://proceedings.mlr.press/v235/smit24a.html) | 正式EMNLP/ICML成本感知比较；复杂debate/反思未稳定超过CoT self-consistency，协议/超参会改变结果 | 不能统称所有条件严格匹配实际总tokens；更不能推出所有协作无效 |
| [R13 等thinking-token比较](https://arxiv.org/abs/2604.02460) | 多跳文本QA中按中间thinking tokens配平后，单体匹配或超过多体 | 排除了prompt/最终答案，provider计量有偏差；不是总成本配平或导航结论 |
| [R23 rover架构比较](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1877762/full) | §3–5：100个合成结构化情境，各5重复；完整输入单体token/时延较低。准确率点估计高，校正后差异不显著 | 不是真实rover闭环导航，不是等推理成本。支持完整结构化输入强基线，不能说统计证实单体准确率胜出 |

负证据不能单独决定停止。基础debate研究和后续Divergent Thinking在算术、事实/语言任务报告协作收益，说明不同意见与真实修订可以有用；但它们没有证明在本网格、同账本、同总成本下有效。[R15 Divergent Thinking](https://aclanthology.org/2024.emnlp-main.992/)、[R16 Multiagent Debate](https://arxiv.org/abs/2305.14325)。更大范围的MAD比较显示收益不稳定；逻辑推理控制实验提醒相关错误、群体压力与合意倾向，需要保留独立初始判断和可验证证据。[R12 If MAD is the Answer](https://arxiv.org/abs/2502.08788)、[R14 Can LLM Agents Really Debate?](https://arxiv.org/abs/2511.07784)。

因此本轮不以“大模型越多越好”或“单体必胜”为前提，而冻结三种口径：**同环境动作预算、同事件调用安排/上限、逐题实耗成本**。相同调用上限不等于相同实际tokens；padding烧tokens、事后挑成本相近子集都不能修复不公平。[成本与失败来源卡](D:/桌面/XDUCS-courses/大四/大数据工程/调研/11b_协作失败与同成本证据.md)。

## 6. 何时介入：合法机会先于模型调用

额外计算应带来超过其成本的决策价值，经典有限资源推理提供规范原则，但不会自动给出航拍失败预测器。[R17 Reflection and Action Under Scarce Resources](https://www.erichorvitz.com/reflect.htm)。现代test-time compute研究说明采样、修订和汇总本身改变资源；Learning When to Plan用SFT＋PPO学习何时规划，prompt-only动态规划不稳定，不能据此宣称通用提示会自然学会失败触发。[R19 Learning When to Plan](https://arxiv.org/abs/2509.03581)、[R20 Scaling Test-time Compute](https://arxiv.org/abs/2506.12928)。

本次建议先做**阶段T：既有轨迹公开前缀的触发机会检查**，只看历史覆盖与预算，不改变动作、不生成新SR。本轮尚未执行这一检查。它应回答：新规则是否在不同源图的失败前有可执行机会、是否只覆盖原成功题、剩余动作是否足够，以及强cue保护会挡住多少状态。

建议v1规则：`active AND no_accepted_cue AND unused_intervention AND (stagnation OR checkpoint)`。停滞为最后两步新增格≤1且存在重复状态；检查点为步数≥ceil(B/3)且从开始没有接受cue。B10为第4步，B20为第7步，每题最多一次；它是公开预算检查，不叫校准失败预测。停滞会漏掉“一直访问新格但没覆盖目标”的失败，检查点对此提供独立机会。

在线只能使用目标图、当前/实际历史图、位置、visited、剩余预算、合法动作、原策略提议和原始cue分数。不可读真目标格/距离、未来成败、事后失败标签、未访问图像/特征、全源图或源ID答案。评测端可用真值做事后分层，不能据最大失败覆盖率挑触发阈值。已接受的0.50 cue保护原动作，不由VLM unknown自动否决；该保护也使首项不覆盖“错误强cue”的另一类失败，结果中须明说。[拟冻结规则和字段](D:/桌面/XDUCS-courses/大四/大数据工程/调研/14_多Agent候选冻结实验设计_v1.md)。

## 7. 五个候选的取舍

以下改善预期均是待检验假设，成本指实现工作和新增推理。论文训练状态与本项目迁移层是否新训练分开。

| 候选 / 排序 | 证据与改善环节 | 权限、训练、成本 | 反证与淘汰条件 |
|---|---|---|---|
| **P1 首选：证据账本→预算探索规划** | MapGPT/SpaceVLN支持持久状态；ReflectVLN支持被消费的子目标反馈。拟补覆盖记忆与预算路线 | 仅已访问证据/网格几何；迁移层可不新训练；2调用/事件，账本、候选与option接口中等工程量 | 单体同账本或确定性规划追平；消息不影响计划；公开T无可执行失败机会，则收束 |
| **P2 条件：盲独立反证→修正** | Debate真实修订、VeGAS候选核验接口；拟减少错误假设坚持 | 同合法图片和候选；首轮通用核验器不训练但能力未证；3调用，独立盲化与来源审计 | critic不能指出可核实错误；反证损伤成功；同3调用自审/采样同样有效，不加辩论轮数 |
| P3 不优先：同质多候选投票 | DiscussNav/debate提供先例，可能采样更多方向 | 同输入、≥2调用；接口便宜但运行成本上升 | 无新增证据、错误相关、只是更多采样；旧小试unknown不能通过改多数规则解决 |
| P4 后续能力路线：专门训练计划/视觉verifier | VeGAS、ReflectVLN、CA-VLN提示训练能力可重要 | 要错误/恢复标注、训练与独立数据；成本高，超出本轮 | 若缺真实图像关系标签或只会文本合法性，不训练；当前Edge已学局部关系，不能重复包装 |
| P5 主线对照：单Agent完整账本＋几何option | MapGPT、结构化rover单体与本地S4访问压缩限制 | 相同已访问100格；先用确定性覆盖/BFS，无新训练、0额外调用 | 若有效，保留为主方法并停止角色扩张；若无效，再查覆盖/表征，而非假设多个角色可补信息缺失 |

P5既是替代方向，也是P1不可删的强对照。P1/P2不同时叠入新策略。多UAV不放进这一排名，因为它增加物理观测并改变问题定义。

## 8. 首项的冻结规格：让一次交接真正发生

P1事件中：E读取公共输入并输出有来源的`observed_facts/counter_evidence/unknowns/evidence_ids`；P读取相同公共输入及**刚产生的E消息**，提出最多2步option。程序检查路径、合法动作和预算，逐格执行、每步获取真实新观测并按实际动作更新原GRU；出现原强cue、到达、预算不足或反证就中断。waypoint不能跳格算一步。

输出区分三类：`exploration_plan`允许目标方向unknown但覆盖路线有几何依据；`target_claim`必须有合法图像关系支持；`abstain`保留原Edge。不要求探索动作先证明朝向隐藏目标，也不把新增覆盖说成目标识别。

| 主臂 | 同一事件调用 | 公共账本/候选/动作预算 | 主要用途 |
|---|---:|---|---|
| M0 原Edge | 0 | 原S4状态与冻结适配 | 已验收参考，资源最低 |
| M1 确定性覆盖规划 | 0 | 同完整visited账本、同T、同2步候选 | 排除几何工具/访问记忆增益 |
| M2 强单Agent结构化工作流 | 2 | 同记忆、图片、E式整理→option规划 | **主要协作对照**；允许同样工具，不故意削弱 |
| M3 单Agent反思 | 2 | 同账本与候选、草案→自审 | 反思协议对照 |
| M4 E→P分离角色 | 2 | 同公共输入；分离角色上下文并传递当前消息 | 检验上下文/职责分工是否增益 |

所有云端臂同模型版本、同最多4图片（目标/当前/最多2实际历史帧）、解码、token上限、触发、调用时序、option与仲裁。M4对M2为主比较，M4只赢M0或M3不构成相对强单体协作收益。一次事件不检验长期私有角色记忆；不能用该成绩宣称多轮持续协作。

模型输入hash和`consumed_message_ids`只能证明消息到达。D同状态诊断加入消息隐藏/打乱；若合理的计划变化消失，才更接近功能使用证据。角色合并若精确复现两次请求，只证明编排等价。新方法的目标遮蔽/错目标可诊断目标机制，但须事前占用冻结额度，不能看到G成绩后补调用。[完整schema、对照与仲裁建议](D:/桌面/XDUCS-courses/大四/大数据工程/调研/14_多Agent候选冻结实验设计_v1.md)。

## 9. 同预算、数据与指标

**建议首项主队列G固定10×10/B20，20张既有S4源文件，每图一题，C12…16各4题，checkpoint0。** 按源文件/任务hash先分层抽取，不读逐题结果选失败；距离只在评测端。API三重复仍只有20源图，不是60独立地图或三训练seed。这是开发机制小试，旧S3/已用SwissView不能再包装成新方法独立确认，最终确认需新源文件及明确地理隔离计划。

诊断D刻意涵盖失败前/成功保护/无信息等合法前缀，报告选择规则和分母，不混入总体SR。动作变化后各臂后继图像自然不同；公平是同传感权限、任务与首次分歧前历史，不是强迫所有未来看同地点。D共享状态评分只评局部决策，不能冒充闭环失败恢复。

| 首轮预算用途 | 建议上限 |
|---|---:|
| D：4状态×M2/M4/预指定消息消融×2调用×2 API重复；隐藏/打乱各2状态 | 48请求 |
| G：20源图×3 API重复×M2/M3/M4×至多1事件×2调用 | 360请求 |
| 工程消息消费/接口检查 | 12请求 |
| **总建议硬上限；本轮未执行** | **420请求** |

每请求max_tokens建议512、至多4图；在provider输出计数约定下，可见输出上限215040 tokens，图片输入至多1680。输入/隐藏思考token、图像收费与单价须另核，不能据此编造人民币/美元总额。固定调用时点敏感性首轮占D的2个第2步状态；它不是固定时钟闭环SR。后者如需测试须事前从G重新分配；没有预算外补跑。D两重复的消息影响仍只是初筛，不能把一次随机动作差异当成因果证明。[详细预算与局限](D:/桌面/XDUCS-courses/大四/大数据工程/调研/14_多Agent候选冻结实验设计_v1.md)。

报告三层结果：

1. 机会：总体触发率、原失败/成功事后触发比例、首次步数、剩余预算、强cue保护与未触发原因。
2. 机制：真实请求、合法非弃权、消息到达/功能使用、改变动作、可核实反证、无根据目标声称、option完成/中断。
3. 效果/资源：恢复失败、损伤成功、SR_gate/SG_gate/Q，以及请求、input/output/thinking tokens、图片、动作、P50/P95时延与接口失败。

沿用项目Q=正常终态/计划题；预算耗尽也正常，但不成功。SR_gate=成功/计划，SG_gate未完成以该网格最大距离保守补入（10×10为18），接口解析率另列。完整配对、Q一致时，`ΔSR=(恢复原失败−损伤原成功)/计划题数`。按源图聚合API重复后计算配对区间；20个已知簇和有限恢复事件无法支持宽泛结论。[已有验收口径](D:/桌面/XDUCS-courses/大四/大数据工程/选题报告相关/分阶段验收标准_v1.md)。

原“SR+2个百分点、至少2/3 API重复正收益、SG不变差”可作为候选小试门槛，**不是文献通用及格线或显著性检验**。20题每轮一题就是5个百分点。必须同时报M4对M2净差、恢复/损伤数与区间；零差异区间或同轨迹不能写成统计等价。阶段T至少需要不同源图的可执行失败前机会与非失败保护状态；D至少要见可追溯消息影响合理计划；达不到则停止，不扩大G。

## 10. 第二项的边界：三调用的盲独立反证

P2调用1由Planner独立起草；调用2的Critic看到相同合法观测和候选集，**不看本轮草案**，输出候选的来源约束/冲突/未知；调用3的Planner读两份意见并修正。这样同事件完成反馈，不依赖下一次触发。明显越界/不可达先由程序处理，不能把廉价规则检查包装成VLM能力。

先做反证能力诊断，包含可核实错误和无信息状态。Critic应允许unknown；多数一致不是证据。主比较是同3调用单Agent草案→自审→修正、单体编排相同盲工具流水线，以及消息隐藏/打乱；还需区分独立采样多样性与反证交流。P2另登记3调用预算，不占P1预算增加第三次调用，不与P1仅按SR排序。VeGAS的零样本失败/训练有效及Planner Matters的verifier退化都使该能力门槛有依据，不能据“critic”名字跳过。[R04](https://arxiv.org/abs/2605.12620)、[R24](https://arxiv.org/abs/2605.02168)。

没有可复核反证、消息不改变建议、损伤≥恢复或同三调用单体追平，停止P2。其价值是检验独立审查，不能预先声称将解决当前视觉方向能力缺口。

## 11. 主线扩展与多无人机的优先级

相较角色扩张，**真实搜索覆盖扩大、真实异时/视角目标、跨步目标关系与完整访问状态**仍更能回答项目泛化问题。10×10未扩大地理面积；SwissView是同模态、同源连续目标。先检查账本/option是否能解释现有覆盖问题，再决定做小模型表征或真区域扩展；已做过旋转、目标扰动、可靠性校准和硬负样本，不能再列为全新首轮。单模型也可通过训练内化推理减少显式调用，FantasyVLN提供另一条研究路线，但其训练资源和VLN权限与当前项目不同，本轮不启动该路线。[本地勘误](D:/桌面/XDUCS-courses/大四/大数据工程/调研/16_原综述状态勘误与本地证据_v1.md)、[R08 FantasyVLN](https://openaccess.thecvf.com/content/CVPR2026/papers/Zuo_FantasyVLN_Unified_Multimodal_Chain-of-Thought_Reasoning_for_Vision-and-Language_Navigation_CVPR_2026_paper.pdf)。

多机器人的新增局部观察是真实信息来源，不能把它和同执行体多提示混在一起。MCoCoNav是RGB-D多机器人语义地图协作；GoalSwarm是多UAV语义任务，其500步限制按object subtask，基线/协作任务数也不同，不构成本网格全队同预算优势证据。[R09 MCoCoNav](https://ojs.aaai.org/index.php/AAAI/article/view/33607)、[R22 GoalSwarm](https://arxiv.org/abs/2603.12908)。

若以后研究多UAV，建议同全队`Σ动作≤B_total`、同任务数，比较单UAV、固定分区、独立探索、地图协作；另报并行完成时间、通信字节/延迟、重复覆盖及模型成本。若每台各获单机B，SR变化首先混入总动作与传感增量。本轮暂不进入这一分支。

## 12. 代码状态、核验更正与置信度

本轮论文以原论文/正式会刊为主，作者仓库核发布状态；论文与仓库不计两项独立研究。`T1`表示正式同行评审原文或官方工程资料，`T1P`表示原论文预印本/正式venue未确认；工程资料权威不等于科学效果已同行评审。协议是否直接支持按核验表的SUPPORTED/PARTIAL记录；跨论文判断另外标置信，不由单篇venue推导普遍效果。

| 综合判断 | 置信与证据范围 |
|---|---|
| 额外编排/计算不保证收益，必须设强单体与实耗成本控制 | [High] R06/R10/R11/R13/R23至少三项独立原始研究共同支持；限一般评测原则，不代表航拍效应 |
| 持久观测状态和阶段计划值得作为单体/协作共同输入 | [Medium] R18/R21及本地S4适配限制；在航拍中的收益未证 |
| 同事件交接能避免“消息发送后无人再次读取”的协议缺口 | [Medium] 本地旧日志与R02接口支持可执行性；不等于SR改善 |
| P1或P2在本网格胜过同成本强单Agent | [Low / 未验证] 没有本项目新实测；这正是实验假设 |

| 项目 | 2026-10-01可确认状态 | 使用限制 |
|---|---|---|
| [GOMAA-Geo](https://github.com/mvrl/GOMAA-Geo) | 作者训练/验证源代码公开；README Model Zoo仍Coming Soon | 权重未就绪；不等于安装即复现 |
| [GeoExplorer](https://github.com/limirs/GeoExplorer) | 作者训练/验证及数据准备实现公开 | 官方checkpoint链接本轮未核实 |
| [IGL-Nav当前README](https://raw.githubusercontent.com/GWxuan/IGL-Nav/main/README.md) | 明确Code is coming soon；ICCV2025为作者公布状态 | 主Agent交付前补读消除了来源卡“当前未核实”；未证明代码已发 |
| [AeroBelief当前README](https://raw.githubusercontent.com/Shawnjx/AeroBelief/main/README.md) | 明确源码/配置/评测在论文接受后发布 | 主Agent补读确认当前声明；不能据此证明论文录用状态 |
| [UniGoal](https://github.com/bagh2178/UniGoal) | 作者实现公开，既有记录支持instance image-goal | 未安装、未验证航拍四动作兼容性 |
| [DiscussNav](https://github.com/LYX0501/DiscussNav) | 作者实现公开 | 视野/模型/调用协议不同 |
| [ReflectVLN](https://github.com/AIprogrammer/ReflectVLN)、[VeGAS](https://github.com/nishadsinghi/vegas) | 本轮README显示代码/数据或Code Coming Soon | 项目页/性能表不等于可复现实现 |

关键核验更正：ReflectVLN在线C由模型预测、训练恢复使用特权监督；VeGAS存在同调用数Self-Consistency消融；Scaling的180/4与260/6属于不同版本；MAST正式D&B venue与人工/扩展数据量要分开。DiscussNav由作者与索引定位到ICRA DOI `10.1109/ICRA57147.2024.10611565`，正式页工具无法访问，保留venue部分核验；VeGAS Findings只按作者声明，不写已核正式主会。这些保留项不影响本次预算/强基线结论。[10项独立核验表](D:/桌面/XDUCS-courses/大四/大数据工程/调研/15_多Agent专项引用核验_v1.md)、[代码与近期缺口卡](D:/桌面/XDUCS-courses/大四/大数据工程/调研/11d_近期缺口与代码状态复核.md)。

## 13. 研究方法、局限和后续决策顺序

用户确认后冻结五个问题：可协作失败、公开触发、角色消息、公平因果、主线比较；四个检索/补缺角色分工，另有独立引用核验与综合反证，共六个研究子Agent，未调用项目导航模型。检索组合导航正例、同成本负例、计算触发、同执行体近期缺口与多UAV边界，主Agent整合本地验收、纠正版本与协议。研究计划和查询族见[11研究计划](D:/桌面/XDUCS-courses/大四/大数据工程/调研/11_多Agent专项研究计划_v1.md)；原始提取卡为[11a](D:/桌面/XDUCS-courses/大四/大数据工程/调研/11a_多Agent导航与角色来源.md)、[11b](D:/桌面/XDUCS-courses/大四/大数据工程/调研/11b_协作失败与同成本证据.md)、[11c](D:/桌面/XDUCS-courses/大四/大数据工程/调研/11c_介入触发与公平评测来源.md)、[11d](D:/桌面/XDUCS-courses/大四/大数据工程/调研/11d_近期缺口与代码状态复核.md)，反证记录见[17](D:/桌面/XDUCS-courses/大四/大数据工程/调研/17_多Agent专项综合与反证_v1.md)。

本专项去重后24项研究，不把同论文HTML/PDF、项目页、仓库或学位论文重复计数；旧综述47项背景不再次包装为新增。独立核验抽查10组高影响claim，未逐条复现每篇表格；预印本、不可读正式venue和未核权重均保留限制。当前检索未找到满足全部同执行体/观测/记忆/总成本/消息消融条件的直接航拍证据；这是本次检索结果，不是对所有存在文献的不存在证明。没有为达到数量目标加入弱相关材料。

后续若用户另行决定执行，顺序应是：公开前缀机会检查→冻结状态/触发/接口→D消息机制和能力诊断→G同成本小试→新独立来源确认。每阶段允许停止。当前交付仅完成研究及建议规格，阶段T/D/G都未启动；受保护旧文件的哈希核查见交付核验记录。

## 14. 去重参考文献与支持定位

以下24项均在正文使用；作者采用首作者等的简记，完整作者见原文。支持位置是本轮来源卡/独立核验实际使用的位置，不暗示复现结果。

| 编号 | 原始研究、时间与状态 | 支持位置 / 来源卡 |
|---|---|---|
| R01 | Long et al. [Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions](https://arxiv.org/abs/2309.11382)，2023预印本；作者列ICRA2024，venue部分核验 | 方法、多专家/消融；A01/C02；T1P原文，作者venue声明 |
| R02 | Wang et al. [ReflectVLN: Training Vision-Language Navigation Agents with Reflective Reasoning](https://arxiv.org/abs/2607.12680)，2026-07预印本 | 意图/执行、反思训练数据、基准比较；A02＋15；T1P |
| R03 | Hariharan et al. [Plan Verification for LLM-Based Embodied Task Completion Agents](https://arxiv.org/abs/2509.02761)，2025；来源卡列NeurIPS Workshop | 方法与TEACh文字计划评估；A03；不当主会证据 |
| R04 | Singhi et al. [Think Twice, Act Once: Verifier-Guided Action Selection For Embodied Agents](https://arxiv.org/abs/2605.12620)，2026-05；作者标CVPR Findings | 核验训练、§6.2/Fig.6；A04＋15；T1P，正式venue未确认 |
| R05 | Zhu et al. [CA-VLN: Collaborative Agents in MLLM-Powered Visual-Language Navigation](https://doi.org/10.3390/s26041254)，Sensors 26(4):1254，2026 | 方法/训练与融合；A05；T1 |
| R06 | Kim et al. [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296)，2025首发，2026-04 v3 | 摘要版本/任务结构/成本；A06/B01/C06；T1P |
| R07 | Cemri et al. [Why Do Multi-Agent LLM Systems Fail?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html)，NeurIPS2025 D&B | 正式摘要、分类/数据；A07/B03/C07；T1 |
| R08 | Zuo et al. [FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-and-Language Navigation](https://openaccess.thecvf.com/content/CVPR2026/papers/Zuo_FantasyVLN_Unified_Multimodal_Chain-of-Thought_Reasoning_for_Vision-and-Language_Navigation_CVPR_2026_paper.pdf)，CVPR2026 | 摘要/推理内化接口；A08；T1，未迁移其效应数值 |
| R09 | Shen et al. [Enhancing Multi-Robot Semantic Navigation Through Multimodal Chain-of-Thought Score Collaboration](https://ojs.aaai.org/index.php/AAAI/article/view/33607)，AAAI2025 | 任务、RGB-D多机器人协作；A09；T1 |
| R10 | Wang et al. [Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies](https://aclanthology.org/2024.emnlp-main.1112/)，EMNLP2024 | 摘要/预算比较；B05；T1 |
| R11 | Smit et al. [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://proceedings.mlr.press/v235/smit24a.html)，ICML2024 | 摘要、成本/时间/准确率、协议敏感性；B06；T1 |
| R12 | Zhang et al. [If Multi-Agent Debate is the Answer, What is the Question?](https://arxiv.org/abs/2502.08788)，2025-02预印本 | 多策略/多基准比较；B07；T1P |
| R13 | Tran & Kiela. [Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets](https://arxiv.org/abs/2604.02460)，2026-04预印本 | §1/§5、thinking-token定义/计量限制；B08＋15；T1P |
| R14 | Wu et al. [Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning](https://arxiv.org/abs/2511.07784)，2025-11预印本 | 逻辑任务、独立/群体意见比较；B09；T1P |
| R15 | Liang et al. [Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate](https://aclanthology.org/2024.emnlp-main.992/)，EMNLP2024 | 方法、修订与任务比较；B10；T1 |
| R16 | Du et al. [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)，2023首发；ICLR2024作者记录 | 独立答案→交叉意见→修订；B11；原文，不作等成本结论 |
| R17 | Horvitz, Cooper & Heckerman. [Reflection and Action Under Scarce Resources: Theoretical Principles and Empirical Study](https://www.erichorvitz.com/reflect.htm)，IJCAI1989 | 作者原文/有限资源计算价值；C01；T1 |
| R18 | Chen et al. [MapGPT: Map-Guided Prompting With Adaptive Path Planning For Vision-and-Language Navigation](https://aclanthology.org/2024.acl-long.529/)，ACL2024 | 在线地图/自适应规划方法；C03；T1 |
| R19 | Paglieri et al. [Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents](https://arxiv.org/abs/2509.03581)，2025首发；2026-02 v3 | §4.3/§4.4/§5.1，SFT＋PPO/动态提示限制；C04；T1P |
| R20 | Zhu et al. [Scaling Test-time Compute for LLM Agents](https://arxiv.org/abs/2506.12928)，2025-06预印本 | 推理扩展策略/资源；C05；T1P |
| R21 | Deng et al. [SpaceVLN: A Zero-Shot Vision-and-Language Navigation Agent with Online Spatial Cognitive Memory and Reasoning](https://arxiv.org/abs/2606.08992)，2026-06预印本 | 空间记忆、感知权限与阶段规划；C08；T1P |
| R22 | James et al. [GoalSwarm: Multi-UAV Semantic Coordination for Open-Vocabulary Object Navigation](https://arxiv.org/abs/2603.12908)，2026-03预印本 | §IV、每子任务500步/样本数；C09＋15；T1P |
| R23 | Sanabria. [OpenAI single-agent LLM architecture reduces computational overhead relative to multi-agent orchestration in a simulated Mars rover decision-support benchmark](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1877762/full)，Frontiers in Robotics and AI，2026-07-06 | §3.5–3.12/§4.2–4.6/§5，结构化输入与成本；D01；T1 |
| R24 | Wu et al. [Planner Matters! An Efficient and Unbalanced Multi-agent Collaboration Framework for Long-horizon Planning](https://arxiv.org/abs/2605.02168)，2026-05预印本；作者[UCSD论文稿](https://escholarship.org/uc/item/2qv9h26p)为同研究补充 | 角色/记忆、附录verifier消融与未配平限制；D02；T1P |

官方代码入口另列于第12节，不进入24项研究计数。2024及更早文献作为[foundational]方法/预算评测基础保留，不用于宣称2026最新性能；2025较早的多机器人文献也仅作问题定义基础。近期工具状态以上述2026-10-01直接复核为准。旧背景综述入口：[2026-09-30主动图像目标搜索综述](D:/桌面/XDUCS-courses/大四/大数据工程/调研/DEEP_RESEARCH_active_image_goal_search.md)。
