# 多Agent专项补充研究计划

日期：2026-10-01（Asia/Shanghai）。Phase 0：用户已确认按任务书范围开展。

## 范围与目标

以2026-09-30主动定位综述为基础，先校正项目状态，再判断同一导航执行体内多决策Agent是否补足明确失败机制。最多推荐1–2项下一轮方案与同成本实验；允许结论为无需扩大协作。多无人机只作次级协议对照。保留旧综述、root research.md、代码、模型、阈值、旧结果与原始数据。

本轮只读当地材料、公开论文/仓库检索和方案写作；不运行训练、项目模型推理、云端VLM/API、导航评测或付费探测。研究分工代理只检索文献。

## 五个区域与问题

1. **角色协作与任务可分解性**：DiscussNav及近两年embodied multiagent如何定义独立状态、消息和执行？是否与单Agent工具/两次建议区分？原论文的导航/目标和当前航拍格子有何差异？
2. **协作退步与计算预算**：Scaling Agent Systems、MAST与equal-budget debate研究是否支持当前串行导航？相关错误、communication与moretokens如何控制？反思和独立复核哪个更有证据？
3. **失败前合法介入**：预算风险、无新覆盖、option停滞、重复假设和证据冲突可以识别什么？哪些trigger无需真目标距离/未访问图像？何时根本没有可推断的新信息？
4. **机制和同成本评测**：代表性队列与失败诊断队列如何分开？触发、弃权、改动作、恢复/损伤和跨轮消息怎样验证？calls、实耗token、图片和时延如何分别配平？
5. **与主线扩展比较**：协作相对扩大真实覆盖、真实异时/任意角度目标和跨步表征哪项更优先？多无人机怎样固定总动作、比较分区、共享证据并计通信？

## 检索词

DiscussNav multiple experts navigation; scaling agent systems sequential task equal compute; MAST multi-agent system failures; multi-agent debate same token budget single agent reflection; independent verifier correlated errors; embodied navigation agents role ablation 2025 2026; vision grounded collaboration failure cases; adaptive deliberation value of computation; event triggered replanning observable failure; budget aware navigation stagnation; goal evidence episodic memory counter evidence; agent collaboration message ablation; self correction without external feedback; navigation multi-agent evaluation paired trajectories; cooperative aerial search same total action budget.

## 方法与产物

Deep流程采用三位并行检索代理，第四波按实际缺口补检，独立引用核验与综合/红队顺序安排。优先相关性与原始证据，数量不能替代质量；不会为30–50目标填无关论文。论文与其代码/项目页去重计数。T1为同行评审一手论文，T1P为预印本，一手代码只证明可用性，不算科学独立佐证。

新补充文件全部写入已有“调研”目录：一页摘要、专项报告、1–2个冻结实验建议、引用核验及旧综述勘误。研究提议与论文事实、本地保存结果分别标记；实验阈值属于将来预登记，不将+2点当普适论文标准。

## 最重要的现有负证据

20张开发图×1题的三轮API重复不增加独立地图数。旧协作在3道原成功题触发、唯一失败题未触发；45导航请求均unknown、0改动作。第二轮从未发生，不能声称已测跨轮反馈，也不能由等SR或零宽区间证明统计等价。新的研究应先找到有机会且有证据的介入，再讨论扩大Agent数。
