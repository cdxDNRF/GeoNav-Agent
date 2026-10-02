# 主动图像目标搜索调研计划

日期：2026-09-30（Asia/Shanghai）。深度：deep。工作范围：研究与方案设计，只读现有代码、数据与实验材料；不执行导航训练或付费 VLM 推理，不修改项目代码。

## 冻结研究问题

给定目标图片，智能体在未知网格内仅通过逐步视觉观测和公开空间状态，如何结合预训练模型、长期证据记忆与规划，在有限动作预算内找到对应位置，并证明收益来自目标视觉信息而非单纯覆盖策略？

用户已确认目标为图片。尚未单独确认的资源项将从当前本地文档只读核验；查不到的项明确记为未知，不以假设冒充事实。

## 六个检索区域

1. 直接任务：GOMAA-Geo / GeoExplorer / DynCur-Geo 的目标与观测模态是什么？训练和推理的信息权限是什么？状态与奖励是什么？哪些组件而非整体可复用？
2. 可迁移视觉导航：ImageNav、实例 ObjectNav 与类别 ObjectNav 如何区分？VLM 直接动作和语义规划是否存在公平比较？室内算法迁移到航拍有哪些假设缺口？
3. 分层与记忆：如何从区域排序发展到持久 waypoint / options？目标相关拓扑、视觉和检索记忆如何实现？哪些记忆解决跨步证据积累而不是防回头？
4. 主动感知与搜索：belief 定义与似然从何而来？什么条件下可以计算真正的信息增益？图搜索 / beam / MCTS 是否需要不可获得的未来图像？
5. 轻训练与学习策略：scorer / adapter / IL / RL 训练预算如何确定？怎样隔离源图与地区？已有目标遮蔽负结果如何改变优先级？
6. 重研究与后续扩展：world model 可以预测什么、不能预测什么？多无人机如何共享证据、分配区域和计通信成本？哪些方案应暂缓？

## 初始检索词

GOMAA-Geo observation goal image; GeoExplorer active geolocalization; DynCur-Geo state dynamics reward; image-goal navigation instance target 2025 2026; VLM navigation planner direct action failure; hierarchical waypoint navigation replanning; topological visual episodic memory navigation; goal-conditioned retrieval navigation; semantic map instance goal confusion; Bayesian active visual search information gain; belief-space planning observation model; learned heuristic graph search visual navigation; zero-shot aerial target search limitations; navigation world model generalization; collaborative embodied exploration communication cost.

## 流程与产物

三个检索子代理并行覆盖前三组证据包，受并发槽限制，额外缺口检索和独立核验顺序安排。目标 30–50 项去重的一手研究来源，重点 2023–2026，经典算法仅作为方法基础。原论文、同一论文项目页和代码不计为三份独立研究证据。

主代理整合任务差异、检查高影响结论、构建约 15 条候选方案与 3–5 个可直接开展的实验。主目录 `research.md` 保存高价值决策摘要，完整十九节报告与证据、论文卡、方法学记录保存在本目录。

## 证据规则

- 论文事实与本项目迁移建议分别标注；后者是待实验验证的设计推断。
- arXiv 预印本不等于同行评审；会议归属、代码发布状态须单独核实。
- 不将不同地图、动作预算、目标模态和成功条件下的 SR 直接排行。
- 无目标坐标、未访问图像或未访问特征进入 Agent；oracle 只允许用于训练标签和隔离的评测诊断。
- 当前协议自动到达成功、没有 stop：不能把目标确认或停步模块的收益直接套用。
- 单个新方法即使有一手论文仍只能支持该实验协议；航拍迁移收益需配对实验验证。
