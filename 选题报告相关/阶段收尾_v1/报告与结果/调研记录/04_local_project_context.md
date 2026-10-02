# 本地项目边界与已有实验：只读核验

核验日期：2026-09-30。此文件记录本次调研时的快照，不替代各批预登记、原始结果或独立复核；未重新运行训练、评测或云端调用。

## 已确认的信息权限

直接读取 `project/src/env/environment.py` 的 `Observation`、`StepInfo`、`GridWorldEnv.step` 和 `evaluator_result`：

- 观察：目标图、当前图、当前网格位置、grid_size、remaining_budget、visited；StepInfo 为越界、重访、步数。
- 实际动作名为 `up/right/down/left`。文献中的 north/east/south/west 仅作概念映射，不改变本地协议。
- 环境为固定 5×5。基础观察的 legal_actions 默认四动作；不同 Agent 的边界筛选需独立登记，不能把环境已免费屏蔽越界作为默认假设。
- 越界原地且耗一步；预算内第一次到达成功，包括最后一步；无 stop。成功导致 episode 终止，目标确认器在已经命中之后没有额外决策机会。
- SG 是终点到目标的曼哈顿距离，对全部 episode 平均，成功为零；不要混同论文其他距离归一化或 SPL。
- 全图、未访问图像/embedding、目标坐标、真实目标距离及答案日志仅属于环境/评测端，禁止进入 Agent。

当前网格内自身位置已知，因此主要未知量是目标位置与未观察区域的视觉内容；并非机器人自身位姿全未知的 SLAM。稀疏的“见过的图像+坐标+线索”状态比新建 SLAM 更贴合当前系统。

## 数据与资源

`README.md` 记载本地 Masa 源图 train/val/test=137/4/10，共151张，和论文的1188张口径不能直接对应。本次读取 `任务清单_v2/manifest_val.json`，确认 val 仅4张源图、100个 episode，5×5、B=10、初始距离4–8、源图每距离5次有放回采样；部分路线重复，episode 不能都视作独立地理样本。test 不作方案选择。

GPU 通过 `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` 当前核验为 NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB。建议冻结视觉编码器、训练小 scorer/记忆头的资源定位有实际依据；未进行吞吐或显存峰值基准，不能据此承诺任何模型的训练时间。

VLM 本地文档记载 gemma4:31b 云端双图服务及另一云端候选。未读取 `.env`、未获取凭据、未探测模型服务在线状态；当前 API 单价、额度、可用性与总预算未知。成本建议以调用数、输入图片数、tokens、延迟和错误率度量，不编造金额。

`论文/GOMAA-Geo.pdf`、`GeoExplorer.pdf`、`DynCur-Geo.pdf` 各为1字节，不是可阅读的论文。本次原论文证据来自在线一手来源。

## 已做方法与不能再当作新建议的项

除了用户附件中的 Random、Frontier、VLM动作/排序、visited、邻域确认和粗区域排序，README 还记载：BC、小GRU、PBRS、冻结动作条件动力学好奇心、合法动作筛选、目标遮蔽/替换、参数容量对照、源图留出、局部匹配、邻接概率与条件方向解耦、图块边缘连续性。

最新本地开发默认由 `project/local_policy_default.json` 确认为 `Small256_NoTarget`，`formal_S2_passed=false`、`vision_target_contribution_proven=false`。NoTarget 保留当前画面与公开状态，因此是“不用目标图”的探索策略，不等于纯无图 Frontier，也不能作为已完成视觉地理定位的证据。

## 影响路线选择的现有结果

以下仅为现有文档与已保存验收文件的读取，未在本次调研复算全部动作与分数。

1. 云端粗区域 G 在小规模 val20 为50%，同批 Frontier也是50%；只可说相对基础VLM观察到收益，不能据此证明目标语义或 hierarchy 超过规则探索。云端 val100 S2 遇到网络/API错误，未完成。
2. 完整val100复验的本地 Small NoTarget 三训练种子平均73.00%；仅4张源图的已知开发证据，不能宣称独立跨地图泛化。
3. 纯冻结 Sat2Cap 输入的邻接线索、邻接/条件方向解耦均未产生合格校准阈值，运行时全部弃权。盲目重复“小目标头+门控”不值得作为新路线。
4. 最新 `边缘连续性可信线索对照_v1` 文档在开发140题报：基线72.62%，Edge真实目标88.57%，均值/错误目标70.24%/70.00%；是真实目标线索的开发候选。读取 `验收结论.json` 确认 candidate_passed=true、default_unchanged=true、formal_S2_passed=false；读取 `独立复核.json` 的通过状态。不能把该批与原val100相减，不能将连续同源切片上的 seam 证据称为通用语义地理理解。

## 对调研的约束

优先讨论持久子目标、目标实例证据记忆、预算内轨迹规划与经源图隔离训练的轻量分数。新技术必须和强 NoTarget 探索器对照，检验目标图是否造成有益的动作改变。局部目标匹配、整体探索与 API 工程可靠性分开度量。

若目标patch与地图patch是同一原图连续切片，匹配边缘可能是合法且高效的任务线索；面向真实无人机跨时相/跨高度/旋转视角，需另立目标图扰动与跨采集协议验证。两者均可研究，但贡献声明应和证据范围一致。

## 来源路径

- [README](/D:/桌面/XDUCS-courses/大四/大数据工程/README.md)（本地链接以桌面客户端实际路径解析为准）
- `D:/桌面/XDUCS-courses/大四/大数据工程/project/src/env/environment.py`
- `D:/桌面/XDUCS-courses/大四/大数据工程/project/src/eval/evaluate.py`
- `D:/桌面/XDUCS-courses/大四/大数据工程/DATA/processed_data/Masa/任务清单_v2/manifest_val.json`
- `D:/桌面/XDUCS-courses/大四/大数据工程/project/local_policy_default.json`
- `D:/桌面/XDUCS-courses/大四/大数据工程/DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1/验收结论.json`
- `D:/桌面/XDUCS-courses/大四/大数据工程/DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1/独立复核.json`

原始本地文件保持原状。研究目录由用户预先提供，主目录 `research.md` 与本目录报告按本次用户指示存放。
