---
format: 1920x1080
duration: 257s
message: "从云端VLM失败到边缘线索突破：可信目标证据让主动地理定位在本地S2正式通过"
arc: "Hook → 任务定义 → 平台评测 → 云端VLM失败 → 四臂诊断 → 本地重建 → 目标难题 → 边缘突破 → S2复验 → 展望"
audience: "课程教师与同组同学：懂机器学习基础，未读过本项目代码"
mode: autonomous
language: zh
music: none
---

## Video direction

- **palette（frame.md 钴蓝坐标纸）**：paper #F0EBDE 为唯一底色，ink #1F2BE0 为唯一主墨（强调/图形/数据线），ink-soft #5560E5 作次级，永久 graph-paper 网格与上下两条 cobalt hairlines 贯穿全片；成功/通过语义用 ink 实心填充，失败/冻结语义用 ink 空心+置灰，不引入第三色相。中文正文用 "Noto Sans SC", "Microsoft YaHei", sans-serif 栈，中文大字用 "Noto Serif SC", "SimSun", serif 栈，数字/代码/英文标签沿用预设 DM Mono 角色；所有数字 tabular-nums。
- **motion grammar + reveal model**：power3 长尾缓动为默认（平滑优于弹跳）；每帧遵循 VO-paced reveal——元素只在旁白说到时入场，禁止 front-load；计数一律 value-scaled counter（数字随值增大），图表生长用 bars/fills，图解用 SVG self-draw；hold 期至多 subtle jitter，禁止 lazy breathing 与无意义 pan/push。
- **rhythm / held 分配**：F9 是全片计数高潮（长 hold），F10 是静收帧；F3 尾、F7 尾、F9 尾为刻意静止读拍；其余帧按句推进。
- **caption keep-out**：底部约 17% 为字幕带，全部关键内容排进上部 83%；居中 hero 锚在 y≈0.42×高。
- **negative list**：不出现真实截图/浏览器 chrome/滚动条、bokeh、"紫蓝 AI 渐变"、stock 照片感、第三色相；两个运动失败模式都禁止——slideshow（开场堆满然后冻结）与 screensaver（元素各自漂浮无叙事）。

## Frame 1 — Hook：没有GPS怎么定位

- status: animated
- src: compositions/frames/01-hook.html
- duration: 18.837s
- transition_in: cut
- poster: 10s
- scene: 大字标语三拍递进：信号消失 → 只凭画面 → 走进目标格；背景坐标纸上浮现5×5网格
- voiceover: 如果导航卫星信号不可用，一架只带相机的无人机，能在大范围未知区域里找到目标吗？这就是主动地理定位。智能体看不到全局地图，只能凭脚下的画面一步一步探索，直到走进目标所在的那一格。
- blueprint: kinetic-type-beats

本帧是冷开场钩子。三段递进短语逐拍落下，随后概念名锁定，5×5网格作为主题动机首次出现（细钴蓝线框，而非具象地图）。全程纯排版+线条图形。

blueprint: kinetic-type-beats (Adapt) — 保留 beat-slam 逐拍入场与收束锁定的 signature；第三拍由纯文字延展为网格+移动小点的图文合拍。
focal: 概念名"主动地理定位"大字（Scene 2 起）
roles: 递进标语行 = foreground subject · 坐标纸网格+5×5线框 = background（dim ~40%）· 移动小点与英文副标 Active Geo-localization = supporting
Scene 1 (0.0–8.1s): asymmetric 60/40。设问三拍 kinetic beat-slam 逐拍砸入："导航卫星信号不可用"（附一个信号图标划掉）→ "只剩一台相机" → "能找到目标吗？"；每拍一行，前一行退为 ink-faint。仅此三行，其余留白。
Scene 2 (8.1–10.4s): scale-swap：设问组缩小上移让位，"主动地理定位" display 大字居中锁定（hierarchy 3:1），下方 mono 副标 Active Geo-localization type-on。
Scene 3 (10.4–18.8s): 背景处 5×5 网格线框 SVG self-draw 亮起（background 层，dim）；一个小方点沿格线走四步（每步落在语流重音上），终点格 ink 实心填充并一次 highlight 脉冲；两侧 supporting 词"探索 → 定位"入场。定格 hold 至帧尾。

## Frame 2 — 任务定义：五乘五网格世界

- status: animated
- src: compositions/frames/02-task.html
- duration: 27.562s
- transition_in: crossfade
- poster: 16s
- scene: 左侧5×5网格图解逐块点亮；右侧目标图卡与当前图卡对置；底部四方向键与预算条依次入场
- voiceover: 我们的平台把这个任务抽象成一个五乘五的网格世界。智能体每一步拿到两样东西：一张给定的目标图，和脚下的航拍图块。它可以选择上下左右移动，越界原地不动也要消耗一步，十步预算内走到目标格才算成功。目标坐标和真实距离是神谕信息，物理隔离在评测端，智能体永远看不到。
- blueprint: grid-card-assemble

本帧教任务规则。图解优先，全部是发明出来的示意图形（抽象航拍色块，不用真实图片）；神谕隔离用"评测端"印章盖住可视化来表现。

blueprint: grid-card-assemble (Adapt) — 保留 stagger 自组装 signature；组装对象从"卡片阵列"变为任务图解部件（网格、图卡、方向键、预算条）分窗入场。
focal: 5×5 网格图解
roles: 网格与移动小点 = foreground subject · 目标图卡/当前图卡 = supporting（Scene 2 起升为次焦点）· 方向键与预算条 = supporting · "评测端"印章卡 = supporting（Scene 4）
Scene 1 (0.0–5.1s): centered，网格占 ~50%。5×5 网格格子 stagger 自组装（cascade），其中 3 格标记 ink-faint 已访问；小点从一角入场。
Scene 2 (5.1–11.5s): 重排为 asymmetric 60/40：网格缩至左侧，右侧两张竖卡对置入场——目标图卡（抽象航拍纹理A：色块+道路线条示意）与当前图卡（纹理B），卡头 mono 标签"目标图 GOAL / 当前图 CURRENT"，卡间一条虚线连接。
Scene 3 (11.5–19.8s): 底部 full-width strip：四方向箭头键逐个弹入；10 步预算条按段点亮（10 格），小点演示越界：撞上边框回弹原位、预算段照常熄灭一格（细节即规则）。
Scene 4 (19.8–27.6s): 右上盖"评测端"斜置印章卡落章（press-release spring）：卡内两行"目标坐标 / 真实距离"被遮罩条盖住，"智能体不可见"高亮；全帧 hold。

## Frame 3 — 平台与评测：工程底线

- status: animated
- src: compositions/frames/03-platform.html
- duration: 27.737s
- transition_in: crossfade
- poster: 14s
- scene: 三个统计卡依次计数：SR分层C4–C8、137张训练源图、196项测试；一条"预登记→冻结→审计"流程线贯穿
- voiceover: 评测的公平性靠工程保证：智能体只能读取公开状态，答案只在评测端解密。主指标是成功率，按起始距离四到八格分层统计；辅助指标是终点距离。数据使用Masa航拍数据集，一百三十七张训练源图，验证与测试严格分开。全套平台有一百九十六项回归测试护住底线。
- blueprint: dataviz-countup

本帧建立可信度：两个计数器、一个指标图解、一条流程线，全部按句入场。

blueprint: dataviz-countup (Adapt) — 保留 value-scaled counter 与 stats 卡 signature；"推镜穿行"简化为一次结尾推近聚焦 196。
focal: "196 项回归测试" hero 数字（Scene 4）
roles: 双通道图解 = foreground（Scene 1）· SR/SG 指标卡 = supporting→次焦点 · Masa 数据卡与 137 计数 = supporting · 196 大数字 = Scene 4 焦点
Scene 1 (0.0–7.9s): split-screen。左盒"智能体：只读公开状态"，右盒"评测端：持有答案"（加锁形线条 SVG self-draw），一条单向箭头连接，箭头上小字"公开 Observation"。
Scene 2 (7.9–15.2s): 下半场两卡并置入场：SR 卡带 C4–C8 五个小分段条（等宽，ink 描边）与"按起始距离分层"注；SG 卡带 0→N 终点距离数轴示意。
Scene 3 (15.2–22.6s): Masa 数据卡入场：137 计数器 value-scaled counter 滚动；下方 train/val/test 三段横条按 137/4/10 比例填充，比例失衡本身即信息（test 段极短+标注"冻结未用"）。
Scene 4 (22.6–27.7s): 一次轻推近（push/focus），"196 项回归测试" hero 数字计数定格居中偏上，一条 ink 勾线 sweep 扫过；hold。

## Frame 4 — 云端VLM：接口通了，策略失败了

- status: animated
- src: compositions/frames/04-vlm.html
- duration: 27.887s
- transition_in: crossfade
- poster: 15s
- scene: 请求构成图解逐层入场；0/20 大字计数落定；网格上小点ABAB往复弹跳；结尾两行对置
- voiceover: 第一阶段接入云端视觉语言模型。每一步把目标图、当前画面，加上位置、预算和已访问格子发给模型，让它输出下一步动作。结果出乎意料：二十道验证题零成功。轨迹分析发现，模型在两个格子之间来回打转，十九道题出现ABAB循环。接口是通的，策略却是失败的。
- blueprint: kinetic-type-beats

本帧是第一个负结果：先讲请求构成，再用 0/20 与 ABAB 循环两个可视化证据收束。

blueprint: kinetic-type-beats (Adapt) — 保留逐拍入场与结尾反差对置的 signature；中段加入网格 ABAB 循环图解作为 element beat。
focal: "0 / 20" 大数字与 ABAB 循环图解
roles: 请求构成图解 = foreground（Scene 2）· 0/20 数字 = Scene 3 焦点 · 两格循环小点与轨迹 = Scene 4 焦点 · 结尾对置句 = foreground
Scene 1 (0.0–3.7s): 标题拍："云端视觉语言模型" kinetic beat-slam 落定，mono 注 gemma4:31b · OpenAI-compatible。
Scene 2 (3.6–12.4s): asymmetric 60/40 图解：左侧三元素按语流逐层入场（目标图卡→当前图卡→公开状态标签组：位置/预算/visited），汇聚箭头指向右侧云形节点；从云节点 type-on 弹出 `{"action":"…"}` JSON 小卡。
Scene 3 (12.4–16.5s): scale-swap 清场换焦点："0 / 20" 大数字 value-scaled counter 落定（3:1 层级），下注"二十道验证题"。
Scene 4 (16.5–24.1s): 网格局部（2×2 特写）入场：小点在 A、B 两格间往复弹跳（bounded loop），身后拖出往返虚线轨迹 self-draw；标签"19/20 题出现 ABAB 循环"。
Scene 5 (24.1–27.9s): 结尾对置：左行"接口是通的"（ink 实心）与右行"策略却是失败的"（ink 空心+删除线质感）hard-cut 同拍落下；hold。

## Frame 5 — 四臂对照：规则赢了模型

- status: animated
- src: compositions/frames/05-fourarm.html
- duration: 26.887s
- transition_in: crossfade
- poster: 14s
- scene: 五根柱状条按叙述顺序依次升起计数：A 5% / B 10% / C 10% / Random 5% / D Frontier 50%；D柱高亮
- voiceover: 四臂对照实验给出了更完整的诊断。单动作模型成功率百分之五，动作排序百分之十，加上防重复治理器仍是百分之十；而不看图的Frontier规则探索，成功率高达百分之五十。治理器把重复访问率降了一半，却换不来定位成功。云端路线按预登记协议暂停，问题转向本地。
- blueprint: dataviz-countup

柱状对比是本帧全部叙事：柱子按叙述顺序生长，Frontier 最后高亮。

blueprint: dataviz-countup (Reproduce) — 柱顶计数 + 逐柱生长即 signature；结尾一次 pull-back 收束。
focal: 五根成功率柱（D 柱为最高焦点）
roles: 柱状图 = foreground subject · 柱顶百分比计数 = supporting（与柱同步）· 注解卡与收束行 = supporting
Scene 1 (0.0–3.7s): full-width strip。空白坐标系轴线 SVG self-draw，五个柱位 mono 标签先行（A 单动作 / B 排序 / C 排序+治理器 / Random / D Frontier），柱体零高。
Scene 2 (3.6–16.9s): 五柱按语流依次生长到各自高度（bars fill + 柱顶 value-scaled counter）：A 5% → B 10% → C 10% → Random 5% → D 50%；D 柱长成时 ink 实心高亮并一次 glow blooms。
Scene 3 (16.9–21.8s): C 柱旁弹出注解卡："重复访问率 55.2%→37.1%"，下一行"SR 未涨"加删除线质感高亮。
Scene 4 (21.8–26.9s): 整图 subtle pull-back，底部一行"云端路线暂停 → 问题转向本地"per-word 入场；hold。

## Frame 6 — 本地重建：小策略网络

- status: animated
- src: compositions/frames/06-local.html
- duration: 26.687s
- transition_in: crossfade
- poster: 15s
- scene: 训练状态剧场：步数计数滚动、编码器置灰；成绩卡44.33%→66.00%依次落定打勾，徽章弹出
- voiceover: 本地分支在八GB显存的笔记本上重新开始。冻结卫星图编码器，只训练一个八十万参数的小策略网络。修正到达奖励的错误之后，奖励塑形把成功率带到百分之四十四；再加入合法动作筛选，禁止越界和无效动作，三个种子全部提升，平均百分之六十六，越界率降到零。
- blueprint: agent-progress-theater

把训练做成工作剧场：机器在转、清单在勾，状态变化即叙事。

blueprint: agent-progress-theater (Adapt) — 保留 loader/状态剧场 + 成绩清单 check-off 的 signature；清单对象换成两张成绩卡+徽章。
focal: 成绩卡对 44.33% → 66.00%
roles: 训练状态条（步数计数/转轮）= foreground 剧场层 · 组件清单（冻结编码器/小策略网络）= supporting · 成绩卡与徽章 = Scene 3 焦点
Scene 1 (0.0–4.8s): 舞台布置：右上角 mono 角标"8GB 显存 · RTX 4060"；中央训练状态条入场——步数计数器开始滚动、一个小转轮自旋（live SVG internals），状态词"训练中"闪现。
Scene 2 (4.8–10.4s): 左侧组件清单两项落位："Sat2Cap 编码器"图标置灰并盖"冻结"小章，"策略网络 ~80万参数 GRU"高亮描边；计数器持续滚动。
Scene 3 (10.4–26.7s): 右侧成绩区 check-off 序列：卡1"PBRS 奖励塑形 44.33%"落定打勾 → 卡2"+ 合法动作筛选（禁越界/无效动作）"滑入 → 卡3"66.00%"落定打勾并弹出两枚徽章"3/3 种子提升""越界率 0"；末尾全帧 hold 定格在 44.33→66.00 的对比上（66.00 放大 3:1 层级）。

## Frame 7 — 目标难题：模型没用目标图

- status: animated
- src: compositions/frames/07-gap.html
- duration: 24.1s
- transition_in: crossfade
- poster: 13s
- scene: 目标图卡被"均值"卡替换后66.00%纹丝不动；四张失败实验卡逐拍落下盖"负结果"章；结论句大字
- voiceover: 但百分之六十六的模型，真的看懂目标了吗？把目标图换成训练集均值，成功率几乎不掉。目标监督、源图留出诊断、五分类邻接头、解耦线索头，四轮实验全部没能建立可靠的目标证据。问题被精确定位成一句话：模型没有真正利用目标图的信息。
- blueprint: kinetic-type-beats

关键诊断帧："数字不动"本身就是可视化证据，四张失败卡是负结果展示。

blueprint: kinetic-type-beats (Adapt) — 保留逐拍与收束大字 signature；第二拍为"换卡数字不动"的图证拍。
focal: 结论句"模型没有真正利用目标图的信息"
roles: 目标图卡/均值卡与 66.00% 数字 = foreground（Scene 2）· 四张失败卡 = supporting（Scene 3 焦点群）· 结论句 = Scene 4 焦点
Scene 1 (0.0–4.0s): centered 设问："真的看懂目标了吗？"大字拍入，右侧一张目标图卡（抽象纹理A）作为被质疑对象。
Scene 2 (4.0–8.5s): 图证拍：目标图卡 hard-cut 换成灰色"训练集均值"卡（纹理退化为模糊色块），旁边"66.00%"数字保持完全不动，下注"几乎不掉"——不动的数字即证据。
Scene 3 (8.5–17.9s): 2×2 grid 卡群 stagger 落下：目标监督 / 源图留出诊断 / 五分类邻接头 / 解耦线索头，每张卡落定即盖一枚 ink 空心"负结果"小章（press-release）。
Scene 4 (17.9–24.1s): 卡群 dim 退后，结论句 per-word staggered reveal 大字居中："模型没有真正利用目标图的信息"；hold。

## Frame 8 — 边缘突破：接缝里的证据

- status: animated
- src: compositions/frames/08-edge.html
- duration: 32.525s
- transition_in: crossfade
- poster: 17s
- scene: 两个图块自左右对开；四条边接缝段点亮并连线；20维/+2560参数标签；72.62%→88.57%对撞替换
- voiceover: 突破来自一个很细的表征：边缘接缝。把当前图块与目标图块的四条边切成小段，逐段比较颜色误差、梯度误差和相关性，构成二十维特征。只新增两千五百六十个参数，线索头第一次给出可信的邻接判断：目标是否就在相邻格、朝哪个方向。同样的开发集，成功率从百分之七十二点六跃升到百分之八十八点六，最难的短距离档提升最大。
- blueprint: comparison-split

核心方法帧：两图块对开是 comparison-split 的 signature；后接参数/判断小卡与数字对撞。

blueprint: comparison-split (Adapt) — 保留两翼对开 mirrored book-open 与内侧徽章 spring-pop 的 signature；徽章从"能力标签"换为"邻接/方向"判断卡，尾段接数字对撞收束。
focal: 对置的两图块与其接缝连线（前半）；72.62%→88.57% 数字对撞（尾段）
roles: 图块A/B = foreground subject · 接缝段与误差连线 = 次焦点（方法本体）· 20维/+2560/判断卡 = supporting · 数字对与徽章 = 尾段焦点
Scene 1 (0.0–3.9s): 概念名拍："边缘接缝"大字居中，mono 副标 seam features · 20-dim。
Scene 2 (3.9–13.6s): 图解主体：两个 3×3 抽象纹理图块自左右两翼 mirrored book-open 对开并置（split-screen）；随后四条边的接缝小段逐条点亮（1/4/8 像素三种宽度的刻度感），块间误差连线 SVG self-draw；三枚标签按语流弹入："颜色误差""梯度误差""相关性"，汇总徽章"20 维特征"收拢。
Scene 3 (13.6–23.3s): 重排 asymmetric 55/45：图解缩至左侧，右侧两张判断卡依次 spring-pop 打勾："目标在相邻格？✓""朝哪个方向？✓"，上方 mono 小卡"+2,560 参数 · 线索头"。
Scene 4 (23.3–32.5s): 底部数字对撞：72.62% 被 88.57% scale-swap 撞出定格，徽章"+15.95 点"弹入，小注"C4/C5 短距离档提升最大"；hold 定格。

## Frame 9 — S2正式复验：数字落定

- status: animated
- src: compositions/frames/09-s2.html
- duration: 23.625s
- transition_in: crossfade
- poster: 14s
- scene: 89.33%主数字滚动计数定格；73.00%基线对照柱；三个证据标签弹入；右侧12/12清单打勾
- voiceover: 冻结全部权重与阈值，一百道验证题乘三个种子正式复验：成功率百分之八十九点三，终点距离零点三二，比基线高出十六点三个百分点，置信区间不包含零。十二项冻结检查全部通过，独立复核确认。默认配置正式切换到边缘线索方案。
- blueprint: dataviz-countup

全片结果高潮：一个 hero 计数、一组对照、一列证据、一次状态翻转。

blueprint: dataviz-countup (Reproduce) — hero 计数 + 证据卡即 signature；结尾状态条翻转收束。
focal: 89.33% hero 数字
roles: hero 数字 = foreground subject · 对照柱与证据标签 = supporting · 验收清单 = 次焦点（Scene 2）· 状态条 = 收束
Scene 1 (0.0–15.2s): centered 偏上。hero"89.33%"value-scaled counter 滚动上升定格（占 ~45%），左下小柱对照"基线 73.00%"（空心）与新柱（ink 实心）并置；mono 标签"val100 × 3 种子 · 5×5 / B=10 · 冻结权重"；三枚证据标签按语流弹入："+16.33 点""95% CI [+9.33, +23.33]""SG 0.320"。
Scene 2 (15.2–19.5s): 右侧验收清单卡快速 check-off：计数"12/12 冻结检查"滚动，三行示例检查项勾线 sweep（任务完成 / SR≥60% / 目标对照+5点），末行"独立复核 ✓ 通过"。
Scene 3 (19.5–23.6s): 底部状态条翻转变色（ink 空心→实心）："默认配置 → EdgeTargetCue"；全帧长 hold（刻意静止读拍）。

## Frame 10 — 展望与收束

- status: animated
- src: compositions/frames/10-outro.html
- duration: 21.262s
- transition_in: crossfade
- poster: 11s
- scene: 台账行浮现；三条下一步路线逐条入场；课程名与仓库名收束为尾卡，静态保持
- voiceover: 项目至今沉淀了完整的实验台账：每个批次可重放审计，失败与负结果全部保留。下一步是独立源图确认、测试集终评与跨数据集验证。主动探索的定位方法，正在从平台走向方法，再走向证据。谢谢观看。
- blueprint: titlecard-reveal

收束帧克制冷静：低运动量，chain 式三卡，最后长 hold。

blueprint: titlecard-reveal (Reproduce) — 卡链 hard-cut 接缝 + 末卡长 hold 即 signature。
focal: 末卡（课程名 + 仓库名）
roles: 台账卡 = 卡1 · 路线卡 = 卡2 · 尾卡 = 卡3（末段焦点）
Scene 1 (0.0–7.9s): 卡1 台账：两行"每个批次可重放审计""失败与负结果全部保留"+ 一条小轨迹回放示意线（SVG self-draw 一条折线掠过 5×5 缩略网格）。
Scene 2 (7.9–13.6s): 卡2 路线（hard-cut 接缝）：三行编号路线逐条 per-word 入场："01 独立源图确认 S3""02 测试集终评""03 跨数据集验证"。
Scene 3 (13.6–21.3s): 卡3 尾卡（hard-cut 接缝）：课程名"基于主动探索的视觉地理定位方法设计与实现"衬线大字 + mono"cdxDNRF/GeoNav-Agent · 2026-09"，"谢谢观看"小字；长 hold 至帧尾（至多 subtle jitter）。
