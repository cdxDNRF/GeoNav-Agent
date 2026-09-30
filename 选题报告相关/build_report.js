// 综合工程设计选题报告生成脚本（agent 路线版）
// 运行: node build_report.js   （需在本目录 npm install docx）
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, PageBreak, VerticalAlign, ImageRun,
} = require("docx");

// ---------- 基础常量 ----------
const CN = { ascii: "Times New Roman", eastAsia: "SimSun", hAnsi: "SimSun" };   // 宋体正文（hAnsi 用宋体，使弯引号/破折号按全角渲染）
const HEI = { ascii: "Times New Roman", eastAsia: "SimHei", hAnsi: "SimHei" };  // 黑体标题
const CONTENT_W = 8306; // A4(11906) - 左右边距(1800*2)

// ---------- 通用构件 ----------
const t = (text, extra = {}) => new TextRun({ text, font: CN, size: 24, ...extra });
const cite = (n) => new TextRun({ text: `[${n}]`, font: CN, size: 24, superScript: true });

const p = (runs, opts = {}) =>
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, lineRule: "auto" },
    indent: { firstLine: 480 },
    children: Array.isArray(runs) ? runs : [t(runs)],
    ...opts,
  });

const h1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1, keepNext: true,
    spacing: { before: 320, after: 160, line: 360, lineRule: "auto" },
    children: [new TextRun({ text, font: HEI, size: 30, bold: true, color: "000000" })],
  });

const h2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2, keepNext: true,
    spacing: { before: 220, after: 120, line: 360, lineRule: "auto" },
    children: [new TextRun({ text, font: HEI, size: 28, bold: true, color: "000000" })],
  });

const numItem = (ref, runs) =>
  new Paragraph({
    numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, lineRule: "auto" },
    children: Array.isArray(runs) ? runs : [t(runs)],
  });

const refP = (text) =>
  new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { line: 300, lineRule: "auto" },
    indent: { left: 480, hanging: 480 },
    children: [new TextRun({ text, font: CN, size: 21 })],
  });

const sp = (n = 1) =>
  Array.from({ length: n }, () => new Paragraph({ spacing: { line: 360, lineRule: "auto" }, children: [] }));

// ---------- 表格构件 ----------
const thin = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: thin, bottom: thin, left: thin, right: thin };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

const tc = (text, w, opts = {}) =>
  new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { line: 280, lineRule: "auto" },
      children: [new TextRun({ text, font: opts.bold ? HEI : CN, size: 20, bold: !!opts.bold })],
    })],
  });

const caption = (text) =>
  new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true,
    spacing: { before: 200, after: 100, line: 300, lineRule: "auto" },
    children: [new TextRun({ text, font: HEI, size: 21, bold: true })],
  });

// cantSplit：禁止行内跨页断行
const mkTable = (widths, headerTexts, dataRows) =>
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true, cantSplit: true,
        children: headerTexts.map((h, i) => tc(h, widths[i], { bold: true, align: AlignmentType.CENTER, fill: "EDEDED" })),
      }),
      ...dataRows.map(cells =>
        new TableRow({
          cantSplit: true,
          children: cells.map((c, i) => tc(c.text, widths[i], c.opts || {})),
        })
      ),
    ],
  });

// ---------- 封面 ----------
const coverLabel = (text, w) =>
  new TableCell({
    borders: noBorders, width: { size: w, type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.RIGHT, spacing: { line: 400, lineRule: "auto" },
      children: [new TextRun({ text, font: HEI, size: 28 })],
    })],
  });

const coverValue = (text, w) =>
  new TableCell({
    borders: noBorders, width: { size: w, type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.LEFT, spacing: { line: 400, lineRule: "auto" },
      children: [new TextRun({ text, font: CN, size: 28 })],
    })],
  });

const coverInfoTable = new Table({
  alignment: AlignmentType.CENTER,
  width: { size: 6400, type: WidthType.DXA },
  columnWidths: [1700, 4700],
  rows: [
    new TableRow({ cantSplit: true, children: [coverLabel("小 组 成 员", 1700), coverValue("＿＿＿＿（学号：＿＿＿＿＿＿）", 4700)] }),
    new TableRow({ cantSplit: true, children: [coverLabel("", 1700), coverValue("＿＿＿＿（学号：＿＿＿＿＿＿）", 4700)] }),
    new TableRow({ cantSplit: true, children: [coverLabel("", 1700), coverValue("＿＿＿＿（学号：＿＿＿＿＿＿）", 4700)] }),
    new TableRow({ cantSplit: true, children: [coverLabel("指 导 教 师", 1700), coverValue("＿＿＿＿＿＿＿＿＿＿", 4700)] }),
    new TableRow({ cantSplit: true, children: [coverLabel("填 表 日 期", 1700), coverValue("2026 年 9 月", 4700)] }),
  ],
});

const cover = [
  ...sp(2),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 400, lineRule: "auto", after: 160 },
    children: [new TextRun({ text: "大数据智能方向综合工程设计", font: HEI, size: 44, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 500, lineRule: "auto", after: 240 },
    children: [new TextRun({ text: "选  题  报  告", font: HEI, size: 56, bold: true })],
  }),
  ...sp(2),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 400, lineRule: "auto" },
    children: [
      new TextRun({ text: "设计题目：", font: HEI, size: 30, bold: true }),
      new TextRun({ text: "基于主动探索的视觉地理定位方法设计与实现", font: HEI, size: 30, bold: true }),
    ],
  }),
  ...sp(7),
  coverInfoTable,
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- 正文 ----------
const body = [];

// 一、选题背景与意义
body.push(h1("一、选题背景与意义"));
body.push(p("在地震搜救、灾后应急巡检、边境监测等任务中，无人机经常需要在导航与通信设施受损或不可用的未知大范围区域执行搜索任务。此时机载卫星定位可能失效或不可信，无人机只能依靠自身携带的相机，通过分析连续获取的航拍画面推断自身或目标所处的地理位置，这一任务被称为视觉地理定位（Visual Geo-localization）。它是构建不依赖卫星导航的自主导航能力的核心环节，在应急救援等领域具有重要应用价值。"));
body.push(p("传统视觉地理定位研究遵循“被动式单次预测”范式：给定一张查询图像，通过图像检索、分类或回归等方法一次性输出位置估计。该范式存在两点固有局限：其一，单张图像携带的信息有限，定位误差往往较大；其二，预测一次成型，即使结果明显偏离也没有任何修正机会。对于具备机动能力的无人机而言，这实际上浪费了“可以飞过去看一眼”的交互潜能。"));
body.push(p([
  t("主动地理定位（Active Geo-localization, AGL）将定位问题建模为序列决策任务：智能体处于网格化的地理环境中，每一步仅能观测到脚下的局部航拍画面，需要在有限的搜索预算（步数）内自主决定移动方向，逐步逼近并最终定位目标。近年来，GOMAA-Geo"),
  cite("1"), t("、GeoExplorer"), cite("2"),
  t("等工作引入多模态大模型（VLM/LLM）作为决策核心，以零样本方式利用大模型的地理常识进行推理式探索，展现出远超传统方法的潜力；最新的 DynCur-Geo"),
  cite("3"), t("则通过强化学习进一步拓展了该任务的性能边界。"),
]));
body.push(p("本课题面向这一前沿方向，设计并实现基于主动探索的视觉地理定位方法及配套实验平台，其意义体现在三个方面：应用层面，面向无人机搜救等真实需求，研究不依赖 GPS 的自主定位能力；学术层面，切入当前领域公认的难题——大尺度任务下探索效率下降与无效移动（“回头路”与“弯路”）问题，在智能体记忆与搜索策略方向开展创新；培养层面，课题覆盖环境建模、智能体系统设计、大模型应用与实验评测全链条，算法与工程并重，契合综合工程设计的训练目标。"));

// 二、国内外研究现状
body.push(h1("二、国内外研究现状"));
body.push(h2("（一）被动式视觉地理定位与跨视角定位"));
body.push(p([
  t("视觉地理定位的早期工作以图像检索为主，将查询图像与带地理标签的图像库进行匹配"),
  cite("4"),
  t("，深度学习时代的方法进一步提升了匹配精度"),
  cite("5"),
  t("。随着研究深入，跨视角地理定位（Cross-View Geo-Localization）成为主流路线：将地面或无人机视角的查询图像与卫星视角的参考图库进行特征匹配，代表性基准包括 CVUSA"),
  cite("6"), t("、University-1652"), cite("7"), t("、VIGOR"), cite("8"),
  t("等。被动式方法的共同局限在于“一次预测、不可修正”，且无法利用智能体的机动能力主动获取信息。"),
]));
body.push(h2("（二）主动地理定位"));
body.push(p([
  t("主动地理定位自 2024 年起快速发展，代表性工作对比见表 1。GOMAA-Geo"),
  cite("1"),
  t("提出目标模态无关的多智能体框架，通过搜索、对话、停止等协议组织多个 VLM 智能体协作，以零样本方式完成主动定位；GeoExplorer"),
  cite("2"),
  t("引入好奇心驱动的探索机制与经验记忆，以逐跳（hop-by-hop）方式逐步逼近目标，但其好奇心权重固定，智能体接近目标后仍易被新奇画面吸引而绕路漂移；DynCur-Geo"),
  cite("3"),
  t("采用强化学习（PPO"),
  cite("13"),
  t("）训练策略网络，通过距离门控的动态好奇心与势函数奖励塑形，在多模态、跨场景与大尺度测试中全面超越前两者，但其奖励设计依赖任务真值，泛化能力受训练分布限制。"),
]));
body.push(caption("表 1  主动地理定位代表性工作对比"));
body.push(mkTable(
  [1450, 950, 1850, 2250, 1806],
  ["方法", "发表", "技术路线", "核心机制", "主要局限"],
  [
    [{ text: "GOMAA-Geo" }, { text: "NeurIPS 2024" }, { text: "零样本多智能体（VLM）" }, { text: "搜索/对话/停止协议，目标模态无关" }, { text: "大尺度任务性能受限" }],
    [{ text: "GeoExplorer" }, { text: "ICCV 2025" }, { text: "好奇心驱动探索（零训练）" }, { text: "逐跳探索与经验记忆" }, { text: "好奇心权重固定，邻近目标时易绕路漂移" }],
    [{ text: "DynCur-Geo" }, { text: "arXiv 2026" }, { text: "强化学习（PPO 训练）" }, { text: "距离门控动态好奇心与势函数奖励塑形" }, { text: "需离线训练，奖励依赖任务真值" }],
    [{ text: "本课题" }, { text: "—" }, { text: "零样本多智能体（VLM/LLM）" }, { text: "空间与经验记忆、层级搜索与邻域确认" }, { text: "—" }],
  ]
));
body.push(new Paragraph({ spacing: { line: 240, lineRule: "auto" }, children: [] }));
body.push(h2("（三）研究空白与本课题切入点"));
body.push(p("综合现有工作，本课题识别出以下三点研究空白："));
body.push(p("（1）尺度问题。DynCur-Geo 的压力测试表明，随着搜索地图从 8×8 扩大到 25×25 网格，现有方法（包括其自身）的成功率显著下降，大尺度长距离主动定位仍是开放难题；"));
body.push(p("（2）无效移动问题。现有工作总结的两大失败模式——“已到达目标邻域又漂移离开”与“视觉相似区域反复回访、来回震荡”——分别对应探索过程中的“弯路”与“回头路”，尚未被显式建模与解决；"));
body.push(p("（3）路线空白。最新研究转向训练策略网络，而零训练的智能体（Agent）路线在记忆机制与搜索策略上的设计空间尚未被充分挖掘。经与指导教师讨论，本课题确定沿“多智能体 + 记忆与搜索策略创新”方向开展。"));
body.push(p([
  t("值得注意的是，上述工作共享同一套实验协议——网格化地图、局部观测、步数预算，统一采用成功率（SR）与终止网格距离（SG）作为评价指标，并在 MASA、MM-GAG、SwissView、xBD"),
  cite("12"),
  t("等公开数据集上评测。这为本课题提供了可直接对比的公开基准，无需自行采集构建数据集。"),
]));

// 三、研究目标
body.push(h1("三、研究目标"));
body.push(p("总体目标：构建智能体与地理环境交互的主动地理定位实验平台，设计并实现一套基于多智能体协作、具备记忆与层级搜索能力的主动视觉地理定位方法（Agent 路线、零训练），在公开基准上系统验证其有效性，重点提升大尺度任务性能并减少无效探索。具体目标如下："));
body.push(numItem("numTargets", "平台目标：实现网格化交互式仿真实验平台，支持多模态目标线索输入，网格尺度（5×5 至 25×25）与搜索预算可配置，内置 SR/SG 统一评测与轨迹可视化；"));
body.push(numItem("numTargets", "方法目标：复现 GOMAA-Geo 式多智能体基础框架，并在此基础上完成两项创新设计——面向“少走回头路”的智能体记忆机制与面向“少走弯路”的层级化搜索策略；"));
body.push(numItem("numTargets", "实验目标：在 MASA 等公开数据集上与 Random、贪心及 GOMAA-Geo、GeoExplorer 基线进行系统对比，标准设置下 SR/SG 达到与基线相当的水平；大网格设置下 SR 优于基线（争取提升 5 个百分点以上）；通过消融实验验证记忆模块可显著降低重复访问率（争取降低 30% 以上）；"));
body.push(numItem("numTargets", "成果目标：交付可运行的实验平台与方法代码、完整的对比与消融实验分析，以及课程设计报告。"));

// 四、研究内容与拟解决的关键问题
body.push(h1("四、研究内容与拟解决的关键问题"));
body.push(h2("（一）关键问题分析"));
body.push(p("结合研究现状分析，本课题拟重点解决以下三个关键问题："));
body.push(numItem("numKp", "探索记忆的高效表示与利用：主动地理定位中途几乎无有效反馈，且 VLM 上下文长度有限。如何将探索历史（已访问网格、已排除区域、观测摘要）紧凑编码进决策上下文，使智能体在长程任务中“记得住、不折返”，是首要关键问题；"));
body.push(numItem("numKp", "大尺度搜索空间中的探索—收敛权衡：随着网格尺度增大，搜索空间迅速膨胀。如何设计区域级粗搜与网格级细搜的切换判据，在保证区域覆盖的同时及时收敛到目标，是大尺度任务性能的核心瓶颈；"));
body.push(numItem("numKp", "视觉歧义区域的目标确认：在视觉高度相似的区域，智能体难以判定“已到达目标邻域”，易出现漂移离开或反复震荡。如何设计可靠的停止与确认机制，是任务成功率的关键保障。"));
body.push(h2("（二）智能体与地理环境交互实验平台的构建"));
body.push(p("基于公开数据集构建网格化仿真环境，作为全部方法的统一实验载体：将搜索区域划分为网格，每个网格对应一张航拍图像块；智能体从起点网格出发，每步仅获得当前网格的局部航拍观测，可在上下左右四个方向移动；在搜索预算内抵达目标网格判定任务成功。平台主要建设内容包括：多数据集数据处理管线（以 MASA 为主基准，MM-GAG 提供多模态目标线索，SwissView 与 xBD 用于跨域测试、视进度选做）；环境与智能体之间的标准交互接口（episode 初始化、观测返回、动作执行、终止判定）；SR/SG 及探索效率指标的统计与轨迹可视化模块。该实验设置与文献[1-3]保持一致，保证实验结果直接可比。"));
body.push(h2("（三）多智能体主动定位基础框架（复现与奠基）"));
body.push(p([
  t("参照 GOMAA-Geo 的多智能体架构搭建零样本主动定位基础系统：感知定位智能体（VLM 判断当前观测与目标线索的地理关联，输出候选方向与置信度）、记忆管理智能体（维护与检索探索历史）、决策协调智能体（综合各方信息选择动作）；智能体间通过搜索协议与停止协议组织交互"),
  cite("10"),
  t("。整个系统不进行任何模型参数训练，决策能力完全来自预训练 VLM/LLM，设计工作集中于提示工程、交互协议与上下文组织。"),
]));
body.push(h2("（四）面向“少走回头路”的记忆机制设计（创新点一）"));
body.push(p("针对现有方法“视觉相似区域反复回访”的失败模式，设计显式的智能体记忆机制，包括：空间记忆——已访问网格的地图与视觉语义摘要，在决策上下文中显式抑制重复访问；经验记忆——跨任务的区域级负面经验（如“此类地貌区域已确认无目标”），支持新任务的快速排除；以及预算受限条件下记忆的紧凑表示与遗忘策略。目标是使智能体“记得来过、不再折返”。"));
body.push(h2("（五）面向“少走弯路”的层级搜索策略设计（创新点二）"));
body.push(p("针对现有方法“邻域漂移”失败模式与大尺度性能衰减，设计由粗到细的层级搜索策略：先依据目标线索与已有观测进行区域级候选圈定与排除，再对候选区域进行网格级细搜，替代单尺度的逐步轨迹搜索；设计目标邻域自适应确认机制——智能体进入候选邻域后切换为核实—收敛模式，抑制“到了又漂走”；并针对大网格场景研究分区探索与预算分配策略。"));
body.push(h2("（六）实验评测与消融分析"));
body.push(p("在统一平台上完成系统评测：基线包括 Random、贪心策略以及 GOMAA-Geo、GeoExplorer（优先采用论文报告值，关键设置自行复现）；消融实验分别剥离记忆模块与层级搜索模块，量化各创新点的贡献；分析不同网格尺度、初始距离与搜索预算下的性能变化，结合轨迹与失败案例可视化定位改进来源。"));

// 五、技术路线与总体方案
body.push(h1("五、技术路线与总体方案"));
body.push(p("系统总体技术路线如图 1 所示，整体为“环境平台 + 多智能体决策闭环”架构：目标线索经实验平台初始化为具体任务后，多智能体系统与网格化环境循环交互，直至停止协议触发，最终由评测模块统计指标并输出结果。"));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 120, after: 140 },
  children: [new ImageRun({
    type: "png", data: fs.readFileSync("figure_tech_route.png"),
    transformation: { width: 545, height: 379 },
    altText: { title: "图1", description: "系统总体技术路线图", name: "fig1" },
  })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 160, line: 300, lineRule: "auto" },
  children: [new TextRun({ text: "图 1  系统总体技术路线图", font: HEI, size: 21, bold: true, color: "000000" })],
}));
body.push(p("其中，单个任务的决策闭环流程如下："));
body.push(numItem("numFlow", "输入目标线索（航拍图、地面照片或文本描述），环境初始化任务（episode），智能体置于起点网格；"));
body.push(numItem("numFlow", "感知定位智能体获取当前网格航拍观测，结合目标线索进行地理关联推理；"));
body.push(numItem("numFlow", "记忆管理智能体更新空间记忆，并检索经验记忆中相关的区域级经验；"));
body.push(numItem("numFlow", "决策协调智能体综合当前推理、记忆状态与剩余预算，确定当前搜索层级（区域级或网格级）与下一步动作；"));
body.push(numItem("numFlow", "环境执行动作并返回新观测；"));
body.push(numItem("numFlow", "重复步骤 2 至 5，直至停止协议触发（自认抵达目标或预算耗尽）；"));
body.push(numItem("numFlow", "平台统计 SR、SG、步数、重复访问率等指标并记录轨迹；"));
body.push(numItem("numFlow", "任务结束后将区域级探索结论写入经验记忆，供后续任务复用。"));
body.push(p([
  t("关键实现选型如下（具体型号在平台搭建阶段结合算力与 API 预算同指导教师确认后锁定）：编程语言采用 Python；作为决策核心的视觉语言模型基于 Transformer 架构"),
  cite("14"),
  t("，在商用 API（如 GPT-4o、GLM-4V 等）与开源模型本地部署（如 Qwen-VL 系列）之间选型；特征编码器采用冻结的 CLIP"),
  cite("9"),
  t("等预训练模型；GeoExplorer 所采用的好奇心机制"),
  cite("11"),
  t("仅在对比实验中涉及。系统不涉及任何模型参数训练，主要资源开销为 VLM 推理调用。"),
]));

// 六、可行性分析
body.push(h1("六、可行性分析"));
body.push(h2("（一）技术可行性"));
body.push(p("本课题采用零训练的 Agent 路线，不进行任何模型参数训练，对算力要求低，主要资源开销为 VLM 推理调用（商用 API 或本地开源模型均可）；网格化实验平台基于 Python 实现，技术成熟；参照系统 GOMAA-Geo 已开源（github.com/mvrl/GOMAA-Geo），其智能体架构与评测协议可直接参考，显著降低实现风险。"));
body.push(h2("（二）数据可行性"));
body.push(p("实验全部基于公开数据集（MASA、MM-GAG、SwissView、xBD），已与指导教师确认无需自行采集构建数据集；上述数据集与三篇核心论文的评测协议一致，实验结果可与已有工作直接对比。"));
body.push(h2("（三）团队与时间可行性"));
body.push(p("小组三人分工分别覆盖实验平台、记忆机制、搜索策略三个核心模块，与研究内容一一对应；18 周进度按“平台—框架—创新—评测—报告”推进并设置六个里程碑，中期检查前完成基础框架，后续留有充分的迭代与缓冲空间。"));
body.push(h2("（四）前期工作基础"));
body.push(p("小组已完成三篇核心论文（GOMAA-Geo、GeoExplorer、DynCur-Geo）的精读，梳理出领域统一的实验协议（网格化环境、SR/SG 指标）与现有方法的两大失败模式；已与指导教师讨论确定“多智能体 + 记忆与搜索策略创新”的技术路线；已完成实验平台的方案设计与技术选型调研，具备立即进入平台搭建阶段的条件。"));

// 七、创新点
body.push(h1("七、创新点"));
body.push(numItem("numInnov", "面向“少走回头路”的智能体记忆机制：将空间记忆与跨任务经验记忆引入主动地理定位的多智能体框架，显式建模“已访问/已排除”信息以抑制重复访问，直接针对现有方法“视觉相似区域来回震荡”的失败模式；"));
body.push(numItem("numInnov", "面向“少走弯路”的层级化搜索策略：设计“区域级圈定—网格级细搜”的由粗到细搜索范式与目标邻域自适应确认机制，直接针对“邻域漂移”失败模式，并重点提升大尺度长距离任务性能；"));
body.push(numItem("numInnov", "统一可比的轻量实验平台：基于公开数据集与领域统一的 SR/SG 评测协议构建网格化交互实验平台，为零样本 Agent 路线与训练路线提供公平对比基础，兼具可复现性与可扩展性。"));

// 八、预期成果与考核指标
body.push(h1("八、预期成果与考核指标"));
body.push(p("预期成果包括："));
body.push(numItem("numOutcomes", "主动地理定位网格化交互实验平台一套（含数据管线、评测与可视化）；"));
body.push(numItem("numOutcomes", "基于多智能体的主动定位方法系统一套（含记忆与层级搜索模块）；"));
body.push(numItem("numOutcomes", "完整的对比实验与消融实验分析；"));
body.push(numItem("numOutcomes", "综合工程设计报告及全部源代码。"));
body.push(caption("表 2  预期考核指标"));
body.push(mkTable(
  [1900, 3700, 2706],
  ["指标", "定义", "预期目标"],
  [
    [{ text: "SR（成功率）" }, { text: "搜索预算内成功抵达目标网格的任务占比" }, { text: "MASA 标准设置下与 GOMAA-Geo 基线相当" }],
    [{ text: "SG（终止网格距离）" }, { text: "任务结束时智能体与目标网格的曼哈顿距离" }, { text: "不高于 GOMAA-Geo 基线" }],
    [{ text: "大网格成功率" }, { text: "10×10 及以上网格设置下的 SR" }, { text: "优于最优基线，争取提升 5 个百分点以上" }],
    [{ text: "重复访问率" }, { text: "重复访问已探索网格的步数占总步数比例" }, { text: "较无记忆消融版本下降 30% 以上（争取）" }],
  ]
));
body.push(new Paragraph({ spacing: { line: 240, lineRule: "auto" }, children: [] }));
body.push(p([
  t("SR 与 SG 为主动地理定位领域统一采用的评价指标"),
  cite("3"),
  t("，保证与已有工作直接可比；平均步数、VLM 调用次数等效率指标作为辅助分析项。上表标注“争取”的量化目标以机制验证为核心，若大模型调用预算等客观条件受限，将以消融实验与机制分析作为等效验收依据。"),
]));
body.push(p("主要风险与对策：VLM 调用成本受限——通过控制上下文长度、优先在中等尺度网格开展实验、必要时切换本地开源模型应对；基线复现工作量大——优先采用论文报告值，仅对关键设置自行复现；大模型输出稳定性不足——采用结构化输出约束与多次采样表决。"));

// 九、进度安排
body.push(h1("九、进度安排"));
body.push(p("本课题按教学日历共 18 周（2026 年 9 月 14 日至 2027 年 1 月 17 日），分六个阶段推进并设置六个里程碑（M1–M6），详见表 3。"));
body.push(caption("表 3  进度安排（2026.9.14–2027.1.17，共 18 周）"));
body.push(mkTable(
  [1750, 750, 1250, 3100, 1456],
  ["阶段", "周次", "时间", "主要任务", "里程碑"],
  [
    [{ text: "一、文献调研与选题论证" }, { text: "1–3", opts: { align: AlignmentType.CENTER } }, { text: "9.14–10.4", opts: { align: AlignmentType.CENTER } }, { text: "精读三篇核心论文；确定技术路线与分工；完成选题报告" }, { text: "M1：选题报告通过" }],
    [{ text: "二、实验平台搭建" }, { text: "4–6", opts: { align: AlignmentType.CENTER } }, { text: "10.5–10.25", opts: { align: AlignmentType.CENTER } }, { text: "数据处理管线；网格环境与交互接口；SR/SG 评测与轨迹可视化；Random 与贪心基线" }, { text: "M2：平台 v1.0，基线跑通" }],
    [{ text: "三、多智能体基础框架" }, { text: "7–9", opts: { align: AlignmentType.CENTER } }, { text: "10.26–11.15", opts: { align: AlignmentType.CENTER } }, { text: "VLM 接入；感知、记忆、协调三类智能体与交互协议；基础系统在 MASA 上出实验结果" }, { text: "M3：中期检查通过" }],
    [{ text: "四、记忆与搜索策略创新" }, { text: "10–13", opts: { align: AlignmentType.CENTER } }, { text: "11.16–12.13", opts: { align: AlignmentType.CENTER } }, { text: "空间与经验记忆模块；层级搜索与邻域确认策略；消融实验" }, { text: "M4：创新模块验证有效" }],
    [{ text: "五、系统评测与分析" }, { text: "14–16", opts: { align: AlignmentType.CENTER } }, { text: "12.14–1.3", opts: { align: AlignmentType.CENTER } }, { text: "多数据集对比实验；大尺度压力测试；失败案例分析" }, { text: "M5：全部实验完成" }],
    [{ text: "六、报告撰写与答辩" }, { text: "17–18", opts: { align: AlignmentType.CENTER } }, { text: "1.4–1.17", opts: { align: AlignmentType.CENTER } }, { text: "课程设计报告撰写；代码整理与文档；答辩准备" }, { text: "M6：提交与答辩" }],
  ]
));
body.push(new Paragraph({ spacing: { line: 240, lineRule: "auto" }, children: [] }));

// 十、小组分工
body.push(h1("十、小组分工"));
body.push(caption("表 4  小组分工（成员信息待填）"));
body.push(mkTable(
  [1950, 2000, 4356],
  ["成员", "角色", "主要职责"],
  [
    [{ text: "＿＿＿＿（组长）" }, { text: "实验平台与评测" }, { text: "环境与数据管线搭建；基线复现；指标统计与轨迹可视化" }],
    [{ text: "＿＿＿＿" }, { text: "多智能体系统与记忆机制" }, { text: "VLM 接入与交互协议设计；空间与经验记忆模块实现" }],
    [{ text: "＿＿＿＿" }, { text: "搜索策略与实验分析" }, { text: "层级搜索与邻域确认策略；消融与大尺度实验；报告统筹" }],
  ]
));
body.push(new Paragraph({ spacing: { line: 240, lineRule: "auto" }, children: [] }));
body.push(p("三人共同参与文献调研、方案讨论与最终报告撰写；每周组会同步进展，分工可根据中期检查结果动态调整。"));

// 十一、参考文献
body.push(h1("十一、参考文献"));
const refs = [
  "[1] Sarkar A, Sastry S, Pirinen A, et al. GOMAA-Geo: GOal Modality Agnostic Active Geo-localization[C]//Advances in Neural Information Processing Systems (NeurIPS), 2024.",
  "[2] Mi L, Bechaz M, Chen Z, et al. GeoExplorer: Active Geo-localization with Curiosity-Driven Exploration[C]//IEEE/CVF International Conference on Computer Vision (ICCV), 2025.",
  "[3] Sun Y, Zhang Y, Zhu P. DynCur-Geo: Dynamic Curiosity Reward Shaping for Multimodal Active Geo-Localization[J]. arXiv preprint arXiv:2608.18673, 2026.",
  "[4] Workman S, Souvenir R, Jacobs N. Wide-Area Image Geolocalization with Aerial Reference Imagery[C]//IEEE International Conference on Computer Vision (ICCV), 2015.",
  "[5] Vo N, Jacobs N, Hays J. Revisiting IM2GPS in the Deep Learning Era[C]//IEEE International Conference on Computer Vision (ICCV), 2017.",
  "[6] Zhai M, Doersch C, Zhou Y, et al. Predicting Ground-Level Scene Layout from Aerial Imagery[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.",
  "[7] Zheng Z, Wei Y, Yang Y. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization[C]//ACM International Conference on Multimedia (ACM MM), 2020.",
  "[8] Zhu S, Yang T, Chen C, et al. VIGOR: Cross-View Image Geo-localization beyond One-to-one Retrieval[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.",
  "[9] Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models from Natural Language Supervision[C]//International Conference on Machine Learning (ICML), 2021.",
  "[10] Yao S, Zhao J, Yu D, et al. ReAct: Synergizing Reasoning and Acting in Language Models[C]//International Conference on Learning Representations (ICLR), 2023.",
  "[11] Pathak D, Agrawal P, Efros A A, et al. Curiosity-driven Exploration by Self-Supervised Prediction[C]//International Conference on Machine Learning (ICML), 2017.",
  "[12] Gupta R, Hosfelt R, Sajeev S, et al. Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery[C]//IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019.",
  "[13] Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.",
  "[14] Vaswani A, Shazeer N, Parmar N, et al. Attention is All You Need[C]//Advances in Neural Information Processing Systems (NeurIPS), 2017.",
];
refs.forEach(r => body.push(refP(r)));

// ---------- 组装文档 ----------
const numberingLevel = {
  level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 480, hanging: 480 } } },
};

const doc = new Document({
  creator: "大数据智能方向综合工程设计小组",
  title: "综合工程设计选题报告——基于主动探索的视觉地理定位方法设计与实现",
  styles: {
    default: {
      document: { run: { font: CN, size: 24 } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEI, size: 30, bold: true, color: "000000" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEI, size: 28, bold: true, color: "000000" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "numTargets", levels: [numberingLevel] },
      { reference: "numKp", levels: [numberingLevel] },
      { reference: "numFlow", levels: [numberingLevel] },
      { reference: "numInnov", levels: [numberingLevel] },
      { reference: "numOutcomes", levels: [numberingLevel] },
    ],
  },
  sections: [{
    properties: {
      titlePage: true,
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, bottom: 1440, left: 1800, right: 1800 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: ["第 ", PageNumber.CURRENT, " 页"], font: CN, size: 18 })],
        })],
      }),
      first: new Footer({ children: [] }),
    },
    children: [...cover, ...body],
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = "选题报告_基于主动探索的视觉地理定位方法设计与实现.docx";
  fs.writeFileSync(out, buf);
  console.log("OK ->", out, buf.length, "bytes");
});
