"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  Packer,
  PageBreak,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableLayoutType,
  TableOfContents,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..", "..");
const CONTENT_PATH = path.join(ROOT, "project", "src", "documents", "final_report_content_v5.json");
const CLAIM_V1_PATH = path.join(ROOT, "总结报告相关", "总结报告_v1", "准备", "论断与证据.json");
const CLAIM_V2_PATH = path.join(ROOT, "总结报告相关", "总结报告_v1", "准备", "增量证据绑定_v2.json");
const LOGO_PATH = path.join(ROOT, "归档", "总结报告相关", "总结报告_v1", "核验", "初稿_v1", "教师封面标志.png");

const OUT_DOCX = path.join(
  ROOT,
  "总结报告相关",
  "总结报告_v1",
  "稿件",
  "总结报告_基于主动探索的视觉地理定位方法设计与实现_终稿_v5.docx"
);
const OUT_MD = path.join(ROOT, "总结报告相关", "总结报告_v1", "稿件", "正文文字稿_v5.md");
const OUT_RECEIPT = path.join(ROOT, "总结报告相关", "总结报告_v1", "核验", "终稿_v5", "生成回执_v5.json");

const PAGE_W = 11906;
const PAGE_H = 16838;
const MARGINS = {
  top: 1440,
  right: 1134,
  bottom: 1440,
  left: 1985,
  header: 851,
  footer: 992,
  gutter: 170,
};
const CONTENT_W = PAGE_W - MARGINS.left - MARGINS.right - MARGINS.gutter; // 8617 dxa

function sha256File(p) {
  return crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
}

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8").replace(/^\uFEFF/, ""));
}

function makeFont(eastAsia = "宋体", ascii = "Times New Roman") {
  return { ascii, hAnsi: ascii, cs: ascii, eastAsia };
}

function makeRun(text, opts = {}) {
  return new TextRun({
    text: String(text),
    font: makeFont(opts.eastAsia || "宋体", opts.ascii || "Times New Roman"),
    size: opts.size || 24,
    bold: Boolean(opts.bold),
    italics: Boolean(opts.italics),
    color: opts.color || "111111",
  });
}

function findFact(claimsMap, claimId, source, pointer) {
  const claim = claimsMap[claimId];
  if (!claim) throw new Error(`Missing claim ${claimId}`);
  const hit = claim.facts.find((f) => f.source === source && f.pointer === pointer);
  if (!hit) throw new Error(`Missing fact ${claimId} -> ${source}:${pointer}`);
  return hit.value;
}

function pctStr(val, digits = 2) {
  return `${(Number(val) * 100).toFixed(digits)}%`;
}

function numStr(val, digits = 2) {
  return Number(val).toFixed(digits);
}

function buildTablesData(v1, v2) {
  const c1 = Object.fromEntries(v1.claims.map((c) => [c.id, c]));
  const c2 = Object.fromEntries(v2.claims.map((c) => [c.id, c]));

  const commonMetrics = findFact(c1, "C11", "common", "/aggregate_metrics");
  const mMap = {};
  for (const row of commonMetrics) mMap[`${row.method}_${row.condition}`] = row;

  const s2Sr = findFact(c1, "C07", "s2", "/averages/Edge/CueFull/sr_mean");
  const s2Sg = findFact(c1, "C07", "s2", "/averages/Edge/CueFull/sg_mean");
  const s3Sr = findFact(c1, "C08", "s3", "/averages/Edge/CueFull/sr_mean");
  const s3Sg = findFact(c1, "C08", "s3", "/averages/Edge/CueFull/sg_mean");
  const s4Sr = findFact(c1, "C09", "s4", "/averages/Edge/CueFull/sr_mean");
  const s4Sg = findFact(c1, "C09", "s4", "/averages/Edge/CueFull/sg_mean");
  const swSr = findFact(c1, "C10", "swiss", "/averages/Edge/CueFull/sr_mean");
  const swSg = findFact(c1, "C10", "swiss", "/averages/Edge/CueFull/sg_mean");

  const roadsRef = findFact(c1, "C16", "roads", "/reference/metrics");
  const roadsCand = findFact(c1, "C16", "roads", "/candidate/metrics");
  const mg10Base = findFact(c1, "C17", "massgis", "/10/comparison/baseline");
  const mg10Cand = findFact(c1, "C17", "massgis", "/10/comparison/candidate");
  const mg15Base = findFact(c1, "C18", "massgis", "/15/comparison/baseline");
  const mg15Cand = findFact(c1, "C18", "massgis", "/15/comparison/candidate");

  const T21 = {
    caption: "表2-1　主动探索网格环境、预算约束与局部观测字段边界规范",
    colWidths: [1650, 2900, 4067],
    headers: ["信息类别", "策略/控制器可读输入（公开契约）", "评测器封闭保管字段（严禁泄漏）"],
    rows: [
      ["空间状态", "当前网格公开坐标 (rₜ,cₜ)、网格尺度 k 与历史访问掩码", "目标真实坐标 (r*,c*)、当前曼哈顿距离与分层难度标签"],
      ["视觉观测", "当前所在单块观测图像/冻结特征与用户给定目标图像/特征", "未访问网格影像像素、全图特征矩阵与全局参考拼图"],
      ["动作与预算", "基础动作{上,右,下,左}、越界原地耗步与剩余可用步数", "跨回合记忆状态（每题重置）与测试队列源文件分组标识"],
      ["UI 与日志", "浏览器全图可视化切换（仅影响 Canvas 渲染透明度）", "终局真实距离 SG 与训练阶段 PBRS 距离塑形信号 Φ(s)"],
    ],
    sourceRows: [],
  };

  const T31 = {
    caption: "表3-1　工程系统五层模块实现架构与双形态工作台技术栈对照",
    colWidths: [1350, 3150, 4117],
    headers: ["系统层次", "核心模块与对应代码位置", "关键运行契约与技术形态"],
    rows: [
      ["数据与环境层", "environment.py / episode.py / opencvl_wild_intake_v1.py", "隔离真值访问；冻结 Masa、SwissView、Roads/MassGIS 与 OpenCVL SE 划分"],
      ["视觉编码与头", "frozen_edge_navigator.py / edge_cue.py / decoupled_cue.py", "冻结骨干；Small256(797,701)+EdgeCue(136,837)=934,538 增量参数；0.50 门控"],
      ["覆盖与 API 网关", "parameterized_coverage.py / service_v17.py", "3 步前瞻（净增益>=2）+径向保护；OpenAI/Gemini/DeepSeek 思考开关与 3 次失败熔断"],
      ["GeoNav V5.0", "workbench_server_v28.py / geonav_v5.{html,js} (8769)", "纳管 192 航拍 960 题 + 192 地面题；支持 n×n 纯内存即时切图与 6 项 E2E 测试"],
      ["GeoNav Lite 与包", "lite_server.py / geonav_lite.{html,js} (8768) / demo/*.zip", "极简答辩聚焦界面；双便携包 360.86 MiB 与 292.01 MiB + 6 项独立回归测试"],
    ],
  };

  const T41 = {
    caption: "表4-1　全阶段评测队列、数据源划分与阶段性统计边界总览",
    colWidths: [1950, 1750, 1550, 3367],
    headers: ["评测协议/任务集", "数据源与独立性", "网格/预算/样本", "核定用途与统计解释边界"],
    rows: [
      ["Masa S2 稳定复验", "4 张独立验证源", "5×5 / 10步 / 300运行", "中期主线稳定复验；同源内任务存在空间关联"],
      ["Masa S3 独立源确认", "10 张测试源文件", "5×5 / 10步 / 750运行", "主确认队列与四臂目标对照；文件隔离非全地理隔离"],
      ["Masa S4 / SwissView", "20消费源 / 20航拍源", "10×10(1500) / 5×5(1500)", "验证切图密化与同模态跨集迁移；均不涉及地面视角"],
      ["MasaRoads / MassGIS", "10完整隔离区(300m/格)", "10×10(2250) / 15×15(3000)", "真实9/20.25 km²扩展；三权重共地图组，置信区间按区重采样"],
      ["V4.18 单API / Lite", "10题开发 / 6源192+192题", "10题×3臂 / 960+192题", "验证云端VLM API(96请求)成本退化及V5/Lite平台规则地板"],
      ["OpenCVL SE 阶段一二", "SE 942→435/116/28对", "25格单步(551开发对)", "dev_val距fit_train>=95.203 km；28盲测>=109.465 km且0评测"],
    ],
  };

  const T42 = {
    caption: "表4-2　四大航拍核心协议与 S3 同题四臂目标干预实证对比",
    colWidths: [2350, 1400, 1200, 1150, 2517],
    headers: ["协议 / 策略与条件", "成功 / 计划", "SR (%)", "SG (格)", "相对基线/对照关键变化"],
    rows: [
      ["Masa S2 · M0 (Full)", "268 / 300", pctStr(s2Sr), numStr(s2Sg, 3), "各权重 [92, 90, 86] / 100；最终验收通过"],
      ["Masa S4 密度扩展 · M0", "1157 / 1500", pctStr(s4Sr), numStr(s4Sg, 3), "各权重 [385, 407, 365] / 500；通过验收"],
      ["SwissView 迁移 · M0", "1302 / 1500", pctStr(swSr), numStr(swSg, 3), "各权重 [430, 434, 438] / 500；通过验收"],
      ["Masa S3 · M0 Full（真目标）", `${mMap.M0_Full.successes} / ${mMap.M0_Full.planned}`, pctStr(mMap.M0_Full.sr), numStr(mMap.M0_Full.sg_cells_all_planned, 3), "各权重 [216, 223, 212] / 250；基准主力"],
      ["Masa S3 · M0 Disabled（禁用）", `${mMap.M0_Disabled.successes} / ${mMap.M0_Disabled.planned}`, pctStr(mMap.M0_Disabled.sr), numStr(mMap.M0_Disabled.sg_cells_all_planned, 3), "相对Full低13.20pp（恢复100，损伤1）"],
      ["Masa S3 · M0 Mean（均值替代）", `${mMap.M0_Mean.successes} / ${mMap.M0_Mean.planned}`, pctStr(mMap.M0_Mean.sr), numStr(mMap.M0_Mean.sg_cells_all_planned, 3), "相对Full低13.60pp（恢复103，损伤1）"],
      ["Masa S3 · M0 Wrong（错误目标）", `${mMap.M0_Wrong.successes} / ${mMap.M0_Wrong.planned}`, pctStr(mMap.M0_Wrong.sr), numStr(mMap.M0_Wrong.sg_cells_all_planned, 3), "相对Full低17.33pp（恢复132，损伤2）"],
      ["Masa S3 · GeoExplorer 有限复现", `${mMap.GeoExplorer_Full.successes} / ${mMap.GeoExplorer_Full.planned}`, pctStr(mMap.GeoExplorer_Full.sr), numStr(mMap.GeoExplorer_Full.sg_cells_all_planned, 3), "Disabled 35.33% / Wrong 36.80%；未过目标门槛"],
    ],
  };

  const T43 = {
    caption: "表4-3　大跨度连续区域独立核验与 MassGIS 新域双网格退化表现（每格300米，预算20步）",
    colWidths: [2550, 1400, 1150, 1300, 2217],
    headers: ["区域协议与对比策略", "成功 / 计划", "SR (%)", "SG (米)", "最终验收与关键诊断注释"],
    rows: [
      ["Roads 9 km² · M0 基线", `${roadsRef.successes} / ${roadsRef.episodes}`, pctStr(roadsRef.sr), numStr(roadsRef.mean_sg_m, 2), "有效移动 5014.13 米；对比基准"],
      ["Roads 9 km² · Coverage3Radial", `${roadsCand.successes} / ${roadsCand.episodes}`, pctStr(roadsCand.sr), numStr(roadsCand.mean_sg_m, 2), "净增+207(+9.20pp)，恢复295/损伤88；通过"],
      ["MassGIS 10×10 · M0 基线", `${mg10Base.successes} / ${mg10Base.planned}`, pctStr(mg10Base.SR), numStr(mg10Base.SG_m, 2), "有效移动 5111.20 米"],
      ["MassGIS 10×10 · Coverage3Radial", `${mg10Cand.successes} / ${mg10Cand.planned}`, pctStr(mg10Cand.SR), numStr(mg10Cand.SG_m, 2), "仅微增3.47pp，未通过新域准入门槛"],
      ["MassGIS 15×15 · M0 基线", `${mg15Base.successes} / ${mg15Base.planned}`, pctStr(mg15Base.SR), numStr(mg15Base.SG_m, 2), "同750物理子集跌至 21.24%"],
      ["MassGIS 15×15 · Coverage3Radial", `${mg15Cand.successes} / ${mg15Cand.planned}`, pctStr(mg15Cand.SR), numStr(mg15Cand.SG_m, 2), "同750子集 22.40%；未通过（列偏置+5漂移）"],
    ],
  };

  const apiM0Sr = findFact(c2, "C24", "v418_api_result", "/arms/M0/SR");
  const apiM0Sg = findFact(c2, "C24", "v418_api_result", "/arms/M0/SG_cells");
  const apiFrSr = findFact(c2, "C24", "v418_api_result", "/arms/frontier/SR");
  const apiFrSg = findFact(c2, "C24", "v418_api_result", "/arms/frontier/SG_cells");
  const apiLmSr = findFact(c2, "C24", "v418_api_result", "/arms/api_m0_fallback/SR");
  const apiLmSg = findFact(c2, "C24", "v418_api_result", "/arms/api_m0_fallback/SG_cells");
  const apiReq = findFact(c2, "C24", "v418_api_result", "/usage/requests");
  const apiTok = findFact(c2, "C24", "v418_api_result", "/usage/usage/total_tokens");
  const apiDam = findFact(c2, "C24", "v418_api_result", "/paired/M0/damaged");

  const gFrSr = findFact(c2, "C25", "multidim_1108", "/capturepoint_192_rule_verification/0/sr");
  const gFrSg = findFact(c2, "C25", "multidim_1108", "/capturepoint_192_rule_verification/0/sg_mean");
  const gCcSr = findFact(c2, "C25", "multidim_1108", "/capturepoint_192_rule_verification/1/sr");
  const gLmSr = findFact(c2, "C25", "multidim_1108", "/capturepoint_192_rule_verification/2/sr");
  const gRwSr = findFact(c2, "C25", "multidim_1108", "/capturepoint_192_rule_verification/3/sr");

  const T44 = {
    caption: "表4-4　V4.18单模型固定十题云端API对照及GeoNav双形态地面规则测试汇总",
    colWidths: [2550, 1300, 1200, 1300, 2267],
    headers: ["测试队列与执行臂", "成功 / 计划", "SR (%)", "SG (格)", "资源开销、配对与验证结论"],
    rows: [
      ["V4.18 十题 · 本地 M0", "8 / 10", pctStr(apiM0Sr), numStr(apiM0Sg, 2), "重访 4 次、越界 0 次；无网络费用延迟"],
      ["V4.18 十题 · 规则 Frontier", "4 / 10", pctStr(apiFrSr), numStr(apiFrSg, 2), "确定性启发对比参考臂"],
      ["V4.18 十题 · 云端 Gemma-4-31B", "1 / 10", pctStr(apiLmSr), numStr(apiLmSg, 2), `${apiReq}请求/0回退/${apiTok} tokens；重访54次、相对M0损伤${apiDam}题`],
      ["地面192题 · Frontier 规则", "138 / 192", pctStr(gFrSr), numStr(gFrSg, 3), "GeoNav V5/Lite 地面演示题库规则上界参考"],
      ["地面192题 · CenterCross", "126 / 192", pctStr(gCcSr), "1.349", "中心十字优先探索对照"],
      ["地面192题 · Lawnmower / Random", "101 / 44 (192)", `${pctStr(gLmSr)} / ${pctStr(gRwSr)}`, "1.771 / 2.917", "双工作台均获 6+6=12 项 E2E 全通过及双压缩交付"],
    ],
  };

  const fitM2cTop1 = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_fit_train/M2c_MultiScale_PriorCalibrated_CV051b/top1_acc");
  const fitM2cErr = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_fit_train/M2c_MultiScale_PriorCalibrated_CV051b/expected_planar_error_m");
  const valAlwTop1 = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/AlwaysCell12Prior/top1_acc");
  const valAlwErr = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/AlwaysCell12Prior/expected_planar_error_m");
  const valEmpErr = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/TrainEmpiricalPrior/expected_planar_error_m");
  const valEmpKl = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/TrainEmpiricalPrior/soft_target_kl_nats");
  const valM1Top1 = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M1_ZeroShot_FullTile_Cosine/top1_acc");
  const valM1Err = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M1_ZeroShot_FullTile_Cosine/expected_planar_error_m");
  const valM1bTop1 = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M1b_ZeroShot_Center9Crop_Cosine/top1_acc");
  const valM1bErr = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M1b_ZeroShot_Center9Crop_Cosine/expected_planar_error_m");
  const valM2aTop1 = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M2a_SpatialHead_PureVisual_NoPrior/top1_acc");
  const valM2aErr = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M2a_SpatialHead_PureVisual_NoPrior/expected_planar_error_m");
  const valM2bTop1 = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M2b_SpatialHead_PriorCalibrated_Full/top1_acc");
  const valM2bErr = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M2b_SpatialHead_PriorCalibrated_Full/expected_planar_error_m");
  const valM2bKl = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M2b_SpatialHead_PriorCalibrated_Full/soft_target_kl_nats");
  const valM2cTop1 = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M2c_MultiScale_PriorCalibrated_CV051b/top1_acc");
  const valM2cTop5 = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M2c_MultiScale_PriorCalibrated_CV051b/top5_acc");
  const valM2cErr = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M2c_MultiScale_PriorCalibrated_CV051b/expected_planar_error_m");
  const valM2cKl = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/M2c_MultiScale_PriorCalibrated_CV051b/soft_target_kl_nats");
  const valGz = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/Control_GroundZeroed/top1_acc");
  const valAs = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/Control_AerialShuffled/top1_acc");
  const valMv = findFact(c2, "C28", "cv051_verdict", "/summary_comparison_all_pairs/development_validation/Control_MeanVisual/top1_acc");

  const T45 = {
    caption: "表4-5　跨视角早期探索负结果（CV-019~041）与 OpenCVL 阶段二单步定位消融实证汇总",
    colWidths: [2650, 1450, 1200, 1200, 2117],
    headers: ["实验任务 / 阶段二评估模型", "测试集与样本规模", "核心精度(%)", "误差 / KL", "门禁判定与机理归因"],
    rows: [
      ["CV-024 / CV-019 (MM-GAG)", "20题 / 125窗", "20.00 / 0.00", "—", "Full 4/20 不及 Wrong 9/20；排序Top1=0；失败"],
      ["CV-031 / CV-034 (CCVPE)", "筛查 / 448双画布", "正例<=5m: 0", "宏BA 68.50%", "累计画布 59.22%，无有效5米定位；失败"],
      ["CV-041 (三城RGB轻量头)", "2850开发(正例9)", "正例联合 0/9", "宏联合 66.67%", "直接跨域回归彻底失效；依停损规则停止"],
      ["OpenCVL · AlwaysCell12 / Empirical", "dev_val (116对)", `${pctStr(valAlwTop1)} / ${pctStr(valAlwTop1)}`, `${numStr(valAlwErr)}m / KL ${numStr(valEmpKl, 3)}`, `经验误差 ${numStr(valEmpErr)}m；零图像纯先验统计地板`],
      ["OpenCVL · M1 / M1b 零样本余弦", "dev_val (116对)", `${pctStr(valM1Top1)} / ${pctStr(valM1bTop1)}`, `${numStr(valM1Err)}m / ${numStr(valM1bErr)}m`, "冻结全图与9格裁切零样本近乎失效"],
      ["OpenCVL · M2a (纯视觉无先验头)", "dev_val (116对)", pctStr(valM2aTop1), `${numStr(valM2aErr)}m / KL 3.292`, "脱离中心格先验后视觉头近乎盲猜"],
      ["OpenCVL · M2b (单尺度先验校准)", "dev_val (116对)", pctStr(valM2bTop1), `${numStr(valM2bErr)}m / KL ${numStr(valM2bKl, 3)}`, "Top-5 79.31%，仍大幅落后中心统计地板"],
      ["OpenCVL · M2c (城内fit_train参考)", "fit_train (435对)", pctStr(fitM2cTop1), `${numStr(fitM2cErr)}m / KL 0.810`, "仅反映同城过拟合与4:3偏置表观高分"],
      ["OpenCVL · M2c (跨城dev_val主模型)", "dev_val (>=95km)", pctStr(valM2cTop1), `${numStr(valM2cErr)}m / KL ${numStr(valM2cKl, 3)}`, `Top-5 ${pctStr(valM2cTop5)}；952×714 42.86% vs ×532 21.57%`],
      ["OpenCVL · GZ / AS / MeanVisual 消融", "dev_val (116对)", `${pctStr(valGz)}/${pctStr(valAs)}/${pctStr(valMv)}`, "Mean KL 1.524", "常量视觉反超M2c；G1–G4全否，28盲测0触碰"],
    ],
  };

  return { T21, T31, T41, T42, T43, T44, T45 };
}

function buildTable(spec) {
  const borderThin = { style: BorderStyle.SINGLE, size: 4, color: "CBD5E1" };
  const borderTopThick = { style: BorderStyle.SINGLE, size: 12, color: "1E293B" };
  const borderHeadBottom = { style: BorderStyle.SINGLE, size: 8, color: "334155" };
  const borderBottomThick = { style: BorderStyle.SINGLE, size: 12, color: "1E293B" };
  const borderNone = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

  const headerRow = new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: spec.headers.map((h, idx) =>
      new TableCell({
        width: { size: spec.colWidths[idx], type: WidthType.DXA },
        verticalAlign: VerticalAlign.CENTER,
        shading: { type: ShadingType.CLEAR, fill: "EEF2F6" },
        margins: { top: 65, bottom: 65, left: 80, right: 80 },
        borders: { top: borderTopThick, bottom: borderHeadBottom, left: borderNone, right: borderNone },
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            keepNext: true,
            spacing: { before: 0, after: 0, line: 260 },
            children: [makeRun(h, { size: 19, bold: true, eastAsia: "黑体", color: "0F172A" })],
          }),
        ],
      })
    ),
  });

  const bodyRows = spec.rows.map((r, rIdx) => {
    const isLast = rIdx === spec.rows.length - 1;
    const fill = rIdx % 2 === 1 ? "F8FAFC" : "FFFFFF";
    return new TableRow({
      cantSplit: true,
      children: r.map((cellText, cIdx) =>
        new TableCell({
          width: { size: spec.colWidths[cIdx], type: WidthType.DXA },
          verticalAlign: VerticalAlign.CENTER,
          shading: { type: ShadingType.CLEAR, fill },
          margins: { top: 55, bottom: 55, left: 75, right: 75 },
          borders: {
            top: borderNone,
            bottom: isLast ? borderBottomThick : borderThin,
            left: borderNone,
            right: borderNone,
          },
          children: [
            new Paragraph({
              alignment: cIdx === 1 || cIdx === 2 ? AlignmentType.CENTER : AlignmentType.LEFT,
              keepNext: !isLast && rIdx === 0,
              spacing: { before: 0, after: 0, line: 250 },
              children: [makeRun(cellText, { size: 18, color: "1E293B" })],
            }),
          ],
        })
      ),
    });
  });

  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 140, after: 60, line: 280 },
      children: [makeRun(spec.caption, { size: 21, bold: true, eastAsia: "黑体", color: "0F172A" })],
    }),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      layout: TableLayoutType.FIXED,
      columnWidths: spec.colWidths,
      rows: [headerRow, ...bodyRows],
    }),
    new Paragraph({
      spacing: { before: 0, after: 80, line: 240 },
      children: [makeRun("", { size: 16 })],
    }),
  ];
}

function buildFigure(figMeta) {
  const imgPath = path.join(ROOT, figMeta.path);
  const targetW = 535;
  const targetH = Math.round((targetW * figMeta.height) / figMeta.width);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 120, after: 50, line: 240 },
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(imgPath),
          transformation: { width: targetW, height: targetH },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 30, after: 120, line: 280 },
      children: [
        makeRun(`${figMeta.number}　${figMeta.title}`, { size: 21, bold: true, eastAsia: "黑体", color: "0F172A" }),
        makeRun(`（${figMeta.caption_boundary}）`, { size: 19, color: "475569" }),
      ],
    }),
  ];
}

function bodyParagraph(text, isEnglish = false) {
  return new Paragraph({
    alignment: AlignmentType.BOTH,
    indent: { firstLine: 480 },
    spacing: { before: 0, after: 65, line: 360 },
    children: [
      makeRun(text, {
        size: 24,
        eastAsia: isEnglish ? "Times New Roman" : "宋体",
        ascii: "Times New Roman",
      }),
    ],
  });
}

function equationParagraph(eq) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 90, after: 90, line: 340 },
    children: [
      makeRun(eq.text, { size: 22, ascii: "Cambria Math", eastAsia: "宋体", color: "0F172A" }),
      makeRun(`　　${eq.num}`, { size: 22, bold: true, color: "334155" }),
    ],
  });
}

function buildMarkdownText(content, tables, figuresById) {
  const lines = [];
  lines.push(`# ${content.cover.title}`);
  lines.push("");
  lines.push(`${content.cover.report_type} · ${content.cover.stage_label}；${content.cover.evidence_cutoff}`);
  lines.push("");
  lines.push("## 中文摘要");
  lines.push("");
  for (const p of content.abstract_cn.paragraphs) lines.push(p, "");
  lines.push(`关键词：${content.abstract_cn.keywords}`, "");
  lines.push("## Abstract", "");
  for (const p of content.abstract_en.paragraphs) lines.push(p, "");
  lines.push(`Keywords: ${content.abstract_en.keywords}`, "");

  for (const ch of content.chapters) {
    lines.push(`## ${ch.title}`, "");
    for (const sec of ch.sections) {
      lines.push(`### ${sec.title}`, "");
      for (const p of sec.paragraphs) lines.push(p, "");
      for (const eq of sec.equations || []) lines.push(`${eq.text}　${eq.num}`, "");
      for (const p of sec.extra_paragraphs || []) lines.push(p, "");
      for (const tid of sec.tables || []) {
        const t = tables[tid];
        lines.push(`**${t.caption}**`, "");
        lines.push(`| ${t.headers.join(" | ")} |`);
        lines.push(`|${t.headers.map(() => "---").join("|")}|`);
        for (const r of t.rows) lines.push(`| ${r.join(" | ")} |`);
        lines.push("");
      }
      for (const fid of sec.figures || []) {
        const f = figuresById[fid];
        lines.push(`> **${f.number}　${f.title}**（${f.caption_boundary}） — 源文件：\`${f.path}\``, "");
      }
    }
  }
  lines.push("## 参考文献", "");
  for (const r of content.references) lines.push(r, "");
  lines.push(`## ${content.appendix.title}`, "");
  for (const s of content.appendix.sections) {
    lines.push(`### ${s.title}`, "");
    for (const p of s.paragraphs) lines.push(p, "");
  }
  return lines.join("\n");
}

async function main() {
  const content = readJson(CONTENT_PATH);
  const v1 = readJson(CLAIM_V1_PATH);
  const v2 = readJson(CLAIM_V2_PATH);

  const figuresById = Object.fromEntries(v2.figures_v5.map((f) => [f.id, f]));
  for (const f of v2.figures_v5) {
    const abs = path.join(ROOT, f.path);
    const actualSha = sha256File(abs);
    if (actualSha !== f.sha256) throw new Error(`SHA mismatch on figure ${f.path}`);
  }

  const tables = buildTablesData(v1, v2);

  // Build Cover section
  const coverChildren = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 320 },
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(LOGO_PATH),
          transformation: { width: 340, height: Math.round((340 * 159) / 851) },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 180 },
      children: [makeRun(content.cover.college, { size: 44, bold: true, eastAsia: "黑体" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 80, after: 760 },
      children: [makeRun(content.cover.report_type, { size: 44, bold: true, eastAsia: "黑体" })],
    }),
    new Paragraph({
      alignment: AlignmentType.LEFT,
      indent: { left: 720 },
      spacing: { before: 120, after: 200, line: 420 },
      children: [makeRun(`题　　目：${content.cover.title}`, { size: 30, bold: true, eastAsia: "黑体" })],
    }),
    ...content.cover.members.map(
      (m) =>
        new Paragraph({
          alignment: AlignmentType.LEFT,
          indent: { left: 720 },
          spacing: { before: 40, after: 140, line: 380 },
          children: [makeRun(`姓　　名：${m.name}　　学号：${m.student_id}`, { size: 30 })],
        })
    ),
    new Paragraph({
      alignment: AlignmentType.LEFT,
      indent: { left: 720 },
      spacing: { before: 40, after: 140, line: 380 },
      children: [makeRun(`指导老师：${content.cover.advisor}`, { size: 30 })],
    }),
    new Paragraph({
      alignment: AlignmentType.LEFT,
      indent: { left: 720 },
      spacing: { before: 40, after: 920, line: 380 },
      children: [makeRun(`起止时间：${content.cover.period}`, { size: 30 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 80 },
      children: [makeRun(content.cover.stage_label, { size: 22, bold: true, eastAsia: "黑体" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 0 },
      children: [makeRun(content.cover.evidence_cutoff, { size: 21, color: "475569" })],
    }),
  ];

  // Build Body section
  const bodyChildren = [];

  // CN Abstract
  bodyChildren.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 120, after: 220, line: 360 },
      children: [makeRun(content.abstract_cn.heading, { size: 32, bold: true, eastAsia: "黑体" })],
    })
  );
  for (const p of content.abstract_cn.paragraphs) bodyChildren.push(bodyParagraph(p, false));
  bodyChildren.push(
    new Paragraph({
      alignment: AlignmentType.BOTH,
      indent: { firstLine: 480 },
      spacing: { before: 120, after: 120, line: 360 },
      children: [
        makeRun("关键词：", { size: 24, bold: true, eastAsia: "黑体" }),
        makeRun(content.abstract_cn.keywords, { size: 24 }),
      ],
    }),
    new Paragraph({ children: [new PageBreak()] })
  );

  // EN Abstract
  bodyChildren.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 120, after: 220, line: 360 },
      children: [makeRun(content.abstract_en.heading, { size: 32, bold: true, eastAsia: "Times New Roman" })],
    })
  );
  for (const p of content.abstract_en.paragraphs) bodyChildren.push(bodyParagraph(p, true));
  bodyChildren.push(
    new Paragraph({
      alignment: AlignmentType.BOTH,
      indent: { firstLine: 480 },
      spacing: { before: 120, after: 120, line: 360 },
      children: [
        makeRun("Keywords: ", { size: 24, bold: true, eastAsia: "Times New Roman" }),
        makeRun(content.abstract_en.keywords, { size: 24, eastAsia: "Times New Roman" }),
      ],
    }),
    new Paragraph({ children: [new PageBreak()] })
  );

  // TOC
  bodyChildren.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 120, after: 200, line: 360 },
      children: [makeRun(content.toc_heading, { size: 32, bold: true, eastAsia: "黑体" })],
    }),
    new TableOfContents("目录", {
      hyperlink: true,
      headingStyleRange: "1-2",
    })
  );

  // Chapters 1-5
  for (const ch of content.chapters) {
    bodyChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        outlineLevel: 0,
        alignment: AlignmentType.CENTER,
        pageBreakBefore: true,
        keepNext: true,
        spacing: { before: 160, after: 220, line: 360 },
        children: [makeRun(ch.title, { size: 32, bold: true, eastAsia: "黑体" })],
      })
    );
    for (const sec of ch.sections) {
      bodyChildren.push(
        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          outlineLevel: 1,
          alignment: AlignmentType.LEFT,
          keepNext: true,
          spacing: { before: 180, after: 110, line: 360 },
          children: [makeRun(sec.title, { size: 28, bold: true, eastAsia: "黑体" })],
        })
      );
      for (const p of sec.paragraphs) bodyChildren.push(bodyParagraph(p, false));
      for (const eq of sec.equations || []) bodyChildren.push(equationParagraph(eq));
      for (const p of sec.extra_paragraphs || []) bodyChildren.push(bodyParagraph(p, false));
      for (const fid of sec.figures || []) bodyChildren.push(...buildFigure(figuresById[fid]));
      for (const tid of sec.tables || []) bodyChildren.push(...buildTable(tables[tid]));
    }
  }

  // References
  bodyChildren.push(
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      outlineLevel: 0,
      alignment: AlignmentType.CENTER,
      pageBreakBefore: true,
      keepNext: true,
      spacing: { before: 160, after: 200, line: 360 },
      children: [makeRun("参考文献", { size: 32, bold: true, eastAsia: "黑体" })],
    })
  );
  for (const refText of content.references) {
    bodyChildren.push(
      new Paragraph({
        alignment: AlignmentType.LEFT,
        indent: { left: 380, hanging: 380 },
        spacing: { before: 0, after: 50, line: 270 },
        children: [makeRun(refText, { size: 20, color: "1E293B" })],
      })
    );
  }

  // Appendix
  bodyChildren.push(
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      outlineLevel: 0,
      alignment: AlignmentType.CENTER,
      pageBreakBefore: true,
      keepNext: true,
      spacing: { before: 160, after: 220, line: 360 },
      children: [makeRun(content.appendix.title, { size: 32, bold: true, eastAsia: "黑体" })],
    })
  );
  for (const s of content.appendix.sections) {
    bodyChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        outlineLevel: 1,
        alignment: AlignmentType.LEFT,
        keepNext: true,
        spacing: { before: 180, after: 110, line: 360 },
        children: [makeRun(s.title, { size: 28, bold: true, eastAsia: "黑体" })],
      })
    );
    for (const p of s.paragraphs) bodyChildren.push(bodyParagraph(p, false));
  }

  const header = new Header({
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 40, line: 240 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CBD5E1", space: 4 } },
        children: [makeRun(content.cover.title, { size: 18, color: "555555" })],
      }),
    ],
  });

  const footer = new Footer({
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 0 },
        children: [
          makeRun("— ", { size: 20, color: "111111" }),
          new TextRun({ children: [PageNumber.CURRENT], font: makeFont("宋体", "Times New Roman"), size: 20 }),
          makeRun(" —", { size: 20, color: "111111" }),
        ],
      }),
    ],
  });

  const doc = new Document({
    features: { updateFields: true },
    styles: {
      default: {
        document: {
          run: { font: makeFont("宋体", "Times New Roman"), size: 24, color: "111111" },
          paragraph: { spacing: { line: 360 } },
        },
        heading1: {
          run: { font: makeFont("黑体", "Times New Roman"), size: 32, bold: true, color: "111111" },
          paragraph: { outlineLevel: 0 },
        },
        heading2: {
          run: { font: makeFont("黑体", "Times New Roman"), size: 28, bold: true, color: "111111" },
          paragraph: { outlineLevel: 1 },
        },
      },
    },
    sections: [
      {
        properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: MARGINS } },
        children: coverChildren,
      },
      {
        properties: {
          page: { size: { width: PAGE_W, height: PAGE_H }, margin: MARGINS, pageNumbers: { start: 1 } },
        },
        headers: { default: header },
        footers: { default: footer },
        children: bodyChildren,
      },
    ],
  });

  fs.mkdirSync(path.dirname(OUT_DOCX), { recursive: true });
  fs.mkdirSync(path.dirname(OUT_RECEIPT), { recursive: true });

  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(OUT_DOCX, buf);

  const mdText = buildMarkdownText(content, tables, figuresById);
  fs.writeFileSync(OUT_MD, mdText, "utf8");

  const receipt = {
    schema: "final-report-build-receipt/v5",
    date: "2026-11-14",
    builder: "project/src/documents/build_final_report_v5.js",
    inputs: {
      content_json: { path: path.relative(ROOT, CONTENT_PATH).replace(/\\/g, "/"), sha256: sha256File(CONTENT_PATH) },
      claim_v1_json: { path: path.relative(ROOT, CLAIM_V1_PATH).replace(/\\/g, "/"), sha256: sha256File(CLAIM_V1_PATH) },
      claim_v2_json: { path: path.relative(ROOT, CLAIM_V2_PATH).replace(/\\/g, "/"), sha256: sha256File(CLAIM_V2_PATH) },
    },
    outputs: {
      docx: {
        path: path.relative(ROOT, OUT_DOCX).replace(/\\/g, "/"),
        bytes: fs.statSync(OUT_DOCX).size,
        sha256: sha256File(OUT_DOCX),
      },
      markdown: {
        path: path.relative(ROOT, OUT_MD).replace(/\\/g, "/"),
        bytes: fs.statSync(OUT_MD).size,
        sha256: sha256File(OUT_MD),
      },
    },
    counts: {
      tables: Object.keys(tables).length,
      body_figures: v2.figures_v5.length,
      total_png_media_including_cover_logo: v2.figures_v5.length + 1,
      references: content.references.length,
    },
  };
  fs.writeFileSync(OUT_RECEIPT, JSON.stringify(receipt, null, 2), "utf8");
  console.log(JSON.stringify(receipt, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
