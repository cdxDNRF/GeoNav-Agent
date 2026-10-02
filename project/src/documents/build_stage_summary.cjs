// Four-page stage summary; leave the original midterm report unchanged.
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, Footer, PageNumber, ExternalHyperlink,
} = require('docx');
const ROOT = path.resolve(__dirname, '../../..');
const OUT = path.join(ROOT, '中期报告相关');
const d = JSON.parse(fs.readFileSync(path.join(ROOT, '绘图/绘图数据/阶段结果_v1.json'), 'utf8'));
const rows = d.edge.rows;
const pct = n => (100*n).toFixed(2)+'%';

function p(text, opts={}) {
  return new Paragraph({spacing:{after:110,line:300},...opts,
    children:[new TextRun({text,...(opts.run||{})})]});
}
function h(text, page=false) {
  return new Paragraph({text,heading:HeadingLevel.HEADING_1,pageBreakBefore:page,
    spacing:{before:120,after:140},keepWithNext:true});
}
function image(rel, width, height) {
  return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:80,after:70},
    children:[new ImageRun({type:'png',data:fs.readFileSync(path.join(ROOT,rel)),
      transformation:{width,height},altText:{title:rel,description:'从本项目既有实验来源生成的科研图',name:rel}})]});
}
function table(headers, data, widths) {
  const border={style:BorderStyle.SINGLE,size:4,color:'CBD5DF'};
  return new Table({width:{size:widths.reduce((a,b)=>a+b,0),type:WidthType.DXA},columnWidths:widths,
    rows:[headers,...data].map((row,i)=>new TableRow({tableHeader:i===0,cantSplit:true,
      children:row.map((value,j)=>new TableCell({width:{size:widths[j],type:WidthType.DXA},
        shading:{type:ShadingType.CLEAR,fill:i===0?'E9EFF5':'FFFFFF'},
        borders:{top:border,bottom:border,left:border,right:border},
        margins:{top:70,bottom:70,left:100,right:100},
        children:[new Paragraph({spacing:{after:0,line:260},
          children:[new TextRun({text:String(value),bold:i===0,size:20})]})]}))}))});
}
function fileLink(label, rel) {
  return new Paragraph({spacing:{after:80,line:260},children:[new ExternalHyperlink({
    link:'file:///'+path.join(ROOT,rel).replace(/\\/g,'/'),
    children:[new TextRun({text:label,style:'Hyperlink',size:21})]})]});
}

const content=[
  new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:90},
    children:[new TextRun({text:'视觉地理定位项目 · 阶段提升总结',size:34,bold:true})]}),
  p('阶段快照：2026-09-30  ｜  开发验证通过，正式S2待复验',
    {alignment:AlignmentType.CENTER,run:{size:21,color:'526674'}}),
  h('1  本阶段的实质进展'),
  p('在相同140项已知开发任务、28张源图、三个配对训练种子下，边缘连续性候选把导航SR从72.62%提高到88.57%，增加15.95个百分点；SG由0.864降到0.348。三个种子均提升，同题置零对照复现旧结果，完整复核通过。'),
  table(['同题条件','成功 / 轨迹','SR','SG（格）'],rows.map(r=>[
    r.label,`${r.successes} / ${r.planned}`,pct(r.sr),r.sg.toFixed(3)]),[3700,2000,1400,1900]),
  image('绘图/数据结果图/01_边缘线索主结果.png',590,295),
  p('图1  同题目标干预与主结果。白色标记为三个训练种子。140×3=420条轨迹，不能视为420张独立源图；SG为所有正常完成轨迹的终点曼哈顿距离均值，成功记0。',
    {run:{size:19,color:'526674'},spacing:{after:80,line:240}}),
  p('当前默认仍为Small256 NoTarget：原val100平均SR73%、SG0.800。最新88.57%属于另一份开发任务集，正确提升比较是88.57%−72.62%，不能减去73%。本次收尾没有运行正式复验、修改默认或新增实验。'),

  h('2  从探索策略到可靠目标线索',true),
  table(['步骤','已观察到的结果','形成的判断'],[
    ['环境与评测基础','修复连接、协议、日志与重放；云端同策略Gemma45%、DeepSeek非思考20%、规则50%（val20）','工程完成不等于任务成功；模型服务更大不保证SR更高'],
    ['PBRS与动作筛选','历史val100：PBRS44.33% → 筛选66%；目标遮蔽65.67%','保留动作筛选，继续检查目标贡献'],
    ['模型与训练方式','扩大GRU未提升；完整导航训练下局部匹配形成观察候选；val100默认无目标73%','完整导航训练和强探索基线重要；不能仅靠扩大参数'],
    ['可信邻接、邻接/方向解耦','同题开发集两个头均无合格阈值，全部弃权，SR均72.62%','保留可信要求；条件方向信号尚不足以介入'],
    ['边缘连续性线索','开发SR88.57%；C4/C5：44.64% → 74.40%；三个种子均提升','形成可靠邻接线索，转为固定配置复验候选'],
  ],[1800,4350,2850]),
  p('上述过程保留正结果、负结果和接口中断记录。每一步的实验列表、训练预算与信息边界以原预登记为准；不同任务集的成绩分开呈现，不把全过程写成一条受控累积提升曲线。'),
  image('绘图/数据结果图/05_同任务集历史方案.png',600,315),
  p('图2  原val100与已知开发140分别展示；白色标记为训练种子。开发面板的不同批次训练设定有差异，仅原配对实验支持对应因素效应。',
    {run:{size:19,color:'526674'},spacing:{after:0,line:240}}),

  h('3  当前Agent与新增机制',true),
  p('当前实现是一个导航智能体：探索器、视觉线索头和行动门控共同形成一个策略。本地分支包含训练；原始“零训练多智能体”的研究目标尚未实现。GUI不是评测必需，本阶段未建设交互轨迹回放界面。'),
  image('绘图/Agent架构图/01_边缘线索Agent架构.png',600,388),
  p('图3  候选Agent的离线训练、在线闭环和评测边界。虚线为冻结配置或评测记录流；真实目标坐标与距离只在评测侧使用。',
    {run:{size:19,color:'526674'},spacing:{after:80,line:240}}),
  p('新增20维边缘关系特征：四个方向，各含三个宽度的RGB沿边误差、沿边梯度误差和去均值相关性。探索器、Sat2Cap编码器和原1041维输入保持固定；ZeroEdge/Edge均136837参数，同初始化、同16轮训练预算，新2560参数投影从0初始化。'),
  p('邻接概率与条件方向概率相乘形成联合置信度；仅当原始五类最高方向≥0.50、动作合法且落点未访问时接受线索，其余情况沿用探索。校准阈值先冻结再验证，不读取未知邻格图像。'),

  h('4  证据、边界与接续',true),
  p('SR增益的28源图成组95%区间为[+11.43,+20.71]个百分点；真实目标相对均值/错误目标分别提高18.33/18.57点。三个头在开发留出图的接受线索一步命中精度为95.11%、94.65%、95.81%；该组件精度不是导航SR。'),
  p('188项测试通过；另行实现的复核检查3425个实际图块profile、694800条配对预测、3360条神经导航及27361个动作、280条规则轨迹，并检查数据隔离、校准、配对初始化、源图区间与哈希。实现及复核由同一执行代理完成，不声称不同人员独立审查。'),
  p('28张图经过多轮开发，不是新独立确认。边缘机制目前适用于同源、连续、方向一致的300×300切图；旋转、异时、非连续裁切、跨视角或跨数据集尚未验证。仅在上述开发范围内主张目标线索有效，不声称通用语义地理理解。'),
  h('收尾文件怎样找到'),
  fileLink('项目总览与当前阶段','选题报告相关/阶段收尾_v1/README.md'),
  fileLink('完整实验索引：包含负结果与未完成评测','选题报告相关/阶段收尾_v1/实验索引.md'),
  fileLink('集中小文件、原路径与SHA-256','选题报告相关/阶段收尾_v1/整理清单.json'),
  fileLink('科研图、矢量格式、图注与drawio源图','绘图/README.md'),
  fileLink('最新验收结论与候选来源','DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1/验收结论.json'),
  h('下一阶段的明确入口'),
  p('冻结三个Edge头、三个探索器、四份均值及0.50阈值，进行正式val100×3同题复验；保留默认/置零/规则基线及真实、均值、错误目标干预，核验距离档、种子、源图区间和完整轨迹。达到既定要求后判定S2并决定默认更新，随后S3再做独立源图确认。'),
  p('正式复验必须正确读取val图片和目标替换路径，不能直接复用训练集专用运行器中的train硬编码分支；本次不加新因素、不重新按val调阈值。原数据、模型、实验路径和原中期报告全部保留。'),
];

const doc=new Document({
  creator:'GeoNav-Agent project',title:'视觉地理定位项目阶段提升总结',
  description:'Verified development evidence and stage-closeout index, 2026-09-30',
  styles:{default:{document:{run:{font:'宋体',size:22},paragraph:{spacing:{after:110,line:300}}}},
    paragraphStyles:[{id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,
      run:{font:'黑体',size:26,bold:true,color:'172D3D'},paragraph:{outlineLevel:0,spacing:{before:140,after:140}}}]},
  sections:[{properties:{page:{size:{width:11906,height:16838},
    margin:{top:1050,bottom:1050,left:1253,right:1253,footer:480,header:480}}},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,
      children:[new TextRun({text:'阶段收尾快照 · 2026-09-30  /  ',font:'宋体',size:18,color:'526674'}),
      new TextRun({children:[PageNumber.CURRENT],size:18})]})]})},children:content}]
});

const md=`# 视觉地理定位项目：阶段提升总结

日期：2026-09-30。开发验证通过，正式S2待复验。本次只整理已有成果，不新增训练、修改默认或改动原中期报告。

## 核心提升

相同已知开发140题、28张源图、3训练种子、5×5/B10：冻结探索/置零SR72.62%、SG0.864；边缘真实目标SR88.57%、SG0.348；均值目标70.24%、SG0.895；错误目标70.00%、SG0.929。真实目标同题增益15.95个百分点，源图成组95%区间[+11.43,+20.71]点；三个种子均提升。C4/C5合并SR由44.64%升至74.40%。当前默认仍是Small NoTarget，原val100 SR73%、SG0.800，不计算跨任务集差值。

## 改进过程

先修复环境与统一评测，完成最小云端VLM；随后逐项对照排序/行动治理、空间记忆、邻域确认、层级搜索，保留故障与负结果。历史val100中，PBRS44.33%经合法动作筛选达到66%，但目标遮蔽65.67%，目标利用证据不足。Gemma/DeepSeek同G策略的val20比较为45%/20%，Frontier50%；DeepSeek为非思考服务配置且正常完成18/20，不能用模型参数大小推断成绩。

本地扩大GRU和直接方向监督迁移未形成新候选，完整PBRS导航训练与局部匹配产生观察候选；固定val100复验选择Small NoTarget73%作开发默认。可信五分类邻接头和邻接/方向解耦头在同题开发集都没有合格阈值，全部弃权，SR仍72.62%。保持可信门槛后，增加20维图块边缘关系的候选才形成可靠邻接信息，最终SR88.57%。历史全过程按任务集分组，不视作一条受控累计提升曲线。

## 当前Agent

单导航智能体由冻结Small NoTarget探索器、冻结Sat2Cap、两图关系特征、解耦邻接/方向头和行动门控组成。本地有训练，零训练多智能体尚未实现。仅使用当前图、给定目标图和公开状态；不读取未知邻格图像、目标坐标或真实距离。四方向各5种边缘关系构成20维特征，原1041维输入不变；两臂均136837参数，新2560参数投影初始化为0，6个头同16epoch训练预算。联合置信度为p(邻接)×p(方向|邻接)，最高方向≥0.50且合法/未访问才接受，否则沿用探索。三阈值在22张校准图上先冻结。

## 证据与边界

三个头在开发留出图的接受线索一步命中精度95.11%/94.65%/95.81%；该精度不是导航SR。188项测试通过，复核3425个实际profile、694800条预测、3360条神经导航及27361动作、280条规则轨迹及相关输入/模型/代码哈希。实现及另行编写复核由同一执行代理完成，不声称不同人员独立审查。

140×3是420条轨迹而非420个独立样本；28张图已参与多轮开发。SG是所有正常完成轨迹的终点距离均值，成功为0。机制当前依赖同源连续、方向一致的300×300切图；异时、旋转、跨视角、独立源图或跨数据集尚未验证。S2正式复验待开展，S3独立确认与大网格未开展。

## 文件入口与下一步

- [阶段总览](../选题报告相关/阶段收尾_v1/README.md)
- [完整实验索引](../选题报告相关/阶段收尾_v1/实验索引.md)
- [小文件快照与哈希](../选题报告相关/阶段收尾_v1/整理清单.json)
- [科研图与完整图注](../绘图/README.md)
- [最新结果和证据](../DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1/结果解读与下一项.md)

下一项固定3头、3探索器、4均值及阈值0.50，做val100×3同题复验，保留基线/规则/目标干预/分档/种子/源图区间和完整重放。正式复验须正确使用val图片与错图路径，不能直接复用train硬编码分支。通过后决定S2与默认，再开展S3。原始数据、模型、实验目录和原中期报告保持原状。
`;
fs.writeFileSync(path.join(OUT,'阶段提升总结_v1.md'),md,'utf8');
Packer.toBuffer(doc).then(buffer=>{
  fs.writeFileSync(path.join(OUT,'阶段提升总结_v1.docx'),buffer);
  console.log(JSON.stringify({docx:'中期报告相关/阶段提升总结_v1.docx',bytes:buffer.length}));
}).catch(err=>{console.error(err);process.exitCode=1;});
