"""校正原报告的 OOXML 副本；绝不调用旧 build_report.js 或覆盖原件。
python project/src/documents/correct_report.py --prepare-figure
python project/src/documents/correct_report.py  # after rendering corrected HTML to PNG
Uses official Document editor, keeps paragraph/run properties; records tracked text
changes in an audit copy, accepts them for the clean final deliverable.
"""
from pathlib import Path
import sys, re, json, zipfile, shutil, tempfile, hashlib, difflib, argparse
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'选题报告相关'
REVIEW=BASE/'校正审阅'
SKILL=Path(r'C:/Users/DFWJ/.zcode/cli/plugins/cache/zcode-plugins-official/documents/0.1.7/skills/docx')
sys.path.insert(0,str(SKILL))
from scripts.document import Document
SOURCE=BASE/'选题报告_基于主动探索的视觉地理定位方法设计与实现.docx'
TARGET=SOURCE.with_name(SOURCE.stem+'_校正版.docx')
G='论文/GOMAA-Geo.pdf，PDF第3–6页（§3、§4；GASP与PPO）；官方 GOMAA-Geo/README.md:78–95、train.py:34–41、models/ppo.py:165–176。'
E='论文/GeoExplorer.pdf，PDF第2–5页（§3.3–3.5；动作—状态联合建模、PPO、预测误差好奇心奖励）。'
D='论文/DynCur-Geo.pdf，PDF第2–4页（Method、Policy Learning and Inference）；第4页明确部分短距离基线更强。仅核验本地论文陈述，未复现实验或核验官方实现。'
P='论文/GOMAA-Geo.pdf，PDF第3、6–7页；GeoExplorer.pdf，第5–6、10、15页；DynCur-Geo.pdf，第4、6、23页。任务族相近不代表数据、训练池与预算一致。'
A='论文/GOMAA-Geo.pdf，第3页；GeoExplorer.pdf，第3页；官方 GOMAA-Geo/config.py:3–7、44（四动作）；models/ppo.py:174–176。sequence.py:14虽有STOP占位token，不构成标准策略动作。'
L='论文/GOMAA-Geo.pdf，第3页明确隐藏目标坐标、仅局部观测；train.py:95–105、models/ppo.py:374–385将目标位置从模型位置序列中切除。更严格的文件名/距离/全图隔离为本课题防泄漏约束。'
M='本课题研究设计约束，非归属于某篇论文：空间记忆限单episode；排除结论依赖目标；独立测试不读写跨测试任务记忆，continual另设协议。'
H='本课题待验证假设；论文不能替代本课题实验，不预先承诺显著性或提升幅度。'
S='论文/GeoExplorer.pdf，第10页5×5/300×300 patch设置、第15页§S4.6（10×10需重训）；GOMAA-Geo.pdf第6页明确B/C。细化同一幅图不改变地理覆盖范围。'
DATA='本地 DATA/raw_data/Masa/png/{train,val,test} 实数为137/4/10（tiff同样）；GeoExplorer.pdf第5页1188与70/15/15，第10页列832/178/179（相加1189，原文计数有不一致）。与本地数据的来源差异未查明。'
CHANGES={}
def c(i,text,basis): CHANGES[i]={'new':text,'basis':basis}
c(27,'传统视觉地理定位常采用被动式单次预测：给定查询图像，通过检索、分类或回归输出位置估计。单张图像的信息与参考图库覆盖可能限制定位效果；这类设置本身不包含主动移动获取新观测的决策闭环，并不意味着所有被动方法都不可迭代修正。对于具备机动能力的无人机，主动获取后续观测是值得研究的补充途径。','GOMAA-Geo.pdf第2页Related Work区分one-shot检索与主动导航；删除对全部被动方法的绝对化断言。')
c(28,'主动地理定位（Active Geo-localization, AGL）将目标定位建模为序列决策：智能体依据目标线索与逐步获得的局部航拍观测，在步数预算内搜索目标。GOMAA-Geo[1]通过目标感知监督预训练与PPO强化学习构建单策略智能体，其零样本能力指训练后的跨模态、跨域迁移，并非零训练；GeoExplorer[2]在动作—状态动力学建模基础上引入好奇心奖励训练策略。DynCur-Geo[3]进一步研究训练阶段的动态好奇心奖励塑形。',G+E+D)
c(29,'本课题面向这一方向，拟设计主动视觉地理定位方法及实验平台：应用层面，以无人机搜救为背景研究局部观测下的目标搜索，不将网格仿真等同于实机导航验证；研究层面，考察较大搜索空间中的重复访问、绕行与收敛问题；培养层面，覆盖环境建模、智能体设计、大模型应用与实验评测，在算法和工程两个层面形成可检验成果。',H)
c(32,'视觉地理定位常通过查询图像与带地理标签的参考图像匹配[4]，深度学习进一步推动了表征与检索方法的发展[5]。跨视角地理定位研究地面、无人机与卫星等视角间的匹配，相关基准包括CVUSA[6]、University-1652[7]、VIGOR[8]。与本课题相比，这些被动检索设置通常不包含逐步移动获取局部观测的交互过程，不能未经协议转换就与AGL方法比较。','GOMAA-Geo.pdf第2页Related Work；本次未重新核验[4–8]全部出版元数据，保留原参考条目。')
c(34,'代表性工作见表1。GOMAA-Geo[1]由跨模态表征、目标感知监督预训练（GASP）与PPO[13]策略学习组成，不是多个VLM协商的零训练架构。GeoExplorer[2]通过有监督的动作—状态联合预测建模环境，再以外部距离奖励与预测误差好奇心奖励训练PPO策略，不能将跨episode经验记忆归为其核心机制。DynCur-Geo[3]在本地论文中提出距离门控好奇心与势函数奖励塑形；距离用于训练奖励而非推理输入，其提升属于论文特定设置，不能概括为所有条件下全面优于前作。',G+E+D)
for i,text,b in [(43,'监督预训练 + PPO单策略',G),(44,'模态对齐、GASP、历史条件决策',G),(45,'需训练；迁移能力需按设置验证',G),(48,'动力学预训练 + PPO',E),(49,'动作—状态建模、好奇心奖励',E),(50,'需训练；探索与收敛需权衡',E),(53,'预训练表征 + PPO',D),(54,'训练期距离门控好奇心、势函数塑形',D),(55,'依赖训练奖励；论文结果需协议核对',D),(58,'拟议零训练多智能体',H),(59,'episode内记忆、层级搜索与确认',M+H),(60,'效果与调用成本待验证',H)]: c(i,text,b)
c(63,'在现有训练式方法的基础上，本课题拟检验以下问题，而不预设相关问题尚未被任何工作研究：',H)
c(64,'（1）尺度问题。需区分同一区域网格细化与地理搜索范围扩大；前者改变单格视野和离散步长，不能直接代表后者。拟在明确覆盖范围、分辨率、步数预算B及初始距离C后评估搜索难度变化；',S)
c(65,'（2）无效移动问题。重复访问、震荡与进入目标邻域后离开是拟分析的轨迹现象；GOMAA-Geo已包含重访惩罚，故不宣称本课题首次解决。拟检验显式记忆与邻域确认能否进一步改善这些现象；',G+' GOMAA-Geo.pdf第6页式(6)；DynCur-Geo.pdf第10、19页路线统计。')
c(66,'（3）方法路线。拟自主设计“多智能体 + 记忆与层级搜索”的免任务训练方案，冻结预训练VLM/LLM参数，通过提示、交互与上下文组织完成决策；其与训练式策略的效果及成本差异均需实验验证。',G+H)
c(67,'上述研究共享网格化局部观测任务的基本形式，但并非全部使用相同数据和协议。开展数值比较前，须核对split、地理区域与目标模态、图像预处理及单格视野、动作与边界规则、B/C、任务采样、训练数据和权重、解码及终止规则。MASA、MM-GAG、SwissView及xBD[12]可作为候选数据来源；论文报告值仅作背景，不可与本地不同数据集结果直接相减并宣称提升。',P+DATA)
c(69,'总体目标：构建局部观测下的主动地理定位实验平台，自主设计零训练的多智能体记忆与层级搜索方法，检验其能否改善定位表现并减少无效探索。这里“零训练”指不进行本课题任务相关的参数训练或微调，不否认基础模型已有预训练。研究目标与创新点均为待验证假设，允许得到不支持假设的结果。具体目标如下：',H+G)
c(70,'平台目标：支持多模态线索、5×5至25×25可配置网格、预算B与初始距离C；记录覆盖范围、单格视野和分辨率，提供SR/SG及轨迹分析，并严格隔离智能体观测与评测真值；',S+L)
c(71,'方法目标：自主设计感知、记忆与决策协调的多智能体基础框架，研究“少走回头路”的episode内记忆与“少走弯路”的层级搜索；GOMAA-Geo作为训练式对照，而非被复现的多智能体原型；',G+M)
c(72,'实验目标：在冻结的本地测试任务上对比Random、受限观测贪心、单智能体及无记忆/无层级搜索消融；训练式基线仅在复现条件与协议对齐后加入。检验大网格SR提高5个百分点、重复访问率相对降低30%的探索性目标，报告不确定性与失败案例，不将其作为已达成结论或必达承诺；',H+P)
c(77,'探索记忆的表示与利用：在局部观测和有限上下文下，如何紧凑记录本episode已访问网格、观测摘要与目标相关证据，减少不必要重访，同时允许合理回退并避免将低置信度判断永久视为排除事实；',M+L)
c(78,'不同搜索尺度下的探索—收敛权衡：如何仅依据已访问观测形成区域候选，并在粗层级规划与相邻网格移动之间分配预算；跨尺度比较需明确B、物理步长及视野变化，避免将额外信息或预算误认作算法收益；',S+L)
c(79,'视觉歧义区域的目标确认：如何依据可见线索核实候选，而非使用真值距离；标准四动作协议由环境判定到达，若增加智能体stop动作，须另列停止定位协议及指标，不能混入标准SR对比。',A+L)
c(81,'拟基于公开数据构建统一环境：每步只返回当前网格局部观测，动作限上下左右，首次到达目标或预算耗尽时由环境终止。MASA作为主要实验来源，MM-GAG用于多模态测试，SwissView与xBD作为条件具备后的跨域测试。Agent不得获取goal坐标、可定位目标的文件名/索引、真值距离、完整搜索图或未访问网格特征；评测器持有真值，oracle只用于环境单元测试或上界诊断，不作为可部署基线。冻结区域级划分和任务清单后再比较方法，不能仅凭接口统一就认定与论文可比。',P+A+L)
c(82,'（三）多智能体主动定位基础框架（自主设计）',G)
c(83,'本课题拟自主设计零训练多智能体系统：感知定位智能体依据当前观测与目标线索形成候选和置信度；记忆管理智能体维护本episode证据；决策协调智能体结合剩余预算选择合法移动。可借鉴推理—行动交替思想[10]组织内部交互，但不将该分工归于GOMAA-Geo。标准实验仅输出四方向动作；stop扩展独立评估。系统冻结VLM/LLM参数，重点研究提示、证据校验与上下文组织，而非复现已有“零训练多VLM架构”。',G+A+H)
c(85,'拟设计显式episode内空间记忆，记录已访问位置、视觉摘要、目标相关匹配证据与置信度，通过紧凑表示和遗忘策略检验能否减少不必要重访。“此处无目标”等负面结论依赖当前目标，不能无条件复用于新任务；独立测试中每个episode重置记忆，开发与测试严格隔离。跨任务经验仅作为可选continual实验，单列任务顺序、允许保留的信息与重置规则，不与独立测试混报。',M+E)
c(87,'拟采用由粗到细的层级规划：仅从目标线索与已访问局部观测更新区域候选，再在候选区域安排网格级搜索；不读取完整地图、缩略全图或未访问特征，也不以“粗搜”名义额外获得观测。层级是内部规划抽象，执行仍为相邻四方向移动且逐步计费。邻域确认根据可见证据触发，效果通过消融验证；若扩展视野、跳跃移动或stop，则建立独立协议并显式核算预算。',L+S+A+H)
c(89,'所有用于提升声明的方法均须在同一冻结测试清单上运行，保持split、区域、观测、动作、B/C、目标模态和终止条件一致。训练式基线需核对其训练数据与权重；无法对齐时，论文值只列作背景。贪心策略不得预读相邻未访问图块，需明确可见信息与打分规则；消融固定模型、提示和预算，分别移除记忆、层级搜索及多智能体协作，统计SR/SG、重复访问与调用成本，并按场景和随机种子报告不确定性。',P+L+H)
c(91,'系统技术路线如图1所示，拟由“环境平台 + 自主设计的多智能体闭环”组成。环境按标准四动作协议返回局部观测并判断首次到达或预算耗尽；真值仅供评测，不进入Agent决策。stop与跨任务记忆扩展分别另设协议。',A+L+M)
c(95,'输入目标线索（航拍图、地面照片或文本）；环境初始化任务并清空episode记忆，隐藏目标坐标、文件名索引与真值距离；',L+M)
c(97,'记忆管理智能体只更新本episode已访问位置与目标相关证据，不检索其他测试任务记录；',M)
c(100,'重复步骤2至5，直至环境判定首次到达或预算耗尽；智能体stop仅在独立扩展协议中启用；',A)
c(101,'评测器用隔离真值计算SR、SG、步数和重复访问率，记录轨迹与VLM调用成本；',L+P)
c(102,'独立测试在任务结束后封存日志并清空记忆；可选continual实验按单独规则保留信息与报告结果。',M)
c(103,'关键实现选型拟在开发阶段锁定：平台采用Python；视觉语言模型可选择基于Transformer架构[14]的商用API或本地开源模型，固定版本、提示和解码参数；特征编码可采用冻结的CLIP[9]等预训练模型。GeoExplorer的好奇心设计[11]属于训练式对照。所提方法不进行任务相关参数训练，但训练式基线的预训练、PPO与权重准备需单独核算；整体成本还包括VLM调用、上下文与多智能体协调开销。',G+E+H)
c(106,'零训练方案避免本课题方法的参数训练开销，但VLM显存、调用费用和时延仍需实测。Python网格环境与模块化接口有利于控制实验条件。GOMAA-Geo官方源码（github.com/mvrl/GOMAA-Geo）可供核对数据处理、预训练与PPO评测流程；使用其权重与对照结果仍需核实版本、训练数据和测试协议，不能视为可直接复用的多智能体系统。',G+P)
c(108,'候选数据来自公开资源，但“公开”不等于已完整取得或划分一致。当前本地Masa为训练137、验证4、测试10幅；GeoExplorer论文描述1188幅及70%/15%/15%划分，二者来源差异尚未查明，不能称本地数据为作者内部重切版。拟保留本地数据版本标识并核查区域重叠；在完成溯源前只做本地协议内对比，不与论文值直接声明提升。',DATA)
c(110,'三人拟分别负责实验平台、记忆机制与搜索策略，按18周教学安排设置六个里程碑。先完成协议与基线，再开展机制实验；训练式基线复现和跨域数据准备需预留时间。若资源不足，缩减实验范围并如实说明，不以未对齐的论文数值替代本地对照。',H+P)
c(111,'（四）文献与方法基础',G+E)
c(112,'本课题以GOMAA-Geo的目标模态无关任务定义、GeoExplorer的动力学建模与好奇心探索、DynCur-Geo的训练奖励设计作为研究参照。三者均不能作为零训练多智能体方案已获验证的证据。后续需先锁定数据版本、观测权限与评测协议，再研究多智能体、episode内记忆及层级搜索的独立贡献。',G+E+D+H)
c(114,'拟议创新一：面向不必要重访的episode内显式空间与证据记忆。在自主设计的多智能体框架中研究置信度、更新与遗忘规则；假设其可降低重复访问，但需与无记忆和单智能体版本对比。跨任务经验仅作为隔离的continual扩展，不预设可无条件迁移负面结论；',M+H)
c(115,'拟议创新二：在局部观测、四方向移动与统一预算下开展区域级规划—网格级执行的层级搜索及证据驱动确认。假设其可改善搜索效率与收敛，需排除额外视野、真值距离及动作权限造成的伪提升，并通过消融检验；',L+S+H)
c(116,'工程贡献：建立版本化、权限隔离、可复现实验平台，显式记录数据split、区域、B/C、视野、动作和目标模态。其作用是支持同协议内公平比较；不宣称存在无需核对即可通用的统一基准。',P+L)
for i,text in [(129,'待检验：同协议下与训练式基线相当'),(132,'待检验：同协议下不高于对照'),(134,'明确B/C、视野和区域的10×10及以上测试'),(135,'探索性假设：SR提高5个百分点'),(138,'探索性假设：较无记忆版相对下降30%')]: c(i,text,H+P)
c(137,'重访动作数/实际动作数；起点计已访问，零步任务单列',M+' 本课题指标操作定义，避免分母不明。')
c(140,'SR与SG见相关AGL论文[2,3]，但指标名称相同不保证实验可比。SR按预算内首次命中率统计，SG为终止位置至目标的曼哈顿距离，均由隔离评测器计算。表2的量化目标为探索性假设，须结合多次运行、置信区间和成本报告检验；未达到目标也应如实分析，不以论文值或事后改变协议替代验证。',P+H)
c(141,'主要风险与对策：调用成本过高时控制上下文与协作轮数，分开统计移动和模型调用预算；基线难以复现时缩减比较范围并标注未复现，论文值仅作背景；模型输出不稳定时采用结构化校验、固定解码与种子并报告多次运行结果；多次投票的额外调用成本同步计入。',P+H)
c(168,'episode内记忆；层级搜索与证据确认；隔离消融实验',M+H)
c(169,'M4：完成假设检验',H)
c(173,'同协议对比；区分网格细化与区域扩大；失败分析',S+P)
c(191,'VLM接入与交互协议；episode内记忆及可选continual隔离',M)

FIG_REPLACEMENTS={
'★1/★2':'1/2', '= 本课题创新点（记忆机制 / 层级搜索）':'= 拟议创新（效果待验证）',
'MASA · MM-GAG · SwissView · xBD':'MASA为主；MM-GAG / SwissView / xBD按条件扩展',
'（全部公开）':'（须核对split）',
'episode 初始化 · 局部观测返回 · 动作执行 · 终止判定':'局部观测 · 四方向动作 · 环境终止；真值仅评测',
'多智能体决策系统（Agent 路线 · 零训练）':'本课题拟议多智能体系统（冻结VLM/LLM）',
'★1':'1','★2':'2',
'空间记忆：已访问网格·排除区<br/>经验记忆：区域级负面经验':'仅本episode已访问位置与证据<br/>独立测试重置；continual单列',
'目标邻域确认与收敛':'证据确认；仍逐格移动计费',
'交互协议：搜索协议 · 对话协议 · 停止协议':'四动作无stop；stop扩展另测；不得读取目标坐标/距离/全图',
'GOMAA-Geo · GeoExplorer · DynCur-Geo · Random · 贪心':'Random / 受限贪心 / 单智能体；训练式基线须同协议复现',
'任务结束：定位结果 + 轨迹':'环境终止：首次到达 / 预算耗尽',
}

def prepare_figure():
    text=(BASE/'figure_tech_route.html').read_text(encoding='utf-8')
    for a,b in FIG_REPLACEMENTS.items():
        assert a in text, a
        text=text.replace(a,b)
    # Visual-gate repair: same original geometry, larger black labels and concise wording.
    for a,b in {
        '仅本episode已访问位置与证据<br/>独立测试重置；continual单列':'episode内证据记忆<br/>独立测试重置；持续另测',
        'VLM 地理关联推理<br/>输出候选方向与置信度':'VLM线索匹配<br/>候选方向与置信度',
        '层级搜索：区域级→网格级<br/>证据确认；仍逐格移动计费':'区域规划→逐格移动<br/>可见证据确认目标',
        '四动作无stop；stop扩展另测；不得读取目标坐标/距离/全图':'四动作无stop；扩展另测；禁用目标坐标、距离与全图',
        'Random / 受限贪心 / 单智能体；训练式基线须同协议复现':'Random / 受限贪心 / 单Agent / 同协议训练式基线',
        '局部观测 · 四方向动作 · 环境终止；真值仅评测':'局部观测 · 四方向动作 · 环境终止；真值仅评测',
        '↑ 动作：上/下/左/右':'↑ 四方向动作',
        'height:112px':'height:126px',
        'top:552px; width:836px; height:96px':'top:552px; width:836px; height:106px',
        'top:172px':'top:188px',
        'font-size:11px':'font-size:18px',
        'font-size:12px':'font-size:18px',
        'font-size: 11px':'font-size: 18px',
        'font-size: 11.5px':'font-size: 18px',
        'font-size: 12px':'font-size: 19px',
        'font-size: 13px':'font-size: 20px',
        'font-size: 13.5px':'font-size: 20px',
        'font-size: 14px':'font-size: 21px',
        'font-size: 10px':'font-size: 16px',
        'line-height: 1.55':'line-height: 1.4',
        'padding: 6px 12px':'padding: 2px 12px',
    }.items():text=text.replace(a,b)
    for a,b in [('#f8fafc','#ffffff'),('#0f172a','#000000'),('#64748b','#000000'),('#cbd5e1','#b8b8b8')]:text=text.replace(a,b)
    text=text.replace('</style>', '.diagram-header { height: 59px; } .diagram-subtitle { display: none; }</style>')
    path=Path(__file__).with_name('figure_tech_route_corrected.html')
    path.write_text(text,encoding='utf-8')
    print(path)

def zip_dir(src,path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(src).as_posix())

def plain(node):return ''.join(ch.data for n in node.getElementsByTagName('w:t') for ch in n.childNodes if ch.nodeType in (ch.TEXT_NODE,ch.CDATA_SECTION_NODE))
def rpr_for(p, superscript=False):
    runs=p.getElementsByTagName('w:r')
    for r in runs:
        if r.getElementsByTagName('w:t') and not r.getElementsByTagName('w:vertAlign'):
            pr=r.getElementsByTagName('w:rPr')
            out=pr[0].toxml() if pr else '<w:rPr/>'
            if superscript:
                out=out.replace('</w:rPr>','<w:vertAlign w:val="superscript"/></w:rPr>') if '</w:rPr>' in out else '<w:rPr><w:vertAlign w:val="superscript"/></w:rPr>'
            return out
    return '<w:rPr/>'
def run(text,pr,deleted=False):return f'<w:r>{pr}<{"w:delText" if deleted else "w:t"} xml:space="preserve">{escape(text)}</{"w:delText" if deleted else "w:t"}></w:r>'
def runs(text,p,deleted=False):
    parts=re.split(r'(\[\d+(?:[,-]\d+)*\])',text)
    return ''.join(run(x,rpr_for(p,bool(re.fullmatch(r'\[\d+(?:[,-]\d+)*\]',x))),deleted) for x in parts if x)

def main():
    REVIEW.mkdir(exist_ok=True)
    tempfile.tempdir=str(REVIEW)  # official library temporary copy stays in authorized directory
    original_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    expected=json.loads((REVIEW/'original-sha256.json').read_text(encoding='utf-8'))
    assert original_hash==expected[str(SOURCE.relative_to(ROOT))], 'Source changed since evidence capture; stop.'
    work=REVIEW/'unpacked-working'
    if work.exists():shutil.rmtree(work)
    with zipfile.ZipFile(SOURCE) as z:z.extractall(work)
    doc=Document(work,author='校正审阅',initials='REV',track_revisions=True,rsid='20260928')
    ed=doc['word/document.xml']; paragraphs=list(ed.dom.getElementsByTagName('w:p'))
    source_paras=json.loads((REVIEW/'source-paragraphs.json').read_text(encoding='utf-8'))
    records=[]
    for i,item in CHANGES.items():
        p=paragraphs[i]; old=plain(p)
        assert old==source_paras[i]['text'],f'Paragraph {i} mismatch'
        new=item['new']; pr=p.getElementsByTagName('w:pPr'); ppr=pr[0].toxml() if pr else ''
        content=[]
        for op,a,b,x,y in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
            if op=='equal':content.append(runs(old[a:b],p))
            else:
                if a!=b:content.append('<w:del>'+runs(old[a:b],p,True)+'</w:del>')
                if x!=y:content.append('<w:ins>'+runs(new[x:y],p)+'</w:ins>')
        ed.replace_node(p,'<w:p>'+ppr+''.join(content)+'</w:p>')
        records.append({'paragraph_index':i,'old':old,'new':new,'basis':item['basis']})
    # Original seven cover spacers each occupy 20.7pt in Word. Consolidate to
    # one exact-height 144.9pt spacer: same cover geometry, no blank-page hazard.
    cover_spacer=paragraphs[7]
    ed.replace_node(cover_spacer,'<w:p><w:pPr><w:spacing w:line="2898" w:lineRule="exact"/></w:pPr></w:p>')
    for p in paragraphs[8:14]:p.parentNode.removeChild(p)
    records.append({'paragraph_index':'封面7–13','old':'7个连续空白段落，用于封面留白','new':'合并为单个144.9pt等高留白段，保留原封面视觉位置；修复postcheck blank-pages错误','basis':'原稿Word段落纵坐标330.40至477.30pt；7×20.7=144.9pt。并非重建封面。'})
    # Visual-gate repair: keep each bibliography entry on a single page.
    for p in paragraphs[198:212]:
        ppr=p.getElementsByTagName('w:pPr')[0]
        if not ppr.getElementsByTagName('w:keepLines'):
            node=ed.dom.createElement('w:keepLines'); ppr.insertBefore(node,ppr.firstChild)
    records.append({'paragraph_index':'参考文献198–211','old':'参考文献[8]可跨页拆分，末页出现孤立年份2021.','new':'所有参考文献段落设置keepLines，避免单条条目跨页孤行','basis':'主代理唯一视觉gate：p11–12发现[8]末行跨页；定向版式修复。'})
    figure=REVIEW/'figure_tech_route_校正版.png'
    assert figure.exists(),'Render corrected HTML into review PNG first.'
    media=list((doc.unpacked_path/'word/media').glob('*.png')); assert len(media)==1
    shutil.copyfile(figure,media[0])
    # Preserve original image display dimensions (545 x 379) and canvas ratio.
    doc.save()
    zip_dir(work,REVIEW/'选题报告_文字修订审计.docx')
    # Clean academic deliverable, while tracked copy and exact old/new ledger retain audit trail.
    ed=doc['word/document.xml']
    for n in list(ed.dom.getElementsByTagName('w:del')):n.parentNode.removeChild(n)
    for n in list(ed.dom.getElementsByTagName('w:ins')):
        parent=n.parentNode
        for ch in list(n.childNodes):parent.insertBefore(ch,n)
        parent.removeChild(n)
    settings=doc['word/settings.xml']
    for n in list(settings.dom.getElementsByTagName('w:trackRevisions')):n.parentNode.removeChild(n)
    doc.save()
    zip_dir(work,TARGET)
    records.append({'paragraph_index':'图1','old':'跨任务区域级负面经验；搜索/对话/停止协议；所有数据集与DynCur均列作基线','new':'本课题拟议多智能体；episode内记忆、独立测试重置；四动作无stop、stop扩展另测；防泄漏；训练式基线须同协议复现','basis':A+M+L+P})
    (REVIEW/'corrections.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['选题报告校正说明与逐项清单','\n一、校正范围与文件保护','在原DOCX的OOXML副本上校正，保留封面、章节顺序、字体、黑白布局、段落样式、表格与图1位置。正文净稿接受文字修订；修订审计副本和本清单保留改动轨迹。原件及原PDF、原图均不覆盖。','原始 build_report.js 仍为旧版，包含已纠正的事实错误，且输出路径会覆盖原稿。不得运行它重生或覆盖校正版；请使用 project/src/documents/correct_report.py。',f'原DOCX SHA-256：{original_hash}','\n二、证据说明','页码均为本地PDF从1起的物理页码，而非文档印刷页码。官方代码仓库origin=git@github.com:mvrl/GOMAA-Geo.git，HEAD=1faa1eee1608529eae95cc0af1d76330c9e46e60；本次只读源码，未运行模型训练或复现实验。', '本地论文与代码完整路径根目录：'+str(ROOT), 'GOMAA-Geo本地PDF为arXiv v1并写Preprint/Under review；NeurIPS 2024由官方README第2、23行佐证。GeoExplorer本地PDF为arXiv v1，ICCV书目信息此处保留原稿，未外部独立核验。','\n三、逐项校正（段落索引为原word/document.xml中w:p从0计数，包含表格）']
    for j,r in enumerate(records,1):lines.extend([f'\n{j}. 原段落/图：{r["paragraph_index"]}','原说法：'+r['old'],'改正：'+r['new'],'依据：'+r['basis']])
    lines.extend(['\n四、未解决限制与边界', '1. 本地Masa 137/4/10与论文1188的来源差异未查明；不能推断为作者内部重切。GeoExplorer补充材料所列832+178+179=1189，与1188有一幅计数不一致，须回溯原始清单。DynCur第4页提到137幅Masa训练图，并不能证明本地数据与该论文一一对应。', '2. DynCur的距离门控、PBRS、PPO以及推理不读距离已在本地论文第2–4页核实；“全面超越”、普适泛化优势与本课题可直接提升均不写入。未核验其实现或重跑其数值。', '3. 参考文献[4–14]沿用原稿，除与三篇核心论文中可交叉核对的部分外，未进行完整外部书目核验；本次不新增未经核实的出版信息。', '4. 原封面成员、学号、导师占位及分工表留白按要求保留，须由小组后续填写；正式报告未写当前代码实现进度。', '5. LibreOffice 26.8.0已从清华镜像完整下载并实际执行msiexec安装，安装退出1603；libreoffice-install.log第301–305行明确缺少管理员权限。未安装成功，故不做PATH注册。仅在该失败后使用独立Word实例只读打开校正版导出PDF；引擎差异可能影响分页与兼容性。', '6. postcheck、PDF页数和visual-judge结果见校正审阅/quality-status.json及对应日志；本清单不把程序检查替代视觉验收。', '\n五、复现', '脚本：'+str(Path(__file__)), '先运行 --prepare-figure；渲染同目录figure_tech_route_corrected.html到选题报告相关/校正审阅/figure_tech_route_校正版.png，再运行主脚本；最后运行export_review.py。原始build_report.js不调用。'])
    (BASE/'选题报告_校正说明.txt').write_text('\n'.join(lines),encoding='utf-8')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original_hash
    with zipfile.ZipFile(SOURCE) as a,zipfile.ZipFile(TARGET) as b:
        checks={name:a.read(name)==b.read(name) for name in ['word/styles.xml','word/numbering.xml','word/footer1.xml','word/footer2.xml'] if name in a.namelist()}
    (REVIEW/'preservation-check.json').write_text(json.dumps({'source_sha256_unchanged':True,'unchanged_parts':checks,'text_changes':len(CHANGES),'image_changes':1},ensure_ascii=False,indent=2),encoding='utf-8')
    del doc
    shutil.rmtree(work)
    print('OUTPUT',TARGET,'CHANGES',len(CHANGES))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-figure',action='store_true');args=ap.parse_args()
    prepare_figure() if args.prepare_figure else main()
