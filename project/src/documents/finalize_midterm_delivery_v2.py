"""Navigation, exact-duplicate cleanup and preservation audit for course delivery."""
from pathlib import Path
from hashlib import sha256
from urllib.parse import quote,unquote
from datetime import datetime,timezone
import argparse,json,re,zipfile

ROOT=Path(__file__).resolve().parents[3]
NAV=ROOT/'项目导航';RECORD=NAV/'整理记录_2026-10-02_v1'
TITLE='基于主动探索的视觉地理定位方法设计与实现'
OUT=ROOT/'中期报告相关'
ALLOWED={'README.md',f'中期报告相关/中期报告_{TITLE}.docx','中期报告相关/编制依据.json','中期报告相关/编制说明.md'}
KEEP=f'选题报告相关/最新版选题报告/18-{TITLE}-陈德昕-李世显-陈龙辉.pdf'
DROP=f'选题报告相关/最新版选题报告/选题报告_{TITLE}_模板版 - 副本.pdf'

def read(p):return json.loads(p.read_text('utf8'))
def digest(p):
    h=sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def new_json(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def navigation():
    if (NAV/'README.md').exists():raise ValueError('navigation already authored; revise in a new version')
    (ROOT/'README.md').write_text('''# 基于主动探索的视觉地理定位方法设计与实现

更新至2026年10月2日。接手项目先读本文件，再查看对应批次协议。当前已完成环境、统一评测与单智能体主线验收，进入阶段收尾；原S2/S3/S4及SwissView同模态迁移有效。继续训练候选的新源文件SR优势不足，未替换默认。

## 先找这些文件

| 用途 | 当前入口 |
|---|---|
| 快速了解方法、阶段与边界 | [当前进度](项目导航/当前进度.md) |
| 查实验、报告、判定和复核 | [全部58批实验索引](项目导航/实验索引.md) |
| 提交中期检查 | [中期报告与材料](中期报告相关/README.md) |
| 汇报与答辩 | [12页可编辑PPT](中期报告相关/答辩材料/中期汇报_基于主动探索的视觉地理定位方法设计与实现.pptx) / [演讲提示与问答](中期报告相关/答辩材料/演讲提示与答辩问答.md) |
| 直接使用科研图 | [科研图目录、图注与来源](绘图/README.md) |
| 继续开发与复现 | [目录结构与维护](项目导航/目录结构与维护.md) / [冻结默认](project/local_policy_default.json) |
| 查早期记录和完整目录登记 | [历史开发记录](项目导航/历史开发记录.md) |
| 查本次整理与恢复 | [整理记录](项目导航/整理记录_2026-10-02_v1/) |

## 当前方法和正式成绩

方法为训练过的单导航Agent：冻结Sat2Cap编码器、Small256 NoTarget探索器、EdgeTargetCue视觉线索头与合法动作门控。NoTarget仅遮蔽探索器的目标通道，视觉头仍读取给定目标图与当前图；整体系统有训练。参考框架与正式要求见[对应说明](选题报告相关/任务要求与论文框架对应说明_v1.md)。

| 已通过验收 | 源文件 / 每权重任务 | 协议 | 三权重平均SR | 平均SG |
|---|---|---|---:|---:|
| [S2稳定验证](DATA/processed_data/Masa/评测结果/边缘线索S2正式复验_v1/验收结论.json) | 4 / 100 | 5×5/B10 | 89.33% | 0.320 |
| [S3独立源文件确认](DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1/验收结论.json) | 10 / 250 | 5×5/B10 | 86.80% | 0.300 |
| [S4密度扩展](DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1/验收结论.json) | 20已见 / 500 | 10×10/B20 | 77.13% | 1.201 |
| [SwissView同模态迁移](DATA/processed_data/SwissView/评测结果/五乘五正式迁移_v1/验收结论.json) | 20当批新源 / 500 | 5×5/B10 | 86.80% | 0.381 |

SR按计划题数计算预算内成功；SG为全部终局曼哈顿距离均值，成功记0。以上为不同协议和队列，不能连接为累计提升曲线、合并成整体SR或直接与论文不同协议的数字相减。

最新Continue5冻结目标证据通过，但固定20新SwissView源文件的10×10仅SR+0.60个百分点、1/3权重正收益，源文件95%差区间[−0.87,+2.27]点；SG由1.230降为0.868。按预登记未通过默认升级，保留权重和低SG结果。详见[独立确认报告](DATA/processed_data/SwissView/评测结果/继续训练独立源图确认_v1/独立源图确认报告.md)。云端多角色、零调用账本、保护规则及旋转支线的负结果按批保留。

## 数据与结论范围

Masa本地151源文件，划分137训练/4验证/10测试；探索器109拟合/28已知开发留出，视觉头87拟合/22校准。SwissView100为100张航拍子集，累计48源文件已跑导航、52尚未跑，见[最新使用状态](选题报告相关/SwissView源文件使用状态_2026-10-02_v1.json)。重复题、训练权重和双网格均不增加独立地图。

当前10×10是在同一原图足迹内增加切图密度，尚未扩大实际覆盖面积。已验证范围是连续航拍目标的本地切图仿真；源文件隔离不证明严格地理无重叠或预训练编码器未见。任意角度、真实异时、跨视角/跨模态目标、更大实际区域和无人机实飞尚待验证。

## 目录用途与登记

| 目录 / 路径模板 | 用途 |
|---|---|
| `project/src/{agents,env,data,train,eval,tests,documents}/` | 模型与策略、交互环境、处理、训练、评测、测试和材料脚本；原脚本保留 |
| `project/` | 默认配置、依赖说明及云端服务配置；`.env*`不进入报告、分享或归档包 |
| `DATA/raw_data/` | 原始数据，只读，不搬移 |
| `DATA/processed_data/{Masa,SwissView}/` | 派生图块、特征、任务及源文件记录 |
| `DATA/processed_data/{Masa,SwissView}/{训练结果,评测结果}/<批次>/` | 原始批次的模型、报告、逐步日志与审计；具体登记见历史记录和实验索引 |
| `models/` | 预训练编码器等权重，不清理 |
| `选题报告相关/` | 选题报告、运行前方案、验收标准、源使用状态与历史阶段快照 |
| `选题报告相关/最新版选题报告/` | 用户填写版本与历史模板副本；[当前版本说明](选题报告相关/最新版选题报告/README.md) |
| `中期报告相关/` | 当前模板版中期报告、阶段提升总结v2、编制依据 |
| `中期报告相关/历史版本/` | 修订前中期报告，历史状态不回写 |
| `中期报告相关/排版审阅/2026-10-02_v2/` | Word全页校样与渲染、视觉核验记录 |
| `中期报告相关/答辩材料/`及`排版审阅/` | PPT、演讲问答与PPT逐页校样 |
| `项目导航/` | 当前进度、58批实验入口、维护说明与历史README |
| `项目导航/整理记录_2026-10-02_v1/{恢复备份,源码快照,审阅}/` | 文件清单、可恢复清理、材料脚本快照与报告/汇报衍生图 |
| `绘图/{数据结果图,Agent架构图,绘图数据,审阅}/` | 35张实验结果图、2张架构/边界图、图注与原始来源；原图不改 |
| `论文/`、`调研/`、`research.md` | 参考论文与已完成专项调研；不当作本项目实验成绩 |
| `课程相关报告模版/` | 教师模板原件，只读 |
| `视频/` | 已有讲解视频工程，入口见[说明](视频/README.md) |
| `GOMAA-Geo/` | 第三方参考仓库，不在其中开发 |

旧目录的细项登记保存在历史开发记录。本次增加导航层和当前材料，没有迁移实验、权重或大型数据。相同字节的源码快照、已冻结报告副本承担批次证据，不按重复删除。

## 协作者必须遵守

1. 源代码写入`project/src/`；非代码目录优先中文命名。每新增目录先在本README登记用途，自动子目录可用路径模板登记。
2. 原始数据只读；派生数据放在对应数据集目录，报告放在选题/中期/导航目录。不得改写历史批次及其判定。
3. 正式任务允许训练小模型，多Agent属于可选拓展。GOMAA-Geo与GeoExplorer均包含训练，借鉴机制不等于完整复现。
4. Agent只读取公开Observation/StepInfo：目标图、当前图、公开位置、预算、访问历史。答案、目标索引、真实距离、全图、未访问图块/特征和评测日志不能进入执行策略；Python私有字段不是恶意代码沙箱。
5. 四方向移动，越界原地并耗一步；预算内首次到达成功，最后一步到达优先判成功。无`stop`扩展动作。跨episode记忆默认清空。
6. 开发在训练/验证源进行；测试和新源文件在配置冻结后使用。最新20确认源已消费，不可重新声明未见；不用开发分数代替独立确认。
7. 固定任务、预算、输入权限与全部权重，保留目标对照、损伤/恢复和复核。门槛按对应预登记，不把SR60%当所有阶段通行证。
8. 删除仅限已核查的冗余文件与可再生缓存，先备份并登记路径/哈希。失败实验、源码快照、数据和权重不作为清理对象。
9. Git保留用户既有改动；本次未提交或推送。提交前排除凭据、原始数据和大型派生文件，按实际任务范围审查。

## 运行与继续工作

环境基础层依赖见[基础依赖](project/requirements-foundation.txt)，云端接口见[接口依赖](project/requirements-vlm.txt)；本地主方法另使用现有PyTorch及编码器依赖，无需Ollama安装。已验收冻结配置仍在原路径，后续S3/S4判定由各批验收文件记录，不回写S2时的配置历史字段。

各批复现入口、完整源码快照及所需数据以实验索引和该批执行协议为准。历史评测器默认输出已经存在，禁止直接重跑覆盖。后续实验先登记新版本目录和单因素预算，验收后再决定是否换默认。材料生成/整理入口在`project/src/documents/`；本次整理记录已封存，不重复运行`prepare/index/cleanup`，旧报告生成器也不应覆盖当前稿。

近期交付是完善中期检查与答辩材料。下一研究因素应依据已保存的恢复/损伤和剩余预算机会单独选择；本轮未启动新训练、API请求或导航实验。
''','utf8')
    (NAV/'README.md').write_text('''# 项目导航

2026-10-02当前阅读入口。目录整理采用索引与当前摘要，原代码、数据、权重、实验报告及日志保留原路径。

| 你要做什么 | 入口 |
|---|---|
| 先看项目到哪一步 | [当前进度与指标](当前进度.md) / [根目录总览](../README.md) |
| 看主方法和近期结论 | [中期报告、阶段总结与汇报](../中期报告相关/README.md) |
| 找某次实验及它的证据 | [58批实验索引](实验索引.md) / [机器可读索引](实验索引.json) |
| 拿论文/PPT图片 | [科研图目录](../绘图/README.md) |
| 继续运行或维护代码 | [目录结构与维护](目录结构与维护.md) |
| 看一路开发记录 | [历史开发记录](历史开发记录.md) |
| 查看清理是否可恢复 | [整理记录](整理记录_2026-10-02_v1/) |

历史日志里的“尚未通过”“预留”是写入当时的状态；当前以本导航及已审计验收为准。指标不能脱离同题、协议和源文件范围单独引用。
''','utf8')
    (NAV/'当前进度.md').write_text('''# 当前进度与阶段结论

截至2026-10-02，环境基础、统一评测和最小VLM接入已完成。本地训练过的单Agent主线通过正式S2/S3、10×10密度扩展S4及SwissView5×5同模态迁移。当前收束已有实验并完善课程材料。

## 默认与验收

冻结Sat2Cap + Small256 NoTarget GRU + EdgeTargetCue，阈值0.50，合法且未访问的最高方向线索才接管。NoTarget仅指探索器目标通道，系统视觉头读取目标图。三训练权重都纳入评测，无集成或挑种子。见[冻结默认](../project/local_policy_default.json)。默认内的独立确认历史字段停留在写入时；后续验收以每批判定为准，不因此修改冻结配置。

| 阶段 | 源文件 / 每权重任务 | SR | SG | 原判定 |
|---|---|---:|---:|---|
| S2，5×5/B10 | 4验证 / 100 | 89.33% | 0.320 | [通过](../DATA/processed_data/Masa/评测结果/边缘线索S2正式复验_v1/验收结论.json) |
| S3，5×5/B10 | 10测试 / 250 | 86.80% | 0.300 | [通过](../DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1/验收结论.json) |
| S4，10×10/B20 | 20已见 / 500 | 77.13% | 1.201 | [密度扩展通过](../DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1/验收结论.json) |
| SwissView，5×5/B10 | 20当批新源 / 500 | 86.80% | 0.381 | [同模态迁移通过](../DATA/processed_data/SwissView/评测结果/五乘五正式迁移_v1/验收结论.json) |

均为三训练权重平均。不同队列分别展示；SR按计划题数，SG为全部终局距离，成功为0。

## 最近继续训练为什么停止

Continue5与Adapt10保持Small256和目标头，每权重同81920动作/1024更新；开发10×10为79.73%/0.823与80.00%/1.059。适配相对继续训练仅+0.27点、SG更差，未升级。

Continue5的[冻结目标证据](../DATA/processed_data/Masa/评测结果/继续训练冻结目标证据复验_v1/验收结论.json)通过：真实79.73%，禁用59.40%、均值52.20%、错误目标57.20%；目标贡献不等于优于旧默认。

固定新20SwissView源文件后，逐题对照如下：

| 新源协议 | 原M0 SR / SG | Continue5 SR / SG |
|---|---:|---:|
| 5×5/B10 | 86.73% / 0.334 | 87.40% / 0.310 |
| 10×10/B20 | 72.07% / 1.230 | 72.67% / 0.868 |

10×10仅+0.60点、1/3权重正、源文件95%差区间[−0.87,+2.27]点；恢复185、损伤176、净9/1500。5×5保护通过，10×10升级未过；[独立判定](../DATA/processed_data/SwissView/评测结果/继续训练独立源图确认_v1/验收结论.json)保持默认。最近两批12000轨迹/181282动作重放通过，未在失败后追加条件对照或调参追过线。

## 其他路线和当前边界

同预算云端反思/双角色没有足够SR增益，支线收束；零调用账本完整队列+1.27点，保留低成本备选；保护、邻域覆盖、旋转负样本等候选未稳定升级。完整正负结果见[实验索引](实验索引.md)。

Masa151文件为137/4/10，探索器109拟合/28开发，视觉头87拟合/22校准。SwissView累计48已导航、52尚未，最新40—59号已消费；见[使用状态](../选题报告相关/SwissView源文件使用状态_2026-10-02_v1.json)。两个网格共用20源，不是40独立地图。10×10只增加原图内密度；真实异时、任意角度、跨视角/跨模态、更大实际范围和实飞未完成。

本轮只整理材料，不启动新因素。下一项先依据新源成功损伤与预算机会确定可检验假设，再冻结单因素、同预算对照和停止条件。已过主线与候选失败分别保留。
''','utf8')
    (NAV/'目录结构与维护.md').write_text('''# 目录结构与维护

## 阅读路径

根目录README说明当前事实；本目录提供完整实验索引和维护说明；中期报告目录提供课程材料；绘图目录提供图、图注、数据和来源。历史开发记录保留早期状态与完整目录登记，不能拿历史“尚未通过”当当前结论。

```text
大数据工程/
├─ README.md                    当前总入口与目录登记
├─ 项目导航/                     当前进度、58批索引、维护、历史记录
│  └─ 整理记录_2026-10-02_v1/     清单、恢复备份、源码快照、审阅
├─ 中期报告相关/                 当前中期报告、阶段提升总结v2、编制依据
│  ├─ 答辩材料/                  12页PPT、演讲问答、内部校样
│  ├─ 历史版本/                  修订前报告
│  └─ 排版审阅/                  Word内部校样与视觉核验
├─ 绘图/                         数据结果图、Agent架构图、图注和来源
├─ 选题报告相关/                 用户报告、预登记、标准、历史阶段快照
├─ project/src/                  原Agent/环境/训练/评测/测试/文档代码
├─ DATA/raw_data/                原始数据，只读
├─ DATA/processed_data/          派生输入与原始实验记录，全部保留原址
├─ models/                       预训练权重，保留
└─ 论文/、调研/、视频/、GOMAA-Geo/  参考资料与已有工程
```

## 维护与复现

1. 新目录在根README先登记用途；非代码目录优先中文。新实验使用新的批次版本，不回写旧协议、源码快照、分数或判定。
2. 查复现入口时先打开实验索引的原目录和执行协议。原始任务、数据、均值、权重、源码和哈希绑定共同构成证据，不能只复制汇总表重跑。
3. 默认仍为`project/local_policy_default.json`引用的三Small256探索器与三Edge视觉头。该文件在S2冻结，之后S3/S4验收存于各批；不能因配置的历史确认字段而否认后续验收。
4. 四方向/边界耗步/最后一步成功规则与Observation信息权限保持；训练真值不进入推理。测试、新源文件只用于冻结配置后的确认，跨episode状态默认清空。
5. 旧批次输出已经存在，运行器拒绝覆盖。基础环境测试与论文效果区分，规则成绩不是VLM成绩；修改导航代码应另立实验和适当回归检查。
6. 材料编制脚本为`project/src/documents/build_midterm_materials_v2.py`，Office导出为`render_midterm_materials_v2.py`。本批已封存，后续生成另登记材料版本；不要让旧`build_midterm_report.py`或旧选题生成器覆盖用户填写的新稿。
7. API配置、`.env*`、凭据不进入文档、源码快照或共享包；当前本地验收不替代任何云端模型验收。用户弃用的8790三选项不继续调用。

## 清理边界与恢复

本轮没有迁移数据、权重、代码或实验目录。可再生`project/src/**/__pycache__/*.pyc`在归档校验后删除；确认同字节且无功能引用的选题PDF副本在保留正式命名原件后删除。所有删除见[整理记录](整理记录_2026-10-02_v1/)。

已冻结源码快照、历史报告、失败实验和用于跨批SHA复用的结果即使字节相同也保留。`node_modules`、第三方仓库和预训练模型不在本次清理范围。报告个人信息取自用户填写的版本，原用户文档未改。

恢复前查看`恢复备份/删除文件.zip`及`删除重复副本.zip`中的原相对路径和清理清单。将所需单个条目恢复到工作区对应路径并核对SHA；不要整包覆盖当前报告。旧材料另在`恢复备份/修订前材料.zip`，修订前中期报告有独立历史副本。
''','utf8')
    (OUT/'README.md').write_text('''# 中期检查与汇报材料

当前阅读版：2026-10-02。材料使用相同源文件范围、SR/SG口径和验收判定；教师模板的封面、人员表、四个栏目与A4格式保留。作者和导师取自用户填写的选题报告。

| 用途 | 当前文件 |
|---|---|
| 中期检查正式正文（11页） | [中期报告Word](中期报告_基于主动探索的视觉地理定位方法设计与实现.docx) |
| 快速回顾方法提升（2页） | [阶段提升总结v2 Word](阶段提升总结_v2.docx) / [Markdown](阶段提升总结_v2.md) |
| 汇报展示（12页） | [可编辑中期汇报PPT](答辩材料/中期汇报_基于主动探索的视觉地理定位方法设计与实现.pptx) |
| 讲解与常见答辩问题 | [演讲提示与答辩问答](答辩材料/演讲提示与答辩问答.md) |
| 核对资料和结果来源 | [编制依据、SHA与判定](编制依据.json) / [编制说明](编制说明.md) |
| 找科研图及完整实验 | [科研图](../绘图/README.md) / [58批实验索引](../项目导航/实验索引.md) |

## 统一结论

原边缘线索主方法通过S2/S3/S4密度扩展与SwissView同模态迁移。Continue5目标图证据通过，但独立新源10×10仅SR+0.60点、1/3权重正，未升级；原默认保持。当前是训练过的单Agent模块系统，未证明真正多Agent收益或无人机实飞。

## 历史材料和内部校样

[修订前中期报告](历史版本/2026-09-30_中期报告_修订前.docx)与[阶段总结v1](阶段提升总结_v1.docx)保留当时状态，v1不作为当前进度稿。旧`阶段提升总结审阅/`等校样也保留。

同名PDF为本机Office导出的固定版面参考；页面PNG、全文校样和渲染/视觉核验记录在`排版审阅/2026-10-02_v2/`与`答辩材料/排版审阅/`，通常不随正文提交。生成材料没有改动用户填写的选题报告、教师模板或历史实验。
''','utf8')
    (ROOT/'选题报告相关/最新版选题报告/README.md').write_text(f'''# 当前选题报告取用说明

2026-10-02核对。当前用户已填写人员和导师信息的版本：

- [Word](18-{TITLE}-陈德昕-李世显-陈龙辉.docx)
- [对应PDF](18-{TITLE}-陈德昕-李世显-陈龙辉.pdf)

本轮仅核对并引用其中的人员表，未修改这些文件。通用模板名DOCX保留现有用户改动。旧`说明.txt`是2026-09-29的历史复制流程，所指通用PDF已在本轮开始前缺失；不要依旧流程重生成覆盖用户填写内容。

与上述PDF字节相同的“模板版 - 副本.pdf”经引用检查、备份与SHA核对后清理。详见[整理记录](../../项目导航/整理记录_2026-10-02_v1/)。
''','utf8')
    print(dict(navigation_authored=6),flush=True)

def duplicate_cleanup():
    p=(ROOT/DROP).resolve();keeper=(ROOT/KEEP).resolve()
    if ROOT.resolve() not in p.parents or p.parent!=keeper.parent:raise ValueError('duplicate escaped named directory')
    if not keeper.is_file() or not p.is_file() or digest(p)!=digest(keeper):raise ValueError('exact named duplicate required')
    needles=[p.name,quote(p.name),DROP,DROP.replace('/',chr(92))];hits=[];bookkeeping=[]
    before=read(RECORD/'整理前文件清单.json');reviewed=0
    for row in before['files']:
        q=ROOT/row['path']
        if not q.is_file() or q.stat().st_size>2*1024*1024:continue
        if q.suffix.lower() in {'.md','.py','.txt','.json','.js','.cjs','.ps1','.html','.xml'}:
            text=q.read_text('utf8',errors='replace');reviewed+=1
            if any(s in text for s in needles):
                if q.name=='原始文件保护清单.json' and '阶段收尾_v1' in q.parts:bookkeeping.append(rel(q))
                else:hits.append(rel(q))
        elif q.suffix.lower() in {'.docx','.pptx','.xlsx'}:
            with zipfile.ZipFile(q) as z:
                for name in z.namelist():
                    if name.endswith('.rels'):
                        text=z.read(name).decode('utf8',errors='replace');reviewed+=1
                        if any(s in unquote(text) for s in needles):hits.append(rel(q)+'!'+name)
    if hits:raise ValueError('duplicate has functional references: '+str(hits))
    h=digest(p);archive=RECORD/'恢复备份/删除重复副本.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:z.write(p,DROP)
    with zipfile.ZipFile(archive) as z:
        if sha256(z.read(DROP)).hexdigest()!=h:raise ValueError('duplicate archive mismatch')
    size=p.stat().st_size
    if digest(p)!=h or digest(keeper)!=h:raise ValueError('duplicate drift before removal')
    p.unlink()
    receipt=dict(deleted_path=DROP,keeper_path=KEEP,sha256=h,deleted_bytes=size,recovery_archive=rel(archive),recovery_sha256=digest(archive),
        functional_references=[],small_text_and_relationship_entries_checked=reviewed,immutable_inventory_mentions_retained=bookkeeping,
        original_user_docx_unchanged=True)
    new_json(RECORD/'重复副本清理.json',receipt);print(dict(duplicate_deleted=1,bytes=size,reference_entries_checked=reviewed),flush=True)

def verify():
    before=read(RECORD/'整理前文件清单.json');deleted=read(RECORD/'重复副本清理.json');changed=[];verified=0;hashed=0;authorized=[]
    assert deleted['deleted_path']==DROP and digest(ROOT/KEEP)==deleted['sha256'] and not (ROOT/DROP).exists()
    with zipfile.ZipFile(ROOT/deleted['recovery_archive']) as z:assert sha256(z.read(DROP)).hexdigest()==deleted['sha256']
    for row in before['files']:
        p=ROOT/row['path']
        if row['path']==DROP:continue
        if row['path'] in ALLOWED:
            if not p.is_file():raise ValueError('required output missing')
            authorized.append(row['path']);continue
        if not p.is_file():changed.append(dict(path=row['path'],reason='missing'));continue
        st=p.stat()
        if (st.st_size,st.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):changed.append(dict(path=row['path'],reason='metadata drift'))
        if 'sha256' in row:
            hashed+=1
            if digest(p)!=row['sha256']:changed.append(dict(path=row['path'],reason='SHA drift'))
        verified+=1
    if changed:raise ValueError('protected changes: '+str(changed[:15]))
    links=0;docs=[ROOT/'README.md',NAV/'README.md',NAV/'实验索引.md',NAV/'目录结构与维护.md',NAV/'当前进度.md',OUT/'README.md',ROOT/'选题报告相关/最新版选题报告/README.md']
    for p in docs:
        for target in re.findall(r'\]\(([^\n)]+)\)',p.read_text('utf8')):
            target=target.strip('<>')
            if re.match(r'^(https?://|#)',target):continue
            if not (p.parent/target).exists():raise ValueError('broken navigation link: '+str(p)+' '+target)
            links+=1
    basis=read(OUT/'编制依据.json')
    for path,h in {**basis['source_sha256'],**basis['revised_outputs']}.items():assert digest(ROOT/path)==h,path
    default=read(ROOT/'project/local_policy_default.json');model_bindings=0
    for row in default['checkpoints']+default['cue_heads']+list(default['means'].values()):
        assert digest(ROOT/row['path'])==row['sha256'],row['path'];model_bindings+=1
    cache=read(RECORD/'清理执行清单.json')
    with zipfile.ZipFile(ROOT/cache['recovery_archive']) as z:
        for row in cache['deleted_files']:
            assert not (ROOT/row['path']).exists(),row['path']
            assert sha256(z.read(row['path'])).hexdigest()==row['current_sha256']
    qa=read(OUT/'排版审阅/2026-10-02_v2/视觉核验.json');assert qa['passed']
    renders=read(OUT/'排版审阅/2026-10-02_v2/渲染收据.json');assert not renders['visual_review_pending']
    for item in renders['artifacts'].values():
        p=ROOT/item['pdf'];assert p.is_file()
        src=p.with_suffix('.pptx' if '中期汇报' in p.name else '.docx');assert digest(src)==item['source_sha256']
        for page in item['pages']:assert (ROOT/page['path']).is_file()
    for path,h in qa['rendered_files_sha256'].items():assert digest(ROOT/path)==h,path
    scripts=['project_materials_organizer.py','build_midterm_materials_v2.py','render_midterm_materials_v2.py','finalize_midterm_delivery_v2.py']
    for name in scripts:assert digest(ROOT/'project/src/documents'/name)==digest(RECORD/'源码快照'/name)
    final=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),initial_files=len(before['files']),protected_files_verified=verified,
        protected_files_sha256_verified=hashed,authorized_modified_materials=authorized,navigation_links_verified=links,
        experimental_batches=len(read(NAV/'实验索引.json')['batches']),deleted_duplicate_files=1,deleted_bytecode_files=cache['deleted_count'],
        deleted_bytes=cache['deleted_bytes']+deleted['deleted_bytes'],recovery_archives_verified=True,all_output_pages_reviewed=25,
        new_training_steps=0,cloud_calls=0,new_navigation_evaluation=0,raw_data_models_experiments_original_source_scripts_unchanged=True,
        default_sha256=digest(ROOT/'project/local_policy_default.json'),default_model_and_mean_bindings_verified=model_bindings,
        large_files_protection='size+mtime; small evidence/source hashes and default referenced weights/means also checked')
    new_json(RECORD/'整理复核.json',final)
    print(final,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['navigation','duplicate','verify']);args=p.parse_args()
    {'navigation':navigation,'duplicate':duplicate_cleanup,'verify':verify}[args.mode]()
