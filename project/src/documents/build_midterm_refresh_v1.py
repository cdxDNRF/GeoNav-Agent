"""DOC-009 teacher-format midterm refresh from sealed results; zero model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT, WD_ROW_HEIGHT_RULE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'中期报告相关/中期成果更新_20261025'
REFERENCE=ROOT/'中期报告相关/排版审阅/中期报告_模板解析.docx'
FILLED=ROOT/'中期报告相关/中期报告_基于主动探索的视觉地理定位方法设计与实现.docx'
TITLE='基于主动探索的视觉地理定位方法设计与实现'
SOURCE_PATHS=[
 'DATA/processed_data/Masa/评测结果/边缘线索S2正式复验_v1/对照汇总.json',
 'DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1/对照汇总.json',
 'DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1/对照汇总.json',
 'DATA/processed_data/SwissView/评测结果/五乘五正式迁移_v1/对照汇总.json',
 'DATA/processed_data/MasaRoads/评测结果/径向保护独立新区域导航确认_v1/主对照/对照汇总.json',
 'DATA/processed_data/MassGIS/评测结果/正式十乘十十五乘十五确认_v1/主对照/汇总.json',
 'DATA/processed_data/OpenCVL/普通RGB空间小头固定开发对照_v1/主对照汇总.json',
 'DATA/processed_data/OpenCVL/普通RGB空间小头固定开发对照_v1/最终判定.json',
 'project/src/agents/edge_cue.py','project/src/agents/decoupled_cue.py',
 'project/src/agents/boundary_policy.py','project/local_policy_default.json',
 'project/defaults/masa_roads_grid10_radial_v1.json','docs/PROJECT_STATE.md',
 '平台/实验工作台_v4/发布修复与存储收束_v1/核验/正式服务复核.json',
 '中期报告相关/中期成果冻结版_v1/图表/01_M0方法与评测边界.png',
 '绘图/数据结果图/50_径向保护独立新区域的分层确认结果.png',
 '课程相关报告模版/综合工程设计-2-中期报告.doc',
]


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)


def prepare():
    assert load(OUT/'核验/用户授权.json')['approved']
    names=['Masa S2','Masa S3','Masa S4','SwissView']
    protocols=['5×5 / B10','5×5 / B10','10×10 / B20','5×5 / B10']
    rows=[]
    for i,p in enumerate(SOURCE_PATHS[:4]):
        m=load(ROOT/p)['averages']['Edge']['CueFull']
        rows.append(dict(name=names[i],protocol=protocols[i],sr=m['sr_mean'],sg=m['sg_mean'],sources=m['source_count'],
                         tasks_per_seed=m['planned_each'],weights=3,source=p,sg_unit='格'))
    assert abs(rows[1]['sr']-.868)<1e-10 and abs(rows[1]['sg']-.3)<1e-10
    a=load(ROOT/SOURCE_PATHS[4]);base=a['reference']['metrics'];cand=a['candidate']['metrics']
    assert a['recovered']-a['harmed']==cand['successes']-base['successes']==207
    area=dict(reference=base,candidate=cand,recovered=a['recovered'],harmed=a['harmed'],source=SOURCE_PATHS[4],
              ci95_sr_gain_pp=[7.07,11.02],regions=10,unique_tasks=750,weights=3,budget=20,cell_m=300,area_km2=9)
    mass=load(ROOT/SOURCE_PATHS[5]);new_domain=[]
    for g in ['10','15']:
        d=mass[g]['comparison']['candidate'];new_domain.append(dict(grid=int(g),sr=d['SR'],sg_m=d['SG_m'],planned=d['planned'],successes=d['successes']))
    c=load(ROOT/SOURCE_PATHS[6]);cv=c['development_means']['Spatial/Full']
    assert cv['positive_joint']==0 and cv['positive_n']==9 and not c['candidate_passed']
    cross=dict(positive_joint=0,positive_n=9,macro_joint=cv['source_macro_joint'],sources=3,
               independent_confirmation_sources=c['independent_confirmation_sources'],by_source=cv['by_source'],
               controls={k:{x:v[x] for x in ['positive_joint','positive_n','source_macro_joint']} for k,v in c['development_means'].items()},
               training_heads=18,updates=3600,training_uses=21600,nontraining_predictions=6750,source=SOURCE_PATHS[6])
    facts=dict(date='2026-10-25',title=TITLE,authors=['陈德昕','李世显','陈龙辉'],advisor='王笛',platform='4.13.1',
               aerial=rows,area=area,new_domain=new_domain,crossview=cross,
               parameters=dict(explorer=797701,edge_head=136837,task_total=934538,encoder='Sat2Cap frozen'),
               current_ground_status='人工连续观测与证据查阅；未完成可靠自主导航',
               assets=dict(method=SOURCE_PATHS[15],area=SOURCE_PATHS[16],platform='图表/平台当前工作台.png',ground='图表/地面目标与空间证据.png'))
    put(OUT/'核验/中期内容与证据.json',facts)
    extra=[REFERENCE.relative_to(ROOT).as_posix(),FILLED.relative_to(ROOT).as_posix()]
    put(OUT/'核验/输入绑定.json',dict(files=[dict(path=n,sha256=sha(ROOT/n)) for n in SOURCE_PATHS+extra],
        metrics_reused_only=True,model_calls=0,new_navigation=0,teacher_template_unchanged=True))
    print('Frozen content prepared:',len(rows),'aerial protocols;crossview 0/9;0 model calls')


def font(run,size=12,bold=False):
    run.font.name='宋体';run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor(0,0,0)
    rp=run._element.get_or_add_rPr();rf=rp.rFonts
    if rf is None:rf=OxmlElement('w:rFonts');rp.insert(0,rf)
    for attr in ['ascii','hAnsi','eastAsia','cs']:rf.set(qn('w:'+attr),'宋体')


def clean(cell):
    for ch in list(cell._tc):
        if ch.tag!=qn('w:tcPr'):cell._tc.remove(ch)
    cell.add_paragraph()


def para(cell,text,kind='body'):
    p=cell.paragraphs[0] if len(cell.paragraphs)==1 and not cell.paragraphs[0].text and not cell.tables else cell.add_paragraph()
    f=p.paragraph_format;f.space_before=Pt(5 if kind=='title' else 0);f.space_after=Pt(5 if kind=='title' else 4)
    f.line_spacing=1.5 if kind=='body' else 1.2;f.first_line_indent=Pt(24 if kind=='body' else 0)
    f.keep_with_next=kind in ['title','sub','caption'];f.widow_control=True
    flag=OxmlElement('w:snapToGrid');flag.set(qn('w:val'),'0');p._p.get_or_add_pPr().append(flag)
    if kind=='caption':p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    font(p.add_run(text),10.5 if kind in ['caption','note'] else 12,kind in ['title','sub','caption'])
    return p


def table(cell,rows,widths):
    t=cell.add_table(rows=len(rows),cols=len(widths));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for column,width in zip(t.columns,widths):column.width=Cm(width)
    borders=OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'5');e.set(qn('w:color'),'D9D9D9');borders.append(e)
    t._tbl.tblPr.append(borders)
    for i,row in enumerate(t.rows):
        row.height_rule=WD_ROW_HEIGHT_RULE.AUTO
        for k,c in enumerate(row.cells):
            c.width=Cm(widths[k]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=c._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
            for edge,val in [('top',70),('bottom',70),('left',90),('right',90)]:
                el=OxmlElement('w:'+edge);el.set(qn('w:w'),str(val));el.set(qn('w:type'),'dxa');m.append(el)
            pr.append(m)
            if i==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E5EDF0');pr.append(shade)
            p=c.paragraphs[0];p.paragraph_format.line_spacing=1.15;p.paragraph_format.space_after=Pt(0)
            if i==0 or k>0:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            font(p.add_run(str(rows[i][k])),10.5,i==0)
        if i==0:row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    return t


def picture(cell,path,width,caption):
    p=cell.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
    p.paragraph_format.space_after=Pt(3);p.add_run().add_picture(str(path),width=Cm(width));para(cell,caption,'caption')


def build(revision='r1'):
    facts=load(OUT/'核验/中期内容与证据.json');assert re.fullmatch(r'r\d+',revision)
    dest=OUT/'临时'/f'中期报告候选_{revision}.docx';assert not dest.exists()
    d=Document(REFERENCE);big=d.tables[0];old=Document(FILLED)
    for p in d.paragraphs:
        if '综合工程设计中期报告' in p.text:
            try:title_style=d.styles['Title']
            except KeyError:
                title_style=d.styles.add_style('Title',WD_STYLE_TYPE.PARAGRAPH);title_style.base_style=p.style
            title_style.font.color.rgb=RGBColor(0,0,0);p.style=title_style
    for a,b in zip(list(big.rows)[:6],list(old.tables[0].rows)[:6]):a._tr.getparent().replace(a._tr,deepcopy(b._tr))
    for p in d.paragraphs:
        text=p.text.strip()
        if re.match(r'^题\s+目',text):value='题    目：'+TITLE;size=14
        elif text.startswith('小组人员'):value='小组人员：陈德昕  李世显  陈龙辉';size=15
        elif text.startswith('指导老师'):value='指导老师：王笛';size=15
        elif text.startswith('填写时间'):value='填写时间：2026 年 10 月 25 日';size=15
        else:continue
        p.clear();font(p.add_run(value),size);p.paragraph_format.first_line_indent=Pt(0)
        ind=p._p.get_or_add_pPr().find(qn('w:ind'))
        if ind is not None:
            for a in ['firstLineChars','leftChars']:ind.attrib.pop(qn('w:'+a),None)
    prototype=deepcopy(big.rows[7]._tr)
    for row in list(big.rows)[7:]:big._tbl.remove(row._tr)
    for row in big.rows[6:]:
        row.height=None;row.height_rule=WD_ROW_HEIGHT_RULE.AUTO
        for flag in row._tr.xpath('./w:trPr/w:cantSplit|./w:trPr/w:trHeight'):flag.getparent().remove(flag)
    first=big.rows[6].cells[0];clean(first);first.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.TOP
    def page(title):
        p=d.add_paragraph();f=p.paragraph_format;f.page_break_before=True;f.keep_with_next=True;f.line_spacing=Pt(1);f.space_after=Pt(0);f.space_before=Pt(0);font(p.add_run(''),1)
        t=OxmlElement('w:tbl');t.append(deepcopy(big._tbl.tblPr));t.append(deepcopy(big._tbl.tblGrid));row=deepcopy(prototype)
        for flag in row.xpath('./w:trPr/w:cantSplit|./w:trPr/w:trHeight'):flag.getparent().remove(flag)
        c=row.find(qn('w:tc'))
        for ch in list(c):
            if ch.tag!=qn('w:tcPr'):c.remove(ch)
        c.append(OxmlElement('w:p'));t.append(row);p._p.addnext(t)
        from docx.table import _Cell
        cell=_Cell(c,d._body);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.TOP;para(cell,title,'title');return cell
    c=first
    para(c,'一、设计目标与任务','title')
    para(c,'本项目面向未知区域中的主动视觉地理定位，构建智能体与地理环境交互的实验平台。智能体从给定起点出发，只根据连续获得的局部航拍信息和目标线索，自主选择下一步方向，并在限定行动预算内定位目标网格。')
    para(c,'截至本次中期检查，航拍图块目标的单智能体导航、分阶段验证及可交互工作台已经形成稳定成果。后续主线进一步面向地面照片指定目标：研究照片与航拍观测的空间关联，并接通自主探索闭环。该跨视角自主能力尚未完成，不用已有航拍成功率替代其验收。')
    table(c,[['任务部分','中期状态','最终仍需完成'],['航拍目标','按指定协议验证通过','作为稳定组件与强基线保留'],['地面照片目标','编码、人工观察与失败诊断','可靠空间关联和自主方向决策'],['实验平台','创建、运行、保存与同题对照','接通地面自主模式及完整交付']],[3.2,4.3,6.5])
    para(c,'项目采用训练过的单导航Agent与线索/规则模块。GOMAA-Geo、GeoExplorer提供任务与机制参考，不宣称完整复现论文；多Agent协作保留为可选后续研究。人员信息和团队分工沿用已确认的选题报告。','note')

    c=page('二、完成情况')
    para(c,'（一）地理环境与可交互平台','sub')
    para(c,'已实现图像切块、任务登记、局部观测、四方向动作、预算扣减和终局反馈。越界动作停留原地并消耗一步；预算内首次到达目标即成功，最后一步到达也计成功。执行策略只能读取当前图、给定目标、公开位置、访问记录和剩余预算，真实目标坐标、距离及未访问图块仅由环境/评测端掌握。')
    para(c,'GeoNav工作台当前为4.13.1，具备场景配置、手动探索与Agent接管、当前题/整队列自动运行、暂停、临时与正式保存、结果重开和同题对照。外部单模型接口与批量框架已经接入；云端调用是否成功和导航是否有效分别记录，不把接口联通当作方法提升。')
    path=OUT/facts['assets']['platform']
    if path.exists():picture(c,path,13.6,'图1  当前工作台提供任务配置与执行过程，界面与实验内核分离')
    para(c,'数据与正式验收、自由测试及人工介入分别记账。保存结果可追溯到任务、配置、权重和源码；不承诺从服务重启前的未完成内存会话直接续跑。平台迁移和独立目录工程验证不替代全科研队列重跑。','note')

    c=page('二、完成情况 续')
    para(c,'（二）航拍目标方法与轻量化实现','sub')
    para(c,'稳定方法M0由冻结Sat2Cap编码器、Small256循环探索器、Edge目标线索头和动作门控组成。探索器通过GRU记忆产生探索建议，其目标通道使用固定拟合均值；给定目标的匹配由专门线索头负责。方向证据不可信时保留探索建议，可信且合法、落点未访问时优先采用线索。')
    picture(c,ROOT/facts['assets']['method'],13.6,'图2  航拍M0的局部观察与行动反馈闭环，答案信息仅属于评测器')
    table(c,[['组成','规模或参数','作用'],['Sat2Cap','冻结骨干','提取当前图与目标图表征'],['Small256','797,701参数','公开状态与历史上的探索动作'],['Edge线索头','136,837参数','邻接识别和条件方向判断'],['动作门控','阈值0.50；合法/未访问','可信线索优先，其余沿用探索']],[3.2,4.2,6.6])
    para(c,'线索联合置信度 q(d)=p(邻接)×p(d|邻接)。任务探索器与线索头合计934,538参数，轻量化针对可训练任务模块，不把冻结视觉骨干从系统资源中隐去。边缘连续性适用于同源、方向一致的航拍图块，不能直接用于地面照片。','note')

    c=page('二、完成情况 续')
    para(c,'（三）分阶段航拍验证与目标有效性','sub')
    para(c,'SR按计划题数统计预算内成功比例；SG为全部终局曼哈顿距离均值，成功记0。格数与米数分别标注，实际移动距离另报。三个权重重复同一队列，不增加独立地理样本量。')
    rows=[['验证批次','源 / 每权重题','网格 / 预算','SR','SG 格']]
    for x in facts['aerial']:rows.append([x['name'],f"{x['sources']} / {x['tasks_per_seed']}",x['protocol'],f"{x['sr']:.2%}",f"{x['sg']:.3f}"])
    table(c,rows,[2.8,3.0,3.8,2.1,2.3])
    para(c,'表1  各行属于不同冻结协议，不能相减排名或连成累计提升曲线','caption')
    para(c,'S3使用10个独立源文件，250题×3权重，真实目标的SR为86.80%、SG为0.300格；目标控制和原门槛通过。源文件独立不等于已经核实地理足迹无重叠。S4是在原图内提高切图密度，不是扩大真实覆盖面积。SwissView为冻结权重的同模态迁移，不证明地面照片或异时成像泛化。')
    para(c,'真实/禁用、均值和错误目标对照按各批原协议保留，错误目标仍对原任务答案计分。本文只引用已封存结果，未为更新材料重跑导航或改变队列、阈值和默认配置。')
    para(c,'学习基线已完成6000条同题评测：M0为86.80%/0.300格，有限预算GeoExplorer为35.60%/2.589格。后者目标有效性未过，训练资源及复现差异仍存在；此结果用于报告本地复现边界，不据此宣称优于原论文。')

    c=page('二、完成情况 续')
    para(c,'（四）真实连续区域与探索保护','sub')
    para(c,'在保持300米/格的连续区域中，冻结Coverage3Radial候选与M0进行同题确认：10个未参与选型的空间隔离区域，每区9km²，750条不同路线×3权重，10×10/B20。候选在可信线索优先的基础上采用三步覆盖规划，并用径向条件保护长距离探索。')
    table(c,[['同题方案','SR','SG 米'],['M0',f"{facts['area']['reference']['sr']:.2%}",f"{facts['area']['reference']['mean_sg_m']:.2f}"],['Coverage3Radial',f"{facts['area']['candidate']['sr']:.2%}",f"{facts['area']['candidate']['mean_sg_m']:.2f}"]],[5.4,3.5,5.1])
    picture(c,ROOT/facts['assets']['area'],13.6,'图3  径向保护在原冻结分层中的独立区域确认结果')
    para(c,'候选SR提高9.20个百分点，按完整区域分组的95%差区间为[+7.07,+11.02]点；恢复295条、损伤88条，净增加207条成功。原升级证据通过，适用范围限定到该数据和预算；不是每题都改善，也不自动替换其它协议的M0。')
    para(c,'MassGIS新产品的10×10/15×15确认仍未通过，Coverage SR为54.62%/20.27%。两网格共用10区，不能当作20独立地区；数据域、区域尺度和表征差异仍需区分。大区域失败作为中期边界保留，不通过反复调参重写原结论。','note')

    c=page('二、完成情况 续')
    para(c,'（五）地面照片目标与跨视角工程','sub')
    para(c,'地面照片与俯视航拍之间存在视角、尺度和场景内容差异；照片中的可见地标也不必等同拍摄点。当前已审查配对来源、目标格语义及标签不确定性，并完成地面/航拍实时编码接口、固定地面目标的人工连续观察和过程保存。该链路验证了环境交互和输入权限，尚未提供可靠地面自主方向决策。')
    path=OUT/facts['assets']['ground']
    if path.exists():picture(c,path,13.6,'图4  地面目标及空间证据可在平台查阅，当前仍是实验模式')
    para(c,'先后验证的全局/局部匹配、小对齐头、CCVPE直接迁移及普通RGB空间小头均按预定停止条件收束。完整参考条件也未支持足够可靠定位，不能把失败只归因于局部遮挡，更不能把原始余弦相似度直接当导航方向或概率。')
    para(c,'当前保留的是可运行接口、可复核负结果和适用边界，不把“材料多”“训练完成”或“界面可操作”当作跨视角定位成功。MM-GAG的6个确认组24素材继续封存，未用于本次材料或新选型。','note')

    c=page('三、存在的主要问题')
    para(c,'1. 地面目标的空间关联与自主闭环尚未完成。最新CV041使用冻结CLIP表征和65,537参数小头，完成18头、3600更新及6750条非训练预测；三城留出开发的“接受且误差≤5米”为0/9，不能作为导航SR。')
    table(c,[['城市','三种子位置误差 米','接受且≤5米'],['Amsterdam','18.306 / 20.384 / 18.306','0 / 3'],['Krakow','6.438 / 6.438 / 6.438','0 / 3'],['Stockholm','1.431 / 1.431 / 1.431','0 / 3']],[3.2,7.1,3.7])
    para(c,'表2  位置误差包括被拒绝的正例；三城×三权重仍仅3个来源','caption')
    para(c,'Full开发450对全部拒绝，宏联合指标66.67%恰为全拒绝参照。固定位置输出仅3/9在5米内，即使只改变接受规则，上界仍低于6/9原门槛。因此同配置加训或扫描阈值缺少依据；本轮保留失败，不扩大模型追分。')
    para(c,'2. 泛化与监督证据有限。新域大区域未通过，照片空间标签仍有估计误差与可辨识性限制；现有材料不足以分离数据多样性、表征、训练目标等原因。需要与最终定位定义匹配的监督证据，而不是继续重复同类局部变体。')
    para(c,'3. 论文与交付仍有缺项。可靠学习基线、新来源共同目标确认、干净环境完整科研复跑尚未闭环；真实无人机、异时影像和任意角度泛化没有证明。平台未结束会话的重启恢复与另一台机器完整验收仍需继续完善。')
    para(c,'4. 工程与研究必须分账。人工操作、自由测试、旧结果回放和正式确认各有用途；可视化中的全图/真值只给观察者，不能回流策略。多个模块及开发助手接手不等于算法实现多Agent。')

    c=page('四、下一步需要完成的工作')
    para(c,'下一阶段保留已验证的航拍系统和当前工作台，优先补齐“地面目标照片→合法连续航拍观测→自主动作→目标格”的核心缺口。先明确可信标签与任务分工，再冻结有机制支持的轻量因素；旧失败小头和阈值不复活，不把所有尺度都达到同一SR作为结题前提。')
    table(c,[['顺序','下一阶段交付','完成条件'],['1 空间证据','固定目标定义、来源与监督','标签和像素对应可信；开发/确认分离'],['2 自主小闭环','目标条件策略与环境连接','同预算规则/无目标基线和三类目标干预；SR/SG/恢复损伤'],['3 新源确认','冻结配置后的独立验证','不在确认源选型；按完整来源统计不确定性'],['4 平台与材料','统一操作、恢复与最终报告','创建→运行→保存→同题对照可演示；结果与文稿一致']],[2.5,4.8,6.7])
    para(c,'最终以限定任务下的可信自主定位、完整平台流程和可追溯交付为标准。多Agent、实飞与任意照片全球定位可列未来工作；已有航拍成果不因后续研究而淘汰。各批具体指标与资源上限须事先冻结，不在看到成绩后放宽。')
    para(c,'资料依据','sub')
    for text in [
        'E1—E4：Masa S2/S3/S4与SwissView正式批次的对照汇总、验收和复核；数据范围分别列于表1。',
        'E5：MasaRoads径向保护独立新区域确认，10区/750路线/3权重；主对照与目标控制原件。',
        'E6：MassGIS正式双网格确认；E7：OpenCVL普通RGB空间小头固定开发及最终判定。',
        'E8：工作台4.13.1发布修复、当前源码及正式服务回执。以上来源路径/原SHA随交付证据表保留。']:
        para(c,text,'note')
    para(c,'报告截止2026年10月25日。旧中期稿转为历史材料，本稿更新阶段成果，不代表整个课题已结束。','note')

    tail=d.add_paragraph();tail.paragraph_format.line_spacing=Pt(1);tail.paragraph_format.space_after=Pt(0);font(tail.add_run(''),1)
    # Preserve template page parts; only the body, metadata and Title style are authored.
    settings=d.settings.element;flag=settings.find(qn('w:updateFields'))
    if flag is None:flag=OxmlElement('w:updateFields');settings.append(flag)
    flag.set(qn('w:val'),'true')
    d.core_properties.title=TITLE+' 中期报告';d.core_properties.subject='综合工程设计中期检查';d.core_properties.author='陈德昕 李世显 陈龙辉';d.core_properties.last_modified_by='项目组'
    d.save(dest)
    # python-docx may reserialize untouched footer/numbering XML. Restore their
    # exact original bytes instead of weakening the preserve-only package gate.
    editable={'word/document.xml','word/styles.xml','word/settings.xml','word/_rels/document.xml.rels',
              '[Content_Types].xml','docProps/core.xml','docProps/app.xml'}
    rewritten=dest.with_suffix('.docx.tmp')
    with ZipFile(REFERENCE) as reference,ZipFile(dest) as generated,ZipFile(rewritten,'w') as fixed:
        source_names=set(reference.namelist())
        for info in generated.infolist():
            raw=reference.read(info.filename) if info.filename in source_names and info.filename not in editable else generated.read(info.filename)
            fixed.writestr(info,raw)
    rewritten.replace(dest)
    print(dest)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='build':build(sys.argv[2] if len(sys.argv)>2 else 'r1')
