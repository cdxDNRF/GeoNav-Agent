"""Evidence-bound closeout of one offline diagnosis and one frozen guard factor."""
from pathlib import Path
import sys,json,hashlib,statistics
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from documents import stage_figures as graphics

ROOT=Path(__file__).resolve().parents[3];FIG=ROOT/'绘图';NAV=ROOT/'项目导航'
DIAG=ROOT/'DATA/processed_data/SwissView/评测结果/继续训练恢复损伤诊断_v1'
RUN=ROOT/'DATA/processed_data/Masa/评测结果/回访新格筛选冻结验证_v1'
DEFAULT=ROOT/'project/local_policy_default.json'

def read(p):return json.loads(p.read_text('utf8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def write(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def new_text(p,text):
    with p.open('x',encoding='utf8') as f:f.write(text)

def required():
    d=read(DIAG/'诊断汇总.json');da=read(DIAG/'独立复核.json');v=read(RUN/'验收结论.json')
    assert da['passed'] and da['summary_sha256']==digest(DIAG/'诊断汇总.json')
    assert v['candidate_passed'] and v['audit_passed']
    assert v['summary_sha256']==digest(RUN/'主对照/对照汇总.json') and v['audit_sha256']==digest(RUN/'独立复核.json')
    s=read(RUN/'主对照/对照汇总.json');c=read(RUN/'目标对照/目标证据汇总.json');reg=read(RUN/'预登记.json')
    assert c['passed'] and digest(DEFAULT)==reg['default_sha256']
    # Closeout cross-checks include presentation aggregates, not only gate deltas.
    for arm in ('M0','F'):
        rr=[]
        for seed in range(3):rr.append(read(RUN/'主对照'/(f'M0_s{seed}_复用结果.json' if arm=='M0' else f'F_s{seed}_CueFull_结果.json')))
        for field,name in [('sr','sr'),('mean_sg_all_episodes','sg')]:
            values=[r['metrics'][field] for r in rr]
            assert s['arms'][arm][name+'_by_seed']==values and abs(s['arms'][arm][name+'_mean']-statistics.mean(values))<1e-12
    assert s['recovery']=={'harmed':3,'recovered':18}
    return d,s,c,v

def reports(d,s,c,v):
    new_text(DIAG/'恢复损伤诊断报告.md','''# Continue5恢复、损伤与失败阶段诊断

本轮完整读取并复核6000条已保存轨迹、74987个动作，无模型推理、训练、API或新增导航。任务为SwissView40—59号20已消费源文件，双网格各500题×3权重×2方法；以下重点为10×10/B20。

## 主要结果

Continue5的410条失败中，406条没有获得仍可行动的目标邻接机会；只有4条曾在有预算时邻接。原M0为412/419条没有这种机会。SG改善主要伴随失败轨迹更接近目标，仍未在预算内完成定位；这描述路径现象，不将责任全部归给探索器。

| 互斥失败类别 | 原M0（419失败） | Continue5（410失败） |
|---|---:|---:|
| 有预算邻接未成功 | 7 | 4 |
| 仅预算耗尽终局邻接 | 72 | 109 |
| 到两格范围但未邻接 | 190 | 184 |
| 全程未进入两格范围 | 150 | 113 |

四类以真实距离作事后标签，终局不当决策机会。原M0失败条件SG4.403，Continue5为3.176；全体SG分别1.230/0.868。两种SG的分母不同，不混用。

## 恢复与损伤为何相抵

同题三权重配对为共同成功905、恢复185、损伤176、共同失败234，净9/1500。恢复中175/185、损伤中167/176的首次动作分歧都发生在“双方都走新格”的状态；此前视觉输入、目标头输出相同，cue都未接管。因此简单保护“走新格”不能识别哪个新方向正确，不能把所有差异解释为回访。

同时，原M0有162/419失败、Continue5有98/410失败出现过“无cue、原动作回访、存在合法新格”的状态。原M0恢复组中92条具有这一现象；Continue5损伤组中52条具有。这些是公开可检测的介入机会，不是可恢复成功数量或因果上界。

目标头也有途中非一步命中的接管：M0接受2671次，其中1610次未立即命中；Continue5接受2653次，其中1579次未立即命中。该诊断按“邻接方向意味着下一步到目标”定义，**未立即命中不代表该动作一定远离目标或无益**；不能据此直接提高阈值。已有目标对照证明整条线索通道在相应队列有净贡献。

## 本轮唯一后续候选

选择原默认M0上的无cue回访新格筛选：仅在原动作回访且有合法未访问邻格时，按原模型logit选择新格。其余状态与已接受cue沿用原策略，不增加训练或模型容量。与旧一次触发的两步账本规划及其新格保护不同，本项只处理实际回访决定。

该候选在另立冻结协议下使用Masa原S4完整dev子集250题/10开发源；此SwissView20图只用于诊断，不再声称独立确认。原Continue5升级失败的判定仍保留。诊断结果已经全部复核，候选结论见后续批次。

[预登记](预登记.json) / [诊断汇总](诊断汇总.json) / [分组结果](分组诊断.json) / [逐题配对](逐题配对诊断.json) / [复核](独立复核.json)
''')
    wrong=read(RUN/'错误目标计划.json');exceptions=sum(not x['matched_distance'] for x in wrong.values())
    new_text(RUN/'回访新格筛选验证报告.md',f'''# 无cue回访新格筛选：冻结开发验证

**候选F通过本批开发与目标证据门槛，原默认保持；尚未完成未参与选型的新源确认。** 没有新增训练或云端调用。F在原M0上加入一个无参数动作筛选：无已接受cue时，若原动作回访且存在合法未访问邻格，则按原explorer logit择优；有cue、原动作已去新格、或无新格时保留原动作。

## 同题主对照

任务在推理前按split==dev取原S4完整250题、10已知开发源，C12—16每源每档5题；10×10/B20、三份冻结原探索器与目标头。原250道test题没有进入本次开发评测。候选设计参考了已消费SwissView诊断，本队列也为已知开发图，不是独立泛化成绩。

| 方法 | seed0 / seed1 / seed2 SR | 平均SR | 全体SG |
|---|---|---:|---:|
| 原M0，同题复用 | 74.40% / 84.40% / 74.80% | 77.87% | 1.321 |
| F回访新格筛选 | 78.00% / 85.20% / 76.40% | 79.87% | 1.188 |

SR恰好提高2.00个百分点，三权重均为正；SG改善0.133格。按10源文件成组、源内三权重平均的95%配对SR差区间[+0.80,+3.60]点，SG差区间[−0.213,−0.045]格。达到预登记开发门槛；小型已知队列上的区间不代替新源确认。

F在70/750条主记录中实际介入，改变140次动作；恢复18条、损伤3条，净15/750。每seed恢复/损伤为11/2、2/0、5/1。该队列与原500题全队列不同，不能拿77.13%作本次基线；也不能称修复了SwissView诊断中的全部176损伤。

## 目标证据

主门槛通过后才按协议运行2250条目标控制；各控制使用相同回访规则，均值和错图同时处理global/local/edge三种目标通道。环境仍按真实目标判定。

| F条件 | 平均SR | 全体SG |
|---|---:|---:|
| 真实目标 | 79.87% | 1.188 |
| 禁用cue | 57.60% | 2.327 |
| 均值目标 | 50.40% | 2.385 |
| 错误目标 | 54.67% | 2.347 |

真实目标对三个控制分别领先22.27、29.47、25.20点，三权重均为正且SG改善。使用原冻结错误目标计划；{exceptions}/250题无法匹配初始距离，例外原样保留，不选择有利错图。

## 复核、冻结与下一步

15项新增相关测试通过（诊断8项、策略7项）。候选新增3000轨迹，原M0复用750条；3750条/65838动作从权重与环境逐步重算，独立实现的新格选择、平分、门槛、恢复损伤与源文件bootstrap通过。197份历史文件、1003份图像/特征输入与冻结核心源码保持。诊断另复核6000条/74987动作。

原M0权重、目标头、0.50阈值和默认配置没有更改，F不是Continue5续训。三份权重完整纳入，不挑最好seed或叠加新因素。本轮固定验证到此结束；下一步使用未参与本项诊断与选型的源文件，另立一次独立确认协议。SwissView40—59已经消费，本轮52尚未导航源均未使用。

已验收S2/S3/S4/原SwissView主线继续有效。F是10×10密度协议开发候选，不能称实际大范围、真实异时、跨视角或多Agent验证通过。

[执行协议](执行协议.md) / [预登记](预登记.json) / [主对照](主对照/对照汇总.json) / [目标对照](目标对照/目标证据汇总.json) / [恢复损伤](逐题恢复与损伤.json) / [复核](独立复核.json) / [验收](验收结论.json)
''')
    reg=read(RUN/'预登记.json');original=read(DEFAULT)
    candidate=dict(version='fresh-alternative-candidate-v1',status='development_and_target_evidence_passed_only',
        grid_size=10,budget=20,rule='agents.fresh_alternative.fresh_alternative',rule_sha256=reg['source_sha256']['agents/fresh_alternative.py'],
        frozen_default_sha256=reg['default_sha256'],original_default_configuration=original,
        actual_evaluation_models=[{'seed':s,'explorer':f'DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1/Edge_s{s}/explorer.pt',
            'head':f'DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1/Edge_s{s}/head.pt'} for s in range(3)],
        threshold=.5,ensemble=False,reference_seed=0,all_evaluated_seeds=[0,1,2],independent_confirmation_passed=False,
        default_replacement_allowed=False,source_sha256=reg['source_sha256'],evaluation_verdict_sha256=digest(RUN/'验收结论.json'))
    for row in candidate['actual_evaluation_models']:
        for field in ('explorer','head'):row[field+'_sha256']=digest(ROOT/row[field])
    write(RUN/'候选配置.json',candidate)
    write(DIAG/'候选选择.json',dict(diagnosis_audit_sha256=digest(DIAG/'独立复核.json'),candidate='fresh_alternative on original M0',
        selection_reason='public no-cue revisit with legal fresh alternative; occurrence is not causality',
        candidate_batch=rel(RUN),candidate_verdict_sha256=digest(RUN/'验收结论.json'),no_second_factor=True))
    for folder,report in [(DIAG,'恢复损伤诊断报告.md'),(RUN,'回访新格筛选验证报告.md')]:
        new_text(folder/'README.md',f'# 本批入口\n\n[报告]({report}) / [复核](独立复核.json) / [预登记](预登记.json)。原始轨迹与配置保持各自来源。\n')
        write(folder/'最终状态.json',dict(completed=True,audit_passed=True,new_training_steps=0,cloud_calls=0,new_source_files=0,default_changed=False,
            report=report,report_sha256=digest(folder/report),audit_sha256=digest(folder/'独立复核.json'),
            state='diagnosis_closed' if folder==DIAG else 'candidate_frozen_pending_independent_sources'))

def plots(d,s,c):
    graphics.style();graphics.ITEMS.clear()
    f,axes=plt.subplots(1,2,figsize=(8.4,4.4),gridspec_kw={'width_ratios':[1.65,1]})
    keys=['missed_actionable_adjacency','adjacent_only_at_terminal','near_without_adjacency','never_within_two']
    labels=['有预算邻接\n未成功','仅终局\n邻接','到两格内\n未邻接','从未进入\n两格范围']
    for arm,off,color,marker in [('M0',-.18,graphics.GRAY,'s'),('Continue5',.18,graphics.BLUE,'o')]:
        values=[d['averages']['10'][arm]['failure_classes'].get(k,0) for k in keys];x=np.arange(4)+off
        axes[0].bar(x,values,width=.33,color=color,edgecolor='#33424E',linewidth=.6)
        axes[0].scatter(x,values,marker=marker,s=20,facecolor='white',edgecolor='#33424E',zorder=3)
        for i,v in zip(x,values):axes[0].text(i,v+7,str(v),ha='center',fontsize=10)
    axes[0].set_xticks(range(4),labels);axes[0].set_ylim(0,238);axes[0].set_ylabel('失败轨迹数（条）');axes[0].set_title('(a) 失败发生阶段',loc='left')
    axes[0].legend([Line2D([],[],color=graphics.GRAY,marker='s'),Line2D([],[],color=graphics.BLUE,marker='o')],['原M0','Continue5'],frameon=False,ncol=2,loc='upper left')
    axes[1].bar([0,1],[185,176],width=.58,color=[graphics.TEAL,graphics.ORANGE],edgecolor='#33424E',linewidth=.6)
    for x,y in enumerate([185,176]):axes[1].text(x,y+7,str(y),ha='center',fontsize=11)
    axes[1].set_xticks([0,1],['恢复','损伤']);axes[1].set_ylim(0,238);axes[1].set_ylabel('配对轨迹数（条）');axes[1].set_title('(b) 净增9 / 1500条',loc='left')
    for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True)
    f.subplots_adjust(left=.08,right=.985,bottom=.27,top=.88,wspace=.30)
    f.text(.5,.105,'SwissView20已消费源文件；10×10/B20；500题×3权重。失败数：M0 419，Continue5 410。',ha='center',fontsize=9)
    f.text(.5,.04,'终局邻接没有剩余行动预算；真值仅用于事后分类，计数不代表因果归因。',ha='center',fontsize=9)
    caption='Continue5的410条失败中406条未在预算内获得可行动的目标邻接机会；109条仅在预算耗尽后邻接。其185条恢复被176条损伤抵消，净增9/1500。左图四类互斥，包含原M0的全部419条失败和Continue5的全部410条失败；方形/圆形区分方法。20已消费SwissView源文件、500题×3训练权重、10×10/B20；无新增导航，事后真值分类不构成因果或策略输入。'
    graphics.save(f,FIG/'数据结果图','36_恢复损伤与失败阶段',caption,[rel(DIAG/'诊断汇总.json'),rel(DIAG/'独立复核.json')],{'sources':20,'planned_each_weight':500,'weights':3,'grid':10,'budget':20,'posthoc':True})
    labels=['原M0\n真实目标','F\n真实目标','F\n禁用cue','F\n均值目标','F\n错误目标']
    values=[s['arms']['M0'],s['arms']['F'],c['arms']['Baseline'],c['arms']['CueMean'],c['arms']['CueWrong']]
    colors=[graphics.GRAY,graphics.BLUE,'#A8AEB4',graphics.ORANGE,graphics.TEAL]
    f,axes=plt.subplots(1,2,figsize=(8.4,4.4));graphics.frames(axes)
    for ax,field,scale,label in [(axes[0],'sr',100,'SR：成功率（%）'),(axes[1],'sg',1,'SG：全体终点距离（格）')]:
        means=[x[field+'_mean']*scale for x in values];seeds=[[v*scale for v in x[field+'_by_seed']] for x in values]
        graphics.bars(ax,labels,means,colors,decimals=2 if scale==100 else 3,seeds=seeds,percent=scale==100)
        ax.set_ylabel(label);ax.set_ylim(0,108 if scale==100 else 3.65)
    axes[0].set_title('(a) 同题SR提升2.00个百分点',loc='left');axes[1].set_title('(b) 主对照与目标干预',loc='left')
    f.subplots_adjust(left=.08,right=.99,bottom=.26,top=.89,wspace=.32)
    f.text(.5,.105,'Masa 10已知dev源文件；250题×3权重；10×10/B20。标记为三权重，非置信区间。',ha='center',fontsize=9)
    f.text(.5,.04,'F对M0源文件配对SR差95%区间：+0.80至+3.60点。开发验证通过，尚未独立确认。',ha='center',fontsize=9)
    caption='回访新格筛选F在同题开发子集把SR从77.87%提高到79.87%，SG从1.321降到1.188；恢复18条、损伤3条，三权重均有SR正收益。Full对禁用、均值、错误目标均通过预定证据门槛。Masa原S4完整dev子集10源文件、250题×3权重、10×10/B20，未纳入原test子集；圆/三角/方形为seed0/1/2而非CI。按源文件成组的SR增益95%区间[+0.80,+3.60]点，仅描述已知开发队列；原默认未替换。'
    graphics.save(f,FIG/'数据结果图','37_回访新格筛选与目标对照',caption,[rel(RUN/'主对照/对照汇总.json'),rel(RUN/'目标对照/目标证据汇总.json'),rel(RUN/'验收结论.json')],{'sources':10,'split':'dev','planned_each_weight':250,'weights':3,'grid':10,'budget':20,'independent_confirmation':False})
    items=graphics.ITEMS
    for item in items:item['reproduction_script']=rel(Path(__file__))
    manifest=dict(date='2026-10-02',script_sha256=digest(Path(__file__)),figures=items,
        source_sha256={p:digest(ROOT/p) for item in items for p in item['sources']},diagnosis_summary=d,primary_summary=s,target_summary=c)
    write(FIG/'绘图数据/恢复损伤与回访筛选_v1.json',manifest)
    new_text(FIG/'恢复损伤与回访筛选图注.md','\n\n'.join('# '+item['id']+'\n\n'+item['caption'] for item in items)+'\n')
    catalog=read(FIG/'图表来源清单.json');assert all(x['id'] not in {y['id'] for y in catalog['figures']} for x in items)
    catalog['figures'].extend(items);(FIG/'图表来源清单.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n','utf8')
    new_text(FIG/'审阅/恢复损伤与回访筛选设计_v1.md','''# 科研图设计与核验依据

采用figure-designer的实验结果图规范。图36用分组计数条形图区分四类互斥失败，再展示配对恢复/损伤；图37用同队列SR/SG条形图及三权重标记，同时展示主因素和目标对照。图36数据为确定性历史计数，图37标记为训练权重，均不伪装成置信区间。

8.4×4.4英寸画布，正文9—11.5pt，建议报告/PPT插入宽度至少18cm；PDF嵌入字体，SVG保留可编辑文字，PNG300dpi。颜色沿用Okabe-Ito系蓝/橙/绿与中灰，方法同时用位置/文字或形状区分；轴从0开始，SG包含成功的0距离。图注说明源文件、题数、预算、三权重及开发范围。

本批结果不属于新的独立地图确认；样本相同与因果归因分别说明。实际像素审阅和导出校验另保存，未审阅前不宣称视觉QA完成。
''')
    sourcecopy=FIG/'审阅/恢复损伤与回访筛选_作图源码_v1.py'
    with sourcecopy.open('xb') as f:f.write(Path(__file__).read_bytes())

def navigation():
    text=(ROOT/'README.md').read_text('utf8')
    summary='''## 最新开发结果：回访新格筛选候选通过

离线复核6000条Continue5/M0既有轨迹后，只新增一个公开状态动作筛选。Masa原S4完整dev子集250题/10源文件×3权重，原M0 **SR77.87%/SG1.321**，F **79.87%/1.188**；SR+2.00点、3/3权重正，源文件95%差[+0.80,+3.60]点；恢复18/损伤3。禁用、均值、错误目标为57.60%、50.40%、54.67%，目标证据通过。

新增3000条/复用750条，共3750条与65838动作完整复核；15项新增相关测试通过。本轮无训练/云端/新源文件，原S2/S3/S4验收与默认保持；F尚待未参与选型的新源确认。原Continue5独立升级失败不回写。

[诊断报告](DATA/processed_data/SwissView/评测结果/继续训练恢复损伤诊断_v1/恢复损伤诊断报告.md) / [候选报告](DATA/processed_data/Masa/评测结果/回访新格筛选冻结验证_v1/回访新格筛选验证报告.md) / [验收](DATA/processed_data/Masa/评测结果/回访新格筛选冻结验证_v1/验收结论.json) / [冻结候选](DATA/processed_data/Masa/评测结果/回访新格筛选冻结验证_v1/候选配置.json) / [图36—37](绘图/恢复损伤与回访筛选图注.md)

'''
    marker='## 本轮开发：继续训练恢复与损伤离线诊断'
    assert marker in text;text=text.replace(marker,summary+marker,1).replace('全部58批实验索引','全部60批实验索引').replace('35张实验结果图','37张实验结果图')
    text=text.replace('进入阶段收尾；','已完成阶段材料收尾并形成回访新格筛选开发候选；')
    text=text.replace('近期交付是完善中期检查与答辩材料。下一研究因素应依据已保存的恢复/损伤和剩余预算机会单独选择；本轮未启动新训练、API请求或导航实验。',
        '中期检查与答辩材料已完成阶段收尾；新F候选通过一次开发与目标对照，下一项适合固定候选做新源确认。本轮新增3000条本地CPU导航，无训练、无API、无新源文件；原默认保持。')
    (ROOT/'README.md').write_text(text,'utf8')
    current=NAV/'当前进度.md';text=current.read_text('utf8')
    text=text.replace('当前收束已有实验并完善课程材料。','阶段材料收尾已完成；随后一次恢复损伤诊断与单因素冻结验证形成回访新格筛选开发候选。')
    text=text.replace('本轮只整理材料，不启动新因素。下一项先依据新源成功损伤与预算机会确定可检验假设，再冻结单因素、同预算对照和停止条件。已过主线与候选失败分别保留。',
        '本轮新增F回访新格筛选开发候选：原M0同题77.87%/SG1.321，F79.87%/1.188，3/3权重为正；恢复18、损伤3，目标控制通过。该结果仅使用Masa原S4的dev子集250题/10源，不等同原500题全队列，也不是独立确认。原默认保持，下一项固定候选做未参与选型的新源确认。')
    text+='\n[本轮诊断](../'+rel(DIAG/'恢复损伤诊断报告.md')+') / [候选报告](../'+rel(RUN/'回访新格筛选验证报告.md')+') / [候选验收](../'+rel(RUN/'验收结论.json')+')。\n'
    current.write_text(text,'utf8')
    idx=read(NAV/'实验索引.json');assert len(idx['batches'])==58
    new=[]
    for folder,ds,category,status,note,report in [(DIAG,'SwissView','离线诊断','诊断与复核完成','6000既有轨迹；无新增导航，分解恢复/损伤与邻接时机','恢复损伤诊断报告.md'),
        (RUN,'Masa','单因素开发','开发候选与目标证据通过','dev250×3；SR+2.00点、3/3正；原默认保持，尚未独立确认','回访新格筛选验证报告.md')]:
        item=dict(dataset=ds,kind='评测结果',name=folder.name,category=category,status=status,scope=note,path=rel(folder),reports=[rel(folder/report)],
            evidence=[rel(folder/n) for n in ('独立复核.json','最终状态.json','验收结论.json') if (folder/n).exists()]);new.append(item)
        with (NAV/'实验索引.md').open('a',encoding='utf8') as f:f.write(f'| {ds} / 评测结果 | {folder.name} | {status} | {note}；[报告](<../{rel(folder/report)}>) / [复核](<../{rel(folder/"独立复核.json")}>) / [原目录](<../{rel(folder)}>) |\n')
    idx['batches'].extend(new);(NAV/'实验索引.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n','utf8')
    for path in (NAV/'README.md',NAV/'目录结构与维护.md'):
        text=path.read_text('utf8').replace('58批','60批');path.write_text(text,'utf8')
    p=FIG/'README.md';text=p.read_text('utf8');header='''## 新增：恢复损伤诊断与回访新格筛选

图36说明Continue5的多数失败在预算内尚未获得目标邻接机会；图37展示原M0加新格筛选后，在Masa dev250×3由77.87%升至79.87%，目标控制通过。新增候选未做独立确认，默认保持。

[图36 PDF](数据结果图/36_恢复损伤与失败阶段.pdf) / [图37 PDF](数据结果图/37_回访新格筛选与目标对照.pdf) / [图注](恢复损伤与回访筛选图注.md) / [数据与SHA](绘图数据/恢复损伤与回访筛选_v1.json)。同目录含300dpi PNG和可编辑SVG；建议插入宽度≥18cm。

'''
    a,b=text.split('\n',1);p.write_text(a+'\n\n'+header+b.lstrip('\n'),'utf8')

def main():
    assert not (RUN/'回访新格筛选验证报告.md').exists(),'immutable closeout exists'
    for name in ('36_恢复损伤与失败阶段','37_回访新格筛选与目标对照'):
        assert not any((FIG/'数据结果图'/f'{name}.{ext}').exists() for ext in ('png','pdf','svg'))
    d,s,c,v=required();reports(d,s,c,v);plots(d,s,c);navigation()
    print(dict(reports=2,figures=2,candidate_passed=True,independent_confirmation=False,default_changed=False),flush=True)

if __name__=='__main__':main()
