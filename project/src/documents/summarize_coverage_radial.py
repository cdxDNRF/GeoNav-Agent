"""Report and scientific plot for the frozen radial protection experiment."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import sys
import warnings

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'project/src'))
from documents import stage_figures as graphics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

RUN=ROOT/'DATA/processed_data/MasaRoads/评测结果/覆盖接管径向保护验证_v1'
PREVIOUS=ROOT/'DATA/processed_data/MasaRoads/评测结果/空间覆盖与误触发单因素验证_v1'
OLD=ROOT/'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
FIG=ROOT/'绘图'
REPORT=RUN/'径向保护开发验证报告.md'
CAPTION=FIG/'覆盖接管径向保护图注.md'
MANIFEST=FIG/'绘图数据/覆盖接管径向保护_v1.json'
STEM='48_覆盖接管径向保护的分层结果'
COHORTS=('all','long_distance','seam_target','interior_target')
LABELS=('混合队列','长距离C12—16','接缝C8—12','内部C8—12')


def read(p):return json.loads(p.read_text('utf-8-sig'))
def digest(p):
    h=sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:
        f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2))
def metric(arm,cohort):return arm['metrics'] if cohort=='all' else arm['by_stratum'][cohort]


def main():
    verdict,state=read(RUN/'验收结论.json'),read(RUN/'最终状态.json')
    audit,summary=read(RUN/'主对照/独立复核.json'),read(RUN/'主对照/对照汇总.json')
    secondary=read(RUN/'主对照/Coverage3机制对照汇总.json')
    diagnostic=read(RUN/'离线诊断/诊断汇总.json')
    assert verdict['completed'] and state['completed'] and audit['passed']
    assert state['verdict_sha256']==digest(RUN/'验收结论.json')
    assert audit['records']==2250 and audit['summary_sha256']==digest(RUN/'主对照/对照汇总.json')
    assert verdict['default_sha256']==digest(ROOT/'project/local_policy_default.json')
    outputs=[REPORT,RUN/'README.md',CAPTION,MANIFEST]
    outputs.extend(FIG/'数据结果图'/f'{STEM}.{ext}' for ext in ('png','pdf','svg'))
    assert not any(p.exists() for p in outputs)
    sources=[p for p in RUN.glob('*') if p.is_file()]
    sources.extend(p for p in (RUN/'离线诊断').glob('*.json'))
    sources.extend(p for p in (RUN/'主对照').glob('*') if p.is_file())
    if verdict['controls_started']:sources.extend(p for p in (RUN/'目标对照').glob('*') if p.is_file())
    sources.extend((ROOT/'project/local_policy_default.json',ROOT/'选题报告相关/覆盖接管径向保护测试_v1.json',
        ROOT/'选题报告相关/覆盖接管径向保护诊断方案_v1.md',ROOT/'选题报告相关/覆盖接管径向保护单因素验证方案_v1.md',
        Path(__file__),ROOT/'project/src/documents/stage_figures.py'))
    next_text=''
    if verdict['eligible_for_new_area_confirmation']:
        assert (RUN/'下一项空间隔离新区域确认草案.md').is_file()
        sources.append(ROOT/'project/src/documents/prepare_coverage_radial_confirmation_draft.py')
        next_text='[下一项新区域确认草案](下一项空间隔离新区域确认草案.md)已保存，尚未准备新图或运行该确认。'
    seeds={'M0':[],'Coverage3':[],'Coverage3Radial':[]}
    folders={'M0':OLD/'独立确认/神经对照','Coverage3':PREVIOUS/'主对照','Coverage3Radial':RUN/'主对照'}
    for name in seeds:
        for seed in range(3):
            p=folders[name]/f'{name}_s{seed}_CueFull_结果.json'
            result=read(p); trace=p.with_name(p.name.replace('结果.json','轨迹.jsonl'))
            assert result['trajectory_sha256']==digest(trace)
            seeds[name].append(result);sources.extend((p,trace))
    arms={'M0':summary['reference'],'Coverage3':secondary['reference'],'Coverage3Radial':summary['candidate']}
    if not verdict['development_main_passed']:status='主保护门槛未通过'
    elif not verdict['development_target_evidence_passed']:status='主门槛通过，目标证据未通过'
    else:status='开发与目标证据通过；待新区域确认'
    table=['| 队列 | M0 SR | Coverage3 SR | 径向保护SR | 对M0 SR差 | M0 SG米 | Coverage3 SG米 | 径向保护SG米 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    physical=['| 队列 | M0失败SG米 | Coverage3失败SG米 | 径向保护失败SG米 | 径向保护平均有效移动米 |','|---|---:|---:|---:|---:|']
    for c,label in zip(COHORTS,LABELS):
        a,b,d=(metric(arms[n],c) for n in arms)
        table.append(f"| {label} | {a['sr']*100:.2f}% | {b['sr']*100:.2f}% | {d['sr']*100:.2f}% | {(d['sr']-a['sr'])*100:+.2f}点 | {a['mean_sg_m']:.2f} | {b['mean_sg_m']:.2f} | {d['mean_sg_m']:.2f} |")
        physical.append(f"| {label} | {a['mean_failed_sg_m']:.2f} | {b['mean_failed_sg_m']:.2f} | {d['mean_failed_sg_m']:.2f} | {d['mean_valid_travel_m']:.2f} |")
    gates=summary['checks'];gate_rows=[('混合SR对M0至少+2点',gates['mixed_SR_gain2pp']),
        ('至少2/3权重正收益',gates['two_positive_weights']),('混合SG不变差',gates['mixed_SG_no_worse'])]
    for c,label in zip(COHORTS[1:],LABELS[1:]):
        gate_rows.extend(((label+' SR损伤不超过2点',gates['strata'][c]['SR_loss_at_most2pp']),
            (label+' SG不变差',gates['strata'][c]['SG_no_worse'])))
    gate_table='| 冻结门槛 | 判定 |\n|---|---|\n'+'\n'.join(f"| {label} | {'通过' if passed else '未通过'} |" for label,passed in gate_rows)
    weight_table='| 权重 | 径向保护混合SR | SG米 | 保护否决次数 | 实际覆盖接管次数 |\n|---:|---:|---:|---:|---:|\n'
    for seed,result in enumerate(seeds['Coverage3Radial']):
        weight_table+=f"| {seed} | {result['metrics']['sr']*100:.2f}% | {result['metrics']['mean_sg_m']:.2f} | {result['guarded_actions']} | {result['changed_actions']} |\n"
    effect=summary['effects']['all']
    intervals='| 队列 | 对M0 SR差95%区域区间 | 对M0 SG差95%区域区间（米） |\n|---|---|---|\n'
    for c,label in zip(COHORTS,LABELS):
        e=summary['effects'][c]
        intervals+=f"| {label} | [{e['source95_SR'][0]*100:+.2f},{e['source95_SR'][1]*100:+.2f}]点 | [{e['source95_SG'][0]*300:+.2f},{e['source95_SG'][1]*300:+.2f}] |\n"
    secondary_effect=secondary['effects']['all']
    control_text=('主门槛通过，已完成6750条条件性目标对照及独立复核，具体证据见目标对照目录。'
        if verdict['controls_started'] else '主门槛未通过，按预先冻结协议未启动6750条目标对照；本因素收束。')
    control_table=''
    if verdict['controls_started']:
        control_summary=read(RUN/'目标对照/对照汇总.json')
        control_audit=read(RUN/'目标对照/独立复核.json')
        assert control_audit['passed'] and control_audit['records']==6750
        assert control_audit['summary_sha256']==digest(RUN/'目标对照/对照汇总.json')
        control_table='| 目标条件 | SR | SG米 | Full SR差 | 区域95% SR差区间 | 全部证据门槛 |\n|---|---:|---:|---:|---|---|\n'
        for condition in ('Baseline','CueMean','CueWrong'):
            item=control_summary['comparisons'][condition]
            m=item['reference']['metrics'];e=item['effects']['all']
            passed=all(control_summary['checks'][condition].values())
            control_table+=f"| {condition} | {m['sr']*100:.2f}% | {m['mean_sg_m']:.2f} | {e['sr_gain']*100:+.2f}点 | [{e['source95_SR'][0]*100:+.2f},{e['source95_SR'][1]*100:+.2f}]点 | {'通过' if passed else '未通过'} |\n"
        control_table+=f"\n目标对照另复核{control_audit['records']}记录/{control_audit['actions']}动作；和主对照合计{control_audit['records']+audit['records']}记录/{control_audit['actions']+audit['actions']}动作。\n"
    caption=('同10个已消费区域、750题×3冻结权重的后验开发。每份权重混合750题、各层250题，'
        '柱为合并均值，空心标记为各权重，不是置信区间。原M0及Coverage3按封存哈希复用，'
        '仅新增Coverage3Radial。唯一新增因素为无已接受cue时保护相对原探索动作的起点径向延伸，'
        '已接受cue始终优先。SG为成功记0的终点曼哈顿距离；不是路径长度。开发证据不替代新区域确认。')
    graphics.style()
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.9))
    colors={'M0':'#0072B2','Coverage3':'#D55E00','Coverage3Radial':'#009E73'}
    names={'M0':'原M0','Coverage3':'三步覆盖','Coverage3Radial':'三步覆盖＋径向保护'}
    titles=[];xs=np.arange(4)
    for panel,(key,scale,ylabel,title) in enumerate((('sr',100,'SR（%）','(a) 混合与各分层成功率'),
            ('mean_sg_m',1,'平均终点SG（米）','(b) 同队列终点距离'))):
        ax=axes[panel];ax.set_axisbelow(True);ax.grid(axis='y',alpha=.65)
        for j,name in enumerate(arms):
            x=xs+(j-1)*.25;values=[metric(arms[name],c)[key]*scale for c in COHORTS]
            ax.bar(x,values,width=.23,color=colors[name],edgecolor='#243746',linewidth=.6,zorder=2)
            for seed,shape in enumerate(('o','^','s')):
                ys=[metric(seeds[name][seed],c)[key]*scale for c in COHORTS]
                ax.scatter(x+(seed-1)*.035,ys,marker=shape,facecolor='white',edgecolor='#243746',s=20,linewidth=.7,zorder=4)
        ax.set_xticks(xs,['混合\n750题','长距离\n250题','接缝\n250题','内部\n250题'])
        ax.tick_params(axis='x',length=0,pad=6);ax.set_ylabel(ylabel);titles.append(ax.set_title(title,loc='left'))
    axes[0].set_ylim(0,105);axes[0].set_yticks(range(0,101,20))
    highest=max(metric(seeds[n][s],c)['mean_sg_m'] for n in seeds for s in range(3) for c in COHORTS)
    axes[1].set_ylim(0,np.ceil((highest+100)/500)*500)
    handles=[Patch(facecolor=colors[n],edgecolor='#243746',label=names[n]) for n in arms]
    handles.extend(Line2D([],[],marker=m,linestyle='',markerfacecolor='white',markeredgecolor='#243746',label=f'权重{s}') for s,m in enumerate(('o','^','s')))
    legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=6,frameon=False,fontsize=9.5)
    fig.subplots_adjust(left=.075,right=.985,top=.78,bottom=.20,wspace=.35)
    fig.text(.5,.035,f'10个已消费区域；三冻结权重；{status}。完整恢复/损伤及区域bootstrap见报告。',ha='center',fontsize=9)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    assert all(not legend.get_window_extent(renderer).overlaps(t.get_window_extent(renderer)) for t in titles)
    files=[]
    for ext in ('png','pdf','svg'):
        buffer=BytesIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');fig.savefig(buffer,format=ext,dpi=300,bbox_inches='tight')
        assert not any('Glyph' in str(w.message) for w in caught)
        p=FIG/'数据结果图'/f'{STEM}.{ext}'
        with p.open('xb') as f:f.write(buffer.getvalue())
        files.append(dict(path=rel(p),sha256=digest(p)))
    plt.close(fig)
    long=diagnostic['strata_outcomes']['long_distance']
    report=f'''# 覆盖接管径向保护开发验证

2026-10-03。**{status}**。固定混合SR {arms['Coverage3Radial']['metrics']['sr']*100:.2f}%，对M0 {effect['sr_gain']*100:+.2f}点；默认M0保持。本批为已消费10区的后验开发，未进行新区域确认。

## 诊断与唯一因素

先只读分析M0/Coverage3共4500条旧记录、{diagnostic['rechecked_actions']}动作并独立计数核验。Coverage3的{long['harmed']['records']}条长距离损伤中{long['harmed']['episodes_with_contraction']}条出现过覆盖动作比原探索动作更靠近起点；{long['recovered']['records']}条长距离恢复中也有{long['recovered']['episodes_with_contraction']}条出现该信号。全队列{diagnostic['overall']['changed_steps']}次接管中的{diagnostic['overall']['contraction_steps']}次符合信号。它是后验关联，不是错误判别器或因果证明。

唯一候选Coverage3Radial在原Coverage3上添加相对起点径向保护，保留原三步/至少2格规则。保护只否决胜出覆盖动作，不重新选择第二候选；只比较下一格的公开曼哈顿半径。已接受cue优先执行，包括向起点靠近的目标动作；原探索器亦可回缩。控制器五输入不包含真值、距离标签、地区身份或未来图像。原模型、0.50、300米每格、10×10/B20冻结。

## 全部主结果与次要机制对照

同750个固定任务×3权重新增2250条候选轨迹。每权重混合750题、每层250题，合并分母2250/750。主比较对象始终是M0；对Coverage3的改善只作次要机制证据，不替代默认升级门槛。

{chr(10).join(table)}

{chr(10).join(physical)}

失败SG只对失败题求均值；全队列SG成功记0，有效移动距离另报。不得混用SG与移动距离。

相较M0，失败题数从{arms['M0']['metrics']['failed_episodes']}减为{arms['Coverage3Radial']['metrics']['failed_episodes']}，但失败题SG从{arms['M0']['metrics']['mean_failed_sg_m']:.2f}升为{arms['Coverage3Radial']['metrics']['mean_failed_sg_m']:.2f}米。两组失败题集合不同，不能把总体SG下降写成所有失败都更接近目标；条件均值与失败数量一并保留。

{weight_table}

对M0恢复{summary['recovered']}、损伤{summary['harmed']}，净增{summary['recovered']-summary['harmed']}成功；对原Coverage3恢复{secondary['recovered']}、损伤{secondary['harmed']}，净增{secondary['recovered']-secondary['harmed']}。配对明细全部保留。

混合SR差的10区域配对95%bootstrap区间[{effect['source95_SR'][0]*100:+.2f},{effect['source95_SR'][1]*100:+.2f}]点，SG差区间[{effect['source95_SG'][0]*300:+.2f},{effect['source95_SG'][1]*300:+.2f}]米，4000次/seed7317。每区域先平均三权重再整组抽取，权重不是独立源图。已查看开发队列的区间不构成未见地区泛化证明。

{intervals}

内部目标SG变化的区域区间包含0，不能把所有分层的均值改善写成稳定优势。对原Coverage3混合SR差{secondary_effect['sr_gain']*100:+.2f}点，95%区域区间[{secondary_effect['source95_SR'][0]*100:+.2f},{secondary_effect['source95_SR'][1]*100:+.2f}]点；该次要SR差也未确认稳定为正。正式开发门槛以对M0的预先规定指标判定，不在看到区间后改变。

## 冻结验收与后续

{gate_table}

{control_text} 未在看到结果后更改门槛、分母、保护或默认。开发与目标证据全部通过才能另立新区域确认协议；当前未下载或消费新区域。

{control_table}

{next_text}

## 核验与结论边界

10项行为/权限/预算/独立规划/分层门槛测试通过。新主评测{audit['records']}记录/{audit['actions']}动作全部以显式神经方程和独立集合式保护重放，并核对终局、物理距离、配对和bootstrap。独立实现由同一会话运行，不冒称另一位人工审核者。旧文件与模型按注册SHA只读核验；0训练、0云端、默认保持。原S2/S3/S4、SwissView及实际区域确认判定有效。

本批仍为Massachusetts发布航拍影像上的本地切图仿真；Sat2Cap预训练地理范围未知，不支持异时、任意角度、跨模态或实飞结论。现有14区域已消费，不能再称新区域确认。

[冻结协议](执行协议.md) / [判定](验收结论.json) / [主复核](主对照/独立复核.json) / [机制对照](主对照/Coverage3机制对照汇总.json) / [诊断复核](离线诊断/独立算术复核.json)。

![图48](../../../../../绘图/数据结果图/{STEM}.png)

图注：{caption}
'''
    write(REPORT,report)
    write(RUN/'README.md',f'# 覆盖接管径向保护验证 v1\n\n{status}；默认M0保持。\n\n[报告](径向保护开发验证报告.md) / [判定](验收结论.json) / [独立复核](主对照/独立复核.json)。\n\n已消费10区的后验开发；4500旧记录诊断、2250新增主轨迹；条件性后续依冻结协议。\n')
    write(CAPTION,'# 覆盖接管径向保护图注\n\n'+caption+'\n')
    item=dict(id=STEM,files=files,caption=caption,protocol='consumed10regions/750tasks/3weights; grid10/B20/300m; one radial guard; posthoc development',sources=[rel(p) for p in sources])
    write(MANIFEST,dict(date='2026-10-03',source_sha256={rel(p):digest(p) for p in sources},
        reproduction_script=rel(Path(__file__)),script_sha256=digest(Path(__file__)),main_summary=summary,
        secondary_summary=secondary,diagnostic_summary=diagnostic,verdict=verdict,per_seed=seeds,figures=[item],
        other_outputs_sha256={rel(p):digest(p) for p in (REPORT,RUN/'README.md',CAPTION)}))
    print(dict(report=rel(REPORT),figure=STEM,status=status),flush=True)


if __name__=='__main__':main()
