"""Audited robustness boundaries, data provenance explanation and reusable scientific plots."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/SwissView/评测结果/目标扰动鲁棒性_v1'
FIG=ROOT/'绘图'
VARIANTS=('Clean','Bright080','JPEG25','Blur1','Crop8','Rot90CW','Ring8')
LABELS=dict(Clean='无扰动',Bright080='亮度 ×0.8',JPEG25='JPEG质量25',Blur1='模糊半径1',Crop8='裁边8后缩放',Rot90CW='顺时针90°',Ring8='灰色边缘8')
def read(p):return json.loads(p.read_text('utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text('utf-8').splitlines() if x]
digest=graphics.digest


def diagnostics():
    tasks={e['episode_id']:e for e in read(OUT/'导航任务.json')};result={}
    def near(a,b):return abs(a//5-b//5)+abs(a%5-b%5)
    for v in VARIANTS:
        seeds=[]
        for s in range(3):
            rows=lines(OUT/f'Edge_s{s}/导航_{v}_CueFull_轨迹.jsonl');failed=[r for r in rows if not r['success']]
            opportunities=sum(any(near(event['patch_id'],tasks[r['episode_id']]['goal'])==1 for event in r['trajectory'][:-1]) for r in failed)
            accepted=correct=0
            for r in rows:
                for i,d in enumerate(r['decisions']):
                    if d['cue_action'] is not None:accepted+=1;correct+=r['trajectory'][i+1]['patch_id']==tasks[r['episode_id']]['goal']
            seeds.append(dict(seed=s,failures=len(failed),failed_with_adjacent_opportunity=opportunities,
                failed_without_adjacent_opportunity=len(failed)-opportunities,accepted_cues=accepted,correct_cues=correct,
                actual_accepted_precision=correct/accepted if accepted else None))
        result[v]=seeds
    return dict(scope='posthoc evaluator truth; no calibration or policy input',variants=result)


def main():
    receipt=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');summary=read(OUT/'对照汇总.json');reg=read(OUT/'预登记.json')
    assert receipt['robustness_confirmation_complete'] and audit['status']=='passed'
    assert digest(OUT/'独立复核.json')==receipt['audit_sha256'] and digest(OUT/'对照汇总.json')==receipt['summary_sha256']
    report=OUT/'鲁棒性与数据范围报告.md';diagpath=OUT/'失败与线索诊断.json'
    names=['15_目标扰动成功率与距离','16_目标扰动配对收益与退化']
    source=FIG/'绘图数据/目标扰动鲁棒性_v1.json'
    if any(p.exists() for p in [report,diagpath,source]) or any((FIG/'数据结果图'/f'{n}.{fmt}').exists() for n in names for fmt in ('png','pdf','svg')):
        raise ValueError('immutable reporting artifacts exist')
    diag=diagnostics();diagpath.write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n','utf-8')
    items=summary['variants'];clean=items['Clean']['Full'];base=items['Clean']['NoTarget']
    retained=[v for v in VARIANTS[1:] if items[v]['performance_retained'] and items[v]['target_gain_retained']]
    lost=[v for v in VARIANTS[1:] if v not in retained]
    text=['# 目标扰动鲁棒性与当前数据范围','',
        '**当前不是20题试跑，而是已完成正式规模验证；地理推断仍只有有限源文件。** 本批在同一SwissView20图/500题/三Masa训练权重上完成七个目标条件×Full/NoTarget共21000条在线轨迹并全量复核。它是已见图上的配对压力测试，不是新的独立地域确认。','',
        f"六个固定强度中，同时保持预定性能与目标收益的条件：{', '.join(LABELS[v] for v in retained) or '无'}；未全部保持：{', '.join(LABELS[v] for v in lost) or '无'}。全部记录/复核完成不等于全条件鲁棒，原干净设置的正式验收不改变。",'',
        '## 数据从哪里来、规模怎么算','',
        '| 数据 | 来源 | 本地与本项目用途 |','|---|---|---|',
        '| Masa | 美国马萨诸塞州Massachusetts Buildings公开航拍 | 原图train137/val4/test10；探索器109拟合，边缘头87拟合+22校准；开发留出28 |',
        '| SwissView100 | EPFL/GeoExplorer作者公开瑞士航拍子集 | 本地100图；试跑4图/20题×3、正式另20图/500题×3；本批沿用正式20图，另外76未跑策略 |',
        '| 其他本地数据 | MM-GAG与SwissViewMonuments | 本批未用于训练/正式导航，地面图或跨视角另开协议 |','',
        '[Masa作者数据页](https://www.cs.toronto.edu/~vmnih/data/) / [SwissView官方数据卡](https://huggingface.co/datasets/EPFL-ECEO/SwissView) / [GeoExplorer项目](https://limirs.github.io/GeoExplorer/)。当前都是本地航拍切图输入，原图来自公开数据，不是模型生成；目标扰动是本批程序确定性合成。', '',
        '正式500题包含424个不同起终点组合与76重复抽样，三个训练权重重复同题，不增加独立地图数量。21,000轨迹也仍来自这20源文件，不是21,000张地图。20图为固定连续编号便利样本；源文件成组区间不代表全国随机抽样或确认地理绝无重叠。试跑与正式源文件不重合；本批压力测试有意重复正式地图以控制差异。Sat2Cap预训练覆盖未知。', '',
        '## 配对成绩与预定判定','',
        '| 目标条件 | 成功/1500 | SR | SG（格） | 对Clean SR差（点） | 对探索SR差（点） | 性能保持 | 目标收益保持 |','|---|---:|---:|---:|---:|---:|---|---|']
    for v in VARIANTS:
        x=items[v];m=x['Full']
        text.append(f"| {LABELS[v]} | {sum(m['successes_by_seed'])}/1500 | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} | {x['versus_Clean']['gain']*100:+.2f} | {x['versus_NoTarget']['gain']*100:+.2f} | {'是' if x['performance_retained'] else '否'} | {'是' if x['target_gain_retained'] else '否'} |")
    text+=['',f"NoTarget在所有七条件都为SR{base['sr_mean']:.2%}、SG{base['sg_mean']:.3f}，每条轨迹均在线复现父批，目标改变不会进入探索输入。Clean SR{clean['sr_mean']:.2%}/SG{clean['sg_mean']:.3f}，完整1500条输入/概率/动作/终态复现父批。所有Q=100%，神经边界动作0。SG为所有成功与失败终点均值，成功为0。",'',
        '性能保持：SR相对Clean下降≤5点、SG增加≤0.30；目标收益保持：相对同扰动NoTarget SR≥+5点、≥2正种子、SG不更差。是运行前工程解释阈值，非论文认证。原跨域SR45/SG2.5/规则+5门槛另列JSON；它不能替代目标贡献。','',
        '| 条件 | 三份权重SR | 对探索SR95%源文件区间（点） | 六项校正尾区间（点） | 对Clean SR95%区间（点） |','|---|---|---|---|---|']
    for v in VARIANTS:
        x=items[v];e=x['versus_NoTarget'];lo,hi=e['source_interval']['interval95'];al,ah=e['six_stress_adjusted_SR_interval'];cl,ch=x['versus_Clean']['source_interval']['interval95']
        seeds=' / '.join(f'{s:.1%}' for s in x['Full']['sr_by_seed'])
        adjusted='不适用Clean' if v=='Clean' else f'[{al*100:+.2f},{ah*100:+.2f}]'
        text.append(f"| {LABELS[v]} | {seeds} | [{lo*100:+.2f},{hi*100:+.2f}] | {adjusted} | [{cl*100:+.2f},{ch*100:+.2f}] |")
    text+=['','先每源文件平均三份权重配对差，再seed3031/2000次bootstrap；普通95%区间逐项解释，六压力条件的SR对探索收益额外给0.05/(2×6)尾分位参考，近似Bonferroni式更保守区间。有限20文件的百分位法不保证精确/全国覆盖，普通区间不可称六项同时95%保证。SG区间及逐图差保存对照汇总JSON。','',
        '| 目标条件 | C4 SR | C5 SR | C6 SR | C7 SR | C8 SR | R三种子 |','|---|---:|---:|---:|---:|---:|---|']
    for v in VARIANTS:
        m=items[v]['Full'];text.append('| '+LABELS[v]+' | '+' | '.join(f"{m['by_distance'][str(d)]['sr']:.2%}" for d in range(4,9))+' | '+' / '.join(f'{r:.2%}' for r in m['repeat_by_seed'])+' |')
    text+=['','## 适用边界与后置诊断','',
        '| 目标条件 | 接受线索数 | 正确接受数 | 实际接受精度 | 失败从未相邻/全部失败 |','|---|---:|---:|---:|---|']
    for v in VARIANTS:
        values=diag['variants'][v];n=sum(x['accepted_cues'] for x in values);k=sum(x['correct_cues'] for x in values)
        precision=f'{k/n:.2%}' if n else '无接受'
        text.append(f"| {LABELS[v]} | {n} | {k} | {precision} | {sum(x['failed_without_adjacent_opportunity'] for x in values)}/{sum(x['failures'] for x in values)} |")
    text+=['','实际接受精度按在线当前位置判断给定动作是否立即到达真目标，不能由总体SR推断每次视觉判断正确。失败机会只查看动作前位置，描述相关性，不证明唯一失败原因或可实现的上限。', '',
        '仅目标受扰动，当前图与地图坐标不变；亮度/压缩/模糊/裁剪/旋转/边缘遮挡分别单独操作，全部目标全局/局部/边缘通道从新像素重算。Crop与Ring也会改变语义特征，不把其效果单独归因于边缘头通道。90度旋转既破坏接缝连续性又改变目标内部方向，Agent未提供角度/反旋转辅助，不能把它解释为“只破坏某一个特征”。', '',
        '每种条件只有一个固定强度；通过不能推断所有亮度、所有压缩率或所有扰动幅度都鲁棒。合成变换不是真实季节/拍摄时间/视角变化；尚未完成真实异时、地面图或跨视角验证。仍为训练过的单智能体模块系统，权重/均值/阈值原样。','',
        '## 可追溯性与后续','',
        f"240项运行前测试通过；独立重建当前500图与3500目标变体，编码器全局/局部最大误差{audit['images']['global_max_abs_error']:.3g}/{audit['images']['local_max_abs_error']:.3g}。全部21000神经轨迹/{audit['neural_actions']}动作、1000原规则记录、{audit['ordinary_SR_SG_intervals']}个普通SR/SG区间及全部校正尾分位核验。规则按父批哈希复用并逐条重查，未复制成7套独立观测。原件/默认/权重哈希保持一致。",'',
        '原图、任务、变换参数、PNG/RGB、特征、源码、权重和完整在线轨迹可追溯；实施与独立算法路径由同一执行代理运行，不声称不同人员复审。原记录audit_pending为采集快照，最终状态看验收与独立复核，不回写原日志。', '',
        '下一项应根据本轮已登记失效条件只选一个改进，在Masa开发数据研究，再用未用于开发的SwissView源图确认。可以先比较无训练图像预处理/旋转候选方法与当前冻结策略，但不能把目标真值/角度泄漏给策略，也不能在本次20正式图上反复调到过线。需要真实异时或跨视角结论时，另准备对应数据与任务。', '',
        '[运行前登记](预登记.json) / [完整复核](独立复核.json) / [验收结论](验收结论.json) / [后置诊断](失败与线索诊断.json)。']
    report.write_text('\n'.join(text)+'\n','utf-8')
    graphics.style();graphics.ITEMS.clear();captions=[]
    # Horizontal panels leave enough room for seven named single-factor conditions.
    fig,axes=plt.subplots(1,2,figsize=(7.6,4.25))
    for ax in axes:ax.set_axisbelow(True);ax.grid(axis='x');ax.set_yticks(range(7),[LABELS[v] for v in VARIANTS],fontsize=9.5);ax.invert_yaxis()
    axes[1].set_yticklabels([])
    maxsg=max([read(OUT/f'Edge_s{s}/导航_{v}_CueFull_结果.json')['metrics']['mean_sg_all_episodes'] for v in VARIANTS for s in range(3)])
    for i,v in enumerate(VARIANTS):
        m=items[v]['Full'];sr=m['sr_mean']*100;sg=m['sg_mean']
        axes[0].barh(i,sr,color=graphics.BLUE,height=.54,edgecolor='#34424F',linewidth=.6)
        axes[1].barh(i,sg,color=graphics.BLUE,height=.54,edgecolor='#34424F',linewidth=.6)
        sgseeds=[read(OUT/f'Edge_s{s}/导航_{v}_CueFull_结果.json')['metrics']['mean_sg_all_episodes'] for s in range(3)]
        for s,(offset,marker) in enumerate([(-.11,'o'),(0,'^'),(.11,'s')]):
            axes[0].scatter(m['sr_by_seed'][s]*100,i+offset,marker=marker,s=20,facecolor='white',edgecolor='#1D2B34',linewidth=.7,zorder=4)
            axes[1].scatter(sgseeds[s],i+offset,marker=marker,s=20,facecolor='white',edgecolor='#1D2B34',linewidth=.7,zorder=4)
        axes[0].text(max(m['sr_by_seed'])*100+2.5,i,f'{sr:.2f}',va='center',fontsize=9)
        axes[1].text(max(sgseeds)+maxsg*.04,i,f'{sg:.3f}',va='center',fontsize=9)
    axes[0].axvline(base['sr_mean']*100,color=graphics.GRAY,ls='--',lw=1.3);axes[1].axvline(base['sg_mean'],color=graphics.GRAY,ls='--',lw=1.3)
    axes[0].set_xlim(0,106);axes[1].set_xlim(0,max(maxsg,base['sg_mean'])*1.25)
    axes[0].set_xlabel('SR：成功率（%）');axes[1].set_xlabel('SG：平均终点距离（格）')
    axes[0].set_title('(a) 目标受扰动的导航成功率',loc='left');axes[1].set_title('(b) SG含成功与失败',loc='left')
    fig.subplots_adjust(left=.21,right=.985,bottom=.17,top=.88,wspace=.21)
    fig.text(.5,.025,'SwissView已见20图/500题×3；5×5/B10；虚线：无目标探索；白色标记：三权重。',ha='center',fontsize=9)
    caption=f"固定目标扰动暴露冻结边缘策略的适用边界，无扰动SR为{clean['sr_mean']:.2%}，无目标探索为{base['sr_mean']:.2%}。每条件同一500题、20已见源文件、三训练权重，目标全通道重算且当前图/真目标/预算不变；白色圆/三角/方块为seed0/1/2，不是CI，虚线为各条件不变的探索基线。各操作仅一个预定强度，不能外推为整段范围或真实异时/跨视角鲁棒性；SG包括全部终点。"
    captions.append(caption);graphics.save(fig,FIG/'数据结果图',names[0],caption,[(OUT/'对照汇总.json').relative_to(ROOT).as_posix()],'known SwissView20/500 x3; seven target-only perturbations; no tuning')
    fig,axes=plt.subplots(1,2,figsize=(7.6,4.25))
    for ax in axes:ax.grid(axis='x');ax.set_axisbelow(True);ax.set_yticks(range(6),[LABELS[v] for v in VARIANTS[1:]],fontsize=9.5);ax.invert_yaxis();ax.axvline(0,color='#34424F',ls='--',lw=.8)
    axes[1].set_yticklabels([])
    for i,v in enumerate(VARIANTS[1:]):
        for ax,key in zip(axes,['versus_NoTarget','versus_Clean']):
            e=items[v][key]['source_interval'];mu=e['mean']*100;lo,hi=np.array(e['interval95'])*100
            ax.errorbar(mu,i,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color=graphics.BLUE,ms=5,capsize=3)
    axes[0].axvline(5,color=graphics.ORANGE,ls=':',lw=1.1);axes[1].axvline(-5,color=graphics.ORANGE,ls=':',lw=1.1)
    axes[0].set_xlabel('SR差：真实目标 − 探索（点）');axes[1].set_xlabel('SR差：扰动 − 无扰动（点）')
    axes[0].set_title('(a) 目标信息还提供收益吗',loc='left');axes[1].set_title('(b) 保留原成功率了吗',loc='left')
    fig.subplots_adjust(left=.21,right=.985,bottom=.18,top=.88,wspace=.28)
    fig.text(.5,.027,'20源文件成组95%区间；橙虚线：+5点目标收益／−5点性能保持尺度；全部条件报告。',ha='center',fontsize=9)
    caption='目标收益保持与性能保持是两项不同检查，不能由高探索成功率推断视觉鲁棒性。左图为扰动Full减同条件NoTarget，右图为扰动Full减Clean；先在每个已见源文件平均三权重配对差，再2000次成组bootstrap（seed3031）。误差线为逐项95%区间，六项更保守尾分位参考另列报告；橙虚线仅为预定SR解释尺度，完整判定还包含SG与种子一致性，不是论文合格线。'
    captions.append(caption);graphics.save(fig,FIG/'数据结果图',names[1],caption,[(OUT/'对照汇总.json').relative_to(ROOT).as_posix()],'paired source-file ordinary95% intervals; known stress-test maps')
    for item in graphics.ITEMS:item['reproduction_script']='project/src/documents/summarize_target_robustness.py'
    manifest=dict(source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [OUT/'对照汇总.json',OUT/'独立复核.json',OUT/'验收结论.json',OUT/'预登记.json']},
        report_sha256=digest(report),script_sha256=digest(Path(__file__)),style_helper_sha256=digest(Path(graphics.__file__)),figures=graphics.ITEMS,measurements=summary['variants'])
    source.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    (FIG/'目标扰动鲁棒性图注.md').write_text('# 目标扰动鲁棒性图注\n\n'+'\n\n'.join(f'## {n}\n\n{c}' for n,c in zip(names,captions))+'\n','utf-8')
    print(json.dumps(dict(report=str(report),retained=retained,lost=lost,figures=names),ensure_ascii=False))


if __name__=='__main__':main()
