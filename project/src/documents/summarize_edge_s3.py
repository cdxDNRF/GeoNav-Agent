"""Generate audited S3 interpretation and standalone vector result figures."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1'
FIG=ROOT/'绘图'


def read(p):return json.loads(p.read_text('utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    receipt=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');summary=read(OUT/'对照汇总.json');reg=read(OUT/'预登记.json')
    assert receipt['audit_sha256']==digest(OUT/'独立复核.json') and audit['status']=='passed'
    assert receipt['summary_sha256']==digest(OUT/'对照汇总.json')
    report=OUT/'独立源图确认报告.md';source=FIG/'绘图数据/S3独立源图确认_v1.json'
    names=['09_S3独立源图结果','10_S3分距离与源图增益']
    paths=[report,source,FIG/'S3独立源图图注.md']+[FIG/f'数据结果图/{n}.{fmt}' for n in names for fmt in ('png','pdf','svg')]
    if any(p.exists() for p in paths):raise ValueError('immutable S3 publication already exists')
    main=summary['averages']['Edge']['CueFull'];base=summary['averages']['Edge']['Baseline']
    edge=summary['effects']['NoTargetBaseline'];ci=edge['source_interval']['interval95'];strongest=summary['strongest_rule']
    labels=[('Edge','Baseline','NoTarget探索'),('ZeroEdge','CueFull','同容量置零'),('Edge','CueFull','Edge真实目标'),
        ('Edge','CueMean','均值目标'),('Edge','CueWrong','错误目标')]
    rows=['# 冻结边缘线索S3：独立源图确认','',
        '最终判定以[验收结论](验收结论.json)为准；运行汇总中的audit_pending与正式未判定字段保留生成时状态，不回写。','',
        f"**S3工程验收：{'通过' if receipt['formal_S3_passed'] else '未通过'}；目标贡献迁移检查：{'全部通过' if receipt['target_transfer_passed'] else '存在未通过项'}。** 默认配置与S2结果原样保持。",'',
        f"固定test250×3；10张原train/val之外的源图文件、{reg['protocol']['unique_routes']}条不同路线；5×5、B10、C4…8各50题。使用S2原训练权重、原均值、原阈值0.50，无训练、调参、模型集成或云端调用。",'',
        '| 方法/目标条件 | 三训练种子SR | 成功 / 计划 | 平均SR | SG | 各种子重访率 |',
        '|---|---|---|---:|---:|---|']
    for a,c,label in labels:
        v=summary['averages'][a][c]
        rows.append(f"| {label} | {' / '.join(f'{p:.1%}' for p in v['sr_by_seed'])} | {sum(v['successes_by_seed'])}/750 | {v['sr_mean']:.2%} | {v['sg_mean']:.3f} | {' / '.join(f'{p:.2%}' for p in v['repeat_by_seed'])} |")
    for n,r in summary['rules'].items():
        v=r['metrics'];rows.append(f"| {n}（确定性，1轮） | {v['sr']:.1%} | {v['successes']}/250 | {v['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} | {v['repeat_visit_rate_micro']:.2%} |")
    rows+=['',f"真实目标相对同题NoTarget探索的SR增益为 **{edge['gain']*100:+.2f}个百分点**，{edge['positive_seeds']}/3种子正增益；按10源图先平均种子再成组bootstrap的95%区间为[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]点。",'',
        '## 原S3门槛逐项判定','',
        'SR≥55%、SG≤2.0、比同题最强预登记规则SR至少+5点且SG不更差、至少6/10源图SR正增益、每种子R≤10%，全部计划任务正常终态并完成复核。目标贡献尺度另外报告，不事后更换工程门槛。','',
        '| 冻结工程检查 | 结果 |','|---|---|']
    for k,v in summary['engineering_checks'].items():rows.append(f'| {k} | {"通过" if v else "未通过"} |')
    rows+=['',f"最强预登记规则为{strongest}；真实方法对该规则在{summary['effects'][strongest]['positive_sources']}/10张源图上有SR正增益。主方法三个种子的Q均为100%；三种子SR极差{summary['diagnostics']['SR_seed_range']*100:.2f}点，作为诊断保留。",'',
        '## 目标控制、规则比较和不确定性','',
        '| 真实目标减对照 | SR差（点） | 正增益种子 | 正增益源图 | SR差95%区间（点） | SG差 | SG差95%区间 | 目标迁移尺度 |',
        '|---|---:|---:|---:|---|---:|---|---|']
    for n,e in summary['effects'].items():
        lo,hi=e['source_interval']['interval95'];sl,sh=e['sg_source_interval']['interval95']
        check='规则比较' if n not in summary['target_transfer_checks'] else ('通过' if summary['target_transfer_checks'][n] else '未通过')
        seeds=f"{e['positive_seeds']}/3" if n not in summary['rules'] else '3配对（规则1轮）'
        rows.append(f"| {n} | {e['gain']*100:+.2f} | {seeds} | {e['positive_sources']}/10 | [{lo*100:+.2f}, {hi*100:+.2f}] | {e['sg_change']:+.3f} | [{sl:+.3f}, {sh:+.3f}] | {check} |")
    rows+=['','目标检查沿用S2尺度：平均SR≥+5点、至少2/3种子正增益、SG不更差。CI是否支持正增益单列，不把工程过线自动说成统计上稳定优于所有对照。规则复用仅作配对，仍只有10个源图簇；所有区间使用2000次bootstrap、seed3031。','',
        '## 分距离与分源图','',
        '| C档 | 探索SR | 真实目标SR | 真实目标SG |','|---|---:|---:|---:|']
    for d in range(4,9):
        b=base['by_distance'][str(d)];v=main['by_distance'][str(d)]
        rows.append(f"| C{d}（50题×3） | {b['sr']:.2%} | {v['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} |")
    rows+=['','| 源图区域 | 原源图文件 | 探索SR | 真实目标SR | SR差（点） | 最强规则SR | 真实目标SG |','|---|---|---:|---:|---:|---:|---:|']
    tasks=read(OUT/'导航任务.json');tile={e['area']:e['source_tile'] for e in tasks}
    for area,v in main['by_source'].items():
        b=base['by_source'][area];rule=summary['rules'][strongest]['by_source'][area]
        rows.append(f"| {area} | {tile[area]} | {b['sr']:.2%} | {v['sr']:.2%} | {(v['sr']-b['sr'])*100:+.2f} | {rule['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} |")
    rows+=['','## 可追溯性与结论范围','',
        f"204项测试通过；3750条神经轨迹、{audit['neural_steps']}个动作及500条规则全部复核。重新提取250块test图的全局/局部/边缘特征，最大全局误差{audit['images']['global_max_abs_error']:.3g}，局部误差{audit['images']['local_max_abs_error']:.3g}。错误目标有{reg['wrong_cue_distance_exceptions']}/250项无法匹配起点距离，已在推理前固定并完整保留，不能把该控制解释为全部距离严格匹配。",'',
        '评测端两份独立加载权重逐题重置并复算；另行实现的S3统计和图像重构，加上冻结S2独立手算动作逻辑，检查公开输入、目标通道、概率、动作、终止和12个SR/SG区间。实施与复核为同一执行代理，不称作不同人员复审。源数据、任务、权重、均值、源码快照及S2原件哈希均核验，默认加载三个种子各重放第一道test题成功。','',
        'train/val/test的源文件名称和图块组文件哈希两两隔离；缺少地理坐标，因此不能保证地理区域绝不重叠。此前test做过Random基础校验；本轮是冻结Edge方案确认，不宣称该数据从未被任何流程读取。221条不同路线与三训练种子不能冒充750张独立地图。','',
        '结论只适用于本地MASA切图协议：同源、连续、方向一致的300×300RGB图块。异时、旋转、非连续裁块、跨视角与跨数据集尚未验证；该方案是本地训练过的单智能体模块系统，不是已实现的零训练多智能体系统。NoTarget是从头训练探索器，ZeroEdge为同容量原校准None头；并非新训练一个完整架构NoTarget视觉头。','',
        'S2的val89.33%与本轮test结果来自不同题和地图，分开列报；方法增益只用本轮同题对照计算。已见本轮test不能继续用于新方法调参后冒充未触及确认集。','',
        '## research.md与下一步','',
        'research.md支持先确认可信目标线索边界、保留NoTarget/Frontier，再分别研究子目标、记忆和预算规划。其“默认未变/S2未通过”是旧快照，本轮未修改原研究记录，也未重新核验其中论文引用。采纳说明见选题报告相关/研究资料与当前路线衔接_v1.md；这些新因素均未接入S3。','',
        ('S3工程门槛和复核已通过，允许开始S4的10×10实验准备。先独立检查100图块数据生成、预算与线索提取是否兼容并预登记同题规则/目标控制；新网格需另外定义源图分组、任务与指标，不能只把5×5参数改为10。原S4规模要求至少20个源图区域、每C每区域5题、三轮；若仍只有10张test源图，不通过切分更多子区域冒充20个独立地图簇，规模不足只能报探索结果。跨数据集和research中的新机制分别成批；本轮停止在S3，不自动运行S4。' if receipt['formal_S3_passed'] else
         '本轮S3未通过项永久保留，回开发数据研究下一版单因素修改；不按test挑新阈值、模型或种子，不直接扩大正式实验。下一版独立确认需要新的未用于开发地图。'),'',
        '[预登记](预登记.json) / [完整复核](独立复核.json) / [默认加载核验](默认加载核验.json)。']
    report.write_text('\n'.join(rows)+'\n','utf-8')

    graphics.style();captions=[]
    # Five categories, direct labels: NoTarget and matched ZeroEdge trajectories coincide.
    values=[summary['averages']['Edge'][c] for c in ('Baseline','CueFull','CueMean','CueWrong')]
    sg_seeds=[];metric_sources=[]
    for c in ('Baseline','CueFull','CueMean','CueWrong'):
        series=[]
        for s in range(3):
            p=OUT/f'Edge_s{s}/导航_{c}_结果.json'
            assert digest(p)==audit['artifacts_sha256'][p.relative_to(OUT).as_posix()]
            series.append(read(p)['metrics']['mean_sg_all_episodes']);metric_sources.append(p)
        sg_seeds.append(series)
    rule=summary['rules'][strongest]['metrics']
    plot_labels=['探索/置零','真实目标','均值目标','错误目标',strongest]
    colors=[graphics.GRAY,graphics.BLUE,graphics.ORANGE,graphics.TEAL,graphics.GRAY]
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.8));graphics.frames(axes)
    graphics.bars(axes[0],plot_labels,[v['sr_mean']*100 for v in values]+[rule['sr']*100],colors,percent=True)
    for i,v in enumerate(values):
        axes[0].texts[i].set_position((i,max(v['sr_by_seed'])*100+1.5))
        for s,(off,marker) in enumerate([(-.13,'o'),(0,'^'),(.13,'s')]):
            axes[0].scatter(i+off,v['sr_by_seed'][s]*100,marker=marker,s=22,facecolor='white',edgecolor='#1D2B34',linewidth=.8,zorder=4)
    axes[0].set_ylim(0,109);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR：成功率（%）');axes[0].set_title('(a) 固定test250×3',loc='left')
    graphics.bars(axes[1],plot_labels,[v['sg_mean'] for v in values]+[rule['mean_sg_all_episodes']],colors,decimals=3)
    for i,series in enumerate(sg_seeds):
        axes[1].texts[i].set_position((i,max(series)+.025))
        for value,off,marker in zip(series,(-.13,0,.13),('o','^','s')):
            axes[1].scatter(i+off,value,marker=marker,s=22,facecolor='white',edgecolor='#1D2B34',linewidth=.8,zorder=4)
    axes[1].set_ylim(0,max([v['sg_mean'] for v in values]+[rule['mean_sg_all_episodes']])*1.30);axes[1].set_ylabel('SG：平均终点距离（格）');axes[1].set_title('(b) 同题SG（包含失败）',loc='left')
    for ax in axes:ax.tick_params(axis='x',labelsize=8.8)
    fig.subplots_adjust(left=.08,right=.985,bottom=.19,top=.88,wspace=.31)
    fig.text(.5,.025,'10张test源图；5×5，B=10；白色标记：三个训练种子；规则仅1轮。',ha='center',fontsize=9)
    caption=f"冻结边缘线索在独立源文件test250上将SR从{base['sr_mean']:.2%}提高至{main['sr_mean']:.2%}，SG从{base['sg_mean']:.3f}降至{main['sg_mean']:.3f}。SR为成功/计划，SG包含成功与失败终点；探索与置零轨迹一致，均值/错误目标干预所有目标通道。10源图、221不同路线、三个训练权重；白色圆/三角/方块分别为seed0/1/2而非CI，最强预登记规则{strongest}只运行一轮。源文件隔离不等于地理绝无重叠，不能据此宣称跨数据集泛化。"
    captions.append(caption);graphics.save(fig,FIG/'数据结果图',names[0],caption,[str((OUT/'对照汇总.json').relative_to(ROOT))],'test250/10source files/3training seeds; frozen 5x5/B10')

    fig,axes=plt.subplots(1,2,figsize=(7.6,3.9));graphics.frames(axes)
    for c,label,color,marker,ls in [('Baseline','探索/置零',graphics.GRAY,'s','--'),('CueFull','真实目标',graphics.BLUE,'o','-'),('CueWrong','错误目标',graphics.TEAL,'^',':')]:
        v=summary['averages']['Edge'][c];axes[0].plot(range(4,9),[v['by_distance'][str(d)]['sr']*100 for d in range(4,9)],label=label,color=color,marker=marker,ls=ls,lw=1.7)
    axes[0].set_ylim(0,107);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_xticks(range(4,9),[f'C{d}' for d in range(4,9)])
    axes[0].set_xlabel('初始曼哈顿距离（格）');axes[0].set_ylabel('SR（%）');axes[0].legend(frameon=False,fontsize=9,loc='lower right');axes[0].set_title('(a) 每档50题×3',loc='left')
    comparisons=[('NoTargetBaseline','真实 − 探索'),('ZeroEdgeFull','真实 − 置零'),('MeanCue','真实 − 均值'),('WrongCue','真实 − 错误'),('Frontier','真实 − Frontier'),('FixedRegion','真实 − FixedRegion')]
    axes[1].grid(False);axes[1].grid(axis='x');axes[1].axvline(0,color='#34424F',ls='--',lw=.8)
    for i,(n,label) in enumerate(comparisons):
        v=summary['effects'][n]['source_interval'];mu=v['mean']*100;lo,hi=np.array(v['interval95'])*100
        axes[1].errorbar(mu,i,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color=graphics.BLUE,capsize=3,ms=5)
    axes[1].set_yticks(range(6),[p[1] for p in comparisons],fontsize=9);axes[1].invert_yaxis();axes[1].set_xlabel('SR差及源图95%区间（百分点）');axes[1].set_title('(b) 10源图成组bootstrap',loc='left')
    fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.88,wspace=.61)
    fig.text(.5,.025,'距离只用于评测分组；源图先平均三种子，再2000次成组重采样。',ha='center',fontsize=9)
    caption=f"真实目标相对探索的test SR收益为{edge['gain']*100:+.2f}点，源图成组95%区间[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]点。左图按初始距离等量分组，C档结构不同，曲线不表示连续难度变化；真距离不提供给Agent。右图为真实目标减各对照的配对SR差，先在每源图内平均三种子，再以10源图为簇作2000次bootstrap；规则仍是一次确定性运行，不把其复制当独立重复。区间仅覆盖本地源文件组，不覆盖尚未验证的异时、旋转、跨视角和地理分布。"
    captions.append(caption);graphics.save(fig,FIG/'数据结果图',names[1],caption,[str((OUT/'对照汇总.json').relative_to(ROOT))],'test250; source-file grouped paired bootstrap')
    for item in graphics.ITEMS:item['reproduction_script']='project/src/documents/summarize_edge_s3.py'
    manifest=dict(data_scope='audited frozen S3 independent source-file confirmation',
        source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [OUT/'对照汇总.json',OUT/'验收结论.json',OUT/'独立复核.json',OUT/'预登记.json',*metric_sources]},
        interpretation_sha256=digest(report),figures=graphics.ITEMS,script_sha256=digest(Path(__file__)),
        plot_style_helper_sha256=digest(Path(graphics.__file__)),
        measurements=summary['averages'],effects=summary['effects'])
    source.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    (FIG/'S3独立源图图注.md').write_text('# S3独立源图确认图注\n\n'+'\n\n'.join(f'## {n}\n\n{c}' for n,c in zip(names,captions))+'\n\n来源与哈希：[本批绘图数据](绘图数据/S3独立源图确认_v1.json)。\n','utf-8')
    print(json.dumps(dict(formal_S3_passed=receipt['formal_S3_passed'],report=str(report),figures=names),ensure_ascii=False))


if __name__=='__main__':main()
