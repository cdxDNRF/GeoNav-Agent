"""Report the frozen local expansion and export publication-ready, traceable plots."""
from collections import Counter
from pathlib import Path
import json
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.local_ledger_expansion import ROOT,OUT,OLD,SEEDS,read,lines,write,digest
from documents import stage_figures as graphics
from matplotlib import pyplot as plt

FIG=ROOT/'绘图';NAME='30_零调用账本完整队列成功率与距离'


def text_new(path,text):
    with Path(path).open('x',encoding='utf-8') as f:f.write(text)


def main():
    verdict=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');summary=read(OUT/'主对照/对照汇总.json')
    if not verdict['audit_passed'] or verdict['audit_sha256']!=digest(OUT/'独立复核.json') or verdict['summary_sha256']!=digest(OUT/'主对照/对照汇总.json'):
        raise ValueError('independent audit required before publication')
    gain=summary['effect'];m0,m1=summary['arms']['M0'],summary['arms']['M1'];delta=100*gain['sr_gain']
    decision='主数值门槛通过，已检查目标证据' if verdict['primary_passed'] else '未达到主数值门槛，停止该因素的后续扩展'
    paired=[];resources=[]
    for seed in SEEDS:
        base={r['episode_id']:r for r in lines(OUT/f'主对照/M0_s{seed}_CueFull_轨迹.jsonl')}
        for r in lines(OUT/f'主对照/M1_s{seed}_CueFull_轨迹.jsonl'):
            b=base[r['episode_id']]
            paired.append(dict(seed=seed,episode_id=r['episode_id'],source=r['source'],split=r['split'],distance=r['distance'],
                M0_success=b['success'],M1_success=r['success'],M0_SG=b['sg'],M1_SG=r['sg'],
                recovered=r['success'] and not b['success'],harmed=b['success'] and not r['success'],
                triggered=any(d['interaction'] for d in r['decisions']),
                changed_actions=sum(d['action']!=d['base']['action'] for d in r['decisions'])))
        for arm in ('M0','M1'):
            result=read(OUT/f'主对照/{arm}_s{seed}_CueFull_结果.json')
            resources.append(dict(seed=seed,arm=arm,elapsed_seconds=result['elapsed_seconds'],records=500,
                triggered=result['triggered'],changed_actions=result['changed_actions'],option_interruptions=result['option_interruptions']))
    write(OUT/'逐题恢复与损伤.json',paired);write(OUT/'资源汇总.json',dict(jobs=resources,cloud_calls=0,new_training_steps=0,
        timing_is_not_a_controlled_speed_benchmark=True,cache_warming_and_shared_CPU_load_may_affect_timing=True))
    tasks=read(OUT/'导航任务.json');unique_routes=len({(r['source_tile'],r['start'],r['goal']) for r in tasks})
    lo,hi=gain['source95_SR'];report=['# 零调用账本规划：完整队列扩展验证','',
        f'**{decision}。** 相对原策略平均SR变化{delta:+.2f}个百分点，SG变化{gain["sg_change"]:+.3f}格；原默认保持。零云端请求、零训练、未使用新源文件。','',
        '## 固定队列与唯一因素','',
        f'从20题点估计扩大到原S4全部500题、20已见源文件、三原冻结权重。每源每C五题，C12…16各100题；500计划题对应{unique_routes}条不同起终点路线（重复抽样保留）。M0与M1各500×3=1,500主轨迹，总3,000条；三训练权重不是API重复，也不是1,500张地图。','',
        'M0为原GRU探索＋合法动作筛选＋冻结Edge视觉线索。M1只新增原20题试验的精确100格访问账本、同一触发和确定性1–2步覆盖option，每题最多一次；排序/同分规则原样不变。每次实际移动仍更新GRU，当前新接受的cue优先打断第二步。没有模型/阈值/训练/预算/目标图变化，控制器只接收公开Observation和原动作建议，不能访问真实目标坐标、真距离、源ID或未访问图像。','',
        '运行前保存验收：平均SR≥+2个百分点、至少2/3权重严格正收益、SG不差、全部正常合法终局及独立复核。主数值通过才追加4,500条固定目标对照；未过则停止，不另改触发/排序调到过线。','',
        '## 主结果','',
        '| 冻结权重 | 原SR | 账本SR | SR差（点） | 原SG | 账本SG | 原失败恢复 | 原成功损伤 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        rec=summary['recovery_by_seed'][str(seed)]
        report.append(f"| {seed} | {100*m0['sr_by_seed'][seed]:.2f}% | {100*m1['sr_by_seed'][seed]:.2f}% | {100*gain['sr_gain_by_seed'][seed]:+.2f} | {m0['sg_by_seed'][seed]:.3f} | {m1['sg_by_seed'][seed]:.3f} | {rec['recovered_failures']} | {rec['harmed_successes']} |")
    report.append(f"| 平均 | {100*m0['sr_mean']:.2f}% | {100*m1['sr_mean']:.2f}% | {delta:+.2f} | {m0['sg_mean']:.3f} | {m1['sg_mean']:.3f} | — | — |")
    recovered=sum(r['recovered'] for r in paired);harmed=sum(r['harmed'] for r in paired)
    report+=['',f'三权重合计恢复{recovered}条原失败、损伤{harmed}条原成功，净增加{recovered-harmed}/1,500。它们是相同500题在三权重下的记录，不是新增独立地图；完整同题分母保留。', '',
        f'SR差的20源文件配对95%区间为[{100*lo:+.2f}, {100*hi:+.2f}]个百分点；SG差区间为[{gain["source95_SG"][0]:+.3f}, {gain["source95_SG"][1]:+.3f}]格。先在各源文件内平均三权重，再成组bootstrap4000次（seed5251）；不是逐条轨迹独立重采样。源文件分组不等于已核验地理独立，不以SG改善代替SR门槛。','',
        '本轮三权重方向一致、源文件SR差区间下界略高于零，支持这套开发队列上的小幅改善。收益幅度仍未到升级要求；源文件收益不均，并且保留32条原成功损伤，不能只报告恢复。','',
        '## 分层与损伤诊断','',
        '| 初始距离C | 原SR | 账本SR | 原SG | 账本SG |','|---|---:|---:|---:|---:|']
    for c in range(12,17):
        a=m0['by_distance'][str(c)];b=m1['by_distance'][str(c)]
        report.append(f"| {c} | {a['sr']:.2%} | {b['sr']:.2%} | {a['sg']:.3f} | {b['sg']:.3f} |")
    report+=['','| 原来源组 | 原SR | 账本SR | 原SG | 账本SG |','|---|---:|---:|---:|---:|']
    for split in ('dev','test'):
        a=m0['by_split'][split];b=m1['by_split'][split]
        report.append(f"| {split}（已见） | {a['sr']:.2%} | {b['sr']:.2%} | {a['sg']:.3f} | {b['sg']:.3f} |")
    positive=sum(x['sr']>1e-12 for x in gain['source_effects'].values());negative=sum(x['sr']<-1e-12 for x in gain['source_effects'].values())
    report+=['',f'源文件平均SR正/负/零变化为{positive}/{negative}/{20-positive-negative}。逐源效果在主汇总，全部逐题恢复、损伤、SG及动作改变量在[配对小数据](逐题恢复与损伤.json)。真值只用于事后评价，不参与触发或动作选择。','',
        '## 冻结判定与目标证据','']
    for key,val in summary['checks'].items():report.append(f"- {key}: {'通过' if val else '未通过'}")
    if verdict['target_controls_started']:
        target=read(OUT/'目标对照/目标证据汇总.json')
        report+=['',f"主数值门槛通过，追加M1禁用cue/均值目标/错目标共4,500条，目标证据门槛{'通过' if target['all_target_numeric_passed'] else '未通过'}。各对照要求SR至少低5点、至少2/3权重正收益及SG不差，完整结果见[目标证据](目标对照/目标证据汇总.json)。M0原控制仍按哈希引用已审计S4，不改历史判断。"]
    else:
        report+=['','主数值门槛未通过，按预定停止条件没有启动4,500条目标对照，也没有消费预留新图。原S4目标证据仍有效，但不能据此宣称新增账本策略已通过新的目标对照。']
    report+=['',
        '更大队列检查的是20题弱信号能否稳定扩展；不把这一阶段称新S2/S3独立确认，不降低门槛或把原已验收默认替换。M1为几何探索增强，视觉目标辨别来自原训练头；本轮仍是单导航实体模块系统。','',
        '## 验证与工程边界','',
        f"311项全套测试通过，新的控制器与上一批20题M1逐动作/轨迹精确复现。M0在三权重全部1,500条轨迹上逐动作/输入/结果复现旧S4；独立审计重放本轮{audit['records']}条轨迹/{audit['actions']}动作，另写公开触发、候选合法路径和排序算术，核查{audit['input_files_verified']}项图像/特征与{audit['protected_files_verified']}项历史文件。复核是同一执行代理另行实现的检查，不声称不同人员盲审。",'',
        '原始大数据、模型、缓存、旧实验和原默认保持；代码在新模块，源快照/协议/任务/哈希保留。CPU单线程、冻结权重、复用特征，不训练或调用云端。计时只作资源日志，缓存预热及共享CPU负载不同，不能据此宣布某方法推理更快。','',
        '[冻结协议](执行协议.md) / [预登记](预登记.json) / [固定任务](导航任务.json) / [主结果](主对照/对照汇总.json) / [逐题恢复损伤](逐题恢复与损伤.json) / [独立复核](独立复核.json) / [验收](验收结论.json) / [资源](资源汇总.json)']
    text_new(OUT/'完整队列验证报告.md','\n'.join(report)+'\n')
    graphics.style();graphics.ITEMS.clear();fig,axes=plt.subplots(1,2,figsize=(7.4,4.4))
    for panel,(field,scale,label) in enumerate([('sr',100,'SR：成功率（%）'),('sg',1,'SG：平均终点距离（格）')]):
        ax=axes[panel];vals=[a[field+'_mean']*scale for a in (m0,m1)]
        ax.bar([0,1],vals,color=[graphics.GRAY,graphics.BLUE],width=.6,edgecolor='#34424F',linewidth=.7,zorder=2)
        highs=[]
        for i,a in enumerate((m0,m1)):
            runs=[v*scale for v in a[field+'_by_seed']];highs.append(max([vals[i]]+runs))
            for seed,v in enumerate(runs):ax.scatter(i+(seed-1)*.11,v,marker=['o','^','s'][seed],facecolor='white',edgecolor='#23313B',s=28,lw=.7,zorder=4)
            ax.text(i,highs[i]+(2.3 if panel==0 else .07),f'{vals[i]:.2f}' if panel==0 else f'{vals[i]:.3f}',ha='center',fontsize=10)
        ax.set_xticks([0,1],['M0 原冻结策略','M1 零调用账本']);ax.set_ylabel(label);ax.grid(axis='y');ax.set_axisbelow(True)
        ax.set_ylim(0,108 if panel==0 else max(highs)*1.3+.1)
        if panel==0:ax.set_yticks([0,20,40,60,80,100])
        ax.set_title('(a) 同题成功率' if panel==0 else '(b) 全轨迹终点距离',loc='left')
    fig.subplots_adjust(left=.095,right=.99,bottom=.20,top=.89,wspace=.32)
    fig.text(.5,.075,'20个已见源文件，500题×3冻结权重/方法；10×10，B=20。',ha='center',fontsize=9)
    fig.text(.5,.025,'白色圆/三角/方形：训练权重0/1/2，非API重复或置信区间；未训练、未调用云端。',ha='center',fontsize=9)
    caption=f"零调用账本相对原策略平均SR变化{delta:+.2f}个百分点、SG变化{gain['sg_change']:+.3f}格，{decision}。M0/M1在同一原S4队列各执行500题×3冻结训练权重，20已见源文件、493不同起终点路线，10×10/B20；不代表新地理泛化或面积扩大。白色圆/三角/方形表示权重0/1/2，非置信区间；配对20源文件95%SR差区间为[{100*lo:+.2f}, {100*hi:+.2f}]点。SG为所有轨迹终点曼哈顿距离，成功记0，全部计划题保留。"
    for ext in ('png','pdf','svg'):
        if (FIG/f'数据结果图/{NAME}.{ext}').exists():raise ValueError('immutable figure exists')
    graphics.save(fig,FIG/'数据结果图',NAME,caption,[(OUT/'主对照/对照汇总.json').relative_to(ROOT).as_posix(),(OUT/'独立复核.json').relative_to(ROOT).as_posix()],summary['scope'])
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/summarize_local_ledger_expansion.py'
    write(FIG/'绘图数据/零调用账本完整队列_v1.json',dict(figures=[item],summary_sha256=digest(OUT/'主对照/对照汇总.json'),
        audit_sha256=digest(OUT/'独立复核.json'),min_font_pt=9,recommended_width_cm=18))
    text_new(FIG/'零调用账本完整队列图注.md','# 零调用账本完整队列图注\n\n'+caption+'\n')
    print(dict(report=(OUT/'完整队列验证报告.md').as_posix(),figure=(FIG/f'数据结果图/{NAME}.png').as_posix()),flush=True)


if __name__=='__main__':main()
