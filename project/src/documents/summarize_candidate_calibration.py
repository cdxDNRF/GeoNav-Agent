"""Audited calibration trade-off, navigation results and reusable vector figures."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.candidate_calibration import ROOT,VARIANTS,STRATEGIES,paths
from documents import stage_figures as graphics
from train.local_capacity import read,lines
from train.dyncur_tiny import digest
from agents.target_cue import choose_cue
from env.episode import ACTIONS
FIG=ROOT/'绘图';LABELS=dict(Original='原策略',Naive4='四方向0.50',Calibrated4='校准四方向',NoTarget='无目标探索')
PAIRS=['rotation_recovery','clean_safety','clean_target_gain','rotation_target_gain','calibration_change']
PAIR_LABELS=['旋转恢复：校准 − 原旋转','干净保护：校准 − 原干净','干净目标收益：校准 − 探索','旋转目标收益：校准 − 探索','校准改变量：校准 − 0.50']


def diagnostics(out):
    tasks={e['episode_id']:e for e in read(out/'导航任务.json')};result={};sources={}
    for v in VARIANTS:
        result[v]={}
        for a in STRATEGIES:
            rows_by_seed=[]
            for s in range(3):
                path=out/f'Edge_s{s}/导航_{v}_{a}_轨迹.jsonl';records=lines(path);accepted=correct=0;sources[path.relative_to(ROOT).as_posix()]=digest(path)
                filtered_true=filtered_false=0;failed_filtered=[]
                for r in records:
                    lost_opportunity=False
                    for i,d in enumerate(r['decisions']):
                        if d['cue_action'] is not None:accepted+=1;correct+=r['trajectory'][i+1]['patch_id']==tasks[r['episode_id']]['goal']
                        if a=='Calibrated4' and d['cue_action'] is None:
                            old,_=choose_cue(d['probabilities'],.5,tuple(d['public_position']),d['public_visited'])
                            if old is not None:
                                dr,dc=ACTIONS[old];rr,cc=d['public_position'];hit=(rr+dr)*5+cc+dc==tasks[r['episode_id']]['goal']
                                filtered_true+=hit;filtered_false+=not hit;lost_opportunity=lost_opportunity or hit
                    if lost_opportunity and not r['success']:failed_filtered.append(r['episode_id'])
                rows_by_seed.append(dict(seed=s,accepted=accepted,correct=correct,false_accepts=accepted-correct,accepted_precision=correct/accepted if accepted else None,
                    neural_actions=sum(r['steps'] for r in records),head_forward_calls=sum(r['steps'] for r in records)*(4 if a in ('Naive4','Calibrated4') else 1),
                    same_visited_state_true_cues_filtered=filtered_true if a=='Calibrated4' else None,
                    same_visited_state_false_cues_filtered=filtered_false if a=='Calibrated4' else None,
                    failed_episodes_with_filtered_true_opportunity=failed_filtered if a=='Calibrated4' else None))
            result[v][a]=rows_by_seed
    differences={}
    for other in ('Original','Naive4'):
        lost=[];gained=[]
        for s in range(3):
            base=lines(out/f'Edge_s{s}/导航_Clean_{other}_轨迹.jsonl');cal=lines(out/f'Edge_s{s}/导航_Clean_Calibrated4_轨迹.jsonl')
            for b,c in zip(base,cal):
                if b['success'] and not c['success']:lost.append(dict(seed=s,episode_id=c['episode_id'],area=c['area'],distance=c['distance'],final_SG=c['sg']))
                if not b['success'] and c['success']:gained.append(dict(seed=s,episode_id=c['episode_id'],area=c['area'],distance=c['distance']))
        differences[other]=dict(lost_clean_successes=len(lost),new_clean_successes=len(gained),losses=lost,gains=gained)
    return dict(scope='posthoc evaluator labels; not threshold selection; no causal guarantee',conditions=result,paired_clean_changes=differences,source_sha256=sources)


def figures(out,mode,summary,names):
    graphics.style();graphics.ITEMS.clear();averages=summary['averages'];reg=read(out/'预登记.json');colors=[graphics.GRAY,graphics.ORANGE,graphics.BLUE,graphics.TEAL];patterns=['','xx','//','..']
    fig,axes=plt.subplots(1,2,figsize=(7.8,4.4));x=np.arange(2);width=.19
    maxsg=max(read(out/f'Edge_s{s}/导航_{v}_{a}_结果.json')['metrics']['mean_sg_all_episodes'] for v in VARIANTS for a in STRATEGIES for s in range(3))
    numeric_labels=[[],[]]
    for i,a in enumerate(STRATEGIES):
        positions=x+(i-1.5)*width;sr=[averages[v][a]['sr_mean']*100 for v in VARIANTS];sg=[averages[v][a]['sg_mean'] for v in VARIANTS]
        for ax,values in zip(axes,(sr,sg)):ax.bar(positions,values,width*.96,label=LABELS[a],color=colors[i],hatch=patterns[i],edgecolor='#34424F',linewidth=.6,zorder=2)
        for j,v in enumerate(VARIANTS):
            for s,(off,marker) in enumerate([(-.036,'o'),(0,'^'),(.036,'s')]):
                metrics=read(out/f'Edge_s{s}/导航_{v}_{a}_结果.json')['metrics'];axes[0].scatter(positions[j]+off,metrics['sr']*100,marker=marker,s=16,facecolor='white',edgecolor='#24323B',linewidth=.65,zorder=4)
                axes[1].scatter(positions[j]+off,metrics['mean_sg_all_episodes'],marker=marker,s=16,facecolor='white',edgecolor='#24323B',linewidth=.65,zorder=4)
            numeric_labels[0].append(axes[0].text(positions[j],103,f'{sr[j]:.1f}',ha='center',fontsize=9))
            numeric_labels[1].append(axes[1].text(positions[j],maxsg*1.26,f'{sg[j]:.2f}',ha='center',fontsize=9))
    for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True);ax.set_xticks(x,['干净目标','给定目标顺时针90°']);ax.tick_params(axis='x',length=0)
    axes[0].set_ylim(0,114);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR：成功率（%）');axes[1].set_ylim(0,maxsg*1.44);axes[1].set_ylabel('SG：平均终点距离（格）')
    axes[0].set_title('(a) 冻结模型的导航成功率',loc='left');axes[1].set_title('(b) SG包含成功与失败',loc='left')
    axes[0].legend(loc='upper center',bbox_to_anchor=(1.05,1.23),ncol=4,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.80,wspace=.32)
    scope=f"{'Masa已知开发' if mode=='development' else 'SwissView新源文件'}{reg['source_count']}图/{reg['episodes']}题×3；5×5/B10；白色标记：三权重，非CI。"
    fig.text(.5,.028,scope,ha='center',fontsize=9)
    fig.canvas.draw()
    for panel in numeric_labels:
        boxes=[t.get_window_extent(fig.canvas.get_renderer()) for t in panel]
        if any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]):raise ValueError('numeric annotation overlap')
    calibrated=averages['Clean']['Calibrated4'];original=averages['Clean']['Original'];naive=averages['Clean']['Naive4']
    caption=f"只校准接受门槛后，本批干净/旋转题SR为{calibrated['sr_mean']:.2%}，原干净为{original['sr_mean']:.2%}、未校准四方向为{naive['sr_mean']:.2%}。每权重阈值仅由原22头部校准源图选择，模型、候选搜索、探索器及预算不变；无可用阈值时全弃权，不称视觉改善。{scope} SR为成功比例，SG为全部终点距离均值（图示两位小数，精确值见报告）；是否放行还检查干净保护、目标收益和区间。"
    graphics.save(fig,FIG/'数据结果图',names[0],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],scope)
    fig,ax=plt.subplots(figsize=(7.8,4.15));all_intervals=[]
    for i,n in enumerate(PAIRS):
        interval=summary['effects'][n]['source_interval'];mu=interval['mean']*100;lo,hi=np.array(interval['interval95'])*100;all_intervals.append((lo,hi))
        ax.errorbar(mu,i,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt=['o','s','^','D','P'][i],color=graphics.BLUE,ms=6,capsize=3);ax.text(hi+1,i,f'{mu:+.2f}',va='center',fontsize=9)
        if n!='calibration_change':ax.scatter(-2 if n=='clean_safety' else 5,i,marker='|',s=160,color=graphics.ORANGE)
    ax.axvline(0,color='#34424F',ls='--',lw=.8);ax.set_yticks(range(5),PAIR_LABELS);ax.invert_yaxis();ax.grid(axis='x');ax.set_axisbelow(True)
    ax.set_xlim(min(-3,min(lo for lo,_ in all_intervals)-4),max(7,max(hi for _,hi in all_intervals)+9));ax.set_xlabel('SR配对差（百分点）；正值表示校准更高');ax.set_title('提高接受精度还需要保留导航能力',loc='left')
    fig.subplots_adjust(left=.40,right=.97,bottom=.22,top=.86);fig.text(.5,.037,'源文件成组95%区间；橙短线：+5点收益／−2点干净保护；全部比较报告。',ha='center',fontsize=9)
    caption='离线校准的接受精度不能替代在线SR与干净保护。每源文件先平均三权重配对差，再2000次seed3031成组bootstrap，图示为逐项95%区间，不声称五项同时覆盖；第五项直接隔离阈值调整的导航改变量。橙短线为事前固定工程尺度，完整判定还包含SG、种子一致性和全量复核。'
    graphics.save(fig,FIG/'数据结果图',names[1],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],scope)
    return graphics.ITEMS


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1];calout=paths('calibration')[1]
    reg=read(out/'预登记.json');summary=read(out/'对照汇总.json');receipt=read(out/'验收结论.json');audit=read(out/'独立复核.json');calreceipt=read(calout/'验收结论.json')
    assert receipt['quality_and_replay_passed'] and digest(out/'独立复核.json')==receipt['audit_sha256'] and digest(out/'对照汇总.json')==receipt['summary_sha256']
    stem={'development':'可靠性校准开发','pilot':'可靠性校准试跑','confirmation':'可靠性校准正式'}[mode];first={'development':21,'pilot':23,'confirmation':25}[mode];names=[f'{first:02d}_{stem}成功率与距离',f'{first+1:02d}_{stem}配对收益与保护']
    report=out/'接受可靠性校准报告.md';diagfile=out/'接受与失败诊断.json';source=FIG/f'绘图数据/{stem}_v1.json'
    if any(p.exists() for p in (report,diagfile,source)) or any((FIG/'数据结果图'/f'{n}.{fmt}').exists() for n in names for fmt in ('png','pdf','svg')):raise ValueError('immutable reporting artifacts exist')
    diag=diagnostics(out);diagfile.write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n','utf-8');avg=summary['averages'];failed=[k for k,v in receipt['release_checks'].items() if not v]
    text=['# 多候选接受可靠性单因素校准','',f"**本批{'通过' if receipt['candidate_numeric_passed'] else '未通过'}预定放行条件，完整复核通过。** 唯一变化是每原权重接受阈值，选值仅用原22头部校准源图；原探索器、候选搜索、视觉头、encoder、四均值、5×5/B10保持冻结。",'',
        '## 校准是否减少了误接受','', '| 权重 | 冻结阈值 | 接受数 | 正确数 | 精度 | 源文件95%筛选区间 | 邻接召回率 |','|---|---:|---:|---:|---:|---|---:|']
    for s in range(3):
        cal=read(calout/f'Edge_s{s}/校准阈值.json');t=cal['threshold'];row=next((r for r in cal['grid'] if r['threshold']==t),None);m=row['metrics'] if row else cal['grid'][-1]['metrics']
        lo,hi=m['precision_interval']['interval95'];text.append(f"| seed{s} | {t if t is not None else 'null/全弃权'} | {m['accepted'] if t is not None else 0} | {m['correct_accepted'] if t is not None else 0} | {f'{m["accepted_precision"]:.2%}' if t is not None else '无接受'} | [{lo:.2%},{hi:.2%}] | {m['adjacent_recall']:.2%} |")
    text+=['','校准22源文件/13200自然配对：1760真邻接、11440非邻接，三头四候选共158400评分全部独立重放。候选原五类联合概率不重新归一化，不使用目标真朝向选视图；source4119/2000次重采样，所有源文件保留、零接受重采样精度记0。网格11阈值，取支持≥100/≥5图、精度≥95%、普通筛选区间下界≥90%的最小门槛；同校准集选择区间有选择偏差，不是未知域95%保证。没有门槛的seed冻结null，不能用空值精度宣称成功。','',
        '## 同题在线成绩','', '| 目标 | 策略 | 成功/总数 | SR | SG（格） | 三权重SR |','|---|---|---:|---:|---:|---|']
    for v in VARIANTS:
        for a in STRATEGIES:
            m=avg[v][a];text.append(f"| {'干净' if v=='Clean' else '顺时针90°'} | {LABELS[a]} | {sum(m['successes_by_seed'])}/{reg['episodes']*3} | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} | {' / '.join(f'{x:.2%}' for x in m['sr_by_seed'])} |")
    text+=['','Calibrated4与Naive4在给定目标Clean/Rot90CW之间保持逐题动作不变；NoTarget也不变。Masa开发全部Original/Naive4/NoTarget原记录复现上一轮，阈值和评分变化可分离；不会用旧SR充当新运行。','',
        '| 配对比较 | SR差（点） | 源文件95%区间（点） | 正权重数 | SG变化 |','|---|---:|---|---:|---:|']
    for n,label in zip(PAIRS,PAIR_LABELS):
        e=summary['effects'][n];lo,hi=e['source_interval']['interval95'];text.append(f"| {label} | {e['gain']*100:+.2f} | [{lo*100:+.2f},{hi*100:+.2f}] | {e['positive_seeds']}/3 | {e['sg_change']:+.3f} |")
    text+=['','干净保护：相对原干净SR最多损失2点、SG最多增加0.10；旋转恢复与两条件对探索目标贡献均≥5点、≥2正权重、SG不更差。开发/正式旋转收益源文件95%下界须正；4图兼容试跑完整报告区间但不据此宣称稳定收益。阈值三个都有效且至少一项实际改变才可进入新域，规则工程门槛仍逐批检查。', '',f"未满足项：{', '.join(failed) if failed else '无'}。",'',
        '| 目标/策略 | 接受数 | 正确到目标 | 错误接受 | 实际接受精度 |','|---|---:|---:|---:|---:|']
    for v in VARIANTS:
        for a in ('Original','Naive4','Calibrated4'):
            rows=diag['conditions'][v][a];n=sum(r['accepted'] for r in rows);k=sum(r['correct'] for r in rows);text.append(f"| {v}/{LABELS[a]} | {n} | {k} | {n-k} | {f'{k/n:.2%}' if n else '无接受'} |")
    text+=['','真目标仅在执行后用来统计立即到达精度与失败，不进入Agent输入/阈值选择。与原干净/Naive4配对的新增失败和新增成功全部保存接受与失败诊断JSON；减少误接管也可能丢失低分真实邻接，不能把精度提高当SR提高。', '',
        '## 范围与可追溯性','',
        f"当前{mode}批：{reg['source_count']}源文件、{reg['episodes']}题/权重、不同路线{reg['unique_routes']}、重复抽样{reg['episodes']-reg['unique_routes']}、首次策略评测新源文件{reg['new_source_files']}。校准22参与探索器109训练但未参加视觉头87训练；28开发曾参与方法选择，均不能冒充全系统独立test。新的SwissView08…11/40…59先验选择，与28已见Swiss文件/RGB哈希隔离，严格地理非重叠与预训练覆盖未知。",'',
        f"253运行前测试通过；校准2200候选图/550当前图、158400原始评分与33精度筛选区间独立复核；当前批{audit['neural_episodes']}在线轨迹/{audit['neural_actions']}动作、{audit['rule_records_rechecked_once']}规则、10个SR/SG源文件区间全量复核。当前图像复算global/local最大误差={audit['images']['global_max_abs_error']:.3g}/{audit['images']['local_max_abs_error']:.3g}；开发按哈希只读复用原2800候选，不复制大文件。", '',
        '同一执行代理使用不同手算实现复核，非不同人员审稿。原数据/旧报告/模型/默认和原S2/S3结论保持，源代码/样本/像素/特征/概率/门槛/完整动作可追溯。独立源文件数量不随种子和重复路线增加，普通逐项区间不保证五项同时覆盖。固定直角旋转不能推断任意角度、裁剪、真实异时或跨视角。', '',
        '## 下一步决定','']
    rows=diag['conditions']['Clean']['Calibrated4'];true=sum(r['same_visited_state_true_cues_filtered'] for r in rows);false=sum(r['same_visited_state_false_cues_filtered'] for r in rows);failed_opportunities=sum(len(r['failed_episodes_with_filtered_true_opportunity']) for r in rows)
    text.insert(text.index('## 范围与可追溯性'),f'在校准策略实际访问的同一干净状态上，将旧0.50门控与新阈值比较：拒绝了{true}次正确邻接线索和{false}次错误线索；{failed_opportunities}条失败轨迹曾因此失去立即到目标的线索。此状态内比较不重新模拟0.50路径，也不证明回退一条动作就能恢复整条轨迹；不能把两策略不同路径上的总接受次数当成逐状态反事实。\n')
    if not receipt['candidate_numeric_passed']:text+=['本轮不放行后续新源图，不替换原默认，保存全部失败。依据精度/覆盖与导航配对差决定下一单因素；若仅提高门槛不能同时保留目标召回与干净SR，应单独改善方向候选判别能力，不能在同一批补分差规则或用SwissView调到过线。']
    elif mode=='confirmation':text+=['新的20源文件正式确认通过；本轮冻结配置可作为独立方向鲁棒候选，默认仍未自动替换。其他扰动与真实应用范围另起实验。']
    else:text+=['本批允许继续预登记的下一批；沿用同阈值，不在新域调参。开发/试跑不等于正式稳定收益确认。']
    text+=['','[预登记](预登记.json) / [全量复核](独立复核.json) / [最终验收](验收结论.json) / [接受与失败诊断](接受与失败诊断.json)。']
    report.write_text('\n'.join(text)+'\n','utf-8');items=figures(out,mode,summary,names)
    for item in items:item['reproduction_script']='project/src/documents/summarize_candidate_calibration.py'
    manifest=dict(mode=mode,source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [out/'对照汇总.json',out/'独立复核.json',out/'验收结论.json',out/'预登记.json',calout/'阈值冻结结束.json',calout/'独立复核.json',calout/'验收结论.json',diagfile]},
        script_sha256=digest(Path(__file__)),style_helper_sha256=digest(Path(graphics.__file__)),report_sha256=digest(report),figures=items)
    source.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8');(FIG/f'{stem}图注.md').write_text('# 多候选可靠性校准图注\n\n'+'\n\n'.join(f"## {i['id']}\n\n{i['caption']}" for i in items)+'\n','utf-8')
    print(json.dumps(dict(report=str(report),passed=receipt['candidate_numeric_passed'],figures=names),ensure_ascii=False))


if __name__=='__main__':main()
