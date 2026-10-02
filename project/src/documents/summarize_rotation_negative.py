"""Evidence-based negative-rotation report and reproducible scientific charts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.rotation_negative_navigation import ROOT,TRAIN,VARIANTS,STRATEGIES,PAIRS,paths
from documents import stage_figures as graphics
from train.local_capacity import read,lines
from train.dyncur_tiny import digest
FIG=ROOT/'绘图'
LABELS=dict(Original='原方向策略',Replay4='原头四方向',RotNeg4='负旋转四方向',NoTarget='无目标探索')
PAIR_LABELS=['旋转恢复：负旋转 − 原旋转','干净保护：负旋转 − 原干净','干净目标收益：负旋转 − 探索','旋转目标收益：负旋转 − 探索','训练因素：负旋转 − 原头四方向']


def folder(out,seed,strategy):return out/f'{dict(Replay4="ReplayR0",RotNeg4="RotNeg").get(strategy,"Original")}_s{seed}'


def diagnostics(out):
    tasks={r['episode_id']:r for r in read(out/'导航任务.json')};summary={};hashes={}
    for v in VARIANTS:
        summary[v]={}
        for a in STRATEGIES:
            seeds=[]
            for s in range(3):
                path=folder(out,s,a)/f'导航_{v}_{a}_轨迹.jsonl';rows=lines(path);accepted=correct=0;false_events=[];hashes[path.relative_to(ROOT).as_posix()]=digest(path)
                for row in rows:
                    for i,decision in enumerate(row['decisions']):
                        if decision['cue_action'] is not None:
                            goal=tasks[row['episode_id']]['goal'];hit=row['trajectory'][i+1]['patch_id']==goal;accepted+=1;correct+=hit
                            if not hit:
                                current=row['trajectory'][i]['patch_id'];distance=abs(current//5-goal//5)+abs(current%5-goal%5);angle=((0 if v=='Clean' else 90)+decision.get('selected_relative_clockwise',0))%360
                                false_events.append(dict(episode_id=row['episode_id'],area=row['area'],step=i+1,current=current,evaluator_target_distance=distance,target_is_adjacent=distance==1,selected_absolute_clockwise=angle,selected_view_aligned=angle==0))
                seeds.append(dict(seed=s,accepted=accepted,correct=correct,false_accepts=accepted-correct,accepted_precision=correct/accepted if accepted else None,neural_actions=sum(r['steps'] for r in rows),
                    false_on_nonadjacent_target=sum(not e['target_is_adjacent'] for e in false_events),false_on_adjacent_wrong_direction=sum(e['target_is_adjacent'] for e in false_events),false_from_misaligned_view=sum(not e['selected_view_aligned'] for e in false_events),false_events=false_events))
            summary[v][a]=seeds
    changes={}
    for v in VARIANTS:
        changes[v]={}
        for a in ('Original','Replay4'):
            losses=[];gains=[]
            for s in range(3):
                new=lines(folder(out,s,'RotNeg4')/f'导航_{v}_RotNeg4_轨迹.jsonl');base=lines(folder(out,s,a)/f'导航_{v}_{a}_轨迹.jsonl')
                for n,b in zip(new,base):
                    item=dict(seed=s,episode_id=n['episode_id'],area=n['area'],distance=n['distance'],final_SG=n['sg'])
                    if b['success'] and not n['success']:losses.append(item)
                    if n['success'] and not b['success']:gains.append(item)
            changes[v][a]=dict(lost=len(losses),gained=len(gains),losses=losses,gains=gains)
    return dict(conditions=summary,paired_changes=changes,source_sha256=hashes,posthoc=True,not_used_to_train_or_select_threshold=True)


def plots(out,mode,summary,names):
    graphics.style();graphics.ITEMS.clear();avg=summary['averages'];reg=read(out/'预登记.json');colors=[graphics.GRAY,graphics.ORANGE,graphics.BLUE,graphics.TEAL];patterns=['','xx','//','..'];width=.19;x=np.arange(2)
    scope=f"{'Masa已知开发' if mode=='development' else 'SwissView新源文件'}{reg['source_count']}图/{reg['episodes']}题×3；5×5/B10；白点：三权重，非CI。"
    fig,axes=plt.subplots(1,2,figsize=(7.8,4.4));maxsg=max(read(folder(out,s,a)/f'导航_{v}_{a}_结果.json')['metrics']['mean_sg_all_episodes'] for v in VARIANTS for a in STRATEGIES for s in range(3));texts=[[],[]]
    for i,a in enumerate(STRATEGIES):
        xx=x+(i-1.5)*width
        for ax,values in zip(axes,([avg[v][a]['sr_mean']*100 for v in VARIANTS],[avg[v][a]['sg_mean'] for v in VARIANTS])):ax.bar(xx,values,width*.96,color=colors[i],hatch=patterns[i],edgecolor='#34424F',linewidth=.6,label=LABELS[a],zorder=2)
        for j,v in enumerate(VARIANTS):
            for s,(off,marker) in enumerate([(-.036,'o'),(0,'^'),(.036,'s')]):
                m=read(folder(out,s,a)/f'导航_{v}_{a}_结果.json')['metrics']
                for ax,value in zip(axes,(m['sr']*100,m['mean_sg_all_episodes'])):ax.scatter(xx[j]+off,value,marker=marker,s=16,facecolor='white',edgecolor='#24323B',linewidth=.65,zorder=4)
            texts[0].append(axes[0].text(xx[j],103,f"{avg[v][a]['sr_mean']*100:.1f}",ha='center',fontsize=9));texts[1].append(axes[1].text(xx[j],maxsg*1.26,f"{avg[v][a]['sg_mean']:.2f}",ha='center',fontsize=9))
    for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True);ax.set_xticks(x,['干净目标','给定目标顺时针90°']);ax.tick_params(axis='x',length=0)
    axes[0].set_ylim(0,114);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR：成功率（%）');axes[1].set_ylim(0,maxsg*1.44);axes[1].set_ylabel('SG：全部终点平均距离（格）')
    axes[0].set_title('(a) 同题导航成功率',loc='left');axes[1].set_title('(b) SG包含失败终点',loc='left');axes[0].legend(loc='upper center',bbox_to_anchor=(1.05,1.23),ncol=4,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.80,wspace=.32);fig.text(.5,.028,scope,ha='center',fontsize=9);fig.canvas.draw()
    for panel in texts:
        boxes=[t.get_window_extent(fig.canvas.get_renderer()) for t in panel]
        if any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]):raise ValueError('overlapping numerical labels')
    caption=f"负旋转头本批SR为{avg['Clean']['RotNeg4']['sr_mean']:.2%}，相同预算原头四方向为{avg['Clean']['Replay4']['sr_mean']:.2%}。只将原非邻接负样本的一半目标旋转，正样本、探索器、encoder、接受阈值0.50与预算冻结；原头对照逐参数复现。{scope} SR为成功比例，SG为所有终点平均曼哈顿距离；图示SG两位小数，精确值见报告。"
    graphics.save(fig,FIG/'数据结果图',names[0],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],scope)
    fig,ax=plt.subplots(figsize=(7.8,4.2));intervals=[]
    for i,(n,_,_,_,_) in enumerate(PAIRS):
        e=summary['effects'][n]['source_interval'];mu=e['mean']*100;lo,hi=np.array(e['interval95'])*100;intervals.append((lo,hi));threshold=-2 if n=='clean_safety' else 2 if n=='training_factor_gain' else 5
        ax.errorbar(mu,i,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt=['o','s','^','D','P'][i],color=graphics.BLUE,ms=6,capsize=3);ax.text(max(hi+1,threshold+1),i,f'{mu:+.2f}',va='center',fontsize=9)
        ax.scatter(threshold,i,marker='|',s=160,color=graphics.ORANGE)
    ax.axvline(0,color='#34424F',ls='--',lw=.8);ax.set_yticks(range(5),PAIR_LABELS);ax.invert_yaxis();ax.grid(axis='x');ax.set_axisbelow(True);ax.set_xlim(min(-3,min(v[0] for v in intervals)-4),max(7,max(v[1] for v in intervals)+9));ax.set_xlabel('SR配对差（百分点）；正值表示负旋转更高');ax.set_title('新训练因素的收益与干净目标保护',loc='left')
    fig.subplots_adjust(left=.42,right=.97,bottom=.22,top=.86);fig.text(.5,.037,'源文件成组95%区间；橙线：+5点目标收益／−2点保护／+2点训练收益。',ha='center',fontsize=9)
    gain=summary['effects']['training_factor_gain'];lo,hi=gain['source_interval']['interval95']
    caption=f"本批负旋转训练相对原头四方向SR变化{gain['gain']*100:+.2f}点，源文件95%区间[{lo*100:+.2f},{hi*100:+.2f}]点。每源文件先平均三权重配对差，再2000次seed3031成组bootstrap；全部五项区间均展示，不声称同时覆盖。开发源文件曾参与方法选择，区间不能当独立test保证；工程判定还检查种子一致性、SG、干净保护与全量重放。"
    graphics.save(fig,FIG/'数据结果图',names[1],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],scope);return graphics.ITEMS


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1];r=read(out/'验收结论.json');g=read(out/'预登记.json');s=read(out/'对照汇总.json');a=read(out/'独立复核.json');ta=read(TRAIN/'独立复核.json')
    assert r['quality_and_replay_passed'] and digest(out/'独立复核.json')==r['audit_sha256'] and digest(out/'对照汇总.json')==r['summary_sha256']
    stem={'development':'旋转负样本开发','pilot':'旋转负样本试跑','confirmation':'旋转负样本正式'}[mode];first={'development':23,'pilot':25,'confirmation':27}[mode];names=[f'{first:02d}_{stem}成功率与距离',f'{first+1:02d}_{stem}配对收益与保护'];report=out/'旋转负样本对照报告.md';diagfile=out/'接受与失败诊断.json';manifest_file=FIG/f'绘图数据/{stem}_v1.json'
    if any(f.exists() for f in (report,diagfile,manifest_file)) or any((FIG/'数据结果图'/f'{n}.{fmt}').exists() for n in names for fmt in ('png','pdf','svg')):raise ValueError('immutable report exists')
    diag=diagnostics(out);diagfile.write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n','utf-8');failed=[n for n,v in r['release_checks'].items() if not v]
    text=['# 旋转负样本视觉头：同预算单因素对照','',f"**本批{'通过' if r['candidate_numeric_passed'] else '未通过'}预定候选放行条件，全量复核通过。** 本批既报告旋转恢复，也直接检查新训练因素是否优于原头四方向。",'',
        '## 训练与诊断','',
        '87拟合源文件/52200自然配对（6960邻接正样本、45240非邻接负样本）。ReplayR0原样训练；RotNeg每epoch将一半非邻接负样本目标分配给R90/R180/R270，各7540，正样本始终R0且标签不变。两臂×三seed都从原初始化开始，AdamW、样本顺序、16epochs、512 batch、1632 optimizer步、835200样本使用完全相同。每头136837可训练参数，原87.85M编码器和无目标探索器冻结。','',
        f"六头全部训练后才评分，原头ReplayR0逐参数复现旧权重；独立功能实现全量复算{ta['optimizer_steps']}训练步、{ta['pair_uses']}配对使用、{ta['rotated_negative_uses']}旋转负样本使用，最终六头全部参数完全一致。新缓存不保存额外训练PNG，也不落盘约900MB密集矩阵，原大数据不移动。",'',
        '| 头 | 接受数 | 错误接受 | 接受精度 | 邻接召回 | 源文件95%诊断区间 |','|---|---:|---:|---:|---:|---|']
    for arm in ('ReplayR0','RotNeg'):
        for seed in range(3):
            m=read(TRAIN/f'{arm}_s{seed}/诊断结果.json');lo,hi=m['precision_interval']['interval95'];text.append(f"| {arm}/s{seed} | {m['accepted']} | {m['false_accepts']} | {m['accepted_precision']:.2%} | {m['adjacent_recall']:.2%} | [{lo:.2%},{hi:.2%}] |")
    text+=['','诊断仅用原22头部校准源文件/13200自然配对、四目标视图、六头，共316800原始预测独立复核；固定0.50门槛，不选择阈值、不按诊断早停或选checkpoint。22源图参加过探索器训练，因此不是全系统独立测试。精度区间source4119/2000次，不能当未知域保证。','',
        '## 同题在线导航','', '| 目标 | 策略 | 成功/总数 | SR | SG | 三权重SR |','|---|---|---:|---:|---:|---|']
    for v in VARIANTS:
        for arm in STRATEGIES:
            m=s['averages'][v][arm];text.append(f"| {v} | {LABELS[arm]} | {sum(m['successes_by_seed'])}/{g['episodes']*3} | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} | {' / '.join(f'{v:.2%}' for v in m['sr_by_seed'])} |")
    text+=['','所有候选四方向及NoTarget在给定Clean/Rot90之间逐题轨迹相同；输入候选按像素SHA确定稳定顺序，以原始方向联合分数选择，接受门槛仍0.50，目标真方位/目标位置不进入选择。旋转不变来自穷举四个直角视图，不能证明任意角度旋转鲁棒。开发Original/Replay4/NoTarget复现旧批完整记录，不以旧数值替代新运行。','',
        '| 对比 | SR变化（点） | 源文件95%区间（点） | 正权重 | SG变化 |','|---|---:|---|---:|---:|']
    for (n,_,_,_,_),label in zip(PAIRS,PAIR_LABELS):
        e=s['effects'][n];lo,hi=e['source_interval']['interval95'];text.append(f"| {label} | {e['gain']*100:+.2f} | [{lo*100:+.2f},{hi*100:+.2f}] | {e['positive_seeds']}/3 | {e['sg_change']:+.3f} |")
    text+=['',f"未满足预定项：{', '.join(failed) if failed else '无'}。",'',
        '训练因素开发收益≥2点、至少2正seed、SG不更差；旋转恢复/两条件目标收益≥5点、至少2正seed、SG不更差；干净SR相对原策略最多损失2点、SG最多增加0.10。开发与正式旋转恢复源图区间下界需正；正式另要求训练因素区间下界正；四新图试跑仅报告区间并检查兼容工程门槛。所有规则/指标均在本批训练前预登记，不因结果改线。','',
        '| 条件/策略 | 实际接受 | 立即到达目标 | 错误接管 | 精度 | 非邻接误接受 | 邻接错方向 | 旋转视图误接受 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        for arm in ('Original','Replay4','RotNeg4'):
            rows=diag['conditions'][v][arm];n=sum(x['accepted'] for x in rows);k=sum(x['correct'] for x in rows);bad_nonadj=sum(x['false_on_nonadjacent_target'] for x in rows);bad_direction=sum(x['false_on_adjacent_wrong_direction'] for x in rows);bad_rotation=sum(x['false_from_misaligned_view'] for x in rows);text.append(f"| {v}/{LABELS[arm]} | {n} | {k} | {n-k} | {k/n:.2%} | {bad_nonadj} | {bad_direction} | {bad_rotation} |")
    newrows=diag['conditions']['Clean']['RotNeg4'];baserows=diag['conditions']['Clean']['Replay4'];effect=s['effects']['training_factor_gain'];paired=diag['paired_changes']['Clean']['Replay4']
    text+=['',f"在各自实际访问的干净路径上，新头错误接管{sum(z['false_accepts'] for z in baserows)}→{sum(z['false_accepts'] for z in newrows)}，选中旋转视图的错误{sum(z['false_from_misaligned_view'] for z in baserows)}→{sum(z['false_from_misaligned_view'] for z in newrows)}；相对原头四方向新增成功{paired['gained']}、新增失败{paired['lost']}，净SR收益{effect['gain']*100:+.2f}点。不同路径上的计数不是同状态反事实，不能单独证明某一次拒绝造成了SR变化。",'',
        '接受精度与配对新增成功/失败使用评测真目标后置统计，不用于训练、选门槛或候选排序；不同策略访问状态不同，总错误次数不能单独证明SR变化的原因。完整逐题变化与来源SHA保存于接受与失败诊断。','',
        '## 范围、资源与复核','',
        f"当前{g['source_count']}源文件/{g['episodes']}题/权重，不同路线{g['unique_routes']}，首次评测新源文件{g['new_source_files']}。Masa28已见开发不能称独立test，种子/重复题不增加独立源图数量。当前{a['neural_episodes']}轨迹/{a['neural_actions']}动作、{a['rule_records_rechecked_once']}规则记录、10个SR/SG区间全部独立复算；258运行前测试通过。训练和动作复核由同一执行代理使用另一个算法实现，不是不同研究人员复审。",'',
        '| 训练头 | 训练用时（秒） | PyTorch峰值（MiB） |','|---|---:|---:|']
    for arm in ('ReplayR0','RotNeg'):
        for seed in range(3):
            m=read(TRAIN/f'{arm}_s{seed}/训练资源.json');text.append(f"| {arm}/s{seed} | {m['seconds']:.1f} | {m['torch_peak_allocated_bytes']/2**20:.1f} |")
    text+=['','峰值含四特征池，排除encoder提取/驱动/其他进程；共享GPU用时只作资源记录，不是速度对比。原S2/S3/S4及Swiss干净确认仍有效，旧文件、数据、原权重和默认配置保持，当前实验没有云端调用。新源文件按ID/文件SHA/RGB SHA隔离，严格地理非重叠及编码器预训练覆盖未知。','', '## 本轮决定','']
    if not r['candidate_numeric_passed']:text+=['不放行后续新源图，保留失败证据并保持原默认。当前均匀旋转负例的SR收益不足，可优先提出“只在87拟合源图内挖掘原头高分旋转负样本”的下一单因素假设，用相同初始化和预算与本轮均匀采样对照；开发/Swiss错误图不得回流拟合。这只是后续方案方向，本批未执行或证明有效。']
    elif mode=='confirmation':text+=['20新源图正式确认通过，可作为冻结直角方向鲁棒候选；默认未自动替换，任意角度/裁剪/真实异时航拍仍需独立实验。']
    else:text+=['可继续已预登记下一批新源文件，保持新头、阈值和策略冻结；开发/四图试跑不是正式泛化确认。']
    text+=['','[预登记](预登记.json) / [独立复核](独立复核.json) / [验收](验收结论.json) / [诊断](接受与失败诊断.json)。']
    report.write_text('\n'.join(text)+'\n','utf-8');items=plots(out,mode,s,names)
    for item in items:item['reproduction_script']='project/src/documents/summarize_rotation_negative.py'
    manifest=dict(mode=mode,source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in (out/'对照汇总.json',out/'独立复核.json',out/'验收结论.json',out/'预登记.json',TRAIN/'独立复核.json',TRAIN/'诊断汇总.json',TRAIN/'验收结论.json',diagfile)},script_sha256=digest(Path(__file__)),style_helper_sha256=digest(Path(graphics.__file__)),report_sha256=digest(report),figures=items)
    manifest_file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8');(FIG/f'{stem}图注.md').write_text('# 旋转负样本视觉头图注\n\n'+'\n\n'.join(f"## {i['id']}\n\n{i['caption']}" for i in items)+'\n','utf-8');print(json.dumps(dict(report=str(report),passed=r['candidate_numeric_passed'],figures=names),ensure_ascii=False))


if __name__=='__main__':main()
