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
from eval.hard_rotation_navigation import ROOT,TRAIN,VARIANTS,STRATEGIES,PAIRS,paths
from documents import stage_figures as graphics
from train.local_capacity import read,lines
from train.dyncur_tiny import digest
FIG=ROOT/'绘图'
LABELS=dict(Original='原方向策略',Raw4='原头四方向',Replay4='均匀负旋转',Hard4='高分负旋转',NoTarget='无目标探索')
PLOT_STRATEGIES=('Raw4','Replay4','Hard4','NoTarget')
PAIR_LABELS=['旋转恢复：高分 − 原旋转','干净保护：高分 − 原干净','干净目标收益：高分 − 探索','旋转目标收益：高分 − 探索','挖掘因素：高分 − 均匀负旋转','总体新头：高分 − 原头四方向']


def folder(out,seed,strategy):return out/f'{dict(Replay4="Uniform",Hard4="HardNeg").get(strategy,"Original")}_s{seed}'


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
        for a in ('Original','Raw4','Replay4'):
            losses=[];gains=[]
            for s in range(3):
                new=lines(folder(out,s,'Hard4')/f'导航_{v}_Hard4_轨迹.jsonl');base=lines(folder(out,s,a)/f'导航_{v}_{a}_轨迹.jsonl')
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
    for i,a in enumerate(PLOT_STRATEGIES):
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
    caption=f"高分负旋转头本批SR为{avg['Clean']['Hard4']['sr_mean']:.2%}，相同预算均匀负旋转为{avg['Clean']['Replay4']['sr_mean']:.2%}。只在拟合源图按原头高分选择相同槽位的旋转负视图，三角配额保持，正样本、探索器、encoder、接受阈值0.50与预算冻结；均匀对照逐参数复现上一轮权重。{scope} SR为成功比例，SG为所有终点平均曼哈顿距离；图示SG两位小数，精确值见报告。"
    graphics.save(fig,FIG/'数据结果图',names[0],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],scope)
    fig,ax=plt.subplots(figsize=(7.8,4.6));intervals=[]
    for i,(n,_,_,_,_) in enumerate(PAIRS):
        e=summary['effects'][n]['source_interval'];mu=e['mean']*100;lo,hi=np.array(e['interval95'])*100;intervals.append((lo,hi));threshold=-2 if n=='clean_safety' else 2 if n=='training_factor_gain' else 0 if n=='original_four_gain' else 5
        display=0.0 if abs(mu)<.005 else mu
        ax.errorbar(mu,i,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt=['o','s','^','D','P','X'][i],color=graphics.BLUE,ms=6,capsize=3);ax.text(max(hi+1,threshold+1),i,f'{display:+.2f}',va='center',fontsize=9)
        
        if n!='original_four_gain':ax.scatter(threshold,i,marker='|',s=160,color=graphics.ORANGE)
    ax.axvline(0,color='#34424F',ls='--',lw=.8);ax.set_yticks(range(6),PAIR_LABELS);ax.invert_yaxis();ax.grid(axis='x');ax.set_axisbelow(True);ax.set_xlim(min(-3,min(v[0] for v in intervals)-4),max(7,max(v[1] for v in intervals)+9));ax.set_xlabel('SR配对差（百分点）；正值表示高分负旋转更高');ax.set_title('高分挖掘的收益与干净目标保护',loc='left')
    fig.subplots_adjust(left=.42,right=.97,bottom=.22,top=.86);fig.text(.5,.037,'源文件成组95%区间；橙线：+5点目标收益／−2点保护／+2点训练收益。',ha='center',fontsize=9)
    gain=summary['effects']['training_factor_gain'];lo,hi=gain['source_interval']['interval95']
    caption=f"本批高分挖掘相对均匀负旋转SR变化{gain['gain']*100:+.2f}点，源文件95%区间[{lo*100:+.2f},{hi*100:+.2f}]点。每源文件先平均三权重配对差，再2000次seed3031成组bootstrap；全部六项区间均展示，不声称同时覆盖。开发源文件曾参与方法选择，区间不能当独立test保证；工程判定还检查种子一致性、SG、干净保护与全量重放。"
    graphics.save(fig,FIG/'数据结果图',names[1],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],scope);return graphics.ITEMS


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1];r=read(out/'验收结论.json');g=read(out/'预登记.json');s=read(out/'对照汇总.json');a=read(out/'独立复核.json');ta=read(TRAIN/'独立复核.json')
    assert r['quality_and_replay_passed'] and digest(out/'独立复核.json')==r['audit_sha256'] and digest(out/'对照汇总.json')==r['summary_sha256']
    stem={'development':'高分旋转负样本开发','pilot':'高分旋转负样本试跑','confirmation':'高分旋转负样本正式'}[mode];first={'development':25,'pilot':27,'confirmation':29}[mode];names=[f'{first:02d}_{stem}成功率与距离',f'{first+1:02d}_{stem}配对收益与保护'];report=out/'高分旋转负样本对照报告.md';diagfile=out/'接受与失败诊断.json';manifest_file=FIG/f'绘图数据/{stem}_v1.json'
    if any(f.exists() for f in (report,diagfile,manifest_file)) or any((FIG/'数据结果图'/f'{n}.{fmt}').exists() for n in names for fmt in ('png','pdf','svg')):raise ValueError('immutable report exists')
    diag=diagnostics(out);diagfile.write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n','utf-8');failed=[n for n,v in r['release_checks'].items() if not v]
    text=['# 拟合源图内高分旋转负例：同预算对照','',f"**本批{'通过' if r['candidate_numeric_passed'] else '未通过'}预定候选放行条件，工程复核通过。** 原S2/S3、10×10密度及SwissView同模态迁移验收保持，本项是目标方向鲁棒性的补充探索。",'',
        '## 挖掘是否改变了训练样本难度','',
        '87拟合源文件/52200自然配对（6960真邻接、45240非邻接）；三原头冻结，在此87图全部负配对的R90/180/270上共407160次评分，不读取22诊断、28开发、原val/test或Swiss来挖掘。分数为原始五类联合概率的前四方向最大值。原头拟合过这些图，因此拟合分数只衡量相对困难度，不能证明泛化，也不意味着每个负例分数高于0.50。','',
        '每epoch两臂同一22620个旋转负槽位，R90/180/270各7540；Uniform按固定rank/epoch选角，HardNeg按该槽位三个旋转中的最高分排序，再平衡贪心选尚有配额的最高分角度，并列按rank/小角度。正样本、原负槽位、标签、每角使用数、优化步数均相同。平衡贪心不声称全局最优，部分槽位因配额会选择较低分视图，全部保留。','',
        '| seed | Uniform平均分 | HardNeg平均分 | 选中分数提高比例 | 分数降低比例 | 配额限制比例 |','|---|---:|---:|---:|---:|---:|']
    mining=read(TRAIN/'挖掘诊断.json')
    for seed,rows in mining.items():
        v={k:sum(x[k] for x in rows)/16 for k in rows[0] if k not in ('epoch','rotated_slots_sha256')};text.append(f"| {seed} | {v['uniform_mean_score']:.4f} | {v['hard_mean_score']:.4f} | {v['fraction_hard_score_higher']:.2%} | {v['fraction_hard_score_lower']:.2%} | {v['quota_constrained_fraction']:.2%} |")
    text+=['','## 匹配训练与22源图诊断','',
        'Uniform/HardNeg两臂×3seed，共六个136837参数视觉头；相同原初始化、CUDA样本排列、AdamW、BCE邻接+真邻接CE方向、16epochs/512 batch/1632步/835200配对使用/111360正使用。两臂每头都361920旋转负使用，每角120640；全部训练后再评诊断/导航，固定0.50门槛，不选epoch/seed/阈值。Uniform逐参数复现上一轮RotNeg，探索器、Sat2Cap、四均值、合法/未访问门控及四候选搜索冻结。','',
        f"独立功能实现复算{ta['mining_predictions_recomputed']}拟合挖掘评分、{ta['mining_epoch_tables_recomputed']}epoch角度表、{ta['optimizer_steps']}训练步、{ta['pair_uses']}配对使用、{ta['rotated_negative_uses']}旋转负使用，六头最终全部参数精确一致；两臂旋转负数分别为{ta['rotated_negative_uses_by_arm']}。冻结审计器原收据沿用单臂旋转总数，另行收据实现按六份资源与完整训练重放汇总，未改变训练/样本/分数/冻结源码。",'',
        '| 头 | 接受数 | 错误接受 | 接受精度 | 邻接召回 | 源文件95%诊断区间 |','|---|---:|---:|---:|---:|---|']
    for arm in ('Uniform','HardNeg'):
        for seed in range(3):
            m=read(TRAIN/f'{arm}_s{seed}/诊断结果.json');lo,hi=m['precision_interval']['interval95'];text.append(f"| {arm}/s{seed} | {m['accepted']} | {m['false_accepts']} | {m['accepted_precision']:.2%} | {m['adjacent_recall']:.2%} | [{lo:.2%},{hi:.2%}] |")
    text+=['','22图/13200自然配对×4目标视图×6头，共316800原始概率完整复算。诊断图未参加视觉头87拟合，但参加过探索器109图训练，不是全系统独立test；source4119/2000次精度区间不当未知域保证。固定缓存按SHA继承上一批像素/编码复算证据，没有重复编码器提取或复制大图。','',
        '## 同题在线成绩','', '| 目标 | 策略 | 成功/总数 | SR | SG | 三权重SR |','|---|---|---:|---:|---:|---|']
    for v in VARIANTS:
        for arm in STRATEGIES:
            m=s['averages'][v][arm];text.append(f"| {v} | {LABELS[arm]} | {sum(m['successes_by_seed'])}/{g['episodes']*3} | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} | {' / '.join(f'{v:.2%}' for v in m['sr_by_seed'])} |")
    text+=['','Raw4保留原头四方向参考，Replay4为本次从头复现的Uniform，Hard4为高分挖掘新头。三个候选策略和NoTarget在两给定条件下逐题动作不变；Original保持原单视图。开发四组控制完整复现各自父批记录，未拿旧数值替代本次运行。穷举直角视图的不变性不能推断任意角度旋转。','',
        '| 配对比较 | SR差（点） | 源文件95%区间（点） | 正seed | SG变化 |','|---|---:|---|---:|---:|']
    for (n,_,_,_,_),label in zip(PAIRS,PAIR_LABELS):
        e=s['effects'][n];lo,hi=e['source_interval']['interval95'];text.append(f"| {label} | {e['gain']*100:+.2f} | [{lo*100:+.2f},{hi*100:+.2f}] | {e['positive_seeds']}/3 | {e['sg_change']:+.3f} |")
    text+=['',f"未满足预定项：{', '.join(failed) if failed else '无'}。",'',
        '挖掘因素相对Uniform SR≥+2点/至少2正seed/SG不更差；旋转恢复和两条件相对NoTarget≥+5点/至少2正seed/SG不更差；干净相对原策略最多损失2点/SG增加0.10。开发/正式旋转恢复SR源文件下界须正；正式另要求挖掘因素下界正；四图试跑只作工程兼容并报告区间。Hard4−Raw4额外比较完整报告，不因开发分数重设门槛。','',
        '| 条件/策略 | 接受 | 立即到目标 | 错误接管 | 精度 | 非邻接误接受 | 邻接错方向 | 旋转视图误接受 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        for arm in ('Original','Raw4','Replay4','Hard4'):
            rows=diag['conditions'][v][arm];n=sum(x['accepted'] for x in rows);k=sum(x['correct'] for x in rows);counts=[sum(x[key] for x in rows) for key in ('false_on_nonadjacent_target','false_on_adjacent_wrong_direction','false_from_misaligned_view')];text.append(f"| {v}/{LABELS[arm]} | {n} | {k} | {n-k} | {f'{k/n:.2%}' if n else '无接受'} | {counts[0]} | {counts[1]} | {counts[2]} |")
    paired=diag['paired_changes']['Clean']['Replay4'];e=s['effects']['training_factor_gain']
    text+=['',f"干净条件相对Uniform新增成功{paired['gained']}、新增失败{paired['lost']}，SR净变化{e['gain']*100:+.2f}点。错误接管统计来自各自实际路径，目标距离/选中旋转的后置标注只在评测端使用，未回流挖掘；不同路径计数不是同状态反事实，不能单独证明因果。",'',
        '## 当前阶段与边界','',
        f"当前属于S4扩展后的目标旋转鲁棒性探索。当前批{g['source_count']}源文件/{g['episodes']}题/seed、{g['unique_routes']}不同路线、首次策略评分新源文件{g['new_source_files']}；{a['neural_episodes']}轨迹/{a['neural_actions']}动作、{a['rule_records_rechecked_once']}规则和12个SR/SG区间全量复核，264运行前测试通过。已见Masa28开发不能当独立test，种子/多次路线不增加独立源图数，普通逐项95%区间不同时覆盖。",'',
        '已完成：环境/统一评测/最小VLM接入；本地视觉S2 SR89.33%/SG0.320，S3本地10独立源图86.80%/0.300，10×10密度77.13%/1.201，SwissView20新源图同模态迁移86.80%/0.381。不同任务/预算分别解释，不合并为一个总成功率。默认仍为Small256 NoTarget探索器+原Edge视觉头，阈值0.50；当前是训练过的单智能体模块系统，原零训练多智能体目标未交付。','',
        '10×10改变切图密度，未扩大真实地理覆盖；SwissView迁移是同模态连续航拍切图，严格地理不重叠及Sat2Cap预训练覆盖未知。真实异时/季节/任意方向/跨视角和25×25尚未验证。当前无云端调用、不移动原数据、不替换原默认或改写旧验收。','',
        '## 本轮决定','']
    if not r['candidate_numeric_passed']:text+=['本候选不进入新源图，原默认与已有正式验收保持；08…11试跑及40…59正式均不消耗。建议先收束本轮旋转训练探索，整理失败边界与剩余误接管；下一假设另行预登记，避免在同一开发集反复增加训练因素追分。']
    elif mode=='confirmation':text+=['新20源文件正式確認通过，高分负例可作为冻结直角旋转鲁棒候选；原默认未自动替换，其他扰动需另行独立验证。']
    else:text+=['允许已预登记下一批沿用冻结模型/阈值/策略；开发或四图试跑不当正式泛化确认。']
    text+=['','[预登记](预登记.json) / [独立复核](独立复核.json) / [验收](验收结论.json) / [诊断](接受与失败诊断.json)。']
    report.write_text('\n'.join(text)+'\n','utf-8');items=plots(out,mode,s,names)
    for item in items:item['reproduction_script']='project/src/documents/summarize_hard_rotation.py'
    manifest=dict(mode=mode,source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in (out/'对照汇总.json',out/'独立复核.json',out/'验收结论.json',out/'预登记.json',TRAIN/'独立复核.json',TRAIN/'诊断汇总.json',TRAIN/'挖掘诊断.json',TRAIN/'验收结论.json',diagfile)},script_sha256=digest(Path(__file__)),style_helper_sha256=digest(Path(graphics.__file__)),report_sha256=digest(report),figures=items)
    manifest_file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8');(FIG/f'{stem}图注.md').write_text('# 高分旋转负例图注\n\n'+'\n\n'.join(f"## {i['id']}\n\n{i['caption']}" for i in items)+'\n','utf-8');print(json.dumps(dict(report=str(report),passed=r['candidate_numeric_passed'],figures=names),ensure_ascii=False))


if __name__=='__main__':main()


