"""Audited orientation ablation, false-acceptance diagnostics and publication figures."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics
from eval.orientation_confirmation import ROOT,VARIANTS,STRATEGIES,paths
from train.local_capacity import read,lines
from train.dyncur_tiny import digest
FIG=ROOT/'绘图'
LABELS=dict(Original='原策略',Candidate4='四方向候选',NoTarget='无目标探索')


def diagnostics(out):
    tasks={e['episode_id']:e for e in read(out/'导航任务.json')};result={};head_calls=0
    def near(a,b):return abs(a//5-b//5)+abs(a%5-b%5)
    for v in VARIANTS:
        result[v]={}
        for a in STRATEGIES:
            seeds=[]
            for s in range(3):
                rows=lines(out/f'Edge_s{s}/导航_{v}_{a}_轨迹.jsonl');accepted=correct=aligned=adjacent=0;calls=0;false_nonadjacent=0
                for r in rows:
                    goal=tasks[r['episode_id']]['goal']
                    for i,d in enumerate(r['decisions']):
                        calls+=4 if a=='Candidate4' else 1
                        if d['cue_action'] is None:continue
                        accepted+=1;hit=r['trajectory'][i+1]['patch_id']==goal;correct+=hit
                        adjacent+=near(r['trajectory'][i]['patch_id'],goal)==1
                        false_nonadjacent+=not hit and near(r['trajectory'][i]['patch_id'],goal)!=1
                        if a=='Candidate4':aligned+=(d['selected_relative_clockwise']+(0 if v=='Clean' else 90))%360==0
                head_calls+=calls
                seeds.append(dict(seed=s,accepted=accepted,correct=correct,false_accepts=accepted-correct,
                    accepted_precision=correct/accepted if accepted else None,accepted_when_adjacent=adjacent,
                    false_accepts_without_adjacent_goal=false_nonadjacent,selected_absolute_R0_on_accepted=aligned if a=='Candidate4' else None,
                    head_forward_calls=calls,neural_actions=sum(r['steps'] for r in rows)))
            result[v][a]=seeds
    return dict(scope='posthoc evaluator truth; not a policy input or calibration',conditions=result,total_head_forward_calls=head_calls,
        online_runtime_note='shared resource load; total run duration is not a controlled speed comparison')


def draw(out,mode,summary,names):
    averages=summary['averages'];graphics.style();graphics.ITEMS.clear();colors=[graphics.GRAY,graphics.BLUE,graphics.TEAL];hatches=['','//','..']
    fig,axes=plt.subplots(1,2,figsize=(7.6,4.25));x=np.arange(2);width=.24
    maxsg=max(averages[v][a]['sg_mean'] for v in VARIANTS for a in STRATEGIES)
    for i,a in enumerate(STRATEGIES):
        sr=[averages[v][a]['sr_mean']*100 for v in VARIANTS];sg=[averages[v][a]['sg_mean'] for v in VARIANTS];pos=x+(i-1)*width
        for ax,values in zip(axes,[sr,sg]):ax.bar(pos,values,width*.94,color=colors[i],hatch=hatches[i],label=LABELS[a],edgecolor='#34424F',linewidth=.6,zorder=2)
        for j,v in enumerate(VARIANTS):
            for seed,(off,marker) in enumerate([(-.045,'o'),(0,'^'),(.045,'s')]):
                metrics=read(out/f'Edge_s{seed}/导航_{v}_{a}_结果.json')['metrics']
                axes[0].scatter(pos[j]+off,metrics['sr']*100,s=18,marker=marker,facecolor='white',edgecolor='#24323B',linewidth=.7,zorder=4)
                axes[1].scatter(pos[j]+off,metrics['mean_sg_all_episodes'],s=18,marker=marker,facecolor='white',edgecolor='#24323B',linewidth=.7,zorder=4)
            axes[0].text(pos[j],102,f'{sr[j]:.2f}',ha='center',fontsize=9)
            axes[1].text(pos[j],maxsg*1.27,f'{sg[j]:.3f}',ha='center',fontsize=9)
    for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True);ax.set_xticks(x,['目标方向一致','给定目标顺时针90°']);ax.tick_params(axis='x',length=0)
    axes[0].set_ylim(0,112);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR：成功率（%）')
    axes[1].set_ylim(0,maxsg*1.43);axes[1].set_ylabel('SG：平均终点距离（格）')
    axes[0].set_title('(a) 同题成功率与全部三份权重',loc='left');axes[1].set_title('(b) SG包含成功与失败',loc='left')
    axes[0].legend(loc='upper center',bbox_to_anchor=(1.05,1.23),ncol=3,frameon=False)
    fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.80,wspace=.31)
    reg=read(out/'预登记.json');scope=f"{'Masa已知开发' if mode=='development' else 'SwissView新源文件'}{reg['source_count']}图/{reg['episodes']}题×3；5×5/B10；白色标记：三份权重，非CI。"
    fig.text(.5,.025,scope,ha='center',fontsize=9)
    clean=averages['Clean'];rot=averages['Rot90CW'];caption=f"四方向候选在本批干净/旋转题上取得相同SR{clean['Candidate4']['sr_mean']:.2%}，原策略分别为{clean['Original']['sr_mean']:.2%}/{rot['Original']['sr_mean']:.2%}。候选来自给定像素的0/90/180/270度无插值变换，原权重、阈值0.50、当前图与预算不变，未提供真角度；全部权重均展示。SR为成功比例，SG为全部终点距离均值；本图范围为{scope}，候选是否放行还取决于干净保护与目标收益。"
    graphics.save(fig,FIG/'数据结果图',names[0],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],scope)
    fig,ax=plt.subplots(figsize=(7.6,3.85));labels=['旋转恢复：候选 − 原旋转','干净保护：候选 − 原干净','干净目标收益：候选 − 探索','旋转目标收益：候选 − 探索'];keys=['rotation_recovery','clean_safety','clean_target_gain','rotation_target_gain']
    for i,k in enumerate(keys):
        interval=summary['effects'][k]['source_interval'];mu=interval['mean']*100;lo,hi=np.array(interval['interval95'])*100
        ax.errorbar(mu,i,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt=['o','s','^','D'][i],color=graphics.BLUE,ms=6,capsize=3)
        ax.text(hi+1,i,f'{mu:+.2f}',va='center',fontsize=9)
        ax.scatter([-2 if k=='clean_safety' else 5],[i],marker='|',s=160,color=graphics.ORANGE,zorder=4)
    ax.axvline(0,color='#34424F',ls='--',lw=.8);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.grid(axis='x');ax.set_axisbelow(True)
    limits=[summary['effects'][k]['source_interval']['interval95'] for k in keys];ax.set_xlim(min(-3,min(lo for lo,_ in limits)*100-4),max(7,max(hi for _,hi in limits)*100+10))
    ax.set_xlabel('SR配对差（百分点）：正值表示候选更高');ax.set_title('四方向搜索需要同时通过旋转收益与干净保护',loc='left')
    fig.subplots_adjust(left=.39,right=.965,bottom=.22,top=.86)
    fig.text(.5,.035,'源文件成组95%区间；橙短线：+5点收益／−2点干净保护；SG与种子要求另列报告。',ha='center',fontsize=9)
    caption='恢复旋转题不代表候选整体可用，干净题的预定保护同样决定放行。四项全部展示，每个源文件先平均三份权重的配对差，再seed3031/2000次成组bootstrap；区间为逐项95%，不声称四项同时保证。橙短线为运行前固定的+5点收益或−2点干净保护尺度，完整门槛还包含SG、至少两份正权重及全量复核。'
    graphics.save(fig,FIG/'数据结果图',names[1],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],scope)
    return graphics.ITEMS


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],default='development');mode=p.parse_args().mode;out=paths(mode)[1]
    summary=read(out/'对照汇总.json');receipt=read(out/'验收结论.json');audit=read(out/'独立复核.json');reg=read(out/'预登记.json')
    assert audit['status']=='passed' and digest(out/'独立复核.json')==receipt['audit_sha256'] and digest(out/'对照汇总.json')==receipt['summary_sha256']
    stem={'development':'方向候选开发','pilot':'方向候选试跑','confirmation':'方向候选独立确认'}[mode];first={'development':17,'pilot':19,'confirmation':21}[mode]
    names=[f'{first:02d}_{stem}成功率与距离',f'{first+1:02d}_{stem}配对收益与保护'];report=out/'方向候选验证报告.md';diagfile=out/'线索接受诊断.json';source=FIG/f'绘图数据/{stem}_v1.json'
    if any(p.exists() for p in (report,diagfile,source)) or any((FIG/'数据结果图'/f'{n}.{fmt}').exists() for n in names for fmt in ('png','pdf','svg')):raise ValueError('immutable reporting artifacts exist')
    diag=diagnostics(out);diagfile.write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n','utf-8');avg=summary['averages'];failed=[k for k,v in receipt['release_checks'].items() if not v]
    text=['# 目标方向候选单因素验证','',f"**本轮{'通过预定放行门槛' if receipt['candidate_numeric_passed'] else '未通过预定放行门槛'}；全部数据与动作独立复核通过。** 固定{reg['source_count']}源文件/{reg['episodes']}题×三权重、两目标条件、Original/Candidate4/NoTarget三臂，共{audit['neural_episodes']}在线轨迹。",'',
        '## 同题成绩','', '| 目标条件 | 策略 | 成功/总数 | SR | SG（格） | 三权重SR |','|---|---|---:|---:|---:|---|']
    for v in VARIANTS:
        for a in STRATEGIES:
            m=avg[v][a];text.append(f"| {'干净' if v=='Clean' else '顺时针90°'} | {LABELS[a]} | {sum(m['successes_by_seed'])}/{3*reg['episodes']} | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} | {' / '.join(f'{s:.2%}' for s in m['sr_by_seed'])} |")
    text+=['','Candidate4在两目标条件的每条动作、所选候选像素、概率和终态完全一致，支持整90度候选集合的不变性实现；不等于找到真朝向，也不支持任意角度。NoTarget两条件轨迹不变，开发Original干净与NoTarget复现旧140题。', '',
        '| 配对比较 | SR差（点） | 源文件95%区间（点） | 正权重数 | SG变化（格） |','|---|---:|---|---:|---:|']
    titles=dict(rotation_recovery='旋转候选−原旋转',clean_safety='干净候选−原干净',clean_target_gain='干净候选−探索',rotation_target_gain='旋转候选−探索')
    for k,e in summary['effects'].items():
        lo,hi=e['source_interval']['interval95'];text.append(f"| {titles[k]} | {e['gain']*100:+.2f} | [{lo*100:+.2f},{hi*100:+.2f}] | {e['positive_seeds']}/3 | {e['sg_change']:+.3f} |")
    text+=['', '预定旋转恢复≥5点、至少两正权重、SG不更差、源文件95%下界>0；干净SR最多损失2点、SG最多增加0.10；两条件对探索均≥5点/至少两正权重/SG不更差。门槛在成绩出现前写定，未因结果改动。干净点估计保护不是统计非劣证明，逐项区间不等于四项同时覆盖。', '',
        f"未满足项：{', '.join(failed) if failed else '无'}。" ,'',
        '## 错误接受与成本','', '| 条件/策略 | 接受线索数 | 正确到目标 | 错误接受数 | 实际接受精度 |','|---|---:|---:|---:|---:|']
    for v in VARIANTS:
        for a in ('Original','Candidate4'):
            rows=diag['conditions'][v][a];n=sum(r['accepted'] for r in rows);correct=sum(r['correct'] for r in rows)
            text.append(f"| {v}/{LABELS[a]} | {n} | {correct} | {n-correct} | {f'{correct/n:.2%}' if n else '无接受'} |")
    text+=['','实际接受精度在执行后用真目标核验，未用于候选排序/阈值。候选取四个原五类向量中的最大原始方向概率，再调用原完整向量门控；最大分数没有角度校准，扩大的搜索空间可能增加误接受。本轮不叠加新阈值、重排合法次候选或歧义筛选。', '',
        f"额外成本：每给定目标最多4次编码（可缓存），每动作4次冻结视觉头评分；本批预提取{audit['images']['candidate_images']}个旋转视图，全部策略在线视觉头调用{diag['total_head_forward_calls']}次。模型/均值不训练、不扩大；运行总耗时{read(out/'执行状态.json')['seconds']:.1f}秒（含数据准备/编码/轨迹，不含独立复核）。共享负载时延不作受控效率基准。", '',
        '## 数据范围与证据','',
        'Masa28图是既有已知开发留出，未参加109探索拟合或87头部拟合/22校准，但曾用于方法选择，不能冒充新独立test。开发全过后使用事先固定SwissView04…07兼容试跑、40…59独立源文件确认；原SwissView20正式图不用于本项调参。独立单位为源文件，重复路线和三权重不会增加地图数量，严格地理不重叠与预训练覆盖未知。', '',
        f"本报告当前批次：{mode}，源文件{reg['source_count']}，首次用于策略评测的新源文件{reg['new_source_files']}；不同路线{reg['unique_routes']}，重复抽样{reg['episodes']-reg['unique_routes']}。", '',
        f"248项运行前测试、{audit['images']['current_images']}当前JPEG原图重建、{audit['images']['candidate_images']}目标PNG与全部全局/局部/profile特征独立复算；{audit['neural_episodes']}轨迹/{audit['neural_actions']}动作、{audit['rule_records_rechecked_once']}规则和8个SR/SG源文件区间复核通过。编码器特征最大绝对误差global/local={audit['images']['global_max_abs_error']:.3g}/{audit['images']['local_max_abs_error']:.3g}。",'',
        '同一执行代理运行不同算法路径复核，不声称不同人员审稿。旧模型、默认、原数据与历史验收按哈希保留；源码/任务/像素/特征/权重/完整决策可追溯。采集快照的audit_pending不回写，最终以验收结论为准。', '',
        '原审计器对完全对称目标的SHA并列假设了绝对角度顺序，导致标签误报；原件保持冻结，另存修复按给定图的相对输入顺序重建并列标签。两项修复回归通过，完整复核重新执行，未重跑实验、改策略或覆盖轨迹。入口`project/src/eval/audit_orientation_confirmation_repair.py`；本批[修复记录](审计修复记录.json)保留前后哈希。', '',
        '## 后续决定','']
    if not receipt['candidate_numeric_passed']:
        text+=['本候选不放行，不使用新SwissView源图，不替换已验证默认。下一项应把多候选错误接受/歧义校准作为单独因素，在开发数据上验证；不要把裁剪、记忆或重训练一起加入。方向枚举可保留为研究候选，但不能宣称方向问题已解决。']
    elif mode=='confirmation':text+=['正式独立源文件确认通过；只支持固定四直角候选和本地连续航拍切图设定。默认仍保持原样，后续裁剪/尺度/真实异时另起验证。']
    else:text+=['本批允许进入预登记的下一批；冻结策略与阈值继续，不在新源图调参。当前批不能代替正式独立确认。']
    text+=['','[预登记](预登记.json) / [完整复核](独立复核.json) / [最终验收](验收结论.json) / [线索诊断](线索接受诊断.json)。']
    report.write_text('\n'.join(text)+'\n','utf-8');items=draw(out,mode,summary,names)
    for item in items:item['reproduction_script']='project/src/documents/summarize_orientation.py'
    manifest=dict(source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [out/'对照汇总.json',out/'独立复核.json',out/'验收结论.json',out/'预登记.json',diagfile]},
        script_sha256=digest(Path(__file__)),style_helper_sha256=digest(Path(graphics.__file__)),report_sha256=digest(report),figures=items,mode=mode)
    source.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    (FIG/f'{stem}图注.md').write_text('# 目标方向候选图注\n\n'+'\n\n'.join(f"## {item['id']}\n\n{item['caption']}" for item in items)+'\n','utf-8')
    print(json.dumps(dict(report=str(report),passed=receipt['candidate_numeric_passed'],figures=names),ensure_ascii=False))


if __name__=='__main__':main()
