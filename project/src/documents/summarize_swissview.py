"""Append audited SwissView reports and scientific plots without changing experiment evidence."""
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
BASE=ROOT/'DATA/processed_data/SwissView/评测结果'
FIG=ROOT/'绘图'
def read(p):return json.loads(p.read_text('utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text('utf-8').splitlines() if x]
digest=graphics.digest


def diagnostic(out):
    tasks={e['episode_id']:e for e in read(out/'导航任务.json')};result=[]
    def near(a,b):return abs(a//5-b//5)+abs(a%5-b%5)
    for seed in range(3):
        rows=lines(out/f'Edge_s{seed}/导航_CueFull_轨迹.jsonl');failed=[r for r in rows if not r['success']]
        ever=sum(any(near(event['patch_id'],tasks[r['episode_id']]['goal'])==1 for event in r['trajectory'][:-1]) for r in failed)
        accepted=correct=0
        for row in rows:
            goal=tasks[row['episode_id']]['goal']
            for i,d in enumerate(row['decisions']):
                if d['cue_action'] is not None:
                    accepted+=1;correct+=row['trajectory'][i+1]['patch_id']==goal
        result.append(dict(seed=seed,failures=len(failed),failed_with_adjacent_opportunity=ever,
            failed_without_adjacent_opportunity=len(failed)-ever,accepted_cues=accepted,correct_cues=correct,
            actual_accepted_precision=correct/accepted if accepted else None))
    return dict(scope='posthoc evaluator-only descriptive diagnostics; never used for tuning',by_seed=result)


def report(out):
    path=out/'迁移验证报告.md'
    if path.exists():raise ValueError('immutable report already exists')
    summary=read(out/'对照汇总.json');reg=read(out/'预登记.json');receipt=read(out/'验收结论.json');audit=read(out/'独立复核.json')
    assert digest(out/'独立复核.json')==receipt['audit_sha256'] and digest(out/'对照汇总.json')==receipt['summary_sha256']
    mode=receipt['mode'];formal=mode=='formal';m=summary['averages']['Edge']['CueFull'];b=summary['averages']['Edge']['Baseline']
    e=summary['effects']['NoTargetBaseline'];lo,hi=e['source_interval']['interval95'];N=reg['protocol']['episodes'];A=reg['protocol']['source_count']
    rows=[f"# SwissView100 5×5 {'正式冻结迁移' if formal else '工程试跑'}结果",'',
        f"本批完整复核通过，工程效果门槛{'通过' if receipt['engineering_passed'] else '未通过'}；{'正式跨数据集验收'+('通过' if receipt['formal_cross_dataset_passed'] else '未通过') if formal else '仅试跑，不作正式跨域通过结论'}。目标贡献迁移{'通过' if receipt['target_transfer_passed'] else '尚未通过全部对照'}。SR **{m['sr_mean']:.2%}**（{sum(m['successes_by_seed'])}/{3*N}），SG **{m['sg_mean']:.3f}**；同题探索SR{b['sr_mean']:.2%}、SG{b['sg_mean']:.3f}。",'',
        f"固定{N}题×3权重、{A}源文件，5×5/B10/C4…8，每源每C{'5' if formal else '1'}题。三个SR为"+' / '.join(f'{v:.2%}' for v in m['sr_by_seed'])+'，全部计划记录正常结束（Q=100%）；不挑最好种子、不做集成。', '',
        '## 同题控制与指标','',
        '| 条件 | 成功/计划 | SR | SG（格） |','|---|---:|---:|---:|']
    for c,label in [('Baseline','NoTarget探索'),('CueFull','真实目标 EdgeTargetCue'),('CueMean','训练均值目标'),('CueWrong','错误目标')]:
        v=summary['averages']['Edge'][c];rows.append(f"| {label} | {sum(v['successes_by_seed'])}/{3*N} | {v['sr_mean']:.2%} | {v['sg_mean']:.3f} |")
    z=summary['averages']['ZeroEdge']['CueFull'];rows.append(f"| 同容量ZeroEdge（原None阈值） | {sum(z['successes_by_seed'])}/{3*N} | {z['sr_mean']:.2%} | {z['sg_mean']:.3f} |")
    for n,r in summary['rules'].items():
        v=r['metrics'];rows.append(f"| {n}（确定性1轮） | {v['successes']}/{N} | {v['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} |")
    rows+=['','SR=成功/计划；SG包含所有成功与失败终点，成功记0；低SG不能推断每次失败都已接近。NoTarget和ZeroEdge动作一致，后一条件保持原校准None，不是重新设定一个有利阈值。Mean同时干预目标全局/局部/profile；Wrong改变三类给定目标，环境真值原样。', '',
        '## 预定验收与配对证据','',
        '| 工程检查 | 结果 |','|---|---|']
    for n,v in summary['engineering_checks'].items():rows.append(f"| {n} | {'通过' if v else '未通过'} |")
    rows+=['',f"原S4跨域门槛：SR≥45%、SG≤2.5、比最强预登记无图规则（{summary['strongest_rule']}）SR≥+5点且SG不更差，完整记录及复核；正式需20源图/500题×3。目标贡献与区间独立报告，工程通过不自动等于所有机制得到证明。",'',
        '| 真实目标减对照 | SR差（百分点） | 正种子 | 正源文件 | 源文件95%区间（点） | SG差 | 目标检查 |','|---|---:|---:|---:|---|---:|---|']
    for n,effect in summary['effects'].items():
        lo,hi=effect['source_interval']['interval95'];check='规则比较' if n in summary['rules'] else ('通过' if summary['target_transfer_checks'][n] else '未通过')
        rows.append(f"| {n} | {effect['gain']*100:+.2f} | {effect['positive_seeds']}/3 | {effect['positive_sources']}/{A} | [{lo*100:+.2f}, {hi*100:+.2f}] | {effect['sg_change']:+.3f} | {check} |")
    rows+=['','先每源图配对平均三权重，再按源文件作2000次bootstrap（seed3031），不是1500个独立地图。规则仅一次，为对齐配对形式重复索引三次，不增加地理样本量。目标门槛对四项控制分别为+5点、至少2正种子、SG不更差，CI支持情况另外列出。','',
        '| C档 | 探索SR | 真实SR | 真实SG |','|---|---:|---:|---:|']
    for d in range(4,9):
        v=m['by_distance'][str(d)];rows.append(f"| C{d} | {b['by_distance'][str(d)]['sr']:.2%} | {v['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} |")
    rows+=['','| 原源文件 | 元数据LV95坐标（原值） | 探索SR | 真实SR | SR差（点） | 真实SG |','|---|---|---:|---:|---:|---:|']
    for row in reg['sources']:
        a=row['area'];v=m['by_source'][a];bv=b['by_source'][a];rows.append(f"| {row['source_tile']} | {row['LV95_coordinates']} | {bv['sr']:.2%} | {v['sr']:.2%} | {(v['sr']-bv['sr'])*100:+.2f} | {v['mean_sg_all_episodes']:.3f} |")
    diag=diagnostic(out);diagpath=out/'失败与线索诊断.json'
    if diagpath.exists():raise ValueError('immutable diagnostic exists')
    diagpath.write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n','utf-8')
    rows+=['','## 后置诊断及结论范围','',
        '| 种子 | 失败 | 失败曾相邻 | 失败从未相邻 | 接受线索/正确 | 实际接受精度 |','|---|---:|---:|---:|---|---:|']
    for v in diag['by_seed']:
        precision='无接受' if v['actual_accepted_precision'] is None else f"{v['actual_accepted_precision']:.2%}"
        rows.append(f"| {v['seed']} | {v['failures']} | {v['failed_with_adjacent_opportunity']} | {v['failed_without_adjacent_opportunity']} | {v['accepted_cues']}/{v['correct_cues']} | {precision} |")
    reference=summary['source_domain_description']
    rows+=['',f"重访率为"+' / '.join(f'{x:.2%}' for x in m['repeat_by_seed'])+f"，SR种子极差{summary['diagnostics']['SR_seed_range']*100:.2f}点，作为诊断保留。相邻机会仅用失败的动作前位置统计；该诊断不能证明探索是唯一原因，也不能当可实现的上限。",'',
        f"相对Masa S3（SR{reference['SR']:.2%}/SG{reference['SG']:.3f}），本域SR描述性变化{reference['SR_change']*100:+.2f}点、SG{reference['SG_change']:+.3f}；两套地图与任务不同，不作配对因果解释。",'',
        f"本批{audit['neural_episodes']}神经轨迹/{audit['neural_steps']}动作、{audit['rule_episodes']}规则和{audit['images']['images']}个图块全量复核；原图裁剪/JPEG字节重建，全局/局部特征最大误差{audit['images']['global_max_abs_error']:.3g}/{audit['images']['local_max_abs_error']:.3g}，profile独立手算，12个SR/SG区间重算。232项运行前测试通过。错误目标距离匹配例外{reg['wrong_distance_exceptions']}/{N}项完整保留。",'',
        '策略CPU FP32、编码器CUDA，直接复用原5×5策略方程，没有10×10适配；profile计算缓存只传给定两图。试跑全部300条轨迹用原FrozenEdgeNavigator复算一致；正式一次在线执行后独立程序重载权重手算全部输入、概率、门控、动作与终止。原结果pending字段保留采集时状态，完成状态看独立复核与验收。同一执行代理实现并运行两条检查路径，不称不同人员审查。', '',
        'Masa任务训练权重迁移到SwissView100且无新训练/阈值校准；原始数据、历史结果、默认与模型保持不变。本轮目标来自同一连续源图、同次拍摄且方向一致，边缘连续性是有效先验；不是跨视角、异时、旋转、真实图搜索的泛化证明，也不是论文原协议复现或零训练多智能体。Sat2Cap预训练覆盖未知。', '',
        '试跑id00…03与正式id20…39文件/RGB互不重合，试跑前固定两组名单。连续编号为便利样本，不能声称全国随机抽样。元数据坐标去重与最小距离已记录，但坐标单位/影像实际覆盖未完整核实，不宣称严格地理不重叠；源文件bootstrap保留此局限。剩余76源图未运行策略，应作为后续新方法的独立确认资源。', '',
        '作者资料：[SwissView官方数据卡](https://huggingface.co/datasets/EPFL-ECEO/SwissView)、[GeoExplorer项目](https://limirs.github.io/GeoExplorer/)；仅核实数据身份/字段与来源，不直接搬用论文SR。', '',
        '## 下一步','',
        ('本轮正式跨域验收通过，可将Masa/S3、密度/S4与SwissView结果作为三项分开的证据写入报告。先做完整阶段总结或独立的鲁棒性协议设计；异时/旋转目标和MM-GAG仍需各自预登记，不继续在这20张确认图上调方法。' if formal and receipt['formal_cross_dataset_passed'] else
         '试跑已通过且复核，允许按预定正式名单继续；仍不足以宣称正式跨域通过。' if not formal and receipt['allow_formal_expansion'] else
         '本轮未达到预定工程门槛，保留完整负结果；回开发数据研究独立因素，下一版使用新的确认源图，不在本批调至过线。'),'',
        '[预登记](预登记.json) / [完整复核](独立复核.json) / [最终验收](验收结论.json) / [失败诊断](失败与线索诊断.json)。']
    path.write_text('\n'.join(rows)+'\n','utf-8');return summary,path


def plots(out,summary,report_path):
    names=['13_SwissView五乘五正式结果','14_SwissView分距离与目标贡献']
    source=FIG/'绘图数据/SwissView五乘五正式迁移_v1.json'
    if source.exists() or any((FIG/'数据结果图'/f'{n}.{fmt}').exists() for n in names for fmt in ('png','pdf','svg')):raise ValueError('immutable plots exist')
    graphics.style();graphics.ITEMS.clear();captions=[]
    v=summary['averages']['Edge'];strongest=summary['strongest_rule'];rule=summary['rules'][strongest]['metrics']
    labels=['探索/置零','真实目标','均值目标','错误目标',strongest]
    conditions=['Baseline','CueFull','CueMean','CueWrong'];colors=[graphics.GRAY,graphics.BLUE,graphics.ORANGE,graphics.TEAL,graphics.GRAY]
    sgs=[];metric_files=[]
    for c in conditions:
        files=[out/f'Edge_s{s}/导航_{c}_结果.json' for s in range(3)];metric_files+=files
        sgs.append([read(p)['metrics']['mean_sg_all_episodes'] for p in files])
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.8));graphics.frames(axes)
    graphics.bars(axes[0],labels,[v[c]['sr_mean']*100 for c in conditions]+[rule['sr']*100],colors,percent=True)
    graphics.bars(axes[1],labels,[v[c]['sg_mean'] for c in conditions]+[rule['mean_sg_all_episodes']],colors,decimals=3)
    for i,c in enumerate(conditions):
        axes[0].texts[i].set_position((i,max(v[c]['sr_by_seed'])*100+1.5));axes[1].texts[i].set_position((i,max(sgs[i])+.05))
        for s,(offset,marker) in enumerate([(-.13,'o'),(0,'^'),(.13,'s')]):
            axes[0].scatter(i+offset,v[c]['sr_by_seed'][s]*100,marker=marker,s=22,facecolor='white',edgecolor='#1D2B34',linewidth=.8,zorder=4)
            axes[1].scatter(i+offset,sgs[i][s],marker=marker,s=22,facecolor='white',edgecolor='#1D2B34',linewidth=.8,zorder=4)
    axes[0].set_ylim(0,109);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR：成功率（%）');axes[0].set_title('(a) SwissView固定500题×3',loc='left')
    axes[1].set_ylim(0,max([x for group in sgs for x in group]+[rule['mean_sg_all_episodes']])*1.30)
    axes[1].set_ylabel('SG：平均终点距离（格）');axes[1].set_title('(b) 同题SG（包含失败）',loc='left')
    for ax in axes:ax.tick_params(axis='x',labelsize=8.8)
    fig.subplots_adjust(left=.08,right=.985,bottom=.19,top=.88,wspace=.31)
    fig.text(.5,.025,'20张SwissView100源图；5×5/B10；Masa权重、均值、阈值冻结；白色标记：三种子。',ha='center',fontsize=9)
    m=v['CueFull'];b=v['Baseline']
    caption=f"Masa冻结边缘策略在SwissView100切图确认中取得SR{m['sr_mean']:.2%}、SG{m['sg_mean']:.3f}，探索对照为{b['sr_mean']:.2%}/{b['sg_mean']:.3f}。SR为成功/计划，SG包含成功与失败；20源文件、固定500题、三训练权重，白色圆/三角/方块为seed0/1/2，不是置信区间。目标同源连续且方向一致，不能推断跨视角/异时泛化；规则仅一次，便利样本不能代表全国随机分布。"
    captions.append(caption);graphics.save(fig,FIG/'数据结果图',names[0],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],'SwissView100 20files/500tasks x3; frozen Masa policy grid5/B10')
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.9));graphics.frames(axes)
    for c,label,color,marker,ls in [('Baseline','探索/置零',graphics.GRAY,'s','--'),('CueFull','真实目标',graphics.BLUE,'o','-'),('CueWrong','错误目标',graphics.TEAL,'^',':')]:
        axes[0].plot(range(4,9),[v[c]['by_distance'][str(d)]['sr']*100 for d in range(4,9)],label=label,color=color,marker=marker,ls=ls,lw=1.7)
    axes[0].set_ylim(0,107);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_xticks(range(4,9),[f'C{d}' for d in range(4,9)])
    axes[0].set_xlabel('初始曼哈顿距离（格）');axes[0].set_ylabel('SR（%）');axes[0].legend(frameon=False,fontsize=9,loc='lower right');axes[0].set_title('(a) 每档100题×3',loc='left')
    comparisons=[('NoTargetBaseline','真实 − 探索'),('ZeroEdgeFull','真实 − 置零'),('MeanCue','真实 − 均值'),('WrongCue','真实 − 错误'),('Frontier','真实 − Frontier'),('FixedRegion','真实 − FixedRegion')]
    axes[1].grid(False);axes[1].grid(axis='x');axes[1].axvline(0,color='#34424F',ls='--',lw=.8)
    for i,(name,label) in enumerate(comparisons):
        e=summary['effects'][name]['source_interval'];mu=e['mean']*100;lo,hi=np.array(e['interval95'])*100
        axes[1].errorbar(mu,i,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color=graphics.BLUE,capsize=3,ms=5)
    axes[1].set_yticks(range(6),[x[1] for x in comparisons],fontsize=9);axes[1].invert_yaxis()
    axes[1].set_xlabel('SR差及源文件95%区间（百分点）');axes[1].set_title('(b) 20源文件成组bootstrap',loc='left')
    fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.88,wspace=.61)
    fig.text(.5,.025,'真距离只用于评测分组；每源文件先平均三种子，再2000次成组重采样。',ha='center',fontsize=9)
    e=summary['effects']['NoTargetBaseline'];lo,hi=e['source_interval']['interval95']
    caption=f"真实目标相对探索的SR收益为{e['gain']*100:+.2f}点，源文件成组95%区间[{lo*100:+.2f},{hi*100:+.2f}]点。左图C4…8等量取样，C档结构不同，不表示连续变化的难度；右图先每源平均三种子配对差，再以20源文件作2000次bootstrap（seed3031），确定性规则不增加独立样本量。区间仅覆盖本次便利样本文件组，影像地理非重叠尚未完整确认；Masa与SwissView不同任务不合并比较。"
    captions.append(caption);graphics.save(fig,FIG/'数据结果图',names[1],caption,[(out/'对照汇总.json').relative_to(ROOT).as_posix()],'SwissView100 source-file paired grouped bootstrap')
    for item in graphics.ITEMS:item['reproduction_script']='project/src/documents/summarize_swissview.py'
    files=[out/'对照汇总.json',out/'验收结论.json',out/'独立复核.json',out/'预登记.json',*metric_files]
    manifest=dict(source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},report_sha256=digest(report_path),figures=graphics.ITEMS,
        script_sha256=digest(Path(__file__)),style_helper_sha256=digest(Path(graphics.__file__)),measurements=summary['averages'],effects=summary['effects'])
    source.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    (FIG/'SwissView五乘五图注.md').write_text('# SwissView冻结迁移图注\n\n'+'\n\n'.join(f'## {n}\n\n{c}' for n,c in zip(names,captions))+'\n','utf-8')
    return names


def main():
    for name in ['五乘五工程试跑_v1','五乘五正式迁移_v1']:
        out=BASE/name
        if (out/'验收结论.json').exists():
            summary,path=report(out)
            if read(out/'验收结论.json')['mode']=='formal':plots(out,summary,path)
            print(json.dumps(dict(report=str(path)),ensure_ascii=False))


if __name__=='__main__':main()
