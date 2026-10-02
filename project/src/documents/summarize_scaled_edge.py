"""Publish audited density-scale reports/figures, preserving all legacy artifacts."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1'
PILOT=ROOT/'DATA/processed_data/Masa/评测结果/S4十乘十工程试跑_v1'
FIG=ROOT/'绘图'

def read(p):return json.loads(p.read_text('utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text('utf-8').splitlines() if v]


def diagnostic():
    """Post-hoc descriptive evaluator truth; never used in policy/threshold choice."""
    tasks={e['episode_id']:e for e in read(OUT/'导航任务.json')};by_seed=[]
    for s in range(3):
        rows=lines(OUT/f'Edge_s{s}/导航_CueFull_轨迹.jsonl');accepted=correct=adjacent=0;failed_adj=0
        for row in rows:
            ep=tasks[row['episode_id']];ever=False
            for d in row['decisions']:
                r,c=d['public_position'];gr,gc=divmod(ep['goal'],10);near=abs(r-gr)+abs(c-gc)==1
                adjacent+=near;ever|=near
                if d['cue_action'] is not None:
                    accepted+=1;dr,dc={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}[d['cue_action']]
                    correct+=(r+dr)*10+c+dc==ep['goal']
            failed_adj+=(not row['success']) and ever
        failures=sum(not r['success'] for r in rows)
        by_seed.append(dict(seed=s,episodes=500,failures=failures,failed_with_pre_action_adjacent_opportunity=failed_adj,
            failed_without_pre_action_adjacent_opportunity=failures-failed_adj,pre_action_adjacent_states=adjacent,
            accepted_cues=accepted,accepted_cues_reaching_true_goal=correct,accepted_cues_not_reaching_true_goal=accepted-correct,
            accepted_cue_precision_on_actual_trajectories=correct/accepted if accepted else None))
    return dict(scope='Post-hoc descriptive evaluator-only diagnostic; not gate, achievable oracle SR, or causal attribution.',by_seed=by_seed,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [OUT/'导航任务.json',*[OUT/f'Edge_s{s}/导航_CueFull_轨迹.jsonl' for s in range(3)]]})


def main():
    receipt=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');summary=read(OUT/'对照汇总.json');reg=read(OUT/'预登记.json')
    assert receipt['audit_sha256']==digest(OUT/'独立复核.json') and audit['status']=='passed'
    assert receipt['summary_sha256']==digest(OUT/'对照汇总.json')
    report=OUT/'十乘十扩展报告.md';diagnostic_path=OUT/'失败与线索诊断.json';plotdata=FIG/'绘图数据/S4十乘十正式扩展_v1.json'
    names=['11_S4十乘十正式结果','12_S4十乘十分距离与增益']
    expected=[report,diagnostic_path,plotdata,FIG/'S4十乘十图注.md']+[FIG/f'数据结果图/{n}.{fmt}' for n in names for fmt in ('png','pdf','svg')]
    if any(p.exists() for p in expected):raise ValueError('immutable publication exists')
    diag=diagnostic();diagnostic_path.write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n','utf-8')
    main=summary['averages']['Edge']['CueFull'];base=summary['averages']['Edge']['Baseline'];e=summary['effects']['NoTargetBaseline'];ci=e['source_interval']['interval95']
    labels=[('Edge','Baseline','NoTarget探索'),('ZeroEdge','CueFull','同容量置零'),('Edge','CueFull','真实目标'),('Edge','CueMean','均值目标'),('Edge','CueWrong','错误目标')]
    rows=['# S4：10×10密度与紧预算正式扩展','',
        f"**本地10×10/B20工程验收：{'通过' if receipt['formal_S4_passed'] else '未通过'}；目标贡献迁移检查：{'全部通过' if receipt['target_transfer_passed'] else '存在不足项'}。** 最终判断见[验收结论](验收结论.json)，运行汇总中的pending字段保留原生成状态。",'',
        f"固定500题×3权重、20源图文件、{reg['task_scope']['unique_routes']}不同路线，C12…16等量、B20、四方向。源图为原test10张和旧28开发留出的前10张，均未用于109拟合，但此前用于S3/方法开发；本轮不是新独立源图确认。",'',
        '| 同题方法/条件 | 三种子SR | 成功 / 计划 | 平均SR | SG | 三种子重访率 |','|---|---|---|---:|---:|---|']
    for a,c,label in labels:
        v=summary['averages'][a][c]
        rows.append(f"| {label} | {' / '.join(f'{p:.1%}' for p in v['sr_by_seed'])} | {sum(v['successes_by_seed'])}/1500 | {v['sr_mean']:.2%} | {v['sg_mean']:.3f} | {' / '.join(f'{p:.2%}' for p in v['repeat_by_seed'])} |")
    for n,r in summary['rules'].items():
        v=r['metrics'];rows.append(f"| {n}（1轮） | {v['sr']:.1%} | {v['successes']}/500 | {v['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} | {v['repeat_visit_rate_micro']:.2%} |")
    rows+=['',f"真实目标相对同题探索的SR收益为 **{e['gain']*100:+.2f}个百分点**，{e['positive_seeds']}/3种子、{e['positive_sources']}/20源图正增益；源文件成组95%区间[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]点。归一化SG为{summary['normalised_SG']:.4f}，口径SG/18。",'',
        '## 原S4门槛与视觉证据','',
        '原门槛：≥20源图、每C每源图5题、三权重、完整正常终态；平均SR≥30%、SG≤4.5（归一化≤0.25），对最强预登记规则SR≥+5点且SG不更差。大网格R作诊断，未套用S3的R10%门槛。目标对照另外判定，CI不代替工程门槛。','',
        '| 工程检查 | 判定 |','|---|---|']
    for k,v in summary['engineering_checks'].items():rows.append(f'| {k} | {"通过" if v else "未通过"} |')
    rows+=['','| 真实目标减对照 | SR差（点） | 正增益种子 | 正增益源图 | SR差95%区间（点） | SG差 | SG差95%区间 | 目标迁移 |',
        '|---|---:|---:|---:|---|---:|---|---|']
    for n,v in summary['effects'].items():
        lo,hi=v['source_interval']['interval95'];sl,sh=v['sg_source_interval']['interval95'];check='规则比较' if n in summary['rules'] else ('通过' if summary['target_transfer_checks'][n] else '未通过')
        rows.append(f"| {n} | {v['gain']*100:+.2f} | {v['positive_seeds']}/3配对 | {v['positive_sources']}/20 | [{lo*100:+.2f},{hi*100:+.2f}] | {v['sg_change']:+.3f} | [{sl:+.3f},{sh:+.3f}] | {check} |")
    rows+=['','源文件先平均三种子，再seed3031/2000次成组bootstrap。规则只执行一次；其复用仅对齐配对，不是三次独立规则或60个地图簇。视觉贡献尺度：SR≥+5点、至少2种子正增益、SG不更差。','',
        '## 新尺度数据与适配','',
        '原1500×1500源图行优先切成100个150格，每格单独BICUBIC放大300后JPEG75。先裁再放大，不为插值跨格取像素。该设定加密同一地理范围，放大不增加原生细节；不能写成更大的真实地理覆盖。','',
        '原探索器1052维/25格访问输入无法直接接受100格。显式适配为位置/9、预算/20，细格访问按2×2汇总25粗槽、cap3/3，丢失格内访问细节；真实100格visited仍用于最高线索未访问筛选。原GRU、Edge/ZeroEdge权重、训练均值、0.50/None阈值、合法动作筛选与argmax不变；新增适配是推理配置变化，不是未经修改的原5×5策略。','',
        'Sat2Cap以CUDA编码；小策略CPU/FP32运行，边缘profile与全局/局部特征一同预计算。行动端每步仅接收给定两张图的特征/profile，无全图bank、源ID、真目标/距离或未访问图块。Mean覆盖全部目标通道，Wrong整题固定替换，真环境目标不变。','',
        '设备及profile缓存路径在原300条CUDA试跑上核验5491动作全部一致，最大概率差4.18e−7、logits差1.91e−6。合成输入设备计时不是算法基准；不宣称普遍加速倍数。正式每题一次在线执行，独立程序重载权重并全量手算复核，不再用相同实现重复一次。','',
        '## 距离与已见来源分组','',
        '| C档（每档100题×3） | 探索SR | 真实目标SR | 真实目标SG |','|---|---:|---:|---:|']
    for d in range(12,17):
        b=base['by_distance'][str(d)];v=main['by_distance'][str(d)]
        rows.append(f"| C{d} | {b['sr']:.2%} | {v['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} |")
    rows+=['','| 已见来源组（各10图/250题×3） | 探索SR | 真实目标SR | 真实目标SG |','|---|---:|---:|---:|']
    for group,v in summary['source_group_results'].items():
        rows.append(f"| {group} | {v['Baseline']['sr']:.2%} | {v['CueFull']['sr']:.2%} | {v['CueFull']['mean_sg_all_episodes']:.3f} |")
    rows+=['','test与dev均已见，只按预定等量权重报告，不挑更好一组。固定试卷有放回采样；20个题ID与试跑相同、按路线组合计有'+str(reg['pilot_route_overlap'])+'次已见路线出现，均重新在线执行，不拼旧日志。500抽样/三权重不是1500张独立地图。','',
        '## 邻接可靠性与失败诊断','',
        '| 头种子 | 邻接方向DA | 已接受raw线索精度 | 非邻接误接受率 |','|---|---:|---:|---:|']
    probe=read(OUT/'邻接探针汇总.json')
    for s,v in probe.items():rows.append(f"| {s} | {v['direction_accuracy_on_adjacent']:.2%} | {v['raw_accepted_precision']:.2%} | {v['false_accept_rate_on_nonadjacent']:.2%} |")
    rows+=['','14400对为7200邻接＋7200非邻接，三头共43200预测，已完整复核。该精度依赖1:1诊断基率，是raw最高联合概率，不是实际导航的后验线索精度，更不等于导航SR；未据此调阈值。','',
        '| 种子 | 失败题 | 失败中曾在动作前邻接 | 失败中从未在动作前邻接 | 实际被接受线索精度 |','|---|---:|---:|---:|---:|']
    for v in diag['by_seed']:
        rows.append(f"| {v['seed']} | {v['failures']} | {v['failed_with_pre_action_adjacent_opportunity']} | {v['failed_without_pre_action_adjacent_opportunity']} | {v['accepted_cue_precision_on_actual_trajectories']:.2%} |")
    rows+=['','以上为事后描述性评测端诊断，仅使用真实目标计算，不输入策略、不参与门槛，不表示某条线索导致了失败或某批失败可被自动挽救。SG含成功的0距离，整体SG小不说明每道失败都接近目标。细粒度访问适配与远距离探索是否是主要原因仍需单因素验证。','',
        '## 完整复核与结论边界','',
        f"224项测试通过；7500条神经轨迹/{audit['neural_actions']}动作、1000规则、2000原图裁剪与全局/局部/profile、43200探针预测及12个SR/SG区间全部复核。裁剪JPEG从原PNG逐字节重建，全局/局部最大全部误差分别为{audit['images']['global_max_abs_error']:.3g}/{audit['images']['local_max_abs_error']:.3g}。Wrong有{reg['wrong_distance_exceptions']}/500个无法匹配起点距离的例外，推理前固定且完整保留。",'',
        '复核由同一执行代理另行实现，手算公开状态、模型输入、概率、门控、合法动作、边界/终止、规则与统计；复用冻结试跑的独立动作检查组件，不称不同人员审查。权重/均值/源码/编码器/数据/旧S2、S3、试跑与默认哈希核验通过。原汇总pending保留原态，最终复核与验收才给出通过与否。','',
        '原冻结审计器在首次重建图片时缺失image_payload导入，保留[原始错误](原始审计异常.json)及原源码，另用audit_scaled_edge_confirmation_repair.py补齐同一元数据剥离函数绑定后完成全部复核。模型、任务、阈值和结果未改动；[修复记录](审计修复记录.json)保存补丁源码/原始源码/原结果哈希，后续审计使用修复入口。','',
        '原S3独立源文件确认与本轮不同：本轮20图未用于109拟合但用于开发/S3，是新尺度工程验收，没有新的未触及独立地图，也无地理不重叠保证。保持连续同源、方向一致、目标是本源格图片的局部协议；未验证异时、旋转、跨视角或跨数据集。不得直接以本分数声称超过论文不同划分/训练/目标模态的结果。仍为训练过的单智能体模块系统。','',
        '## 下一步','',
        ('本地10×10紧预算密度扩展完成。下一项优先选新数据集，保持5×5/A、B10、C4…8，先冻结原权重/阈值/采样与图像裁剪规则，再20题工程验证及来源隔离。原跨域标准要求至少20区域/500题×3、SR≥45%、SG≤2.5、相对同题预登记规则+5点；不把单数据集新网格通过当跨域已经验证。不自动启动25×25或叠加记忆/子目标/规划。' if receipt['formal_S4_passed'] else
         '本轮未达标项保留，先回开发集分析公开状态适配或线索迁移的单因素问题，下一版独立确认需新的未用于开发材料。不得调本批500题至过线，不直接开展跨域/25×25。'),'',
        '[预登记](预登记.json) / [完整复核](独立复核.json) / [后验失败诊断](失败与线索诊断.json)。']
    report.write_text('\n'.join(rows)+'\n','utf-8')

    # Append a small explanation to the completed pilot; no existing artifact rewritten.
    pilot_report=PILOT/'试跑结果说明.md'
    if not pilot_report.exists():
        pilot_summary=read(PILOT/'对照汇总.json');v=pilot_summary['averages']['Edge']['CueFull'];b=pilot_summary['averages']['Edge']['Baseline']
        pilot_report.write_text('# 10×10工程试跑：结果说明\n\n'+f"四张已见test源图、20题×3；真实目标SR{v['sr_mean']:.2%}/SG{v['sg_mean']:.3f}，探索{b['sr_mean']:.2%}/{b['sg_mean']:.3f}。300神经/5491动作、40规则、400格与8640探针预测复核通过，达到预登记的扩大探索条件；不是正式S4。\n\n"+
            '原5×5输入/动作兼容核验通过；新10×10密度数据和2×2访问汇总属于新增推理配置，原权重/阈值不改。CPU与缓存输入复算保持全部300条试跑的输入、动作、门控、终态一致。正式500题/20图结果另见../S4十乘十正式扩展_v1/十乘十扩展报告.md。两批分数不相减作算法提升。\n','utf-8')

    graphics.style();captions=[];plot_labels=['探索/置零','真实目标','均值目标','错误目标',summary['strongest_rule']]
    values=[summary['averages']['Edge'][c] for c in ('Baseline','CueFull','CueMean','CueWrong')];rule=summary['rules'][summary['strongest_rule']]['metrics']
    colors=[graphics.GRAY,graphics.BLUE,graphics.ORANGE,graphics.TEAL,graphics.GRAY]
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.8));graphics.frames(axes)
    for ax,key,scale,decimals in [(axes[0],'sr_mean',100,2),(axes[1],'sg_mean',1,3)]:
        final=rule['sr']*100 if key=='sr_mean' else rule['mean_sg_all_episodes']
        graphics.bars(ax,plot_labels,[v[key]*scale for v in values]+[final],colors,decimals=decimals,percent=key=='sr_mean')
        for i,v in enumerate(values):
            series=v['sr_by_seed'] if key=='sr_mean' else v['sg_by_seed'];ax.texts[i].set_position((i,max(series)*scale+(1.5 if scale==100 else .025)))
            for y,off,marker in zip(series,(-.13,0,.13),('o','^','s')):ax.scatter(i+off,y*scale,s=22,marker=marker,facecolor='white',edgecolor='#1D2B34',linewidth=.8,zorder=4)
        ax.tick_params(axis='x',labelsize=8.8)
    axes[0].set_ylim(0,109);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR：成功率（%）');axes[0].set_title('(a) 固定500题×3',loc='left')
    axes[1].set_ylim(0,max([rule['mean_sg_all_episodes'],*[max(v['sg_by_seed']) for v in values]])*1.3);axes[1].set_ylabel('SG：平均终点距离（格）');axes[1].set_title('(b) 包含成功与失败',loc='left')
    fig.subplots_adjust(left=.08,right=.985,bottom=.19,top=.88,wspace=.31)
    fig.text(.5,.025,'10×10，B=20；20张已见源图；白色标记为三训练种子，规则1轮。',ha='center',fontsize=9)
    caption=f"冻结边缘线索及公开状态适配在10×10密度/紧预算确认中将SR从{base['sr_mean']:.2%}提高至{main['sr_mean']:.2%}，SG从{base['sg_mean']:.3f}降至{main['sg_mean']:.3f}。固定500题×3、20张已见但未拟合源图、C12…16/B20，探索与置零轨迹一致，规则仅1轮；白色圆/三角/方块为seed0/1/2而非CI。数据是原1500图细切150格放大300，未扩大真实地理覆盖，也不是新的独立源图或跨数据集确认。"
    captions.append(caption);graphics.save(fig,FIG/'数据结果图',names[0],caption,[str((OUT/'对照汇总.json').relative_to(ROOT))],'formal500x3/grid10/B20/density/20already-used non-fit source files')

    fig,axes=plt.subplots(1,2,figsize=(7.6,3.9));graphics.frames(axes)
    for c,label,color,marker,ls in [('Baseline','探索/置零',graphics.GRAY,'s','--'),('CueFull','真实目标',graphics.BLUE,'o','-'),('CueWrong','错误目标',graphics.TEAL,'^',':')]:
        v=summary['averages']['Edge'][c];axes[0].plot(range(12,17),[v['by_distance'][str(d)]['sr']*100 for d in range(12,17)],label=label,color=color,marker=marker,ls=ls,lw=1.7)
    axes[0].set_ylim(0,107);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_xticks(range(12,17),[f'C{d}' for d in range(12,17)])
    axes[0].set_xlabel('初始曼哈顿距离（格）');axes[0].set_ylabel('SR（%）');axes[0].legend(frameon=False,loc='lower right',fontsize=9);axes[0].set_title('(a) 每档100题×3',loc='left')
    comparisons=[('NoTargetBaseline','真实 − 探索'),('ZeroEdgeFull','真实 − 置零'),('MeanCue','真实 − 均值'),('WrongCue','真实 − 错误'),('Frontier','真实 − Frontier'),('FixedRegion','真实 − FixedRegion')]
    axes[1].grid(False);axes[1].grid(axis='x');axes[1].axvline(0,color='#34424F',ls='--',lw=.8)
    for i,(n,label) in enumerate(comparisons):
        v=summary['effects'][n]['source_interval'];mu=v['mean']*100;lo,hi=np.array(v['interval95'])*100
        axes[1].errorbar(mu,i,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color=graphics.BLUE,capsize=3,ms=5)
    axes[1].set_yticks(range(6),[v[1] for v in comparisons],fontsize=9);axes[1].invert_yaxis();axes[1].set_xlabel('SR差及源图95%区间（百分点）');axes[1].set_title('(b) 20源文件成组bootstrap',loc='left')
    fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.88,wspace=.61)
    fig.text(.5,.025,'源文件先平均三种子，再2000次成组重采样；同地理范围的密度压力测试。',ha='center',fontsize=9)
    caption=f"10×10真实目标相对同题探索的SR收益为{e['gain']*100:+.2f}点，20源文件成组95%区间[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]点。左图为C12…16等量组，真实距离仅用于评测、不输入策略；右图源文件内先平均配对种子，再以20文件为簇作2000次bootstrap。10张已用S3图与10张已用开发图预定等权；区间不是新地理分布的独立确认，不将三份权重或规则复制当更多地图。"
    captions.append(caption);graphics.save(fig,FIG/'数据结果图',names[1],caption,[str((OUT/'对照汇总.json').relative_to(ROOT))],'grid10/C12..16/B20; grouped paired intervals on20known source files')
    for item in graphics.ITEMS:item['reproduction_script']='project/src/documents/summarize_scaled_edge.py'
    source_files=[OUT/'对照汇总.json',OUT/'验收结论.json',OUT/'独立复核.json',OUT/'预登记.json',OUT/'邻接探针汇总.json',diagnostic_path]
    manifest=dict(source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_files},report_sha256=digest(report),
        script_sha256=digest(Path(__file__)),plot_style_helper_sha256=digest(Path(graphics.__file__)),figures=graphics.ITEMS,
        measurements=summary['averages'],effects=summary['effects'],scope='audited local grid-density formal confirmation,20already-used non-fit source files')
    plotdata.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    (FIG/'S4十乘十图注.md').write_text('# S4 10×10正式结果图注\n\n'+'\n\n'.join(f'## {n}\n\n{c}' for n,c in zip(names,captions))+'\n\n[来源与哈希](绘图数据/S4十乘十正式扩展_v1.json)。\n','utf-8')
    print(json.dumps(dict(formal_S4_passed=receipt['formal_S4_passed'],report=str(report),figures=names),ensure_ascii=False))


if __name__=='__main__':main()
