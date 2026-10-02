"""Audited neighborhood-planning development report and scientific figure."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.neighborhood_coverage import ROOT,OUT,PRIOR,SEEDS,read,write,digest
from documents import stage_figures as graphics
from matplotlib import pyplot as plt

FIG=ROOT/'绘图';NAME='32_目标邻域覆盖规划成功率与距离'


def text_new(path,text):
    with path.open('x',encoding='utf8') as f:f.write(text)


def main():
    verdict=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');s=read(OUT/'主对照/对照汇总.json');diag=read(OUT/'原失败与静态机会诊断.json')
    if not verdict['audit_passed'] or verdict['audit_sha256']!=digest(OUT/'独立复核.json') or verdict['summary_sha256']!=digest(OUT/'主对照/对照汇总.json'):raise ValueError('audited summary required')
    a=s['arms'];rec=s['recovery'];budget_diag=read(OUT/'终步预算补充诊断.json');status='开发候选条件通过' if verdict['candidate_passed'] else '未通过升级条件，收束本因素'
    report=['# 目标邻域覆盖规划：单因素开发验证','',f'**{status}。** 本轮无云端请求、训练或新源图，默认保持。用户弃用的8790三个模型没有调用。','',
        '## 已有失败与新假设','',
        f"原M0的1500条记录中有{diag['failures']}条失败，其中{diag['failed_never_adjacent']}条未在仍有行动预算的决策状态进入目标相邻格，{diag['failed_adjacent_reached']}条有这种邻接机会仍失败。统计决策前状态，不把最后一次移动后的零预算邻接视为可利用的成功机会。这是查看既有数据后的开发诊断，不能据此证明唯一失败原因；真值只进入事后评价。",'',
        f"补充诊断：上述332条中，{budget_diag['first_adjacency_at_zero_budget']}条第一次邻接发生在最后一次移动后、预算已为零；{budget_diag['never_adjacent_including_terminal']}条直到终局仍未邻接。补充在主评测后另存，只澄清失败口径，不改控制器或候选门槛。",'',
        '冻结视觉头识别目标是否与当前图邻接，新规划尝试创造更多可由该视觉头检查的候选位置。设H(S)为S内格子及合法四邻并集，候选路径P的新增机会为|H(P)−H(visited)|。它是几何计数，不是视觉概率、未知图像评分或真实目标方向。过去邻域已覆盖也不代表其中目标已被排除，仍逐步运行原视觉头。','',
        f"在旧M1的{diag['static_events']}次介入状态中，新评分改变了{diag['static_changed_paths']}条候选路径；静态诊断不产生新的SR，不把预测的覆盖增加当成实际成功。",'',
        '## 冻结队列与唯一因素','',
        'M0为原已验收冻结策略；M1为原两步账本几何规划；N只把新增目标邻域机会放在候选排序第一项，之后原几何排序、explorer首动作匹配和候选编号同分规则保持。当前接受cue仍最高优先；原最早触发、每题一次介入、两步上限、权重/阈值/预算/目标图与GRU逐步更新均保持，不叠加上一轮保护否决。','',
        '相同500题、20已见源文件、三训练权重，10×10/B20、C12…16各100题。N新执行1500条；M0/M1各1500已审计记录按SHA复用，没有重复运行旧臂。已见源文件开发对照不称独立test、S3或严格地理泛化。','',
        '候选条件为N对M0平均SR至少+2点、至少2/3正权重、SG不差；对M1平均SR严格正收益且至少2/3正权重、SG不差；全部正常合法终局及全量复核。仅主数值通过才运行无cue/均值目标/错目标4500条证据对照；新候选也不自动改默认。','',
        '## 主结果','',
        '| 方法 | 平均SR | 平均SG（所有终局） | 权重0/1/2 SR |','|---|---:|---:|---|']
    for arm,label in [('M0','原冻结策略'),('M1','原账本规划'),('N','目标邻域覆盖规划')]:
        r=a[arm];report.append(f"| {label} | {r['sr_mean']:.2%} | {r['sg_mean']:.3f} | {' / '.join(f'{x:.2%}' for x in r['sr_by_seed'])} |")
    report+=['','| 对照 | SR差（百分点） | SG差（格） | 正权重 | 源文件95% SR差区间（点） |','|---|---:|---:|---:|---|']
    for key,label in [('N_vs_M0','N对M0'),('N_vs_M1','N对M1')]:
        e=s['effects'][key];lo,hi=e['source95_SR'];report.append(f"| {label} | {100*e['sr_gain']:+.2f} | {e['sg_change']:+.3f} | {e['positive_seeds']}/3 | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    report+=['',f"N相对M0恢复{rec['recovered_vs_M0']}条、损伤{rec['harmed_vs_M0']}条，净增加{rec['recovered_vs_M0']-rec['harmed_vs_M0']}/1500；相对M1恢复{rec['recovered_vs_M1']}条、损伤{rec['harmed_vs_M1']}条。所有计划题保留，同一500题在三权重重复不是1500张地图。",'',
        '源文件配对bootstrap4000次、seed5251，先在各源内平均三训练权重，再成组采样20源文件；不把每题当独立源文件。SR/SG区间均在主汇总，SG为所有终局曼哈顿距离，成功记0。','',
        '## 分距离结果','',
        '| 初始C | M0 SR | M1 SR | N SR | N SG |','|---|---:|---:|---:|---:|']
    for c in range(12,17):report.append(f"| {c} | {a['M0']['by_distance'][str(c)]['sr']:.2%} | {a['M1']['by_distance'][str(c)]['sr']:.2%} | {a['N']['by_distance'][str(c)]['sr']:.2%} | {a['N']['by_distance'][str(c)]['sg']:.3f} |")
    report+=['','## 冻结判定与后续边界','']
    for key,val in s['checks'].items():report.append(f"- {key}: {'通过' if val else '未通过'}")
    if verdict['target_controls_started']:
        target=read(OUT/'目标对照/目标证据汇总.json');report+=['',f"主数值通过，新增4500条无cue/均值目标/错目标对照，目标门槛{'通过' if target['all_target_numeric_passed'] else '未通过'}。详见目标证据汇总；不能借用原S4验收替代新增N策略目标证据。"]
    else:report+=['','按预定停止条件没有启动4500条目标对照，也未消费预留源图；保留原默认与既有S2/S3/S4/SwissView结论。']
    report+=['',
        '本轮只检验一次局部规划目标替换。几何机会更多可能改变长期路径，也可能损伤原本有效的探索；两步局部计数不代表全回合最优预算规划。不在本批调评分权重、触发或步数寻找过线方案。下一项优先检查5×5/B10探索器训练与10×10/B20评测之间的协议适配，保留架构与冻结目标头、设置同预算5×5继续训练对照；这是一项待验证假设，不把协议差异说成已证明的原因。下一项草案尚未启动训练或消耗新图。仍是训练过的单导航智能体模块系统，未通过的支线不计为多Agent协作成果。','',
        f"新增5项测试通过，旧运行模块保持，之前319项全套记录按SHA引用。独立实现候选邻域计算、排序、触发与控制，冻结权重重放{audit['records']}条轨迹/{audit['actions']}动作；独立核验失败诊断/静态路径、配对恢复损伤、源文件区间、{audit['input_files_verified']}项输入和{audit['protected_files_verified']}项历史文件。复核由同一执行代理另行实现，不声称不同人员盲审。",'',
        '[执行协议](执行协议.md) / [预登记](预登记.json) / [失败与静态机会诊断](原失败与静态机会诊断.json) / [终步预算诊断](终步预算补充诊断.json) / [主结果](主对照/对照汇总.json) / [逐题恢复损伤](逐题恢复与损伤.json) / [独立复核](独立复核.json) / [验收](验收结论.json)']
    text_new(OUT/'目标邻域覆盖规划报告.md','\n'.join(report)+'\n')
    text_new(OUT/'README.md','# 目标邻域覆盖规划开发验证\n\n[结果报告](目标邻域覆盖规划报告.md) / [验收](验收结论.json) / [复核](独立复核.json) / [冻结设计](执行协议.md) / [逐题](逐题恢复与损伤.json)。\n\nN新1500条，原M0/M1按已审计哈希复用；20已见源文件开发验证。默认保持，8790三个选项不调用。\n')
    resources=[dict(seed=seed,elapsed_seconds=r['elapsed_seconds'],triggered=r['triggered'],changed_actions=r['changed_actions']) for seed in SEEDS for r in [read(OUT/f'主对照/N_s{seed}_CueFull_结果.json')]]
    write(OUT/'资源汇总.json',dict(jobs=resources,cloud_calls=0,new_training_steps=0,controlled_speed_benchmark=False))
    graphics.style();graphics.ITEMS.clear();fig,axes=plt.subplots(1,2,figsize=(7.6,4.2));labels=['M0 原策略','M1 账本','N 邻域覆盖'];colors=[graphics.GRAY,graphics.BLUE,graphics.TEAL]
    for panel,(field,scale,ylabel) in enumerate([('sr',100,'SR：成功率（%）'),('sg',1,'SG：平均终点距离（格）')]):
        ax=axes[panel];highs=[]
        for i,arm in enumerate(('M0','M1','N')):
            r=a[arm];v=r[field+'_mean']*scale;seeds=[x*scale for x in r[field+'_by_seed']];high=max([v]+seeds);highs.append(high)
            ax.bar(i,v,width=.58,color=colors[i],edgecolor='#34424F',linewidth=.7,zorder=2)
            for seed,value in enumerate(seeds):ax.scatter(i+(seed-1)*.10,value,marker=['o','^','s'][seed],facecolor='white',edgecolor='#23313B',s=26,lw=.7,zorder=4)
            ax.text(i,high+(2.5 if panel==0 else .065),f'{v:.2f}' if panel==0 else f'{v:.3f}',ha='center',fontsize=10)
        ax.set_xticks(range(3),labels);ax.set_ylabel(ylabel);ax.set_title('(a) 同题成功率' if panel==0 else '(b) 全终局距离',loc='left');ax.grid(axis='y');ax.set_axisbelow(True)
        ax.set_ylim(0,108 if panel==0 else max(highs)*1.27+.1)
        if panel==0:ax.set_yticks([0,20,40,60,80,100])
    fig.subplots_adjust(left=.09,right=.99,bottom=.22,top=.89,wspace=.34)
    fig.text(.5,.08,'20已见源文件，500题×3冻结权重；10×10，B=20，未训练、未调用云端。',ha='center',fontsize=9)
    fig.text(.5,.025,'M0/M1引用已审计结果，N新执行；白色标记为权重0/1/2，非置信区间。',ha='center',fontsize=9)
    caption=f"目标邻域覆盖单因素开发验证：M0 SR{a['M0']['sr_mean']:.2%}/SG{a['M0']['sg_mean']:.3f}，M1 {a['M1']['sr_mean']:.2%}/{a['M1']['sg_mean']:.3f}，N {a['N']['sr_mean']:.2%}/{a['N']['sg_mean']:.3f}。{status}，默认保持。20已见源文件、同500题、三冻结权重、10×10/B20；N新1500轨迹，旧M0/M1按已审计SHA引用。标记为训练权重而非CI，SG为所有终局曼哈顿距离；不代表新源图或更大实际面积确认。"
    for ext in ('png','pdf','svg'):
        if (FIG/f'数据结果图/{NAME}.{ext}').exists():raise ValueError('immutable figure exists')
    graphics.save(fig,FIG/'数据结果图',NAME,caption,[(OUT/'主对照/对照汇总.json').relative_to(ROOT).as_posix(),(OUT/'独立复核.json').relative_to(ROOT).as_posix()],
        'known-source neighborhood-planning development,500tasks/20sources/3frozen weights')
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/summarize_neighborhood_coverage.py'
    write(FIG/'绘图数据/目标邻域覆盖规划_v1.json',dict(figures=[item],summary_sha256=digest(OUT/'主对照/对照汇总.json'),audit_sha256=digest(OUT/'独立复核.json'),min_font_pt=9,recommended_width_cm=18))
    text_new(FIG/'目标邻域覆盖规划图注.md','# 目标邻域覆盖规划图注\n\n'+caption+'\n')
    print(dict(report=str(OUT/'目标邻域覆盖规划报告.md'),figure=str(FIG/f'数据结果图/{NAME}.png')),flush=True)


if __name__=='__main__':main()
