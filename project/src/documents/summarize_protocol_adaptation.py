"""Audited equal-budget training adaptation report and vector scientific figures."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.protocol_adaptation_navigation import ROOT,OUT,TRAIN,ARMS,SEEDS,read,write,digest,lines
from documents import stage_figures as graphics
from matplotlib import pyplot as plt

FIG=ROOT/'绘图';NAME='33_探索器训练协议适配成功率与距离'


def text_new(path,text):
    with path.open('x',encoding='utf8') as f:f.write(text)


def main():
    verdict=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');training=read(TRAIN/'独立复核.json');s=read(OUT/'主对照/对照汇总.json');reg=read(TRAIN/'预登记.json')
    if not verdict['audit_passed'] or verdict['audit_sha256']!=digest(OUT/'独立复核.json') or verdict['summary_sha256']!=digest(OUT/'主对照/对照汇总.json') or not training['passed']:raise ValueError('complete independent audit required')
    a=s['arms'];rec=s['recovery'];secondary=read(OUT/'继续训练对原基线_补充比较.json')['effect'];status='开发候选条件通过，仍须独立确认' if verdict['candidate_passed'] else '10×10协议适配未通过升级条件，收束本适配因素'
    diagnostics=read(OUT/'邻接时机诊断.json');timing={}
    for arm in ('M0','Continue5','Adapt10'):
        rows=[r for r in diagnostics if r['arm']==arm];near=[r for r in rows if r['budgeted_adjacency_observed']]
        timing[arm]=dict(records=len(rows),budgeted_adjacency_episodes=len(near),budgeted_adjacency_rate=len(near)/len(rows),
            mean_first_adjacency_step_among_reached=sum(r['first_adjacency_step'] for r in near)/len(near) if near else None,
            mean_remaining_at_first_adjacency_among_reached=sum(r['remaining_at_first_adjacency'] for r in near)/len(near) if near else None,
            failed_without_budgeted_adjacency=sum(not r['success'] and not r['budgeted_adjacency_observed'] for r in rows))
    write(OUT/'邻接时机汇总.json',dict(arms=timing,scope='posthoc true-goal diagnostics, conditional means have different reached cohorts; never used for model selection'))
    report=['# 探索器训练协议适配：同预算对照','',f'**{status}。** 原默认和既有验收保持；本轮没有云端调用、模型扩容或新的独立评测源文件。','',
        '## 唯一训练因素与公平预算','',
        '继续训练采用的任务协议为唯一对照因素：Continue5为原5×5/B10，Adapt10为10×10/B20，包括对应初始距离分布C4…8与C12…16。两臂不只差网格大小，因此不将效果拆归因于网格、预算或采样中的某一项。三原NoTarget权重逐seed作为共同起点，Small256总797701参数、可训练666117，encoder/目标头/0.50阈值/合法动作规则保持。没有新增100格输入、账本或邻域覆盖因素。','',
        '两臂每模型81920真实环境动作、1024优化器更新、128次rollout更新；64并行环境×10步rollout、2次pass×4个16序列mini-batch。GRU跨rollout携带，终局重置；不是因Adapt10预算20就把rollout改成20。PPO配方相同，优化器均重新建立；六模型全使用最后权重，全部训练和训练复核结束后才统一评测，未按开发SR早停或挑轮次。','',
        '原训练奖励success1 + .1格距离进展；PBRS beta.5、gamma.99、势函数按2(K−1)归一化，终局势为0，K5严格等于旧d/8。训练goal只用于奖励，公开NoTarget输入含fit均值、当前图特征与公开状态。没有目标坐标或真距离输入。','',
        '原109拟合/28已知开发源文件隔离，S4的20评测源文件与fit按文件名和原图SHA隔离。只在109-fit源图派生10900个10×10虚拟JPEG图块的全局特征，约21.3MiB；JPEG在内存编码，不保存新图块目录。原数据只读，encoder冻结，全部10900特征独立重新提取一致。','',
        '## 训练验证','',
        f"六新模型合计{training['new_training_actions']}动作、{training['optimizer_updates']}优化器更新。独立世界算术核查每次输入/转移/奖励，另重训六模型，所有最终参数与所有训练轨迹数组精确复现；验证重放另执行{training['verification_replay_actions']}动作，不计作方法额外训练预算、不使用重放权重替换原评测权重。",'',
        '6项新增测试通过，包括K5旧环境逐项等价、K10部署输入一致、真实goal变化不改变公开输入、预算与首次到达终局、PBRS尺度、内存JPEG与S4切图一致。旧运行模块没有改写。训练计时与显存只作资源记录，共享GPU/缓存预热不能当受控速度比较。','',
        '## 同题主评测','',
        '相同500题/20已见源文件、三权重、10×10/B20；Continue5/Adapt10新3000轨迹，M0已审计1500轨迹按SHA引用。不把三权重×题数称独立地图，也不把旧test文件名当本次未见确认。','',
        '| 方法 | 平均SR | 平均SG（所有终局） | 权重0/1/2 SR |','|---|---:|---:|---|']
    for arm,label in [('M0','原冻结策略'),('Continue5','5×5同预算继续训练'),('Adapt10','10×10同预算适配')]:
        r=a[arm];report.append(f"| {label} | {r['sr_mean']:.2%} | {r['sg_mean']:.3f} | {' / '.join(f'{x:.2%}' for x in r['sr_by_seed'])} |")
    report+=['','| 对照 | SR差（百分点） | SG差（格） | 正权重 | 源文件95% SR差区间（点） |','|---|---:|---:|---:|---|']
    for key,label in [('Adapt10_vs_Continue5','适配对继续训练'),('Adapt10_vs_M0','适配对原冻结策略')]:
        e=s['effects'][key];lo,hi=e['source95_SR'];report.append(f"| {label} | {100*e['sr_gain']:+.2f} | {e['sg_change']:+.3f} | {e['positive_seeds']}/3 | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lo,hi=secondary['source95_SR']
    report+=['',f"补充比较：Continue5对原M0平均SR{100*secondary['sr_gain']:+.2f}点、SG{secondary['sg_change']:+.3f}格，{secondary['positive_seeds']}/3权重正收益，源文件95% SR差区间[{100*lo:+.2f}, {100*hi:+.2f}]点。这是辅助对照，不改变本批Adapt10放行门槛、不释放条件目标对照预算。区间覆盖零，不能只凭三权重均正或平均+2.6点声称源文件层面稳定优势。",'',
        f"相对Continue5恢复{rec['recovered_vs_Continue5']}条、损伤{rec['harmed_vs_Continue5']}条；相对M0恢复{rec['recovered_vs_M0']}条、损伤{rec['harmed_vs_M0']}条。所有计划题保留，既报告恢复，也报告原成功损伤。",'',
        '配对源文件bootstrap4000次、seed5251，先在各源内平均三权重，再按20源成组重采样。SR/SG差区间均保存，不按1500条独立题计算。SG为全终局曼哈顿距离，成功记0。','',
        '## 获取线索时机：事后诊断','',
        '| 方法 | 预算内邻接记录 | 邻接覆盖率 | 已邻接者平均首邻接步 | 已邻接者平均剩余预算 | 失败且无预算内邻接 |','|---|---:|---:|---:|---:|---:|']
    for arm,label in [('M0','原策略'),('Continue5','继续训练'),('Adapt10','适配')]:
        t=timing[arm];step='—' if t['mean_first_adjacency_step_among_reached'] is None else f"{t['mean_first_adjacency_step_among_reached']:.2f}";remaining='—' if t['mean_remaining_at_first_adjacency_among_reached'] is None else f"{t['mean_remaining_at_first_adjacency_among_reached']:.2f}"
        report.append(f"| {label} | {t['budgeted_adjacency_episodes']}/1500 | {t['budgeted_adjacency_rate']:.2%} | {step} | {remaining} | {t['failed_without_budgeted_adjacency']} |")
    report+=['','这里的邻接真值仅用于事后诊断，不输入策略、不挑训练轮次。首次邻接统计有行动预算的决策前状态，零预算终步相邻不计可利用机会。条件均值的已邻接样本集合可能不同，不单凭均值变化断言所有题更早获得线索；SR仍为主指标。','',
        '## 冻结验收与边界','']
    for k,val in s['checks'].items():report.append(f"- {k}: {'通过' if val else '未通过'}")
    if verdict['target_controls_started']:
        target=read(OUT/'目标对照/目标证据汇总.json');report+=['',f"主数值通过，追加Adapt10无cue/均值目标/错目标4500轨迹，目标门槛{'通过' if target['all_target_numeric_passed'] else '未通过'}，详情在目标证据汇总。仍须独立确认，默认保持。"]
    else:report+=['','按协议不运行4500条Adapt10条件目标对照，不消费预留源图，不改训练轮数、lr或挑seed追求过线。收束当前密度扩展协议适配因素，保留已通过的主方法及适用边界。Continue5保留为新增训练预算的开发信号：成功率与Adapt10相近、SG更小；下一项先冻结这三最终权重，另立目标证据复验和独立确认方案，不能据未执行的控制或旧验收切换默认。']
    report+=['',f"评测独立复核重放{audit['records']}轨迹/{audit['actions']}动作，核查冻结目标头/两图输入/GRU、所有转移/终局、配对恢复损伤/邻接时机/源文件区间以及{audit['protected_files_verified']}历史文件和{audit['input_files_verified']}评测输入。复核由同一执行代理另行实现，不声称不同人员盲审。",'',
        '当前仍是训练过的单导航智能体模块系统。10×10是原同一地理足迹内切图密度提高，未验证更大实际地理面积、任意旋转或真实异时。用户弃用的8790三个模型未调用。','',
        '[执行协议](执行协议.md) / [预登记](预登记.json) / [训练复核](../../训练结果/探索器协议适配同预算对照_v1/独立复核.json) / [主结果](主对照/对照汇总.json) / [继续训练补充比较](继续训练对原基线_补充比较.json) / [逐题恢复损伤](逐题恢复与损伤.json) / [邻接时机](邻接时机汇总.json) / [评测复核](独立复核.json) / [验收](验收结论.json)']
    text_new(OUT/'探索器协议适配报告.md','\n'.join(report)+'\n')
    text_new(OUT/'README.md','# 探索器训练协议适配开发验证\n\n[报告](探索器协议适配报告.md) / [验收](验收结论.json) / [复核](独立复核.json) / [冻结协议](执行协议.md) / [逐题](逐题恢复与损伤.json)。\n\n两训练路线各500题×3，原M0按审计SHA复用；已见源文件开发对照，原默认保持。\n')
    text_new(TRAIN/'README.md','# 同预算训练协议适配\n\n[冻结协议](执行协议.md) / [预登记](预登记.json) / [全部训练结束](全部训练结束.json) / [训练复核](独立复核.json)。\n\n原三权重共同起点，Continue5/Adapt10各81920动作和1024优化器更新，六最终模型逐参数复现。训练复核目录是证据重放，不是新增方法训练预算。\n')
    resources={f'{arm}_s{seed}':read(TRAIN/f'{arm}_s{seed}/训练资源.json') for seed in SEEDS for arm in ARMS}
    write(TRAIN/'资源汇总.json',dict(models=resources,matched_new_training_actions=491520,verification_replay_actions=491520,shared_GPU_timing_not_controlled_benchmark=True))
    graphics.style();graphics.ITEMS.clear();fig,axes=plt.subplots(1,2,figsize=(7.6,4.2));labels=['原冻结策略','5×5继续训练','10×10适配'];colors=[graphics.GRAY,graphics.BLUE,graphics.TEAL]
    for panel,(field,scale,label) in enumerate([('sr',100,'SR：成功率（%）'),('sg',1,'SG：平均终点距离（格）')]):
        ax=axes[panel];highs=[]
        for i,arm in enumerate(('M0','Continue5','Adapt10')):
            r=a[arm];v=r[field+'_mean']*scale;seeds=[x*scale for x in r[field+'_by_seed']];high=max([v]+seeds);highs.append(high)
            ax.bar(i,v,width=.58,color=colors[i],edgecolor='#34424F',linewidth=.7,zorder=2)
            for seed,value in enumerate(seeds):ax.scatter(i+(seed-1)*.1,value,marker=['o','^','s'][seed],facecolor='white',edgecolor='#23313B',s=26,lw=.7,zorder=4)
            ax.text(i,high+(2.5 if panel==0 else .065),f'{v:.2f}' if panel==0 else f'{v:.3f}',ha='center',fontsize=10)
        ax.set_xticks(range(3),labels);ax.set_ylabel(label);ax.set_title('(a) 同题成功率' if panel==0 else '(b) 全终局距离',loc='left');ax.grid(axis='y');ax.set_axisbelow(True)
        ax.set_ylim(0,108 if panel==0 else max(highs)*1.27+.1)
        if panel==0:ax.set_yticks([0,20,40,60,80,100])
    fig.subplots_adjust(left=.09,right=.99,bottom=.22,top=.89,wspace=.34)
    fig.text(.5,.08,'20已见源文件，500题×3冻结目标头；两训练臂各81920动作、1024优化器更新。',ha='center',fontsize=9)
    fig.text(.5,.025,'评测均为10×10/B20；白色标记为seed0/1/2，非置信区间；参数量保持。',ha='center',fontsize=9)
    caption=f"探索器训练协议同预算适配：M0 SR{a['M0']['sr_mean']:.2%}/SG{a['M0']['sg_mean']:.3f}，Continue5 {a['Continue5']['sr_mean']:.2%}/{a['Continue5']['sg_mean']:.3f}，Adapt10 {a['Adapt10']['sr_mean']:.2%}/{a['Adapt10']['sg_mean']:.3f}。{status}，默认保持。20已见源文件、同500题×三权重、10×10/B20；两训练臂各81920实际动作与1024优化器更新，Small256/encoder/目标头/阈值保持。训练协议包含网格密度、预算及距离分布，不单独归因网格。标记为权重seed非CI，SG为所有终局曼哈顿距离；不代表独立新图或更大实际面积确认。"
    for ext in ('png','pdf','svg'):
        if (FIG/f'数据结果图/{NAME}.{ext}').exists():raise ValueError('immutable figure exists')
    graphics.save(fig,FIG/'数据结果图',NAME,caption,[(OUT/'主对照/对照汇总.json').relative_to(ROOT).as_posix(),(TRAIN/'独立复核.json').relative_to(ROOT).as_posix(),(OUT/'独立复核.json').relative_to(ROOT).as_posix()],
        'equal-budget continued training,20known evaluation sources,500tasks/3weights')
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/summarize_protocol_adaptation.py'
    write(FIG/'绘图数据/探索器训练协议适配_v1.json',dict(figures=[item],summary_sha256=digest(OUT/'主对照/对照汇总.json'),audit_sha256=digest(OUT/'独立复核.json'),min_font_pt=9,recommended_width_cm=18))
    text_new(FIG/'探索器训练协议适配图注.md','# 探索器训练协议适配图注\n\n'+caption+'\n')
    print(dict(report=str(OUT/'探索器协议适配报告.md'),figure=str(FIG/f'数据结果图/{NAME}.png')),flush=True)


if __name__=='__main__':main()
