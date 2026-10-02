"""Audited development report and traceable vector result figure."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.ledger_protection import ROOT,OUT,PRIOR,SEEDS,read,write,digest
from documents import stage_figures as graphics
from matplotlib import pyplot as plt

FIG=ROOT/'绘图';NAME='31_账本探索保护规则成功率与距离'


def text_new(path,text):
    with path.open('x',encoding='utf8') as f:f.write(text)


def main():
    verdict=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');s=read(OUT/'主对照/对照汇总.json');diag=read(OUT/'上一轮损伤诊断.json')
    if not verdict['audit_passed'] or verdict['audit_sha256']!=digest(OUT/'独立复核.json') or verdict['summary_sha256']!=digest(OUT/'主对照/对照汇总.json'):raise ValueError('audited summary required')
    a=s['arms'];rec=s['recovery'];status='开发候选门槛通过' if verdict['candidate_passed'] else '保护因素未通过升级条件，收束该因素'
    report=['# 账本探索保护规则：诊断与单因素验证','',f'**{status}。** 原默认保持；无新增训练、云端导航调用或未见源图。','',
        '## 先分析所有配对记录','',
        '上一轮M1恢复51条、损伤32条；损伤来自32个不同题ID、17个源文件。以下公开状态诊断使用全部1500对记录，标签只用于事后评价。规则不是在未见集上提出的，不能把本轮称独立确认。','',
        '| 旧结果组 | 记录 | 首次偏离时原动作进入新格 | 首次偏离在第二步 |','|---|---:|---:|---:|']
    for k,label in [('harm','损伤'),('recover','恢复'),('same_success','保持成功'),('same_fail','保持失败')]:
        r=diag['groups'][k];report.append(f"| {label} | {r['total']} | {r.get('first_diverge_base_fresh',0)} | {r.get('first_diverge_second_step',0)} |")
    report+=['','损伤30/32与恢复49/51均会被“保护原新格动作”条件覆盖，两类信息重叠。24条损伤与37条恢复首次偏离在两步计划第二步；这说明强制第二步有介入风险，同时也提供探索收益，不能只看损伤提出乐观效果承诺。','',
        '## 唯一新因素与冻结队列','',
        'P继承M1公开账本、原触发、一次介入和两步候选。每一步若option要改原动作、原动作下一格尚未访问，就保留原动作并取消剩余option；当前已接受目标cue仍优先。否决消耗原一次介入机会，不延迟重试。控制器只接收公开Observation和原动作建议，不接收真实goal、真距离、源ID、结果标签或未访问图像。无阈值扫描、重训、候选改排序或多规则叠加。','',
        '同原S4全部500题/20已见源文件、C12…16、10×10/B20、冻结权重0/1/2。P新执行1500轨迹；M0与M1各1500已审计记录按哈希引用，无重复运行。源文件名dev/test在本开发对照中均已被查看，不称新测试集。','',
        '预定候选要求：对M0平均SR至少+2点、至少2/3权重正收益、SG不差；对M1平均SR/SG均不差、损伤少于32；完整合法终局及全量复核。仅主数值全部通过才执行4500目标对照，默认不自动切换。','',
        '## 同题主结果','',
        '| 方法 | 平均SR | 平均SG（全部终局） | 权重0/1/2 SR |','|---|---:|---:|---|']
    for arm,label in [('M0','原冻结策略'),('M1','原账本规划'),('P','保护新格动作')]:
        r=a[arm];report.append(f"| {label} | {r['sr_mean']:.2%} | {r['sg_mean']:.3f} | {' / '.join(f'{x:.2%}' for x in r['sr_by_seed'])} |")
    report+=['','| 对照 | SR差（百分点） | SG差（格） | 正权重 | 源文件95% SR差区间（点） |','|---|---:|---:|---:|---|']
    for key,label in [('P_vs_M0','P对M0'),('P_vs_M1','P对M1')]:
        e=s['effects'][key];lo,hi=e['source95_SR'];report.append(f"| {label} | {100*e['sr_gain']:+.2f} | {e['sg_change']:+.3f} | {e['positive_seeds']}/3 | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    report+=['',f"P相对M0恢复{rec['recovered']}条、损伤{rec['harmed']}条，净增加{rec['recovered']-rec['harmed']}/1500。原M1的32条损伤修复{rec['old_harm_repaired']}条；原51条恢复保留{rec['old_recovery_retained']}条、失去{51-rec['old_recovery_retained']}条。修复损伤不等于提高整体SR，完整同题分母与失去的收益均保留。",'',
        '源文件配对bootstrap4000次、seed5251，先在各源内平均三冻结权重，再对20源文件成组采样。不是按1500条独立地图采样，不代表地理独立或跨数据集确认。每项SG差区间保存于主汇总；SG是全部终局曼哈顿距离，成功记0。','',
        '## 冻结验收与边界','']
    for key,val in s['checks'].items():report.append(f"- {key}: {'通过' if val else '未通过'}")
    report+=['',('主数值通过，目标对照另存目标证据汇总。' if verdict['target_controls_started'] else '主数值未通过，按协议不执行条件目标对照、不消费预留源图。原S2/S3/S4与SwissView正式结论保持有效，但不借用它们声称本保护规则已通过新的视觉证据验收。'),'',
        '本规则是一次保守介入假设的检查，不能从本轮否定所有保护机制或所有多Agent方法。当前账本规划仍缺少可区分成功/失败的额外目标证据；不在同队列继续调门槛寻找过线版本。默认继续采用原已验收本地策略，M1仅为低成本备选。后续如启用新云端模型，应先验证账本消息能提供不同且合理的计划证据，再谈同预算SR对照。','',
        f"319项运行前测试通过；独立代码重新枚举候选/触发/新格否决，并从冻结权重重放{audit['records']}条轨迹/{audit['actions']}动作。另核查{audit['input_files_verified']}项输入、{audit['protected_files_verified']}项历史文件；M0/M1已审计日志的哈希与指标重算一致。独立实现的复核由同一执行代理完成，不声称不同人员盲审。",'',
        '[执行协议](执行协议.md) / [预登记](预登记.json) / [全配对诊断](上一轮损伤诊断.json) / [主结果](主对照/对照汇总.json) / [逐题保护与损伤](逐题保护与损伤.json) / [独立复核](独立复核.json) / [验收](验收结论.json)']
    text_new(OUT/'保护规则验证报告.md','\n'.join(report)+'\n')
    text_new(OUT/'README.md','# 账本探索保护规则验证\n\n[结果报告](保护规则验证报告.md) / [验收](验收结论.json) / [复核](独立复核.json) / [冻结设计](执行协议.md) / [逐题](逐题保护与损伤.json)。\n\n旧M0/M1按预登记SHA引用上一轮，P新执行1500记录；已见源文件开发验证，不替换原默认。\n')
    resources=[dict(seed=seed,elapsed_seconds=r['elapsed_seconds'],protection_vetoes=r['protection_vetoes'],
        changed_actions=r['changed_actions'],triggered=r['triggered']) for seed in SEEDS for r in [read(OUT/f'主对照/P_s{seed}_CueFull_结果.json')]]
    write(OUT/'资源汇总.json',dict(jobs=resources,new_training_steps=0,local_experiment_cloud_calls=0,controlled_speed_benchmark=False))
    graphics.style();graphics.ITEMS.clear();fig,axes=plt.subplots(1,2,figsize=(7.6,4.2));labels=['M0 原策略','M1 账本','P 保护规则'];colors=[graphics.GRAY,graphics.BLUE,graphics.ORANGE]
    for panel,(field,scale,ylabel) in enumerate([('sr',100,'SR：成功率（%）'),('sg',1,'SG：平均终点距离（格）')]):
        ax=axes[panel];heights=[]
        for i,arm in enumerate(('M0','M1','P')):
            r=a[arm];v=r[field+'_mean']*scale;seeds=[x*scale for x in r[field+'_by_seed']];height=max([v]+seeds);heights.append(height)
            ax.bar(i,v,width=.58,color=colors[i],edgecolor='#34424F',linewidth=.7,zorder=2)
            for seed,value in enumerate(seeds):ax.scatter(i+(seed-1)*.10,value,marker=['o','^','s'][seed],facecolor='white',edgecolor='#23313B',s=26,lw=.7,zorder=4)
            ax.text(i,height+(2.5 if panel==0 else .065),f'{v:.2f}' if panel==0 else f'{v:.3f}',ha='center',fontsize=10)
        ax.set_xticks(range(3),labels);ax.set_ylabel(ylabel);ax.set_title('(a) 同题成功率' if panel==0 else '(b) 全终局距离',loc='left');ax.grid(axis='y');ax.set_axisbelow(True)
        ax.set_ylim(0,108 if panel==0 else max(heights)*1.27+.1)
        if panel==0:ax.set_yticks([0,20,40,60,80,100])
    fig.subplots_adjust(left=.09,right=.99,bottom=.22,top=.89,wspace=.34)
    fig.text(.5,.08,f"20已见源文件，500题×3冻结权重；P恢复{rec['recovered']}、损伤{rec['harmed']}，M1恢复51、损伤32。",ha='center',fontsize=9)
    fig.text(.5,.025,'M0/M1引用已审计结果，P新执行；白色标记为权重0/1/2，非置信区间。',ha='center',fontsize=9)
    caption=f"公开新格动作保护单因素开发验证：原策略M0 SR{a['M0']['sr_mean']:.2%}/SG{a['M0']['sg_mean']:.3f}，M1 {a['M1']['sr_mean']:.2%}/{a['M1']['sg_mean']:.3f}，P {a['P']['sr_mean']:.2%}/{a['P']['sg_mean']:.3f}。{status}，默认保持。相同500题、20已见源文件、三冻结权重、10×10/B20；P新执行1500轨迹，旧M0/M1按已审计SHA复用。权重标记不是CI，SG为所有终局的曼哈顿距离；不代表独立新图确认。"
    for ext in ('png','pdf','svg'):
        if (FIG/f'数据结果图/{NAME}.{ext}').exists():raise ValueError('immutable figure exists')
    graphics.save(fig,FIG/'数据结果图',NAME,caption,[(OUT/'主对照/对照汇总.json').relative_to(ROOT).as_posix(),(OUT/'独立复核.json').relative_to(ROOT).as_posix()],
        'known-source protection development,500tasks/20sources/3frozen weights')
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/summarize_ledger_protection.py'
    write(FIG/'绘图数据/账本探索保护规则_v1.json',dict(figures=[item],summary_sha256=digest(OUT/'主对照/对照汇总.json'),audit_sha256=digest(OUT/'独立复核.json'),min_font_pt=9,recommended_width_cm=18))
    text_new(FIG/'账本探索保护规则图注.md','# 账本探索保护规则图注\n\n'+caption+'\n')
    print(dict(report=str(OUT/'保护规则验证报告.md'),figure=str(FIG/f'数据结果图/{NAME}.png')),flush=True)


if __name__=='__main__':main()
