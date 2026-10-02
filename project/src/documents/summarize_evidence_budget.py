"""Publish static opportunity and incomplete-interface evidence without new SR claims."""
from pathlib import Path
import sys
import json
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matplotlib import pyplot as plt
from documents import stage_figures as graphics
from eval.evidence_budget_stage_t import ROOT,OUT,read,write,digest,lines

FIG=ROOT/'绘图';NAME='28_证据账本触发机会与接口诊断'


def text_new(p,text):
    with Path(p).open('x',encoding='utf-8') as f:f.write(text)


def main():
    r=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');t=read(OUT/'阶段T/机会汇总.json')
    if digest(OUT/'独立复核.json')!=r['audit_sha256']:raise ValueError('audit before report')
    d=read(OUT/'阶段D来源格式澄清_v3/机制汇总.json');primary=t['per_checkpoint']['0'];requests=list(lines(OUT/'API响应.jsonl'))
    report=['# 证据账本→预算规划：T/D/G分关验证','',
        '**阶段T通过；阶段D接口工程未通过，消息机制诊断未完成；阶段G未放行。** 本批共14/420请求，无训练、无新导航评测、无新SR，原默认及S2/S3/S4/SwissView验收保持。','',
        '## T：公开前缀确实能提供机会','',
        '读取既有S4三份权重各500题/20源文件的轨迹，共1500已保存记录、25476行动前公开前缀。不是新导航执行。主检查固定权重0；另外两份仅作描述，不按最好种子选结果。','',
        '| 已有权重 | 原失败数 | 旧冲突触发覆盖失败 | 新规则覆盖失败 | 新规则覆盖原成功 | 失败来源数 |','|---|---:|---:|---:|---:|---:|']
    for seed,x in t['per_checkpoint'].items():
        report.append(f"| {seed} | {x['original_failures']} | {x['old_failure_trigger_tasks']} | {x['new_failure_trigger_tasks']} | {x['new_success_trigger_tasks']}/{x['original_successes']} | {x['new_failure_source_count']} |")
    report+=['',
        f"权重0的新触发覆盖{primary['new_failure_trigger_tasks']}/{primary['original_failures']}失败和{primary['new_success_trigger_tasks']}/{primary['original_successes']}成功。这是公开停滞/预算检查点，不是失败预测器，更不说明能恢复这些失败。既有真值只用于事后分层，未进入规则或模型。",'',
        'T窗口固定最后2次移动，新格≤1且存在重访；或已花7/20步且此前无接受cue；当前cue接受则保护、剩余至少2步，每题一次。新总体任务按源文件/任务哈希选择20题，C12…16均衡，无成败选题；尚未执行。','',
        '## D：三个工程记录版本及费用边界','',
        '| 版本 | 新聊天请求 | 结果 |','|---|---:|---|',
        '| 原严格schema | 6 | 前两接口事件有效；第6回复geometry:e0被拒绝，停止 |',
        '| 已知类型—事实引用规范化v2 | 7 | 复用旧6请求确认唯一可验证别名；诊断遇到空/数字来源，连续错误关闭 |',
        '| 显式来源目录v3 | 1 | 来源标签符合要求，但counter_evidence为null，工程关卡停止 |','',
        '全部14请求HTTP200，原严格解析8次有效/6次拒绝。其中1个已知几何事实别名可验证修复，余5条结构/来源问题保留：空来源3、数字来源1、null反证列表1。旧回复、失败和每版预登记均保留，未静默改判原日志。', '',
        '所有版本沿用同Gemma4:31b、512输出token、最多4图、同四个状态、同触发/候选/几何评分/消息门槛。额度跨版本累加，engineering实际7≤12、D实际7≤48、G实际0≤360，总14≤420。没有因为额度耗尽而停止，也没有将剩余额度转成额外尝试。', '',
        'v2仅1组M4完整/隐藏消息计划均可执行，二者选择同一个c02；第二重复未完成，其他对照也未完整执行。这不构成稳定消息效应、单Agent胜出或P1无效证据。像素/HTTP接口成功也不代表结构化账本稳定可用。', '',
        '## 归因与下一步','',
        '目前已确认“介入机会”这一前置条件；“可靠账本接口→消息功能→闭环恢复失败”尚未建立。尤其不能把格式错误解释成探索算法退步、数据无效或模型没有视觉能力。没有新SR，不与之前95%或正式阶段成绩相减。', '',
        '本批按冻结停止条件收束；G计划20题×3及M0/M1/M2/M3/M4对照未运行，不把not_run当新失败轨迹。P2未启动，预留SwissView源文件未使用。未来先改小而明确的E接口：程序负责可计算几何项，模型只填写明确来源枚举与少量可验证视觉关系，并以离线畸形响应/真实工程样例验证；另立版本后才再做消息诊断，不通过更改导航门槛补过线。', '',
        '## 验证与文件入口','',
        '288项离线全套测试、3项引用修复与2项目录澄清检查通过。独立实现复核重建14请求的公共状态、图像/消息哈希、角色上下文、严格JSON与候选几何评分，核查全部版本额度及旧失败保留。来源引用只证明可追溯性，视觉事实语义准确性尚未验证；同一执行代理完成另行复核，不声称不同人员审查。','',
        '[执行协议](执行协议.md) / [阶段T统计](阶段T/机会汇总.json) / [T复核](阶段T/独立复核.json) / [T放行](阶段T/放行结论.json) / [四诊断状态](阶段T/诊断状态.json) / [未执行总体任务](总体任务.json)', '',
        '[原D记录](阶段D/机制汇总.json) / [引用修复收据](阶段D格式修复_v2/工程引用修复复核.json) / [v2诊断](阶段D格式修复_v2/机制汇总.json) / [v3诊断](阶段D来源格式澄清_v3/机制汇总.json)', '',
        '[最终验收](验收结论.json) / [独立复核](独立复核.json) / [最终执行状态](最终执行状态.json) / [请求意图](请求意图.jsonl) / [响应](API响应.jsonl)']
    text_new(OUT/'分关验证报告.md','\n'.join(report)+'\n')
    graphics.style();graphics.ITEMS.clear();fig,axes=plt.subplots(1,2,figsize=(7.8,4.6));old=[];new=[]
    for x in t['per_checkpoint'].values():
        old.append(100*x['old_failure_trigger_tasks']/x['original_failures']);new.append(100*x['new_failure_trigger_tasks']/x['original_failures'])
    axes[0].bar([0,1],[old[0],new[0]],color=[graphics.GRAY,graphics.BLUE],width=.62,edgecolor='#34424F',linewidth=.7)
    for values,j in [(old,0),(new,1)]:
        for i,v in enumerate(values):axes[0].scatter(j+(i-1)*.10,v,marker=['o','^','s'][i],facecolor='white',edgecolor='#24323B',s=24,zorder=4)
    axes[0].text(0,old[0]+5,'12/115',ha='center',fontsize=10)
    axes[0].text(1,max(new)+5,'111/115',ha='center',fontsize=10)
    axes[0].set_xticks([0,1],['旧cue冲突','公开预算/停滞']);axes[0].set_ylim(0,112)
    axes[0].set_ylabel('原失败题获得介入机会（%）');axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_title('(a) T：机会覆盖，非新SR',loc='left')
    axes[1].bar([0,1],[8,6],color=[graphics.TEAL,graphics.ORANGE],width=.62,edgecolor='#34424F',linewidth=.7)
    for i,value in enumerate((8,6)):axes[1].text(i,value+.3,str(value),ha='center',fontsize=10)
    axes[1].set_xticks([0,1],['原schema有效','原schema拒绝']);axes[1].set_ylim(0,10);axes[1].set_yticks([0,2,4,6,8,10])
    axes[1].set_ylabel('实际聊天请求数');axes[1].set_title('(b) D：14请求的原解析结果',loc='left')
    for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True)
    fig.subplots_adjust(left=.1,right=.98,bottom=.22,top=.86,wspace=.42)
    fig.text(.5,.085,'T柱：固定权重0；白点：已有三权重描述，非新模型执行或CI。',ha='center',fontsize=9)
    fig.text(.5,.035,'D拒绝中1条已知别名可验证修复；其余5条保留。G未运行，无新SR。',ha='center',fontsize=9)
    caption='阶段T静态读取1500条既有S4记录、25476公开前缀；权重0原失败115题中旧冲突12、新公开预算/停滞111获得介入机会，不代表恢复成功。原成功385题中也有372触发，不能当失败预测。三白点是已有训练权重的描述，不增加源图或形成CI。阶段D跨格式版本14实际请求，原解析8有效/6拒绝，其中1已知类型事实别名复核修复；其余5失败保留。工程与消息关卡未通过，闭环G未启动，无新SR、无新训练或新地图。'
    for ext in ('png','pdf','svg'):
        if (FIG/f'数据结果图/{NAME}.{ext}').exists():raise ValueError('figure exists')
    graphics.save(fig,FIG/'数据结果图',NAME,caption,[(OUT/'阶段T/机会汇总.json').relative_to(ROOT).as_posix(),(OUT/'独立复核.json').relative_to(ROOT).as_posix()],
        'static known S4 prefix check, incomplete D interface diagnosis')
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/summarize_evidence_budget.py'
    write(FIG/'绘图数据/证据账本分关验证_v1.json',dict(figures=[item],T_sha256=digest(OUT/'阶段T/机会汇总.json'),
        audit_sha256=digest(OUT/'独立复核.json'),new_SR_available=False,min_font_pt=9,recommended_width_cm=18))
    text_new(FIG/'证据账本分关验证图注.md','# 证据账本分关验证\n\n'+caption+'\n')
    p=ROOT/'README.md';old=p.read_text(encoding='utf-8');old=old.replace('## 进行中：证据账本与预算规划分关验证','## 已收尾：证据账本与预算规划分关验证')
    heading='## 已收尾：证据账本与预算规划分关验证（2026-10-01）'
    entry=f'''\n\n本批T通过：权重0原115失败中111获得机会（旧12），但372/385原成功也触发，不是失败预测。D工程未通过，跨版本14/420请求均HTTP成功，原schema8有效/6拒绝，1已知引用别名另行复核修复；消息机制未完成验证，G未放行，无新SR、无训练或新源图。原默认与既有验收保持。

[分关报告]({OUT.relative_to(ROOT).as_posix()}/分关验证报告.md) / [最终验收]({OUT.relative_to(ROOT).as_posix()}/验收结论.json) / [复核]({OUT.relative_to(ROOT).as_posix()}/独立复核.json) / [科研图](绘图/数据结果图/{NAME}.pdf)。所有格式失败和预登记版本保留；额度跨版本累加，未以模型调用填满剩余预算。
'''
    p.write_text(old.replace(heading,heading+entry,1),encoding='utf-8')
    fp=FIG/'README.md';old=fp.read_text(encoding='utf-8');pos=old.index('\n## ')
    fp.write_text(old[:pos]+f'''\n## 新增：证据账本T/D分关验证

[PNG](数据结果图/{NAME}.png) / [PDF](数据结果图/{NAME}.pdf) / [SVG](数据结果图/{NAME}.svg) / [图注](证据账本分关验证图注.md) / [数据来源](绘图数据/证据账本分关验证_v1.json)。T为静态机会检查，D为工程接口诊断，G未运行；不把触发率当成功率。
'''+old[pos:],encoding='utf-8')
    print(dict(report=True,figure=(FIG/f'数据结果图/{NAME}.png').as_posix(),G_started=False))


if __name__=='__main__':main()
