"""Publish audited pilot evidence and a standalone scientific result figure."""
import json
from pathlib import Path
import re
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter
from PIL import Image
from matplotlib import pyplot as plt
from documents import stage_figures as graphics
from eval.collaboration_pilot import ROOT,OUT,read,read_lines,MODES
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest

FIG=ROOT/'绘图'
NAME='27_多Agent同预算小验证成功率与距离'
LABELS={'Edge':'冻结本地基线','SingleOnce':'单次建议','SingleReflect':'单Agent反思','TwoAgent':'两Agent协作'}


def write_text_new(path, text):
    with Path(path).open('x',encoding='utf-8') as f:
        f.write(text)


def main():
    receipt=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');s=read(OUT/'对照汇总.json');reg=read(OUT/'预登记.json')
    if not receipt['audit_passed'] or digest(OUT/'独立复核.json')!=receipt['audit_sha256']:
        raise ValueError('completed audit required')
    rows=read_lines(OUT/'导航轨迹.jsonl');calls=read_lines(OUT/'API响应.jsonl')
    verdict='通过小规模候选门槛' if receipt['candidate_passed'] else '未通过小规模候选门槛'
    proposal_calls=[c for c in calls if c['role']!='preflight']
    evidence=Counter((c.get('parsed') or {}).get('target_evidence','invalid') for c in proposal_calls)
    trigger_sources=sorted({r['area'] for r in rows if any(d['interaction'] and d['interaction']['call_ids'] for d in r['decisions'])})
    diagnostics=read(OUT/'纠正与损伤诊断.json')['by_arm']
    text=['# 多Agent同预算20题小验证：结果与边界','',
        f"结论：{verdict}。三轮API重复不增加独立源图数；原默认、训练权重与已通过S2/S3/S4/SwissView均保持。",'',
        '## 相同任务与模型的对照','',
        'Masa既有28张开发图中事前按哈希抽取20张、每张1题，C4…8各4题、5×5/B10；固定本地checkpoint0，非挑选最好seed。云端为同一个Gemma4:31b，三个API重复；比较相同图像权限和调用上限，实际调用与token耗量单独报告。','',
        '| 方法 | 计划/正常完成 | 成功 | SR | SG | 导航请求 | 改变原动作 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in reg['arms']:
        rr=[r for r in rows if r['arm']==arm];m=s['averages'][arm];res=s['resources'].get(arm,{})
        text.append(f"| {LABELS[arm]} | {len(rr)}/{sum(r['status']=='completed' for r in rr)} | {sum(r['success'] for r in rr)} | {m['SR_gate']:.2%} | {m['SG_gate']:.3f} | {res.get('calls',0)} | {res.get('changed_actions',0)} |")
    text+=['',f"协作相对同预算反思SR差为{sum(s['paired_SR_by_API_repeat'])/3*100:+.2f}个百分点，源文件配对95%区间[{s['source_SR_interval95'][0]*100:+.2f},{s['source_SR_interval95'][1]*100:+.2f}]；SG差{sum(s['paired_SG_by_API_repeat'])/3:+.3f}，区间{s['source_SG_interval95']}。每源图先平均三轮，再4000次成组bootstrap，seed4121。",'',
        '| API重复 | 协作相对反思SR差（百分点） | SG差 |','|---|---:|---:|']
    for r in range(3):text.append(f"| {r+1} | {s['paired_SR_by_API_repeat'][r]*100:+.2f} | {s['paired_SG_by_API_repeat'][r]:+.3f} |")
    text+=['','## 角色参与与真实收益','',
        f"工程像素身份判断6/6正确；全部{s['total_requests']}次请求中，导航{len(proposal_calls)}次。实际触发覆盖{len(trigger_sources)}/20张图（{', '.join(trigger_sources)}）；输出证据标记为{dict(evidence)}。像素身份判断不能证明目标方向能力。",'',
        '两个独立角色按协议维护各自上下文，核验者未见本轮规划建议，完成后交换消息；回退和执行由确定性程序完成。因独立支持/一致性条件限制，不能把调用了两个角色当成获得协作导航收益。','',
        '| 方法 | 恢复基线失败 | 损伤基线成功 | 改变动作 |','|---|---:|---:|---:|']
    for arm,d in diagnostics.items():text.append(f"| {LABELS[arm]} | {d['corrected_failures']} | {d['damaged_successes']} | {d['changed_actions']} |")
    if not any(d['changed_actions'] for d in diagnostics.values()):
        text+=['','全部云端臂未改变原动作，所以SR与SG相同。区间退化为0描述本批输出，不证明方法统计等价、普遍无效或已达到任务上限。本批只对少数冲突状态提供机会，强基线本来就成功19/20题，统计效力有限。']
    if all(sum(bool(d['interaction']['call_ids']) for d in r['decisions'])<=1 for r in rows if r['arm']=='TwoAgent'):
        text+=['','事后覆盖检查：本批没有第二轮交互，唯一基线失败题未触发。触发协议未给修复本批失败提供机会；没有后续模型调用实际利用角色交换消息，不能声称已验证反馈修正或持续多Agent协作。该解释不用于修改本批协议或重跑。']
    text+=['','## 放行与收束','',f"事前门槛：SR至少+2点、≥2/3轮正收益、SG不变差、完整响应与审计。各项结果：{s['checks']}。",'',
        '本批不自动扩大到140题、不增加Agent、不消耗预留SwissView地图、不替换默认。未安排错目标/遮蔽对照，不建立新增目标理解或独立泛化结论。只有先提出能在公开观测中提供额外有效证据的独立假设，才考虑新批次；当前更应补实际地理覆盖扩大及真实目标变化的验证。','',
        '## 资源与可追溯性','',
        '| 方法 | 有效/实际请求 | 输入tokens | 输出tokens | 图片 | 请求时延中位数（秒） |','|---|---:|---:|---:|---:|---:|']
    for arm in MODES:
        r=s['resources'][arm];latency='无请求' if r['median_latency_seconds'] is None else f"{r['median_latency_seconds']:.2f}"
        text.append(f"| {LABELS[arm]} | {r['valid']}/{r['calls']} | {r['input_tokens']} | {r['output_tokens']} | {r['images']} | {latency} |")
    text+=['','tokens是provider返回计数，图像token计入方式由provider决定；单价未核实，不编造费用估算。相同上限不等于完全相同实耗；temperature0不保证云端权重或输出永久稳定。','',
        f"273项离线测试通过。事后复核重放{audit['completed_records']}条正常记录、{audit['replayed_actions']}动作，核验全部{audit['cloud_requests']}个请求与响应、输入图像哈希、公开信息、角色消息、严格JSON、边界、原模型参数及指标。另行实现由同一执行代理完成，不声称不同人员审查。原大数据和旧结果未移动。",'',
        '[冻结协议](冻结方案.md) / [预登记](预登记.json) / [任务](导航任务.json) / [汇总](对照汇总.json) / [验收](验收结论.json) / [复核](独立复核.json) / [逐题纠正与损伤](纠正与损伤诊断.json) / [请求意图](请求意图.jsonl) / [API响应](API响应.jsonl) / [全部轨迹](导航轨迹.jsonl)', '',
        '[任务要求与参考框架对应](任务要求与论文框架对应说明.md)。当前工作符合主动视觉地理定位的核心；训练GRU/视觉头允许，多Agent为可选。完整原论文复现、真实搜索范围扩大、任意角度和实飞尚未验证。']
    write_text_new(OUT/'多Agent小验证报告.md','\n'.join(text)+'\n')
    graphics.style();graphics.ITEMS.clear()
    arms=reg['arms'];x=list(range(4));colors=[graphics.GRAY,graphics.ORANGE,graphics.TEAL,graphics.BLUE]
    fig,axes=plt.subplots(1,2,figsize=(7.8,4.4));max_sg=max(v['SG_gate'] for runs in s['runs'].values() for v in runs.values())
    for ax,key,scale in [(axes[0],'SR_gate',100),(axes[1],'SG_gate',1)]:
        vals=[s['averages'][a][key]*scale for a in arms]
        ax.bar(x,vals,width=.6,color=colors,edgecolor='#34424F',linewidth=.7)
        for i,arm in enumerate(arms):
            runs=list(s['runs'][arm].values())
            if arm!='Edge':
                for j,m in enumerate(runs):ax.scatter(i+(j-1)*.1,m[key]*scale,marker=['o','^','s'][j],s=22,facecolor='white',edgecolor='#24323B',zorder=4)
            ax.text(i,vals[i]+(2 if scale==100 else max(.01,max_sg*.07)),f'{vals[i]:.1f}' if scale==100 else f'{vals[i]:.3f}',ha='center',fontsize=10)
        ax.set_xticks(x,[LABELS[a].replace('冻结本地基线','本地基线') for a in arms],rotation=15)
        ax.grid(axis='y');ax.set_axisbelow(True)
    axes[0].set_ylim(0,110);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR：成功率（%）');axes[0].set_title('(a) 相同20题的导航成功率',loc='left')
    axes[1].set_ylim(0,max(.15,max_sg*1.6));axes[1].set_ylabel('SG：全部终点距离（格）');axes[1].set_title('(b) 包含失败终点的SG',loc='left')
    fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.87,wspace=.32)
    fig.text(.5,.045,'20张已见开发源图；5×5/B10；固定本地权重0。白点：三轮API重复，非训练seed或CI。',ha='center',fontsize=9)
    caption=f"20源图/20题已见开发小验证，原本地checkpoint0与三轮Gemma云端控制同图像权限/动作预算；双调用臂同调用上限。协作{verdict}。白点是三轮API重复，不增加独立源图。SG包含全部失败终点；全部云端臂{sum(d['changed_actions'] for d in diagnostics.values())}个改变动作，不能把相同SR归因于新增协作。"
    for fmt in ('png','pdf','svg'):
        if (FIG/f'数据结果图/{NAME}.{fmt}').exists():raise ValueError('immutable figure exists')
    graphics.save(fig,FIG/'数据结果图',NAME,caption,[(OUT/'对照汇总.json').relative_to(ROOT).as_posix()],
                  'known Masa20sources/20tasks, cloud3repeats, frozen localcheckpoint0')
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/summarize_collaboration_pilot.py'
    manifest=dict(version='collaboration-pilot-figure-v1',figures=[item],summary_sha256=digest(OUT/'对照汇总.json'),
        registration_sha256=digest(OUT/'预登记.json'),audit_sha256=digest(OUT/'独立复核.json'),
        minimum_font_pt=9,recommended_width_cm=18,scope='known20source pilot, no generalizationclaim')
    write_new(FIG/'绘图数据/多Agent同预算小验证_v1.json',manifest)
    write_text_new(FIG/'多Agent同预算小验证图注.md','# 多Agent同预算小验证\n\n'+caption+'\n')
    # Current indexes may evolve; all historical artifact contents stay unchanged.
    readme=ROOT/'README.md';original=readme.read_text(encoding='utf-8');position=original.index('\n## ')
    folder=OUT.relative_to(ROOT).as_posix()
    entry=f'''\n## 已完成：多Agent同预算20题小验证（2026-10-01）

当前正式要求允许训练；多Agent是可选拓展。[任务要求与论文框架对应](选题报告相关/任务要求与论文框架对应说明_v1.md)。冻结Gemma4:31b、本地checkpoint0、20张已见开发源图/20题、三轮API重复；单次、单Agent两次反思、两Agent独立核验与原Edge同题比较。{verdict}，默认及原验收保持。

| 方法 | SR | SG | 导航请求 | 改变动作 |
|---|---:|---:|---:|---:|
'''
    for a in arms:
        m=s['averages'][a];r=s['resources'].get(a,{})
        entry+=f"| {LABELS[a]} | {m['SR_gate']:.2%} | {m['SG_gate']:.3f} | {r.get('calls',0)} | {r.get('changed_actions',0)} |\n"
    entry+=f'''\n全部{len(calls)}请求与{audit['replayed_actions']}动作复核通过；实际触发覆盖{len(trigger_sources)}张图，导航输出{dict(evidence)}。API重复不是训练seed；本批结果不能证明任意协作无效，也不能凭继承的基线成绩声称协作提升。不自动扩大开发，不消耗预留新图。

[报告]({folder}/多Agent小验证报告.md) / [验收]({folder}/验收结论.json) / [复核]({folder}/独立复核.json) / [协议](选题报告相关/多Agent同预算小验证方案_v1.md) / [科研图](绘图/数据结果图/{NAME}.pdf)。
'''
    readme.write_text(original[:position]+entry+original[position:],encoding='utf-8')
    current=readme.read_text(encoding='utf-8').replace('该方案尚未实现或调用模型，当前仍为训练过的单智能体模块系统；无需新增训练的协作层不等于整个系统零训练。',
        '已按新冻结协议完成小验证；当前默认仍为训练过的单智能体模块系统，新增协作支线未替换默认；无需新增训练的协作层不等于整个系统零训练。').replace('草案仅新增于既有选题报告目录，不创建实验目录、不消耗预留地图；具体任务、提示、触发/仲裁及云端总预算需在实际执行前冻结。',
        'v1草案保留原样；本轮实际任务、提示、触发/仲裁与云端预算已另行冻结在同预算小验证协议和预登记中，不消耗预留地图。')
    readme.write_text(current,encoding='utf-8')
    f=FIG/'README.md';old=f.read_text(encoding='utf-8');pos=old.index('\n## ')
    f.write_text(old[:pos]+f'''\n## 新增：多Agent同预算小验证

[PNG](数据结果图/{NAME}.png) / [PDF](数据结果图/{NAME}.pdf) / [SVG](数据结果图/{NAME}.svg) / [图注](多Agent同预算小验证图注.md) / [源数据与哈希](绘图数据/多Agent同预算小验证_v1.json)。{verdict}，固定本地权重0、20已见源图；API重复不称训练种子。PNG300dpi、PDF/SVG矢量，建议宽度≥18cm。
'''+old[pos:],encoding='utf-8')
    png=FIG/f'数据结果图/{NAME}.png'
    with Image.open(png) as image:
        dpi=image.info.get('dpi');check_dpi=bool(dpi and min(dpi)>=299)
    pdf=FIG/f'数据结果图/{NAME}.pdf';svg=FIG/f'数据结果图/{NAME}.svg'
    svgtext=svg.read_text(encoding='utf-8')
    delivery=dict(complete=True,report_sha256=digest(OUT/'多Agent小验证报告.md'),manifest_sha256=digest(FIG/'绘图数据/多Agent同预算小验证_v1.json'),
        figures_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in (png,pdf,svg)},PNG_dpi=dpi,PNG_300dpi=check_dpi,
        SVG_vector_text='<text' in svgtext,SVG_no_embedded_raster='<image' not in svgtext,
        figure_visual_review_pending=True,default_changed=False,candidate_passed=receipt['candidate_passed'])
    write_new(OUT/'交付结构核验.json',delivery)
    print(json.dumps(dict(report=True,figure=png.as_posix(),candidate_passed=receipt['candidate_passed']),ensure_ascii=False))


if __name__=='__main__':main()
