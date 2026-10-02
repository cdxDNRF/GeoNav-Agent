"""Publish audited Continue5 evidence and fresh-source confirmation without changing runs."""
import argparse,json,hashlib,re,sys
from pathlib import Path
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics
from matplotlib import pyplot as plt
from PIL import Image
import fitz

ROOT=Path(__file__).resolve().parents[3];FIG=ROOT/'绘图'
EVIDENCE=ROOT/'DATA/processed_data/Masa/评测结果/继续训练冻结目标证据复验_v1'
CONFIRM=ROOT/'DATA/processed_data/SwissView/评测结果/继续训练独立源图确认_v1'


def read(p):return json.loads(p.read_text('utf8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def text_new(p,s):
    with p.open('x',encoding='utf8') as f:f.write(s)
def write(p,x):text_new(p,json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


def audit_required(out,summary):
    v=read(out/'验收结论.json');a=read(out/'独立复核.json')
    if not a['passed'] or not v['audit_passed'] or v['audit_sha256']!=digest(out/'独立复核.json') or v['summary_sha256']!=digest(out/summary):raise ValueError('complete audited receipt required')
    return v,a


def save(fig,name,caption,refs,label):
    for ext in ('png','pdf','svg'):
        if (FIG/f'数据结果图/{name}.{ext}').exists():raise ValueError('immutable figure exists')
    graphics.ITEMS.clear();graphics.save(fig,FIG/'数据结果图',name,caption,refs,label)
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/summarize_continued_validation.py'
    manifest=FIG/f'绘图数据/{label}_v1.json'
    write(manifest,dict(figures=[item],min_font_pt=9,recommended_width_cm=18,source_sha256={n:digest(ROOT/n) for n in refs},script_sha256=digest(Path(__file__))))
    text_new(FIG/f'{label}图注.md','# '+label+'图注\n\n'+caption+'\n')
    catalog=read(FIG/'图表来源清单.json')
    if name in {x['id'] for x in catalog['figures']}:raise ValueError('duplicate catalog entry')
    catalog['figures'].append(item)
    (FIG/'图表来源清单.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n','utf8')
    return item


def evidence():
    out=EVIDENCE;v,a=audit_required(out,'目标证据汇总.json');s=read(out/'目标证据汇总.json');reg=read(out/'预登记.json');r=s['arms']
    lines=['# Continue5冻结目标证据复验','',f"**目标证据{'通过' if v['target_evidence_passed'] else '未通过'}；默认保持。** 本批没有训练、云端调用或新评测源文件。已见500任务/20源文件、10×10/B20、三份Continue5最终权重。",'',
        '真实目标1500条按SHA复用上轮审计轨迹，另运行禁用cue、均值目标、错目标各1500条。四条件全部6000条在本轮重放。均值同时遮蔽目标全局、局部和边缘通道；错目标替换三个目标通道，环境目标保持真实。','',
        '| 条件 | 平均SR | 全终局SG | seed0/1/2 SR |','|---|---:|---:|---|']
    labels={'CueFull':'真实目标','Baseline':'禁用视觉线索','CueMean':'均值目标','CueWrong':'错误目标'}
    for c in labels:
        z=r[c];lines.append(f"| {labels[c]} | {z['sr_mean']:.2%} | {z['sg_mean']:.3f} | {' / '.join(f'{x:.2%}' for x in z['sr_by_seed'])} |")
    lines+=['','| 真实目标相对 | SR差（点） | SG差 | 正权重 | 20源文件95% SR差区间（点） | 门槛 |','|---|---:|---:|---:|---|---|']
    for c in ('Baseline','CueMean','CueWrong'):
        e=s['effects'][c];lo,hi=e['source95_SR'];lines.append(f"| {labels[c]} | {100*e['sr_gain']:+.2f} | {e['sg_change']:+.3f} | {e['positive_seeds']}/3 | [{100*lo:+.2f}, {100*hi:+.2f}] | {'通过' if all(s['checks'][c].values()) else '未通过'} |")
    e=s['Continue5_vs_M0'];lo,hi=e['source95_SR'];rec=s['recovery_vs_M0']
    lines+=['',f"对原M0恢复{rec['recovered']}条、损伤{rec['harmed']}条，净{rec['recovered']-rec['harmed']:+d}/1500；SR{100*e['sr_gain']:+.2f}点、SG{e['sg_change']:+.3f}格，源文件95% SR差区间[{100*lo:+.2f}, {100*hi:+.2f}]点。区间仍含零：本批证明该队列上目标图有用，不证明继续训练已稳定优于原默认。",'',
        f"固定错误目标计划共{len(reg['wrong_distance_exceptions'])}题无法保持初始距离一致，例外在预登记完整保存；三权重复用同计划，不按结果改错图。配对bootstrap按20源文件重采样4000次/seed5251，先平均源内三权重，不把1500记录称1500独立地图。",'',
        f"全量复核重放{a['records']}轨迹/{a['actions']}动作，其中新增控制{a['new_control_actions']}动作；核查每次模型、两图通道、GRU、合法动作、首次到达终局及SR/SG/四项源文件差区间。{a['protected_files_verified']}历史文件、{a['input_files_verified']}图像/特征输入SHA保持。5项新门槛/队列回归测试通过；审计独立指另行实现重算，由同一执行代理完成。",'',
        '通过本批后，允许按另立冻结协议开展新源文件确认；未以旧开发图替代确认，也未自动切换默认。当前仍为训练过的单导航智能体模块系统。', '',
        '[执行协议](执行协议.md) / [预登记](预登记.json) / [目标汇总](目标证据汇总.json) / [逐题恢复损伤](继续训练对原基线_逐题恢复与损伤.json) / [复核](独立复核.json) / [验收](验收结论.json)']
    text_new(out/'冻结目标证据复验报告.md','\n'.join(lines)+'\n');text_new(out/'README.md','# Continue5冻结目标证据\n\n[报告](冻结目标证据复验报告.md) / [验收](验收结论.json) / [复核](独立复核.json) / [协议](执行协议.md)。\n\n4500新控制、1500按SHA复用真实目标；全部6000重放，原默认保持。\n')
    graphics.style();fig,axes=plt.subplots(1,2,figsize=(8.2,4.2));cs=list(labels);colors=[graphics.BLUE,graphics.GRAY,graphics.ORANGE,graphics.TEAL]
    for j,(field,scale,ylabel) in enumerate([('sr',100,'SR：成功率（%）'),('sg',1,'SG：平均终点距离（格）')]):
        ax=axes[j];vals=[r[c][field+'_mean']*scale for c in cs];weights=[[x*scale for x in r[c][field+'_by_seed']] for c in cs]
        graphics.bars(ax,[labels[c] for c in cs],vals,colors,decimals=2 if j==0 else 3,seeds=weights,percent=j==0)
        ax.set_ylabel(ylabel);ax.set_title('(a) 成功率与目标依赖' if j==0 else '(b) 所有终局距离',loc='left');ax.grid(axis='y');ax.set_axisbelow(True)
        ax.set_ylim(0,105 if j==0 else max(max(z) for z in weights)*1.2+.1)
        if j==0:ax.set_yticks([0,20,40,60,80,100])
    fig.subplots_adjust(left=.085,right=.99,bottom=.23,top=.89,wspace=.32)
    fig.text(.5,.085,'20已见源文件；每条件500题×3最终权重，10×10/B20；未进行新增训练。',ha='center',fontsize=9)
    fig.text(.5,.03,'白色标记为seed0/1/2，非置信区间；真实目标轨迹按SHA复用并重新审计。',ha='center',fontsize=9)
    caption=f"Continue5冻结目标证据：真实目标SR{r['CueFull']['sr_mean']:.2%}/SG{r['CueFull']['sg_mean']:.3f}，禁用cue{r['Baseline']['sr_mean']:.2%}/{r['Baseline']['sg_mean']:.3f}、均值目标{r['CueMean']['sr_mean']:.2%}/{r['CueMean']['sg_mean']:.3f}、错目标{r['CueWrong']['sr_mean']:.2%}/{r['CueWrong']['sg_mean']:.3f}。每条件500题×三最终权重、20已见源文件、10×10/B20；均值/错图作用于全局、局部、边缘全部目标通道，禁用cue保留同一探索器。SR主指标，SG为所有终局曼哈顿距离。白色点是训练权重，不是CI。目标证据通过不等于独立确认或默认替换。"
    item=save(fig,'34_继续训练冻结目标证据',caption,[(out/'目标证据汇总.json').relative_to(ROOT).as_posix(),(out/'独立复核.json').relative_to(ROOT).as_posix()],'继续训练冻结目标证据')
    print(dict(report=str(out/'冻结目标证据复验报告.md'),figure=str(FIG/f"数据结果图/{item['id']}.png")),flush=True)


def confirmation():
    out=CONFIRM;v,a=audit_required(out,'主对照汇总.json');s=read(out/'主对照汇总.json');r=s['averages']
    title='独立候选确认通过' if v['independent_candidate_confirmed'] else '独立候选未通过升级条件，收束新增训练支线'
    report=['# Continue5独立源文件确认','',f'**{title}。** 原默认与既有验收保持。预留SwissView40…59号20源文件本轮已消费，后续不得再次作为未见确认。','',
        '源名单继承早已登记的预留40…59号列表，并在运行前核查既有全部导航轨迹实际源文件、原文件/RGB身份及元数据SHA。新20源文件与已见28源文件的文件与RGB哈希不重叠，两协议共用同20源文件。仅证明源文件隔离，不声称严格地理无重叠或冻结encoder预训练未见。','',
        '固定原M0与Continue5三权重/目标头/均值/0.50阈值/合法动作规则，零新增训练、零云端。5×5/B10原生300图块，10×10/B20原生150图块经BICUBIC放大；10×10只提高同足迹切图密度。每协议500题，每源五距离档各5题。','',
        '| 新源协议 | 原M0 SR / SG | Continue5 SR / SG | SR差（点） | SG差（格） | 正权重 | 源文件95% SR差区间（点） |','|---|---:|---:|---:|---:|---:|---|']
    for k in ('5','10'):
        old,new=r[k]['M0'],r[k]['Continue5'];e=s['effects'][k];lo,hi=e['source95_SR']
        report.append(f"| {k}×{k} | {old['sr_mean']:.2%} / {old['sg_mean']:.3f} | {new['sr_mean']:.2%} / {new['sg_mean']:.3f} | {100*e['sr_gain']:+.2f} | {e['sg_change']:+.3f} | {e['positive_seeds']}/3 | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    report+=['','## 完整配对与预定判定','']
    for k in ('5','10'):
        q=s['recovery_by_protocol'][k];report.append(f"{k}×{k}恢复{q['recovered']}、损伤{q['harmed']}，净{q['recovered']-q['harmed']:+d}/1500。Continue5三权重SR：{' / '.join(f'{x:.2%}' for x in r[k]['Continue5']['sr_by_seed'])}；原M0：{' / '.join(f'{x:.2%}' for x in r[k]['M0']['sr_by_seed'])}。")
    report+=['']+[f"- {key}: {'通过' if value else '未通过'}" for key,value in s['checks'].items()]
    if v['target_controls_started']:
        target=read(out/'目标证据汇总.json');report+=['','## 新10×10目标对照','','| 控制 | SR | SG | Full SR差（点） | 正权重 | 通过 |','|---|---:|---:|---:|---:|---|']
        for c in ('Baseline','CueMean','CueWrong'):
            z=target['arms'][c];e=target['effects'][c];report.append(f"| {c} | {z['sr_mean']:.2%} | {z['sg_mean']:.3f} | {100*e['sr_gain']:+.2f} | {e['positive_seeds']}/3 | {target['checks'][c]} |")
    else:report+=['','按冻结门槛停止，不追加4500新目标对照，不调参数或更换源名单追求通过。已见源文件的目标证据仍成立；独立源文件上的升级优势按本批结果判定。']
    report+=['',f"全量复核：{a['records']}轨迹/{a['actions']}动作；2500派生JPEG从原图重建一致，global/local重新提取精确一致，边缘profile另行实现计算一致。全部3000条5×5记录与原包装器逐步完全一致；检查手工几何/首次到达、全体SR/SG/目标三通道遮蔽与配对源文件区间。{a['protected_files_verified']}历史文件和{a['derived_input_files_verified']}派生输入文件SHA保持。新5项门槛/双网格队列测试通过。",'',
        '独立源文件bootstrap沿用seed5251/4000次，源内先平均三权重。5×5容忍最多2点平均下降只是一项保护规则，不是统计学非劣效证据；报告区间不删去零或负值。任务和权重重复不当独立源文件。','',
        '当前仍为训练过的单导航智能体模块系统；真实多Agent协作、任意角度、异时与无人机实飞未由本批验证。原S2/S3/S4及原SwissView迁移不回写。','',
        '[执行协议](执行协议.md) / [预登记](预登记.json) / [源使用核查](源图使用核查.json) / [主汇总](主对照汇总.json) / [逐题恢复损伤](逐题恢复与损伤.json) / [复核](独立复核.json) / [验收](验收结论.json)']
    text_new(out/'独立源图确认报告.md','\n'.join(report)+'\n');text_new(out/'README.md','# Continue5独立源文件确认\n\n[报告](独立源图确认报告.md) / [验收](验收结论.json) / [复核](独立复核.json) / [源图核查](源图使用核查.json)。\n\n新20源文件已消费，三最终权重、两个网格；原默认保持。\n')
    graphics.style();fig,axes=plt.subplots(1,2,figsize=(8.2,4.2));colors=[graphics.GRAY,graphics.BLUE];xs=[-.17,.17]
    for j,(field,scale,ylabel) in enumerate([('sr',100,'SR：成功率（%）'),('sg',1,'SG：平均终点距离（格）')]):
        ax=axes[j];maxv=0
        for i,k in enumerate(('5','10')):
            for arm,c,off in zip(('M0','Continue5'),colors,xs):
                z=r[k][arm];v=z[field+'_mean']*scale;ws=[x*scale for x in z[field+'_by_seed']];maxv=max(maxv,max(ws));x=i+off
                ax.bar(x,v,width=.3,color=c,edgecolor='#34424F',lw=.7,label='原M0' if arm=='M0' else 'Continue5',zorder=2)
                for n,w in enumerate(ws):ax.scatter(x+(n-1)*.045,w,marker=['o','^','s'][n],facecolor='white',edgecolor='#23313B',s=24,lw=.7,zorder=4)
                ax.text(x,max(ws)+(.9 if j==0 else .025),f'{v:.2f}' if j==0 else f'{v:.3f}',ha='center',fontsize=9.3)
        ax.set_xticks([0,1],['5×5 / B10','10×10 / B20']);ax.set_ylabel(ylabel);ax.set_title('(a) 新源图成功率' if j==0 else '(b) 所有终局距离',loc='left');ax.grid(axis='y');ax.set_axisbelow(True)
        ax.set_ylim(0,109 if j==0 else maxv*1.25+.1)
        if j==0:ax.set_yticks([0,20,40,60,80,100]);ax.legend([ax.containers[0],ax.containers[1]],['原M0','Continue5'],loc='lower left',frameon=False)
    fig.subplots_adjust(left=.085,right=.99,bottom=.23,top=.89,wspace=.32)
    fig.text(.5,.085,'20新SwissView源文件，两协议共用；每方法/网格500任务×3最终权重。',ha='center',fontsize=9)
    fig.text(.5,.03,'白色标记为seed0/1/2，非置信区间；10×10提高同足迹密度，非更大实际面积。',ha='center',fontsize=9)
    caption='Continue5固定20新SwissView源文件确认。'+ '; '.join(f"{k}×{k}：原M0 SR{r[k]['M0']['sr_mean']:.2%}/SG{r[k]['M0']['sg_mean']:.3f}，Continue5 {r[k]['Continue5']['sr_mean']:.2%}/{r[k]['Continue5']['sg_mean']:.3f}" for k in ('5','10'))+f'。{title}。两协议共用同20源文件、每方法/网格500任务×三最终权重；白色点为训练权重而非CI。SG为所有终局曼哈顿距离；源文件隔离不代表地理足迹无重叠，10×10不是实际面积扩大，原默认保持。'
    item=save(fig,'35_继续训练独立源图双网格确认',caption,[(out/'主对照汇总.json').relative_to(ROOT).as_posix(),(out/'独立复核.json').relative_to(ROOT).as_posix()],'继续训练独立源图确认')
    print(dict(report=str(out/'独立源图确认报告.md'),figure=str(FIG/f"数据结果图/{item['id']}.png")),flush=True)


def delivery(which):
    out=EVIDENCE if which=='evidence' else CONFIRM;label='继续训练冻结目标证据' if which=='evidence' else '继续训练独立源图确认';report='冻结目标证据复验报告.md' if which=='evidence' else '独立源图确认报告.md'
    name='34_继续训练冻结目标证据' if which=='evidence' else '35_继续训练独立源图双网格确认';summary='目标证据汇总.json' if which=='evidence' else '主对照汇总.json'
    v,a=audit_required(out,summary);manifest=FIG/f'绘图数据/{label}_v1.json';item=read(manifest)['figures'][0];review=FIG/f'审阅/{label}设计与核验_v1.md'
    if not review.is_file():raise ValueError('actual PNG review required before delivery')
    for p in item['files']:
        if digest(ROOT/p['path'])!=p['sha256']:raise ValueError('media drift')
    with Image.open(FIG/f'数据结果图/{name}.png') as im:
        size=im.size;dpi=im.info.get('dpi');assert min(dpi)>299
    tags=[e.tag.rsplit('}',1)[-1] for e in ET.parse(FIG/f'数据结果图/{name}.svg').getroot().iter()]
    assert 'text' in tags and 'image' not in tags
    with fitz.open(FIG/f'数据结果图/{name}.pdf') as pdf:
        fonts=pdf[0].get_fonts(full=True);assert len(pdf)==1 and not pdf[0].get_images() and fonts and all(pdf.extract_font(z[0])[-1] for z in fonts)
    links=0
    for p in (out/report,out/'README.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text('utf8')):
            if not (p.parent/target).is_file():raise ValueError('broken link '+target)
            links+=1
    write(out/'交付核验.json',dict(passed=True,report_sha256=digest(out/report),PNG_size=size,PNG_dpi=dpi,PDF_embedded_fonts=len(fonts),SVG_editable_text=True,
        review_sha256=digest(review),figure_manifest_sha256=digest(manifest),report_links=links,default_changed=False,new_training_steps=0,cloud_calls=0))
    write(out/'最终状态.json',dict(status='closed',completed=True,target_evidence_passed=v['target_evidence_passed'],
        independent_candidate_confirmed=v.get('independent_candidate_confirmed'),new_source_files=v.get('new_source_files',0),records=a['records'],
        default_changed=False,new_training_steps=0,cloud_calls=0,verdict_sha256=digest(out/'验收结论.json'),delivery_sha256=digest(out/'交付核验.json')))
    print(read(out/'交付核验.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('evidence','confirmation','deliver-evidence','deliver-confirmation'));args=p.parse_args()
    {'evidence':evidence,'confirmation':confirmation,'deliver-evidence':lambda:delivery('evidence'),'deliver-confirmation':lambda:delivery('confirmation')}[args.mode]()
