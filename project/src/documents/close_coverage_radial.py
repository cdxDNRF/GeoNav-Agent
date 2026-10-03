"""Index, verify and seal the completed radial protection batch."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import fitz
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'DATA/processed_data/MasaRoads/评测结果/覆盖接管径向保护验证_v1'
FIG=ROOT/'绘图'
REPORT=RUN/'径向保护开发验证报告.md'
MANIFEST=FIG/'绘图数据/覆盖接管径向保护_v1.json'
VISUAL=FIG/'审阅/覆盖接管径向保护视觉核验_v1.json'
FILE_QA=FIG/'审阅/覆盖接管径向保护文件核验_v1.json'
DELIVERY=RUN/'阶段交付核验.json'


def read(p):return json.loads(p.read_text('utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def write(p,v,exclusive=False):
    with p.open('x' if exclusive else 'w',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)


def description():
    s=read(RUN/'主对照/对照汇总.json');v=read(RUN/'验收结论.json')
    a,b=s['candidate']['metrics'],s['reference']['metrics']
    status=('开发与目标证据通过，待新区域确认' if v['eligible_for_new_area_confirmation'] else
            '主保护门槛未通过' if not v['development_main_passed'] else '目标证据未通过')
    long=s['candidate']['by_stratum']['long_distance']
    return (f"同10个已消费区域、750题×3权重的后验开发，唯一新增径向保护：混合SR {a['sr']*100:.2f}%"
        f"（M0 {b['sr']*100:.2f}%，{s['effects']['all']['sr_gain']*100:+.2f}点）；"
        f"混合SG {b['mean_sg_m']:.2f}→{a['mean_sg_m']:.2f}米，长距离SR {long['sr']*100:.2f}%。"
        f"恢复{s['recovered']}、损伤{s['harmed']}，净增{s['recovered']-s['harmed']}成功。"
        f"{status}；10测试及完整重放通过，0训练/0云端，默认M0保持，未进行新区域确认。")


def index():
    verdict=read(RUN/'验收结论.json');manifest=read(MANIFEST)
    assert verdict['completed'] and read(VISUAL)['passed'] and REPORT.is_file()
    value=read(ROOT/'项目导航/实验索引.json')
    assert len(value['batches'])==67 and not any(b['path']==rel(RUN) for b in value['batches'])
    text=description()
    value['date']='2026-10-03'
    value['batches'].append(dict(dataset='MasaRoads',kind='评测',name=RUN.name,category='公开径向保护后验开发',
        status='开发通过待独立确认' if verdict['eligible_for_new_area_confirmation'] else '保护或目标证据门槛未通过',
        scope='10已消费区域/750任务/3权重；无训练/云端/新区域确认',path=rel(RUN),reports=[rel(REPORT)],
        evidence=[rel(RUN/'验收结论.json'),rel(RUN/'主对照/独立复核.json'),rel(RUN/'离线诊断/独立算术复核.json')]))
    # Check required anchors before writing any navigation changes.
    root_text=(ROOT/'README.md').read_text('utf-8')
    progress=(ROOT/'项目导航/当前进度.md').read_text('utf-8')
    assert '## 最新阶段：三步覆盖接管开发验证收尾' in root_text
    assert '## 最新：三步覆盖单因素开发收尾' in progress
    assert '## 下一项方向' in progress
    catalogue=read(FIG/'图表来源清单.json')
    assert not any(f['id']==manifest['figures'][0]['id'] for f in catalogue['figures'])
    write(ROOT/'项目导航/实验索引.json',value)
    p=ROOT/'项目导航/实验索引.md'
    p.write_text(p.read_text('utf-8').replace('67批','68批').replace('67个','68个')+
        f'\n\n## 覆盖接管径向保护验证_v1\n\n{text}\n\n[报告](../{rel(REPORT)}) / [判定](../{rel(RUN/"验收结论.json")}) / [复核](../{rel(RUN/"主对照/独立复核.json")})。\n','utf-8')
    latest=f'## 最新阶段：覆盖接管径向保护验证完成\n\n{text}\n\n[报告]({rel(REPORT)}) / [判定]({rel(RUN/"验收结论.json")}) / [图48]({rel(FIG/"覆盖接管径向保护图注.md")})。\n\n'
    root_text=root_text.replace('## 最新阶段：三步覆盖接管开发验证收尾',latest+'## 前一轮：三步覆盖接管开发验证收尾',1)
    start,end=root_text.index('更新至'),root_text.index('## 最新阶段')
    root_text=root_text[:start]+'更新至2026年10月03日。原S2/S3/S4、SwissView及实际面积空间隔离确认保持有效；最新径向保护单因素开发已完成，默认M0保持。具体结果与边界见最新阶段和当前进度。\n\n'+root_text[end:]
    start=root_text.index('后续开发已登记：');end=root_text.index('\n\n',start)
    root_text=root_text[:start]+('本轮开发目录：`DATA/processed_data/MasaRoads/评测结果/覆盖接管径向保护验证_v1/`及`离线诊断/`、`主对照/`、条件性`目标对照/`、`源码快照/{agents,eval,tests}/`。唯一径向保护候选已冻结并完成验证；旧数据、代码、模型和判定保留。')+root_text[end:]
    (ROOT/'README.md').write_text(root_text.replace('全部67批','全部68批').replace('67批实验','68批实验').replace('47张实验与数据协议图','48张实验与数据协议图'),'utf-8')
    latest=f'## 最新：径向保护单因素开发完成\n\n{text}\n\n[报告](../{rel(REPORT)}) / [判定](../{rel(RUN/"验收结论.json")}) / [图48](../{rel(FIG/"覆盖接管径向保护图注.md")})。\n\n'
    progress=progress.replace('## 最新：三步覆盖单因素开发收尾',latest+'## 前一轮：三步覆盖单因素开发收尾',1)
    start,end=progress.index('截至'),progress.index('## 最新：')
    progress=progress[:start]+'截至2026-10-03，原正式验收保持有效；Coverage3及径向保护均为已消费10区的后验开发。默认M0保持，实际判定以每批冻结门槛为准。\n\n'+progress[end:]
    start=progress.index('## 下一项方向')
    next_text=('开发与目标证据全部通过，下一步另立空间隔离新区域确认协议，冻结全部任务后再导航。当前10区域不得再次标为未消费。' if verdict['eligible_for_new_area_confirmation'] else
        '径向保护按一次冻结验证收束，不以调整门槛或继续叠加规则追分。后续依据已保存分层证据决定搜索覆盖目标与预算分配的新假设，再另立协议；本批未通过判定保留。')
    body_start=start+len('## 下一项方向\n\n')
    body_end=progress.find('\n\n',body_start)
    if body_end<0:body_end=len(progress)
    progress=progress[:start]+'## 下一项方向\n\n'+next_text+'默认M0及原正式验收保持，真值仅用于事后诊断。\n'+progress[body_end:]
    (ROOT/'项目导航/当前进度.md').write_text(progress,'utf-8')
    for name in ('项目导航/README.md','项目导航/目录结构与维护.md'):
        p=ROOT/name;p.write_text(p.read_text('utf-8').replace('67批','68批').replace('67个','68个'),'utf-8')
    catalogue['figures'].extend(manifest['figures']);write(FIG/'图表来源清单.json',catalogue)
    p=FIG/'README.md';stem=manifest['figures'][0]['id']
    p.write_text(p.read_text('utf-8')+f'\n\n## 图48：覆盖接管径向保护\n\n{text}\n\n[PDF](数据结果图/{stem}.pdf) / [SVG](数据结果图/{stem}.svg) / [PNG](数据结果图/{stem}.png) / [图注](覆盖接管径向保护图注.md) / [来源](绘图数据/覆盖接管径向保护_v1.json)。\n','utf-8')
    print(dict(index_entries=68,logical_figures=48),flush=True)


def check():
    assert not FILE_QA.exists() and not DELIVERY.exists()
    reg=read(RUN/'预登记.json');v=read(RUN/'验收结论.json');state=read(RUN/'最终状态.json')
    assert read(RUN/'预登记封存.json')['registration_sha256']==digest(RUN/'预登记.json')
    for field in ('protected_sha256','source_sha256','frozen_sha256'):
        for name,expected in reg[field].items():assert digest(ROOT/name)==expected,name
    assert v['completed'] and state['completed'] and not v['default_changed']
    assert v['new_training_steps']==v['cloud_calls']==0 and not state['new_area_confirmation_started']
    assert v['default_sha256']==digest(ROOT/'project/local_policy_default.json')
    assert state['verdict_sha256']==digest(RUN/'验收结论.json')
    summary=read(RUN/'主对照/对照汇总.json');audit=read(RUN/'主对照/独立复核.json')
    assert audit['passed'] and audit['records']==2250 and audit['summary_sha256']==digest(RUN/'主对照/对照汇总.json')
    assert v['SR']==summary['candidate']['metrics']['sr'] and v['original_SR']==summary['reference']['metrics']['sr']
    assert v['development_main_passed']==summary['main_numeric_passed']
    control_records=control_actions=0
    if v['controls_started']:
        ca=read(RUN/'目标对照/独立复核.json');cs=read(RUN/'目标对照/对照汇总.json')
        assert v['development_main_passed'] and ca['passed'] and ca['records']==6750
        assert ca['summary_sha256']==digest(RUN/'目标对照/对照汇总.json')
        assert v['development_target_evidence_passed']==cs['target_numeric_passed']
        control_records,control_actions=ca['records'],ca['actions']
    else:assert not list((RUN/'目标对照').glob('*轨迹.jsonl'))
    assert v['records']==state['records']==2250+control_records
    for folder in ('主对照','目标对照'):
        for p in (RUN/folder).glob('*结果.json'):
            assert read(p)['trajectory_sha256']==digest(p.with_name(p.name.replace('结果.json','轨迹.jsonl')))
    diag=read(RUN/'离线诊断/独立算术复核.json')
    assert diag['passed'] and (diag['navigation_records'],diag['rechecked_actions'])==(4500,74189)
    for name,expected in diag['files_sha256'].items():assert digest(RUN/'离线诊断'/name)==expected
    tests=read(ROOT/'选题报告相关/覆盖接管径向保护测试_v1.json');assert tests['passed'] and tests['tests_run']==10
    visual,manifest=read(VISUAL),read(MANIFEST)
    assert visual['passed'] and visual['PNG_and_rendered_PDF_reviewed'] and visual['final_plots']==1
    for mapping in (visual['files_sha256'],manifest['source_sha256'],manifest['other_outputs_sha256']):
        for name,expected in mapping.items():assert digest(ROOT/name)==expected,name
    assert manifest['main_summary']==summary and manifest['verdict']==v
    assert manifest['script_sha256']==digest(ROOT/manifest['reproduction_script'])
    figure=manifest['figures'][0];files={}
    for item in figure['files']:
        p=ROOT/item['path'];assert digest(p)==item['sha256'];files[p.suffix]=p
    with Image.open(files['.png']) as im:
        assert min(im.info['dpi'])>299;size,dpi=im.size,im.info['dpi']
    with fitz.open(files['.pdf']) as pdf:
        assert len(pdf)==1 and not pdf[0].get_images()
        fonts=pdf[0].get_fonts(full=True);assert fonts and all(f[2]!='Type3' and pdf.extract_font(f[0])[3] for f in fonts)
    svg=Counter(n.tag.rsplit('}',1)[-1] for n in ET.parse(files['.svg']).iter())
    assert svg['text']>0 and svg['image']==0
    assert figure==next(f for f in read(FIG/'图表来源清单.json')['figures'] if f['id']==figure['id'])
    ids={int(p.name.split('_')[0]) for p in (FIG/'数据结果图').glob('*.png') if re.match(r'^\d{2}_',p.name)}
    assert ids==set(range(1,49))
    idx=read(ROOT/'项目导航/实验索引.json');assert len(idx['batches'])==len({b['path'] for b in idx['batches']})==68
    entry=next(b for b in idx['batches'] if b['path']==rel(RUN))
    assert all((ROOT/n).is_file() for n in (*entry['reports'],*entry['evidence']))
    markdowns=[ROOT/'README.md',ROOT/'项目导航/当前进度.md',ROOT/'项目导航/实验索引.md',
        REPORT,RUN/'README.md',FIG/'README.md',FIG/'覆盖接管径向保护图注.md']
    links=0
    for p in markdowns:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text('utf-8')):
            link=link.strip('<>').split('#')[0]
            if not link or re.match(r'[a-z]+://',link):continue
            assert (p.parent/link).resolve().exists(),(p,link)
            links+=1
    write(FILE_QA,dict(passed=True,index_entries=68,logical_figures=48,local_links_verified=links,
        PNG_size=size,DPI=dpi,PDF_embedded_fonts=True,PDF_raster_images=0,SVG_editable_texts=svg['text'],
        original_data_model_runtime_hashes_unchanged=True),True)
    paths=markdowns+[RUN/n for n in ('预登记.json','预登记封存.json','验收结论.json','最终状态.json',
        '主对照/独立复核.json','离线诊断/独立算术复核.json')]
    paths.extend((MANIFEST,VISUAL,FILE_QA,Path(__file__)))
    paths.extend(files.values());paths.extend(ROOT/n for n in ('项目导航/实验索引.json','项目导航/README.md',
        '项目导航/目录结构与维护.md','绘图/图表来源清单.json'))
    result=dict(completed=True,development_main_passed=v['development_main_passed'],
        target_evidence_passed=v['development_target_evidence_passed'],eligible_for_new_area_confirmation=v['eligible_for_new_area_confirmation'],
        default_changed=False,new_training_steps=0,cloud_calls=0,new_area_confirmation_started=False,
        diagnostic_old_records=4500,diagnostic_old_actions=74189,new_main_records=2250,new_main_actions=audit['actions'],
        conditional_control_records=control_records,conditional_control_actions=control_actions,
        tests_passed=10,index_entries=68,logical_figures=48,protected_files_verified=len(reg['protected_sha256']),
        all_original_data_models_results_and_runtime_preserved=True,source_sha256={rel(p):digest(p) for p in paths})
    write(DELIVERY,result,True);print({k:v for k,v in result.items() if k!='source_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('index','check'))
    {'index':index,'check':check}[parser.parse_args().mode]()
