"""Index and seal the independently audited data-only preparation stage."""
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
RUN=ROOT/'DATA/processed_data/MasaRoads/径向保护新区域确认_v1'
QA,META,DATA=RUN/'核验',RUN/'元数据',RUN/'工程数据'
FIG=ROOT/'绘图'
REPORT=RUN/'新区域确认数据准备报告.md'
DRAFT=RUN/'下一项冻结径向保护新区域导航确认方案_v1.md'
MANIFEST=FIG/'绘图数据/径向保护新区域确认准备_最终_v2.json'
VISUAL=FIG/'审阅/径向保护新区域准备视觉核验_v2.json'
FILE_QA=FIG/'审阅/径向保护新区域准备文件核验_v2.json'
DELIVERY=QA/'阶段交付核验.json'


def read(p):return json.loads(p.read_text('utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def write(p,v,exclusive=False):
    with p.open('x' if exclusive else 'w',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)


def description():
    a=read(QA/'独立复核.json');s=read(META/'连续区域选择.json')
    return (f"径向保护的10个预留确认新区已完成数据准备，每区9 km²、300米/格、10×10/B20，"
        f"1000图块、750固定不同任务、1200几何探针。新增下载{s['newly_downloaded_sources']}源/"
        f"{s['received_bytes']/1024**2:.2f} MiB，旧152源只读复用；新至旧151最短"
        f"{a['minimum_old_region_gap_m']/1000:.2f}公里，至已消费14区及新区间均至少3公里（0.001米容差）。"
        "12测试及标签/像素/采样/空间独立复核通过。尚未调用编码器、视觉头或导航，不产生新SR，默认M0保持。")


def index():
    assert read(QA/'验收结论.json')['data_engineering_passed'] and read(VISUAL)['passed']
    manifest=read(MANIFEST);text=description()
    idx=read(ROOT/'项目导航/实验索引.json')
    assert len(idx['batches'])==68 and not any(b['path']==rel(RUN) for b in idx['batches'])
    root_text=(ROOT/'README.md').read_text('utf-8');progress=(ROOT/'项目导航/当前进度.md').read_text('utf-8')
    assert '## 最新阶段：覆盖接管径向保护验证完成' in root_text
    assert '## 最新：径向保护单因素开发完成' in progress and '## 下一项方向' in progress
    catalog=read(FIG/'图表来源清单.json')
    assert not any(f['id']==manifest['figures'][0]['id'] for f in catalog['figures'])
    idx['date']='2026-10-03'
    idx['batches'].append(dict(dataset='MasaRoads',kind='数据',name=RUN.name,category='径向保护独立新区域准备',
        status='数据/空间通过；未导航、无新SR',scope='10预留新区/40源/1000图块/750路线/1200几何探针；预训练地理未知',
        path=rel(RUN),reports=[rel(REPORT),rel(RUN/'科研图排版修订_v2.md'),rel(DRAFT)],
        evidence=[rel(QA/'验收结论.json'),rel(QA/'独立复核.json'),rel(QA/'输入冻结结束.json')]))
    write(ROOT/'项目导航/实验索引.json',idx)
    p=ROOT/'项目导航/实验索引.md';p.write_text(p.read_text('utf-8').replace('68批','69批').replace('68个','69个')+
        f'\n\n## 径向保护新区域确认_v1（数据准备）\n\n{text}\n\n[报告](../{rel(REPORT)}) / [判定](../{rel(QA/"验收结论.json")}) / [下一项草案](../{rel(DRAFT)})。\n','utf-8')
    latest=f'## 最新阶段：径向保护独立新区域准备通过\n\n{text}\n\n[报告]({rel(REPORT)}) / [数据复核]({rel(QA/"独立复核.json")}) / [下一项导航草案]({rel(DRAFT)}) / [图49修订版]({rel(FIG/"径向保护新区域确认准备图注_v2.md")})。\n\n'
    root_text=root_text.replace('## 最新阶段：覆盖接管径向保护验证完成',latest+'## 前一轮：覆盖接管径向保护验证完成',1)
    start,end=root_text.index('更新至'),root_text.index('## 最新阶段')
    root_text=root_text[:start]+'更新至2026年10月03日。原正式验收与默认M0保持；径向保护开发/目标证据通过，独立确认的10个新区数据已准备并复核，尚未进行新区域导航。见最新阶段和下一项冻结草案。\n\n'+root_text[end:]
    start=root_text.index('独立确认的前置数据准备已登记：');end=root_text.index('\n\n',start)
    directory=('本轮数据目录：`DATA/raw_data/MasaRoads/径向保护新区域确认_v1/tiff/`、`DATA/processed_data/MasaRoads/径向保护新区域确认_v1/`及`元数据/`、`核验/源码快照/{data,env,eval,tests}/`、`工程数据/{mosaics,previews,patches/test/img_3000…img_3009}/`。测试临时目录`选题报告相关/径向新区域测试临时_*/grid/patches/test/img_3000/`已清理；旧影像只读、不复制或移动。')
    root_text=root_text[:start]+root_text[end+2:]
    pos=root_text.index('## 前一轮：覆盖接管径向保护验证完成');root_text=root_text[:pos]+directory+'\n\n'+root_text[pos:]
    (ROOT/'README.md').write_text(root_text.replace('全部68批','全部69批').replace('68批实验','69批实验').replace('48张实验与数据协议图','49张实验与数据协议图'),'utf-8')
    latest=f'## 最新：径向保护新区域数据准备通过\n\n{text}\n\n[报告](../{rel(REPORT)}) / [复核](../{rel(QA/"独立复核.json")}) / [下一项草案](../{rel(DRAFT)}) / [图49](../{rel(FIG/"径向保护新区域确认准备图注_v2.md")})。\n\n'
    progress=progress.replace('## 最新：径向保护单因素开发完成',latest+'## 前一轮：径向保护单因素开发完成',1)
    start,end=progress.index('截至'),progress.index('## 最新：')
    progress=progress[:start]+'截至2026-10-03，径向保护开发及目标证据通过；10个独立确认新区的数据/协议已通过，尚未调用模型或产生新SR。原正式验收和默认M0保持。\n\n'+progress[end:]
    start=progress.index('## 下一项方向');body_start=start+len('## 下一项方向\n\n');body_end=progress.find('\n\n',body_start)
    if body_end<0:body_end=len(progress)
    next_text=('先按已保存草案冻结新的M0/Coverage3Radial正式评测批次与接口，再提取冻结编码器特征并登记缓存。'
        '主对照4500轨迹，数值与重放通过才追加6750候选目标控制。新数据只通过工程，独立导航还未验收；'
        '全部独立门槛通过后才决定默认替换，当前M0保持。准备批次的未消费账本保持历史快照，正式调用状态另记。')
    progress=progress[:start]+'## 下一项方向\n\n'+next_text+'\n'+progress[body_end:]
    (ROOT/'项目导航/当前进度.md').write_text(progress,'utf-8')
    for name in ('项目导航/README.md','项目导航/目录结构与维护.md'):
        p=ROOT/name;p.write_text(p.read_text('utf-8').replace('68批','69批').replace('68个','69个'),'utf-8')
    catalog['figures'].extend(manifest['figures']);write(FIG/'图表来源清单.json',catalog)
    p=FIG/'README.md';stem=manifest['figures'][0]['id']
    p.write_text(p.read_text('utf-8')+f'\n\n## 图49：径向保护的新区域与任务准备\n\n{text}\n\n[PDF](数据结果图/{stem}.pdf) / [SVG](数据结果图/{stem}.svg) / [PNG](数据结果图/{stem}.png) / [图注](径向保护新区域确认准备图注_v2.md) / [来源](绘图数据/径向保护新区域确认准备_最终_v2.json)。原版及图例修订记录均保留。\n','utf-8')
    print(dict(index_entries=69,logical_figures=49,new_SR_produced=False),flush=True)


def check():
    assert not FILE_QA.exists() and not DELIVERY.exists() and not (QA/'最终状态.json').exists()
    reg=read(QA/'预登记.json');frozen=read(QA/'输入冻结结束.json');v=read(QA/'验收结论.json');a=read(QA/'独立复核.json')
    assert read(QA/'预登记封存.json')['sha256']==digest(QA/'预登记.json')
    for field in ('protected_sha256','source_sha256','frozen_metadata_sha256'):
        for name,expected in reg[field].items():assert digest(ROOT/name)==expected,name
    for name,expected in frozen['files_sha256'].items():assert digest(ROOT/name)==expected,name
    assert reg['protocol_sha256']==digest(ROOT/'选题报告相关/径向保护新区域数据准备冻结方案_v1.md')
    assert reg['default_sha256']==digest(ROOT/'project/local_policy_default.json')
    assert a['passed'] and v['data_engineering_passed'] and not v['new_SR_produced']
    assert v['audit_sha256']==digest(QA/'独立复核.json') and v['raw_and_derived_input_freeze_sha256']==digest(QA/'输入冻结结束.json')
    assert (a['isolated_regions'],a['exact_patches'],a['tasks'],a['probe_records'])==(10,1000,750,1200)
    assert a['new_downloaded_sources']<=200 and a['network_received_bytes']<=3*1024**3//2
    assert all(a[k]>=3000-.001 for k in ('minimum_old_region_gap_m','minimum_consumed14_region_gap_m','minimum_new_region_pair_gap_m'))
    assert a['navigation_model_calls']==a['new_navigation_records']==a['cloud_calls']==a['new_training_steps']==0
    usage=read(META/'本项目模型使用状态.json')
    assert len(usage['regions'])==10 and all(not r[k] for r in usage['regions'] for k in ('visual_encoder_called','visual_head_called','navigation_called','used_for_candidate_selection'))
    assert read(RUN/'测试记录.json')['successful'] and read(RUN/'测试记录.json')['tests_run']==12
    assert not list((ROOT/'选题报告相关').glob('径向新区域测试临时_*'))
    manifest=read(MANIFEST);visual=read(VISUAL)
    assert visual['passed'] and visual['PNG_and_rendered_PDF_reviewed'] and visual['final_plots']==1
    assert not read(FIG/'审阅/径向保护新区域准备视觉核验_v1.json')['passed']
    for mapping in (manifest['source_sha256'],manifest['other_outputs_sha256'],visual['files_sha256']):
        for name,expected in mapping.items():assert digest(ROOT/name)==expected,name
    assert manifest['audit']==a and manifest['verdict']==v
    assert manifest['script_sha256']==digest(ROOT/manifest['reproduction_script'])
    files={}
    for item in manifest['figures'][0]['files']:
        p=ROOT/item['path'];assert digest(p)==item['sha256'];files[p.suffix]=p
    with Image.open(files['.png']) as im:
        assert min(im.info['dpi'])>299;size,dpi=im.size,im.info['dpi']
    with fitz.open(files['.pdf']) as pdf:
        assert len(pdf)==1 and not pdf[0].get_images()
        fonts=pdf[0].get_fonts(full=True);assert fonts and all(f[2]!='Type3' and pdf.extract_font(f[0])[3] for f in fonts)
    svg=Counter(n.tag.rsplit('}',1)[-1] for n in ET.parse(files['.svg']).iter());assert svg['text']>0 and svg['image']==0
    assert manifest['figures'][0]==next(x for x in read(FIG/'图表来源清单.json')['figures'] if x['id']==manifest['figures'][0]['id'])
    ids={int(p.name.split('_')[0]) for p in (FIG/'数据结果图').glob('*.png') if re.match(r'^\d{2}_',p.name)}
    assert ids==set(range(1,50))
    idx=read(ROOT/'项目导航/实验索引.json');assert len(idx['batches'])==len({r['path'] for r in idx['batches']})==69
    entry=next(r for r in idx['batches'] if r['path']==rel(RUN))
    assert all((ROOT/n).is_file() for n in (*entry['reports'],*entry['evidence']))
    mds=[ROOT/'README.md',ROOT/'项目导航/当前进度.md',ROOT/'项目导航/实验索引.md',FIG/'README.md',
        REPORT,DRAFT,RUN/'README.md',RUN/'科研图排版修订_v2.md',FIG/'径向保护新区域确认准备图注_v2.md']
    links=0
    for p in mds:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text('utf-8')):
            link=link.strip('<>').split('#')[0]
            if not link or re.match(r'[a-z]+://',link):continue
            assert (p.parent/link).resolve().exists(),(p,link)
            links+=1
    write(FILE_QA,dict(passed=True,index_entries=69,logical_figures=49,local_links_verified=links,
        PNG_size=size,DPI=dpi,PDF_embedded_fonts=True,PDF_raster_images=0,SVG_editable_texts=svg['text'],
        original_runtime_and_data_unchanged=True,new_SR_produced=False),True)
    paths=mds+[MANIFEST,VISUAL,FILE_QA,Path(__file__),FIG/'审阅/径向保护新区域准备视觉核验_v1.json']
    paths.extend(files.values());paths.extend(QA/n for n in ('预登记.json','预登记封存.json','输入冻结结束.json','独立复核.json','验收结论.json'))
    paths.extend(ROOT/n for n in ('项目导航/实验索引.json','项目导航/README.md','项目导航/目录结构与维护.md','绘图/图表来源清单.json'))
    result=dict(completed=True,data_engineering_passed=True,new_confirmation_regions=10,exact_native_patches=1000,
        unique_navigation_tasks=750,geometric_probes=1200,new_downloaded_sources=a['new_downloaded_sources'],
        received_bytes=a['network_received_bytes'],tests_passed=12,protected_files_verified=len(reg['protected_sha256']),
        frozen_input_files_verified=len(frozen['files_sha256']),new_navigation_records=0,navigation_model_calls=0,
        new_training_steps=0,cloud_calls=0,new_SR_produced=False,default_changed=False,old_raw_moved_or_copied=False,
        index_entries=69,logical_figures=49,figure_revision='v2;original v1 preserved',
        source_sha256={rel(p):digest(p) for p in paths})
    write(DELIVERY,result,True)
    write(QA/'最终状态.json',dict(completed=True,data_engineering_passed=True,delivery_sha256=digest(DELIVERY),
        navigation_started=False,new_SR_produced=False,default_changed=False,next='freeze new same-task navigation confirmation'),True)
    print({k:v for k,v in result.items() if k!='source_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('index','check'))
    {'index':index,'check':check}[parser.parse_args().mode]()
