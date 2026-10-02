"""Independent published means, diagnostic counts, links and media QA."""
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import fitz
from eval.ledger_protection import ROOT,OUT,PRIOR,SEEDS,read,lines,write,digest


def main():
    s=read(OUT/'主对照/对照汇总.json')
    for arm in ('M0','M1','P'):
        folder=OUT if arm=='P' else PRIOR;jobs=[read(folder/f'主对照/{arm}_s{seed}_CueFull_结果.json') for seed in SEEDS]
        for metric,key in [('sr','sr'),('mean_sg_all_episodes','sg')]:
            vals=[x['metrics'][metric] for x in jobs]
            if vals!=s['arms'][arm][key+'_by_seed'] or abs(sum(vals)/3-s['arms'][arm][key+'_mean'])>1e-12:raise ValueError('published means')
            for bucket in ('by_distance','by_split'):
                for k,v in s['arms'][arm][bucket].items():
                    if abs(sum(x[bucket][k][metric] for x in jobs)/3-v[key])>1e-12:raise ValueError('published strata')
    pairs=read(OUT/'逐题保护与损伤.json');diag=read(OUT/'上一轮损伤诊断.json');oldpairs={(r['seed'],r['episode_id']):r for r in read(PRIOR/'逐题恢复与损伤.json')}
    counts={'harm':[0,0,0],'recover':[0,0,0],'same_success':[0,0,0],'same_fail':[0,0,0]}
    for seed in SEEDS:
        for row in lines(PRIOR/f'主对照/M1_s{seed}_CueFull_轨迹.jsonl'):
            old=oldpairs[seed,row['episode_id']];group='harm' if old['harmed'] else 'recover' if old['recovered'] else 'same_success' if old['M0_success'] else 'same_fail'
            counts[group][0]+=1
            first=next((d for d in row['decisions'] if d['action']!=d['base']['action']),None)
            if first is not None:
                b=first['base'];r,c=b['public_position'];dr,dc={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}[b['action']]
                counts[group][1]+=(r+dr)*10+c+dc not in b['public_visited'];counts[group][2]+=first['interaction'] is None
    for group,v in counts.items():
        r=diag['groups'][group]
        if v!=[r['total'],r.get('first_diverge_base_fresh',0),r.get('first_diverge_second_step',0)]:raise ValueError('independent diagnostic counts')
    links=0
    gateway=ROOT/'DATA/processed_data/Masa/评测结果/云端8790接口检查_v1'
    for f in [OUT/'保护规则验证报告.md',OUT/'README.md',gateway/'接口检查报告.md',gateway/'README.md']:
        for link in re.findall(r'\]\(([^)]+)\)',f.read_text(encoding='utf8')):
            if not (f.parent/link).is_file():raise ValueError('broken report link '+link)
            links+=1
    manifest=ROOT/'绘图/绘图数据/账本探索保护规则_v1.json';item=read(manifest)['figures'][0]
    for f in item['files']:
        if digest(ROOT/f['path'])!=f['sha256']:raise ValueError('figure hash')
    name=item['id'];base=ROOT/'绘图/数据结果图'
    with Image.open(base/(name+'.png')) as image:
        dims=image.size;dpi=image.info.get('dpi')
        if not dpi or min(dpi)<299:raise ValueError('PNG resolution')
    tags=[x.tag.rsplit('}',1)[-1] for x in ET.parse(base/(name+'.svg')).getroot().iter()]
    if 'image' in tags or 'text' not in tags:raise ValueError('editable vector SVG')
    with fitz.open(base/(name+'.pdf')) as doc:
        fonts=doc[0].get_fonts(full=True)
        if len(doc)!=1 or doc[0].get_images() or not fonts or any(not doc.extract_font(f[0])[-1] for f in fonts):raise ValueError('PDF fonts/vector')
    review=ROOT/'绘图/审阅/账本探索保护规则设计与核验_v1.md'
    if not review.is_file():raise ValueError('actual image review required')
    verdict=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json')
    if not audit['passed'] or verdict['audit_sha256']!=digest(OUT/'独立复核.json'):raise ValueError('audit receipt')
    # Re-score gateway final answers separately; no provider calls.
    intents=list(lines(gateway/'请求意图.jsonl'));responses=list(lines(gateway/'请求响应.jsonl'));gsm=read(gateway/'检查汇总.json')
    if len(intents)!=9 or len(responses)!=9 or gsm['generation_requests']!=9:raise ValueError('gateway accounting')
    for m,row in gsm['models'].items():
        text=image_count=correct=0
        for intent,response in zip(intents,responses):
            if response['requested_model']!=m:continue
            final=response.get('raw_content');actual=False
            if isinstance(final,str) and response.get('finish_reason')!='length':
                import json
                clean=re.sub(r'^```(?:json)?\s*\n|\n```$','',final.strip(),flags=re.I)
                try:
                    answer=json.loads(clean);expected=intent['expected']
                    actual=isinstance(answer,dict) and set(answer)==set(expected) and all(type(answer[k]) is type(v) and answer[k]==v for k,v in expected.items())
                except (ValueError,TypeError):pass
            if actual!=response['correct']:raise ValueError('gateway raw final scoring')
            if response['kind']=='text':text+=actual
            else:image_count+=1;correct+=actual
        if (bool(text),image_count,correct,correct==2)!=(row['text_passed'],row['image_requests'],row['image_correct'],row['limited_visual_check_passed']):raise ValueError('gateway per-model summary')
    write(OUT/'交付核验.json',dict(passed=True,aggregate_and_stratified_means_checked=True,all1500_diagnostic_labels_checked=True,
        report_links=links,PNG_pixels=dims,PNG_dpi=dpi,PDF_embedded_fonts=len(fonts),SVG_editable_text=True,
        review_sha256=digest(review),figure_manifest_sha256=digest(manifest),gateway_generations_independently_rescored=9,
        gateway_report_sha256=digest(gateway/'接口检查报告.md'),audit_sha256=digest(OUT/'独立复核.json'),default_changed=False))
    print(read(OUT/'交付核验.json'))


if __name__=='__main__':main()
