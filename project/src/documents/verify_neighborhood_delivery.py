"""Check published statistics, supplemental budget diagnosis, links and exports."""
import json,re,sys
from pathlib import Path
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import fitz
from eval.neighborhood_coverage import ROOT,OUT,PRIOR,SEEDS,read,lines,write,digest


def main():
    s=read(OUT/'主对照/对照汇总.json');v=read(OUT/'验收结论.json');a=read(OUT/'独立复核.json')
    if not a['passed'] or v['audit_sha256']!=digest(OUT/'独立复核.json') or v['summary_sha256']!=digest(OUT/'主对照/对照汇总.json'):raise ValueError('audit receipt')
    for arm in ('M0','M1','N'):
        folder=OUT if arm=='N' else PRIOR;jobs=[read(folder/f'主对照/{arm}_s{seed}_CueFull_结果.json') for seed in SEEDS]
        for metric,key in [('sr','sr'),('mean_sg_all_episodes','sg')]:
            vals=[r['metrics'][metric] for r in jobs]
            if vals!=s['arms'][arm][key+'_by_seed'] or abs(sum(vals)/3-s['arms'][arm][key+'_mean'])>1e-12:raise ValueError('aggregate means')
            for bucket in ('by_distance','by_split'):
                for k,r in s['arms'][arm][bucket].items():
                    if abs(sum(j[bucket][k][metric] for j in jobs)/3-r[key])>1e-12:raise ValueError('stratified means')
    tasks={x['episode_id']:x for x in read(OUT/'导航任务.json')};late=[];never=budgeted=0
    for seed in SEEDS:
        for row in lines(PRIOR/f'主对照/M0_s{seed}_CueFull_轨迹.jsonl'):
            if row['success']:continue
            goal=tasks[row['episode_id']]['goal'];previous=[d['base']['public_position'] for d in row['decisions']]
            before=min(abs(r-goal//10)+abs(c-goal%10) for r,c in previous);cell=row['trajectory'][-1]['patch_id'];terminal=abs(cell//10-goal//10)+abs(cell%10-goal%10)
            budgeted+=before==1;never+=before>1 and terminal>1
            if before>1 and terminal==1:late.append(dict(seed=seed,episode_id=row['episode_id'],source=row['source'],minimum_distance_with_action_budget=before,terminal_distance=terminal,remaining_budget=0))
    supplement=read(OUT/'终步预算补充诊断.json')
    if late!=supplement['cases'] or len(late)!=supplement['first_adjacency_at_zero_budget'] or never!=supplement['never_adjacent_including_terminal'] or budgeted!=supplement['failed_with_budgeted_adjacency']:raise ValueError('budget diagnosis')
    report=OUT/'目标邻域覆盖规划报告.md';links=0
    for path in (report,OUT/'README.md'):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf8')):
            if not (path.parent/target).is_file():raise ValueError('broken link '+target)
            links+=1
    manifest=ROOT/'绘图/绘图数据/目标邻域覆盖规划_v1.json';item=read(manifest)['figures'][0]
    for entry in item['files']:
        if digest(ROOT/entry['path'])!=entry['sha256']:raise ValueError('figure SHA')
    base=ROOT/'绘图/数据结果图';name=item['id']
    with Image.open(base/(name+'.png')) as image:
        dimensions=image.size;dpi=image.info.get('dpi')
        if not dpi or min(dpi)<299:raise ValueError('PNG resolution')
    tags=[x.tag.rsplit('}',1)[-1] for x in ET.parse(base/(name+'.svg')).getroot().iter()]
    if 'image' in tags or 'text' not in tags:raise ValueError('SVG vector/editable text')
    with fitz.open(base/(name+'.pdf')) as pdf:
        fonts=pdf[0].get_fonts(full=True)
        if len(pdf)!=1 or pdf[0].get_images() or not fonts or any(not pdf.extract_font(f[0])[-1] for f in fonts):raise ValueError('PDF embedded fonts')
    review=ROOT/'绘图/审阅/目标邻域覆盖规划设计与核验_v1.md'
    if not review.is_file():raise ValueError('actual image review first')
    reg=read(OUT/'预登记.json')
    for key,basepath in [('protected_sha256',ROOT),('input_sha256',ROOT),('source_sha256',ROOT/'project/src'),('frozen_core_sha256',ROOT/'project/src')]:
        for n,h in reg[key].items():
            if digest(basepath/n)!=h:raise ValueError('preservation '+n)
    if digest(ROOT/'project/local_policy_default.json')!=reg['default_sha256'] or digest(ROOT/'project/cloud_provider_preferences.json')!=reg['cloud_preferences_sha256']:raise ValueError('default/preferences drift')
    write(OUT/'交付核验.json',dict(passed=True,aggregate_and_stratified_means_checked=True,late_adjacency_records_verified=len(late),
        all_report_links_verified=links,PNG_pixels=dimensions,PNG_dpi=dpi,PDF_embedded_fonts=len(fonts),SVG_editable_text=True,
        actual_review_sha256=digest(review),figure_manifest_sha256=digest(manifest),supplement_sha256=digest(OUT/'终步预算补充诊断.json'),
        report_sha256=digest(report),preserved_historical_files=len(reg['protected_sha256']),input_files=len(reg['input_sha256']),default_changed=False,cloud_calls=0))
    write(OUT/'最终状态.json',dict(status='closed',completed=True,candidate_passed=v['candidate_passed'],default_changed=False,
        new_navigation_records=1500,target_records=0 if not v['target_controls_started'] else 4500,cloud_calls=0,new_training_steps=0,new_unseen_source_files=0,
        verdict_sha256=digest(OUT/'验收结论.json'),delivery_sha256=digest(OUT/'交付核验.json'),next_plan='选题报告相关/十乘十探索器训练适配下一项草案_v1.md',next_training_started=False))
    print(read(OUT/'交付核验.json'))


if __name__=='__main__':main()
