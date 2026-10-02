"""Verify published means, budgets, provenance, links and vector media artifacts."""
import re,sys
from pathlib import Path
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import fitz
from eval.protocol_adaptation_navigation import ROOT,OUT,TRAIN,PRIOR,ARMS,SEEDS,read,write,digest
from eval.audit_local_ledger_expansion import independent_effect,compare_effect


def main():
    summary=read(OUT/'主对照/对照汇总.json');v=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');training=read(TRAIN/'独立复核.json');reg=read(TRAIN/'预登记.json')
    if not audit['passed'] or not training['passed'] or v['audit_sha256']!=digest(OUT/'独立复核.json') or v['summary_sha256']!=digest(OUT/'主对照/对照汇总.json'):raise ValueError('audit receipt')
    for arm in ('M0',*ARMS):
        base=PRIOR if arm=='M0' else OUT;jobs=[read(base/f'主对照/{arm}_s{seed}_CueFull_结果.json') for seed in SEEDS]
        for metric,key in [('sr','sr'),('mean_sg_all_episodes','sg')]:
            values=[r['metrics'][metric] for r in jobs]
            if values!=summary['arms'][arm][key+'_by_seed'] or abs(sum(values)/3-summary['arms'][arm][key+'_mean'])>1e-12:raise ValueError('published means')
            for bucket in ('by_distance','by_split'):
                for k,r in summary['arms'][arm][bucket].items():
                    if abs(sum(j[bucket][k][metric] for j in jobs)/3-r[key])>1e-12:raise ValueError('published strata')
    secondary=read(OUT/'继续训练对原基线_补充比较.json')['effect']
    compare_effect(secondary,independent_effect([read(OUT/f'主对照/Continue5_s{s}_CueFull_结果.json') for s in SEEDS],[read(PRIOR/f'主对照/M0_s{s}_CueFull_结果.json') for s in SEEDS]))
    total_actions=total_optimizer=0
    for seed in SEEDS:
        for arm in ARMS:
            folder=TRAIN/f'{arm}_s{seed}';r=read(folder/'训练资源.json');b=read(folder/'训练预算收据.json')
            if r['parameters']!=797701 or r['trainable_parameters']!=666117 or r['steps']!=81920 or b['optimizer_updates']!=1024:raise ValueError('architecture/budget')
            total_actions+=r['steps'];total_optimizer+=b['optimizer_updates']
    if total_actions!=491520 or total_optimizer!=6144:raise ValueError('total budget')
    rows=read(OUT/'邻接时机诊断.json');timing=read(OUT/'邻接时机汇总.json')['arms']
    for arm in ('M0',*ARMS):
        selected=[r for r in rows if r['arm']==arm];near=[r for r in selected if r['budgeted_adjacency_observed']];t=timing[arm]
        if len(selected)!=1500 or len(near)!=t['budgeted_adjacency_episodes'] or t['budgeted_adjacency_rate']!=len(near)/1500:raise ValueError('diagnostic rate')
        if near and (abs(sum(r['first_adjacency_step'] for r in near)/len(near)-t['mean_first_adjacency_step_among_reached'])>1e-12 or abs(sum(r['remaining_at_first_adjacency'] for r in near)/len(near)-t['mean_remaining_at_first_adjacency_among_reached'])>1e-12):raise ValueError('diagnostic means')
    paired_timing=read(OUT/'邻接时机配对补充.json')['comparisons'];timing_index={(r['arm'],r['seed'],r['episode_id']):r for r in rows}
    for arm,reference in [('Continue5','M0'),('Adapt10','M0'),('Adapt10','Continue5')]:
        pairs=[(r,timing_index[reference,r['seed'],r['episode_id']]) for r in rows if r['arm']==arm]
        changes=[a['first_adjacency_step']-b['first_adjacency_step'] for a,b in pairs if a['budgeted_adjacency_observed'] and b['budgeted_adjacency_observed']]
        r=paired_timing[arm+'_vs_'+reference]
        if r['both_reached_records']!=len(changes) or (r['earlier'],r['same'],r['later'])!=(sum(x<0 for x in changes),sum(x==0 for x in changes),sum(x>0 for x in changes)) or abs(r['mean_first_step_change_on_common_reached']-sum(changes)/len(changes))>1e-12:raise ValueError('common-cohort timing')
        wait=sum((a['first_adjacency_step'] if a['budgeted_adjacency_observed'] else 20)-(b['first_adjacency_step'] if b['budgeted_adjacency_observed'] else 20) for a,b in pairs)/len(pairs)
        if abs(wait-r['restricted_mean_wait_change_all_planned'])>1e-12:raise ValueError('censored diagnostic wait')
    links=0
    for f in (OUT/'探索器协议适配报告.md',OUT/'获取线索时机配对补充说明.md',OUT/'README.md',TRAIN/'README.md'):
        for target in re.findall(r'\]\(([^)]+)\)',f.read_text(encoding='utf8')):
            if not (f.parent/target).is_file():raise ValueError('broken link '+target)
            links+=1
    manifest=ROOT/'绘图/绘图数据/探索器训练协议适配_v1.json';item=read(manifest)['figures'][0]
    for entry in item['files']:
        if digest(ROOT/entry['path'])!=entry['sha256']:raise ValueError('figure hash')
    base=ROOT/'绘图/数据结果图';name=item['id']
    with Image.open(base/(name+'.png')) as image:
        dimensions=image.size;dpi=image.info.get('dpi')
        if not dpi or min(dpi)<299:raise ValueError('PNG resolution')
    tags=[x.tag.rsplit('}',1)[-1] for x in ET.parse(base/(name+'.svg')).getroot().iter()]
    if 'image' in tags or 'text' not in tags:raise ValueError('editable vector SVG')
    with fitz.open(base/(name+'.pdf')) as pdf:
        fonts=pdf[0].get_fonts(full=True)
        if len(pdf)!=1 or pdf[0].get_images() or not fonts or any(not pdf.extract_font(f[0])[-1] for f in fonts):raise ValueError('PDF vector/fonts')
    review=ROOT/'绘图/审阅/探索器训练协议适配设计与核验_v1.md'
    if not review.exists():raise ValueError('actual PNG review first')
    for key,basepath in [('protected_sha256',ROOT),('source_sha256',ROOT/'project/src'),('frozen_core_sha256',ROOT/'project/src'),('encoder_sha256',ROOT),('training_input_sha256',ROOT),('evaluation_input_sha256',ROOT)]:
        for n,h in reg[key].items():
            if digest(basepath/n)!=h:raise ValueError('final preservation '+n)
    for r in reg['source_rows']:
        if digest(ROOT/r['raw_path'])!=r['raw_sha256']:raise ValueError('fit raw source preservation')
    cache=ROOT/'DATA/processed_data/Masa/探索器协议适配_v1/训练特征'
    if digest(cache/'全局特征.npz')!=training['cache_sha256']:raise ValueError('verified cache unchanged')
    if digest(ROOT/'project/local_policy_default.json')!=reg['default_sha256'] or digest(ROOT/'project/cloud_provider_preferences.json')!=reg['cloud_preferences_sha256']:raise ValueError('default/preferences drift')
    write(OUT/'交付核验.json',dict(passed=True,aggregate_and_stratified_means_verified=True,all_training_budgets_verified=True,new_training_actions=total_actions,
        optimizer_updates=total_optimizer,report_links=links,PNG_pixels=dimensions,PNG_dpi=dpi,PDF_embedded_fonts=len(fonts),SVG_editable_text=True,
        actual_review_sha256=digest(review),figure_manifest_sha256=digest(manifest),report_sha256=digest(OUT/'探索器协议适配报告.md'),
        protected_files=len(reg['protected_sha256']),input_files=len(reg['evaluation_input_sha256']),default_changed=False,cloud_calls=0))
    final=dict(status='closed',completed=True,candidate_passed=v['candidate_passed'],new_primary_records=3000,
        target_records=4500 if v['target_controls_started'] else 0,new_training_actions=491520,verification_replay_actions=491520,
        default_changed=False,cloud_calls=0,new_unseen_evaluation_sources=0,verdict_sha256=digest(OUT/'验收结论.json'),delivery_sha256=digest(OUT/'交付核验.json'))
    write(OUT/'最终状态.json',final);write(TRAIN/'最终状态.json',dict(status='closed',completed=True,training_audit_passed=True,models=6,
        new_training_actions=491520,verification_replay_actions=491520,evaluation_final_state=final,default_changed=False))
    print(read(OUT/'交付核验.json'))


if __name__=='__main__':main()
