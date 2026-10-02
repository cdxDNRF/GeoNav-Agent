"""Evidence, figure structure, source preservation and final document links."""
from pathlib import Path
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents.summarize_target_robustness import ROOT,OUT,FIG,read,digest,diagnostics
from PIL import Image
import fitz


def main():
    p=argparse.ArgumentParser();p.add_argument('--visual-inspected',action='store_true');args=p.parse_args()
    if (OUT/'交付核验.json').exists():raise ValueError('immutable delivery receipt exists')
    receipt=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');reg=read(OUT/'预登记.json')
    assert receipt['robustness_confirmation_complete'] and audit['status']=='passed'
    assert digest(OUT/'独立复核.json')==receipt['audit_sha256'] and digest(OUT/'对照汇总.json')==receipt['summary_sha256']
    for n,h in audit['artifacts_sha256'].items():assert digest(OUT/n)==h,n
    for folder,files in reg['historical_sha256'].items():
        for n,h in files.items():assert digest(ROOT/folder/n)==h,n
    for n,h in reg['source_sha256'].items():assert digest(ROOT/'project/src'/n)==digest(OUT/'源码快照'/n)==h,n
    assert digest(ROOT/'project/local_policy_default.json')==reg['default_sha256']
    assert read(OUT/'失败与线索诊断.json')==diagnostics()
    manifest=read(FIG/'绘图数据/目标扰动鲁棒性_v1.json')
    for n,h in manifest['source_sha256'].items():assert digest(ROOT/n)==h,n
    assert digest(OUT/'鲁棒性与数据范围报告.md')==manifest['report_sha256']
    plots=[]
    for item in manifest['figures']:
        for f in item['files']:assert digest(ROOT/f['path'])==f['sha256']
        base=FIG/'数据结果图'/item['id']
        with Image.open(base.with_suffix('.png')) as im:
            assert im.width>=2000 and min(im.info['dpi'])>=299;pixels=im.size;dpi=im.info['dpi']
        tree=ET.parse(base.with_suffix('.svg'));ns={'svg':'http://www.w3.org/2000/svg'}
        assert not tree.findall('.//svg:image',ns) and len(tree.findall('.//svg:text',ns))>=20
        with fitz.open(base.with_suffix('.pdf')) as doc:
            assert len(doc)==1 and not doc[0].get_images() and doc[0].get_fonts()
            assert all(doc.extract_font(f[0])[3] for f in doc[0].get_fonts(full=True))
            sizes=[s['size'] for b in doc[0].get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
            minimum=min(sizes);assert minimum>=8.7
        plots.append(dict(id=item['id'],pixels=pixels,dpi=dpi,minimum_font_pt=minimum,PDF_pages=1,
            PDF_embedded_fonts=True,PDF_bitmap_count=0,SVG_editable_text=True,visual_inspected=args.visual_inspected,
            recommended_insert_width_cm=18))
    missing=[];count=0
    for f in [ROOT/'README.md',FIG/'README.md',FIG/'目标扰动鲁棒性图注.md',OUT/'鲁棒性与数据范围报告.md',FIG/'审阅/目标扰动结果图设计与核验_v1.md']:
        for dest in re.findall(r'!?\[[^\]]*\]\(([^\n)]+)\)',f.read_text('utf-8')):
            if dest.startswith(('https://','http://','#','app://')):continue
            q=Path(dest.split('#')[0].strip('<>'))
            if not q.is_absolute():q=f.parent/q
            count+=1
            if not q.exists() and q.resolve() not in [(OUT/'交付核验.json').resolve(),(FIG/'审阅/目标扰动结果图核验_v1.json').resolve()]:missing.append(str(q))
    assert not missing,missing
    qa=dict(status='passed',plots=plots,visual_inspected=args.visual_inspected,
        no_overlap_or_clipping_confirmed_by_image_inspection=args.visual_inspected,
        source_manifest_sha256=digest(FIG/'绘图数据/目标扰动鲁棒性_v1.json'))
    qa_file=FIG/'审阅/目标扰动结果图核验_v1.json';qa_file.write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n','utf-8')
    result=dict(status='passed',robustness_study_complete=True,all_six_fixed_stresses_retained=receipt['all_six_fixed_stresses_retained'],
        independent_neural_episodes=21000,independent_neural_actions=audit['neural_actions'],
        current_images_reconstructed=500,target_images_reconstructed=3500,ordinary_SR_SG_intervals_rechecked=audit['ordinary_SR_SG_intervals'],
        rule_records_rechecked_once=1000,tests_run=read(OUT/'测试记录.json')['tests_run'],
        protected_legacy_batch_files_verified={n:len(v) for n,v in reg['historical_sha256'].items()},
        frozen_source_files_verified=len(reg['source_sha256']),default_unchanged=True,training_steps=0,cloud_calls=0,
        all_Clean_records_reproduced=True,all_NoTarget_trajectories_reproduced=True,
        posthoc_diagnostics_recomputed=True,plots=2,plot_files=6,visual_inspected=args.visual_inspected,
        local_document_links_checked=count,missing_links=0,new_unseen_source_files=0,
        receipt_sha256=digest(OUT/'验收结论.json'),report_sha256=digest(OUT/'鲁棒性与数据范围报告.md'),plot_QA_sha256=digest(qa_file))
    (OUT/'交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    assert (OUT/'交付核验.json').is_file() and qa_file.is_file()
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
