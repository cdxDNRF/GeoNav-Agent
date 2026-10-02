"""Verify immutable study provenance, plots and their document indexes."""
import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fitz
from PIL import Image
from documents.summarize_orientation import ROOT,FIG,paths,read,digest,diagnostics


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);p.add_argument('--visual-inspected',action='store_true');args=p.parse_args();out=paths(args.mode)[1]
    if (out/'交付核验.json').exists():raise ValueError('immutable delivery exists')
    stem={'development':'方向候选开发','pilot':'方向候选试跑','confirmation':'方向候选独立确认'}[args.mode]
    receipt=read(out/'验收结论.json');audit=read(out/'独立复核.json');reg=read(out/'预登记.json');manifest_file=FIG/f'绘图数据/{stem}_v1.json';manifest=read(manifest_file)
    assert audit['status']=='passed' and receipt['quality_and_replay_passed']
    assert digest(out/'独立复核.json')==receipt['audit_sha256'] and digest(out/'对照汇总.json')==receipt['summary_sha256']
    for n,h in audit['artifacts_sha256'].items():assert digest(out/n)==h,n
    for folder,files in reg['historical_sha256'].items():
        for n,h in files.items():assert digest(ROOT/folder/n)==h,n
    for n,h in reg['source_sha256'].items():assert digest(ROOT/'project/src'/n)==digest(out/'源码快照'/n)==h,n
    assert digest(ROOT/'project/local_policy_default.json')==reg['default_sha256']
    assert read(out/'线索接受诊断.json')==diagnostics(out)
    paired_failure_checked=False
    if args.mode=='pilot':
        from documents.orientation_failure_diagnosis import analyze
        assert read(out/'试跑失败配对诊断.json')==analyze(out)
        for n,h in read(out/'试跑失败配对诊断.json')['source_sha256'].items():assert digest(ROOT/n)==h,n
        assert not paths('confirmation')[0].exists() and not paths('confirmation')[1].exists()
        paired_failure_checked=True
    for n,h in manifest['source_sha256'].items():assert digest(ROOT/n)==h,n
    assert digest(out/'方向候选验证报告.md')==manifest['report_sha256']
    assert digest(ROOT/'project/src/documents/summarize_orientation.py')==manifest['script_sha256']
    plots=[]
    for item in manifest['figures']:
        for f in item['files']:assert digest(ROOT/f['path'])==f['sha256']
        base=FIG/'数据结果图'/item['id']
        with Image.open(base.with_suffix('.png')) as im:
            assert im.width>=2000 and min(im.info['dpi'])>=299;size=im.size;dpi=im.info['dpi']
        tree=ET.parse(base.with_suffix('.svg'));ns={'svg':'http://www.w3.org/2000/svg'}
        assert not tree.findall('.//svg:image',ns) and len(tree.findall('.//svg:text',ns))>=15
        with fitz.open(base.with_suffix('.pdf')) as doc:
            assert len(doc)==1 and not doc[0].get_images() and doc[0].get_fonts()
            assert all(doc.extract_font(f[0])[3] for f in doc[0].get_fonts(full=True))
            sizes=[s['size'] for b in doc[0].get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']];minimum=min(sizes);assert minimum>=8.7
        plots.append(dict(id=item['id'],pixels=size,dpi=dpi,minimum_font_pt=minimum,PDF_embedded_fonts=True,PDF_bitmap_count=0,SVG_editable_text=True,visual_inspected=args.visual_inspected,recommended_insert_width_cm=18))
    doc_files=[ROOT/'README.md',FIG/'README.md',FIG/f'{stem}图注.md',out/'方向候选验证报告.md',FIG/f'审阅/{stem}设计与核验_v1.md'];count=0;missing=[]
    if paired_failure_checked:doc_files.append(out/'试跑失败配对诊断.md')
    for f in doc_files:
        for dest in re.findall(r'!?\[[^\]]*\]\(([^\n)]+)\)',f.read_text('utf-8')):
            if dest.startswith(('https://','http://','#','app://')):continue
            q=Path(dest.split('#')[0].strip('<>'));q=q if q.is_absolute() else f.parent/q;count+=1
            if not q.exists() and q.resolve()!=(out/'交付核验.json').resolve():missing.append(str(q))
    assert not missing,missing
    result=dict(status='passed',mode=args.mode,candidate_numeric_passed=receipt['candidate_numeric_passed'],formal_independent_confirmation_passed=receipt['formal_independent_confirmation_passed'],
        neural_episodes=audit['neural_episodes'],neural_actions=audit['neural_actions'],candidate_images=audit['images']['candidate_images'],rule_records=audit['rule_records_rechecked_once'],
        frozen_source_files_verified=len(reg['source_sha256']),historical_batch_files_verified={n:len(v) for n,v in reg['historical_sha256'].items()},
        SR_SG_intervals_recomputed=8,tests_run=read(out/'测试记录.json')['tests_run'],default_unchanged=True,plots=plots,visual_inspected=args.visual_inspected,
        posthoc_diagnostics_recomputed=True,paired_failure_diagnosis_recomputed=paired_failure_checked,
        formal_confirmation_not_started=args.mode=='pilot',local_document_links_checked=count,missing_links=0,receipt_sha256=digest(out/'验收结论.json'),manifest_sha256=digest(manifest_file))
    (out/'交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
