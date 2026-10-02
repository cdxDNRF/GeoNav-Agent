"""Verify immutable training/navigation evidence, posthoc tables and publication exports."""
import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fitz
from PIL import Image
from documents.summarize_rotation_negative import ROOT,FIG,TRAIN,paths,read,digest,diagnostics


def provenance(out):
    r=read(out/'验收结论.json');a=read(out/'独立复核.json');g=read(out/'预登记.json');assert r['quality_and_replay_passed'] and a['status']=='passed' and digest(out/'独立复核.json')==r['audit_sha256']
    for n,h in a['artifacts_sha256'].items():assert digest(out/n)==h,n
    for root,files in g['historical_sha256'].items():
        for n,h in files.items():assert digest(ROOT/root/n)==h,n
    for n,h in g['source_sha256'].items():assert digest(ROOT/'project/src'/n)==digest(out/'源码快照'/n)==h,n
    for n,h in g.get('read_only_data_sha256',{}).items():assert digest(ROOT/n)==h,n
    cache=ROOT/g.get('fit_cache_root',g.get('data_root',''))
    for n,h in read(out/'特征冻结结束.json')['files_sha256'].items():assert digest(cache/n)==h,n
    assert digest(ROOT/'project/local_policy_default.json')==g['default_sha256'];return r,a,g


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);p.add_argument('--visual-inspected',action='store_true');args=p.parse_args();out=paths(args.mode)[1]
    if (out/'交付核验.json').exists():raise ValueError('immutable delivery exists')
    r,a,g=provenance(out);tr,ta,tg=provenance(TRAIN);assert digest(out/'对照汇总.json')==r['summary_sha256'];assert read(out/'接受与失败诊断.json')==diagnostics(out)
    stem={'development':'旋转负样本开发','pilot':'旋转负样本试跑','confirmation':'旋转负样本正式'}[args.mode];manifest_file=FIG/f'绘图数据/{stem}_v1.json';manifest=read(manifest_file)
    for n,h in manifest['source_sha256'].items():assert digest(ROOT/n)==h,n
    assert digest(out/'旋转负样本对照报告.md')==manifest['report_sha256'];assert digest(ROOT/'project/src/documents/summarize_rotation_negative.py')==manifest['script_sha256'];plots=[]
    for item in manifest['figures']:
        for f in item['files']:assert digest(ROOT/f['path'])==f['sha256']
        base=FIG/'数据结果图'/item['id']
        with Image.open(base.with_suffix('.png')) as im:assert im.width>=2000 and min(im.info['dpi'])>=299;size=im.size;dpi=im.info['dpi']
        tree=ET.parse(base.with_suffix('.svg'));ns={'svg':'http://www.w3.org/2000/svg'};assert not tree.findall('.//svg:image',ns) and len(tree.findall('.//svg:text',ns))>=15
        with fitz.open(base.with_suffix('.pdf')) as pdf:
            assert len(pdf)==1 and not pdf[0].get_images() and pdf[0].get_fonts() and all(pdf.extract_font(f[0])[3] for f in pdf[0].get_fonts(full=True));sizes=[s['size'] for b in pdf[0].get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']];minimum=min(sizes);assert minimum>=8.7
        plots.append(dict(id=item['id'],pixels=size,dpi=dpi,minimum_font_pt=minimum,PDF_embedded_fonts=True,PDF_bitmap_count=0,SVG_editable_text=True,visual_inspected=args.visual_inspected,recommended_insert_width_cm=18))
    docs=[ROOT/'README.md',FIG/'README.md',FIG/f'{stem}图注.md',FIG/f'审阅/{stem}设计与核验_v1.md',out/'旋转负样本对照报告.md'];count=0;missing=[]
    for f in docs:
        for dest in re.findall(r'!?\[[^\]]*\]\(([^\n)]+)\)',f.read_text('utf-8')):
            if dest.startswith(('https://','http://','#','app://')):continue
            q=Path(dest.split('#')[0].strip('<>'));q=q if q.is_absolute() else f.parent/q;count+=1
            if not q.exists() and q.resolve()!=(out/'交付核验.json').resolve():missing.append(str(q))
    assert not missing,missing;forbidden=[]
    if not r['candidate_numeric_passed']:
        for mode in (['pilot','confirmation'] if args.mode=='development' else ['confirmation'] if args.mode=='pilot' else []):assert not paths(mode)[0].exists() and not paths(mode)[1].exists();forbidden.append(mode)
    result=dict(status='passed',mode=args.mode,candidate_numeric_passed=r['candidate_numeric_passed'],formal_independent_confirmation_passed=r['formal_independent_confirmation_passed'],
        trained_models=ta['models'],training_steps_recomputed=ta['optimizer_steps'],training_pair_uses_recomputed=ta['pair_uses'],diagnostic_predictions_recomputed=ta['diagnostic_predictions'],
        neural_episodes=a['neural_episodes'],neural_actions=a['neural_actions'],rule_records=a['rule_records_rechecked_once'],SR_SG_intervals_recomputed=10,tests_run=read(out/'测试记录.json')['tests_run'],
        plots=plots,visual_inspected=args.visual_inspected,posthoc_diagnostics_recomputed=True,local_document_links_checked=count,missing_links=0,default_unchanged=True,
        unstarted_following_stages=forbidden,receipt_sha256=digest(out/'验收结论.json'),manifest_sha256=digest(manifest_file))
    (out/'交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
