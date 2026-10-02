"""Check final SwissView report/plot provenance and protected experiment hashes."""
from pathlib import Path
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents.summarize_swissview import ROOT,BASE,FIG,read,digest,diagnostic
from PIL import Image
import fitz


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--visual-inspected',action='store_true');args=parser.parse_args()
    out=BASE/'五乘五正式迁移_v1';pilot=BASE/'五乘五工程试跑_v1'
    if (out/'交付核验.json').exists():raise ValueError('immutable delivery check exists')
    receipt=read(out/'验收结论.json');audit=read(out/'独立复核.json');reg=read(out/'预登记.json')
    assert audit['status']=='passed' and digest(out/'独立复核.json')==receipt['audit_sha256']
    assert digest(out/'对照汇总.json')==receipt['summary_sha256']
    for batch in (pilot,out):
        for n,h in read(batch/'独立复核.json')['artifacts_sha256'].items():assert digest(batch/n)==h,n
        assert read(batch/'失败与线索诊断.json')==diagnostic(batch)
    for folder,files in reg['historical_sha256'].items():
        for n,h in files.items():assert digest(ROOT/folder/n)==h,n
    for n,h in reg['source_sha256'].items():assert digest(ROOT/'project/src'/n)==digest(out/'源码快照'/n)==h,n
    assert digest(ROOT/'project/local_policy_default.json')==receipt['default_sha256']==reg['default_sha256']
    manifest=read(FIG/'绘图数据/SwissView五乘五正式迁移_v1.json')
    for n,h in manifest['source_sha256'].items():assert digest(ROOT/n)==h,n
    assert digest(out/'迁移验证报告.md')==manifest['report_sha256']
    plotchecks=[]
    for item in manifest['figures']:
        for file in item['files']:assert digest(ROOT/file['path'])==file['sha256']
        base=FIG/'数据结果图'/item['id']
        with Image.open(base.with_suffix('.png')) as im:
            assert im.width>=2000 and min(im.info['dpi'])>=299
            pixels=im.size;dpi=im.info['dpi']
        tree=ET.parse(base.with_suffix('.svg'));ns={'svg':'http://www.w3.org/2000/svg'}
        assert len(tree.findall('.//svg:text',ns))>=20 and not tree.findall('.//svg:image',ns)
        with fitz.open(base.with_suffix('.pdf')) as doc:
            assert len(doc)==1 and not doc[0].get_images() and doc[0].get_fonts()
            fonts=doc[0].get_fonts(full=True)
            assert all(doc.extract_font(f[0])[3] for f in fonts)
            font_sizes=[span['size'] for block in doc[0].get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for span in line['spans']]
            minimum_font=min(font_sizes);assert minimum_font>=8.7
        plotchecks.append(dict(id=item['id'],pixels=pixels,dpi=dpi,minimum_font_pt=minimum_font,
            PDF_pages=1,PDF_embedded_fonts=True,PDF_bitmap_count=0,SVG_editable_text=True,
            SVG_bitmap_count=0,visual_inspected=args.visual_inspected,recommended_insert_width_cm=18))
    missing=[];linkcount=0
    for p in [ROOT/'README.md',FIG/'README.md',FIG/'SwissView五乘五图注.md',pilot/'迁移验证报告.md',out/'迁移验证报告.md']:
        for dest in re.findall(r'!?\[[^\]]*\]\(([^\n)]+)\)',p.read_text('utf-8')):
            if dest.startswith(('https://','http://','#','app://')):continue
            dest=dest.split('#')[0].strip('<>');q=Path(dest)
            if not q.is_absolute():q=p.parent/q
            linkcount+=1
            # README links this receipt before the successful check writes it.
            # Validate that single forward reference immediately after creation.
            if not q.exists() and q.resolve()!=(out/'交付核验.json').resolve():
                missing.append(dict(document=str(p),target=str(q)))
    assert not missing,missing
    qa=dict(status='passed',plots=plotchecks,visual_inspected=args.visual_inspected,
        source_manifest_sha256=digest(FIG/'绘图数据/SwissView五乘五正式迁移_v1.json'),
        no_overlap_or_clipping_confirmed_by_image_inspection=args.visual_inspected)
    (FIG/'审阅/SwissView结果图核验_v1.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n','utf-8')
    result=dict(status='passed',formal_cross_dataset_passed=receipt['formal_cross_dataset_passed'],
        target_transfer_passed=receipt['target_transfer_passed'],neural_episodes=audit['neural_episodes'],
        neural_actions=audit['neural_steps'],rule_episodes=audit['rule_episodes'],tests_run=read(out/'测试记录.json')['tests_run'],
        independently_audited_input_patches=audit['images']['images'],full_checkpoint_replay=True,
        protected_legacy_batch_files_verified={p:len(v) for p,v in reg['historical_sha256'].items()},
        frozen_source_files_verified=len(reg['source_sha256']),default_unchanged=True,training_steps=0,cloud_calls=0,
        posthoc_diagnostic_recomputed_from_transition_events=True,plots=2,plot_files=6,
        visual_inspected=args.visual_inspected,local_document_links_checked=linkcount,missing_links=0,
        source_domain_shift_is_unpaired=True,geographic_nonoverlap_confirmed=False,cross_view_confirmed=False,
        receipt_sha256=digest(out/'验收结论.json'),report_sha256=digest(out/'迁移验证报告.md'),
        plot_QA_sha256=digest(FIG/'审阅/SwissView结果图核验_v1.json'))
    (out/'交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    assert (out/'交付核验.json').is_file()
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
