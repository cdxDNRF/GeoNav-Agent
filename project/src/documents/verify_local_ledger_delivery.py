"""Check published means, local links and vector exports after actual PNG review."""
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import fitz
from eval.local_ledger_expansion import ROOT,OUT,SEEDS,read,write,digest


def main():
    summary=read(OUT/'主对照/对照汇总.json')
    for arm in ('M0','M1'):
        jobs=[read(OUT/f'主对照/{arm}_s{s}_CueFull_结果.json') for s in SEEDS]
        actual=summary['arms'][arm]
        for field,name in [('sr','sr'),('mean_sg_all_episodes','sg')]:
            values=[r['metrics'][field] for r in jobs]
            if actual[name+'_by_seed']!=values or abs(actual[name+'_mean']-sum(values)/3)>1e-12:raise ValueError('published aggregate means')
        for bucket in ('by_distance','by_split'):
            for key in actual[bucket]:
                for field,name in [('sr','sr'),('mean_sg_all_episodes','sg')]:
                    expected=sum(r[bucket][key][field] for r in jobs)/3
                    if abs(actual[bucket][key][name]-expected)>1e-12:raise ValueError('published stratified means')
    report=OUT/'完整队列验证报告.md';links=re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8'))
    for link in links:
        if not (OUT/link).is_file():raise ValueError('broken report link '+link)
    manifest=ROOT/'绘图/绘图数据/零调用账本完整队列_v1.json';item=read(manifest)['figures'][0]
    for entry in item['files']:
        if digest(ROOT/entry['path'])!=entry['sha256']:raise ValueError('figure hash')
    base=ROOT/'绘图/数据结果图';name=item['id']
    with Image.open(base/(name+'.png')) as img:
        dims=img.size;dpi=img.info.get('dpi')
        if dpi is None or min(dpi)<299:raise ValueError('PNG resolution')
    tags=[x.tag.rsplit('}',1)[-1] for x in ET.parse(base/(name+'.svg')).getroot().iter()]
    if 'image' in tags or 'text' not in tags:raise ValueError('SVG must be vector with editable text')
    doc=fitz.open(base/(name+'.pdf'));fonts=doc[0].get_fonts(full=True)
    if len(doc)!=1 or doc[0].get_images() or not fonts or any(not doc.extract_font(f[0])[-1] for f in fonts):raise ValueError('PDF vector/embedded fonts')
    review=ROOT/'绘图/审阅/零调用账本完整队列设计与核验_v1.md'
    if not review.is_file():raise ValueError('actual visual review first')
    audit=read(OUT/'独立复核.json');verdict=read(OUT/'验收结论.json')
    if not audit['passed'] or verdict['audit_sha256']!=digest(OUT/'独立复核.json'):raise ValueError('completed independent audit required')
    result=dict(passed=True,aggregate_and_stratified_means_checked=True,report_links=len(links),PNG_pixels=dims,PNG_dpi=dpi,
        PDF_embedded_fonts=len(fonts),SVG_editable_text=True,SVG_raster_images=0,figure_manifest_sha256=digest(manifest),
        review_sha256=digest(review),audit_sha256=digest(OUT/'独立复核.json'),default_changed=False)
    write(OUT/'交付核验.json',result);print(result)


if __name__=='__main__':main()
