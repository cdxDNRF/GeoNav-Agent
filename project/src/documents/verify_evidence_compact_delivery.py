"""Verify report links and exported scientific artifacts after visual inspection."""
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import fitz
from eval.evidence_compact_g import ROOT,OUT,G,read,write,digest


def main():
    report=OUT/'简化接口与闭环验证报告.md';links=re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8'))
    for link in links:
        if not (OUT/link).is_file():raise ValueError('broken report link: '+link)
    manifest=ROOT/'绘图/绘图数据/证据账本简化接口闭环_v2.json';item=read(manifest)['figures'][0]
    for obj in item['files']:
        if digest(ROOT/obj['path'])!=obj['sha256']:raise ValueError('figure hash mismatch')
    name=item['id'];base=ROOT/'绘图/数据结果图'
    with Image.open(base/(name+'.png')) as img:
        size=img.size;dpi=img.info.get('dpi')
        if dpi is None or min(dpi)<299:raise ValueError('PNG dpi')
    svg=ET.parse(base/(name+'.svg')).getroot();tags=[n.tag.rsplit('}',1)[-1] for n in svg.iter()]
    if 'image' in tags or 'text' not in tags:raise ValueError('SVG must contain editable vector text, no raster')
    doc=fitz.open(base/(name+'.pdf'));font_lengths=[]
    if len(doc)!=1 or doc[0].get_images():raise ValueError('PDF one vector page required')
    for font in doc[0].get_fonts(full=True):
        data=doc.extract_font(font[0]);font_lengths.append(len(data[-1]))
    if not font_lengths or any(n==0 for n in font_lengths):raise ValueError('PDF fonts must be embedded')
    result=dict(passed=True,report_links=len(links),PNG_pixels=size,PNG_dpi=dpi,PDF_pages=len(doc),
                PDF_embedded_fonts=len(font_lengths),SVG_editable_text=True,SVG_raster_images=0,
                source_manifest_sha256=digest(manifest),G_verdict_sha256=digest(G/'验收结论.json'),
                visual_review_record='绘图/审阅/证据账本简化接口闭环设计与核验_v2.md')
    if not (ROOT/result['visual_review_record']).is_file():raise ValueError('actual PNG review required first')
    write(OUT/'交付核验.json',result);print(result)


if __name__=='__main__':main()
