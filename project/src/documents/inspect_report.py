"""Read-only source inventory and page-numbered local evidence extraction."""
from pathlib import Path
import hashlib, json, zipfile
import fitz
from lxml import etree
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '选题报告相关/校正审阅'
OUT.mkdir(exist_ok=True)
source = ROOT / '选题报告相关/选题报告_基于主动探索的视觉地理定位方法设计与实现.docx'
ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(source) as z:
    tree=etree.fromstring(z.read('word/document.xml'))
    paras=[{'index':i,'text': ''.join(p.xpath('.//w:t/text()', namespaces=ns)), 'xml':etree.tostring(p,encoding='unicode')} for i,p in enumerate(tree.xpath('//w:body//w:p',namespaces=ns))]
    (OUT/'source-paragraphs.json').write_text(json.dumps(paras,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'source-text.txt').write_text('\n'.join(f"[{p['index']}] {p['text']}" for p in paras),encoding='utf-8')
    print('PARAGRAPHS',len(paras),'MEDIA',[x for x in z.namelist() if '/media/' in x])
    print('FONTS',sorted(set(tree.xpath('//w:rFonts/@w:eastAsia | //w:rFonts/@w:ascii',namespaces=ns))))
for p in (ROOT/'论文').glob('*.pdf'):
    d=fitz.open(p)
    text='\n\n'.join(f'=== PDF PAGE {i+1} ===\n'+pg.get_text() for i,pg in enumerate(d))
    (OUT/f'{p.stem}-pages.txt').write_text(text,encoding='utf-8')
    print(p.name,'PAGES',len(d))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,source.with_suffix('.pdf'), ROOT/'选题报告相关/build_report.js',ROOT/'选题报告相关/figure_tech_route.html',ROOT/'选题报告相关/figure_tech_route.png']}
(OUT/'original-sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
