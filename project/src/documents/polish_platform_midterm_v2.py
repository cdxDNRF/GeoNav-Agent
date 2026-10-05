"""Preserve v1; repair inherited Title border and title wrapping in a new v2."""
from pathlib import Path
import hashlib
import json
from docx import Document
from docx.shared import Pt,RGBColor
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[3]
FOLDER=ROOT/'总结报告相关/平台交付_v1'
SOURCE=FOLDER/'主动探索视觉地理定位中期平台交付总结_v1.docx'
OUT=FOLDER/'主动探索视觉地理定位中期平台交付总结_v2.docx'
if OUT.exists():raise ValueError('v2已存在，不覆盖')
d=Document(SOURCE)
for style in d.styles:
    for border in style.element.xpath('.//w:pBdr'):border.getparent().remove(border)
p=d.paragraphs[0];p.text='主动探索视觉地理定位\n中期平台交付总结';p.style='Title'
for run in p.runs:
    run.font.name='Microsoft YaHei';run.font.size=Pt(22);run.font.bold=True
    run.font.color.rgb=RGBColor(0,0,0);run.font.underline=False
    fonts=run._element.get_or_add_rPr().rFonts
    for name in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):
        fonts.attrib.pop(qn('w:'+name),None)
    fonts.set(qn('w:eastAsia'),'微软雅黑')
d.save(OUT)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
(FOLDER/'排版审阅/格式修订说明_v2.json').write_text(json.dumps(dict(source_sha256=sha(SOURCE),output_sha256=sha(OUT),
    changes=['remove inherited paragraph borders in styles','natural two-line black title'],
    body_unchanged=True,original_user_document_modified=False),ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT)
