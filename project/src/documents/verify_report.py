"""Reproducible non-visual checks; never treats these as visual acceptance."""
from pathlib import Path
import json,zipfile,hashlib
from lxml import etree
ROOT=Path(__file__).resolve().parents[3]; BASE=ROOT/'选题报告相关'; Q=BASE/'校正审阅'
SOURCE=BASE/'选题报告_基于主动探索的视觉地理定位方法设计与实现.docx'
FINAL=SOURCE.with_name(SOURCE.stem+'_校正版.docx')
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
def xml(p):
    with zipfile.ZipFile(p) as z:return etree.fromstring(z.read('word/document.xml'))
def texts(tree):return tree.xpath('//w:t/text()',namespaces=NS)
def signature(node):return (node.tag,sorted(node.attrib.items()),[signature(c) for c in node])
a=xml(SOURCE); b=xml(FINAL)
audit=xml(Q/'选题报告_文字修订审计.docx')
for node in audit.xpath('//w:ins',namespaces=NS):node.getparent().remove(node)
old=''.join(audit.xpath('//w:t/text() | //w:delText/text()',namespaces=NS))
manifest=json.loads((Q/'original-sha256.json').read_text(encoding='utf-8'))
checks={
'original_files_unchanged':{k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items()},
'audit_reverted_text_equals_original':old==''.join(texts(a)),
'no_tracked_changes_in_final':not b.xpath('//w:ins | //w:del',namespaces=NS),
'section_properties_preserved':signature(a.xpath('//w:sectPr',namespaces=NS)[0])==signature(b.xpath('//w:sectPr',namespaces=NS)[0]),
'paragraph_count_source':len(a.xpath('//w:p',namespaces=NS)),
'paragraph_count_final':len(b.xpath('//w:p',namespaces=NS)),
'original_cover_text_preserved': ''.join(a.xpath('//w:p[position()<25]//w:t/text()',namespaces=NS))[:100]==''.join(b.xpath('//w:t/text()',namespaces=NS))[:100],
'correction_records':len(json.loads((Q/'corrections.json').read_text(encoding='utf-8'))),
}
assert checks['audit_reverted_text_equals_original']
assert checks['no_tracked_changes_in_final']
assert checks['section_properties_preserved']
assert all(checks['original_files_unchanged'].values())
assert checks['paragraph_count_source']-checks['paragraph_count_final']==6
(Q/'final-integrity.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
