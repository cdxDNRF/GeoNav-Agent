"""Export only the corrected copy in an isolated Word instance after LO failure.
Render every PDF page to PNG without opening/visually inspecting those images.
"""
from pathlib import Path
import json, hashlib, subprocess, os, sys
import fitz
import win32com.client
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'选题报告相关'; REVIEW=BASE/'校正审阅'
DOCX=BASE/'选题报告_基于主动探索的视觉地理定位方法设计与实现_校正版.docx'
PDF=DOCX.with_suffix('.pdf')
app=win32com.client.DispatchEx('Word.Application')
app.Visible=False; app.DisplayAlerts=0
try:
    doc=app.Documents.Open(str(DOCX),ReadOnly=True,AddToRecentFiles=False,Visible=False)
    doc.Repaginate()
    positions=[]
    for i in range(1,26):
        p=doc.Paragraphs(i)
        positions.append({'i':i,'text':p.Range.Text,'y':p.Range.Information(6),'line_spacing':p.Format.LineSpacing})
    (REVIEW/'cover-positions.json').write_text(json.dumps(positions,ensure_ascii=False,indent=2),encoding='utf-8')
    doc.ExportAsFixedFormat(str(PDF),17,OpenAfterExport=False)
    doc.Close(False)
finally:
    app.Quit(False)
pdf=fitz.open(PDF)
render=REVIEW/'pages';render.mkdir(exist_ok=True)
for p in render.glob('page-*.png'):p.unlink()
for i,p in enumerate(pdf):p.get_pixmap(matrix=fitz.Matrix(1.8,1.8),alpha=False).save(render/f'page-{i+1:02d}.png')
(REVIEW/'corrected-pdf-text.txt').write_text('\n\n'.join(f'=== PAGE {i+1} ===\n'+p.get_text() for i,p in enumerate(pdf)),encoding='utf-8')
status={'pdf_pages':len(pdf),'pdf_bytes':PDF.stat().st_size,'converter':'Microsoft Word isolated COM instance; LibreOffice MSI failed 1603 (insufficient administrator privileges)','visual_gate':'pending documents:visual-judge; author did not inspect rendered pages','png_paths':[str(render/f'page-{i+1:02d}.png') for i in range(len(pdf))]}
(REVIEW/'quality-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(status,ensure_ascii=False,indent=2))
