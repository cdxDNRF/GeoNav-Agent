"""Isolated native Office export, followed by per-page PNGs for Windows QA."""
from pathlib import Path
import json,hashlib
import fitz
import win32com.client
import pywintypes

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'中期报告相关'
REVIEW=OUT/'排版审阅/2026-10-02_v2';TITLE='基于主动探索的视觉地理定位方法设计与实现'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def render(pdf,folder,prefix):
    pages=[];alltext=[]
    with fitz.open(pdf) as d:
        for i,page in enumerate(d):
            image=folder/f'{prefix}_页{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(image)
            text=page.get_text();alltext.append(text)
            pages.append(dict(page=i+1,path=image.relative_to(ROOT).as_posix(),characters=len(text),size_points=[page.rect.width,page.rect.height],
                first_text=text[:140],last_text=text[-140:]))
    (folder/f'{prefix}_全文校样.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(alltext)),'utf8')
    return pages


def main():
    docs=[(OUT/f'中期报告_{TITLE}.docx','中期报告'),(OUT/'阶段提升总结_v2.docx','阶段总结')];results={}
    for docx,prefix in docs:
        word=win32com.client.DispatchEx('Word.Application');word.Visible=False;word.DisplayAlerts=0
        try:
            before=digest(docx);doc=word.Documents.Open(str(docx),ReadOnly=True,AddToRecentFiles=False,Visible=False)
            try:doc.Repaginate();doc.ExportAsFixedFormat(str(docx.with_suffix('.pdf')),17,OpenAfterExport=False)
            finally:doc.Close(False)
            assert digest(docx)==before
            results[prefix]=dict(source_sha256=before,pdf=docx.with_suffix('.pdf').relative_to(ROOT).as_posix(),
                pages=render(docx.with_suffix('.pdf'),REVIEW,prefix),converter='Microsoft Word isolated read-only COM')
            print(dict(exported=prefix,pages=len(results[prefix]['pages'])),flush=True)
        finally:
            try:word.Quit(False)
            except (pywintypes.com_error,AttributeError):pass  # Server may exit after closing its last document.
    source=OUT/f'答辩材料/中期汇报_{TITLE}.pptx';before=digest(source);powerpoint=win32com.client.DispatchEx('PowerPoint.Application')
    try:
        pres=powerpoint.Presentations.Open(str(source),ReadOnly=True,Untitled=False,WithWindow=False)
        try:pres.SaveAs(str(source.with_suffix('.pdf')),32)
        finally:pres.Close()
    finally:
        try:powerpoint.Quit()
        except (pywintypes.com_error,AttributeError):pass
    assert digest(source)==before
    results['中期汇报']=dict(source_sha256=before,pdf=source.with_suffix('.pdf').relative_to(ROOT).as_posix(),
        pages=render(source.with_suffix('.pdf'),OUT/'答辩材料/排版审阅','中期汇报'),converter='Microsoft PowerPoint isolated COM,WithWindow=False')
    (REVIEW/'渲染收据.json').write_text(json.dumps(dict(artifacts=results,visual_review_pending=True,no_source_modified=True),ensure_ascii=False,indent=2)+'\n','utf8')
    print(dict(exported='中期汇报',pages=len(results['中期汇报']['pages'])),flush=True)


if __name__=='__main__':main()
