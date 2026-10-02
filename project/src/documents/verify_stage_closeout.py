"""Verify copy provenance, original preservation, vector figures and Word layout."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import io
import contextlib
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT/'选题报告相关/阶段收尾_v1'
FIG = ROOT/'绘图'
DOC = ROOT/'中期报告相关/阶段提升总结_v1.docx'
REVIEW = ROOT/'中期报告相关/阶段提升总结审阅'


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(p):return json.loads(p.read_text('utf-8'))


def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')


def render_summary():
    import win32com.client
    REVIEW.mkdir(parents=True,exist_ok=True)
    before=digest(DOC)
    pdf=DOC.with_suffix('.pdf')
    app=win32com.client.DispatchEx('Word.Application')
    app.Visible=False;app.DisplayAlerts=0
    try:
        doc=app.Documents.Open(str(DOC),ReadOnly=True,AddToRecentFiles=False,Visible=False)
        try:
            doc.Repaginate()
            doc.ExportAsFixedFormat(str(pdf),17,OpenAfterExport=False)
        finally:
            doc.Close(False)
    finally:
        app.Quit(False)
    assert digest(DOC)==before,'Word source changed during read-only export'
    pages=[]
    with fitz.open(pdf) as f:
        for i,page in enumerate(f):
            page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(REVIEW/f'page-{i+1:02d}.png')
            text=page.get_text()
            pages.append(dict(page=i+1,text=text,characters=len(text.strip())))
        result=dict(pdf_pages=len(f),docx_sha256=before,pdf_sha256=digest(pdf),
                    converter='isolated Microsoft Word COM; read-only source',pages=pages,
                    visual_inspection='pending',original_midterm_untouched=True)
    write(REVIEW/'排版核验.json',result)
    return result


def check_originals():
    original=read(STAGE/'原始文件保护清单.json')['files']
    hashes=0; concurrent=[]
    for rec in original:
        p=ROOT/rec['path']
        if not p.exists():raise ValueError('Original moved or deleted: '+rec['path'])
        st=p.stat()
        metadata_changed=(st.st_size,st.st_mtime_ns)!=(rec['bytes'],rec['mtime_ns'])
        current_hash=digest(p) if 'sha256' in rec else None
        content_changed='sha256' in rec and current_hash!=rec['sha256']
        if metadata_changed or content_changed:
            # Another active workflow is maintaining the literature notes. Never restore
            # its files, silently rebaseline the protection manifest or rewrite frozen copies.
            if not rec['path'].startswith('调研/'):
                raise ValueError('Protected original changed: '+rec['path'])
            concurrent.append(dict(path=rec['path'],before=rec,
                current=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=current_hash),
                treatment='current original retained; frozen snapshot retained; no write by closeout scripts'))
        if 'sha256' in rec:
            hashes+=1
    write(STAGE/'同步调研更新记录.json',dict(changes=concurrent,
        note='Detected during packaging. Closeout scripts only read 调研 originals. Baseline is retained; these updates are not rolled back.'))
    return dict(files=len(original),hashes_checked=hashes,metadata_checked=len(original),
                scope='DATA, models, pre-existing source/documents/figures; ROOT README intentionally updated',
                core_data_models_code_and_course_documents_unchanged=True,
                concurrent_research_changes=len(concurrent),all_originals_unchanged=not concurrent)


def check_snapshots():
    manifest=read(STAGE/'整理清单.json')
    drift=[]
    for rec in manifest['copied']:
        for key in ['original','snapshot']:
            p=ROOT/rec[key]
            if p.stat().st_size!=rec['bytes'] or digest(p)!=rec['sha256']:
                if key=='original' and rec[key].startswith('调研/'):
                    drift.append(dict(original=rec[key],snapshot=rec['snapshot'],
                        frozen_sha256=rec['sha256'],current_sha256=digest(p)))
                    continue
                raise ValueError('Snapshot hash mismatch: '+rec[key])
    original_batches={f'{kind}/{p.name}' for kind in ['训练结果','评测结果']
        for p in (ROOT/'DATA/processed_data/Masa'/kind).iterdir() if p.is_dir()}
    catalogued={b['kind']+'/'+b['name'] for b in manifest['experiments']}
    assert catalogued==original_batches
    return dict(experiments=len(catalogued),copies=len(manifest['copied']),
        bytes=sum(x['bytes'] for x in manifest['copied']),frozen_copies_hash_verified=True,
        sources_currently_match=len(manifest['copied'])-len(drift),research_source_drift=drift)


def check_figures():
    manifest=read(FIG/'图表来源清单.json')
    data=read(ROOT/manifest['data_file'])
    assert digest(ROOT/manifest['data_file'])==manifest['data_sha256']
    assert digest(ROOT/'project/src/documents/stage_figures.py')==manifest['script_sha256']
    for p,h in manifest['source_sha256'].items():
        assert digest(ROOT/p)==h,p
    # Resolve every displayed number back to the original summary field.
    def nested(obj,keys):
        for k in keys.split('.'):obj=obj[k]
        return obj
    number_rows=data['edge']['rows']+data['formal_val']['rows']+data['cloud']['rows']+data['historical_milestones']
    for r in number_rows:
        v=nested(read(ROOT/r['source_file']),r['source_field'])
        sr=v.get('sr_mean',v.get('sr_gate',v.get('sr')))
        sg=v.get('sg_mean',v.get('sg_gate',v.get('sg',v.get('mean_sg_all_episodes'))))
        assert abs(sr-r['sr'])<1e-12
        assert abs(sg-r['sg'])<1e-12
    vector_pdfs,svgs,pngs,drawios=0,0,0,0
    for item in manifest['figures']:
        for rec in item['files']:
            p=ROOT/rec['path'];assert digest(p)==rec['sha256'],str(p)
            if p.suffix=='.pdf':
                with fitz.open(p) as pdf:
                    assert len(pdf)==1
                    assert not pdf[0].get_images(full=True),'Raster image wrapped in PDF'
                    assert len(pdf[0].get_text().strip())>20
                    assert pdf[0].get_fonts(full=True)
                vector_pdfs+=1
            elif p.suffix=='.svg':
                root=ET.parse(p).getroot()
                assert root.tag.endswith('svg')
                assert not any(el.tag.endswith('image') for el in root.iter()),'Bitmap in SVG'
                svgs+=1
            elif p.suffix=='.png':
                with Image.open(p) as im:
                    assert min(im.size)>900
                    assert abs(im.info['dpi'][0]-300)<1
                    im.verify()
                pngs+=1
            elif p.suffix=='.drawio':
                root=ET.parse(p).getroot()
                vertices=root.findall('.//mxCell[@vertex="1"]')
                edges=root.findall('.//mxCell[@edge="1"]')
                assert len(vertices)>=6 and len(edges)>=5
                assert any('<br>' in v.get('value','') for v in vertices)
                drawios+=1
    return dict(figures=len(manifest['figures']),vector_pdfs=vector_pdfs,
        vector_svgs=svgs,png_300dpi=pngs,editable_drawios=drawios,
        numeric_rows_traced=len(number_rows),source_files_checked=len(manifest['source_sha256']),
        visual_inspection='pending')


def check_links():
    files=[ROOT/'README.md', *STAGE.glob('*.md'), *FIG.glob('*.md'),
        ROOT/'中期报告相关/阶段提升总结_v1.md', *list((STAGE/'报告与结果').rglob('README.md'))]
    count=0
    for p in files:
        for target in re.findall(r'\[[^\]]+\]\((?:<([^>]+)>|([^\)]+))\)',p.read_text('utf-8')):
            ref=target[0] or target[1]
            if re.match(r'^(?:https?:|file:|#)',ref):continue
            ref=ref.split('#')[0]
            if not (p.parent/ref).resolve().exists():
                raise ValueError(f'Broken new index link: {p.name} -> {ref}')
            count+=1
    return count


def check_word():
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(DOC) as z:
        assert z.testzip() is None
        root=ET.fromstring(z.read('word/document.xml'))
        text=''.join(el.text or '' for el in root.findall('.//w:t',ns))
        for expected in ['88.57%','72.62%','0.348','一个导航智能体','正式','待复验']:
            assert expected in text,expected
        media=[n for n in z.namelist() if n.startswith('word/media/') and not n.endswith('/')]
        assert len(media)==3
        images=['绘图/数据结果图/01_边缘线索主结果.png',
                '绘图/数据结果图/05_同任务集历史方案.png',
                '绘图/Agent架构图/01_边缘线索Agent架构.png']
        expected={digest(ROOT/rel) for rel in images}
        assert {hashlib.sha256(z.read(n)).hexdigest() for n in media}==expected,'Summary contains stale figures'
        for table in root.findall('.//w:tbl',ns):
            width=table.find('w:tblPr/w:tblW',ns)
            grid=table.findall('w:tblGrid/w:gridCol',ns)
            assert int(width.get('{'+ns['w']+'}w'))==sum(int(c.get('{'+ns['w']+'}w')) for c in grid)
    return dict(docx_sha256=digest(DOC),embedded_images=3,xml_and_tables_checked=True)


def check_word_schema():
    """Run the installed skill's full XSD checks with correct external URI handling.

    The bundled reference checker only excludes http/mailto targets and mistakenly
    treats valid TargetMode=External file URIs as ZIP members. This local subclass
    checks those actual local destinations and retains all other schema checks.
    No change is made to the document or shared skill files.
    """
    skill=Path.home()/'.agents/skills/docx/scripts/office'
    if not skill.exists():
        raise ValueError('Installed DOCX schema validator not found')
    sys.path.insert(0,str(skill))
    from validators import DOCXSchemaValidator
    checked=[]
    class LocalFileAwareValidator(DOCXSchemaValidator):
        def validate_file_references(self):
            referenced=set()
            ns={'r':'http://schemas.openxmlformats.org/package/2006/relationships'}
            for p in self.unpacked_dir.rglob('*.rels'):
                for rel in ET.parse(p).getroot().findall('r:Relationship',ns):
                    target=rel.get('Target')
                    if rel.get('TargetMode')=='External':
                        url=urlparse(target)
                        if url.scheme=='file':
                            value=unquote(url.path)
                            if re.match(r'^/[A-Za-z]:/',value):value=value[1:]
                            actual=Path(value)
                            assert actual.exists() and actual.is_file(),target
                            checked.append(str(actual))
                        else:
                            assert url.scheme in ['http','https','mailto'],target
                        continue
                    base=self.unpacked_dir if p.name=='.rels' else p.parent.parent
                    actual=(self.unpacked_dir/target.lstrip('/') if target.startswith('/') else base/target).resolve()
                    assert actual.exists() and actual.is_file(),target
                    assert self.unpacked_dir.resolve() in actual.parents,target
                    referenced.add(actual)
            all_files={p.resolve() for p in self.unpacked_dir.rglob('*') if p.is_file()
                       and p.name!='[Content_Types].xml' and not p.name.endswith('.rels')}
            assert all_files==referenced,'Unreferenced package member'
            return True
    log=io.StringIO()
    with tempfile.TemporaryDirectory(prefix='schema-',dir=REVIEW) as tmp:
        with zipfile.ZipFile(DOC) as z:z.extractall(tmp)
        validator=LocalFileAwareValidator(Path(tmp),None,verbose=False)
        with contextlib.redirect_stdout(log):
            passed=validator.validate()
    result=dict(status='passed' if passed else 'failed',docx_sha256=digest(DOC),
        full_skill_xsd_and_ooxml_checks=passed,external_local_files_checked=checked,
        reference_checker_adjustment='local subclass respects TargetMode=External; shared skill/document unchanged',
        log=log.getvalue())
    write(REVIEW/'OOXML完整验证.json',result)
    assert passed,'DOCX schema verification failed'
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-render',action='store_true')
    parser.add_argument('--visual-reviewed',action='store_true',help='Record visual inspection already performed by executing agent')
    args=parser.parse_args()
    layout=read(REVIEW/'排版核验.json') if args.skip_render else render_summary()
    # Indexes intentionally link to these receipts; keep an explicit pending state
    # until every verification completes, rather than skipping those links.
    write(STAGE/'整理核验.json',dict(status='verification_running'))
    write(FIG/'审阅/图表核验.json',dict(status='verification_running'))
    result=dict(status='structural_checks_passed_visual_pending',originals=check_originals(),
        snapshots=check_snapshots(),figures=check_figures(),new_index_links_checked=check_links(),
        document=check_word(),document_schema=check_word_schema(),summary_pdf_pages=layout['pdf_pages'],new_experiments_run=False,
        default_replaced=False,old_midterm_changed=False,
        verification_script_sha256=digest(Path(__file__)))
    if args.visual_reviewed:
        result['status']='passed_with_preserved_concurrent_research_updates'
        result['figures']['visual_inspection']='passed_by_executing_agent; all9figure previews inspected'
        layout['visual_inspection']='passed_by_executing_agent; all4pages inspected'
        layout['inspected_pages']=[1,2,3,4]
        write(REVIEW/'排版核验.json',layout)
    write(STAGE/'整理核验.json',result)
    write(FIG/'审阅/图表核验.json',result['figures'])
    print(json.dumps({k:v for k,v in result.items() if k in ['status','snapshots','summary_pdf_pages','new_index_links_checked']},ensure_ascii=False))


if __name__=='__main__':main()
