"""DOC009 cached native-PDF raster adapter and report/source integrity checks."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import shutil
from zipfile import ZipFile
from lxml import etree
from docx import Document
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'中期报告相关/中期成果更新_20261025'
SKILL=Path('C:/Users/DFWJ/.codex/plugins/cache/openai-primary-runtime/documents/26.1022.11436/skills/documents')
DEPS=Path('C:/Users/DFWJ/.cache/codex-runtimes/codex-primary-runtime/dependencies')
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)


def render(revision):
    doc=OUT/'临时'/f'中期报告候选_{revision}.docx';pdf=doc.with_suffix('.pdf')
    receipt=read(OUT/'核验'/f'报告原生导出_{revision}.json')
    assert receipt['passed'] and pdf.stat().st_size>0
    original_doc,original_pdf=sha(doc),sha(pdf)
    os.environ['TEMP']=str(OUT/'临时');os.environ['TMP']=str(OUT/'临时');tempfile.tempdir=str(OUT/'临时')
    os.environ['PATH']=str(DEPS/'native/poppler/Library/bin')+os.pathsep+os.environ.get('PATH','')
    spec=importlib.util.spec_from_file_location('packaged_docx_renderer',SKILL/'render_docx.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def native_pdf_adapter(doc_path,user_profile,convert_tmp_dir,stem,verbose):
        if Path(doc_path).resolve()!=doc.resolve():raise ValueError('cached PDF belongs to another document')
        return str(pdf),'WPS native export already complete;cached verified PDF used;no re-export'
    module.convert_to_pdf=native_pdf_adapter
    dest=OUT/'审阅'/f'报告_{revision}'
    assert not list(dest.glob('page-*.png')),'avoid overwriting a prior review'
    pages=module.rasterize(str(doc),str(dest),130,verbose=False,emit_pdf=False)
    assert len(pages)==receipt['pages']==len(PdfReader(pdf).pages)
    assert sha(doc)==original_doc and sha(pdf)==original_pdf
    put(OUT/'核验'/f'报告栅格化_{revision}.json',dict(passed=True,pages=len(pages),
        renderer=str(SKILL/'render_docx.py'),converter='cached WPS Writer native PDF adapter',
        missing_bundled_windows_LibreOffice=True,managed_code_modified=False,
        docx_sha256=original_doc,pdf_sha256=original_pdf,outputs=[Path(p).relative_to(ROOT).as_posix() for p in pages]))
    print('Native report rasterized:',len(pages),'pages')


def audit(revision):
    doc=OUT/'临时'/f'中期报告候选_{revision}.docx';pdf=doc.with_suffix('.pdf')
    facts=read(OUT/'核验/中期内容与证据.json');base=read(OUT/'核验/模板包与样式证据.json');old=read(OUT/'核验/旧稿与模板保护.json')
    for name,h in old.items():assert sha(ROOT/name)==h,name
    binding=read(OUT/'核验/输入绑定.json')
    for row in binding['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    original=Document(ROOT/'中期报告相关/中期报告_基于主动探索的视觉地理定位方法设计与实现.docx');new=Document(doc)
    for a,b in zip(original.tables[0].rows[:6],new.tables[0].rows[:6]):
        assert [c.text for c in a.cells]==[c.text for c in b.cells]
    with ZipFile(ROOT/base['reference']) as zr,ZipFile(doc) as zn:
        source=etree.fromstring(zr.read('word/document.xml'));target=etree.fromstring(zn.read('word/document.xml'))
        a=source.find('.//w:sectPr',NS);b=target.find('.//w:sectPr',NS)
        assert etree.tostring(a)==etree.tostring(b),'teacher section/page geometry changed'
        preserved=[]
        for name in zr.namelist():
            if name.endswith('/'):continue
            if name.startswith(('word/header','word/footer','word/media/')):
                assert zr.read(name)==zn.read(name),name;preserved.append(name)
        text=''.join(target.xpath('//w:t/text()',namespaces=NS))
        for required in ['一、设计目标与任务','二、完成情况','三、存在的主要问题','四、下一步需要完成的工作','2026 年 10 月 25 日','0/9','86.80%','68.04%','54.62%/20.27%','0.300','797,701','136,837','934,538']:
            assert required in text,required
        assert '【待填写】' not in text and '尚未完成' in text
    reader=PdfReader(pdf);pdf_text='\n'.join(p.extract_text() or '' for p in reader.pages)
    for required in ['86.80','68.04','934,538','66.67']:
        assert required in pdf_text,required
    assert len(reader.pages)==9
    put(OUT/'核验'/f'报告结构与数字复核_{revision}.json',dict(passed=True,old_artifacts_unchanged=len(old),
        teacher_sections_and_metadata_preserved=True,preserved_package_parts=preserved,page_count=len(reader.pages),
        source_binding_files=len(binding['files']),ground_navigation_claimed_passed=False,model_calls=0))
    print('Report structure/metrics/old artifacts verified')


def materials(_=None):
    """Promote the verified report and prepare a self-contained demo card."""
    dest=OUT/'交付'
    for suffix in ['docx','pdf']:
        source=OUT/'临时'/f'中期报告候选_r2.{suffix}';target=dest/f'中期报告.{suffix}'
        assert not target.exists()
        shutil.copyfile(source,target);assert sha(source)==sha(target)
    card='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GeoNav · 中期演示操作卡</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f7f6;color:#173f49;font-family:"Microsoft YaHei",sans-serif;line-height:1.8}main{max-width:1040px;margin:auto;background:white;padding:56px 64px}header{border-top:5px solid #087f83;padding-top:22px}small,.muted{color:#567178}h1{font-size:36px;line-height:1.4;margin:12px 0}h2{font-size:24px;margin:38px 0 14px}p{margin:10px 0}a{color:#087f83;text-decoration-thickness:1px;text-underline-offset:4px}nav{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}nav a,button{border:1px solid #bfd4d4;background:#edf5f4;padding:9px 16px;border-radius:4px;color:#173f49;font:inherit;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:left;padding:14px 12px;border-bottom:1px solid #dce6e6;vertical-align:top}th{background:#edf5f4}td:first-child{white-space:nowrap;color:#087f83;font-weight:bold}aside{border-left:4px solid #b16c1b;padding:12px 20px;background:#fff7ed;margin:24px 0}details{border-bottom:1px solid #dce6e6;padding:14px 0}summary{cursor:pointer;font-weight:bold}details p{color:#4e6870}figure{margin:20px 0}img{max-width:100%;height:auto;border:1px solid #dce6e6}figcaption{font-size:13px;color:#567178}footer{margin-top:40px;border-top:1px solid #dce6e6;padding-top:15px;font-size:13px;color:#567178}@media(max-width:650px){main{padding:28px 20px}h1{font-size:28px}table{font-size:13px}td,th{padding:9px 6px}}@media print{body,main{background:white}main{padding:0;max-width:none}nav,button,.screen-only{display:none}h2,summary{break-after:avoid}tr,details,aside{break-inside:avoid}figure{break-inside:avoid}a{color:inherit}footer{font-size:10px}}</style>
<main><header><small>GEONAV / MIDTERM / 2026.10.25</small><h1>三分钟，说明平台能力与研究边界</h1><p>当前工作台 4.13.1 · 中期展示 · 单导航Agent</p></header>
<nav><a href="http://127.0.0.1:8769/" target="_blank" rel="noopener">打开工作台</a><a href="http://127.0.0.1:8769/ground-walk" target="_blank" rel="noopener">人工连续观察</a><a href="http://127.0.0.1:8769/spatial-evidence?case=112&amp;method=Spatial&amp;seed=0&amp;arm=Full" target="_blank" rel="noopener">空间证据样例</a><button onclick="document.querySelectorAll('details').forEach(x=>x.open=true);window.print()">展开问答并打印</button></nav>
<aside><strong>开场一句话</strong><p>我们已完成航拍目标的轻量主动导航和可交互实验平台；地面照片目标已接入人工连续观察与感知证据查阅，可靠自主导航仍是下一阶段任务。</p></aside>
<h2>演示前检查</h2><p>用根目录“平台/启动平台.cmd”启动。影像与模型按项目绑定准备，启动器不自动下载或安装。关闭无关窗口，先打开工作台和本操作卡。当前端口为8769；不要把历史验收端口当成正式入口。</p><p class="muted">本卡默认只查旧记录和静态证据，不创建实验，不调用编码器、模型API或执行移动。旧实验记录若未在当前库中，用下方历史截图替代，不为了演示补跑。</p>
<h2>三分钟操作顺序</h2><table><thead><tr><th>时间</th><th>操作</th><th>向评审说明</th></tr></thead><tbody>
<tr><td>0:00–0:35</td><td>工作台 → 创建实验。展示数据、策略、权重、预算及单题／队列按钮；不点击开始。</td><td>Web是可视化层，场景与执行由同一内核管理；新运行与历史回放分开。</td></tr>
<tr><td>0:35–1:15</td><td>进入“历史案例”，打开已有航拍成功与失败轨迹；拖动逐步回放。</td><td>Agent每步只能看当前局部观测和给定目标；全图与真值仅供观察者。回放不调用模型。</td></tr>
<tr><td>1:15–1:50</td><td>人工连续观察页：有保存记录则打开；没有则展示本卡的四步工程截图。</td><td>固定地面照片，人工移动后更新当前航拍与证据；越界和重访也记录。尚不是可靠的地面目标自主导航。</td></tr>
<tr><td>1:50–2:35</td><td>空间证据页打开案例112；切换Full、Mean、Wrong，查看保存的分数与预测。</td><td>读取既有结果，不产生新预测。CV041的接受且≤5米为0/9，全拒绝66.67%不等于定位成功。</td></tr>
<tr><td>2:35–3:00</td><td>返回汇报第11、12页。</td><td>航拍成果与交互平台作为中期交付；下一阶段补任务匹配空间监督、目标有效性与新源自主闭环。</td></tr>
</tbody></table>
<h2>历史工程样例备用画面</h2><figure><img src="../../../平台/实验工作台_v4/地面目标连续观测与人工探索_v1/审阅/连续探索桌面.png" alt="PLATFORM024既有四步人工探索记录，含目标照片、当前航拍、访问格和行动账本"><figcaption>PLATFORM-024既有四步人工工程记录，当时界面；本轮未重跑。该画面不作为当前自主导航成绩。</figcaption></figure>
<h2>现场常见问题</h2>
<details><summary>是不是多Agent系统？</summary><p>当前默认是训练过的单导航Agent，包含探索器、目标线索头和规则控制。已有多角色原型，但收益不足以升级默认；多个模块不能等同多个自主Agent。</p></details>
<details><summary>轻量化体现在哪里？</summary><p>Small256探索器797,701参数，Edge线索头136,837参数，任务模块合计934,538参数；冻结视觉骨干仍占模型、内存及推理资源，不能把任务头参数当作全系统规模。</p></details>
<details><summary>通过了哪些测试？</summary><p>Masa S2/S3/S4、SwissView同模态迁移，以及指定9km²新区域协议通过；Masa S3 SR86.80%、SG0.300格，9km²径向保护SR68.04%。不同队列、网格和预算的数字不能当作一条累计提升曲线。</p></details>
<details><summary>为什么15×15仍只给20步？</summary><p>它是固定行动预算的尺度压力确认，用来检查区域扩大后的退化；20步不是所有15×15任务的合理最优预算。新预算研究需另立协议，不能修改已经失败的旧验收来追过线。</p></details>
<details><summary>SG很低是否就证明成功？</summary><p>SR按计划任务数计算；SG是全部任务的终局曼哈顿距离，成功为0。还需看目标对照、恢复与损伤、来源分层以及实际移动距离，不能由单个指标代替验收。</p></details>
<details><summary>已经比论文更好吗？</summary><p>还不能作这一宣称。数据切分、信息权限、动作和预算存在复现差异；有限资源学习基线未稳定利用目标。可靠学习基线、新来源共同确认和完整干净环境科研复跑仍待补证。</p></details>
<details><summary>地面照片现在究竟能做什么？</summary><p>可以做实时编码、固定目标人工连续观察、保存与只读证据比较；CV041空间小头未通过，当前不能把原始余弦当成可靠方向雷达，也不宣称可定位任意手持照片。</p></details>
<details><summary>如果需要现场真正运行？</summary><p>由使用者显式创建临时实验，确认当前数据与模型可用，先用“当前一道题”，演示完释放或明确保存。人工、混合、自由测试及正式Agent评测分账；这不是本轮材料任务的已执行验收。</p></details>
<footer>来源：本批核验/中期内容与证据.json、项目当前状态及PLATFORM-024原验收截图。报告和PPT均保留原指标单位、分母与失败边界。“视频”目录由ZCode负责，本材料任务不读取或管理。</footer></main></html>'''
    (dest/'演示操作卡.html').write_text(card,encoding='utf-8')
    notes='''# 中期汇报讲稿与材料使用

建议7–9分钟讲解，另留3分钟平台查阅。12页PPT的备注区已给出来源，图表、表格及方法图可编辑；最终PDF用于固定排版展示。

| 页 | 讲解要点 |
|---|---|
| 1 | 项目题目、成员及教师；说明这是中期阶段成果。 |
| 2 | 最终任务是地面照片目标的主动定位；当前已形成航拍方法与交互平台，地面自主链路未完成。 |
| 3 | 四动作与局部观测；真值、未访图块、目标坐标只属于环境及事后诊断。 |
| 4 | Small256负责探索，Edge负责目标邻接线索；给出概率门控、任务模块参数和冻结骨干边界。 |
| 5 | 按原协议分别报S2/S3/S4和SwissView；权重重复不增加独立地图。 |
| 6 | 9km²十新区，同题SR58.84%→68.04%，恢复295、损伤88；区域分组区间见备注。 |
| 7 | MassGIS54.62%/20.27%未过，保留真实泛化边界；不靠改旧预算重新宣布通过。 |
| 8 | 工作台能创建、单题/队列运行、保存与同题对照；本页截图未启动实验。 |
| 9 | 既有四步人工连续观察实例；当时界面，不是本轮新运行，更不是自主导航成绩。 |
| 10 | 跨视角空间小头Full0/9，全拒绝66.67%不是定位准确率；停止同因素加训。 |
| 11 | 中期交付与最终未完成项并列。论文公平对标和地面自主导航均未收口。 |
| 12 | 先取得目标定义与监督证据，再冻结轻量闭环和新源确认；多Agent后置。 |

演示操作与问答见同目录“演示操作卡.html”。只读演示失败时使用已绑定截图和PDF，不临时重训、换阈值或消费封存组。
'''
    (dest/'汇报讲稿.md').write_text(notes,encoding='utf-8')
    print('Verified report promoted; demo card and speaker guide written')


def pdf_slides(_=None):
    from pdf2image import convert_from_path
    os.environ['TEMP']=str(OUT/'临时');os.environ['TMP']=str(OUT/'临时');tempfile.tempdir=str(OUT/'临时')
    pdf=OUT/'交付/中期成果汇报.pdf';h=sha(pdf);assert len(PdfReader(pdf).pages)==12
    images=convert_from_path(pdf,dpi=96,poppler_path=str(DEPS/'native/poppler/Library/bin'),thread_count=1)
    for i,im in enumerate(images,1):
        target=OUT/'审阅'/f'native-slide-{i:02d}.png';assert not target.exists();im.save(target)
    assert sha(pdf)==h
    put(OUT/'核验/PPT原生PDF栅格化.json',dict(passed=True,pages=12,pdf_sha256=h,engine='bundled Poppler',model_calls=0))
    print('12 native PDF pages rendered')


def final_check(_=None):
    facts=read(OUT/'核验/中期内容与证据.json');old=read(OUT/'核验/旧稿与模板保护.json')
    for name,h in old.items():assert sha(ROOT/name)==h,name
    mutable={'docs/PROJECT_STATE.md':'原入口_PROJECT_STATE.md'}
    for item in read(OUT/'核验/输入绑定.json')['files']:
        p=ROOT/item['path']
        if item['path'] in mutable and (OUT/'核验'/mutable[item['path']]).exists():p=OUT/'核验'/mutable[item['path']]
        assert sha(p)==item['sha256'],item['path']
    deliver=OUT/'交付';report=PdfReader(deliver/'中期报告.pdf');deck=PdfReader(deliver/'中期成果汇报.pdf')
    assert len(report.pages)==9 and len(deck.pages)==12
    all_text='\n'.join(p.extract_text() or '' for p in deck.pages)
    for n in ['86.80','68.0','54.6','20.3','66.67']:assert n in all_text,n
    assert facts['area']['recovered']-facts['area']['harmed']==207
    a=facts['area'];assert abs((a['candidate']['successes']-a['reference']['successes'])/2250-.092)<1e-12
    with ZipFile(deliver/'中期成果汇报.pptx') as z:
        slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
        charts=[n for n in z.namelist() if re.fullmatch(r'ppt/(?:slides/)?charts/chart\d+\.xml',n)]
        assert len(slides)==12 and len(charts)==3
        assert sum(z.read(n).count(b'<a:tbl>') for n in slides)==3
        assert len([n for n in z.namelist() if n.startswith('ppt/embeddings/') and n.endswith('.xlsx')])==3
        ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        vals=[list(map(float,etree.fromstring(z.read(n)).xpath('//c:val//c:numCache/c:pt/c:v/text()',namespaces=ns))) for n in charts]
        expected=[[round(a['reference']['sr'],4),round(a['candidate']['sr'],4)],[755.2,648.53],[.5462,.2027]]
        assert vals==expected,(vals,expected)
    assert sha(deliver/'中期成果汇报.pptx')==read(OUT/'核验/PPT结构与可编辑性.json')['finalSha256']
    source_names=['build_midterm_refresh_v1.py','build_midterm_showcase_v1.mjs','audit_midterm_refresh_v1.py','export_midterm_refresh_v1.ps1']
    paths=[ROOT/'project/src/documents'/n for n in source_names]+list(deliver.iterdir())
    paths += [ROOT/'平台/实验工作台_v4/地面目标连续观测与人工探索_v1/审阅/连续探索桌面.png']
    binding=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in paths if p.is_file()]
    put(OUT/'核验/交付与源码绑定.json',dict(files=binding,model_calls=0,new_navigation=0,scope='DOC-009; ZCode video directory excluded'))
    put(OUT/'核验/最终数字与交付复核.json',dict(passed=True,report_pages=9,slides=12,native_charts=3,native_tables=3,embedded_editable_workbooks=3,
        chart_literals_verified=expected,old_artifacts_unchanged=len(old),alternate_arithmetic=dict(recovered=295,harmed=88,net=207,sr_gain_pp=9.2),
        reviewer='same Codex agent; alternate implementation, not independent personnel',ground_navigation_passed=False,model_calls=0,new_navigation=0,
        video_directory_read_or_modified=False,network='localhost read-only screenshots only; no external proxy',git_commit_or_push=False))
    print('Final deliverable metrics, native charts and old originals verified')


if __name__=='__main__':
    import sys
    {'render':render,'audit':audit,'materials':materials,'pdf-slides':pdf_slides,'final':final_check}[sys.argv[1]](sys.argv[2] if len(sys.argv)>2 else 'r2')
