"""Curated Git surface: current runtime, frozen research evidence, current reports.

Builds manifests only; it does not delete local research files, stage, or push.
"""
import ast
from collections import Counter
import hashlib
import gzip
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'项目导航/Git精简发布_20261025_v1'
OLD=('实时导航平台_v2','实时导航平台_v3','实时导航平台_上传测试_v1')
ARCHIVE=('实时导航平台_v2','实时导航平台_v3','上传测试台_v1')


def read(p):
    p=Path(p)
    if p.exists():return json.loads(p.read_text(encoding='utf-8-sig'))
    return json.loads(gzip.decompress(Path(str(p)+'.gz').read_bytes()).decode('utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(name,value):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,separators=(',',':'));f.write('\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def names(*args):return [n for n in git(*args,'-z').decode('utf-8').split('\0') if n]


def prepare():
    if git('diff','--cached','--name-only').strip():raise ValueError('已有暂存内容，先协调')
    moves=[]
    for old,archive in zip(OLD,ARCHIVE):
        folder=ROOT/'平台'/old;dest=ROOT/'平台/历史版本'/archive
        if not folder.is_dir() or not dest.is_dir():raise ValueError('旧平台目录现场变化')
        for p in sorted(folder.rglob('*')):
            if p.is_symlink():raise ValueError('迁移不接受符号链接')
            if p.is_file():
                rel=p.relative_to(folder);new=dest/('旧目录说明.md' if str(rel)=='README.md' else rel)
                if new.exists():raise ValueError('历史目的已存在，禁止覆盖')
                moves.append(dict(old=p.relative_to(ROOT).as_posix(),new=new.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    put('旧平台迁移预登记.json',dict(files=moves,action='move unique evidence into existing local archive;do not publish old platforms'))
    put('Git基线.json',dict(head=git('rev-parse','HEAD').decode().strip(),branch=git('branch','--show-current').decode().strip(),
        origin=git('rev-parse','origin/main').decode().strip(),staged_empty=True,tracked=names('ls-files')))
    print('migration registered',len(moves),'files')


def audit_move():
    rows=read(OUT/'旧平台迁移预登记.json')['files']
    for row in rows:
        assert not (ROOT/row['old']).exists(),row['old']
        assert sha(ROOT/row['new'])==row['sha256'],row['new']
    assert all(not (ROOT/'平台'/old).exists() for old in OLD)
    put('核验/旧平台迁移复核.json',dict(passed=True,files=len(rows),bytes=sum(r['bytes'] for r in rows),original_bytes_preserved=True))


def source_refs(value,result,eligible=False):
    if isinstance(value,dict):
        for k,v in value.items():
            if any(word in k.lower() for word in ['protected','protection','baseline','old_sources','current_entry']):continue
            context=eligible or any(word in k.lower() for word in ['sources','source_sha','source_files','code_sha','源码','executor','runner','driver'])
            if context and k.startswith('project/src/') and k.endswith('.py') and (ROOT/k).is_file():result.add(k)
            source_refs(v,result,context)
    elif isinstance(value,list):
        for v in value:source_refs(v,result,eligible)
    elif eligible and isinstance(value,str) and value.startswith('project/src/') and value.endswith('.py') and (ROOT/value).is_file():result.add(value)


def plan():
    from project.src.webapp.binding_v3 import source_closure
    tracked=set(names('ls-files'));available=tracked|set(names('ls-files','--others','--exclude-standard'))
    chosen={};metadata=set();source_roots=set()
    index=read(ROOT/'项目导航/实验索引.json');life=read(ROOT/'docs/维护/版本生命周期.json')
    manifest=read(ROOT/'平台/实验工作台_v4/发布修复与存储收束_v1/搬迁发布清单.json')
    runtime={n for n in manifest['files'] if n.startswith('project/src/')}
    # Explicit workers and current tests are needed even if only spawned/imported dynamically.
    runtime.update(source_closure(ROOT,['project/src/webapp/workbench_server_v21.py','project/src/platform_core/cli_v2.py',
        'project/src/platform_core/cli_v4.py','project/src/tools/ground_live_worker_v1.py','project/src/tools/ground_frame_worker_v1.py',
        'project/src/tests/test_release_maintenance_v1.py','project/src/tools/repository_publication_v1.py']))
    runtime.update({'project/src/README.md','project/src/webapp/README.md','project/src/tools/README.md'})
    # Root entry fields are not all runnable defaults; retain their small report/config targets.
    for v in life['current'].values():
        if isinstance(v,str) and (ROOT/v).is_file():metadata.add(v)
    for batch in index['batches']:
        files=batch.get('reports',[])+batch.get('evidence',[])
        for name in files:
            p=ROOT/name
            if not p.is_file():continue
            metadata.add(name)
            if p.suffix=='.json' and p.stat().st_size<=2*1024**2 and not str(batch.get('category','')).startswith('PLATFORM'):
                try:source_refs(read(p),source_roots)
                except (ValueError,UnicodeError):pass
        # A stage's source binding is frequently in a sibling seal, not the short index.
        folder=ROOT/batch['path']
        if folder.is_dir() and not str(batch.get('category','')).startswith('PLATFORM'):
            for pattern in ['*封存*.json','*源码*绑定*.json','*输入*绑定*.json']:
                for p in folder.glob(pattern):
                    if p.stat().st_size<=2*1024**2:
                        try:source_refs(read(p),source_roots)
                        except (ValueError,UnicodeError):pass
    research={};broken_legacy=[]
    for source in sorted(source_roots):
        try:research.update(source_closure(ROOT,[source]))
        except (SyntaxError,UnicodeError) as exc:broken_legacy.append(dict(path=source,reason=type(exc).__name__))
    runtime.update(research)
    # Model algorithms/data contracts are small and distinct research mechanisms.
    for area in ['agents','data','env','train']:
        runtime.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'project/src'/area).glob('*.py'))
    runtime.update(source_closure(ROOT,[n for n in runtime if n.endswith('.py')]))
    plots=[p.relative_to(ROOT).as_posix() for p in (ROOT/'project/src/documents').glob('plot*.py')]
    plots.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'project/src/documents').glob('render_method*.py'))
    runtime.update(source_closure(ROOT,plots))
    exact_platform={n for n in manifest['files'] if n.startswith('平台/')}
    exact_platform.update({'平台/README.md','平台/启动平台.cmd','平台/启动平台.ps1',
        '平台/实验工作台_v4/发布修复与存储收束_v1/README.md',
        '平台/实验工作台_v4/发布修复与存储收束_v1/目录用途登记沿革.md',
        '平台/实验工作台_v4/发布修复与存储收束_v1/项目状态沿革.md',
        '平台/实验工作台_v4/发布修复与存储收束_v1/入口快照/02.before',
        '平台/实验工作台_v4/发布修复与存储收束_v1/入口快照/03.before',
        '平台/实验工作台_v4/发布修复与存储收束_v1/核验/核心HTTP回归_v2.log',
        '平台/实验工作台_v4/发布修复与存储收束_v1/核验/GUI回归.json',
        '平台/实验工作台_v4/发布修复与存储收束_v1/核验/正式服务复核.json',
        '平台/实验工作台_v4/发布修复与存储收束_v1/核验/原生清理回执.json',
        '平台/实验工作台_v4/发布修复与存储收束_v1/核验/原件与用户记录复核.json',
        '平台/实验工作台_v4/发布修复与存储收束_v1/环境重建元数据.json'})
    materials={v for k,v in life['current'].items() if isinstance(v,str) and k in ['report','report_pdf','final_report','final_report_pdf']}
    rows=[]
    for n in sorted(available|runtime|exact_platform|materials|metadata):
        p=ROOT/n
        if not p.is_file():rows.append([n,'absent_or_migrated',0]);continue
        size=p.stat().st_size;suffix=p.suffix.lower();parts=p.parts;reason=None
        unsafe= n.startswith(('.agents/本地/','models/','node_modules/','GOMAA-Geo/','DATA/raw_data/','project/.runtime/','project/src/tools/local/','平台/历史版本/','平台/主动探索演示_v1/','视频/')) or '__pycache__' in parts
        if unsafe:reason='local_asset_or_retired_entry'
        elif p.name.startswith('.env') and not p.name.endswith('.example'):reason='credentials'
        elif suffix in ['.pt','.pth','.ckpt','.npy','.npz','.safetensors','.sqlite','.zip','.7z','.tif','.tiff']:reason='model_data_cache_or_package'
        elif '/源码快照/' in n:reason='historical_duplicate_source_snapshot'
        elif n.startswith('project/src/'):
            if n in runtime:chosen[n]='current_dependency_or_bound_research_source'
            else:reason='inactive_source_or_superseded_platform_asset'
        elif n.startswith('project/'):
            if p.name.endswith('.example') or p.name.startswith('requirements') or '/defaults/' in n or 'default' in p.name:chosen[n]='configuration_or_dependencies'
            else:reason='local_project_payload'
        elif n.startswith('平台/'):
            if n in exact_platform and size<=1024**2:chosen[n]='current_platform_or_required_guide'
            else:reason='historical_platform_QA_or_payload'
        elif n.startswith('DATA/processed_data/'):
            if '/记录/' in n or suffix=='.jsonl':reason='full_prediction_or_step_log'
            elif suffix=='.md' and size<=256*1024:chosen[n]='experiment_report'
            elif n in metadata and suffix in ['.json','.csv'] and size<=1024**2:chosen[n]='indexed_verdict_summary_protocol_or_binding'
            else:reason='local_detailed_experiment_payload'
        elif n.startswith(('总结报告相关/','中期报告相关/')):
            if n in materials:chosen[n]='current_report'
            elif suffix=='.md' and '/历史版本/' not in n and '/审阅/' not in n:chosen[n]='report_guide_or_argument'
            elif suffix in ['.json','.csv'] and '/核验/' not in n and '/审阅/' not in n and size<=256*1024:chosen[n]='report_evidence_index'
            else:reason='superseded_draft_or_layout_QA'
        elif n.startswith('绘图/'):
            if '/审阅/' in n or '/论文图_v1/' in n or '/方法图_v1/审阅' in n:reason='figure_review_or_superseded_draft'
            elif suffix in ['.png','.pdf','.svg','.md','.json','.csv','.py'] and (suffix!='.svg' or size<=512*1024) and size<=4*1024**2:chosen[n]='scientific_figure_or_source'
            else:reason='large_embedded_or_duplicate_figure'
        elif n.startswith('论文/'):
            if p.name=='README.md':chosen[n]='primary_source_links'
            else:reason='reference_paper_PDF_local_only'
        elif n.startswith('项目导航/'):
            if '/Git精简发布_20261025_v1/' in n and p.name not in ['README.md','发布核验摘要.json']:reason='local_publication_inventory'
            elif any(x in n for x in ['/源码快照/','/公开来源/','/输入/','/运行/','/清单/','/备份/']):reason='local_audit_input_or_copy'
            elif suffix=='.md':chosen[n]='research_and_maintenance_guide'
            elif suffix=='.json' and (n=='项目导航/实验索引.json' or size<=256*1024):chosen[n]='small_research_or_validation_summary'
            else:reason='detailed_audit_payload'
        elif n.startswith('docs/'):
            if '/历史版本/' not in n and suffix in ['.md','.json']:chosen[n]='current_coordination_and_policy'
            else:reason='old_document_copy'
        elif n.startswith('.agents/'):
            if n in ['.agents/TASKS.md','.agents/HANDOFF.md','.agents/README.md']:chosen[n]='coordination'
            else:reason='local_agent_history'
        elif suffix=='.md' or n=='.gitignore' or n.startswith('课程相关报告模版/') or (n.startswith('调研/') and suffix=='.json' and size<=768*1024):chosen[n]='project_document_or_template'
        else:reason='nonessential_root_payload'
        if n in chosen:
            if size>5*1024**2:raise ValueError('selected file too large:'+n)
        else:rows.append([n,reason or 'not_selected',size])
    selected=[dict(path=n,reason=why,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n,why in sorted(chosen.items())]
    result=dict(selected=selected,excluded=rows,tracked_before=len(tracked),remove_from_index=sorted(tracked-set(chosen)),
                total_bytes=sum(r['bytes'] for r in selected),runtime_sources=len(runtime),omitted_broken_legacy_roots=broken_legacy)
    put('本地发布清单.json',result)
    (OUT/'待加入.paths').write_bytes(b'\0'.join(r['path'].encode('utf-8') for r in selected)+b'\0')
    (OUT/'退出索引.paths').write_bytes(b'\0'.join(n.encode('utf-8') for n in result['remove_from_index'])+b'\0')
    print(json.dumps(dict(selected=len(selected),MiB=round(result['total_bytes']/2**20,2),remove=len(result['remove_from_index']),source_files=len(runtime),excluded_reasons=dict(Counter(r[1] for r in rows))),ensure_ascii=False))


def audit_public():
    final=read(OUT/'最终暂存清单.json');selected={r['path'] for r in final['selected']}
    secret_values=[]
    for p in (ROOT/'project').glob('.env*'):
        if p.name.endswith('.example'):continue
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            k,sep,v=line.partition('=')
            if sep and re.search(r'KEY|TOKEN|SECRET|PASSWORD',k,re.I):
                v=v.strip().strip('"\'')
                if len(v)>=12:secret_values.append(v.encode())
    problems=[];python_count=json_count=0
    for name in sorted(selected):
        p=ROOT/name;raw=p.read_bytes()
        if p.name.startswith('.env') and not p.name.endswith('.example'):problems.append([name,'credentials_path'])
        if p.suffix=='.py':ast.parse(raw.decode('utf-8-sig'),filename=name);python_count+=1
        if p.suffix=='.json':json.loads(raw.decode('utf-8-sig'));json_count+=1
        parts=[raw]
        if p.suffix=='.docx':
            with zipfile.ZipFile(p) as z:
                parts.extend(z.read(i) for i in z.namelist() if i.endswith('.xml') and z.getinfo(i).file_size<16*1024**2)
        for blob in parts:
            if any(v in blob for v in secret_values):problems.append([name,'actual_local_credential_value'])
            if re.search(rb'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----',blob):problems.append([name,'private_key_material'])
            if re.search(rb'\b(?:sk-|gsk_)[A-Za-z0-9_-]{28,}\b',blob):problems.append([name,'possible_provider_token'])
    if problems:
        put('核验/敏感项待核.json',dict(problems=problems,values_redacted=True));raise ValueError('Public credential check needs review;see redacted report')
    from project.src.webapp.binding_v3 import source_closure
    closure=source_closure(ROOT,['project/src/webapp/workbench_server_v21.py','project/src/tools/package_workbench_v25.py',
        'project/src/tools/ground_live_worker_v1.py','project/src/tools/ground_frame_worker_v1.py',
        'project/src/tests/test_release_maintenance_v1.py'])
    missing=set(closure)-selected
    if missing:raise ValueError('Missing current dependency: '+str(sorted(missing)))
    if any(n.startswith(('平台/历史版本/','平台/实时导航平台_v2/','平台/实时导航平台_v3/','平台/实时导航平台_上传测试_v1/','models/','DATA/raw_data/')) for n in selected):
        raise ValueError('Retired platform/model/raw data selected')
    put('核验/公开源码与凭据核验.json',dict(passed=True,selected_files=len(selected),python_syntax=python_count,json_parse=json_count,
        current_dependency_closure=len(closure),missing_dependencies=[],credential_matches=0,actual_secrets_never_printed=True,
        model_forward=0,navigation=0))
    print(json.dumps(dict(passed=True,python_syntax=python_count,json_parse=json_count,current_dependencies=len(closure),credential_matches=0)))


if __name__=='__main__':
    import sys
    {'prepare':prepare,'audit-move':audit_move,'plan':plan,'audit-public':audit_public}[sys.argv[1]]()
