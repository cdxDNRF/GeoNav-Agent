"""Reference-aware navigation and reversible cleanup; experiment/data paths stay intact."""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
from collections import defaultdict
import argparse,ast,json,os,re,zipfile

ROOT=Path(__file__).resolve().parents[3]
NAV=ROOT/'项目导航';RECORD=NAV/'整理记录_2026-10-02_v1'
ALLOWED_MODIFIED={
    'README.md','中期报告相关/中期报告_基于主动探索的视觉地理定位方法设计与实现.docx',
    '中期报告相关/编制依据.json','中期报告相关/编制说明.md'}
IGNORED_DIRS={'.git','node_modules','.zcode','__pycache__','.pytest_cache','.mypy_cache','.ipynb_checkpoints'}
SENSITIVE_NAMES={'.env','.env.deepseek','.env.example','.env.deepseek.example'}


def digest(p):
    h=sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)


def read(p):return json.loads(p.read_text('utf8'))
def rel(p):return p.relative_to(ROOT).as_posix()
def link(p,label,base=NAV):return f'[{label}](<{os.path.relpath(p,base).replace(chr(92),"/")}>)'


def scan():
    for folder in ('DATA','models','project','论文','课程相关报告模版','选题报告相关','中期报告相关','绘图','调研','视频'):
        for dp,ds,fs in os.walk(ROOT/folder):
            ds[:]=[n for n in ds if n not in IGNORED_DIRS]
            for n in fs:
                p=Path(dp)/n
                if p.name.startswith('.env') or p.name.startswith('~$') or p.is_symlink():continue
                yield p


def prepare():
    if RECORD.exists():raise ValueError('immutable organization record exists')
    for p in (RECORD/'恢复备份',RECORD/'源码快照',RECORD/'审阅',ROOT/'中期报告相关/历史版本',ROOT/'中期报告相关/排版审阅/2026-10-02_v2',ROOT/'中期报告相关/答辩材料/排版审阅'):
        p.mkdir(parents=True,exist_ok=True)
    items=[]
    for p in scan():
        st=p.stat();r=dict(path=rel(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
        if p.suffix in {'.py','.md','.json','.docx','.pdf','.png','.svg','.js','.cjs','.ps1','.txt'} and st.st_size<=2*1024*1024:r['sha256']=digest(p)
        items.append(r)
    for p in (ROOT/'README.md',ROOT/'KNOWLEDGE.md',ROOT/'research.md',ROOT/'.gitignore'):
        if p.exists():st=p.stat();items.append(dict(path=rel(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=digest(p)))
    write(RECORD/'整理前文件清单.json',dict(utc=datetime.now(timezone.utc).isoformat(),files=items,allowed_modified=sorted(ALLOWED_MODIFIED),
        large_files='metadata protected; source/small evidence files also SHA protected',new_experiments=0,cloud_calls=0))
    with zipfile.ZipFile(RECORD/'恢复备份/修订前材料.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(ALLOWED_MODIFIED):
            p=ROOT/name
            if p.is_file():z.write(p,name)
    source=ROOT/'中期报告相关/中期报告_基于主动探索的视觉地理定位方法设计与实现.docx'
    (ROOT/'中期报告相关/历史版本/2026-09-30_中期报告_修订前.docx').write_bytes(source.read_bytes())
    caches=[]
    for p in (ROOT/'project/src').rglob('__pycache__'):
        if p.is_dir():
            for q in p.rglob('*'):
                if q.is_file():caches.append(dict(path=rel(q),bytes=q.stat().st_size,sha256=digest(q),reason='Python regenerable bytecode'))
    small=[p for p in scan() if p.stat().st_size<=2*1024*1024 and p.suffix.lower() in {'.md','.json','.txt','.docx','.pdf','.png'}]
    bysize=defaultdict(list)
    for p in small:bysize[(p.stat().st_size,p.suffix)].append(p)
    dup=[]
    for ps in bysize.values():
        if len(ps)<2:continue
        groups=defaultdict(list)
        for p in ps:groups[digest(p)].append(rel(p))
        for h,paths in groups.items():
            if len(paths)>1:dup.append(dict(sha256=h,bytes=(ROOT/paths[0]).stat().st_size,paths=paths))
    write(RECORD/'重复文件检查.json',dict(groups=dup,policy='Frozen source snapshots/results and template inputs retain paths; equality alone never authorizes deletion.'))
    write(RECORD/'清理候选.json',dict(regenerable_bytecode=caches,duplicate_copy_candidates=[g for g in dup if any('副本' in p for p in g['paths'])],
        not_deleted='Datasets,weights,logs,failed experiments,source snapshots,node_modules,reference repository,user-edited proposal'))
    print(dict(prepared=True,protected_files=len(items),bytecode_files=len(caches),bytecode_bytes=sum(r['bytes'] for r in caches),duplicate_groups=len(dup)),flush=True)


def historic_readme():
    p=NAV/'历史开发记录.md'
    if p.exists():raise ValueError('immutable historical README copy exists')
    old=(ROOT/'README.md').read_text('utf8')
    def rebase(m):
        target=m.group(2).strip('<>')
        if re.match(r'^(https?://|app:|codex:|#)',target):return m.group(0)
        return m.group(1)+'(<../'+target+'>)'
    old=re.sub(r'(\]\()([^\n)]+)\)',lambda m:rebase(m).replace(']((<','](<'),old)
    # Explicit parser below handles ordinary Markdown links without doubling parentheses.
    old=(ROOT/'README.md').read_text('utf8')
    def convert(m):
        label,target=m.groups();target=target.strip('<>')
        if re.match(r'^(https?://|app:|codex:|#)',target):return m.group(0)
        return f'[{label}](<../{target}>)'
    old=re.sub(r'\[([^\]]+)\]\(([^\n)]+)\)',convert,old)
    p.write_text('# 历史开发记录\n\n本文件保存整理前README的逐批记录；记录中的“尚未”表示当时状态。当前进度和默认以根目录README与对应正式验收为准。相对链接已改为从本目录指向原件。原始字节存于整理记录的修订前材料备份。\n\n'+old,'utf8')


def index():
    tree=ast.parse((ROOT/'project/src/documents/stage_closeout.py').read_text('utf8'));desc={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DESCRIPTIONS' for t in n.targets):desc=ast.literal_eval(n.value)
    overrides={
        '边缘线索S2正式复验_v1':('正式主线','S2通过','val100×3，4源文件，5×5/B10，SR89.33%/SG0.320'),
        '边缘线索S3独立源图确认_v1':('正式主线','S3通过','test250×3，10源文件，5×5/B10，SR86.80%/SG0.300'),
        'S4十乘十正式扩展_v1':('正式扩展','S4密度扩展通过','500题×3，20已见源文件，10×10/B20，77.13%/1.201'),
        '五乘五正式迁移_v1':('正式迁移','SwissView迁移通过','500题×3，20新源文件，5×5/B10，86.80%/0.381'),
        '继续训练冻结目标证据复验_v1':('最新复验','目标证据通过','已见20源文件；Full79.73%/SG0.823；新增4500控制，复用1500真实目标'),
        '继续训练独立源图确认_v1':('最新确认','升级未通过','新20SwissView源文件；10×10仅SR+0.60点，1/3正，SG改善；默认保持'),
        '探索器协议适配开发验证_v1':('训练因素','协议适配升级未通过','Continue5 79.73%/0.823、Adapt10 80.00%/1.059；同预算，不增加参数'),
        '探索器协议适配同预算对照_v1':('训练记录','六模型训练及重放完成','Continue5/Adapt10各三最终权重；491520新训练动作，6144优化器更新'),
        '零调用账本完整队列验证_v1':('探索规划','未达到SR+2点门槛','M1 78.40%/1.110，对M0+1.27点；低成本备选'),
        '账本探索保护规则验证_v1':('探索保护','候选未通过','77.07%/1.199；修复旧损伤，也失去大部分恢复'),
        '目标邻域覆盖规划验证_v1':('探索覆盖','候选未通过','76.93%/1.281，未增加净成功'),
        '证据账本简化接口验证_v2':('云端协作','接口和D通过，G未达扩大门槛','同20题，云端三重复；M0 55%、零调用60%、云端三方案均55%'),
        '证据账本预算规划验证_v1':('云端接口','T通过，旧D未通过','14请求保留；修复在另批，不把请求成功当导航有效'),
        '多Agent同预算小验证_v1':('云端协作','小验证未通过扩大门槛','原触发只覆盖成功题；未产生有效计划修正'),
        '云端8790接口检查_v1':('云端接口','已弃用备选','9生成+1列表；用户已决定不继续考虑三个选项'),
        '高分旋转负样本开发验证_v1':('目标鲁棒性','新候选未通过','训练预算一致，未优于均匀负旋转；未消费该批新确认图'),
        '旋转负样本开发验证_v1':('目标鲁棒性','新候选未通过','增加旋转负例未达到SR一致收益门槛'),
        '候选可靠性开发验证_v1':('目标校准','新候选未通过','0.90阈值提高精度但损伤召回和干净SR'),
        '候选可靠性离线校准_v1':('校准诊断','离线校准完成','接受精度不是导航SR；在线验证另列'),
        '目标方向候选试跑确认_v1':('目标旋转','新候选未通过','04…07新源文件试跑；四朝向降低干净接受可靠性'),
        '目标方向候选开发验证_v1':('目标旋转','开发条件通过后试跑失败','已见Masa28；不能把旋转集合不变性当任意旋转鲁棒'),
        '目标扰动鲁棒性_v1':('适用边界','扰动诊断完成','揭示边缘连续性对真实目标条件的依赖，不修改默认'),
    }
    desc.update(overrides);entries=[]
    rows=['# 全部实验索引','',
        '每行对应原始批次目录。数据、权重、轨迹、源码快照均保留原址；相同名称的训练和评测目录分别列出。历史状态只描述对应批次，不是当前默认状态。','',
        '| 数据集与类型 | 批次 | 判定 | 范围及入口 |','|---|---|---|---|']
    for dataset in ('Masa','SwissView'):
        for kind in ('训练结果','评测结果'):
            base=ROOT/'DATA/processed_data'/dataset/kind
            if not base.exists():continue
            for p in sorted(base.iterdir()):
                if not p.is_dir():continue
                cat,status,note=desc.get(p.name,('历史记录','保留该批原判定','早期调试、训练或工程记录；不作为当前正式成绩'))
                evid=[q for q in (p/'验收结论.json',p/'独立复核.json',p/'最终状态.json',p/'README.md') if q.is_file()]
                reports=[q for q in sorted(p.glob('*.md')) if any(s in q.name for s in ('报告','说明','解读')) and q.name not in ('执行协议.md','冻结方案.md')]
                a=[]
                if reports:a.append(link(reports[0],'报告'))
                a.extend(link(q,{'验收结论.json':'验收','独立复核.json':'复核','最终状态.json':'最终状态','README.md':'原目录说明'}[q.name]) for q in evid)
                a.append(link(p,'原目录'))
                rows.append(f'| {dataset} / {kind} | {p.name} | {status} | {note}；'+ ' / '.join(a)+' |')
                entries.append(dict(dataset=dataset,kind=kind,name=p.name,category=cat,status=status,scope=note,path=rel(p),reports=[rel(q) for q in reports],evidence=[rel(q) for q in evid]))
    (NAV/'实验索引.md').write_text('\n'.join(rows)+'\n','utf8');write(NAV/'实验索引.json',dict(date='2026-10-02',batches=entries,source_only=True))
    historic_readme();print(dict(indexed_batches=len(entries)),flush=True)


def cleanup():
    if (RECORD/'清理执行清单.json').exists():raise ValueError('cleanup receipt exists')
    source=(ROOT/'project/src').resolve();plan=read(RECORD/'清理候选.json');targets=[]
    for row in plan['regenerable_bytecode']:
        p=(ROOT/row['path']).resolve()
        if source not in p.parents or '__pycache__' not in p.parts or p.suffix!='.pyc':raise ValueError('unsafe cache target')
        if p.is_file():targets.append(dict(row,current_sha256=digest(p),current_bytes=p.stat().st_size))
    # Later authoring may create more project bytecode; include only exact pyc entries.
    existing={r['path'] for r in targets}
    for p in source.rglob('*.pyc'):
        q=p.resolve()
        if source not in q.parents or '__pycache__' not in q.parts or rel(p) in existing:continue
        targets.append(dict(path=rel(p),bytes=p.stat().st_size,sha256=digest(p),current_sha256=digest(p),current_bytes=p.stat().st_size,reason='regenerable Python bytecode'))
    archive=RECORD/'恢复备份/删除文件.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for r in targets:z.write(ROOT/r['path'],r['path'])
    with zipfile.ZipFile(archive) as z:
        for r in targets:
            if sha256(z.read(r['path'])).hexdigest()!=r['current_sha256']:raise ValueError('backup mismatch')
    for r in targets:
        p=(ROOT/r['path']).resolve()
        if source not in p.parents or digest(p)!=r['current_sha256']:raise ValueError('delete target drift')
        p.unlink()
    removed=[]
    for p in sorted(source.rglob('__pycache__'),key=lambda p:len(p.parts),reverse=True):
        if source not in p.resolve().parents:raise ValueError('cache folder escaped source')
        if p.is_dir() and not any(p.iterdir()):p.rmdir();removed.append(rel(p))
    receipt=dict(deleted_files=targets,deleted_count=len(targets),deleted_bytes=sum(r['current_bytes'] for r in targets),removed_empty_cache_directories=removed,
        recovery_archive=rel(archive),recovery_sha256=digest(archive),deleted_failed_experiment_evidence=False,deleted_dataset_or_model=False,
        duplicate_groups_retained='Frozen snapshots,original report versions and active template dependencies are intentional evidence; no path-breaking deletions.')
    write(RECORD/'清理执行清单.json',receipt);print(dict(deleted_count=receipt['deleted_count'],deleted_bytes=receipt['deleted_bytes'],recovery_archive=rel(archive)),flush=True)


def verify():
    before=read(RECORD/'整理前文件清单.json');changed=[];verified=0;authorized=[]
    for r in before['files']:
        p=ROOT/r['path']
        if r['path'] in ALLOWED_MODIFIED:
            if not p.is_file():raise ValueError('required revised material missing')
            if 'sha256' in r and digest(p)!=r['sha256']:authorized.append(r['path'])
            continue
        if not p.is_file():changed.append(dict(path=r['path'],reason='missing'));continue
        st=p.stat()
        if (st.st_size,st.st_mtime_ns)!=(r['bytes'],r['mtime_ns']):changed.append(dict(path=r['path'],reason='metadata drift'))
        elif 'sha256' in r and digest(p)!=r['sha256']:changed.append(dict(path=r['path'],reason='SHA drift'))
        verified+=1
    if changed:write(RECORD/'保护核查异常.json',changed);raise ValueError('unexpected protected file changes: '+str(len(changed)))
    links=0
    for p in (ROOT/'README.md',NAV/'README.md',NAV/'实验索引.md',NAV/'目录结构与维护.md',NAV/'当前进度.md',ROOT/'中期报告相关/README.md'):
        for target in re.findall(r'\]\(([^\n)]+)\)',p.read_text('utf8')):
            target=target.strip('<>')
            if re.match(r'^(https?://|#)',target):continue
            if not (p.parent/target).exists():raise ValueError('broken navigation link '+str(p)+' '+target)
            links+=1
    final=dict(passed=True,protected_files_verified=verified,authorized_modified_materials=authorized,navigation_links_verified=links,
        deleted_files=read(RECORD/'清理执行清单.json')['deleted_count'],new_training_steps=0,cloud_calls=0,new_navigation_evaluation=0,
        raw_data_models_experiments_source_scripts_unchanged=True)
    write(RECORD/'整理复核.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','index','cleanup','verify'));args=p.parse_args()
    {'prepare':prepare,'index':index,'cleanup':cleanup,'verify':verify}[args.mode]()
