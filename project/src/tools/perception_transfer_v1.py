"""Trust the original CV031 seal before collecting or transferring its assets."""
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote,urlsplit
import re
from project.src.platform_core.perception_reader_v1 import PerceptionReader,BATCH,SEAL_SHA,EvidenceUnavailable

GUIDE='平台/实验工作台_v4/局部感知证据查阅_v1/README.md'
GUIDE_SEAL='平台/实验工作台_v4/局部感知证据查阅_v1/阶段封存.json'
GUIDE_SEAL_SHA='8ba08d618e69ace20036a8a799f00a78a586be1fc887172b948aaf8548dbcc2a'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def checked(root,relative,expected):
    p=(Path(root)/relative).resolve()
    if not p.is_relative_to(Path(root).resolve()) or not p.is_file():raise EvidenceUnavailable('缺少绑定的搬迁资产：'+relative)
    if digest(p)!=expected:raise EvidenceUnavailable('搬迁资产与原封存SHA不同：'+relative)
    return p


def verified_perception_assets(root):
    """172 fixed items, each compared to its original seal/image binding."""
    root=Path(root).resolve();reader=PerceptionReader(root)
    refs,b,s,cal,cases=reader._bundle()
    expected={BATCH+'/阶段封存.json':SEAL_SHA}
    for name in ('冻结执行绑定.json','开发筛查汇总.json','唯一校准阈值.json'):expected[BATCH+'/'+name]=refs[BATCH+'/'+name]
    for n in range(21,101):
        for name in (f'预测_{n:03d}.json',f'roi_{n:03d}.npz'):expected[BATCH+'/'+name]=refs[BATCH+'/'+name]
    for e in b['images']:expected[e['path']]=e['sha256']
    if set(expected)!=set(reader.assets()) or len(expected)!=172:raise EvidenceUnavailable('固定证据资产集合变化')
    for relative,h in expected.items():checked(root,relative,h)
    return expected


def verified_guide_assets(root):
    """Retain the original guide and all its frozen small release attachments."""
    root=Path(root).resolve();p=checked(root,GUIDE_SEAL,GUIDE_SEAL_SHA)
    seal=json.loads(p.read_text(encoding='utf-8-sig'))
    expected={e['path']:e['sha256'] for e in seal['files']};expected[GUIDE_SEAL]=GUIDE_SEAL_SHA
    if GUIDE not in expected:raise EvidenceUnavailable('当前指南未在原交付封存中')
    for relative,h in expected.items():checked(root,relative,h)
    return expected


def require_guide_links(root,files,guide_relative=GUIDE,min_links=8):
    """Every Markdown local link in the new feature guide must travel with it."""
    root=Path(root).resolve();guide=root/guide_relative;found=[]
    for raw in re.findall(r'\]\(([^)]+)\)',guide.read_text(encoding='utf-8-sig')):
        target=urlsplit(raw)
        if target.scheme or not target.path:continue
        p=(guide.parent/unquote(target.path)).resolve()
        if not p.is_relative_to(root) or p.relative_to(root).as_posix() not in files:
            raise EvidenceUnavailable('指南本地链接未进入搬迁资产：'+raw)
        found.append(p.relative_to(root).as_posix())
    if len(found)<min_links:raise EvidenceUnavailable('指南链接数异常')
    return found


def verify_expected(root,expected):
    for relative,h in expected.items():checked(root,relative,h)
