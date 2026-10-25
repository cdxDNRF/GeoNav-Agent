"""Original-SHA asset closure for CV041 viewer; no weights, features or trainer."""
import hashlib
from pathlib import Path
import re
from project.src.platform_core.spatial_evidence_reader_v1 import SpatialEvidenceReader

GUIDE='平台/实验工作台_v4/空间小头对照证据查阅_v1/使用与恢复.md'


def verified_spatial_assets(root):
    root=Path(root).resolve();assets=SpatialEvidenceReader(root).assets()
    for name,expected in assets.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file():raise ValueError('空间证据缺失：'+name)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('空间证据与原封存SHA不一致：'+name)
        if p.suffix in ('.pt','.npz','.safetensors') or name.startswith(('models/','DATA/raw_data/')):raise ValueError('查阅资产中混入模型/特征或原数据库')
    return assets


def require_spatial_guide_links(root,files):
    root=Path(root).resolve();p=root/GUIDE;links=[]
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        if '://' in target or target.startswith('#'):continue
        path=(p.parent/target).resolve()
        if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in files:raise ValueError('空间查阅指南链接未完整携带：'+target)
        links.append(path.relative_to(root).as_posix())
    if len(links)<3:raise ValueError('空间查阅指南缺少恢复/源码入口')
    return links
