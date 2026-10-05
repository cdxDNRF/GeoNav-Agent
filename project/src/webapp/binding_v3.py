"""Read-only dependency fingerprinting; never imports scientific/model code."""
import ast
from pathlib import Path
from .live_v2 import DEV, read, sha


def source_closure(root, starts):
    root=Path(root);base=root/'project/src';pending=[root/p for p in starts];seen={}
    def resolve(module):
        module=module.removeprefix('project.src.')
        p=base.joinpath(*module.split('.'))
        if p.with_suffix('.py').is_file():return p.with_suffix('.py')
        if (p/'__init__.py').is_file():return p/'__init__.py'
        return None
    while pending:
        p=pending.pop();key=p.relative_to(root).as_posix()
        if key in seen:continue
        seen[key]=sha(p)
        for parent in p.parents:
            if parent==base.parent:break
            init=parent/'__init__.py'
            if init.is_file():pending.append(init)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    parent=p.parent
                    for _ in range(node.level-1):parent=parent.parent
                    prefix=parent.relative_to(base).as_posix().replace('/','.')
                    name=prefix+('.'+node.module if node.module else '')
                else:name=node.module or ''
                modules=[name]+[name+'.'+a.name for a in node.names]
            for module in modules:
                found=resolve(module)
                if found:pending.append(found)
    return seen


def validate_bindings(root, inputs, sources):
    for relative, expected in {**inputs,**sources}.items():
        if sha(Path(root)/relative)!=expected:raise ValueError('实际执行依赖SHA已变化，请重新登记新运行')


def execution_bindings(root):
    root=Path(root);folder=root/DEV;registration=read(folder/'核验/预登记.json')
    inputs={}
    # Include the original image registration itself, actual data geometry,
    # tasks and every accessible g5/g10 image, never expose this to the policy.
    for relative,expected in registration['input_sha256'].items():
        if relative.startswith(DEV+'/工程数据/grid5/') or relative.startswith(DEV+'/工程数据/grid10/') or relative in (
            DEV+'/元数据/grid5任务.json',DEV+'/元数据/grid10任务.json',DEV+'/元数据/原冻结模型配置.json'):
            inputs[relative]=expected
    for relative in (DEV+'/核验/预登记.json',DEV+'/核验/特征完成.json',
        'project/local_policy_default.json','project/local_policy_defaults_v1.json',
        'project/defaults/masa_roads_grid10_radial_v1.json'):
        inputs[relative]=sha(root/relative)
    inputs[DEV+'/特征/native225.npz']=read(folder/'核验/特征完成.json')['sha256']
    config=read(folder/'元数据/原冻结模型配置.json')
    for rec in config['checkpoints']+config['cue_heads']+list(config['means'].values()):
        inputs[rec['path']]=rec['sha256']
    sources=source_closure(root,['project/src/webapp/server_v3.py'])
    validate_bindings(root,inputs,sources)
    return inputs,sources
