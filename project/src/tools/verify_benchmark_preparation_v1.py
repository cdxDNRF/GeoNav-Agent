"""One-shot CPU delivery: bind assets/code, test and replay synthetic records.

Outputs are exclusive. Re-run without --save only reads existing evidence.
Never loads a scientific model or starts navigation on the real task queue.
"""
import ast
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from agents.benchmark_contract_v1 import ScriptedPolicy
from eval.benchmark_runner_v1 import Payload, Task, FitMean
from eval.benchmark_runner_v2 import RealPlan, RegisteredPolicy, evaluate, hashed
from eval.audit_benchmark_runner_v2 import audit
from tools.benchmark_assets_v1 import build, bundle, digest, verify_bindings, canonical

ROOT = SRC.parents[1]
BATCH = ROOT / '项目导航/共同真实评测执行准备_v2'
PRIMARY = ('tools/benchmark_assets_v1.py', 'eval/audit_benchmark_runner_v2.py',
           'tests/test_benchmark_assets_audit_v1.py', 'tools/verify_benchmark_preparation_v1.py')


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def local_closure(paths):
    """AST inventory of local imports, including lazy ones; no import execution.

    Does not claim to resolve dynamic module loaders or third-party libraries.
    Actual-model closure and environment are a later GPU/formal gate.
    """
    found = {}; todo = list(paths)
    while todo:
        path = todo.pop().resolve()
        if not path.is_file() or path in found: continue
        if SRC not in path.parents: raise ValueError('source outside project/src')
        found[path] = digest(path)
        for node in ast.walk(ast.parse(path.read_text('utf-8'))):
            names = []
            if isinstance(node, ast.Import): names = [(SRC, n.name) for n in node.names]
            if isinstance(node, ast.ImportFrom):
                base = path.parent
                if node.level:
                    for _ in range(node.level-1): base = base.parent
                else: base = SRC
                if node.module: names.append((base, node.module))
                else: names += [(base, n.name) for n in node.names]
            for base, name in names:
                p = base.joinpath(*name.split('.'))
                candidate = p.with_suffix('.py')
                if candidate.is_file(): todo.append(candidate)
                elif (p/'__init__.py').is_file(): todo.append(p/'__init__.py')
    return {p.relative_to(ROOT).as_posix(): h for p, h in sorted(found.items())}


def fake_assets():
    import numpy as np
    def feat(n):
        return bundle(np.full(512,n,np.float32),np.full((4,768),n,np.float32),
                      np.full((4,3,64,3),n,np.float32))
    cells = {a: {i: Payload(f'{a}-cell-{i}'.encode(), feat(i+1)) for i in range(25)} for a in ('one','two')}
    tasks = tuple(Task(f'{a}-{j}',a,5*j,5*j+1,24,'C1') for a in cells for j in (0,1))
    means = {m: FitMean(Payload(b'artificial-fit-placeholder',feat(n)), 'a'*64)
             for m,n in [('M0',17),('GeoExplorer',18)]}
    return RealPlan(tasks,'synthetic_interface'),cells,means


def prepare():
    marker = BATCH/'核验/执行准备绑定_v1.json'
    if marker.exists() or (BATCH/'核验/合成回归_v1').exists():
        raise FileExistsError('preparation already exists; immutable outputs cannot be repeated')
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise ValueError('CPU preparation must not import tensor/model libraries')
    data = build(ROOT)
    evidence = BATCH/'核验/资产复核_v1.json'
    prior = json.loads(evidence.read_text('utf-8'))
    if prior != data.evidence: raise ValueError('asset evidence changed after its first exclusive save')
    bindings = local_closure([SRC/p for p in PRIMARY])
    bindings.update(data.evidence['bindings'])
    bindings[evidence.relative_to(ROOT).as_posix()] = digest(evidence)
    bindings['project/local_policy_defaults_v1.json'] = '8e1b85eb74021ac18f8246f8146f7db451d68f423d7152f6c3a5d7c86b4d16fa'
    bindings['项目导航/共同真实评测准备_v1/共同对照冻结草案.md'] = digest(ROOT/'项目导航/共同真实评测准备_v1/共同对照冻结草案.md')
    verify_bindings(ROOT, bindings)
    save(marker, dict(stage='CPU_preparation', date=datetime.now().astimezone().isoformat(),
        scientific_execution_allowed=False, bindings=bindings,
        primary_sources=list(PRIMARY), local_import_closure_only=True,
        excludes_dynamic_original_model_and_environment=True,
        pending=['complete seed2 and full training audit','actual public GPU loader and M0 trajectory regression',
                 'final weight/dynamic code closure/environment and real execution protocol']))
    test = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','project/src/tests',
                           '-p','test_benchmark_assets_audit_v1.py','-v'],cwd=ROOT,capture_output=True,text=True)
    save(BATCH/'核验/CPU测试记录_v1.json',dict(command=test.args,exit_code=test.returncode,
        stdout=test.stdout,stderr=test.stderr,source_bindings={p:digest(SRC/p) for p in PRIMARY},
        real_model_forward=0,real_navigation=0))
    if test.returncode: raise RuntimeError('CPU boundary tests failed; preserve outputs')
    artifact = BATCH/'核验/假权重标识_v1.txt'
    with artifact.open('xb') as stream: stream.write(b'CPU synthetic artifact, no scientific weights\n')
    plan, cells, means = fake_assets()
    regs = [RegisteredPolicy(m,seed,
        (lambda: ScriptedPolicy(('right',))) if m=='M0' else (lambda: ScriptedPolicy(('down','up')*4+('right',))),
        ((artifact,digest(artifact)),), tuple((ROOT/p,h) for p,h in bindings.items() if p.startswith('project/src/')),
        means[m],means[m].fit_manifest_sha256)
        for m in ('M0','GeoExplorer') for seed in (0,1,2)]
    trial = BATCH/'核验/合成回归_v1'
    evaluate(plan,cells,regs,hashed(plan.record()),trial)
    result = audit(trial,plan,cells,means)
    save(BATCH/'核验/合成独立复核_v1.json',result)
    for p in PRIMARY:
        source = SRC/p; target = BATCH/'源码快照'/source.name
        with target.open('xb') as stream: stream.write(source.read_bytes())
    verify_bindings(ROOT, bindings)
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise ValueError('unexpected tensor/model import in preparation')
    save(BATCH/'核验/准备完成_v1.json',dict(status='CPU_preparation_passed_real_execution_pending',
        date=datetime.now().astimezone().isoformat(),bindings_checked=len(bindings),
        snapshots_checked={p.name:digest(p) for p in (BATCH/'源码快照').iterdir()},
        synthetic_episodes=result['episodes_checked'],synthetic_public_decisions=result['public_decisions_checked'],
        source_file_groups_real=10,real_tasks=250,real_patches=250,
        planned_real_episodes=6000,new_model_forward=0,new_real_navigation=0,new_training=0,cloud_calls=0,
        reserve_regions_consumed=0,scientific_sr=False,training_completed=False,
        independent_person_review=False))
    return dict(status='CPU_preparation_passed_real_execution_pending',bindings=len(bindings),
                synthetic_episodes=result['episodes_checked'],synthetic_actions=result['public_decisions_checked'])


def check_existing():
    marker = json.loads((BATCH/'核验/执行准备绑定_v1.json').read_text('utf-8'))
    verify_bindings(ROOT,marker['bindings'])
    done = json.loads((BATCH/'核验/准备完成_v1.json').read_text('utf-8'))
    for name,h in done['snapshots_checked'].items():
        if digest(BATCH/'源码快照'/name)!=h: raise ValueError('source snapshot drift')
    plan,cells,means=fake_assets()
    result=audit(BATCH/'核验/合成回归_v1',plan,cells,means)
    return dict(status='existing_CPU_preparation_verified',bindings=len(marker['bindings']),
                synthetic_episodes=result['episodes_checked'],synthetic_actions=result['public_decisions_checked'],
                scientific_sr=False,model_forward=0,new_real_navigation=0)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--save',action='store_true');args=parser.parse_args()
    print(json.dumps(prepare() if args.save else check_existing(),ensure_ascii=False))
