"""Freeze BENCH-012 before arithmetic; verify or seal new delivery only."""
import argparse
import ast
import json
import sys
from pathlib import Path
from eval.compile_geoexplorer_evidence_close_v1 import ROOT,BATCH,INPUTS,SEAL_BATCHES,read,sha,write_new,verify_binding

SOURCES=[
    'project/src/tools/prepare_geoexplorer_evidence_close_v1.py',
    'project/src/eval/compile_geoexplorer_evidence_close_v1.py',
    'project/src/eval/audit_geoexplorer_evidence_close_v1.py',
    'project/src/tests/test_geoexplorer_evidence_close_v1.py',
    'project/src/documents/build_geoexplorer_evidence_close_v1.py']


def forbidden_import_check(path):
    parsed=ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in ast.walk(parsed):
        if isinstance(node,ast.Import): names=[n.name for n in node.names]
        elif isinstance(node,ast.ImportFrom): names=[node.module or '']
        else: continue
        if any(n.split('.')[0] in ['torch','transformers','train','agents','requests','httpx','urllib','subprocess'] for n in names):
            raise ValueError('forbidden dependency in static task: '+str(path))


def prepare():
    if BATCH.exists(): raise ValueError('occupied batch; do not prepare again')
    bindings={p:sha(ROOT/p) for p in INPUTS.values()}
    seals={}
    for directory in SEAL_BATCHES:
        seal_path=ROOT/directory/'核验/阶段封存_v1.json'
        bindings[seal_path.relative_to(ROOT).as_posix()]=sha(seal_path)
        files=read(seal_path)['files']
        for p,h in files.items():
            name=(Path(directory)/p).as_posix()
            if sha(ROOT/name)!=h: raise ValueError('old sealed file changed: '+name)
            bindings[name]=h
        seals[directory]=len(files)
    for source in SOURCES:
        forbidden_import_check(ROOT/source); bindings[source]=sha(ROOT/source)
    for sub in ['元数据','核验','源码快照']:(BATCH/sub).mkdir(parents=True,exist_ok=False)
    for source in SOURCES:
        dest=BATCH/'源码快照'/Path(source).name
        with dest.open('xb') as stream:stream.write((ROOT/source).read_bytes())
        bindings[dest.relative_to(ROOT).as_posix()]=sha(dest)
    protocol={'task':'BENCH-012','date':'2026-10-08','mode':'existing-evidence and coordinate arithmetic only',
              'bindings':bindings,'historical_seal_counts':seals,'sources':SOURCES,
              'grid':5,'near_tolerance_l2':1e-6,'phase_multipliers':[2,1],
              'factor_selection_rule':'retain at most half-circle coordinate variant as untested hypothesis if analytic boundary alias and FP32 near pairs agree, half-circle removes aliases; no causal/performance claim',
              'full_model_input_alias_assumed':False,'inference_allowed':False,'training_allowed':False,
              'navigation_allowed':False,'new_sr':False,'reserve_consumption':0,'api_calls':0,
              'wall_seconds_cap':120,'output_bytes_cap':8*1024*1024,
              'historical_weights_reread':False,
              'historical_verification_scope':'stage seals, bound selected reports/audits/protocol/source/default bytes; no new full checkpoint/log audit',
              'numpy_version':__import__('numpy').__version__}
    write_new(BATCH/'元数据/冻结收束协议_v1.json',protocol)
    print(json.dumps({'frozen_bindings':len(bindings),'historical_seals':seals},ensure_ascii=False))


def verify():
    protocol=read(BATCH/'元数据/冻结收束协议_v1.json');verify_binding(protocol)
    for source in SOURCES:forbidden_import_check(ROOT/source)
    for p in BATCH.rglob('*'):
        if p.is_file() and p.suffix=='.json':read(p)
    for forbidden in ['torch','transformers']:
        if forbidden in sys.modules:raise ValueError('unexpected model module loaded')
    seal=BATCH/'核验/阶段封存_v1.json'
    if seal.exists():
        for p,h in read(seal)['files'].items():
            if sha(BATCH/p)!=h:raise ValueError('new delivery changed: '+p)
    size=sum(p.stat().st_size for p in BATCH.rglob('*') if p.is_file())
    if size>protocol['output_bytes_cap']:raise ValueError('output cap exceeded')
    return {'status':'passed','bindings_checked':len(protocol['bindings']),
            'historical_files_checked':sum(protocol['historical_seal_counts'].values()),
            'sources_checked':len(SOURCES),'new_batch_bytes':size,'model_imports':0,'model_forward':0,
            'new_training':0,'new_navigation':0,'new_sr':False,'api_calls':0,'reserve_consumption':0,
            'independent_person_review':False,'full_checkpoint_audit_repeated':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','verify','seal']);args=parser.parse_args()
    if args.mode=='prepare':prepare();return
    result=verify()
    if args.mode=='seal':
        if read(BATCH/'核验/另实现复核_v1.json')['status']!='passed':raise ValueError('audit required')
        write_new(BATCH/'核验/交付核验_v1.json',result)
        files={p.relative_to(BATCH).as_posix():sha(p) for p in sorted(BATCH.rglob('*')) if p.is_file()}
        write_new(BATCH/'核验/阶段封存_v1.json',{'status':'completed','files':files})
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
