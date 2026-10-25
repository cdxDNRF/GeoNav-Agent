"""Alternate CV-023 cosine, ranks, paired counts and source-bootstrap audit; no model/raw pixels."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/单表征目标对照开发_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def check_file(item):
    p=ROOT/item['path']
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest==item['sha256'] and p.stat().st_size==item['bytes'],item['path']


def normalized(x):
    x=[float(v) for v in x]
    norm=math.sqrt(math.fsum(v*v for v in x))
    assert norm>1e-12 and all(math.isfinite(v) for v in x)
    return [v/norm for v in x]


def dot(x,y):return math.fsum(a*b for a,b in zip(x,y))


def rank_facts(scores,truth):
    order=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    rank=1+order.index(truth)
    return order,rank,1/rank,rank==1,rank<=5


def quantile(values,q):
    s=sorted(values);point=(len(s)-1)*q
    lo=math.floor(point);hi=math.ceil(point)
    return s[lo]+(point-lo)*(s[hi]-s[lo])


def audit(destination=None):
    protocol=read(OUT/'冻结协议.json');binding=read(OUT/'输入源码与运行环境绑定.json')
    for x in binding['files']+[binding['weight'],binding['protocol']]:check_file(x)
    protection=read(OUT/'历史与用户保护.json')
    for x in protection['files']:check_file(x)
    pre=read(ROOT/'DATA/processed_data/MM-GAG/同任务首批图像轴向预检_v1/首像素冻结协议.json')
    originals={s['source_id']:s for s in pre['sources']}
    ledger={s['source_id']:s for s in read(ROOT/'DATA/processed_data/MM-GAG/同任务来源用途账本_v1.json')['records']}
    assert len(protocol['sources'])==14
    ins={x['id']:x for x in protocol['inputs']}
    for source in protocol['sources']:
        sid=source['source_id'];o=originals[sid]
        assert ledger[sid]['representative'] and ledger[sid]['role'] in ('fit_preparation','development')
        assert ins[source['ground_id']]['raw']==o['ground'] and ins[source['ground_id']]['view']=='ground'
        if source['role']=='development':
            assert [ins[i]['raw'] for i in source['tile_ids']]==o['tiles']
            assert [Path(x['path']).name for x in o['tiles']]==[f'sat_{r}_{c}.jpg' for r in range(1,10) for c in range(1,10)]
            assert source['truth_index_evaluator_only']==40
    assert [s['source_id'] for s in protocol['sources'][9:]]==['mmgag_062','mmgag_073','mmgag_079','mmgag_039','mmgag_081']
    assert len({s['group'] for s in protocol['sources']})==14
    reuse=protocol['reuse'];check_file(reuse['feature_file'])
    ref=np.load(ROOT/reuse['feature_file']['path'],allow_pickle=False)[reuse['row']]
    assert hashlib.sha256(ref.tobytes()).hexdigest()==reuse['feature_sha256']
    cachepath=OUT/'特征/特征.sqlite'
    database=sqlite3.connect(cachepath.resolve().as_uri()+'?mode=ro',uri=True)
    assert database.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    vectors={reuse['input_id']:ref}
    for idx,blob,digest in database.execute('SELECT id,vector,digest FROM features ORDER BY id'):
        assert idx in ins and idx not in vectors
        assert len(blob)==4096 and hashlib.sha256(blob).digest()==digest
        value=np.frombuffer(blob,dtype=np.float32)
        assert np.isfinite(value).all()
        vectors[idx]=value
    database.close()
    assert set(vectors)==set(ins) and len(vectors)==protocol['unique_inputs']
    log=[json.loads(x) for x in (OUT/'编码事件.jsonl').read_text(encoding='utf-8').splitlines()]
    new_ids=set(ins)-{reuse['input_id']}
    for phase in ('before_raw_read','raw_complete','before_forward'):
        ids=[x['input_id'] for x in log if x['phase']==phase]
        assert len(ids)==len(new_ids) and set(ids)==new_ids
    completed=[x for x in log if x['phase'] in ('forward_complete','receipt_reconciled_from_cache')]
    assert len(completed)==len(new_ids) and {x['input_id'] for x in completed}==new_ids
    for x in completed:assert hashlib.sha256(vectors[x['input_id']].tobytes()).hexdigest()==x['feature_sha256']
    for x in [x for x in log if x['phase']=='before_raw_read']:assert x['raw']==ins[x['input_id']]['raw']
    assert not any(x['phase']=='failed' for x in log)
    scorebind=read(OUT/'评分输入与均值绑定.json');check_file(scorebind['cache']);check_file(scorebind['protocol'])
    loaded=read(OUT/'模型冻结核验.json')
    assert loaded['all343_tensor_SHA_before_after_equal'] and loaded['parameters']==87566465
    assert loaded['trainable_parameters']==0 and not loaded['optimizer_created'] and not loaded['gradients']
    means=[normalized(vectors[s['ground_id']]) for s in protocol['sources'][:9]]
    mean=normalized([math.fsum(row[i] for row in means)/9 for i in range(1024)])
    assert max(abs(a-b) for a,b in zip(mean,scorebind['mean_vector']))<1e-14
    dev=protocol['sources'][9:]
    targets=[normalized(vectors[s['ground_id']]) for s in dev]
    banks=[[normalized(vectors[k]) for k in s['tile_ids']] for s in dev]
    scenes=[normalized([math.fsum(row[k] for row in bank)/81 for k in range(1024)]) for bank in banks]
    detail=read(OUT/'评分明细.json');summary=read(OUT/'对照汇总.json')
    assert len(detail['grid_cases'])==20 and len(detail['scene_cases'])==20
    errors=[]
    for scope in ('grid','scene'):
        cases=detail[scope+'_cases'];seen=set()
        for c in cases:
            i=next(i for i,s in enumerate(dev) if s['source_id']==c['source'])
            key=(c['source'],c['condition']);assert key not in seen;seen.add(key)
            target={'Full':targets[i],'Mean':mean,'Wrong':targets[(i+1)%5]}.get(c['condition'])
            bank=banks[i] if scope=='grid' else scenes
            expected=[0.0]*len(bank) if c['condition']=='Disabled' else [dot(target,row) for row in bank]
            assert len(expected)==len(c['scores'])
            errors.extend(abs(a-b) for a,b in zip(expected,c['scores']))
            truth=40 if scope=='grid' else i
            assert c['truth_index_evaluator_only']==truth
            order,rank,rr,top1,top5=rank_facts(c['scores'],truth)
            assert order==c['order'] and rank==c['rank'] and rr==c['rr'] and top1==c['top1'] and top5==c['top5']
            assert c['rank_best_tie']==1+sum(x>c['scores'][truth] for x in c['scores'])
            assert c['rank_worst_tie']==sum(x>=c['scores'][truth] for x in c['scores'])
            if scope=='grid':
                assert c['selected_cell_manhattan_error_not_SG']==abs(order[0]//9-4)+abs(order[0]%9-4)
        for condition in ('Full','Disabled','Mean','Wrong'):
            subset=[c for c in cases if c['condition']==condition];assert len(subset)==5
            stored=summary['screen'][scope][condition]
            assert stored['planned']==5 and stored['top1_hits']==sum(x['top1'] for x in subset)
            assert stored['top5_hits']==sum(x['top5'] for x in subset)
            assert abs(stored['MRR']-math.fsum(x['rr'] for x in subset)/5)<1e-14
            assert stored['mean_rank']==sum(x['rank'] for x in subset)/5
    assert max(errors)<1e-12
    old=read(ROOT/'DATA/processed_data/MM-GAG/场景与精确格信号分解_v1/格级已存分数重组.json')
    for c in detail['old_identity_reference']:
        previous=[x for x in old['cases'] if x['source']==c['source'] and x['condition']==c['condition'] and x['arm']=='Identity']
        expected=[0.0]*81 if c['condition']=='Disabled' else previous[0]['scores']
        assert c['scores']==expected and c['rank']==rank_facts(expected,40)[1]
    delta_checks=0
    for scope in ('grid','scene'):
        cases=detail[scope+'_cases']
        for pair in summary['paired_'+scope]:
            full={c['source']:c for c in cases if c['condition']=='Full'}
            control={c['source']:c for c in cases if c['condition']==pair['control']}
            assert pair['independent_units']==5 and pair['bootstrap_draws']==3125
            per=pair['per_source']
            for item in per:
                a=full[item['source']];b=control[item['source']]
                assert item['rank_improvement']==b['rank']-a['rank']
                assert item['MRR_delta']==a['rr']-b['rr']
                assert item['top1_delta']==int(a['top1'])-int(b['top1'])
            assert pair['recovered_top1']==sum(item['top1_delta']==1 for item in per)
            assert pair['damaged_top1']==sum(item['top1_delta']==-1 for item in per)
            for metric in ('rank_improvement','MRR_delta','top1_delta'):
                values=[x[metric] for x in per]
                resamples=[math.fsum(values[i] for i in row)/5 for row in itertools.product(range(5),repeat=5)]
                bounds=pair['source_bootstrap'][metric]['CI95_percentile']
                assert max(abs(a-b) for a,b in zip(bounds,[quantile(resamples,.025),quantile(resamples,.975)]))<1e-12
                delta_checks+=1
    grid={(x['source'],x['condition']):x for x in detail['grid_cases']}
    gm=summary['screen']['grid'];sm=summary['screen']['scene']
    dual=sum(grid[s['source_id'],'Full']['rank']<grid[s['source_id'],'Mean']['rank'] and grid[s['source_id'],'Full']['rank']<grid[s['source_id'],'Wrong']['rank'] for s in dev)
    gates=[gm['Full']['top1_hits']>=1,dual>=3,all(gm['Full']['MRR']>gm[c]['MRR'] for c in ('Disabled','Mean','Wrong')),
           all(gm['Full']['top1_hits']>=gm[c]['top1_hits'] for c in ('Disabled','Mean','Wrong')),
           sm['Full']['top1_hits']>=3 and all(sm['Full']['top1_hits']-sm[c]['top1_hits']>=2 for c in ('Mean','Wrong'))]
    assert gates==[x['passed'] for x in summary['screen']['gates']]
    assert summary['screen']['all_effect_gates_passed']==all(gates)
    result=read(OUT/'执行结果.json')
    assert result['new_encoded_inputs']==len(new_ids) and result['confirmation_access']==result['training']==result['navigation']==0
    assert result['new_SR'] is None and result['elapsed_seconds']<=1800
    features=sum(p.stat().st_size for p in (OUT/'特征').iterdir() if p.is_file())
    assert features<=2*1024**2 and sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<8*1024**2
    report={'task':'CV-023','passed':True,'reviewer':'same Agent, alternate equations/counts, not independent personnel',
        'binding_records_checked':len(binding['files'])+2,'historical_protection_records_checked':len(protection['files']),
        'users_protected':protection['users_count'],'unique_features':len(vectors),'new_encoded_inputs':len(new_ids),'reused':1,
        'grid_score_slots_recomputed':1620,'scene_score_slots_recomputed':100,'old_score_slots_referenced':1215,
        'max_cosine_error':max(errors),'paired_source_CI_checks':delta_checks,'grid_and_scene_cases':40,
        'independent_units':5,'no_raw_rereads':True,'no_torch_import_or_model_forward':True,
        'confirmation_access':0,'training':0,'navigation':0,'new_SR':None,'feature_bytes':features,
        'all_effect_gates_passed':all(gates),'dual_control_rank_sources':dual}
    if destination:
        with Path(destination).open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--save');audit(parser.parse_args().save)
