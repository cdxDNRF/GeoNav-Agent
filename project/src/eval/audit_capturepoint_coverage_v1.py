"""CV-018 alternate pixel-cache equations and protocol accounting, no models/raw pixels."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'三新增来源工程与配对覆盖协议_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def verify(items):
    for e in items:
        data=(ROOT/e['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==e['sha256'] and len(data)==e['bytes'],e['path']


def save(name,value):
    p=OUT/'核验'/name
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(value,ensure_ascii=False))


def engineering():
    b=read(OUT/'输入与源码绑定.json');verify(b['inputs']+b['sources']+b['readonly_math_dependencies']+b['protected']+[b['protocol'],b['authorization']])
    scope=read(OUT/'首像素冻结范围.json');summary=read(OUT/'像素工程汇总.json')
    assert summary['planned_sources']==3 and summary['raw_reads']==249 and summary['raw_decodes']==249
    expected={e['path']:e for e in scope['raw_expected']};assert len(expected)==249
    events=[json.loads(line) for line in (OUT/'输入消费.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(events)==498
    for kind in ('attempt','decoded'):assert Counter(e['path'] for e in events if e['event']==kind)==Counter({p:1 for p in expected})
    allreceipts=[];maxdiff=0
    transforms={
        'row':lambda r,c:(r,c),'column':lambda r,c:(c,r),'flip_row':lambda r,c:(10-r,c),
        'flip_column':lambda r,c:(r,10-c),'rotate180':lambda r,c:(10-r,10-c),
        'rotate90':lambda r,c:(c,10-r),'rotate270':lambda r,c:(10-c,r),'antitranspose':lambda r,c:(10-c,10-r)}
    for s,reported in zip(scope['sources'],summary['sources']):
        assert s['source_id']==reported['source_id'] and s['role']=='fit_preparation'
        receipt=read(OUT/'核验'/f"{s['source_id']}_回执.json");assert receipt['result']==reported
        assert len(receipt['receipts'])==83;allreceipts.extend(receipt['receipts'])
        verify([reported['thumbnail']])
        with np.load(ROOT/reported['thumbnail']['path'],allow_pickle=False) as cache:
            ground=cache['ground'].astype(np.float64);tiles=cache['tiles'].astype(np.float64);probes=cache['probes'].astype(np.float64)
        assert ground.shape==(16,16,3) and tiles.shape==(81,16,16,3) and probes.shape==(8,16,16,3)
        values={}
        for name,transform in transforms.items():
            indices=[]
            for r,c in scope['axis_probes']:
                y,x=transform(r,c);indices.append((y-1)*9+x-1)
            value=float(np.abs(tiles[indices]-probes).reshape(-1).mean());values[name]=value
            maxdiff=max(maxdiff,abs(value-reported['axis']['scores'][name]))
        order=sorted(values,key=lambda k:(values[k],k));margin=values[order[1]]-values[order[0]]
        assert maxdiff<1e-12 and order[0]==reported['axis']['best'] and abs(margin-reported['axis']['margin'])<1e-12
        def std_max(a):
            v=a.reshape(-1,3);return float(np.sqrt(((v-v.mean(axis=0))**2).sum(axis=0)/len(v)).max())
        gs,cs=std_max(ground),std_max(tiles[40]);assert abs(gs-reported['ground_std16_max'])<1e-12 and abs(cs-reported['center_std16_max'])<1e-12
        passed=order[0]=='row' and values['row']<=8 and margin>=3 and gs>1 and cs>1
        assert passed==reported['passed']
    assert len(allreceipts)==249 and {x['path'] for x in allreceipts}==set(expected)
    for r in allreceipts:assert r['sha256']==expected[r['path']]['sha256'] and r['bytes']==expected[r['path']]['bytes']
    assert summary['all_required_passed']==all(r['passed'] for r in summary['sources'])
    save('工程另实现复核.json',{'task':'CV-018','passed':True,'all_sources_engineering_passed':summary['all_required_passed'],
        'sources':3,'raw_receipts':249,'events':498,'axis_scores':24,'max_axis_score_difference':maxdiff,
        'raw_images_redecoded_by_auditor':0,'new_model_calls':0,'training_steps':0,'confirmation_access':0,'independent_person_review':False})


def protocol():
    binding=read(OUT/'协议源码与输入绑定.json');verify(binding['sources']+binding['readonly_math_dependencies']+binding['inputs']+[binding['protocol'],binding['queue'],binding['engineering'],binding['engineering_audit']])
    p=read(OUT/'配对覆盖同预算冻结协议.json');q=read(OUT/'固定训练与开发队列.json');old=read(BASE/'同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json')
    scope=read(OUT/'首像素冻结范围.json')
    assert q['development_tasks']==old['development_tasks'] and q['ground_encoder_inputs'][:14]==old['ground_encoder_inputs'] and q['aerial_encoder_inputs'][:1134]==old['aerial_encoder_inputs']
    assert len(q['ground_encoder_inputs'])==17 and len(q['aerial_encoder_inputs'])==1377 and len(q['training_queries'])==12
    assert q['training_queries'][:9]==old['training_queries'] and len({t['group_evaluator_only'] for t in q['training_queries']})==9
    assert [t['source_id_evaluator_only'] for t in q['training_queries']]==p['ordered12_training_sources']
    for j,s in enumerate(scope['sources']):
        t=q['training_queries'][9+j];assert t['source_id_evaluator_only']==s['source_id'] and t['group_evaluator_only']==s['group']
        assert t['ground_feature_index']==14+j and t['aerial_feature_indices']==list(range(1134+81*j,1215+81*j)) and t['positive_candidate_index_evaluator_only']==40
        assert q['ground_encoder_inputs'][14+j]==s['ground'] and q['aerial_encoder_inputs'][1134+81*j:1215+81*j]==s['tiles']
    phases=[list(range(9)),[9,10,11,0,1,2,3,4,5],[6,7,8,9,10,11,0,1,2],list(range(3,12))]
    counts=Counter(i for step in range(500) for i in phases[step%4]);assert counts==Counter({k:375 for k in range(12)})
    assert p['sampler']['mathematically_verified_uses']==[counts[i] for i in range(12)] and p['sampler']['total_query_uses']==4500
    assert p['training']['updates']==500 and p['head']['trainable_parameters']==8192
    assert q['mean_ground_feature_indices']==list(range(9)) and p['mean_ground_indices']==list(range(9))
    assert q['planned_rankings']==1625 and q['reuse_original_rankings']==1250 and q['new_rankings']==375
    assert len(q['development_tasks'])*4*3+125==1625 and len({t['group_evaluator_only'] for t in q['development_tasks']})==5
    oldp=read(BASE/'同任务对齐训练与目标对照协议_v1/同任务对齐训练冻结协议.json')
    expected={k:v for k,v in oldp['required_development_gates'].items() if k!='all1250_rankings_and_all_audits'}
    assert p['evaluation']['required_effect_thresholds_from_CV012']==expected
    assert p['research_design_frozen'] and p['source_engineering_passed'] and not p['training_authorized'] and not p['model_forward_authorized']
    assert p['actual_executor_binding'] is None and p['actual_feature_binding'] is None and p['candidate_final_checkpoint'] is None
    assert p['current_activity']['new_training_steps']==0 and p['current_activity']['new_model_forward_calls']==0
    save('协议另实现复核.json',{'task':'CV-018','passed':True,'training_queries':12,'fit_source_groups':9,'known_dev_queries_sources':5,
        'development_windows':125,'planned_rankings':1625,'old_prefix_ground_aerial_preserved':[14,1134],
        'feature_reference_counts_only':[17,1377],'sampler_uses':4500,'uses_per12_query':375,'source_group_independence_not_increased':True,
        'actual_features_read':0,'model_calls':0,'training_steps':0,'confirmation_access':0,'independent_person_review':False})


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('engineering','protocol'):raise SystemExit('engineering | protocol; each once')
    {'engineering':engineering,'protocol':protocol}[sys.argv[1]]()
