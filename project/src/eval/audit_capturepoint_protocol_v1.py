"""CV-012 alternate metadata/task/gate audit. No raw images or model execution."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.capturepoint_alignment_protocol_v1 import gate
BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'同任务对齐训练与目标对照协议_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    assert not Path(p).resolve().is_relative_to((ROOT/'DATA/raw_data').resolve())
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def audit(save=False):
    b=read(OUT/'输入与协议源码绑定.json');cache={};checked=0
    def verify(items):
        nonlocal checked
        for e in items:
            if e['path'] not in cache:cache[e['path']]=sha(ROOT/e['path'])
            assert cache[e['path']]==e['sha256'],e['path'];checked+=1
    for k in ['inputs','sources','reference_math_sources_only','previous_seals','protected']:verify(b[k])
    verify([b['protocol']]);old_items=0
    for e in b['previous_seals']:
        s=read(ROOT/e['path']);verify(s['files']+s['sources']);old_items+=len(s['files'])+len(s['sources'])
    for e in b['sources']:
        for n in ast.walk(ast.parse((ROOT/e['path']).read_text(encoding='utf-8-sig'))):
            if isinstance(n,ast.Import):assert not any(a.name.split('.')[0] in {'torch','transformers','safetensors','numpy','PIL','requests','httpx','subprocess','socket','importlib'} for a in n.names)
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in {'torch','transformers','safetensors','numpy','PIL'}
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in {'eval','exec','__import__'}
    p=read(OUT/'同任务对齐训练冻结协议.json');verify([p['queue']]);q=read(ROOT/p['queue']['path'])
    pre=read(BASE/'同任务首批图像轴向预检_v1/首像素冻结协议.json')['sources']
    roles=read(BASE/'同任务来源用途账本_v1.json')['records'];by_role={r['source_id']:r for r in roles}
    assert len(q['ground_encoder_inputs'])==14 and len(q['aerial_encoder_inputs'])==1134
    for i,s in enumerate(pre):
        assert q['ground_encoder_inputs'][i]==s['ground'] and q['aerial_encoder_inputs'][i*81:(i+1)*81]==s['tiles']
        assert by_role[s['source_id']]['role']==('fit_preparation' if i<9 else 'development')
        assert by_role[s['source_id']]['representative'] and len({v['sha256'] for v in s['tiles']})==81
    assert len(q['training_queries'])==9
    for i,t in enumerate(q['training_queries']):
        assert t['source_id_evaluator_only']==pre[i]['source_id'] and t['group_evaluator_only']==pre[i]['group']
        assert t['ground_feature_index']==i and t['positive_candidate_index_evaluator_only']==40
        assert t['aerial_feature_indices']==list(range(i*81,(i+1)*81))
        assert pre[i]['tiles'][40]['path'].endswith('/sat_5_5.jpg')
    tasks=q['development_tasks'];assert len(tasks)==125
    offset=0
    for i in range(9,14):
        s=pre[i];goals=sorted(range(25),key=lambda g:hashlib.sha256(('CV-012/goal-order/v1:'+s['group']+':'+str(g)).encode()).hexdigest())
        path_to_index={e['path']:i*81+j for j,e in enumerate(s['tiles'])};prefix=s['tiles'][0]['path'].rsplit('/',1)[0]
        for goal in goals:
            t=tasks[offset];assert t['pair_id']==f'cptd_{offset:03d}';offset+=1
            assert t['source_id_evaluator_only']==s['source_id'] and t['group_evaluator_only']==s['group']
            assert t['ground_feature_index']==i and t['wrong_ground_feature_index']==(i+1 if i<13 else 9)
            assert t['goal_index_evaluator_only']==goal and t['not_navigation']
            y,x=divmod(goal,5)
            names=[prefix+f'/sat_{r}_{c}.jpg' for r in range(5-y,10-y) for c in range(5-x,10-x)]
            indices=[path_to_index[name] for name in names]
            assert t['aerial_feature_indices']==indices and indices[goal]==i*81+40
    used={s['source_id'] for s in pre};withheld={r['source_id'] for r in roles if r['role']=='confirmation_reserved'}
    assert len(used)==14 and len(withheld)==24 and not used&withheld
    assert q['planned_rankings']==125*(3*3+1)==1250 and q['shared_disabled_target_records']==125
    assert p['evaluation']['planned_rankings']==1250 and p['evaluation']['planned_windows']==125 and p['evaluation']['unique_development_sources']==5
    assert p['evaluation']['arms']==['Identity','SwissRank8Frozen','MMRank8Final'] and p['evaluation']['conditions_per_arm']==['Full','Mean','Wrong']
    assert p['head']['trainable_parameters']==8192 and p['head']['rank']==8 and p['head']['U_init']=='zeros'
    train=p['training'];old=read(BASE/'配对派生清单与弱目标准入_v1/训练设计冻结.json')['training']
    for k in ['seed','unique_ground_queries','aerial_candidates_per_query','updates','temperature','optimizer','learning_rate','weight_decay','gradient_clip_norm','schedule']:assert train[k]==old[k]
    assert train['source_ids_evaluator_only']==[s['source_id'] for s in pre[:9]]
    assert train['source_groups']==[s['group'] for s in pre[:9]]
    for a,f in [('Identity','initial.npz'),('SwissRank8Frozen','final.npz')]:
        weight=p['baselines'][a]['weight'];assert weight['path']==f'DATA/processed_data/MM-GAG/弱场景对齐头工程与训练_v3/权重/{f}';verify([weight])
    assert p['baselines']['MMRank8Final']['weight'] is None
    assert 'mean of9 normalized MM fit' in p['controls']['Mean'] and 'head is not called' in p['controls']['Disabled']
    thresholds=p['required_development_gates'];assert thresholds=={'all1250_rankings_and_all_audits':True,'Full_top1_minimum':32,
        'Full_minus_Mean_minimum':16,'Full_minus_Wrong_minimum':16,'Full_minus_Disabled_minimum':16,
        'Full_minus_Identity_minimum':5,'Full_minus_SwissFrozen_minimum':5,
        'source_groups_Full_above_both_controls_minimum':3,'source_groups_Full_above_both_baselines_minimum':3}
    # Alternate integer gate equations checked on several synthetic count arrangements.
    cases=[]
    for full,mean,swiss in [([15]*5,[5]*5,[10]*5),([6,6,6,6,7],[0]*5,[1]*5),
                            ([25,25,0,0,0],[0]*5,[0]*5),([10]*5,[7]*5,[9]*5),([10]*5,[7,7,7,7,6],[9]*5)]:
        counts={str(i):{'planned':25,'Identity':{'Full':min(1,full[i]),'Mean':0,'Wrong':0},
            'SwissRank8Frozen':{'Full':swiss[i],'Mean':0,'Wrong':0},
            'MMRank8Final':{'Full':full[i],'Mean':mean[i],'Wrong':0},'Disabled':0} for i in range(5)}
        actual=gate(counts,complete=True,engineering_and_audit=True)
        positive_controls=sum(full[i]>mean[i] and full[i]>0 for i in range(5))
        positive_bases=sum(full[i]>counts[str(i)]['Identity']['Full'] and full[i]>swiss[i] for i in range(5))
        expected=sum(full)>=32 and sum(full)-sum(mean)>=16 and sum(full)>=16 and sum(full)-sum(swiss)>=5 and sum(full)-sum(c['Identity']['Full'] for c in counts.values())>=5 and positive_controls>=3 and positive_bases>=3
        assert actual['passed']==expected;cases.append(expected)
    assert p['resources']['maximum_encoder_inputs']==1148 and p['resources']['maximum_encoder_forward_calls']==144
    assert p['resources']['head_training_device']=='cpu' and p['resources']['maximum_CUDA_allocated_bytes']==2147483648
    assert p['resources']['whole_batch_outputs_limit_bytes']==41943040
    assert p['research_design_frozen'] and not p['execution_admitted'] and not p['training_authorized'] and not p['training_started']
    assert p['actual_executor_binding'] is None and p['actual_feature_binding'] is None
    assert p['confirmation']['groups']==6 and p['confirmation']['sources']==24 and not p['confirmation']['pixel_or_model_access_this_protocol']
    assert all(v==0 for v in p['current_task_activity'].values())
    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<=3*1024**2
    result={'task':'CV-012','passed':True,'binding_checks':checked,'old_sealed_items':old_items,
        'training_queries':9,'development_sources':5,'windows_verified':125,'candidate_references_verified':125*25,
        'encoder_input_references':1148,'planned_rankings':1250,'synthetic_gate_cases':len(cases),
        'raw_image_reads':0,'model_calls':0,'training_steps':0,'new_predictions':0,'confirmation_pixels':0,
        'independent_person_review':False,'method':'same Agent alternate filename arithmetic, role/reference/cardinality reconstruction and integer gate equations'}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
