"""CV-003 alternate equations and counts; saved arrays only, no model runtime."""
import argparse,collections,hashlib,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/局部语义匹配单因素验证_v1'
PREV=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def unit(a):
    a=np.array(a,dtype=np.float64);assert a.ndim==2 and np.isfinite(a).all()
    d=np.sqrt(np.einsum('ij,ij->i',a,a));assert (d>1e-12).all()
    return a/d[:,None]
def score(a,b):
    matrix=np.einsum('ik,jk->ij',unit(a),unit(b),optimize=False)
    forward=sum(float(max(row)) for row in matrix)/matrix.shape[0]
    backward=sum(float(max(col)) for col in matrix.T)/matrix.shape[1]
    return (forward+backward)/2
def audit(save=False):
    state=read(OUT/'执行状态.json');assert state['state']=='completed'
    first=read(OUT/'首forward前绑定.json');details=read(OUT/'输入与源码绑定.json')
    checked=0
    for key in ['files','sources','dependencies','models','raw_used','previous_bound_files','protected_current_files']:
        for e in first[key]:assert sha(ROOT/e['path'])==e['sha256'],e['path'];checked+=1
    previous_seal=details['previous_seal'];assert sha(ROOT/previous_seal['path'])==previous_seal['sha256']
    seal=read(ROOT/previous_seal['path'])
    for e in seal['files']+seal['sources']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    impl=details['installed_clip_implementation'];assert sha(impl['path'])==impl['sha256']
    task=read(OUT/'冻结执行任务.json');pairs=task['pairs'];split=read(OUT/'来源分组副本.json');protocol=read(OUT/'冻结协议.json')
    assert task==read(PREV/'冻结执行任务.json') and split==read(PREV/'划分与任务.json')
    dev={x for group in split['dev_groups'] for x in group};reserve={x for group in split['reserved_groups'] for x in group}
    assert len(pairs)==16 and len(split['dev_groups'])==10 and len(split['reserved_groups'])==20 and not dev&reserve
    assert all(p['source_id_evaluator_only'] in dev for p in pairs)
    assert len({p['source_id_evaluator_only'] for p in pairs})==16 and len({p['goal_index_evaluator_only'] for p in pairs})==16
    assert max(collections.Counter(p['group_id'] for p in pairs).values())<=4
    for p in pairs:
        gr,gc=divmod(p['goal_index_evaluator_only'],5);assert len(p['aerial_window_paths'])==25
        for j,path in enumerate(p['aerial_window_paths']):
            r,c=divmod(j,5);indices=tuple(map(int,Path(path).stem.split('_')[1:]))
            assert task['first_numeric_index']=='row' and indices==(5-gr+r,5-gc+c)
        assert next(q for q in pairs if q['pair_id']==p['wrong_target_pair_id'])['group_id']!=p['group_id']
    resource=read(OUT/'运行资源.json');replay=read(OUT/'全局复现核验.json');deltas={}
    for e in resource['feature_files']:assert sha(ROOT/e['path'])==e['sha256'] and (ROOT/e['path']).stat().st_size==e['bytes']
    for kind,n,ref,count in [('ground','地面全局复现.npy','地面特征.npy',16),('aerial','航拍全局复现.npy','航拍特征.npy',400)]:
        current=np.load(OUT/n,allow_pickle=False);old=np.load(PREV/ref,allow_pickle=False)
        assert current.dtype==np.float32 and current.shape==old.shape==(count,512) and np.isfinite(current).all()
        delta=float(np.max(np.abs(current.astype(float)-old.astype(float))));deltas[kind]=delta
        assert delta<=protocol['global_replay_tolerance'] and delta==replay['by_encoder'][kind]['maximum_global_delta']
        assert replay['by_encoder'][kind]['maximum_cls_delta']<=protocol['cls_reconstruction_tolerance']
    assert replay['passed'] and replay['before_local_scoring']
    ground=np.load(OUT/'地面局部特征.npy',allow_pickle=False);aerial=np.load(OUT/'航拍局部特征.npy',allow_pickle=False)
    assert ground.dtype==aerial.dtype==np.float16 and ground.shape==(16,49,512) and aerial.shape==(400,49,512)
    assert np.isfinite(ground).all() and np.isfinite(aerial).all()
    g=[unit(v) for v in ground];mean=unit(sum(g)/16);index={p['pair_id']:i for i,p in enumerate(pairs)}
    rows=read(OUT/'逐对排序结果.json')['rows'];assert len(rows)==48
    lookup={(r['pair_id'],r['condition']):r for r in rows};assert len(lookup)==48
    conditions=['Full','Mean','Wrong'];counts=dict.fromkeys(conditions,0);top5=dict(counts);ranks={k:[] for k in counts};groups={};results={};max_delta=0
    for i,p in enumerate(pairs):
        group=groups.setdefault(p['group_id'],{'planned':0,**dict.fromkeys(conditions,0)});group['planned']+=1
        for name,target in [('Full',g[i]),('Mean',mean),('Wrong',g[index[p['wrong_target_pair_id']]])]:
            scores=[score(aerial[i*25+j],target) for j in range(25)];row=lookup[p['pair_id'],name]
            assert np.isfinite(row['scores']).all() and len(row['scores'])==25
            max_delta=max(max_delta,max(abs(x-y) for x,y in zip(scores,row['scores'])))
            # Rank persisted float64 decisions after independently checking every value.
            # This preserves the frozen hash rule for exact/numerical ties.
            ranked=sorted(range(25),key=lambda j:(-row['scores'][j],hashlib.sha256(f"CV-002/tie/v1:{p['pair_id']}:{j}".encode()).hexdigest()))
            pos=ranked.index(p['goal_index_evaluator_only'])+1
            assert row['ranking']==ranked and row['true_rank']==pos and row['top1']==(pos==1) and row['top5']==(pos<=5)
            truth=p['goal_index_evaluator_only'];margin=scores[truth]-max(scores[j] for j in range(25) if j!=truth)
            assert abs(row['true_score']-scores[truth])<5e-12 and abs(row['true_minus_best_other']-margin)<1e-11
            counts[name]+=pos==1;top5[name]+=pos<=5;group[name]+=pos==1;ranks[name].append(pos);results[p['pair_id'],name]=pos==1
    assert max_delta<5e-12,max_delta
    report=read(OUT/'局部匹配对照汇总.json')
    assert counts==report['top1_counts'] and top5==report['top5_counts'] and groups==report['per_group']
    assert {k:statistics.median(v) for k,v in ranks.items()}==report['median_true_rank']
    assert {k:v/16 for k,v in counts.items()}==report['top1_rates']
    positive=sum(v['Full']>v['Mean'] and v['Full']>v['Wrong'] for v in groups.values());assert positive==report['groups_positive_against_both_controls']
    gates={'all_16_complete':True,'full_top1_at_least4':counts['Full']>=4,'full_minus_mean_at_least2':counts['Full']-counts['Mean']>=2,
      'full_minus_wrong_at_least2':counts['Full']-counts['Wrong']>=2,'positive_groups_at_least3':positive>=3}
    assert gates==report['gates'] and all(gates.values())==report['candidate_passed']==state['candidate_passed']
    for name in ['Mean','Wrong']:
        rec=sum(results[p['pair_id'],'Full'] and not results[p['pair_id'],name] for p in pairs)
        harm=sum(results[p['pair_id'],name] and not results[p['pair_id'],'Full'] for p in pairs)
        assert report['contrasts'][name]=={'recovered':rec,'harmed':harm,'net':rec-harm}
    old={(r['pair_id'],r['condition']):r for r in read(PREV/'逐对排序结果.json')['rows']}
    for metric in ['top1','top5']:
        rec=sum(lookup[p['pair_id'],'Full'][metric] and not old[p['pair_id'],'Full'][metric] for p in pairs)
        harm=sum(old[p['pair_id'],'Full'][metric] and not lookup[p['pair_id'],'Full'][metric] for p in pairs)
        assert report['local_vs_global_full'][metric]=={'recovered':rec,'harmed':harm,'net':rec-harm}
    assert state['attempted_input_images']==resource['attempted_input_images']==416
    assert state['successful_forward_calls']==resource['successful_forward_calls']==52
    assert state['ground_done']==16 and state['aerial_done']==400
    assert resource['reserved_source_images_encoded']==resource['new_training_parameters']==resource['training_steps']==resource['navigation_episodes']==resource['model_api_calls']==resource['downloads']==0
    assert sum(e['images'] for e in resource['encoders'])==416
    assert all(e['parameters_used']==87849216 and not e['training'] and not e['gradients_enabled'] and e['peak_cuda_allocated_bytes']<=2*1024**3 for e in resource['encoders'])
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert total<=protocol['output_limit_bytes']
    result={'task':'CV-003','passed':True,'binding_checks':checked,'previous_seal_files':len(seal['files']),'previous_seal_sources':len(seal['sources']),
      'planned_pairs':16,'development_groups':10,'reserved_groups':20,'scalars_recomputed':1200,'maximum_score_delta':max_delta,'global_replay_max_delta':deltas,
      'top1_counts':counts,'top5_counts':top5,'candidate_passed':report['candidate_passed'],'model_forward_calls_in_audit':0,
      'independent_implementation':True,'independent_person_review':False,'ranking_method':'saved float64 decisions checked against alternate einsum equations; unchanged hash ties',
      'output_bytes':total,'navigation_SR':None,'original_defaults_and_current_report_unchanged':True}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
