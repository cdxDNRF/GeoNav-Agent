"""Independent CV-002 audit: saved features only, no model imports or forward."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def unit(v):
    v=[float(x) for x in v];d=math.sqrt(math.fsum(x*x for x in v));assert d>1e-12
    return [x/d for x in v]
def main(save):
    state=read('执行状态.json');assert state['state']=='completed'
    first=read('首forward前绑定.json')
    for e in first['files']+first['sources']+first['models']+first['raw_used']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    split=read('划分与任务.json');pairs=read('冻结执行任务.json')['pairs'];protocol=read('冻结协议.json');report=read('感知对照汇总.json');rows=read('逐对排序结果.json')['rows'];resource=read('运行资源.json')
    assert len(split['dev_groups'])==10 and len(split['reserved_groups'])==20 and len(pairs)==16
    dev={s for group in split['dev_groups'] for s in group};reserved={s for group in split['reserved_groups'] for s in group}
    assert not dev&reserved
    assert all(p['source_id_evaluator_only'] in dev for p in pairs)
    assert len({p['source_id_evaluator_only'] for p in pairs})==16 and len({p['goal_index_evaluator_only'] for p in pairs})==16
    assert max(collections.Counter(p['group_id'] for p in pairs).values())<=4
    for p in pairs:
        goal=p['goal_index_evaluator_only'];gr,gc=divmod(goal,5)
        assert len(p['aerial_window_paths'])==25
        for j,path in enumerate(p['aerial_window_paths']):
            r,c=divmod(j,5);bits=Path(path).stem.split('_');indices=(int(bits[1]),int(bits[2]))
            if read('冻结执行任务.json')['first_numeric_index']=='column':indices=indices[::-1]
            assert indices==(5-gr+r,5-gc+c),(p['pair_id'],j,indices)
        wrong=next(x for x in pairs if x['pair_id']==p['wrong_target_pair_id']);assert wrong['group_id']!=p['group_id']
    ground=np.load(OUT/'地面特征.npy',allow_pickle=False);aerial=np.load(OUT/'航拍特征.npy',allow_pickle=False)
    assert ground.shape==(16,512) and aerial.shape==(400,512)
    assert np.isfinite(ground).all() and np.isfinite(aerial).all()
    for e in resource['feature_files']:assert sha(ROOT/e['path'])==e['sha256']
    g=[unit(v) for v in ground];mean=unit([math.fsum(g[i][j] for i in range(16))/16 for j in range(512)])
    conditions=['Full','Mean','Wrong'];counts={k:0 for k in conditions};top5=dict(counts);ranks={k:[] for k in conditions};groups={};results={};max_delta=0
    for i,p in enumerate(pairs):
        w=next(j for j,x in enumerate(pairs) if x['pair_id']==p['wrong_target_pair_id'])
        query={'Full':g[i],'Mean':mean,'Wrong':g[w]};grp=groups.setdefault(p['group_id'],{'planned':0,**{k:0 for k in conditions}});grp['planned']+=1
        for name in conditions:
            scores=[math.fsum(a*b for a,b in zip(unit(aerial[i*25+j]),query[name])) for j in range(25)]
            ranked=sorted(range(25),key=lambda j:(-scores[j],hashlib.sha256(f"CV-002/tie/v1:{p['pair_id']}:{j}".encode()).hexdigest()))
            pos=ranked.index(p['goal_index_evaluator_only'])+1;correct=pos==1
            row=next(x for x in rows if x['pair_id']==p['pair_id'] and x['condition']==name)
            max_delta=max(max_delta,max(abs(x-y) for x,y in zip(scores,row['scores'])))
            assert ranked==row['ranking'] and pos==row['true_rank'] and correct==row['top1'] and (pos<=5)==row['top5']
            counts[name]+=correct;top5[name]+=pos<=5;grp[name]+=correct;ranks[name].append(pos);results[(p['pair_id'],name)]=correct
    assert max_delta<=5e-12,max_delta
    assert counts==report['top1_counts'] and top5==report['top5_counts'] and groups==report['per_group']
    assert {k:statistics.median(v) for k,v in ranks.items()}==report['median_true_rank']
    positive=sum(x['Full']>x['Mean'] and x['Full']>x['Wrong'] for x in groups.values())
    gates=[counts['Full']>=4,counts['Full']-counts['Mean']>=2,counts['Full']-counts['Wrong']>=2,positive>=3]
    assert all(gates)==report['candidate_passed']
    for name in ['Mean','Wrong']:
        rec=sum(results[(p['pair_id'],'Full')] and not results[(p['pair_id'],name)] for p in pairs)
        harm=sum(results[(p['pair_id'],name)] and not results[(p['pair_id'],'Full')] for p in pairs)
        assert report['contrasts'][name]=={'recovered':rec,'harmed':harm,'net':rec-harm}
    assert resource['attempted_input_images']==416 and resource['successful_forward_calls']==52
    assert sum(e['images'] for e in resource['encoders'])==416 and resource['reserved_source_images_encoded']==0
    assert all(e['peak_cuda_allocated_bytes']<=2*1024**3 for e in resource['encoders'])
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert total<=protocol['output_limit_bytes']
    result={'task':'CV-002','passed':True,'planned_pairs':16,'conditions':3,'scalars_recomputed':1200,'maximum_cosine_delta':max_delta,
      'top1_counts':counts,'top5_counts':top5,'development_groups':len(groups),'reserved_groups':20,'reserved_encoded':0,
      'candidate_passed':report['candidate_passed'],'navigation_SR':None,'model_forward_calls_in_audit':0,'independent_implementation':True,'independent_person_review':False,'output_bytes':total}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');args=p.parse_args();main(args.save)
