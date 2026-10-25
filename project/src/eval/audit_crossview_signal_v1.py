"""Alternate arithmetic/counts for CV-021. No project scorer/head imports."""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'场景与精确格信号分解_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(v):
    n=math.sqrt(math.fsum(float(x)*float(x) for x in v))
    if n<=1e-12:raise ValueError('zero vector')
    return [float(x)/n for x in v]
def rank(values,source):
    return sorted(enumerate(values),key=lambda p:(-p[1],hashlib.sha256(('CV-002/tie/v1:'+source+':'+str(p[0])).encode()).hexdigest()))

def main():
    path=OUT/'另实现复核.json'
    if path.exists():raise FileExistsError('Audit output already consumed')
    bind=read(OUT/'输入与源码绑定.json');a=bind['assets']
    entries=bind['sources']+bind['dependencies']+bind['inputs']+[bind['protocol']]+a['feature_entries']+list(a['head_entries'].values())+a['ranking_entries']+a['protected']
    for e in entries:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    queue=read(BASE/'三新增来源工程与配对覆盖协议_v1/固定训练与开发队列.json')
    tasks={x['pair_id']:x for x in queue['development_tasks']}
    rows=[]
    for e in a['ranking_entries']:rows.extend(read(ROOT/e['path'])['rows'])
    spatial=read(OUT/'格级已存分数重组.json');scene=read(OUT/'场景分数与查询投影.json');decision=read(OUT/'开发信号筛选.json')
    assert len(rows)==1625 and len(spatial['cases'])==60 and len(spatial['sources'])==5
    by_source={s['source']:s for s in spatial['sources']};score_errors=[]
    for case in spatial['cases']:
        source=by_source[case['source']]
        available={i:[] for i in source['indices']}
        matched=[r for r in rows if r['arm']==case['arm'] and r['condition']==case['condition'] and tasks[r['pair_id']]['source_id_evaluator_only']==case['source']]
        assert len(matched)==25
        for r in matched:
            for i,v in zip(tasks[r['pair_id']]['aerial_feature_indices'],r['scores']):available[i].append(v)
        for j,i in enumerate(source['indices']):
            assert available[i] and max(available[i])-min(available[i])<=1e-12
            score_errors.append(abs(case['scores'][j]-available[i][0]))
        ordered=[p[0] for p in rank(case['scores'],case['source'])]
        assert case['ranking']==ordered
        truth_set={t['aerial_feature_indices'][t['goal_index_evaluator_only']] for t in tasks.values() if t['source_id_evaluator_only']==case['source']}
        assert len(truth_set)==1
        truth=source['indices'].index(next(iter(truth_set)))
        coords=[]
        for i in source['indices']:
            stem=Path(queue['aerial_encoder_inputs'][i]['path']).stem.split('_');coords.append([int(stem[1])-1,int(stem[2])-1])
        assert coords==source['coords'] and len(set(map(tuple,coords)))==81
        assert coords[truth]==[4,4] and case['truth_index_evaluator_only']==truth and case['true_rank']==ordered.index(truth)+1
        d=sum(abs(x-y) for x,y in zip(coords[ordered[0]],coords[truth]));assert d==case['best_cell_distance']
        assert case['top1']==(case['true_rank']==1) and case['top5']==(case['true_rank']<=5)
        assert case['best_within_one_cell_diagnostic_only']==(d<=1) and case['best_cell']==coords[ordered[0]]
        assert abs(case['truth_minus_best_negative']-(case['scores'][truth]-max(v for i,v in enumerate(case['scores']) if i!=truth)))<1e-12
        for name,layer in case['distance_layers'].items():
            distances=[sum(abs(x-y) for x,y in zip(c,coords[truth])) for c in coords]
            values=[v for v,dist in zip(case['scores'],distances) if (dist==0 if name=='exact' else dist==1 if name=='adjacent' else dist>=2)]
            assert layer['count']==len(values)
            assert abs(layer['mean_score']-math.fsum(values)/len(values))<1e-12 and layer['max_score']==max(values)
    assert len(score_errors)==4860 and max(score_errors)==0
    g=np.load(ROOT/a['feature_entries'][0]['path'],allow_pickle=False)
    aerial=np.load(ROOT/a['feature_entries'][1]['path'],allow_pickle=False)
    assert g.shape==(14,512) and aerial.shape==(1134,512)
    gn=[norm(x) for x in g];mean=norm([math.fsum(gn[i][j] for i in range(9))/9 for j in range(512)])
    queries=gn[9:14]+[mean];banks=[]
    for s in spatial['sources']:
        unit_rows=[norm(aerial[i]) for i in s['indices']]
        banks.append(norm([math.fsum(row[j] for row in unit_rows)/81 for j in range(512)]))
    projection_errors=[];scene_errors=[];gates_by_arm={}
    for arm,e in a['head_entries'].items():
        with np.load(ROOT/e['path'],allow_pickle=False) as state:
            u=state['U'];v=state['V'];assert set(state.files)=={'U','V'} and u.shape==(512,8) and v.shape==(8,512)
        projected=[]
        for q in queries:
            hidden=[math.fsum(float(v[k,j])*q[j] for j in range(512)) for k in range(8)]
            projected.append(norm([q[j]+math.fsum(float(u[j,k])*hidden[k] for k in range(8)) for j in range(512)]))
        saved=scene['arms'][arm]
        projection_errors.extend(abs(x-y) for row,other in zip(projected,saved['projected_targets']) for x,y in zip(row,other))
        matrix=[[math.fsum(x*y for x,y in zip(q,bank)) for bank in banks] for q in projected]
        scene_errors.extend(abs(x-y) for row,other in zip(matrix,saved['score_matrix_6x5']) for x,y in zip(row,other))
        counts={c:0 for c in ['Full','Mean','Wrong']};better={'Mean':[],'Wrong':[]}
        for i,source in enumerate(scene['source_order']):
            assert saved['per_source'][i]['source']==source
            for c,qi in [('Full',i),('Mean',5),('Wrong',(i+1)%5)]:
                record=saved['per_source'][i]['conditions'][c];order=[p[0] for p in rank(saved['score_matrix_6x5'][qi],source)]
                assert record['scores']==saved['score_matrix_6x5'][qi] and record['ranking']==order
                assert record['true_rank']==order.index(i)+1 and record['top1']==(order[0]==i)
                assert record['best_source']==scene['source_order'][order[0]]
                counts[c]+=int(record['top1'])
            cases={x['condition']:x for x in spatial['cases'] if x['source']==source and x['arm']==arm}
            for c in better:
                if cases['Full']['true_rank']<cases[c]['true_rank']:better[c].append(source)
        joint=sorted(set(better['Mean']).intersection(better['Wrong']))
        gates={'scene_Full_at_least3':counts['Full']>=3,'scene_Full_minus_Mean_at_least2':counts['Full']>=counts['Mean']+2,
            'scene_Full_minus_Wrong_at_least2':counts['Full']>=counts['Wrong']+2,'cell_rank_above_both_controls_at_least3_sources':len(joint)>=3}
        exp=decision['arms'][arm]
        assert exp['gates']==gates and exp['signal_screen_passed']==all(gates.values())
        assert exp['scene_top1_counts']==counts and exp['cell_rank_better_than_both']==joint and exp['cell_rank_better_than']==better
        for c in ['Full','Mean','Wrong']:
            cc=[x for x in spatial['cases'] if x['arm']==arm and x['condition']==c]
            assert exp['spatial_81_counts'][c]=={'top1':sum(x['top1'] for x in cc),'top5':sum(x['top5'] for x in cc),'best_within_one':sum(x['best_cell_distance']<=1 for x in cc)}
        gates_by_arm[arm]=gates
    assert max(projection_errors)<1e-12 and max(scene_errors)<1e-12 and len(scene_errors)==120
    eligible=[arm for arm in read(OUT/'冻结诊断协议.json')['arms'] if all(gates_by_arm[arm].values())]
    assert decision['eligible']==eligible and decision['direction_signal_present']==bool(eligible)
    assert decision['selected_by_frozen_priority']==(eligible[0] if eligible else None)
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    for e in entries:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    audit={'task':'CV-021','passed':True,'saved_slots_checked':4860,'scene_scores_checked':120,
        'max_saved_score_error':max(score_errors),'max_projection_arithmetic_error':max(projection_errors),'max_scene_arithmetic_error':max(scene_errors),
        'all_source_counts_controls_and_gates_match':True,'backbone_forward':0,'head_model_forward':0,'training':0,'raw_images':0,
        'same_agent_alternate_implementation_not_independent_person':True,'bound_inputs_unchanged':True}
    path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
