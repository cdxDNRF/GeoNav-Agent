"""CV-019 offline375 new rankings,1625 referenced records, six paired effects."""
import itertools
import numpy as np
from agents.capturepoint_target_head_v1 import TargetHead
from agents.crossview_local_match_v1 import rank_cells
from data.mmgag_views_v1 import normalize_features
from data.capturepoint_coverage_protocol_v1 import CANDIDATE


def rank_candidate(queue,g,a,weights):
    if g.shape!=(17,512) or a.shape!=(1377,512):raise ValueError('Frozen17/1377 feature layout required')
    g=normalize_features(g);a=normalize_features(a);mean=normalize_features(g[:9].mean(0));head=TargetHead(weights);rows=[]
    for task in queue['development_tasks']:
        for condition in ('Full','Mean','Wrong'):
            target=g[task['ground_feature_index']] if condition=='Full' else mean if condition=='Mean' else g[task['wrong_ground_feature_index']]
            scores=a[task['aerial_feature_indices']]@head(target)
            order=rank_cells(scores,task['pair_id']);truth=task['goal_index_evaluator_only'];rank=order.index(truth)+1
            rows.append({'pair_id':task['pair_id'],'group':task['group_evaluator_only'],'arm':CANDIDATE,'condition':condition,
                'truth_evaluator_only':truth,'scores':scores.tolist(),'ranking':order,'true_rank':rank,'top1':rank==1,'top5':rank<=5})
    if len(rows)!=375:raise ValueError('125x3 new records required')
    return rows


def summarize(old_rows,new_rows):
    if len(old_rows)!=1250 or len(new_rows)!=375:raise ValueError('Fixed old1250/new375 planned records')
    rows=old_rows+new_rows;lookup={(r['pair_id'],r['arm'],r['condition']):r for r in rows}
    if len(lookup)!=1625:raise ValueError('Duplicate planned record')
    pairs=sorted({r['pair_id'] for r in rows});groups=sorted({r['group'] for r in rows})
    if len(pairs)!=125 or len(groups)!=5:raise ValueError('Only five source units and125 windows')
    arms=['Identity','SwissRank8Frozen','MMRank8Final',CANDIDATE]
    combos=[(a,c) for a in arms for c in ('Full','Mean','Wrong')]+[('SharedDisabled','Disabled')]
    if set(lookup)!={(p,a,c) for p in pairs for a,c in combos}:raise ValueError('Planned combination mismatch')
    counts={};source={};gate={}
    for arm,c in combos:
        subset=[lookup[p,arm,c] for p in pairs]
        counts.setdefault(arm,{})[c]={'planned':125,'top1':sum(r['top1'] for r in subset),'top5':sum(r['top5'] for r in subset),'median_rank':float(np.median([r['true_rank'] for r in subset]))}
        for group in groups:
            selected=[r for r in subset if r['group']==group]
            if len(selected)!=25:raise ValueError('Every source keeps25 windows')
            source.setdefault(group,{}).setdefault(arm,{})[c]={'planned':25,'top1':sum(r['top1'] for r in selected),'top5':sum(r['top5'] for r in selected)}
    for group,values in source.items():
        gate[group]={'planned':25,'Disabled':values['SharedDisabled']['Disabled']['top1']}
        for arm in arms:gate[group][arm]={c:values[arm][c]['top1'] for c in ('Full','Mean','Wrong')}
    specs=[('Full_vs_'+c,(CANDIDATE,'Full'),(CANDIDATE,c)) for c in ('Mean','Wrong')]
    specs += [('Full_vs_Disabled',(CANDIDATE,'Full'),('SharedDisabled','Disabled'))]
    specs += [('Full_vs_'+a,(CANDIDATE,'Full'),(a,'Full')) for a in arms[:-1]]
    contrasts={}
    for name,target,control in specs:
        effect={}
        for metric in ('top1','top5'):
            recovered=harmed=0;deltas=[]
            for group in groups:
                ids=[p for p in pairs if lookup[p,*target]['group']==group]
                rec=sum(lookup[p,*target][metric] and not lookup[p,*control][metric] for p in ids)
                harm=sum(lookup[p,*control][metric] and not lookup[p,*target][metric] for p in ids)
                recovered+=rec;harmed+=harm;deltas.append((rec-harm)/25)
            values=[sum(x)/5 for x in itertools.product(deltas,repeat=5)]
            effect[metric]={'recovered':recovered,'harmed':harmed,'net':recovered-harmed,'source_units':5,'resamples':3125,
                'per_source_rate_differences':dict(zip(groups,deltas)),'paired_source_percentile95':np.quantile(values,[.025,.975]).tolist()}
        contrasts[name]=effect
    return {'planned_records':1625,'old_reused_records':1250,'new_records':375,'planned_windows':125,'source_units':5,
        'counts':counts,'per_source':source,'gate_counts':gate,'contrasts':contrasts,'navigation_SR':None,
        'scope':'known-development camera-proxy association;125 windows are5 target photos, not125 independent regions'}
