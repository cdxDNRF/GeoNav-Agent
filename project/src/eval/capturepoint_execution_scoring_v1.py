"""Private offline evaluator for the CV-012 queue. Never an online navigation policy."""
import itertools
import numpy as np
from agents.capturepoint_target_head_v1 import TargetHead
from agents.crossview_local_match_v1 import rank_cells
from data.mmgag_views_v1 import normalize_features
from data.capturepoint_alignment_protocol_v1 import ARMS,CONDITIONS,gate


def scores_for(task,condition,g,a,head,mean):
    if condition=='Disabled':return np.zeros(25,np.float64)
    if condition=='Full':target=g[task['ground_feature_index']]
    elif condition=='Wrong':target=g[task['wrong_ground_feature_index']]
    elif condition=='Mean':target=mean
    else:raise ValueError('Unknown frozen condition')
    # Hidden goal/group/source never enter the public head call.
    return a[task['aerial_feature_indices']]@head(target)


def rank_all(queue,ground,aerial,states):
    if ground.shape!=(14,512) or aerial.shape!=(1134,512) or set(states)!=set(ARMS):raise ValueError('Frozen queue/features/three arms required')
    g=normalize_features(ground);a=normalize_features(aerial);mean=normalize_features(g[:9].mean(0))
    heads={k:TargetHead(v) for k,v in states.items()};rows=[]
    for task in queue['development_tasks']:
        conditions=[(arm,c) for arm in ARMS for c in CONDITIONS]+[('SharedDisabled','Disabled')]
        for arm,condition in conditions:
            scores=scores_for(task,condition,g,a,heads.get(arm),mean)
            order=rank_cells(scores,task['pair_id']);truth=task['goal_index_evaluator_only'];pos=order.index(truth)+1
            rows.append({'pair_id':task['pair_id'],'group':task['group_evaluator_only'],'arm':arm,'condition':condition,
                'truth_evaluator_only':truth,'scores':scores.tolist(),'ranking':order,'true_rank':pos,'top1':pos==1,'top5':pos<=5})
    assert len(rows)==1250
    return rows


def summarize(rows):
    if len(rows)!=1250:raise ValueError('Planned1250 rankings required')
    keys={(r['pair_id'],r['arm'],r['condition']) for r in rows}
    if len(keys)!=1250:raise ValueError('Duplicate planned ranking')
    lookup={(r['pair_id'],r['arm'],r['condition']):r for r in rows};groups=sorted({r['group'] for r in rows})
    if len(groups)!=5:raise ValueError('Five source units required')
    counts={};source={};gate_counts={}
    combinations=[(arm,c) for arm in ARMS for c in CONDITIONS]+[('SharedDisabled','Disabled')]
    for arm,c in combinations:
        subset=[r for r in rows if r['arm']==arm and r['condition']==c]
        if len(subset)!=125:raise ValueError('Fixed125 windows per arm/condition')
        counts.setdefault(arm,{})[c]={'planned':125,'top1':sum(r['top1'] for r in subset),'top5':sum(r['top5'] for r in subset),
            'median_rank':float(np.median([r['true_rank'] for r in subset]))}
        for group in groups:
            selected=[r for r in subset if r['group']==group]
            if len(selected)!=25:raise ValueError('25 windows per source required')
            source.setdefault(group,{}).setdefault(arm,{})[c]={'planned':25,'top1':sum(r['top1'] for r in selected),'top5':sum(r['top5'] for r in selected)}
    for group,s in source.items():
        gate_counts[group]={'planned':25,'Disabled':s['SharedDisabled']['Disabled']['top1']}
        for arm in ARMS:gate_counts[group][arm]={c:s[arm][c]['top1'] for c in CONDITIONS}
    comparisons=[(arm+'_Full_vs_'+c,(arm,'Full'),(arm,c)) for arm in ARMS for c in ['Mean','Wrong']]
    comparisons += [(arm+'_Full_vs_Disabled',(arm,'Full'),('SharedDisabled','Disabled')) for arm in ARMS]
    comparisons += [('MM_vs_'+arm+'_'+c,('MMRank8Final',c),(arm,c)) for arm in ['Identity','SwissRank8Frozen'] for c in CONDITIONS]
    contrasts={}
    for name,t,c in comparisons:
        result={}
        for metric in ['top1','top5']:
            pairs=sorted({r['pair_id'] for r in rows});rec=harm=0;deltas=[]
            for group in groups:
                selected=[p for p in pairs if lookup[p,*t]['group']==group]
                recovered=sum(lookup[p,*t][metric] and not lookup[p,*c][metric] for p in selected)
                harmed=sum(lookup[p,*c][metric] and not lookup[p,*t][metric] for p in selected)
                rec+=recovered;harm+=harmed;deltas.append((recovered-harmed)/25)
            samples=[sum(x)/5 for x in itertools.product(deltas,repeat=5)]
            result[metric]={'recovered':rec,'harmed':harm,'net':rec-harm,'per_source_rate_differences':dict(zip(groups,deltas)),
                'paired_source_percentile95':[float(v) for v in np.quantile(samples,[.025,.975])],'source_units':5,'resamples':3125}
        contrasts[name]=result
    return {'planned_rankings':1250,'planned_windows':125,'development_source_units':5,'counts':counts,'per_source':source,
        'gate_counts':gate_counts,'contrasts':contrasts,'navigation_SR':None,'note':'development association only;25windows/source are repeated target queries'}
