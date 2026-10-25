"""Frozen single-encoder target interventions. Evaluator-only banks, never policy input."""
from __future__ import annotations

import itertools
import math
import numpy as np

CONDITIONS = ('Full', 'Disabled', 'Mean', 'Wrong')


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2) or not np.isfinite(x).all():
        raise ValueError('finite vector/matrix required')
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('zero target/feature norm')
    return x / norms


def rank_case(scores, truth):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 <= truth < len(scores):
        raise ValueError('invalid scores/truth')
    order = np.lexsort((np.arange(len(scores)), -scores))
    r = int(np.flatnonzero(order == truth)[0]) + 1
    return {'scores': scores.tolist(), 'order': order.tolist(), 'truth_index_evaluator_only': int(truth),
            'rank': r, 'rr': 1.0/r, 'top1': r == 1, 'top5': r <= 5,
            'rank_best_tie': int(np.sum(scores > scores[truth]))+1,
            'rank_worst_tie': int(np.sum(scores >= scores[truth])),
            'selected_index': int(order[0]), 'selected_tie_size': int(np.sum(scores == scores[order[0]]))}


def aggregate(cases):
    n = len(cases)
    if not n:
        raise ValueError('empty planned cases')
    return {'planned': n, 'top1_hits': sum(x['top1'] for x in cases),
            'top5_hits': sum(x['top5'] for x in cases),
            'top1_rate': sum(x['top1'] for x in cases)/n,
            'top5_rate': sum(x['top5'] for x in cases)/n,
            'MRR': sum(x['rr'] for x in cases)/n,
            'mean_rank': sum(x['rank'] for x in cases)/n}


def paired_delta(cases, condition, scope):
    full = sorted([x for x in cases if x['condition'] == 'Full'], key=lambda x:x['source'])
    control = sorted([x for x in cases if x['condition'] == condition], key=lambda x:x['source'])
    if [x['source'] for x in full] != [x['source'] for x in control] or len(full) != 5:
        raise ValueError('five-source pairing mismatch')
    per = [{'source': a['source'], 'group': a['group'], 'rank_improvement': b['rank']-a['rank'],
            'MRR_delta': a['rr']-b['rr'], 'top1_delta': int(a['top1'])-int(b['top1']),
            'recovered_top1': a['top1'] and not b['top1'],
            'damaged_top1': b['top1'] and not a['top1']} for a,b in zip(full,control)]
    # Exact enumeration of all empirical source bootstrap draws: 5^5, no random seed selection.
    draws = np.asarray(list(itertools.product(range(5), repeat=5)))
    cis = {}
    for metric in ('rank_improvement', 'MRR_delta', 'top1_delta'):
        v = np.array([x[metric] for x in per])
        means = v[draws].mean(axis=1)
        cis[metric] = {'mean':float(v.mean()), 'CI95_percentile': np.quantile(means,[.025,.975]).tolist()}
    return {'scope':scope, 'control':condition, 'independent_units':5,
            'source_geographic_independence_not_certified':True,
            'recovered_top1':sum(x['recovered_top1'] for x in per),
            'damaged_top1':sum(x['damaged_top1'] for x in per),
            'rank_wins':sum(x['rank_improvement']>0 for x in per),
            'rank_ties':sum(x['rank_improvement']==0 for x in per),
            'rank_losses':sum(x['rank_improvement']<0 for x in per),
            'per_source':per, 'source_bootstrap':cis,
            'bootstrap_draws':3125, 'CI_is_development_description_not_confirmation':True}


def screen(grid, scene):
    gm = {c:aggregate([x for x in grid if x['condition']==c]) for c in CONDITIONS}
    sm = {c:aggregate([x for x in scene if x['condition']==c]) for c in CONDITIONS}
    by = {(x['source'],x['condition']):x for x in grid}
    sources=sorted({x['source'] for x in grid})
    dual=sum(by[s,'Full']['rank'] < min(by[s,'Mean']['rank'],by[s,'Wrong']['rank']) for s in sources)
    gates = [
        {'id':'G1_exact_top1','passed':gm['Full']['top1_hits']>=1,'actual':gm['Full']['top1_hits'],'required':'>=1/5'},
        {'id':'G2_dual_control_rank','passed':dual>=3,'actual':dual,'required':'>=3/5 sources strictly better than both Mean and Wrong'},
        {'id':'G3_MRR_each_control','passed':all(gm['Full']['MRR']>gm[c]['MRR'] for c in CONDITIONS[1:]),
         'actual':{c:gm['Full']['MRR']-gm[c]['MRR'] for c in CONDITIONS[1:]},'required':'Full MRR strictly greater than all3 controls'},
        {'id':'G4_no_net_top1_damage','passed':all(gm['Full']['top1_hits']>=gm[c]['top1_hits'] for c in CONDITIONS[1:]),
         'actual':{c:gm['Full']['top1_hits']-gm[c]['top1_hits'] for c in CONDITIONS[1:]},'required':'>=0 recovered-minus-damaged for each control'},
        {'id':'G5_scene_signal','passed':sm['Full']['top1_hits']>=3 and all(sm['Full']['top1_hits']-sm[c]['top1_hits']>=2 for c in ('Mean','Wrong')),
         'actual':{c:sm[c]['top1_hits'] for c in CONDITIONS},'required':'Full scene Top1>=3/5 and >=2 more hits than Mean and Wrong'},
    ]
    return {'gates':gates,'all_effect_gates_passed':all(x['passed'] for x in gates),
            'dual_control_rank_sources':dual,'grid':gm,'scene':sm,
            'interpretation':'development signal only; not navigation or formal acceptance'}


def evaluate(protocol, vectors, old):
    expected = {x['id'] for x in protocol['inputs']}
    if set(vectors) != expected:
        raise ValueError('planned feature denominator mismatch')
    if any(np.asarray(x).shape!=(1024,) for x in vectors.values()):
        raise ValueError('1024-dimensional frozen features required')
    source = protocol['sources']
    fit = source[:9]
    dev = source[9:]
    if len(fit)!=9 or len(dev)!=5:
        raise ValueError('9fit/5dev required')
    mean = unit(np.mean([unit(vectors[x['ground_id']]) for x in fit],axis=0))
    ground = [unit(vectors[x['ground_id']]) for x in dev]
    banks = [unit(np.stack([vectors[i] for i in x['tile_ids']])) for x in dev]
    scenes=unit(np.stack([unit(b.mean(axis=0)) for b in banks]))
    grid=[]; scene=[]
    for i,s in enumerate(dev):
        targets={'Full':ground[i],'Mean':mean,'Wrong':ground[(i+1)%5]}
        for c in CONDITIONS:
            scores=np.zeros(81) if c=='Disabled' else banks[i] @ targets[c]
            result=rank_case(scores,s['truth_index_evaluator_only'])
            predicted=result['selected_index']; truth=s['truth_index_evaluator_only']
            result.update({'source':s['source_id'],'group':s['group'],'condition':c,
                           'wrong_source_evaluator_only':dev[(i+1)%5]['source_id'] if c=='Wrong' else None,
                           'selected_cell_manhattan_error_not_SG':abs(predicted//9-truth//9)+abs(predicted%9-truth%9)})
            grid.append(result)
            sr=rank_case(np.zeros(5) if c=='Disabled' else scenes @ targets[c],i)
            sr.update({'source':s['source_id'],'group':s['group'],'condition':c})
            scene.append(sr)
    # Reuse only old stored Identity scores; use new frozen index tie rule in this reference view.
    old_cases=[]
    old_metadata={x['source']:x for x in old['sources']}
    for s in dev:
        om=old_metadata[s['source_id']]
        if om['truth_index']!=s['truth_index_evaluator_only'] or om['coords']!=s['tile_coords']:
            raise ValueError('old Identity candidate/label layout mismatch')
        for c in CONDITIONS:
            if c=='Disabled': scores=[0.0]*81
            else:
                matches=[x for x in old['cases'] if x['source']==s['source_id'] and x['arm']=='Identity' and x['condition']==c]
                if len(matches)!=1:raise ValueError('unique old score reference required')
                scores=matches[0]['scores']
            r=rank_case(scores,s['truth_index_evaluator_only'])
            r.update({'source':s['source_id'],'group':s['group'],'condition':c,
                      'reference':'CV021 stored Identity scores; Disabled fixed zero; tie rule unified in this new report'})
            old_cases.append(r)
    newscreen=screen(grid,scene)
    summary={'planned_sources':5,'planned_grid_queries':20,'planned_scene_queries':20,
             'grid_score_slots':1620,'scene_score_slots':100,'screen':newscreen,
             'paired_grid':[paired_delta(grid,c,'grid') for c in CONDITIONS[1:]],
             'paired_scene':[paired_delta(scene,c,'scene') for c in CONDITIONS[1:]],
             'old_identity':{c:aggregate([x for x in old_cases if x['condition']==c]) for c in CONDITIONS},
             'Full_vs_old_identity':paired_delta(grid+[dict(x,condition='Identity') for x in old_cases if x['condition']=='Full'],'Identity','grid'),
             'fit_mean':mean.tolist(),'new_SR':None,'navigation':0,'training':0,
             'confirmation_access':0,'sample_unit':'five previously consumed development source groups',
             'not_agent_or_closed_loop':True,
             'decision':'retain_for_one_new_local_navigation_protocol' if newscreen['all_effect_gates_passed'] else 'close_this_fixed_representation_no_training_or_confirmation'}
    return {'grid_cases':grid,'scene_cases':scene,'old_identity_reference':old_cases,'summary':summary}
