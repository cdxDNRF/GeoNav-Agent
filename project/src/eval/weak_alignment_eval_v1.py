"""CV-008 offline rankings. Candidate libraries are evaluator-private, not online input."""
import hashlib
import numpy as np
import torch
from agents.weak_alignment_head_v1 import WeakAlignmentHead
from agents.crossview_local_match_v1 import rank_cells
from data.mmgag_views_v1 import normalize_features


def eval_tasks(bank, manifest, mm_pairs, dev_groups):
    by_offset = {(c['scene'], tuple(c['offset'])): i for i, c in enumerate(bank['cells'])}
    query_index = {q['scene']: i for i, q in enumerate(bank['queries'])}
    result = []
    for row in manifest:
        if row['role'] != 'development': continue
        r, c = divmod(row['variant'], 5)
        candidates = [by_offset[row['scene'], (y-r, x-c)] for y in range(5) for x in range(5)]
        wrong = dev_groups[(dev_groups.index(row['scene']) + 1) % len(dev_groups)]
        result.append({'dataset': 'weak_dev', 'pair_id': 'weak_'+row['original_id'], 'group': row['scene'],
                       'query': query_index[row['scene']], 'wrong_query': query_index[wrong],
                       'candidates': candidates, 'truth': row['positive_index']})
    mm_index = {p['pair_id']: i for i, p in enumerate(mm_pairs)}
    for i, p in enumerate(mm_pairs):
        assert p['group_id'] != mm_pairs[mm_index[p['wrong_target_pair_id']]]['group_id']
        result.append({'dataset': 'MMGAG', 'pair_id': p['pair_id'], 'group': p['group_id'], 'query': i,
                       'wrong_query': mm_index[p['wrong_target_pair_id']], 'candidates': list(range(i*25, (i+1)*25)),
                       'truth': p['goal_index_evaluator_only']})
    assert len(result) == 141 and len({t['pair_id'] for t in result}) == 141
    return result


def transformed(vector, state):
    head = WeakAlignmentHead().double()
    head.load_state_dict({k: torch.from_numpy(v).double() for k, v in state.items()}, strict=True)
    head.eval().requires_grad_(False)
    with torch.inference_mode(): return head(torch.from_numpy(vector).double()).numpy()


def rankings(tasks, features, fit_indices, states):
    rows = []
    means = {'weak_dev': normalize_features(features['weak_dev'][0][fit_indices].mean(axis=0)),
             'MMGAG': normalize_features(features['MMGAG'][0].mean(axis=0))}
    for arm, state in states.items():
        for t in tasks:
            g, a = features[t['dataset']]
            for condition, vector in [('Full', g[t['query']]), ('Mean', means[t['dataset']]), ('Wrong', g[t['wrong_query']])]:
                target = transformed(vector, state)
                scores = a[t['candidates']] @ target
                order = rank_cells(scores, t['pair_id']); position = order.index(t['truth']) + 1
                rows.append({'dataset': t['dataset'], 'pair_id': t['pair_id'], 'group': t['group'], 'arm': arm,
                             'condition': condition, 'truth_evaluator_only': t['truth'], 'scores': scores.tolist(),
                             'ranking': order, 'true_rank': position, 'top1': position == 1, 'top5': position <= 5})
    assert len(rows) == 846
    return rows


def summarize(rows):
    counts = {}; sources = {}; contrasts = {}
    for dataset in ['weak_dev', 'MMGAG']:
        subset = [r for r in rows if r['dataset'] == dataset]
        counts[dataset] = {}; sources[dataset] = {}
        for arm in ['Identity', 'Rank8Final']:
            counts[dataset][arm] = {}
            for cond in ['Full', 'Mean', 'Wrong']:
                selected = [r for r in subset if r['arm'] == arm and r['condition'] == cond]
                counts[dataset][arm][cond] = {'planned': len(selected), 'top1': sum(r['top1'] for r in selected),
                                               'top5': sum(r['top5'] for r in selected),
                                               'median_true_rank': float(np.median([r['true_rank'] for r in selected]))}
                for group in sorted({r['group'] for r in selected}):
                    group_rows = [r for r in selected if r['group'] == group]
                    sources[dataset].setdefault(group, {}).setdefault(arm, {})[cond] = {
                        'planned': len(group_rows), 'top1': sum(r['top1'] for r in group_rows), 'top5': sum(r['top5'] for r in group_rows)}
        lookup = {(r['pair_id'], r['arm'], r['condition']): r for r in subset}
        pairs = sorted({r['pair_id'] for r in subset}); contrasts[dataset] = {}
        for arm in ['Identity', 'Rank8Final']:
            for control in ['Mean', 'Wrong']:
                contrasts[dataset][arm+'_Full_vs_'+control] = contrast(lookup, pairs, (arm, 'Full'), (arm, control))
        for cond in ['Full', 'Mean', 'Wrong']:
            contrasts[dataset]['Final_vs_Identity_'+cond] = contrast(lookup, pairs, ('Rank8Final', cond), ('Identity', cond))
    m = counts['MMGAG']['Rank8Final']; w = sources['weak_dev']
    weak_gains = {group: (r['Rank8Final']['Full']['top1']-r['Identity']['Full']['top1']) / r['Identity']['Full']['planned'] for group, r in w.items()}
    positive_mm = sum(r['Rank8Final']['Full']['top1'] > max(r['Rank8Final'][c]['top1'] for c in ['Mean', 'Wrong']) for r in sources['MMGAG'].values())
    gates = {'MMGAG_all16_complete': m['Full']['planned'] == 16, 'MMGAG_Full_top1_minimum4': m['Full']['top1'] >= 4,
             'MMGAG_Full_minus_Mean_minimum2': m['Full']['top1']-m['Mean']['top1'] >= 2,
             'MMGAG_Full_minus_Wrong_minimum2': m['Full']['top1']-m['Wrong']['top1'] >= 2,
             'MMGAG_groups_positive_against_both_minimum3': positive_mm >= 3,
             'weak_macro_top1_gain_minimum004': float(np.mean(list(weak_gains.values()))) >= .04-1e-12,
             'weak_groups_improving_minimum3': sum(x > 0 for x in weak_gains.values()) >= 3}
    return {'task': 'CV-008', 'counts': counts, 'per_source': sources, 'contrasts': contrasts,
            'weak_source_top1_gains': weak_gains, 'MMGAG_groups_positive_against_both': positive_mm,
            'effect_gates': gates, 'effect_gates_passed': all(gates.values()),
            'navigation_SR': None, 'independent_geographic_samples_added': 0,
            'scope': '5 weak-development groups and10 known MMGAG development groups; geographic independence unknown'}


def contrast(lookup, pairs, treatment, control):
    result = {}
    for metric in ['top1', 'top5']:
        t = [bool(lookup[(p, *treatment)][metric]) for p in pairs]
        c = [bool(lookup[(p, *control)][metric]) for p in pairs]
        recovered = sum(x and not y for x, y in zip(t, c)); harmed = sum(y and not x for x, y in zip(t, c))
        result[metric] = {'recovered': recovered, 'harmed': harmed, 'net': recovered-harmed}
    return result
