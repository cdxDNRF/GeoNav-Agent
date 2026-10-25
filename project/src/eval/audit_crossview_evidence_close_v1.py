"""Alternate counting/physical-index implementation, without importing the producer."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'DATA/processed_data/MM-GAG'
OUT = BASE / '跨视角阶段证据收束_v1'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def count(saved):
    ranks = []
    for row in saved:
        truth = row.get('truth_evaluator_only', row.get('goal_index_evaluator_only'))
        key = lambda j: (-row['scores'][j], hashlib.sha256(('CV-002/tie/v1:%s:%d' % (row['pair_id'], j)).encode()).hexdigest())
        assert sorted(row['ranking']) == list(range(25))
        assert all(key(a) <= key(b) for a, b in zip(row['ranking'], row['ranking'][1:]))
        rank = 1 + sum(key(j) < key(truth) for j in range(25) if j != truth)
        assert rank == row['true_rank'] and (rank == 1) == row['top1'] and (rank <= 5) == row['top5']
        ranks.append(rank)
    return {'planned': len(saved), 'top1': ranks.count(1), 'top5': len([r for r in ranks if r < 6]), 'median_rank': median(ranks)}


def audit():
    binding = load(OUT / '输入与源码绑定.json')
    for item in binding['inputs'] + binding['sources'] + binding['protected_current_files']:
        raw = (ROOT / item['path']).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
    result = load(OUT / '证据重计汇总.json')
    checked = 0
    for series in result['historical_series']:
        folder = ROOT / series['batch_path']
        filename = '逐题排序结果.json' if series['series'].startswith('CV-008') else '逐对排序结果.json'
        rows = load(folder / filename)['rows']
        if series['series'].startswith('CV-008'):
            dataset = 'weak_dev' if series['series'].endswith('weak') else 'MMGAG'
            rows = [r for r in rows if r['dataset'] == dataset and r['arm'] == 'Rank8Final']
        elif series['series'] == 'CV-005': rows = [r for r in rows if r['arm'] == 'FullFrame']
        for condition in ('Full', 'Mean', 'Wrong'):
            sub = [r for r in rows if r['condition'] == condition]
            assert count(sub) == series['counts'][condition]
            checked += len(sub)
        assert len({r.get('group', r.get('group_id')) for r in rows}) == series['source_groups']
    rows = load(BASE / '同任务拍摄点对齐执行_v1/逐题排序结果.json')['rows']
    tasks = load(BASE / '同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json')['development_tasks']
    expected = result['current']
    assert len(rows) == 1250 and len(tasks) == 125
    lookup = {(r['pair_id'], r['arm'], r['condition']): r for r in rows}
    assert len(lookup) == 1250
    for arm, conditions in expected['counts'].items():
        for condition, stats in conditions.items():
            assert count([r for r in rows if r['arm'] == arm and r['condition'] == condition]) == stats
    for arm, conditions in expected['target_sensitivity'].items():
        for condition, stats in conditions.items():
            comparisons = [(lookup[t['pair_id'], arm, 'Full'], lookup[t['pair_id'], arm, condition]) for t in tasks]
            assert stats['same_top1_cell'] == sum(x['ranking'][0] == y['ranking'][0] for x, y in comparisons)
            assert [stats['full_target_rank_better'], stats['full_target_rank_worse'], stats['full_target_rank_tied']] == [
                sum(x['true_rank'] < y['true_rank'] for x,y in comparisons),
                sum(x['true_rank'] > y['true_rank'] for x,y in comparisons),
                sum(x['true_rank'] == y['true_rank'] for x,y in comparisons)]
            overlap = sum(sum(j in y['ranking'][:5] for j in x['ranking'][:5]) for x,y in comparisons) / 625
            assert abs(overlap - stats['mean_top5_overlap']) < 1e-12
    for group, source in expected['per_source'].items():
        selected = [t for t in tasks if t['group_evaluator_only'] == group]
        assert len(selected) == 25 and len({t['ground_feature_index'] for t in selected}) == 1
        assert {t['source_id_evaluator_only'] for t in selected} == {source['source_id']}
        assert sorted(t['goal_index_evaluator_only'] for t in selected) == list(range(25))
        assert {t['aerial_feature_indices'][t['goal_index_evaluator_only']] for t in selected} == {selected[0]['ground_feature_index'] * 81 + 40}
        assert len({i for t in selected for i in t['aerial_feature_indices']}) == 81
        for arm, conditions in source['arms'].items():
            for condition, stats in conditions.items():
                selected_rows = [lookup[t['pair_id'], arm, condition] for t in selected]
                assert all(r['group'] == group and r['truth_evaluator_only'] == t['goal_index_evaluator_only'] for t,r in zip(selected,selected_rows))
                assert count(selected_rows) == {k: stats[k] for k in ('planned', 'top1', 'top5', 'median_rank')}
                winners, distances, margins, stronger = [], [], [], []
                first_scores, deltas = {}, []
                for task, row in zip(selected, selected_rows):
                    j = row['ranking'][0]; truth = row['truth_evaluator_only']
                    global_index = task['aerial_feature_indices'][j]
                    name = task['aerial_feature_indices'][truth]
                    tile_r, tile_c = divmod(global_index % 81, 9); goal_r, goal_c = divmod(name % 81, 9)
                    winners.append(global_index % 81); distances.append(abs(tile_r-goal_r) + abs(tile_c-goal_c))
                    negative = sorted(row['scores'][i] for i in range(25) if i != truth)
                    margins.append(row['scores'][truth] - negative[-1]); stronger.append(negative[-1] > row['scores'][truth])
                    for i, score in zip(task['aerial_feature_indices'], row['scores']):
                        if i in first_scores: deltas.append(abs(first_scores[i] - score))
                        else: first_scores[i] = score
                histogram = Counter(winners); mode = max(histogram.values())
                assert len(histogram) == stats['unique_top1_physical_tiles'] and mode == stats['modal_top1_windows']
                assert sorted(k for k,v in histogram.items() if v == mode) == stats['modal_top1_tiles']
                assert median(distances) == stats['median_top1_to_proxy_grid_L1']
                assert abs(median(margins)-stats['median_target_minus_best_negative']) < 1e-12
                assert sum(stronger) == stats['windows_with_strictly_higher_negative']
                assert max(deltas, default=0) < 1e-12 and stats['repeated_tile_score_max_delta'] < 1e-12
    assert checked == 567 and len(expected['per_source']) == 5 and len(result['effect_gates_failed']) == 8
    assert all(result[k] == 0 for k in ('new_raw_reads','new_model_forward_calls','new_predictions','new_training_steps','new_navigation_episodes','new_API_calls','confirmation_access'))
    assert result['new_candidate_admitted'] is False
    report = {'task': 'CV-015', 'passed': True, 'saved_ranking_decisions_checked': checked + len(rows),
        'current_saved_scores_checked': len(rows)*25, 'current_source_groups':5,
        'alternate_method': 'pairwise outranking and adjacent-order validation; saved-score/physical-index table, loop counters and medians',
        'new_model_forward_calls':0, 'new_predictions':0, 'new_training_steps':0, 'confirmation_access':0,
        'independent_person_review':False}
    path=OUT/'核验/另实现复核.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__': audit()
