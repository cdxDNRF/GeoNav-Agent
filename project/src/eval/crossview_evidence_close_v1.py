"""CV-015: recount saved decisions only. No encoders, features, weights or raw images."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[3]
BASE = Path('DATA/processed_data/MM-GAG')
OUT = BASE / '跨视角阶段证据收束_v1'
RUN = BASE / '同任务拍摄点对齐执行_v1'
QUEUE = BASE / '同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json'
ARMS = ('Identity', 'SwissRank8Frozen', 'MMRank8Final')
SOURCES = [
    'project/src/eval/crossview_evidence_close_v1.py',
    'project/src/eval/audit_crossview_evidence_close_v1.py',
    'project/src/tests/test_crossview_evidence_close_v1.py',
    'project/src/documents/close_crossview_evidence_v1.py',
]
SERIES = [
    ('CV-002', '跨视角感知最小验证_v1', '感知对照汇总.json', '逐对排序结果.json', None, None),
    ('CV-003', '局部语义匹配单因素验证_v1', '局部匹配对照汇总.json', '逐对排序结果.json', None, None),
    ('CV-005', '拍摄点语义与完整目标图验证_v1', '完整目标图对照汇总.json', '逐对排序结果.json', None, 'FullFrame'),
    ('CV-008-weak', '弱场景对齐头工程与训练_v3', '感知对照汇总.json', '逐题排序结果.json', 'weak_dev', 'Rank8Final'),
    ('CV-008-MM', '弱场景对齐头工程与训练_v3', '感知对照汇总.json', '逐题排序结果.json', 'MMGAG', 'Rank8Final'),
]


def text_json(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT / BASE) or path.suffix != '.json':
        raise ValueError('Only registered processed MM-GAG JSON inputs are readable')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    path = ROOT / path
    if path.suffix.lower() not in {'.json', '.jsonl', '.md', '.py', '.docx', '.pdf'}:
        raise ValueError('No raw, feature, cache or model tensor access')
    if not path.resolve().is_relative_to(ROOT) or 'raw_data' in path.parts or 'models' in path.parts:
        raise ValueError('Outside offline evidence scope')
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def save(name, value):
    path = ROOT / OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def validate_row(row):
    scores, ranking = row['scores'], row['ranking']
    if len(scores) != 25 or not all(type(x) in (int, float) and math.isfinite(x) for x in scores):
        raise ValueError('25 finite saved scores required')
    if len(ranking) != 25 or set(ranking) != set(range(25)):
        raise ValueError('Invalid saved permutation')
    truth = row.get('truth_evaluator_only', row.get('goal_index_evaluator_only'))
    if type(truth) != int or not 0 <= truth < 25:
        raise ValueError('Invalid evaluator truth')
    expected = sorted(range(25), key=lambda j: (-scores[j], hashlib.sha256(f"CV-002/tie/v1:{row['pair_id']}:{j}".encode()).hexdigest()))
    rank = ranking.index(truth) + 1
    if expected != ranking or row['true_rank'] != rank or row['top1'] != (rank == 1) or row['top5'] != (rank <= 5):
        raise ValueError('Saved decision/count inconsistency')
    return truth


def counts(rows):
    if len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate task within one condition')
    for row in rows:
        validate_row(row)
    return {'planned': len(rows), 'top1': sum(r['true_rank'] == 1 for r in rows),
            'top5': sum(r['true_rank'] <= 5 for r in rows), 'median_rank': median(r['true_rank'] for r in rows)}


def summarize_current(rows, queue):
    tasks = {t['pair_id']: t for t in queue['development_tasks']}
    if len(tasks) != 125 or len(queue['development_tasks']) != 125 or len(rows) != 1250:
        raise ValueError('Fixed125 tasks/1250 records required')
    expected = {(p, a, c) for p in tasks for a in ARMS for c in ('Full', 'Mean', 'Wrong')}
    expected |= {(p, 'SharedDisabled', 'Disabled') for p in tasks}
    keys = [(r['pair_id'], r['arm'], r['condition']) for r in rows]
    if len(set(keys)) != 1250 or set(keys) != expected:
        raise ValueError('Duplicate/missing/unplanned record')
    lookup = {(r['pair_id'], r['arm'], r['condition']): r for r in rows}
    source_tasks = defaultdict(list)
    for task in tasks.values():
        gr, gc = divmod(task['goal_index_evaluator_only'], 5)
        expected_indices = [task['ground_feature_index']*81 + (4-gr+r)*9 + 4-gc+c for r in range(5) for c in range(5)]
        if task['aerial_feature_indices'] != expected_indices:
            raise ValueError('Physical5x5 camera-center mapping drift')
        source_tasks[task['group_evaluator_only']].append(task)
    if len(source_tasks) != 5 or any(len(v) != 25 for v in source_tasks.values()):
        raise ValueError('Five complete source units required')
    full_counts, per_source, controls = {}, {}, {}
    for arm in ARMS:
        full_counts[arm] = {c: counts([lookup[p, arm, c] for p in tasks]) for c in ('Full', 'Mean', 'Wrong')}
        controls[arm] = {}
        for condition in ('Mean', 'Wrong'):
            records = [(lookup[p, arm, 'Full'], lookup[p, arm, condition]) for p in tasks]
            controls[arm][condition] = {
                'planned': 125,
                'same_top1_cell': sum(a['ranking'][0] == b['ranking'][0] for a, b in records),
                'full_target_rank_better': sum(a['true_rank'] < b['true_rank'] for a, b in records),
                'full_target_rank_worse': sum(a['true_rank'] > b['true_rank'] for a, b in records),
                'full_target_rank_tied': sum(a['true_rank'] == b['true_rank'] for a, b in records),
                'mean_top5_overlap': mean(len(set(a['ranking'][:5]) & set(b['ranking'][:5])) / 5 for a, b in records),
            }
    full_counts['SharedDisabled'] = {'Disabled': counts([lookup[p, 'SharedDisabled', 'Disabled'] for p in tasks])}
    for group, selected in sorted(source_tasks.items()):
        ids = {t['source_id_evaluator_only'] for t in selected}
        queries = {t['ground_feature_index'] for t in selected}
        physical_truths = {t['aerial_feature_indices'][t['goal_index_evaluator_only']] for t in selected}
        union = set().union(*(set(t['aerial_feature_indices']) for t in selected))
        common = set.intersection(*(set(t['aerial_feature_indices']) for t in selected))
        if len(ids) != 1 or len(queries) != 1 or len(physical_truths) != 1 or len(union) != 81 or common != physical_truths:
            raise ValueError('Broken physical/source/query identity')
        if sorted(t['goal_index_evaluator_only'] for t in selected) != list(range(25)):
            raise ValueError('Each source must retain all25 proxy placements')
        source = {'source_id': next(iter(ids)), 'windows': 25, 'unique_ground_queries': 1,
                  'unique_positive_tiles': 1, 'unique_aerial_tiles': 81, 'positive_local_indices': sorted(t['goal_index_evaluator_only'] for t in selected), 'arms': {}}
        for arm in ARMS:
            arm_stats = {}
            for condition in ('Full', 'Mean', 'Wrong'):
                scores_by_tile = defaultdict(list)
                winners, offsets, margins, stronger = Counter(), [], [], []
                subset = []
                for task in selected:
                    row = lookup[task['pair_id'], arm, condition]
                    truth = validate_row(row)
                    if row['group'] != group or truth != task['goal_index_evaluator_only']:
                        raise ValueError('Saved task/truth mapping drift')
                    indices = task['aerial_feature_indices']
                    for j, score in enumerate(row['scores']):
                        scores_by_tile[indices[j]].append(score)
                    winner = indices[row['ranking'][0]] % 81
                    positive = indices[truth] % 81
                    winners[winner] += 1
                    offsets.append(abs(winner // 9 - positive // 9) + abs(winner % 9 - positive % 9))
                    margins.append(row['scores'][truth] - max(s for j, s in enumerate(row['scores']) if j != truth))
                    stronger.append(sum(s > row['scores'][truth] for s in row['scores']))
                    subset.append(row)
                delta = max(max(v) - min(v) for v in scores_by_tile.values())
                if delta > 1e-12:
                    raise ValueError('Repeated physical candidate scores changed')
                arm_stats[condition] = dict(counts(subset), unique_top1_physical_tiles=len(winners),
                    modal_top1_windows=max(winners.values()), modal_top1_tiles=sorted(k for k, v in winners.items() if v == max(winners.values())),
                    median_top1_to_proxy_grid_L1=median(offsets), median_target_minus_best_negative=median(margins),
                    windows_with_strictly_higher_negative=sum(v > 0 for v in stronger), repeated_tile_score_max_delta=delta)
            source['arms'][arm] = arm_stats
        per_source[group] = source
    training = queue['training_queries']
    fit_groups = {t['group_evaluator_only'] for t in training}
    if len(training) != 9 or len(fit_groups) != 9 or fit_groups & set(source_tasks):
        raise ValueError('Fit/development groups must be isolated in the frozen queue')
    return {'rankings': 1250, 'windows': 125, 'distinct_development_sources': 5,
        'distinct_development_ground_queries': 5, 'unique_development_physical_aerial_tiles': 405,
        'fit_queries': 9, 'fit_source_groups': 9, 'fit_positive_tiles': 9,
        'fit_query_positive_relations_per_group': 1,
        'counts': full_counts, 'target_sensitivity': controls, 'per_source': per_source,
        'error_distance_unit': 'posthoc 9x9 bank grid L1 to camera proxy; NOT navigation SG, meters, or a new accepted target',
        'new_model_predictions': 0}


def collect():
    index = json.loads((ROOT / '项目导航/实验索引.json').read_text(encoding='utf-8-sig'))
    batches = [b for b in index['batches'] if re.fullmatch(r'CV-0(?:0[1-9]|1[0-4])', b.get('category', ''))]
    if len(batches) != 14:
        raise ValueError('Fourteen sealed predecessors required')
    paths = {str(BASE / '同任务来源用途账本_v1.json'), str(QUEUE), str(RUN / '逐题排序结果.json'), str(RUN / '开发对照汇总.json'),
        str(RUN / '核验/最终门槛判定.json'), str(RUN / '核验/另实现复核.json'), str(RUN / '训练日志.json'), str(RUN / '执行完成.json'),
        str(BASE / '目标格定义与视觉尺度审查_v1/审查汇总.json'), str(BASE / '弱场景与拍摄点迁移差异诊断_v1/目标语义与证据边界.json')}
    for _, folder, summary, rows, _, _ in SERIES:
        paths.update((str(BASE / folder / summary), str(BASE / folder / rows)))
    paths.update(str(Path(b['path']) / '阶段封存.json') for b in batches)
    protected = text_json(RUN / '首forward绑定.json')['protected'][:4]
    output = ROOT / OUT
    if output.exists():
        raise FileExistsError('Never overwrite a completed/partial CV-015 batch')
    output.mkdir(); (output / '核验').mkdir()
    binding = {'task': 'CV-015', 'date': '2026-10-18', 'inputs': [digest(p) for p in sorted(paths)],
        'sources': [digest(p) for p in SOURCES], 'protected_current_files': [digest(e['path']) for e in protected],
        'scope': 'saved JSON rankings and metadata only; no raw images, feature arrays, weights, model inference or new predictions',
        'outputs_limit_bytes': 3 * 1024**2, 'confirmation_groups_access': 0,
        'statistical_plan': ['fixed-series count reconciliation without combining protocols', '1250 stored rankings and25-score tie/count checks',
            'physical candidate reuse and query/positive identity per source', 'Full/Mean/Wrong winner and rank comparisons',
            'posthoc error offset and saved target score margins by source', 'failure boundaries and conditional next direction, no efficacy gate changes']}
    save('输入与源码绑定.json', binding)
    old_evidence = {}
    for batch in batches:
        seal = text_json(Path(batch['path']) / '阶段封存.json')
        for item in seal['files']:
            old_evidence[item['path']] = item['sha256']
    for item in binding['inputs']:
        if not item['path'].endswith('阶段封存.json') and item['path'] != (BASE / '同任务来源用途账本_v1.json').as_posix():
            if old_evidence.get(item['path']) != item['sha256']:
                raise ValueError(('Input no longer matches original stage seal', item['path']))
    history = []
    for category, folder, _, rows_name, dataset, arm in SERIES:
        rows = text_json(BASE / folder / rows_name)['rows']
        if dataset: rows = [r for r in rows if r['dataset'] == dataset]
        if arm: rows = [r for r in rows if r.get('arm') == arm]
        if not rows: raise ValueError((category, 'empty frozen selection'))
        history.append({'series': category, 'batch_path': (BASE / folder).as_posix(), 'arm': arm or 'original',
            'label_semantics': 'weak released common-cell proxy' if dataset == 'weak_dev' else 'CSV/code camera-point proxy',
            'counts': {c: counts([r for r in rows if r['condition'] == c]) for c in ('Full', 'Mean', 'Wrong')},
            'source_groups': len({r.get('group', r.get('group_id')) for r in rows}), 'navigation_SR': None})
    current = summarize_current(text_json(RUN / '逐题排序结果.json')['rows'], text_json(QUEUE))
    if current['counts'] != text_json(RUN / '开发对照汇总.json')['counts']:
        raise ValueError('Existing CV-014 aggregate disagrees')
    roles = text_json(BASE / '同任务来源用途账本_v1.json')['records']
    role_counts = Counter(r['role'] for r in roles)
    reserved = [r for r in roles if r['role'] == 'confirmation_reserved']
    if len(reserved) != 24 or len({r['group'] for r in reserved}) != 6:
        raise ValueError('Confirmation ledger changed')
    log = text_json(RUN / '训练日志.json')
    result = {'task': 'CV-015', 'historical_series': history, 'current': current, 'current_role_counts': dict(role_counts),
        'reserved_groups': 6, 'reserved_records': 24, 'fit_loss_first': log['log'][0]['loss_before_update'], 'fit_loss_final': log['final_fit_loss'],
        'fit_updates': log['updates'], 'fit_supervision_uses': log['supervision_uses'],
        'effect_gates_failed': [k for k,v in text_json(RUN / '核验/最终门槛判定.json')['gates'].items() if not v],
        'new_raw_reads': 0, 'new_model_forward_calls': 0, 'new_predictions': 0, 'new_training_steps': 0,
        'new_navigation_episodes': 0, 'new_API_calls': 0, 'confirmation_access': 0,
        'new_candidate_admitted': False, 'next_direction': 'audit the camera-point labels and within-area distinguishability on the14 consumed fit/dev representatives before proposing same-area multi-query supervision; no training or new-source use admitted'}
    save('证据重计汇总.json', result)
    for e in binding['inputs'] + binding['sources'] + binding['protected_current_files']:
        if digest(e['path']) != e: raise ValueError('Input/source/protected file changed')
    print(json.dumps({'collected': True, 'historical_series': len(history), 'saved_rankings_checked': 1250 + 48 + 48 + 48 + 375 + 48,
                      'new_model_calls': 0, 'next_candidate_admitted': False}, ensure_ascii=False))


if __name__ == '__main__':
    collect()
