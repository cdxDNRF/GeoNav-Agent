"""CV-020: text-only CV-019 review; synthetic fault injection, no real inference.

Does not import project executors, torch or transformers. The only compiled
historical function is AST-extracted encode(), supplied entirely in-memory
stubs. Never read raw imagery, parameters, or feature files.
"""
import ast
import contextlib
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'DATA/processed_data/MM-GAG'
OLD = BASE / '同任务拍摄点对齐执行_v1'
RUN = BASE / '配对覆盖同预算执行_v1'
DESIGN = BASE / '三新增来源工程与配对覆盖协议_v1'
OUT = BASE / '配对覆盖复查与后续计划_v1'
CANDIDATE = 'MMRank8Coverage12'
TEXT_SUFFIXES = {'.py', '.json', '.jsonl', '.md', '.txt'}
READS = {}


def text_file(path):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    if relative.startswith('DATA/raw_data/') or path.suffix not in TEXT_SUFFIXES:
        raise ValueError('Review scope excludes raw/binary inputs: ' + relative)
    data = path.read_bytes()
    READS[relative] = {'path': relative, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return data.decode('utf-8-sig')


def read(path):
    return json.loads(text_file(path))


def verify(entry):
    text_file(ROOT / entry['path'])
    actual = READS[entry['path']]
    assert actual['sha256'] == entry['sha256'], entry['path']
    assert 'bytes' not in entry or actual['bytes'] == entry['bytes'], entry['path']


def percentile(values, probability):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * probability
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)


def injected_interruption(source):
    """Run historical control flow with fake 16-row input; no I/O/real model."""
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'encode')
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    persisted, events, buffers = {}, [], []

    class FakeTensor:
        def __init__(self, values): self.values = values
        def to(self, *_): return self
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.values

    class FakeEncoder:
        training = False
        calls = 0
        def parameters(self): return []
        def __call__(self, **_):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('CV020_SYNTHETIC_SECOND_BATCH_FAILURE')
            return SimpleNamespace(image_embeds=FakeTensor(np.ones((8, 512), dtype=np.float32)))

    class FakeDestination:
        def __init__(self, shape): self.array = np.full(shape, np.nan, dtype=np.float32)
        def __setitem__(self, index, value): self.array[index] = value
        def flush(self): pass

    def open_memory_array(*_, shape, **__):
        value = FakeDestination(shape)
        buffers.append(value)
        return value

    model = FakeEncoder()
    fake_torch = SimpleNamespace(from_numpy=FakeTensor, inference_mode=contextlib.nullcontext)
    fake_numpy = SimpleNamespace(float32=np.float32, nan=np.nan, stack=np.stack, isfinite=np.isfinite,
        lib=SimpleNamespace(format=SimpleNamespace(open_memmap=open_memory_array)))
    namespace = {'np': fake_numpy, 'OUT': Path('in_memory_only'), 'OLD': Path('in_memory_only'),
        'load_encoder': lambda *_: (model, []), 'tensor_sha': lambda _: 'fake_same_tensor',
        'read': lambda _: {'tensor_sha': 'fake_same_tensor'},
        'put': lambda name, value: persisted.__setitem__(name, value),
        'event': lambda event_type, **fields: events.append({'event': event_type, **fields}),
        'pixels': lambda item, kind: (np.zeros((3, 2, 2), np.float32),
            {'raw_sha256': 'fake_raw_' + item['path'], 'pixel_sha256': 'fake_pixel_' + item['path']})}
    previous = sys.modules.get('torch')
    sys.modules['torch'] = fake_torch
    try:
        exec(compile(tree, '<CV019-encode-in-memory-stubs>', 'exec'), namespace)
        try:
            namespace['encode']('aerial', [{'path': 'synthetic_' + str(i)} for i in range(16)], {'device': 'cpu'})
        except RuntimeError as error:
            assert str(error) == 'CV020_SYNTHETIC_SECOND_BATCH_FAILURE'
        else:
            raise AssertionError('Expected synthetic interruption')
    finally:
        if previous is None:
            del sys.modules['torch']
        else:
            sys.modules['torch'] = previous
    saved = [e for e in events if e['event'] == 'encoder_saved']
    finite_rows = np.flatnonzero(np.isfinite(buffers[0].array).all(axis=1)).tolist()
    assert model.calls == 2 and finite_rows == list(range(8))
    assert saved == [{'event': 'encoder_saved', 'kind': 'aerial', 'indices': list(range(8))}]
    assert '核验/aerial_像素回执.json' not in persisted
    return {'reproduced': True, 'synthetic_encoder_calls': model.calls,
        'successful_feature_rows': finite_rows, 'saved_events': saved,
        'pixel_receipt_file_present_after_interruption': False,
        'actual_model_forward_calls': 0, 'actual_raw_reads': 0, 'actual_feature_writes': 0,
        'method': 'AST encode control flow, fake torch/model/pixels/memmap/put/event, second-batch injected failure'}


def check():
    binding = read(RUN / '首forward绑定.json')
    seal = read(RUN / '阶段封存.json')
    for key in ('sources', 'dependencies', 'inputs'):
        for entry in binding[key]: verify(entry)
    for key in ('authorization', 'reuse_plan'): verify(binding[key])
    for entry in binding['protected'][:2]: verify(entry)
    hashed, size_only = [], []
    for entry in seal['files'] + seal['sources'] + seal['readonly_dependencies']:
        path = ROOT / entry['path']
        assert path.is_file(), path
        if 'bytes' in entry: assert path.stat().st_size == entry['bytes'], path
        if path.suffix in TEXT_SUFFIXES:
            verify(entry)
            hashed.append(entry['path'])
        else:
            size_only.append(entry['path'])

    queue = read(DESIGN / '固定训练与开发队列.json')
    prior_queue = read(BASE / '同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json')
    assert len(queue['ground_encoder_inputs']) == 17 and len(queue['aerial_encoder_inputs']) == 1377
    assert queue['ground_encoder_inputs'][:14] == prior_queue['ground_encoder_inputs']
    assert queue['aerial_encoder_inputs'][:1134] == prior_queue['aerial_encoder_inputs']
    assert queue['development_tasks'] == prior_queue['development_tasks']
    fit = queue['training_queries']
    train_g = [q['ground_feature_index'] for q in fit]
    dev = queue['development_tasks']
    dev_g = {q['ground_feature_index'] for q in dev}
    assert train_g == list(range(9)) + [14, 15, 16] and dev_g == set(range(9, 14))
    assert not set(train_g) & dev_g and queue['mean_ground_feature_indices'] == list(range(9))
    assert len({q['group_evaluator_only'] for q in fit}) == 9
    assert len({q['group_evaluator_only'] for q in dev}) == 5
    train_a = set(itertools.chain.from_iterable(q['aerial_feature_indices'] for q in fit))
    dev_a = set(itertools.chain.from_iterable(q['aerial_feature_indices'] for q in dev))
    assert not train_a & dev_a
    for q in fit:
        assert len(q['aerial_feature_indices']) == 81 and q['positive_candidate_index_evaluator_only'] == 40
    for task in dev:
        assert task['wrong_ground_feature_index'] == 9 + ((task['ground_feature_index'] - 9 + 1) % 5)
    roles = read(BASE / '同任务来源用途账本_v1.json')['records']
    role_by_id = {r['source_id']: r for r in roles}
    assert all(role_by_id[q['source_id_evaluator_only']]['role'] == 'fit_preparation' for q in fit)
    assert all(role_by_id[q['source_id_evaluator_only']]['role'] == 'development' for q in dev)
    reserved = [r for r in roles if r['role'] == 'confirmation_reserved']
    assert len(reserved) == 24 and len({r['group'] for r in reserved}) == 6

    plan = read(RUN / '特征复用计划.json')
    reused = [x for x in plan['aerial_assignment'] if x['source'] == 'old']
    fresh = [x for x in plan['aerial_assignment'] if x['source'] == 'new']
    assert len(reused) == 35 and len(fresh) == 208
    for x in reused:
        assert x['index'] < 729
        assert queue['aerial_encoder_inputs'][x['index']]['sha256'] == x['raw_sha256']
    for x in plan['aerial_assignment']:
        assert queue['aerial_encoder_inputs'][x['global_index']]['sha256'] == x['raw_sha256']
    assert len(plan['new_ground_items']) == 3 and plan['new_unique_encoder_inputs'] == 211 and plan['new_encoder_calls'] == 27
    event_rows = [json.loads(line) for line in text_file(RUN / '事件.jsonl').splitlines()]
    saved_rows = [(r['kind'], i) for r in event_rows if r['event'] == 'encoder_saved' for i in r['indices']]
    assert Counter(saved_rows) == Counter([('ground', i) for i in range(3)] + [('aerial', i) for i in range(208)])
    for kind, count in [('ground', 3), ('aerial', 208)]:
        receipts = read(RUN / ('核验/' + kind + '_像素回执.json'))['receipts']
        assert len(receipts) == count and {x['index'] for x in receipts} == set(range(count))

    log = read(RUN / '训练日志.json')
    assert len(log['log']) == 500 and log['supervision_uses'] == 4500
    uses = Counter()
    for t, item in enumerate(log['log']):
        assert item['step'] == t + 1
        expected = [(9 * t + j) % 12 for j in range(9)]
        assert item['query_indices'] == expected
        uses.update(expected)
    assert list(sorted(uses.items())) == [(i, 375) for i in range(12)]
    assert log['uses_per_query'] == [375] * 12
    assert log['trainable_parameters'] == 8192 and log['encoder_optimizer_parameters'] == 0

    refs = read(RUN / '合并结果引用.json')
    verify(refs['old']); verify(refs['new'])
    old_rows = read(ROOT / refs['old']['path'])['rows']
    new_rows = read(ROOT / refs['new']['path'])['rows']
    assert len(old_rows) == 1250 and len(new_rows) == 375
    rows = old_rows + new_rows
    task_by_id = {t['pair_id']: t for t in dev}
    lookup = {(r['pair_id'], r['arm'], r['condition']): r for r in rows}
    arms = ['Identity', 'SwissRank8Frozen', 'MMRank8Final', CANDIDATE]
    combos = [(a, c) for a in arms for c in ('Full', 'Mean', 'Wrong')] + [('SharedDisabled', 'Disabled')]
    assert len(task_by_id) == 125 and len(lookup) == 1625
    assert set(lookup) == {(p, a, c) for p in task_by_id for a, c in combos}
    for r in rows:
        task = task_by_id[r['pair_id']]
        assert r['group'] == task['group_evaluator_only'] and r['truth_evaluator_only'] == task['goal_index_evaluator_only']
        order, scores = r['ranking'], r['scores']
        assert sorted(order) == list(range(25)) and len(scores) == 25
        assert all(math.isfinite(x) for x in scores)
        assert all(scores[order[i]] >= scores[order[i+1]] for i in range(24))
        rank = order.index(r['truth_evaluator_only']) + 1
        assert rank == r['true_rank'] and r['top1'] == (rank == 1) and r['top5'] == (rank <= 5)
    summary = read(RUN / '开发对照汇总.json')
    for arm, condition in combos:
        selected = [r for r in rows if r['arm'] == arm and r['condition'] == condition]
        expected = {'planned': 125, 'top1': sum(r['top1'] for r in selected),
            'top5': sum(r['top5'] for r in selected), 'median_rank': statistics.median(r['true_rank'] for r in selected)}
        assert expected == summary['counts'][arm][condition]
        for group in summary['per_source']:
            subset = [r for r in selected if r['group'] == group]
            assert len(subset) == 25
            assert {'planned':25, 'top1':sum(r['top1'] for r in subset), 'top5':sum(r['top5'] for r in subset)} == summary['per_source'][group][arm][condition]
    specs = [('Full_vs_' + c, (CANDIDATE, c)) for c in ('Mean', 'Wrong')]
    specs += [('Full_vs_Disabled', ('SharedDisabled', 'Disabled'))]
    specs += [('Full_vs_' + a, (a, 'Full')) for a in arms[:-1]]
    for name, control in specs:
        for metric in ('top1', 'top5'):
            delta = {}
            recovered = harmed = 0
            for group in sorted(summary['per_source']):
                ids = [p for p, t in task_by_id.items() if t['group_evaluator_only'] == group]
                differences = [int(lookup[p, CANDIDATE, 'Full'][metric]) - int(lookup[p, *control][metric]) for p in ids]
                recovered += differences.count(1); harmed += differences.count(-1)
                delta[group] = sum(differences) / 25
            exp = summary['contrasts'][name][metric]
            assert (recovered, harmed, recovered-harmed) == (exp['recovered'], exp['harmed'], exp['net'])
            assert delta == exp['per_source_rate_differences']
            values = [sum(x) / 5 for x in itertools.product(delta.values(), repeat=5)]
            for got, want in zip([percentile(values, .025), percentile(values, .975)], exp['paired_source_percentile95']): close(got, want)

    decision = read(RUN / '核验/最终门槛判定.json')
    top = lambda a, c='Full': summary['counts'][a][c]['top1']
    per = summary['per_source']; full = top(CANDIDATE)
    positives = lambda tests: sum(all(topper(g, a, c) < topper(g, CANDIDATE, 'Full') for a, c in tests) for g in per)
    def topper(g, a, c): return per[g][a][c]['top1']
    gates = {'all1625_planned_rankings_complete': len(lookup) == 1625,
        'all_engineering_and_audit_passed': read(RUN / '核验/工程另实现复核.json')['passed'] and read(RUN / '核验/完整另实现复核.json')['passed'],
        'Full_top1_at_least32': full >= 32,
        'Full_minus_Mean_at_least16': full - top(CANDIDATE, 'Mean') >= 16,
        'Full_minus_Wrong_at_least16': full - top(CANDIDATE, 'Wrong') >= 16,
        'Full_minus_Disabled_at_least16': full - top('SharedDisabled', 'Disabled') >= 16,
        'Full_minus_Identity_at_least5': full - top('Identity') >= 5,
        'Full_minus_Swiss_at_least5': full - top('SwissRank8Frozen') >= 5,
        'sources_above_both_controls_at_least3': positives([(CANDIDATE, 'Mean'), (CANDIDATE, 'Wrong')]) >= 3,
        'sources_above_both_baselines_at_least3': positives([('Identity', 'Full'), ('SwissRank8Frozen', 'Full')]) >= 3,
        'Full_minus_MM9_at_least5': full - top('MMRank8Final') >= 5,
        'sources_above_MM9_at_least3': positives([('MMRank8Final', 'Full')]) >= 3}
    assert gates == decision['gates'] and not any(list(gates.values())[2:]) and not decision['passed']
    runner_source = text_file(ROOT / 'project/src/tools/run_capturepoint_coverage_v1.py')
    proof = injected_interruption(runner_source)
    return {'task':'CV-020', 'review_checks_passed':True, 'candidate_passed':False,
        'sealed_text_and_source_hashes_verified':len(set(hashed)), 'sealed_binary_presence_size_only':size_only,
        'binary_integrity_limit':'No fresh binary SHA or tensor/score recomputation; CV019 original alternate-equation audit retained, not rerun.',
        'fit_ground_indices':train_g, 'development_ground_indices':sorted(dev_g), 'reused_old_train_aerial_rows':len(reused),
        'updates':500, 'query_uses':4500, 'uses_per_query':375, 'head_parameters':8192,
        'ranking_records':1625, 'planned_windows':125, 'target_photos_and_source_units':5,
        'control_counts':summary['counts'][CANDIDATE], 'all_full_counts':{a:summary['counts'][a]['Full'] for a in arms},
        'source_full_counts':{g:per[g][CANDIDATE]['Full'] for g in per},
        'MM12_vs_MM9':summary['contrasts']['Full_vs_MMRank8Final'], 'gates':gates,
        'reserved_groups':6, 'reserved_records':24, 'finding_count':1,
        'findings':[{'priority':'P2', 'file':'project/src/tools/run_capturepoint_coverage_v1.py', 'line':122,
            'title':'Per-batch features survive interruption without their pixel receipts',
            'rule':'AGENTS.md:49', 'proof':'中断注入证据.json',
            'impact':'Reproducibility/recovery gap on a future interrupted encoding run; completed CV019 receipts all present, negative result not invalidated.'}],
        'scope':{'real_model_forward_calls':0,'new_model_scores':0,'raw_images_opened':0,
            'feature_or_parameter_files_opened':0,'training_updates':0,'navigation':0,'confirmation_consumption':0},
        'same_agent_review_not_independent_person':True}, proof


def main():
    result, proof = check()
    text_file(Path(__file__))
    assert len(READS) > 0
    binding = {'task':'CV-020','source':READS[Path(__file__).relative_to(ROOT).as_posix()],
        'inputs':[READS[k] for k in sorted(READS) if k != Path(__file__).relative_to(ROOT).as_posix()],
        'binary_files_not_opened':True,'immutable_inputs_only':True}
    OUT.mkdir(exist_ok=True)
    for name, value in [('输入与源码绑定.json',binding),('中断注入证据.json',proof),('复查汇总.json',result)]:
        path = OUT / name
        if path.exists(): raise FileExistsError('Do not overwrite an existing review: '+str(path))
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'review_checks_passed':True,'findings':1,'ranking_records':1625,
        'candidate_passed':False,'real_forward_calls':0,'outputs_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())},ensure_ascii=False))


if __name__ == '__main__': main()
