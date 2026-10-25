"""CV-024 one bounded cached-feature closed loop; freeze/run are exclusive.

No encoder loading, raw pixel reads, API, training or reserved-group access.
Invoke from repository root with python -B -m project.src.tools.run_ground_local_loop_v1.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'project/src'))
BASE = ROOT / 'DATA/processed_data/MM-GAG'
OUT = BASE / '目标语义与局部闭环工程_v1'
QUEUE = BASE / '同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json'
LEDGER = BASE / '同任务来源用途账本_v1.json'
SRC = ['agents/ground_local_loop_v1.py', 'tools/run_ground_local_loop_v1.py',
       'eval/audit_ground_local_loop_v1.py', 'tests/test_ground_local_loop_v1.py']
ORDER = ('Full', 'Disabled', 'Mean', 'Wrong', 'Frontier')
SOURCE_ORDER = ('mmgag_062', 'mmgag_073', 'mmgag_079', 'mmgag_039', 'mmgag_081')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def entry(p):
    p = Path(p)
    return dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), bytes=p.stat().st_size)


def verify(items):
    for e in items:
        p = ROOT / e['path']
        if ('bytes' in e and p.stat().st_size != e['bytes']) or sha(p) != e['sha256']:
            raise ValueError('Input changed: ' + e['path'])


def put(name, obj):
    with (OUT/name).open('x', encoding='utf-8', newline='\n') as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
        f.write('\n')


def tag(x): return hashlib.sha256(('CV-024/tasks/v1:' + x).encode()).hexdigest()


def distance(a, b): return abs(a//5-b//5) + abs(a%5-b%5)


def make_tasks(queue):
    result = []
    for source in SOURCE_ORDER:
        choices = [t for t in queue['development_tasks'] if t['source_id_evaluator_only'] == source]
        if len(choices) != 25: raise ValueError('Expected old25 windows/source')
        for d in (4, 5, 6, 8):
            eligible = [(t, i) for t in choices for i in range(25) if distance(i, t['goal_index_evaluator_only']) == d]
            t, start = min(eligible, key=lambda x: tag(f'{source}/{d}/{x[0]["pair_id"]}/{x[1]}'))
            row = dict(t)
            row.pop('not_navigation')
            row.update(task_id=f'cv024_{source}_c{d}', parent_window=t['pair_id'],
                       start=start, distance_evaluator_only=d, grid=5, budget=10)
            # Independent original numeric-array mapping: no source filename,
            # crop origin or this goal index is sent to the policy.
            gr, gc = divmod(row['goal_index_evaluator_only'], 5)
            expected = [(5-gr+r, 5-gc+c) for r in range(5) for c in range(5)]
            actual = []
            for i in row['aerial_feature_indices']:
                match = re.search(r'/sat_(\d+)_(\d+)\.jpg$', queue['aerial_encoder_inputs'][i]['path'])
                if not match: raise ValueError('Unexpected raw tile name')
                actual.append(tuple(map(int, match.groups())))
            if actual != expected or actual[row['goal_index_evaluator_only']] != (5, 5):
                raise ValueError('Original center/window axis differs')
            result.append(row)
    return result


def freeze():
    if OUT.exists(): raise FileExistsError('Already prepared: inspect existing state; never refreeze')
    queue = read(QUEUE)
    roles = read(LEDGER)['records']
    dev = {r['source_id'] for r in roles if r['role'] == 'development' and r.get('representative')}
    if dev != set(SOURCE_ORDER): raise ValueError('Known-dev source roles drifted')
    assets = read(BASE/'配对覆盖同预算执行_v1/首forward绑定.json')
    features = assets['old_features']
    models = read(BASE/'同任务拍摄点对齐执行_v1/首forward绑定.json')['models']
    # Protect all explicitly bound historical inputs without copying any data.
    protected = features + models + assets['protected']
    platform_seal = ROOT/'平台/实验工作台_v4/试跑保存与同题对照_v1/阶段封存.json'
    platform = read(platform_seal)
    protected += platform['files'] + platform['sources']
    historical = [BASE/'跨视角感知最小验证_v1/阶段封存.json',
                  BASE/'场景与精确格信号分解_v1/阶段封存.json',
                  BASE/'单表征目标对照开发_v1/阶段封存.json']
    for p in historical:
        d = read(p); protected += d['files'] + d['sources']
    protected = list({e['path']: e for e in protected}.values())
    verify(protected)
    tasks = make_tasks(queue)
    references = [ROOT/'GOMAA-Geo/models/ppo.py', ROOT/'GOMAA-Geo/data_utils/get_sat_embeddings_sat2cap.py',
                  ROOT/'GOMAA-Geo/data_utils/sequence.py', ROOT/'DATA/raw_data/MM-GAG/coordinates.csv',
                  BASE/'拍摄点语义与完整目标图验证_v1/目标语义约定.json',
                  BASE/'同任务拍摄点对齐执行_v1/首forward绑定.json',
                  BASE/'同任务对齐执行器工程_v2/阶段封存.json',
                  BASE/'配对覆盖同预算执行_v1/首forward绑定.json']
    OUT.mkdir()  # Exact user-approved path, registered in root README first.
    put('目标语义与资产核验.json', {
        'task': 'CV-024', 'code_reference': 'GOMAA-Geo/models/ppo.py:validate_ground_level_custom',
        'original_center': [5,5], 'window': 'rows[5-goal//5:10-goal//5], cols[5-goal%5:10-goal%5]',
        'all20_windows_match_original_center_and_axis': True,
        'operational_target': 'the dataset paired center proxy cell, not an independently surveyed camera point or annotated visible landmark',
        'old_capture_point_contract': 'preserved verbatim; this batch clarifies evidential scope without relabeling',
        'CSV': 'filename/lat/lon present; no pixel georeferencing/heading measurement inferred from this table',
        'encoders': 'existing public CLIP ViT-B/32 ground and local Sat2Cap aerial 512D cached outputs, no trained head',
        'compatibility': 'existing shape/preprocessor/strict-load receipts and actual SHA verified; dimensional compatibility does not prove alignment accuracy',
        'limits': 'local Sat2Cap model card is a template, original paper run/encoder mapping unresolved; not a paper reproduction',
        'cvusa_and_tiny_heads': 'not used or revived', 'reserved_groups_read': 0})
    put('任务清单.json', {'tasks': tasks, 'sources': 5, 'independent_geography_certified': False})
    protocol = {
        'task': 'CV-024', 'date': '2026-10-22', 'status': 'engineering development, not formal confirmation',
        'hypothesis': 'a local affine semantic trend can guide equal-cost frontier choices using only visited observations',
        'one_factor': 'semantic tie-break of nearest-frontier exploration; frozen paired encoders, no head/capacity/threshold change',
        'arms': list(ORDER), 'unique_tasks': 20, 'planned_episodes': 100, 'grid': 5, 'budget': 10,
        'task_selection': 'hash of fixed known-dev window/start, without feature scores;5 sources x C4/5/6/8',
        'formula': 's_i=cos(a_i,g); b=lstsq(X-mean(X),s-mean(s)); choose min(path_cost,-s_hat,public_cell_id)',
        'forecast_precision': 'round predicted score to12 decimal places for stable exact ties',
        'model_input': 'one goal512, current512, public cell/grid/step/budget only; internal history reset per episode',
        'topology': 'known grid geometry; paths through visited cells with at most one unknown endpoint',
        'conditions': {'Full': 'same paired ground vector', 'Disabled': 'goal=None',
                       'Mean': 'unit(mean(unit(fit ground[0:9]))) only',
                       'Wrong': 'fixed previous-protocol cyclic dev ground index',
                       'Frontier': 'same path and tie ordering, no target evidence'},
        'actions': 'up/right/down/left; out-of-bounds would cost one step; first arrival/last-step success; no stop',
        'gate_scope': 'bounded mechanism screen only; no static Top-k pass required; historical failed gates unchanged',
        'screen': {'Full_net_success_vs_each_control_and_Frontier_min': 2,
                   'Full_SG_no_worse_than_each': True, 'positive_sources_vs_Frontier_min': 3,
                   'audit_and_permission_tests_required': True},
        'stop': 'single execution; failed screen closes this simple local-gradient factor, no tuning/relabel/new-source/parameter scan',
        'resources': {'output_bytes': 8*1024**2, 'CPU_seconds': 600, 'max_actions': 1000,
                      'encoder_forward': 0, 'train_updates': 0, 'raw_pixel_reads': 0, 'downloads': 0, 'API': 0, 'reserved_group_features': 0},
        'not_authorized_by_result': 'automatic default/GUI release, new model, training or confirmation',
        'retention': 'small code/config/task/trajectory/metric/audit evidence; no feature/model copies'}
    put('冻结协议.json', protocol)
    bound = {'task': 'CV-024', 'sources': [entry(ROOT/'project/src'/p) for p in SRC],
             'inputs': [entry(p) for p in [QUEUE, LEDGER, platform_seal, *historical, *references]],
             'features': features, 'models': models, 'protected': protected,
             'protocol': entry(OUT/'冻结协议.json'), 'tasks': entry(OUT/'任务清单.json')}
    put('首动作绑定.json', bound)
    print(json.dumps({'frozen': True, 'tasks': len(tasks), 'planned_episodes': 100, 'protected': len(protected), 'model_calls': 0}))


def summary(rows):
    result = {}
    for arm in ORDER:
        a = [r for r in rows if r['arm'] == arm]
        result[arm] = {'planned': 20, 'completed': len(a), 'success': sum(r['success'] for r in a),
                       'SR': sum(r['success'] for r in a)/20, 'SG_cells': sum(r['SG'] for r in a)/20,
                       'mean_moves': sum(r['moves'] for r in a)/20,
                       'mean_coverage': sum(r['coverage'] for r in a)/20,
                       'by_source': {}, 'by_distance': {}}
        for field, vals, key in [('source', SOURCE_ORDER, 'by_source'), ('distance', (4,5,6,8), 'by_distance')]:
            for v in vals:
                subset = [r for r in a if r[field] == v]
                result[arm][key][str(v)] = {'n': len(subset), 'success': sum(r['success'] for r in subset),
                                           'SR': sum(r['success'] for r in subset)/len(subset),
                                           'SG': sum(r['SG'] for r in subset)/len(subset)}
    return result


def execute():
    if (OUT/'执行开始.json').exists(): raise FileExistsError('Consumed/partial execution. Read saved outputs; no blind rerun')
    b = read(OUT/'首动作绑定.json'); protocol = read(OUT/'冻结协议.json')
    verify(b['sources']+b['inputs']+b['features']+b['models']+b['protected']+[b['protocol'], b['tasks']])
    from tests.test_ground_local_loop_v1 import Tests
    tests = unittest.TestResult(); unittest.defaultTestLoader.loadTestsFromTestCase(Tests).run(tests)
    put('测试核验.json', {'tests': tests.testsRun, 'failures': tests.failures, 'errors': tests.errors,
                         'passed': tests.wasSuccessful(), 'real_inputs': 0})
    if not tests.wasSuccessful(): raise ValueError('Engineering tests failed; no navigation')
    # Import the policy only after binding. Cache banks stay in evaluator scope.
    from agents.ground_local_loop_v1 import GroundLocalLoop
    tasks = read(OUT/'任务清单.json')['tasks']
    ground = np.load(ROOT/b['features'][0]['path'], allow_pickle=False)
    aerial = np.load(ROOT/b['features'][1]['path'], allow_pickle=False)
    if ground.shape != (14,512) or aerial.shape != (1134,512): raise ValueError('Frozen cache dimensions differ')
    if not np.isfinite(ground).all() or not np.isfinite(aerial).all(): raise ValueError('Nonfinite cache')
    g = ground.astype(np.float64); g /= np.linalg.norm(g, axis=1, keepdims=True)
    mean = g[:9].mean(0); mean /= np.linalg.norm(mean)
    put('均值与权限绑定.json', {'fit_indices': list(range(9)), 'mean': mean.tolist(),
                             'sha256_float64': hashlib.sha256(mean.tobytes()).hexdigest(),
                             'goal_shape': [512], 'public_fields': ['position','grid_size','step_index','remaining_steps','current_feature'],
                             'cache_bank_never_policy_argument': True})
    put('执行开始.json', {'task': 'CV-024', 'single_run': True, 'model_forward': 0})
    started = time.perf_counter(); rows = []
    with (OUT/'轨迹.jsonl').open('x', encoding='utf-8') as output:
        for t in tasks:
            for arm in ORDER:
                goals = {'Full': g[t['ground_feature_index']], 'Disabled': None, 'Mean': mean,
                         'Wrong': g[t['wrong_ground_feature_index']], 'Frontier': None}
                p = GroundLocalLoop(); p.begin(grid_size=5, budget=10, goal_feature=goals[arm], use_evidence=arm!='Frontier')
                pos, truth = t['start'], t['goal_index_evaluator_only']; visited = {pos}; events = []
                for step in range(10):
                    event = p.observe({'position': pos, 'grid_size': 5, 'step_index': step,
                                       'remaining_steps': 10-step,
                                       'current_feature': aerial[t['aerial_feature_indices'][pos]].copy()})
                    action = event['action']; r,c = divmod(pos,5)
                    dr,dc = ((-1,0),(0,1),(1,0),(0,-1))[action]
                    rr,cc = r+dr,c+dc
                    nxt = rr*5+cc if 0<=rr<5 and 0<=cc<5 else pos
                    event.update(next_position=nxt, moved=nxt!=pos)
                    events.append(event); pos=nxt; visited.add(pos)
                    if pos==truth: break
                p.finish()
                row = {'task_id': t['task_id'], 'arm': arm, 'source': t['source_id_evaluator_only'],
                       'group': t['group_evaluator_only'], 'distance': t['distance_evaluator_only'],
                       'start': t['start'], 'goal_evaluator_only': truth, 'final': pos,
                       'success': pos==truth, 'SG': distance(pos,truth), 'steps': len(events),
                       'moves': sum(e['moved'] for e in events), 'coverage': len(visited)/25,
                       'events': events, 'experimental_cached_closed_loop': True}
                output.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); output.flush(); rows.append(row)
                if time.perf_counter()-started>600: raise TimeoutError('Bounded execution stopped; retain partial rows')
    sums = summary(rows); full = {r['task_id']:r for r in rows if r['arm']=='Full'}
    contrasts = {}
    for arm in ORDER[1:]:
        control = {r['task_id']:r for r in rows if r['arm']==arm}
        recover = sum(full[k]['success'] and not control[k]['success'] for k in full)
        harm = sum(control[k]['success'] and not full[k]['success'] for k in full)
        deltas = [sums['Full']['by_source'][s]['SR']-sums[arm]['by_source'][s]['SR'] for s in SOURCE_ORDER]
        samples = [sum(deltas[i] for i in ids)/5 for ids in itertools.product(range(5),repeat=5)]
        contrasts[arm] = {'recovered': recover, 'harmed': harm, 'net': recover-harm,
                          'source_SR_deltas': deltas, 'source_bootstrap95': np.quantile(samples,[.025,.975]).tolist(),
                          'positive_sources': sum(d>0 for d in deltas),
                          'actions_changed_paired_tasks': sum([e['action'] for e in full[k]['events']] != [e['action'] for e in control[k]['events']] for k in full)}
    gates = {'all100_completed': len(rows)==100,
             'net_at_least2_each': all(c['net']>=2 for c in contrasts.values()),
             'SG_no_worse_each': all(sums['Full']['SG_cells']<=sums[a]['SG_cells'] for a in ORDER[1:]),
             'positive_frontier_sources_at_least3': contrasts['Frontier']['positive_sources']>=3}
    put('结果汇总.json', {'task': 'CV-024', 'arms': sums, 'contrasts': contrasts, 'gates': gates,
                         'mechanism_screen_passed_pending_audit': all(gates.values()),
                         'formal_acceptance': False, 'goal_semantics': 'dataset paired center proxy',
                         'sources': 5, 'unique_tasks': 20, 'episodes': 100, 'seed_repeats': 1,
                         'not_paper_reproduction_or_default_upgrade': True})
    if any(x in sys.modules for x in ('torch','transformers','timm','requests','urllib.request')):
        raise ValueError('Forbidden framework/network module imported')
    verify(b['protected']+b['sources']+b['inputs'])
    put('执行完成.json', {'state':'completed','seconds':time.perf_counter()-started,'episodes':len(rows),
                           'actions':sum(r['steps'] for r in rows), 'encoder_forward':0,'training':0,'API':0,
                           'raw_pixel_reads':0,'downloads':0,'reserved_groups_consumed':0,
                           'protected_verified':len(b['protected']), 'forbidden_modules_imported':False})
    if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())>protocol['resources']['output_bytes']:
        raise ValueError('Output size exceeded')
    print(json.dumps({a: {'success':sums[a]['success'], 'SR':sums[a]['SR'], 'SG':sums[a]['SG_cells']} for a in ORDER}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['freeze','execute'])
    {'freeze':freeze, 'execute':execute}[parser.parse_args().phase]()
