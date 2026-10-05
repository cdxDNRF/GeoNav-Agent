"""DEV-001开发批次执行器：预登记、τ双臂导航、离线审计与门槛判定。

只使用DATA-007已消费开发区img_6100（grid5/10/15嵌套窗）与EVAL-002封存校准
产物；正式10区与3备用区不参与。输出一次性：已存在文件一律拒绝覆盖。
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import ast
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'project/src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from agents.massgis_threshold_navigator_v1 import ThresholdNativeNavigator
from agents.frozen_edge_navigator import load_frozen_edge_default
from env.massgis_native_area_v2 import NativeAreaEnv, NativeEpisode, native_spec
from env.episode import ACTIONS
from env.scaled_grid import distance
from eval.massgis_threshold_calibration_v1 import (
    DEV7, CONFIRM, OUT, AREAS, GRIDS, DEV_NAV_GRIDS, SEEDS, TAU_GRID, DEV_GATES,
    LIMITS, read_json, read_jsonl, sha_file, rel, stamp, write_json, manhattan)

DEV_PLAN = ROOT / 'DATA/processed_data/MassGIS/评测结果/新域误触发与覆盖诊断_v1/下一批误触发治理冻结方案_v1.md'
META = OUT / '元数据'
QA = OUT / '核验'
RUNS = OUT / '主对照'
ARMS = {'T050': 0.50}  # τ*臂名在校准结果读取后动态补充
CODE = ('agents/massgis_threshold_navigator_v1.py',
        'eval/massgis_threshold_calibration_v1.py',
        'eval/massgis_threshold_dev_run_v1.py',
        'tests/test_massgis_threshold_navigator_v1.py')


def setup_torch():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def mapping(k):
    return [r * 15 + c + (15 - k) for r in range(k) for c in range(k)]


def banks(k):
    path = DEV7 / '特征/native225.npz'
    assert sha_file(path) == read_json(DEV7 / '核验/特征完成.json')['sha256'], 'feature cache drift'
    with np.load(path, allow_pickle=False) as f:
        return tuple(f[name][mapping(k)] for name in ('global_features', 'local_features', 'profiles'))


def dev_root(k):
    return DEV7 / '工程数据' / ('grid' + str(k))


def legal(cell, k):
    r, c = divmod(cell, k)
    return {a: (r + dr) * k + c + dc for a, (dr, dc) in ACTIONS.items()
            if 0 <= r + dr < k and 0 <= c + dc < k}


def grid15_tasks():
    """冻结grid15开发任务采样：结构与DATA-007 grid10一致（75题、8/25接缝）。

    在预登记时一次生成并封存；rng种子为专用冻结值，不在导航阶段重算。
    """
    cells = read_json(DEV7 / '元数据/grid15逐格来源.json')['cells']
    rng = np.random.default_rng(71015)
    tasks = []
    used = set()
    ds = ([(d, 'short') for d in range(4, 9) for _ in range(5)] +
          [(d, 'middle') for d in range(9, 13) for _ in range(7 if d == 9 else 6)] +
          [(d, 'long') for d in range(13, 17) for _ in range(7 if d == 13 else 6)])
    spec = native_spec(15)
    for i, (d, stratum) in enumerate(ds):
        mixed = i % 25 < 8
        pairs = [(a, b) for a in range(225) for b in range(225)
                 if distance(a, b, 15) == d
                 and bool(cells[b]['crosses_source_seam']) == mixed and (a, b) not in used]
        if not pairs:
            raise ValueError('grid15 task infeasible at ' + str(d))
        a, b = pairs[int(rng.integers(len(pairs)))]
        used.add((a, b))
        ep = NativeEpisode(f'native_dev_g15_{i}', 'dev', 'img_6100', a, b, d, spec.budget, 15,
                           spec.name, 'MassGIS2005/DATA005_known_full4')
        ep.validate()
        tasks.append(dict(**asdict(ep), stratum=stratum, target_mixed_source=mixed))
    assert len(tasks) == 75
    return tasks


def source_snapshot():
    pending = list(CODE)
    found = set()

    def push(module):
        for name in (module.replace('.', '/') + '.py', module.replace('.', '/') + '/__init__.py'):
            if (SRC / name).is_file() and name not in found:
                pending.append(name)

    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        tree = ast.parse((SRC / name).read_text('utf-8'))
        package = name[:-3].replace('/', '.').split('.')[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    push(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:max(0, len(package) - (node.level - 1))] if node.level else []
                module = '.'.join(prefix + ([node.module] if node.module else []))
                push(module)
                for alias in node.names:
                    push(module + '.' + alias.name)
    return sorted(found)


def prepare():
    """预登记：校准完成→grid15任务冻结→源码快照→绑定→封存。"""
    if not (OUT / '元数据/校准结果.json').exists():
        raise FileNotFoundError('先运行校准')
    if (QA / '预登记.json').exists():
        raise FileExistsError('预登记已存在，不重复')
    tau = read_json(META / '校准结果.json')['tau_star']
    # grid15任务先落盘再进入输入绑定；已存在时须与冻结采样器逐字节语义一致
    tasks15 = grid15_tasks()
    dest15 = META / 'grid15任务.json'
    if dest15.exists():
        if json.loads(dest15.read_text(encoding='utf-8')) != tasks15:
            raise ValueError('已存在grid15任务与冻结采样器不一致；人工核查')
    else:
        write_json(dest15, tasks15)
    import shutil
    sources = {}
    for name in source_snapshot():
        src = SRC / name
        dest = OUT / '源码快照' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        sources[rel(src)] = sha_file(src)
        sources[rel(dest)] = sha_file(dest)
    inputs = {rel(DEV_PLAN): sha_file(DEV_PLAN)}
    for name in ('校准规范.json', '校准结果.json'):
        inputs[rel(OUT / '元数据' / name)] = sha_file(OUT / '元数据' / name)
    for p in [DEV7 / '元数据/原冻结模型配置.json', DEV7 / '特征/native225.npz',
              DEV7 / '核验/特征完成.json'] + \
             [DEV7 / '元数据' / f'grid{k}{suffix}' for k in (5, 10, 15)
              for suffix in ('逐格来源.json', '视觉探针.json')] + \
             [DEV7 / '元数据' / f'grid{k}任务.json' for k in (5, 10)] + \
             [DEV7 / '工程数据' / f'grid{k}' / '数据清单.json' for k in (5, 10, 15)] + \
             [META / 'grid15任务.json']:
        inputs[rel(p)] = sha_file(p)
    model_cfg = read_json(DEV7 / '元数据/原冻结模型配置.json')
    model_paths = [ROOT / 'project/local_policy_default.json', ROOT / 'project/local_policy_defaults_v1.json',
                   ROOT / 'project/defaults/masa_roads_grid10_radial_v1.json']
    for group in ('checkpoints', 'cue_heads'):
        for rec in model_cfg[group]:
            model_paths.append(ROOT / rec['path'])
    for rec in model_cfg['means'].values():
        model_paths.append(ROOT / rec['path'])
    for p in model_paths:
        inputs[rel(p)] = sha_file(p)
    arms = {'T050': 0.50, f'T{str(tau).replace(".", "")}': tau}
    registration = dict(utc=stamp(), tau_star=tau, arms=arms,
                        grids=list(DEV_NAV_GRIDS), seeds=list(SEEDS), condition='CueFull',
                        grid5_policy='M0_only', grid15_tasks_frozen='元数据/grid15任务.json',
                        dev_gates=DEV_GATES, limits=LIMITS,
                        source_sha256=sources, input_sha256=inputs,
                        dev_geography='img_6100 DATA007 consumed development window only',
                        formal_regions_used=0, spare_regions_used=0, training_steps=0,
                        network_requests=0, default_changed=False,
                        calibration_selection=read_json(META / '校准结果.json')['selection_rule_used'])
    write_json(QA / '预登记.json', registration)
    write_json(QA / '预登记封存.json', dict(sha256=sha_file(QA / '预登记.json')))
    print(json.dumps({'prepared': True, 'tau_star': tau, 'arms': list(arms),
                      'sources': len(sources) // 2, 'inputs': len(inputs)}, ensure_ascii=False))


def check():
    reg = read_json(QA / '预登记.json')
    if sha_file(QA / '预登记.json') != read_json(QA / '预登记封存.json')['sha256']:
        raise ValueError('预登记漂移')
    for field in ('source_sha256', 'input_sha256'):
        for p, h in reg[field].items():
            if sha_file(ROOT / p) != h:
                raise ValueError('冻结绑定漂移: ' + p)
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(reg['utc'])).total_seconds()
    if elapsed > LIMITS['wall_seconds']:
        raise ValueError('登记时间预算耗尽')
    return reg


def freeze_first_action():
    bindings = {rel(p): sha_file(p) for p in
                [DEV7 / '特征/native225.npz', DEV7 / '核验/特征完成.json']
                + [DEV7 / '元数据' / n for n in ('grid5任务.json', 'grid10任务.json',
                                                 'grid5逐格来源.json', 'grid10逐格来源.json', 'grid15逐格来源.json')]
                + [META / 'grid15任务.json']}
    path = QA / '首动作前封存.json'
    if path.exists():
        if read_json(path)['files_sha256'] != bindings:
            raise ValueError('首动作输入漂移')
    else:
        write_json(path, dict(utc=stamp(), files_sha256=bindings,
                              models=read_json(DEV7 / '元数据/原冻结模型配置.json'),
                              navigation_records_started=0))


def episode(t):
    return NativeEpisode(**{name: t[name] for name in NativeEpisode.__dataclass_fields__})


def diag(current, goal, decision, remaining, k, cells):
    dr, dc = ACTIONS[decision['action']]
    y, x = divmod(current, k)
    nxt = (y + dr) * k + (x + dc) if 0 <= y + dr < k and 0 <= x + dc < k else current
    accepted = decision['cue_action'] is not None
    d = distance(current, goal, k)
    return dict(true_distance=d, actionable_adjacency=d == 1, remaining=remaining,
                cue_accepted=accepted, accepted_true_hit=accepted and nxt == goal,
                current_mixed_source=cells[current]['crosses_source_seam'],
                target_mixed_source=cells[goal]['crosses_source_seam'])


def run():
    reg = check()
    setup_torch()
    freeze_first_action()
    tau_star = reg['tau_star']
    arms = reg['arms']
    total = 0
    for k in DEV_NAV_GRIDS:
        tasks = read_json(DEV7 / f'元数据/grid{k}任务.json') if k in (5, 10) else read_json(META / 'grid15任务.json')
        cells = read_json(DEV7 / f'元数据/grid{k}逐格来源.json')['cells']
        wrong = None
        g, l, p = banks(k)
        env = NativeAreaEnv(dev_root(k))
        policies = ('M0',) if k == 5 else ('M0', 'Coverage3Radial')
        for policy in policies:
            for seed in SEEDS:
                for arm_name, tau in arms.items():
                    path = RUNS / f'g{k}_{policy}_s{seed}_{arm_name}.jsonl'
                    seal = path.with_suffix('.seal.json')
                    if seal.exists():
                        if sha_file(path) != read_json(seal)['sha256'] or len(read_jsonl(path)) != len(tasks):
                            raise ValueError('已完成臂发生变化: ' + path.name)
                        continue
                    if path.exists():
                        raise ValueError('未封存的部分输出，停止: ' + path.name)
                    legacy = load_frozen_edge_default(read_json(DEV7 / '元数据/原冻结模型配置.json'), ROOT, seed, 'cpu')
                    model = ThresholdNativeNavigator(legacy, k, policy, 'CueFull', threshold=tau)
                    with path.open('x', encoding='utf-8', newline='\n') as stream:
                        for t in tasks:
                            model.reset()
                            obs = env.reset(episode(t))
                            cue = t['goal']
                            target = env.payload(cue)
                            decisions = []
                            diagnostics = []
                            while not env.done:
                                view = replace(obs, target_image=target)
                                current = view.position[0] * k + view.position[1]
                                result = model.act_with_profiles(view, g[current], l[current], g[cue], l[cue],
                                                                 p[current], p[cue])
                                if result['action'] not in legal(current, k):
                                    raise ValueError('模型执行了非法动作')
                                decisions.append(result)
                                diagnostics.append(diag(current, t['goal'], result, view.remaining_budget, k, cells))
                                obs, _, _ = env.step(result['action'])
                            row = dict(**env.evaluator_result(), grid_size=k, seed=seed, policy=policy,
                                       condition='CueFull', threshold=tau, arm=arm_name, stratum=t['stratum'],
                                       distance=t['dist'], target_mixed_source=t['target_mixed_source'],
                                       decisions=decisions, evaluation_diagnostics=diagnostics, status='completed')
                            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                            stream.flush()
                    write_json(seal, dict(sha256=sha_file(path), planned=len(tasks), completed=len(tasks),
                                          threshold=tau, arm=arm_name))
                    rows = read_jsonl(path)
                    total += len(rows)
                    print(json.dumps({'done': path.name, 'records': len(rows),
                                      'SR': sum(r['success'] for r in rows) / len(rows)}, ensure_ascii=False), flush=True)
    write_json(QA / '导航完成.json', dict(utc=stamp(), neural_records=total, rule_records=0,
                                         geographic_regions=1, grid15_navigation_records_included=True,
                                         formal_regions_used=0, spare_regions_used=0,
                                         files_sha256={rel(f): sha_file(f) for f in RUNS.glob('*')}))
    print(json.dumps({'run_complete': True, 'total_records': total}, ensure_ascii=False))


# ---------------- 离线审计与门槛判定 ----------------

def replay_record(row, task, k):
    """独立重放：从start按动作delta重建轨迹并校验记录指标。"""
    pos = task['start']
    remaining = task['budget']
    visited = [pos]
    revisits = 0
    oob = 0
    travel = 0
    traj = [pos]
    for dec in row['decisions']:
        dr, dc = ACTIONS[dec['action']]
        y, x = divmod(pos, k)
        ny, nx = y + dr, x + dc
        outside = not (0 <= ny < k and 0 <= nx < k)
        if not outside:
            new = ny * k + nx
            travel += 300
            pos = new
        else:
            oob += 1
        if pos in visited:
            revisits += 1
        visited.append(pos)
        remaining -= 1
        traj.append(pos)
    sg = manhattan(pos, task['goal'], k)
    assert (pos == task['goal']) == row['success'], row['episode_id']
    assert row['steps'] == len(row['decisions']), row['episode_id']
    assert remaining == 0 or pos == task['goal'], row['episode_id']
    assert row['sg'] == sg and row['sg_m'] == sg * 300, row['episode_id']
    assert row['revisits'] == revisits and row['out_of_bounds'] == oob, row['episode_id']
    assert row['valid_travel_m'] == travel, row['episode_id']
    return sg, revisits


def audit():
    reg = check()
    tau_star = reg['tau_star']
    result = dict(utc=stamp(), tau_star=tau_star, regression={}, per_arm={}, gates={}, passed=None)
    # 1) 全部记录重放 + 绑定
    for path in sorted(RUNS.glob('g*.jsonl')):
        seal = read_json(path.with_suffix('.seal.json'))
        assert sha_file(path) == seal['sha256'], path.name
        rows = read_jsonl(path)
        assert seal['planned'] == len(rows) == seal['completed'], path.name
        k = rows[0]['grid_size']
        tasks = {t['episode_id']: t for t in
                 (read_json(DEV7 / f'元数据/grid{k}任务.json') if k in (5, 10) else read_json(META / f'grid{k}任务.json')).copy()}
        for row in rows:
            replay_record(row, tasks[row['episode_id']], k)
            assert row['threshold'] == seal['threshold'], path.name
        result['per_arm'][path.stem] = dict(records=len(rows),
                                            SR=sum(r['success'] for r in rows) / len(rows),
                                            SG_m=sum(r['sg_m'] for r in rows) / len(rows))
    # 2) 回归核对：grid5/10 T050臂与DATA-007 CueFull逐记录全等
    mismatches = []
    for k in (5, 10):
        for policy in (('M0',) if k == 5 else ('M0', 'Coverage3Radial')):
            for seed in SEEDS:
                mine = read_jsonl(RUNS / f'g{k}_{policy}_s{seed}_T050.jsonl')
                base = read_jsonl(DEV7 / '主对照' / f'g{k}_{policy}_s{seed}_CueFull.jsonl')
                assert len(mine) == len(base)
                for a, b in zip(mine, base):
                    if a['episode_id'] != b['episode_id'] or a['decisions'] != b['decisions'] \
                            or a['trajectory'] != b['trajectory'] or a['success'] != b['success'] \
                            or a['sg'] != b['sg']:
                        mismatches.append(f'g{k}_{policy}_s{seed}:{a["episode_id"]}')
    result['regression'] = dict(grid5_10_T050_vs_DATA007=dict(matches=not mismatches,
                                                              mismatches=mismatches[:10],
                                                              compared='decisions+trajectory+success+sg per record'))
    # 3) 门槛判定（τ*臂 vs 0.50基线臂，逐臂同题配对）
    gates = {}
    for k in DEV_NAV_GRIDS:
        for policy in (('M0',) if k == 5 else ('M0', 'Coverage3Radial')):
            base_rows = read_jsonl(RUNS / f'g{k}_{policy}_s0_T050.jsonl')  # placeholder replaced below
    # 逐种子配对比较
    gate_items = {}
    for k in DEV_NAV_GRIDS:
        for policy in (('M0',) if k == 5 else ('M0', 'Coverage3Radial')):
            tau_name = f'T{str(tau_star).replace(".", "")}'
            base_all = [r for s in SEEDS for r in read_jsonl(RUNS / f'g{k}_{policy}_s{s}_T050.jsonl')]
            cand_all = [r for s in SEEDS for r in read_jsonl(RUNS / f'g{k}_{policy}_s{s}_{tau_name}.jsonl')]
            assert len(base_all) == len(cand_all)
            by_ep_base = {(r['seed'], r['episode_id']): r for r in base_all}
            by_ep_cand = {(r['seed'], r['episode_id']): r for r in cand_all}
            assert set(by_ep_base) == set(by_ep_cand)
            restored = sum(1 for key in by_ep_base if not by_ep_base[key]['success'] and by_ep_cand[key]['success'])
            harmed = sum(1 for key in by_ep_base if by_ep_base[key]['success'] and not by_ep_cand[key]['success'])
            steps = sum(r['steps'] for r in cand_all)
            wrong_cues = sum(1 for r in cand_all for d, diagr in zip(r['decisions'], r['evaluation_diagnostics'])
                             if diagr['cue_accepted'] and not diagr['accepted_true_hit'])
            strata = {}
            for stratum in ('short', 'middle', 'long'):
                bs = [r for r in base_all if r['stratum'] == stratum]
                cs = [r for r in cand_all if r['stratum'] == stratum]
                if not bs:
                    continue
                strata[stratum] = dict(base_SR=sum(r['success'] for r in bs) / len(bs),
                                       cand_SR=sum(r['success'] for r in cs) / len(cs),
                                       base_SG_m=sum(r['sg_m'] for r in bs) / len(bs),
                                       cand_SG_m=sum(r['sg_m'] for r in cs) / len(cs))
            gate_items[f'g{k}_{policy}'] = dict(
                planned=len(base_all),
                base_SR=sum(r['success'] for r in base_all) / len(base_all),
                cand_SR=sum(r['success'] for r in cand_all) / len(cand_all),
                SR_difference=(sum(r['success'] for r in cand_all) - sum(r['success'] for r in base_all)) / len(base_all),
                base_SG_m=sum(r['sg_m'] for r in base_all) / len(base_all),
                cand_SG_m=sum(r['sg_m'] for r in cand_all) / len(cand_all),
                restored=restored, harmed=harmed,
                wrong_cue_action_rate=wrong_cues / steps if steps else None,
                accepted_precision=_accepted_precision(cand_all),
                strata=strata)
    for name, item in gate_items.items():
        gates[name] = dict(
            wrong_cue=item['wrong_cue_action_rate'] <= DEV_GATES['wrong_cue_action_rate_at_most'],
            SR_not_lower=item['SR_difference'] >= 0,
            stratum_SG_no_worse=all(v['cand_SG_m'] <= v['base_SG_m'] + 1e-9 for v in item['strata'].values()),
            probe_recall=True)  # 校准结果已在gate4单独记录
    result['gates'] = dict(items=gate_items, verdicts=gates,
                           gate4_probe_adjacent_recall=read_json(META / '校准结果.json')['gate4_probe_adjacent_recall_satisfied'],
                           all_passed=all(all(v.values()) for v in gates.values())
                           and result['gates']['gate4_probe_adjacent_recall'])
    result['passed'] = result['gates']['all_passed']
    write_json(QA / '门槛判定.json', result)
    print(json.dumps({'passed': result['passed'],
                      'summary': {name: {'SR_delta': round(item['SR_difference'], 4),
                                         'wrong_cue': item['wrong_cue_action_rate']}
                                  for name, item in gate_items.items()}}, ensure_ascii=False, indent=1))
    return result


def _accepted_precision(rows):
    accepted = 0
    correct = 0
    for r in rows:
        for d, diagr in zip(r['decisions'], r['evaluation_diagnostics']):
            if diagr['cue_accepted']:
                accepted += 1
                correct += int(diagr['accepted_true_hit'])
    return correct / accepted if accepted else None


def report():
    verdict = read_json(QA / '门槛判定.json')
    reg = read_json(QA / '预登记.json')
    calib = read_json(META / '校准结果.json')
    lines = ['# DEV-001 误触发治理（cue接受门槛校准）开发验证报告', '',
             f'执行：ZCode / zcode-dev001-20261005；唯一因素=cue接受置信阈值 0.50→{reg["tau_star"]}；模型权重、特征、探索器、覆盖控制器、预算与协议全部不变。', '',
             '## 校准（只用EVAL-002封存探针概率）', '',
             f'- 冻结τ网格 {list(TAU_GRID)}；选择规则：{calib["selection_rule_used"]}。',
             f'- **τ\\* = {reg["tau_star"]}**；探针邻接召回 g10={calib["probe_adjacent_recall_at_tau_star"]["g10"]:.4f}、g15={calib["probe_adjacent_recall_at_tau_star"]["g15"]:.4f}（门槛④≥0.90：{"满足" if calib["gate4_probe_adjacent_recall_satisfied"] else "未满足"}）。', '',
             '## 同题开发导航（img_6100，已消费开发区，正式10区/3备用未参与）', '',
             '| 网格/策略 | 0.50基线SR | τ*臂SR | SR差 | 0.50 SG(米) | τ* SG(米) | 恢复 | 损伤 | 错误cue/动作 | 接受精度 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, item in verdict['gates']['items'].items():
        lines.append(f'| {name} | {item["base_SR"]:.4f} | {item["cand_SR"]:.4f} | {item["SR_difference"]:+.4f} '
                     f'| {item["base_SG_m"]:.2f} | {item["cand_SG_m"]:.2f} | {item["restored"]} | {item["harmed"]} '
                     f'| {item["wrong_cue_action_rate"]:.4f} | {item["accepted_precision"] if item["accepted_precision"] is not None else "—"} |')
    lines += ['', '## 回归核对', '',
              f'grid5/10 τ=0.50臂与DATA-007封存CueFull轨迹逐记录全等（decisions+trajectory+success+sg）：{"通过" if verdict["regression"]["grid5_10_T050_vs_DATA007"]["matches"] else "未通过"}。',
              'grid15无历史基线，本批τ=0.50臂即其基线（新证据，仅开发口径）。', '',
              '## 门槛判定（全部必要）', '']
    for name, v in verdict['gates']['verdicts'].items():
        lines.append(f'- **{name}**：错误cue/动作≤5% {"通过" if v["wrong_cue"] else "未过"}；SR不降 {"通过" if v["SR_not_lower"] else "未过"}；分层SG不变差 {"通过" if v["stratum_SG_no_worse"] else "未过"}。')
    lines.append(f'- **门槛④探针邻接召回≥0.90**：{"通过" if verdict["gates"]["gate4_probe_adjacent_recall"] else "未过"}。')
    lines += ['', f'## 总判定：{"通过，可进入确认准备" if verdict["passed"] else "未通过——按冻结方案停止条件收束该因素，不调门槛、不改探索器或训练"}', '',
              '## 边界', '',
              '- 本批为开发验证（已消费地理img_6100，单一地理样本），不输出地区置信区间、不冒充独立确认、不升级默认。',
              '- 离线重放、回归核对与门槛判定由同一开发者实现，不称独立人员审查。',
              '- 全部输入/源码绑定见 预登记.json；校准规范先于曲线落盘（frozen_before_curve=true）。']
    path = OUT / '误触发治理开发验证报告.md'
    if path.exists():
        raise FileExistsError(str(path))
    path.write_text('\n'.join(lines), encoding='utf-8')
    print('report written')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'audit', 'report'))
    args = parser.parse_args()
    globals()[args.action]()
