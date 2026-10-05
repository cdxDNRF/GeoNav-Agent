"""EVAL-003新域误触发与覆盖只读诊断（最终版）。

只读取EVAL-002封存轨迹/决策/诊断/探针概率与冻结任务清单；统计失败轨迹的
真实目标邻接机会分类、误触发重定向反事实、距离2窗口、配对恢复/损伤迁移、
raw头负样本方向分解与分层/接缝/区域证据。真值只作事后诊断；不重新编码、
不预测、不导航、不调用模型、不产生新SR。所有输出一次性，不覆盖已存在文件。
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIRM = ROOT / 'DATA/processed_data/MassGIS/评测结果/正式十乘十十五乘十五确认_v1'
FROZEN = ROOT / 'DATA/processed_data/MassGIS/工程准备/正式导航协议冻结_v1'
OUT = ROOT / 'DATA/processed_data/MassGIS/评测结果/新域误触发与覆盖诊断_v1'
AREAS = tuple(f'img_{n}' for n in (7005, 7008, 7009, 7010, 7011, 7012, 7013, 7014, 7015, 7016))
GRIDS = (10, 15)
SEEDS = (0, 1, 2)
POLICIES = ('M0', 'Coverage3Radial')
THRESHOLD = 0.5
DELTAS = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}
DIST_BINS = ((1, 'd1'), (2, 'd2'), (4, 'd3_4'), (8, 'd5_8'), (10 ** 9, 'd9p'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def sha_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def manhattan(a, b, k):
    return abs(a // k - b // k) + abs(a % k - b % k)


def steps_of(record):
    return [event['patch_id'] for event in record['trajectory']]


def dist_bin(d):
    for upper, name in DIST_BINS:
        if d <= upper:
            return name
    raise AssertionError(d)


def verify_inputs(binding):
    """校验EVAL-002阶段封存与全部诊断输入绑定。"""
    problems = []
    stage = read_json(CONFIRM / '核验/阶段封存.json')
    for rel, digest in stage['files_sha256'].items():
        path = ROOT / rel
        if not path.is_file():
            problems.append(f'封存文件缺失: {rel}')
        elif sha_file(path) != digest:
            problems.append(f'封存漂移: {rel}')
        break_flag = bool(problems)
        if break_flag:
            break
    if stage['default_upgraded']:
        problems.append('EVAL-002不应登记默认升级')
    if stage['spare_regions_model_consumed'] != 0:
        problems.append('备用区域不应被消费')
    ledger = read_json(CONFIRM / '元数据/正式池导航完成账本.json')
    if ledger['main_records'] != 10500 or ledger['rule_records'] != 3500 or ledger['probe_predictions'] != 282480:
        problems.append('完成账本记录数与预期不符')
    for rel, digest in binding.items():
        path = ROOT / rel
        if not path.is_file():
            problems.append(f'诊断输入缺失: {rel}')
        elif sha_file(path) != digest:
            problems.append(f'诊断输入漂移: {rel}')
        if problems:
            break
    if problems:
        raise ValueError('; '.join(problems))
    return dict(utc=stamp(), stage_seal_files=len(stage['files_sha256']), checked=len(binding), passed=True)


def plan_denominator():
    """计划分母：grid10=750、grid15=1000（750共同+250远距），逐区75/100题。"""
    plans = {}
    for k in GRIDS:
        tasks = read_json(FROZEN / f'元数据/grid{k}任务.json')
        assert len(tasks) == (750 if k == 10 else 1000), k
        by_area = Counter(t['area'] for t in tasks)
        assert set(by_area) == set(AREAS) and set(by_area.values()) == {75 if k == 10 else 100}
        plans[k] = {t['episode_id']: t for t in tasks}
    common_g10 = {t['pair_id'] for t in plans[10].values() if t['cohort'] == 'common'}
    common_g15 = {t['pair_id'] for t in plans[15].values() if t['cohort'] == 'common'}
    assert len(common_g10) == 750 and common_g10 == common_g15
    return plans


def verify_trajectory_files():
    """每轨迹文件seal与记录一致性；返回(k, policy, seed, area)->records。"""
    jobs = {}
    total = 0
    for k in GRIDS:
        for policy in POLICIES:
            for seed in SEEDS:
                for area in AREAS:
                    path = CONFIRM / '主对照' / f'g{k}_{policy}_s{seed}_CueFull_{area}.jsonl'
                    seal = read_json(path.with_suffix('.seal.json'))
                    data = read_jsonl(path)
                    assert sha_file(path) == seal['sha256'], path.name
                    assert seal['records'] == len(data), path.name
                    assert sum(r['steps'] for r in data) == seal['actions'], path.name
                    binding = seal['binding']
                    assert binding['grid'] == k and binding['policy'] == policy
                    assert binding['seed'] == seed and binding['area'] == area
                    assert binding['condition'] == 'CueFull'
                    for rec in data:
                        assert rec['status'] == 'completed'
                        assert rec['grid_size'] == k and rec['policy'] == policy and rec['seed'] == seed
                        assert rec['area'] == area
                        assert len(rec['decisions']) == rec['steps'] == len(rec['evaluation_diagnostics'])
                    jobs[(k, policy, seed, area)] = data
                    total += len(data)
    assert total == 10500, total
    return jobs


def load_probe(k, seed, area):
    import numpy as np
    path = CONFIRM / '特征' / f'probe_g{k}_s{seed}_CueFull_{area}.npy'
    seal = read_json(path.with_suffix('.seal.json'))
    assert sha_file(path) == seal['sha256']
    arr = np.load(path, allow_pickle=False)
    assert arr.shape[1] == 5 and len(arr) == 1404 if k == 10 else True
    return arr


def opportunity_facts(record, task):
    """单轨迹邻接机会事实（主实现）。

    可行动机会 = 执行动作前位于真实目标邻接格且剩余预算>=1。
    决策记录的 remaining_budget 是执行该步动作前的剩余预算。
    """
    k = task['grid_size']
    goal = task['goal']
    budget = task['budget']
    traj = steps_of(record)
    decisions = record['decisions']
    positions = traj[:-1]
    opp_steps = [i for i, cell in enumerate(positions)
                 if manhattan(cell, goal, k) == 1 and decisions[i]['remaining_budget'] >= 1]
    first = None
    if opp_steps:
        i = opp_steps[0]
        dec = decisions[i]
        cue = dec['cue_action']
        correct = wrong = None
        if cue is not None:
            dr, dc = DELTAS[cue]
            cur = positions[i]
            ny, nx = cur // k + dr, cur % k + dc
            dest = ny * k + nx if 0 <= ny < k and 0 <= nx < k else cur
            correct = dest == goal
            wrong = dest != goal
        first = dict(step_index=i + 1, remaining_budget=dec['remaining_budget'],
                     cue_accepted=cue is not None, accepted_direction=cue,
                     accepted_direction_correct=correct, accepted_direction_wrong=wrong,
                     explorer_action=dec['explorer_action'], executed_action=dec['action'])
    dists = [manhattan(cell, goal, k) for cell in traj]
    dmin = min(dists)
    terminal_adjacent_only = (not record['success'] and not opp_steps and dists[-1] == 1)
    return dict(opportunities=len(opp_steps), had_opportunity=bool(opp_steps),
                first_opportunity=first, min_distance=dmin,
                first_min_step=dists.index(dmin), terminal_adjacent_only=terminal_adjacent_only)


def classify_all(jobs, plans):
    """全部10500条记录的机会事实 + 误触发反事实 + d2窗口。"""
    facts = {}
    redirect = defaultdict(Counter)
    override = Counter()
    d2_window = Counter()
    toward = defaultdict(lambda: [0, 0])
    fired_true_dist = Counter()
    reason_by = defaultdict(Counter)
    cov_stats = Counter()
    cov_gains = Counter()
    for (k, policy, seed, area), records in jobs.items():
        for rec in records:
            task = plans[k][rec['episode_id']]
            goal = task['goal']
            fact = opportunity_facts(rec, task)
            facts[(k, policy, seed, rec['episode_id'])] = fact
            traj = steps_of(rec)
            dists = [manhattan(cell, goal, k) for cell in traj]
            outcome = 'success' if rec['success'] else 'fail'
            for i, dec in enumerate(rec['decisions']):
                d0 = dists[i]
                d1 = dists[i + 1]
                reason_by[(f'g{k}', policy)][dec['reason']] += 1
                if d1 < d0:
                    toward[(f'g{k}', policy, 'cue' if dec['cue_action'] else 'explore')][0] += 1
                toward[(f'g{k}', policy, 'cue' if dec['cue_action'] else 'explore')][1] += 1
                if dec['cue_action'] is not None:
                    fired_true_dist[dist_bin(d0)] += 1
                    if d0 >= 2:
                        dr, dc = DELTAS[dec['action']]
                        y, x = divmod(pos := traj[i], k)
                        actual = (y + dr) * k + (x + dc) if 0 <= y + dr < k and 0 <= x + dc < k else pos
                        dr2, dc2 = DELTAS[dec['explorer_action']]
                        ny2, nx2 = y + dr2, x + dc2
                        exp_d = ny2 * k + nx2 if 0 <= ny2 < k and 0 <= nx2 < k else pos
                        da, de = manhattan(actual, goal, k), manhattan(exp_d, goal, k)
                        for tag, delta in (('cue', da), ('exp', de)):
                            redirect[(f'g{k}', policy, outcome, tag)][
                                'toward' if delta < d0 else ('away' if delta > d0 else 'stay')] += 1
                        if dec['explorer_action'] != dec['action']:
                            override['total'] += 1
                            override['exp_better' if de < da else ('exp_worse' if de > da else 'equal')] += 1
                            better = 'exp_better' if de < da else ('exp_worse' if de > da else 'equal')
                            override[f'g{k}_{outcome}_{better}'] += 1
                if d0 == 2 and outcome == 'fail':
                    d2_window[f'g{k}_d2_accepted' if dec['cue_action'] is not None else f'g{k}_d2_not_accepted'] += 1
                    d2_window[f'g{k}_d2_next_' + ('adjacent' if d1 == 1 else
                               ('stay2' if d1 == 2 else 'farther'))] += 1
                if policy == 'Coverage3Radial':
                    cov = dec.get('coverage') or {}
                    control = cov.get('control')
                    if control is not None:
                        cov_stats[(f'g{k}', control)] += 1
                        if cov.get('changed'):
                            cov_stats[(f'g{k}', control + '_changed')] += 1
                        if cov.get('radial_guard_blocked'):
                            cov_stats[(f'g{k}', 'radial_guard_blocked')] += 1
                        if control == 'coverage_lookahead':
                            cov_gains[(f'g{k}', cov.get('selected_gain'))] += 1
    return dict(facts=facts, redirect={str(k): dict(v) for k, v in redirect.items()},
                override=dict(override), d2_window=dict(d2_window),
                toward={str(k): {'toward': v[0], 'total': v[1]} for k, v in toward.items()},
                fired_true_dist=dict(fired_true_dist), reason_by={str(k): dict(v) for k, v in reason_by.items()},
                cov_stats={str(k): v for k, v in cov_stats.items()},
                cov_gains={str(k): v for k, v in cov_gains.items()})


def failure_min_distance(jobs, plans):
    """失败轨迹的最小真实距离直方图与终局邻接统计。"""
    hist = defaultdict(Counter)
    stratum_close = defaultdict(Counter)
    for (k, policy, seed, area), records in jobs.items():
        for rec in records:
            if rec['success']:
                continue
            task = plans[k][rec['episode_id']]
            traj = steps_of(rec)
            dists = [manhattan(cell, task['goal'], k) for cell in traj]
            dmin = min(dists)
            hist[f'g{k}']['d1_terminal' if dmin == 1 else
                         ('d2' if dmin == 2 else ('d3_4' if dmin <= 4 else 'd5p'))] += 1
            remaining_after_min = task['budget'] - dists.index(dmin)
            stratum_close[(f'g{k}', task['stratum'], policy)][
                'd1_terminal' if dmin == 1 else ('d2' if dmin == 2 else 'd3p')] += 1
            if dmin >= 3 and remaining_after_min >= 5:
                hist[f'g{k}']['never_close_budget5p'] += 1
    return dict(hist={k: dict(v) for k, v in hist.items()},
                stratum={str(key): dict(v) for key, v in stratum_close.items()})


def raw_breakdown():
    """保存探针概率的接受分解：邻接正确/错方向、距离2、更远负样本。"""
    import numpy as np
    result = {}
    for k in GRIDS:
        totals = Counter()
        per_kind = defaultdict(Counter)
        per_area_neg = {}
        for area in AREAS:
            pairs = read_json(FROZEN / f'元数据/{area}_g{k}_探针.json')
            area_neg_acc = 0
            area_neg_n = 0
            for seed in SEEDS:
                probs = load_probe(k, seed, area)
                assert len(probs) == len(pairs)
                for row, pair in zip(probs, pairs):
                    top = int(np.argmax(row))
                    confidence = float(row[top])
                    accepted = top != 4 and confidence >= THRESHOLD
                    kind = pair['kind']
                    label = pair['label']
                    if not accepted:
                        per_kind[kind]['rejected'] += 1
                        continue
                    per_kind[kind]['accepted'] += 1
                    if label == 4:
                        totals[f'{kind}_false_accept'] += 1
                        area_neg_acc += 1
                    elif top != label:
                        totals['adjacent_wrong_direction'] += 1
                    else:
                        totals['adjacent_correct_direction'] += 1
                    if kind == 'adjacent':
                        per_kind['adjacent']['correct' if top == label else 'wrong_direction'] += 1
                area_neg_n += int(np.count_nonzero([p['label'] == 4 for p in pairs]))
            per_area_neg[area] = f'{area_neg_acc}/{area_neg_n}'
        positives = sum(1 for area in AREAS for _seed in SEEDS
                        for p in read_json(FROZEN / f'元数据/{area}_g{k}_探针.json') if p['label'] != 4)
        accepted_total = sum(totals.values())
        correct = totals['adjacent_correct_direction']
        result[k] = dict(totals=dict(totals),
                         per_kind={kind: dict(v) for kind, v in per_kind.items()},
                         negative_false_accept_per_area=per_area_neg,
                         derived=dict(positives_3seeds=positives, accepted_total=accepted_total,
                                      precision=correct / accepted_total if accepted_total else None,
                                      recall=correct / positives if positives else None))
    return result


def paired_migration(jobs, plans, facts):
    """M0与Coverage按(episode,seed)配对的恢复/损伤及距离分解。"""
    migration = defaultdict(Counter)
    harm_decomp = defaultdict(Counter)
    for k in GRIDS:
        for seed in SEEDS:
            for area in AREAS:
                base_by_ep = {r['episode_id']: r for r in jobs[(k, 'M0', seed, area)]}
                for rec in jobs[(k, 'Coverage3Radial', seed, area)]:
                    base = base_by_ep[rec['episode_id']]
                    task = plans[k][rec['episode_id']]
                    key = f'g{k}'
                    if not base['success'] and rec['success']:
                        migration[key]['restored'] += 1
                    elif base['success'] and not rec['success']:
                        migration[key]['harmed'] += 1
                        base_f = facts[(k, 'M0', seed, rec['episode_id'])]
                        cov_f = facts[(k, 'Coverage3Radial', seed, rec['episode_id'])]
                        tag = (f"cov_d{min(cov_f['min_distance'], 9)}_m0_d{min(base_f['min_distance'], 9)}"
                               if not cov_f['terminal_adjacent_only'] else 'cov_d1_terminal')
                        harm_decomp[key][tag] += 1
                    elif base['success'] and rec['success']:
                        migration[key]['both_success'] += 1
                    else:
                        migration[key]['both_fail'] += 1
                    assert task['episode_id'] == base['episode_id']
    return dict(migration={key: dict(v) for key, v in migration.items()},
                harm_min_distance={key: dict(v) for key, v in harm_decomp.items()})


def stratum_seam_region(jobs, plans, facts):
    """分层SR、接缝目标对照、逐区机会饥饿与common队列逐距离SR。"""
    per = defaultdict(lambda: Counter())
    common_sr = defaultdict(lambda: [0, 0])
    for (k, policy, seed, area), records in jobs.items():
        for rec in records:
            task = plans[k][rec['episode_id']]
            fact = facts[(k, policy, seed, rec['episode_id'])]
            outcome = 'success' if rec['success'] else ('fail_with_opp' if fact['had_opportunity'] else 'fail_no_opp')
            for key in ((f'g{k}', policy, 'stratum', task['stratum']),
                        (f'g{k}', policy, 'seam' if task['target_mixed_source'] else 'clean'),
                        (f'g{k}', policy, 'area', area)):
                per[key]['n'] += 1
                per[key]['success'] += rec['success']
            if policy == 'M0' and task['cohort'] == 'common':
                common_sr[(f'g{k}', task['dist'])][0] += rec['success']
                common_sr[(f'g{k}', task['dist'])][1] += 1
            per[(f'g{k}', policy, 'failclass')][outcome] += 1
    return dict(
        stratum={str(key): dict(v) for key, v in per.items() if 'stratum' in key},
        seam={str(key): dict(v) for key, v in per.items() if 'seam' in key or 'clean' in key},
        area={str(key): dict(v) for key, v in per.items() if 'area' in key},
        failclass={str(key): dict(v) for key, v in per.items() if 'failclass' in key},
        common_sr_by_distance={f'{g}_d{d}': dict(success=v[0], planned=v[1], sr=v[0] / v[1])
                               for (g, d), v in sorted(common_sr.items())})


def independent_recount(jobs, plans):
    """另实现计数器：从任务start按动作delta重建全程，不使用trajectory字段。"""
    cross = Counter()
    for (k, policy, seed, area), records in jobs.items():
        for rec in records:
            task = plans[k][rec['episode_id']]
            goal = task['goal']
            pos = task['start']
            remaining = task['budget']
            opp = 0
            acc_correct = 0
            acc_wrong = 0
            min_dist = manhattan(pos, goal, k)
            for dec in rec['decisions']:
                if manhattan(pos, goal, k) == 1 and remaining >= 1:
                    opp += 1
                    if dec['cue_action'] is not None:
                        dr, dc = DELTAS[dec['cue_action']]
                        ny, nx = pos // k + dr, pos % k + dc
                        dest = ny * k + nx if 0 <= ny < k and 0 <= nx < k else pos
                        if dest == goal:
                            acc_correct += 1
                        else:
                            acc_wrong += 1
                dr, dc = DELTAS[dec['action']]
                ny, nx = pos // k + dr, pos % k + dc
                if 0 <= ny < k and 0 <= nx < k:
                    pos = ny * k + nx
                remaining -= 1
                min_dist = min(min_dist, manhattan(pos, goal, k))
            assert manhattan(pos, goal, k) == rec['sg'], rec['episode_id']
            assert (pos == goal) == rec['success'], rec['episode_id']
            cross['episodes'] += 1
            cross['actionable_opportunities'] += opp
            cross['accepted_at_opp_correct'] += acc_correct
            cross['accepted_at_opp_wrong'] += acc_wrong
            cross['fail_terminal_adjacent'] += int(not rec['success'] and min_dist == 1 and opp == 0)
    return dict(cross)


def main_cross_check(jobs, plans, facts, cross):
    """主实现与独立重建计数一致性。"""
    main_opp = sum(f['opportunities'] for f in facts.values())
    main_fail_term = sum(1 for (k, p, s, e), f in facts.items() if f['terminal_adjacent_only'])
    acc_correct = cross['accepted_at_opp_correct']
    return dict(main_opportunities=main_opp, independent_opportunities=cross['actionable_opportunities'],
                opportunities_match=main_opp == cross['actionable_opportunities'],
                main_fail_terminal_adjacent=main_fail_term,
                independent_fail_terminal_adjacent=cross['fail_terminal_adjacent'],
                terminal_match=main_fail_term == cross['fail_terminal_adjacent'],
                accepted_at_opportunity=dict(correct=cross['accepted_at_opp_correct'],
                                             wrong=cross['accepted_at_opp_wrong']))


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'输出已存在，不覆盖: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')
    return path


def main():
    binding = read_json(OUT / '核验/输入绑定.json')
    verification = verify_inputs(binding['inputs'])
    plans = plan_denominator()
    jobs = verify_trajectory_files()
    facts_block = classify_all(jobs, plans)
    facts = facts_block.pop('facts')
    failure_block = failure_min_distance(jobs, plans)
    raw = raw_breakdown()
    paired = paired_migration(jobs, plans, facts)
    layers = stratum_seam_region(jobs, plans, facts)
    cross = independent_recount(jobs, plans)
    check = main_cross_check(jobs, plans, facts, cross)
    assert check['opportunities_match'] and check['terminal_match'], check
    # 机会=成功的一致性断言：任何失败轨迹都不得拥有可行动邻接机会
    fail_with_opp = sum(1 for f in facts.values() if f['had_opportunity'] and f['min_distance'] != 0)
    successes = sum(r['success'] for records in jobs.values() for r in records)
    assert fail_with_opp == 0
    summary = dict(
        protocol='eval003-offline-diagnosis-v1',
        utc=stamp(),
        verification=verification,
        denominators=dict(planned={f'g{k}': len(plans[k]) for k in GRIDS},
                          main_records=10500, successes=successes,
                          actionable_opportunities=cross['actionable_opportunities'],
                          opportunities_equal_successes=successes == cross['actionable_opportunities'],
                          fail_with_actionable_opportunity=fail_with_opp),
        opportunity_facts=dict(
            accepted_at_first_opportunity=Counter(
                f['first_opportunity']['cue_accepted'] for f in facts.values() if f['first_opportunity']),
            first_opp_remaining_by_policy={
                policy: dict(sorted(Counter(f['first_opportunity']['remaining_budget']
                                            for (k, p, s, e), f in facts.items()
                                            if f['first_opportunity'] and p == policy).items()))
                for policy in POLICIES}),
        redirect_and_override=dict(
            redirect=facts_block['redirect'], override=facts_block['override'],
            toward_rates=facts_block['toward'], fired_true_dist=facts_block['fired_true_dist'],
            reason_by=facts_block['reason_by']),
        distance2_window=facts_block['d2_window'],
        coverage_controller=dict(stats=facts_block['cov_stats'], gains=facts_block['cov_gains']),
        failure_min_distance=failure_block,
        raw_breakdown=raw,
        paired_migration=paired,
        layers=layers,
        cross_check={**cross, **check},
    )
    write_json(OUT / '误触发与覆盖诊断汇总.json', summary)
    print(json.dumps({'written': '误触发与覆盖诊断汇总.json',
                      'opportunities_match': check['opportunities_match'],
                      'opportunities_equal_successes': successes == cross['actionable_opportunities']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
