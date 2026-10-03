"""Read-only, one-signal diagnosis before the radial candidate is frozen."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
PREVIOUS = ROOT / 'DATA/processed_data/MasaRoads/评测结果/空间覆盖与误触发单因素验证_v1'
OUT = ROOT / 'DATA/processed_data/MasaRoads/评测结果/覆盖接管径向保护验证_v1/离线诊断'
MOVES = dict(up=(-1, 0), right=(0, 1), down=(1, 0), left=(0, -1))


def read(p):
    return json.loads(p.read_text('utf-8-sig'))


def digest(p):
    h = sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)


def lines(p):
    with p.open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def distance(a, b):
    return abs(a // 10 - b // 10) + abs(a % 10 - b % 10)


def destination(cell, action):
    dr, dc = MOVES[action]
    r, c = cell // 10 + dr, cell % 10 + dc
    assert 0 <= r < 10 and 0 <= c < 10
    return r * 10 + c


def validate(row, task, seed):
    assert row['episode_id'] == task['episode_id'] and row['local_checkpoint_seed'] == seed
    assert row['condition'] == 'CueFull' and row['status'] == 'completed'
    cells = [x['patch_id'] for x in row['trajectory']]
    assert len(cells) == len(row['decisions']) + 1 == row['steps'] + 1
    assert cells[0] == task['start'] and task['goal'] not in cells[:-1]
    for i, d in enumerate(row['decisions']):
        assert d['public_position'] == list(divmod(cells[i], 10))
        assert d['public_visited'] == cells[:i + 1] and d['remaining_budget'] == 20 - i
        assert destination(cells[i], d['action']) == cells[i + 1]
        assert row['trajectory'][i + 1]['action'] == d['action']
    assert row['success'] == (cells[-1] == task['goal'])
    assert row['success'] or row['steps'] == 20
    assert row['sg'] == distance(cells[-1], task['goal'])
    return cells


def main():
    assert not OUT.exists(), 'exclusive new diagnostic directory'
    delivery = read(PREVIOUS / '阶段交付核验.json')
    assert delivery['completed'] and not delivery['development_candidate_passed']
    for name, expected in delivery['source_sha256'].items():
        # Mutable navigation is a historical delivery snapshot, not a runtime input.
        if name.startswith(('project/src/', 'DATA/', '选题报告相关/', '绘图/数据结果图/', '绘图/绘图数据/')):
            assert digest(ROOT / name) == expected, name
    paths = [OLD / '导航任务.json', OLD / '任务分层.json', PREVIOUS / '验收结论.json',
        PREVIOUS / '主对照/独立复核.json', PREVIOUS / '阶段交付核验.json',
        ROOT / 'project/local_policy_default.json', Path(__file__),
        ROOT / '选题报告相关/覆盖接管径向保护诊断方案_v1.md']
    for seed in range(3):
        paths.extend((OLD / f'独立确认/神经对照/M0_s{seed}_CueFull_轨迹.jsonl',
                      PREVIOUS / f'主对照/Coverage3_s{seed}_CueFull_轨迹.jsonl'))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    OUT.mkdir(parents=True)
    write(OUT / '输入登记.json', dict(date='2026-10-03', source_sha256=hashes,
          source_scope='consumed10regions_posthoc_development', new_navigation=0, cloud_calls=0, training_steps=0))
    write(OUT / '输入封存.json', dict(sha256=digest(OUT / '输入登记.json')))
    tasks = {t['episode_id']: t for t in read(OLD / '导航任务.json') if t['split'] == 'test'}
    strata = read(OLD / '任务分层.json')
    records = []
    actions = 0
    for seed in range(3):
        base = lines(OLD / f'独立确认/神经对照/M0_s{seed}_CueFull_轨迹.jsonl')
        candidate = lines(PREVIOUS / f'主对照/Coverage3_s{seed}_CueFull_轨迹.jsonl')
        assert len(base) == len(candidate) == 750
        for b, c in zip(base, candidate):
            assert b['episode_id'] == c['episode_id']
            t = tasks[b['episode_id']]
            b_cells, c_cells = validate(b, t, seed), validate(c, t, seed)
            actions += b['steps'] + c['steps']
            changes, contractions = [], []
            for i, d in enumerate(c['decisions']):
                if not d['coverage']['changed']:
                    continue
                assert d['cue_action'] is None and d['action'] != d['explorer_action']
                cell = c_cells[i]
                original_dest = destination(cell, d['explorer_action'])
                radial_loss = distance(original_dest, t['start']) - distance(c_cells[i + 1], t['start'])
                item = dict(step=i, radius=distance(cell, t['start']), remaining_budget=20-i,
                    radial_loss=radial_loss, actual_action=d['action'], original_action=d['explorer_action'],
                    # All following goal geometry is evaluation-only.
                    evaluation_only_target_progress=distance(cell, t['goal']) - distance(c_cells[i + 1], t['goal']))
                changes.append(item)
                if radial_loss > 0:
                    contractions.append(item)
            outcome = ('recovered' if c['success'] and not b['success'] else
                       'harmed' if b['success'] and not c['success'] else
                       'both_success' if c['success'] else 'both_failure')
            records.append(dict(episode_id=c['episode_id'], seed=seed, stratum=strata[c['episode_id']]['stratum'],
                outcome=outcome, original_success=b['success'], candidate_success=c['success'],
                original_sg=b['sg'], candidate_sg=c['sg'], original_final_radius=distance(b_cells[-1],t['start']),
                candidate_final_radius=distance(c_cells[-1],t['start']),
                original_max_radius=max(distance(v,t['start']) for v in b_cells),
                candidate_max_radius=max(distance(v,t['start']) for v in c_cells),
                changed_steps=len(changes), radial_contraction_steps=len(contractions),
                first_change=changes[0] if changes else None, first_contraction=contractions[0] if contractions else None,
                contraction_target_progress=dict(Counter(v['evaluation_only_target_progress'] for v in contractions))))
    def aggregate(rows):
        return dict(records=len(rows), outcomes=dict(Counter(r['outcome'] for r in rows)),
            changed_steps=sum(r['changed_steps'] for r in rows), contraction_steps=sum(r['radial_contraction_steps'] for r in rows),
            episodes_with_contraction=sum(r['radial_contraction_steps'] > 0 for r in rows),
            first_change_contraction=sum(r['first_change'] is not None and r['first_change']['radial_loss'] > 0 for r in rows),
            original_mean_max_radius=sum(r['original_max_radius'] for r in rows)/len(rows),
            candidate_mean_max_radius=sum(r['candidate_max_radius'] for r in rows)/len(rows),
            original_mean_final_radius=sum(r['original_final_radius'] for r in rows)/len(rows),
            candidate_mean_final_radius=sum(r['candidate_final_radius'] for r in rows)/len(rows))
    summary = dict(records=4500, paired_episodes=2250, rechecked_actions=actions,
        signal='Coverage3 next radius < original explorer next radius from public start',
        overall=aggregate(records),
        strata={s: aggregate([r for r in records if r['stratum']==s]) for s in ('long_distance','seam_target','interior_target')},
        outcomes={o: aggregate([r for r in records if r['outcome']==o]) for o in ('recovered','harmed','both_success','both_failure')},
        strata_outcomes={s:{o:aggregate(rows) for o in ('recovered','harmed','both_success','both_failure')
            if (rows:=[r for r in records if r['stratum']==s and r['outcome']==o])}
            for s in ('long_distance','seam_target','interior_target')},
        causal_claim=False, new_navigation=0, training_steps=0, cloud_calls=0)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected
    write(OUT / '逐题诊断.json', records)
    write(OUT / '诊断汇总.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
