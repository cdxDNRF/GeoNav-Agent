"""Independent feature re-extraction, equation/geometry and whole-pilot replay."""
from collections import Counter, deque
from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from agents.target_cue import cue_features
from agents.edge_cue import edge_features
from eval.audit_edge_cue import manual_profile
from eval.audit_continued_source import same_values
from eval.audit_area_data import independent_tasks
from eval.actual_area_pilot import (ROOT, OUT, DATA, PREVIOUS, OLD, SEEDS, CONDITIONS, RULES, MEANS,
                                    read, write, digest, lines, check_bindings, load_banks, make_agent)
from env.actual_area import AreaEpisode, ActualAreaGridEnv, PROTOCOL

MOVES = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}


def need(value, message):
    if not value:
        raise ValueError('actual-area pilot audit: ' + message)


def near(a, b):
    return abs(a//10-b//10) + abs(a%10-b%10)


def advance(cell, action):
    y, x = divmod(cell, 10)
    dy, dx = MOVES[action]
    if not (0 <= y+dy < 10 and 0 <= x+dx < 10):
        return cell, True
    return 10*(y+dy)+x+dx, False


def independent_rule(name, obs):
    """Independent public-state BFS and original unfiltered region arithmetic."""
    start = obs.position[0]*10 + obs.position[1]
    seen_visits = set(obs.visited)
    if name == 'Frontier':
        pending = deque([(start, ())])
        seen = {start}
        while pending:
            cell, route = pending.popleft()
            for action in MOVES:
                neighbor, blocked = advance(cell, action)
                if blocked:
                    continue
                route_next = route + (action,)
                if neighbor not in seen_visits:
                    return route_next[0]
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append((neighbor, route_next))
        return next(a for a in MOVES if not advance(start, a)[1])
    if name != 'FixedRegion':
        raise ValueError('two frozen rules only')
    visits = Counter(obs.visited)
    region = lambda cell: 3*(cell//10//4) + cell%10//4
    region_target = 0
    for r in range(9):
        if any(region(cell) == r and visits[cell] == 0 for cell in range(100)):
            region_target = r
            break
    destinations = {a: advance(start, a)[0] for a in MOVES}
    fresh = [a for a in MOVES if destinations[a] not in seen_visits]
    scores = [(region(destinations[a]) != region_target, region(destinations[a]),
               visits[destinations[a]], i, a) for i, a in enumerate(MOVES) if not fresh or a in fresh]
    return min(scores)[-1]


def reextract():
    from transformers import CLIPVisionModelWithProjection
    encoder = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / 'models/Sat2Cap'), local_files_only=True).to('cuda').eval()
    encoder.requires_grad_(False)
    g, l, p = load_banks()
    manifest = read(DATA / '数据清单.json')
    provenance = read(DATA / '图块来源.json')
    keys = {'dev__' + r['area'] for r in manifest['regions']}
    need(set(g) == set(l) == set(p) == keys and len(provenance) == 300, 'all feature/provenance keys')
    total = 0
    with torch.inference_mode():
        for region in manifest['regions']:
            key = 'dev__' + region['area']
            need(g[key].shape == (100, 512) and l[key].shape == (100, 4, 768) and p[key].shape == (100, 4, 3, 64, 3), 'bank shapes')
            with Image.open(DATA / 'mosaics' / (region['area'] + '.tiff')) as raw:
                raw.load()
                need(raw.size == (3000, 3000) and raw.mode == 'RGB', 'native mosaic shape')
                need(sha256(raw.tobytes()).hexdigest() == region['native_mosaic_rgb_sha256'], 'mosaic decoded SHA')
                inputs = []
                for cell in range(100):
                    y, x = divmod(cell, 10)
                    crop = raw.crop((300*x, 300*y, 300*(x+1), 300*(y+1)))
                    record = provenance[region['area'] + '/' + str(cell)]
                    need(sha256(crop.tobytes()).hexdigest() == record['native_rgb_sha256'], 'native pixels')
                    buffer = BytesIO()
                    crop.save(buffer, format='JPEG', quality=75)
                    path = DATA / 'patches/dev' / region['area'] / f'patch_{cell}.jpg'
                    need(buffer.getvalue() == path.read_bytes() and digest(path) == record['file_sha256'], 'native JPEG reconstruction')
                    with Image.open(path) as im:
                        rgb = np.asarray(im.convert('RGB'), np.uint8)
                        pixel = np.asarray(im.resize((224, 224), Image.Resampling.BICUBIC), np.float32)/255
                    need(sha256(rgb.tobytes()).hexdigest() == record['jpeg_rgb_sha256'], 'decoded patch SHA')
                    np.testing.assert_array_equal(manual_profile(rgb), p[key][cell])
                    input_ = ((pixel-np.array([.3670, .3827, .3338], np.float32))/np.array([.2209, .1975, .1988], np.float32)).transpose(2, 0, 1)
                    inputs.append(torch.from_numpy(input_))
                vg, vl = [], []
                for i in range(0, 100, 25):
                    z = encoder(torch.stack(inputs[i:i+25]).to('cuda'))
                    vg.append(z.image_embeds.cpu().numpy())
                    h = encoder.vision_model.post_layernorm(z.last_hidden_state[:, 1:]).reshape(25, 7, 7, 768)
                    vl.append(torch.stack((h[:, :3, :3].mean((1, 2)), h[:, :3, 3:].mean((1, 2)),
                                           h[:, 3:, :3].mean((1, 2)), h[:, 3:, 3:].mean((1, 2))), 1).cpu().numpy())
                np.testing.assert_array_equal(np.concatenate(vg), g[key])
                np.testing.assert_array_equal(np.concatenate(vl), l[key])
                total += 100
            print(dict(reextracted_map=region['area'], exact_patches=100), flush=True)
    del encoder
    torch.cuda.empty_cache()
    return total


def independent_diagnostic(cell, goal, cue, action, accepted, remaining, region):
    next_cell, oob = advance(cell, action)
    # Source identities are taken from the independently validated raster order.
    source = lambda c: region['sources'][2*(c//10//5)+c%10//5]['id']
    n = near(cell, goal)
    opportunity = n == 1 and remaining > 0
    return dict(current_source=source(cell), true_target_source=source(goal), given_target_source=source(cue),
                destination_source=source(next_cell), crossed_source_boundary=source(next_cell) != source(cell),
                true_target_distance=n, actionable_true_adjacency=opportunity,
                true_adjacency_cross_source=opportunity and source(cell) != source(goal), cue_accepted=accepted,
                accepted_immediate_true_hit=accepted and not oob and next_cell == goal,
                accepted_immediate_given_hit=accepted and not oob and next_cell == cue, effective_move_m=300*int(not oob))


def independent_diagnosis(rows):
    records = [d for row in rows for d in row['evaluation_diagnostics']]
    adjacency = {}
    for group in ('same_source', 'cross_source'):
        candidates = [d for d in records if d['actionable_true_adjacency'] and
                      d['true_adjacency_cross_source'] == (group == 'cross_source')]
        n = len(candidates)
        count = sum(d['cue_accepted'] for d in candidates)
        hit = sum(d['accepted_immediate_true_hit'] for d in candidates)
        adjacency[group] = dict(opportunities=n, accepted=count, accepted_correct=hit,
                                acceptance_rate=count/n if n else None, correct_acceptance_rate=hit/n if n else None)
    categories = dict.fromkeys(('missed_actionable_adjacency', 'adjacent_only_terminal', 'within2_never_adjacent', 'never_within2'), 0)
    for row in rows:
        if row['success']:
            continue
        opportunity = any(d['actionable_true_adjacency'] for d in row['evaluation_diagnostics'])
        close = any(d['true_target_distance'] <= 2 for d in row['evaluation_diagnostics']) or row['sg'] <= 2
        category = 'missed_actionable_adjacency' if opportunity else ('adjacent_only_terminal' if row['sg'] == 1 else
                        ('within2_never_adjacent' if close else 'never_within2'))
        categories[category] += 1
    accepts = sum(d['cue_accepted'] for d in records)
    hits = sum(d['accepted_immediate_true_hit'] for d in records)
    return dict(action_count=len(records), crossings=sum(d['crossed_source_boundary'] for d in records),
                crossing_records=sum(any(d['crossed_source_boundary'] for d in row['evaluation_diagnostics']) for row in rows),
                cue_accepts=accepts, accepted_true_hits=hits, accepted_given_hits=sum(d['accepted_immediate_given_hit'] for d in records),
                accepted_not_true_hit=accepts-hits, accepted_true_hit_rate=hits/accepts if accepts else None,
                actionable_adjacency_records=sum(any(d['actionable_true_adjacency'] for d in row['evaluation_diagnostics']) for row in rows),
                adjacency=adjacency, failure_categories=categories,
                reasons=dict(Counter(d['reason'] for row in rows for d in row['decisions'])))


def independent_metrics(rows):
    n = len(rows)
    successes = sum(row['success'] for row in rows)
    steps = sum(row['steps'] for row in rows)
    revisits = sum(row['revisits'] for row in rows)
    oob = sum(row['out_of_bounds'] for row in rows)
    return dict(episodes=n, successes=successes, sr=successes/n,
                mean_sg_all_episodes=float(np.mean([row['sg'] for row in rows])), mean_steps=steps/n,
                repeat_visit_rate_micro=revisits/steps, repeat_visit_rate_macro=float(np.mean([row['repeat_visit_rate'] for row in rows])),
                out_of_bounds_rate=oob/steps, mean_sg_m=float(np.mean([row['sg_m'] for row in rows])),
                mean_valid_travel_m=sum(row['valid_travel_m'] for row in rows)/n,
                total_valid_travel_m=sum(row['valid_travel_m'] for row in rows), total_steps=steps, total_revisits=revisits, total_oob=oob,
                terminations=dict(Counter(row['termination'] for row in rows)))


def independent_stats(rows):
    return dict(metrics=independent_metrics(rows),
                by_source={s: independent_metrics([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
                by_distance={str(d): independent_metrics([r for r in rows if r['distance'] == d]) for d in range(12, 17)},
                diagnostics=independent_diagnosis(rows))


@torch.no_grad()
def manual_equations(agent, obs, cell, cue, key, banks, hidden):
    g, l, p = banks
    counts = np.zeros(25, np.float32)
    for v in obs.visited:
        counts[(v//10//2)*5+(v%10//2)] += 1
    counts = np.minimum(counts, 3)/3
    feature = np.concatenate((agent.em, g[key][cell], np.array([obs.position[0]/9, obs.position[1]/9,
                                obs.remaining_budget/20], np.float32), counts)).astype(np.float32)
    logits, _, _, hidden = agent.explorer.step(torch.from_numpy(feature)[None], hidden)
    explorer_action = tuple(MOVES)[int(torch.argmax(logits))]
    mask = agent.condition == 'CueMean'
    vector = np.concatenate((cue_features(agent.hm if mask else g[key][cue], g[key][cell], agent.lm if mask else l[key][cue], l[key][cell]),
                             edge_features(agent.pm if mask else p[key][cue], p[key][cell]))).astype(np.float32)
    probability = agent.head(torch.from_numpy(vector)[None]).softmax(-1)[0].numpy()
    top = int(np.argmax(probability))
    chosen = None
    if agent.condition == 'Baseline':
        reason = 'uncalibrated_abstain'
    elif top == 4:
        reason = 'not_adjacent'
    elif probability[top] < .5:
        reason = 'low_confidence'
    else:
        action = tuple(MOVES)[top]
        target, blocked = advance(cell, action)
        if blocked:
            reason = 'illegal_top_direction'
        elif target in obs.visited:
            reason = 'visited_top_destination'
        else:
            reason = 'accepted'
            chosen = action
    return dict(explorer_action=explorer_action, explorer_logits=logits[0].tolist(), probabilities=probability.tolist(),
                cue_action=chosen, action=chosen or explorer_action, reason=reason,
                explorer_features_sha256=sha256(feature.tobytes()).hexdigest(), cue_features_sha256=sha256(vector.tobytes()).hexdigest()), hidden


def replay(seed, condition, episodes, banks, means, wrong, regions):
    neural = seed is not None
    folder = OUT / ('神经对照' if neural else '规则基线')
    prefix = f'M0_s{seed}_{condition}' if neural else condition
    path = folder / (prefix + '_轨迹.jsonl')
    rows = list(lines(path))
    need([r['episode_id'] for r in rows] == [e.episode_id for e in episodes], 'all75 ordered terminals')
    agent = make_agent(seed, means, condition) if neural else None
    env = ActualAreaGridEnv(DATA)
    g, l, p = banks
    actions = 0
    for row, ep in zip(rows, episodes):
        need((row['area'], row['source'], row['split'], row['distance'], row['condition'], row['status']) ==
              (ep.area, ep.source_tile, 'dev', ep.dist, condition, 'completed'), 'metadata identity')
        if neural:
            need(row['local_checkpoint_seed'] == seed, 'checkpoint identity')
            agent.reset()
        obs = env.reset(ep)
        need(set(vars(obs)) == {'current_image', 'target_image', 'position', 'grid_size', 'remaining_budget', 'visited',
                                'legal_actions', 'media_type'}, 'public permission boundary')
        cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
        target_image = env.payload(cue)
        key = 'dev__' + ep.area
        position = ep.start
        visited = [position]
        trajectory = [dict(step=0, patch_id=position, action=None, out_of_bounds=False, revisited=False)]
        hidden = None
        need(len(row['decisions']) == len(row['evaluation_diagnostics']) == row['steps'], 'one decision/diagnostic per step')
        for index, d in enumerate(row['decisions']):
            need(not env.done and index < 20, 'no post-terminal actions')
            if neural:
                view = replace(obs, target_image=target_image)
                expected = agent.act_with_profiles(view, g[key][position], l[key][position], g[key][cue], l[key][cue], p[key][position], p[key][cue])
                need(d == expected, 'complete checkpoint/GRU replay')
                manual, hidden = manual_equations(agent, view, position, cue, key, banks, hidden)
                need(all(d[k] == value for k, value in manual.items()), 'independent input/masking/selection equations')
                need(d['current_image_sha256'] == sha256(view.current_image).hexdigest() and
                     d['target_image_sha256'] == sha256(target_image).hexdigest(), 'given image payloads')
                accepted = d['reason'] == 'accepted'
            else:
                action = independent_rule(condition, obs)
                expected = dict(action=action, reason='rule', public_position=list(obs.position),
                                public_visited=list(obs.visited), remaining_budget=obs.remaining_budget)
                need(d == expected, 'independent public-only rule choice')
                accepted = False
            diagnostic = independent_diagnostic(position, ep.goal, cue, d['action'], accepted, 20-index, regions[ep.area])
            need(diagnostic == row['evaluation_diagnostics'][index], 'independent source/seam/goal/physical diagnostic')
            position, oob = advance(position, d['action'])
            need(not neural or not oob, 'neural zero out of bounds')
            revisited = position in visited
            visited.append(position)
            trajectory.append(dict(step=index+1, patch_id=position, action=d['action'], out_of_bounds=oob, revisited=revisited))
            obs, done, info = env.step(d['action'])
            need((obs.position, obs.remaining_budget, info.out_of_bounds, info.revisited) ==
                  (divmod(position, 10), 19-index, oob, revisited), 'manual state transition')
            need(done == (position == ep.goal or index == 19), 'first arrival/budget termination')
            if position == ep.goal:
                need(index+1 == len(row['decisions']), 'first-arrival immediate stop')
            actions += 1
        need(env.done and all(row[k] == v for k, v in env.evaluator_result().items()), 'all terminal environment fields')
        need(row['trajectory'] == trajectory and row['steps'] == len(trajectory)-1, 'manual complete trajectory')
        need(row['success'] == (position == ep.goal) and row['sg'] == near(position, ep.goal), 'manual true goal/SG')
        need(row['success'] or row['steps'] == 20, 'all failures exhaust budget')
        need(row['sg_m'] == 300*near(position, ep.goal), 'SG meters')
        need(row['valid_travel_m'] == 300*sum(t['action'] is not None and not t['out_of_bounds'] for t in trajectory), 'physical travel meters')
        need(row['revisits'] == sum(t['revisited'] for t in trajectory) and row['out_of_bounds'] == sum(t['out_of_bounds'] for t in trajectory), 'manual revisit/boundary counts')
    expected_stats = independent_stats(rows)
    stored = read(folder / (prefix + '_结果.json'))
    need(all(same_values(stored[k], v) for k, v in expected_stats.items()), 'all/map/C/diagnostic statistics')
    need(stored['trajectory_sha256'] == digest(path), 'trajectory SHA')
    print(dict(replayed=prefix, records=75, actions=actions), flush=True)
    return expected_stats, rows, actions


def independent_effect(left, right):
    sources = sorted(left[0]['by_source'])
    matrix = []
    effects = {}
    for source in sources:
        sr = sum(a['by_source'][source]['sr']-b['by_source'][source]['sr'] for a, b in zip(left, right))/3
        sg = sum(a['by_source'][source]['mean_sg_all_episodes']-b['by_source'][source]['mean_sg_all_episodes'] for a, b in zip(left, right))/3
        effects[source] = dict(sr=sr, sg=sg)
        matrix.append([sr, sg])
    matrix = np.asarray(matrix)
    samples = np.random.default_rng(5251).integers(3, size=(4000, 3))
    sampled = np.stack([matrix[draw].mean(0) for draw in samples])
    ci = np.quantile(sampled, [.025, .975], axis=0)
    gains = [a['metrics']['sr']-b['metrics']['sr'] for a, b in zip(left, right)]
    return dict(sr_gain=sum(gains)/3, sr_gain_by_seed=gains, positive_seeds=sum(g > 1e-12 for g in gains),
                sg_change=sum(a['metrics']['mean_sg_all_episodes']-b['metrics']['mean_sg_all_episodes'] for a, b in zip(left, right))/3,
                source_effects=effects, source95_SR=ci[:, 0].tolist(), source95_SG=ci[:, 1].tolist(), source_count=3,
                bootstrap_seed=5251, bootstrap_resamples=4000, interpretation='descriptive three known mosaics; not independent geography')


def main():
    need(not (OUT / '独立复核.json').exists(), 'immutable audit output')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reg = read(OUT / '预登记.json')
    check_bindings(reg)
    need(reg['planned_total'] == 1050 and reg['weights'] == [0, 1, 2] and reg['conditions'] == list(CONDITIONS)
         and reg['rules'] == list(RULES) and reg['target_gate'] == dict(SR_gain=.05, positive_weights=2, SG_no_worse=True), 'fixed design')
    status = read(OUT / '执行状态.json')
    need(status['completed'] and status['episodes'] == 1050, 'complete all planned terminals')
    frozen = read(OUT / '特征冻结结束.json')
    need(not frozen['model_evaluation_started'], 'features frozen before navigation')
    for name, h in frozen['files_sha256'].items():
        need(digest(ROOT / name) == h, 'feature SHA')
    patches = reextract()
    manifest = read(DATA / '数据清单.json')
    regions = {r['area']: r for r in manifest['regions']}
    bank = read(OUT / '导航任务.json')
    need(bank == independent_tasks(manifest['regions']), 'original deterministic75task sampler')
    need((OUT / '导航任务.json').read_bytes() == (DATA / '导航任务.json').read_bytes(), 'original task bytes')
    need((OUT / '错误目标计划.json').read_bytes() == (DATA / '错误目标计划.json').read_bytes(), 'original wrong-target bytes')
    episodes = [AreaEpisode(**r) for r in bank]
    for e in episodes:
        e.validate()
    need(len({(e.area, e.start, e.goal) for e in episodes}) == 73, '73 unique routes; two duplicates retained')
    wrong = read(OUT / '错误目标计划.json')
    need(sum(not w['matched_distance'] for w in wrong.values()) == 13, '13 documented distance exceptions')
    for e in episodes:
        w = wrong[e.episode_id]
        need(w['cue_cell'] not in (e.start, e.goal) and 0 <= w['cue_cell'] < 100 and
             w['matched_distance'] == (near(e.start, w['cue_cell']) == e.dist), 'manual wrong target binding')
    banks = load_banks()
    means = {name: np.load(OLD / name, allow_pickle=False) for name in MEANS}
    results, rows_by, rules, rule_rows = {}, {}, {}, {}
    actions = neural_actions = rule_actions = 0
    for seed in SEEDS:
        for condition in CONDITIONS:
            rs, rows, n = replay(seed, condition, episodes, banks, means, wrong, regions)
            results[condition, seed] = rs
            rows_by[condition, seed] = rows
            actions += n
            neural_actions += n
    for name in RULES:
        rs, rows, n = replay(None, name, episodes, banks, means, wrong, regions)
        rules[name], rule_rows[name] = rs, rows
        actions += n
        rule_actions += n
    summary = read(OUT / '对照汇总.json')
    for c in CONDITIONS:
        rows = [r for s in SEEDS for r in rows_by[c, s]]
        expected = dict(**independent_stats(rows), sr_by_seed=[results[c, s]['metrics']['sr'] for s in SEEDS],
                        sg_by_seed=[results[c, s]['metrics']['mean_sg_all_episodes'] for s in SEEDS],
                        successes_by_seed=[results[c, s]['metrics']['successes'] for s in SEEDS], tasks_each=75, weight_count=3)
        need(same_values(summary['arms'][c], expected), 'pooled3weight summaries ' + c)
    for name in RULES:
        expected = dict(**rules[name], trajectory_sha256=digest(OUT / '规则基线' / (name + '_轨迹.jsonl')), weight_count=0, tasks_each=75)
        need(same_values(summary['arms'][name], expected), 'single run rule summary')
    left = [results['CueFull', s] for s in SEEDS]
    checks = {}
    paired = []
    for c in (*CONDITIONS[1:], *RULES):
        right = [results[c, s] for s in SEEDS] if c in CONDITIONS else [rules[c]]*3
        ee = independent_effect(left, right)
        need(same_values(summary['effects'][c], ee), 'whole-mosaic paired interval ' + c)
        if c in CONDITIONS:
            checks[c] = dict(SR_gain5pp=ee['sr_gain'] >= .05-1e-12, two_positive_weights=ee['positive_seeds'] >= 2, SG_no_worse=ee['sg_change'] <= 1e-12)
    for seed in SEEDS:
        for c in (*CONDITIONS[1:], *RULES):
            refs = rows_by[c, seed] if c in CONDITIONS else rule_rows[c]
            for a, b in zip(rows_by['CueFull', seed], refs):
                paired.append(dict(seed=seed, comparison=c, episode_id=a['episode_id'], source=a['source'], distance=a['distance'],
                                   full_success=a['success'], reference_success=b['success'], full_sg=a['sg'], reference_sg=b['sg'],
                                   recovered=a['success'] and not b['success'], harmed=b['success'] and not a['success']))
    need(paired == read(OUT / '逐题配对.json'), 'all paired recoveries/harms')
    recovery = {c: {k: sum(r[k] for r in paired if r['comparison'] == c) for k in ('recovered', 'harmed')} for c in summary['effects']}
    need(recovery == summary['paired_recovery'], 'paired recovery totals')
    target = all(all(x.values()) for x in checks.values())
    need(checks == summary['target_checks'] and target == summary['target_numeric_passed'], 'unchanged target gates')
    check_bindings(reg)
    audit = dict(passed=True, records=1050, neural_records=900, rule_records=150, actions=actions,
                 neural_actions=neural_actions, rule_actions=rule_actions, reconstructed_native_patches=patches,
                 global_local_reextraction_exact=True, independent_edge_profiles_exact=True,
                 all_checkpoint_GRU_environment_steps_replayed=True, independent_policy_equations_checked=True,
                 independent_rules_checked=True, physical_distance_and_seam_geometry_checked=True,
                 all_planned_denominators_and_paired_intervals_checked=True, posthoc_truth_excluded_from_agent=True,
                 protected_files_verified=len(reg['protected_sha256']), code_and_snapshots_verified=len(reg['source_sha256']),
                 data_input_files_verified=len(reg['input_sha256']), feature_files_verified=len(frozen['files_sha256']),
                 new_training_steps=0, cloud_calls=0, default_changed=False)
    write(OUT / '独立复核.json', audit)
    verdict = dict(completed=True, execution_compatibility_passed=True, audit_passed=True, target_evidence_passed=target,
                   known_area_pilot_passed=target, formal_unknown_geography_passed=False, formal_area_navigation_passed=False,
                   known_training_maps_only=True, target_checks=checks, maps=3, tasks_each=75, unique_routes=73,
                   records=1050, source95_intervals_descriptive_only=True, default_changed=False, new_training_steps=0,
                   cloud_calls=0, summary_sha256=digest(OUT / '对照汇总.json'), audit_sha256=digest(OUT / '独立复核.json'),
                   default_sha256=digest(ROOT / 'project/local_policy_default.json'))
    write(OUT / '验收结论.json', verdict)
    print(audit, flush=True)
    print(verdict, flush=True)


if __name__ == '__main__':
    main()
