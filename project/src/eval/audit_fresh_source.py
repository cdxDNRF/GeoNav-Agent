"""Independent source/crop/feature, public guard and complete neural replay."""
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from agents.target_cue import cue_features
from agents.edge_cue import edge_features
from eval.audit_edge_cue import manual_profile
from eval.audit_trusted_cue import navigation_metrics
from eval.audit_local_ledger_expansion import independent_effect, compare_effect
from eval.audit_continued_source import same_values, verify_average
from eval.audit_fresh_alternative import independent_guard
from eval.fresh_source_confirmation import (ROOT, SRC, OUT, DATA, OLD, SEEDS, CONTROLS, MEANS,
    read, write, digest, lines, check_bindings, select_sources, validate_cohort, load_banks, make_agent)
from data.scaled_masa import scaled_tasks, wrong_plan
from env.scaled_grid import ScaledEpisode, ScaledGridEnv


def check(value, message):
    if not value:
        raise ValueError('fresh-source audit: ' + message)


def pixels_and_features(reg):
    from transformers import CLIPVisionModelWithProjection
    encoder = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / 'models/Sat2Cap'), local_files_only=True).to('cuda').eval()
    encoder.requires_grad_(False)
    g, l, p = load_banks()
    manifest = read(DATA / '数据清单.json')
    provenance = read(DATA / '图块来源.json')
    check(manifest['source_count'] == 20 and manifest['patch_count'] == 2000, 'derived cohort counts')
    check(manifest['sources'] and len(manifest['sources']) == 20, 'all input sources')
    keys = {'test__' + r['area'] for r in reg['sources']}
    check(set(g) == set(l) == set(p) == keys and len(provenance) == 2000, 'all bank/provenance keys')
    total = 0
    with torch.inference_mode():
        for source in reg['sources']:
            key = 'test__' + source['area']
            inputs = []
            with Image.open(ROOT / source['raw_path']) as raw:
                raw.load()
                check((raw.mode, raw.size) == ('RGB', (1500, 1500)), 'raw dimensions')
                check(sha256(raw.tobytes()).hexdigest() == source['rgb_sha256'], 'raw decoded pixels')
                for cell in range(100):
                    r, c = divmod(cell, 10)
                    crop = raw.crop((c*150, r*150, (c+1)*150, (r+1)*150)).resize((300, 300), Image.Resampling.BICUBIC)
                    buf = BytesIO()
                    crop.save(buf, format='JPEG', quality=75)
                    path = DATA / 'patches/test' / source['area'] / f'patch_{cell}.jpg'
                    check(path.read_bytes() == buf.getvalue(), 'reconstructed JPEG bytes')
                    with Image.open(path) as im:
                        rgb = np.asarray(im.convert('RGB'), np.uint8)
                        pixel = np.asarray(im.resize((224, 224), Image.Resampling.BICUBIC), np.float32) / 255
                    check(provenance[key + '/' + str(cell)] == dict(file_sha256=digest(path), rgb_sha256=sha256(rgb.tobytes()).hexdigest()), 'per-patch provenance')
                    np.testing.assert_array_equal(manual_profile(rgb), p[key][cell])
                    x = ((pixel - np.array([.3670, .3827, .3338], np.float32)) / np.array([.2209, .1975, .1988], np.float32)).transpose(2, 0, 1)
                    inputs.append(torch.from_numpy(x))
            vg, vl = [], []
            for start in range(0, 100, 25):
                z = encoder(torch.stack(inputs[start:start+25]).to('cuda'))
                vg.append(z.image_embeds.cpu().numpy())
                h = encoder.vision_model.post_layernorm(z.last_hidden_state[:, 1:]).reshape(25, 7, 7, 768)
                vl.append(torch.stack((h[:, :3, :3].mean((1, 2)), h[:, :3, 3:].mean((1, 2)),
                                       h[:, 3:, :3].mean((1, 2)), h[:, 3:, 3:].mean((1, 2))), 1).cpu().numpy())
            np.testing.assert_array_equal(np.concatenate(vg), g[key])
            np.testing.assert_array_equal(np.concatenate(vl), l[key])
            total += 100
            print(dict(reconstructed_source=source['area'], patches=100), flush=True)
    del encoder
    torch.cuda.empty_cache()
    return total


def audited_stats(rows):
    return dict(metrics=navigation_metrics(rows),
                by_source={s: navigation_metrics([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
                by_distance={str(d): navigation_metrics([r for r in rows if r['distance'] == d]) for d in range(12, 17)})


def replay(arm, seed, condition, folder, episodes, banks, means, wrong):
    rows = list(lines(folder / f'{arm}_s{seed}_{condition}_轨迹.jsonl'))
    check([r['episode_id'] for r in rows] == [e.episode_id for e in episodes], 'all ordered500 tasks')
    agent = make_agent(seed, means, condition)
    g, l, p = banks
    env = ScaledGridEnv(DATA)
    actions = changed = triggered_records = 0
    for row, ep in zip(rows, episodes):
        check((row['arm'], row['condition'], row['local_checkpoint_seed'], row['source'], row['area'], row['split'], row['distance'], row['grid_size'])
              == (arm, condition, seed, ep.source_tile, ep.area, 'test', ep.dist, 10), 'metadata bindings')
        agent.reset()
        obs = env.reset(ep)
        cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
        target = env.payload(cue)
        key = 'test__' + ep.area
        position = ep.start
        visited = [position]
        trajectory = [dict(step=0, patch_id=position, action=None, out_of_bounds=False, revisited=False)]
        triggered = False
        for i, record in enumerate(row['decisions']):
            check(not env.done and i < 20, 'no post-terminal step')
            view = replace(obs, target_image=target)
            cell = obs.position[0]*10 + obs.position[1]
            base = agent.act_with_profiles(view, g[key][cell], l[key][cell], g[key][cue], l[key][cue], p[key][cell], p[key][cue])
            if arm == 'M0':
                check(record == base, 'exact original checkpoint/GRU replay')
                action = base['action']
            else:
                check(record['base'] == base, 'exact F checkpoint/GRU replay')
                expected = independent_guard(view, base)
                check({k: record[k] for k in expected} == expected, 'independent public new-cell choice')
                action = expected['action']
                changed += int(expected['triggered'])
                triggered |= expected['triggered']
            masked = condition == 'CueMean'
            x = np.concatenate((cue_features(agent.hm if masked else g[key][cue], g[key][cell], agent.lm if masked else l[key][cue], l[key][cell]),
                                edge_features(agent.pm if masked else p[key][cue], p[key][cell]))).astype(np.float32)
            check(sha256(x.tobytes()).hexdigest() == base['cue_features_sha256'], 'independent global/local/edge target masking')
            check(base['current_image_sha256'] == sha256(obs.current_image).hexdigest() and base['target_image_sha256'] == sha256(target).hexdigest(), 'two image payloads')
            dr, dc = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}[action]
            r, c = divmod(position, 10)
            r += dr; c += dc
            check(0 <= r < 10 and 0 <= c < 10, 'manual legal geometry')
            position = r*10 + c
            revisited = position in visited
            visited.append(position)
            trajectory.append(dict(step=i+1, patch_id=position, action=action, out_of_bounds=False, revisited=revisited))
            obs, _, info = env.step(action)
            check(not info.out_of_bounds and obs.position == (r, c) and obs.remaining_budget == 19-i, 'environment transition')
            if position == ep.goal:
                check(i+1 == len(row['decisions']), 'first-arrival termination')
            actions += 1
        check(env.done and row['status'] == 'completed', 'normal terminal')
        check(all(row[k] == v for k, v in env.evaluator_result().items()), 'all environment fields')
        check(row['trajectory'] == trajectory and row['steps'] == len(trajectory)-1, 'manual step trajectory')
        check(row['success'] == (position == ep.goal), 'manual success')
        check(row['sg'] == abs(position//10-ep.goal//10) + abs(position%10-ep.goal%10), 'manual SG')
        check(row['success'] or len(trajectory)-1 == 20, 'budget terminal for failure')
        check(row['revisits'] == sum(t['revisited'] for t in trajectory), 'manual revisit count')
        triggered_records += int(triggered)
    rs = audited_stats(rows)
    path = folder / f'{arm}_s{seed}_{condition}_轨迹.jsonl'
    stored = read(folder / f'{arm}_s{seed}_{condition}_结果.json')
    check(all(same_values(stored[k], v) for k, v in rs.items()), 'independent all/source/distance metrics')
    check(stored['trajectory_sha256'] == digest(path), 'trajectory SHA')
    check((stored['changed_actions'], stored['triggered_records']) == (changed, triggered_records), 'independent intervention counts')
    print(dict(replayed_arm=arm, seed=seed, condition=condition, records=500), flush=True)
    return rs, rows, actions, changed, triggered_records


def main():
    check(not (OUT / '独立复核.json').exists(), 'immutable audit')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reg = read(OUT / '预登记.json')
    check_bindings(reg)
    status = read(OUT / '执行状态.json')
    check(status['completed'], 'evaluation completed')
    check(reg['candidate_gate'] == dict(SR_gain=.02, positive_weights=2, SG_no_worse=True, source95_SR_lower_positive=True), 'frozen primary gate')
    sources, usage = select_sources()
    check(sources == reg['sources'] and usage == reg['source_usage'], 'actual unseen source history')
    frozen = read(OUT / '特征冻结结束.json')
    check(not frozen['model_evaluation_started'], 'feature freeze predates inference')
    for name, h in frozen['files_sha256'].items():
        check(digest(ROOT / name) == h, 'immutable derived input ' + name)
    patches = pixels_and_features(reg)
    episodes = [ScaledEpisode(**r) for r in read(OUT / '导航任务.json')]
    validate_cohort(episodes)
    check([asdict(e) for e in episodes] == [asdict(e) for e in scaled_tasks(sources, 5)], 'original deterministic sampler')
    wrong = read(OUT / '错误目标计划.json')
    check(wrong == wrong_plan(episodes), 'wrong targets and matching exceptions')
    banks = load_banks()
    means = {name: np.load(OLD / name, allow_pickle=False) for name in MEANS}
    results, rows_by, paired = {}, {}, []
    actions = changed = primary_triggers = primary_changes = 0
    for seed in SEEDS:
        for arm in ('M0', 'F'):
            rs, rr, n, c, t = replay(arm, seed, 'CueFull', OUT / '主对照', episodes, banks, means, wrong)
            results[arm, seed] = rs; rows_by[arm, seed] = rr
            actions += n; changed += c
            if arm == 'F': primary_triggers += t; primary_changes += c
        for a, b in zip(rows_by['M0', seed], rows_by['F', seed]):
            paired.append(dict(seed=seed, episode_id=a['episode_id'], source=a['source'], distance=a['distance'],
                               original_success=a['success'], candidate_success=b['success'], original_sg=a['sg'], candidate_sg=b['sg'],
                               recovered=b['success'] and not a['success'], harmed=a['success'] and not b['success']))
    check(paired == read(OUT / '逐题恢复与损伤.json'), 'every recovery and harm')
    left = [results['F', s] for s in SEEDS]
    right = [results['M0', s] for s in SEEDS]
    e = independent_effect(left, right)
    checks = dict(SR_gain2pp=e['sr_gain'] >= .02-1e-12, two_positive_weights=e['positive_seeds'] >= 2,
                  SG_no_worse=e['sg_change'] <= 1e-12, source95_SR_positive=e['source95_SR'][0] > 0)
    passed = all(checks.values())
    summary = read(OUT / '主对照汇总.json')
    compare_effect(summary['effect'], e)
    for arm in ('M0', 'F'):
        verify_average(summary['arms'][arm], [results[arm, s] for s in SEEDS], 10)
    check(summary['checks'] == checks and summary['primary_numeric_passed'] == passed == status['target_controls_started'], 'frozen acceptance decision')
    check(summary['recovery'] == {k: sum(r[k] for r in paired) for k in ('recovered', 'harmed')}, 'recovery totals')
    check(summary['recovery_by_seed'] == {str(s): {k: sum(r[k] for r in paired if r['seed'] == s) for k in ('recovered', 'harmed')} for s in SEEDS}, 'recovery by seed')
    check(summary['triggered_records'] == primary_triggers and summary['changed_actions'] == primary_changes, 'primary intervention totals')
    target_pass = None
    if passed:
        target = read(OUT / '目标证据汇总.json')
        tc = {}
        for condition in CONTROLS:
            controls = []
            for seed in SEEDS:
                rs, _, n, c, _ = replay('F', seed, condition, OUT / '目标对照', episodes, banks, means, wrong)
                controls.append(rs); actions += n; changed += c
            ee = independent_effect(left, controls)
            compare_effect(target['effects'][condition], ee)
            verify_average(target['arms'][condition], controls, 10)
            tc[condition] = ee['sr_gain'] >= .05-1e-12 and ee['positive_seeds'] >= 2 and ee['sg_change'] <= 1e-12
        target_pass = all(tc.values())
        check(target['checks'] == tc and target['target_numeric_passed'] == target_pass, 'target acceptance')
    else:
        check(not (OUT / '目标对照').exists() and not (OUT / '目标证据汇总.json').exists(), 'no controls after failed primary')
    check_bindings(reg)
    audit = dict(passed=True, records=7500 if passed else 3000, actions=actions, changed_actions=changed,
                 reconstructed_patches=patches, independent_global_local_reextraction_exact=True, independent_edge_profiles_exact=True,
                 source_file_and_RGB_disjoint_verified=True, all_checkpoint_and_environment_steps_replayed=True,
                 independent_public_guard_verified=True, manual_geometry_and_target_masks_checked=True,
                 source_intervals_recomputed=True, protected_files_verified=len(reg['protected_sha256']),
                 derived_input_files_verified=len(frozen['files_sha256']), new_source_files=20, new_training_steps=0,
                 cloud_calls=0, default_changed=False)
    write(OUT / '独立复核.json', audit)
    confirmed = passed and target_pass is True
    verdict = dict(completed=True, audit_passed=True, primary_passed=passed, target_controls_started=passed,
                   target_evidence_passed=target_pass, independent_candidate_confirmed=confirmed,
                   failed_primary_checks=[k for k, v in checks.items() if not v], source_files_consumed=True,
                   new_source_files=20, new_training_steps=0, cloud_calls=0, grid5_default_changed=False,
                   grid10_recommendation_upgrade_allowed=confirmed, default_changed=False,
                   summary_sha256=digest(OUT / '主对照汇总.json'), audit_sha256=digest(OUT / '独立复核.json'),
                   next='install separately registered grid10 recommendation with equivalence verification' if confirmed
                        else 'close this factor; retain M0 and prepare actual-area expansion')
    write(OUT / '验收结论.json', verdict)
    print(audit, flush=True)
    print(verdict, flush=True)


if __name__ == '__main__':
    main()
