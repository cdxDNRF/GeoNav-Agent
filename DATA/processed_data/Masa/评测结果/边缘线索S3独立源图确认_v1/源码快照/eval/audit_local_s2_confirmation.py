"""Independent complete val confirmation audit; publish default only after passing."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.spatial_relation import make_policy
from env.episode import manhattan
from eval.evaluate import verify_task_file
from eval.audit_navigation_spatial import check_metrics, metrics, check_comparison, close
from eval.audit_navigation_spatial_repair import check_record
from eval.local_s2_confirmation import (OUTPUT, PREVIOUS, HISTORICAL, DOC, DEFAULT_CONFIG,
    ARCHITECTURES, MAIN_ARMS, MASA, ROOT, SRC, STANDARD, read, lines, variants_for)
from train.curiosity_controlled import write_new
from train.dyncur_tiny import EmbeddingStore, digest
from train.target_controlled import wrong_cue_plan


def verify_engineering(runs, rule, saved):
    average_sr = np.mean([r['metrics']['sr'] for r in runs])
    average_sg = np.mean([r['metrics']['mean_sg_all_episodes'] for r in runs])
    expected = dict(all_100_tasks_completed=all(r['metrics']['episodes'] == 100 for r in runs),
        mean_SR_60pct=average_sr >= .60-1e-12, mean_SG_1p8=average_sg <= 1.8+1e-12,
        rule_SR_gain_5pp=average_sr >= rule['sr'] + .05-1e-12,
        SG_no_worse_than_rule=average_sg <= rule['mean_sg_all_episodes'] + 1e-12,
        each_distance_SR_30pct=all(np.mean([r['by_distance'][str(d)]['sr'] for r in runs]) >= .30-1e-12 for d in range(4, 9)),
        C7_C8_SR_40pct=all(np.mean([r['by_distance'][str(d)]['sr'] for r in runs]) >= .40-1e-12 for d in (7, 8)),
        two_seeds_above_rule=sum(r['metrics']['sr'] > rule['sr']+1e-12 for r in runs) >= 2)
    assert saved == expected


def reproduce_selection(main, rule, saved):
    """Direct arithmetic decision independent of the runner's choose_default()."""
    def better(a, b):
        differences = [x['metrics']['sr']-y['metrics']['sr'] for x, y in zip(main[a], main[b])]
        sg = np.mean([x['metrics']['mean_sg_all_episodes']-y['metrics']['mean_sg_all_episodes'] for x, y in zip(main[a], main[b])])
        return np.mean(differences) >= .02-1e-12 and sum(v > 1e-12 for v in differences) >= 2 and sg <= 1e-12
    full = 'Spatial256_Full' if better('Spatial256_Full', 'Small256_Full') else 'Small256_Full'
    no_target = 'Spatial256_NoTarget' if better('Spatial256_NoTarget', 'Small256_NoTarget') else 'Small256_NoTarget'
    candidate = no_target if better(no_target, full) else full
    assert (saved['full_winner'], saved['no_target_winner'], saved['engineering_candidate']) == (full, no_target, candidate)
    check_comparison(main[no_target], main[full], saved['no_target_vs_full'])
    check_comparison(main[candidate], main['HistoricalBoundary'], saved['candidate_vs_incumbent'])
    verify_engineering(main[candidate], rule, saved['engineering_candidate_checks'])
    selected = candidate if better(candidate, 'HistoricalBoundary') and all(saved['engineering_candidate_checks'].values()) else 'HistoricalBoundary'
    assert saved['proposed_default'] == selected and saved['changed'] == (selected != 'HistoricalBoundary')
    assert saved['reference_seed'] == 0
    return selected


def publish_default(out, reg, summary, audit):
    """Deterministic seed-zero reference, not a new ensemble or best-seed selection."""
    if audit['status'] != 'passed':
        raise ValueError('cannot publish a default without a completed independent audit')
    selected = summary['default_selection']['proposed_default']
    selected_plans = [p for p in reg['checkpoint_provenance'] if p['arm'] == selected]
    assert [p['seed'] for p in selected_plans] == [0, 1, 2]
    plan = selected_plans[0]
    config = dict(version='local-policy-default-v1', selected_arm=selected,
        architecture=plan['architecture'], trained_condition=plan['trained'],
        inference_condition='NoTarget' if plan['trained'] == 'NoTarget' else 'Full',
        reference_seed=0, selected_by_best_seed=False, ensemble=False,
        checkpoints=[dict(seed=p['seed'], path=p['source_checkpoint'], sha256=p['checkpoint_sha256']) for p in selected_plans],
        means={n: dict(path=(out / n).relative_to(ROOT).as_posix(), sha256=digest(out / n))
               for n in ('全局拟合均值.npy', '局部拟合均值.npy')},
        evidence=dict(audit_path=(out / '独立复核.json').relative_to(ROOT).as_posix(),
                      audit_sha256=digest(out / '独立复核.json'),
                      registration_path=(out / '预登记.json').relative_to(ROOT).as_posix(),
                      registration_sha256=digest(out / '预登记.json')),
        protocol=dict(grid_size=5,budget=10,actions=['up','right','down','left'],decode='argmax',boundary_filter=True,memory='episode_isolated'),
        scope='Local development default from known validation; no independent map confirmation.',
        vision_target_contribution_proven=bool(summary['local_visual_S2_numeric_and_target_checks_passed']),
        formal_S2_passed=bool(summary['local_visual_S2_numeric_and_target_checks_passed']))
    if DEFAULT_CONFIG.exists():
        raise ValueError('existing local default must be versioned explicitly before replacement')
    write_new(DEFAULT_CONFIG, config)
    write_new(out / '默认方案配置.json', config)
    conclusion = dict(status='completed_and_audited', default_arm=selected,
        changed=summary['default_selection']['changed'], chosen_by_preregistered_order=True,
        reference_seed=0, no_best_seed_or_ablation_selection=True,
        formal_S2_passed=bool(summary['local_visual_S2_numeric_and_target_checks_passed']),
        independent_map_confirmation_completed=False, audit_sha256=digest(out / '独立复核.json'),
        default_config_path=DEFAULT_CONFIG.relative_to(ROOT).as_posix(), default_config_sha256=digest(DEFAULT_CONFIG),
        selection=summary['default_selection'])
    write_new(out / '默认选择结论.json', conclusion)
    return config, conclusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args(); out = args.output_dir
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    reg = read(out / '预登记.json'); device = torch.device(reg['runtime']['device'])
    assert read(out / '执行状态.json')['status'] == 'completed' and not (out / '执行异常.json').exists()
    assert reg['training_steps'] == reg['cloud_calls'] == 0 and reg['test_used'] is False
    assert reg['planned_neural_episodes'] == 2700 and reg['planned_rule_episodes'] == 200
    for name, value in reg['source_sha256'].items():
        assert digest(SRC / name) == digest(out / '源码快照' / name) == value
    for name, value in reg['data_sha256'].items():
        assert digest(MASA / name) == value
    for name, value in reg['frozen_inputs_sha256'].items():
        assert digest(out / name) == value
    assert digest(DOC) == digest(out / '冻结方案.md') == reg['document_sha256']
    assert digest(STANDARD) == digest(out / '冻结标准.md') == reg['standard_sha256']
    assert digest(ROOT / 'models/Sat2Cap/model.safetensors') == reg['encoder_sha256']
    assert digest(ROOT / 'models/Sat2Cap/config.json') == reg['encoder_config_sha256']
    assert digest(PREVIOUS / '独立复核.json') == reg['previous_audit_sha256']
    assert digest(PREVIOUS / '验收结论.json') == reg['previous_acceptance_sha256']
    episodes, val_manifest = verify_task_file(MASA, MASA / '任务清单_v2/episodes_val.jsonl', 'val')
    train_episodes, train_manifest = verify_task_file(MASA, MASA / '任务清单_v2/episodes_train.jsonl', 'train')
    assert val_manifest['area_sha256'] == reg['val_area_sha256'] and train_manifest['area_sha256'] == reg['train_area_sha256']
    assert not {e.source_tile for e in episodes} & {e.source_tile for e in train_episodes}
    assert not set(reg['val_area_sha256'].values()) & set(reg['train_area_sha256'].values())
    task_rows = read(out / '导航任务.json'); ep_by_id = {e['episode_id']: e for e in task_rows}
    assert len(ep_by_id) == 100 and Counter(e['dist'] for e in task_rows) == {d: 20 for d in range(4, 9)}
    assert len({e['source_tile'] for e in task_rows}) == 4
    assert len({(e['area'],e['start'],e['goal']) for e in task_rows}) == reg['unique_routes'] == 86
    assert all(e['split'] == 'val' and e['budget'] == 10 and e['dist'] == manhattan(e['start'],e['goal']) for e in task_rows)
    from dataclasses import asdict
    assert task_rows == [asdict(e) for e in episodes]
    wrong = read(out / '错误目标计划.json')
    assert wrong == wrong_cue_plan(episodes)
    assert sum(not v['matched_distance'] for v in wrong.values()) == reg['wrong_cue_distance_exceptions']
    for e in task_rows:
        cue = wrong[e['episode_id']]
        assert cue['cue_cell'] not in (e['start'],e['goal'])
        assert cue['matched_distance'] == (manhattan(e['start'],cue['cue_cell']) == e['dist'])
    store = EmbeddingStore(MASA / 'papr_val_sat_embeds_grid_5.npy')
    training_store = EmbeddingStore(MASA / 'papr_train_sat_embeds_grid_5.npy')
    mean = np.load(out / '全局拟合均值.npy'); local_mean = np.load(out / '局部拟合均值.npy')
    split = read(out / '源图划分.json'); fit = split['fit']
    np.testing.assert_array_equal(mean, np.concatenate([training_store.data[a] for a in sorted(fit)]).mean(0))
    with np.load(PREVIOUS / '局部区域特征.npz') as train_local:
        np.testing.assert_array_equal(local_mean, np.concatenate([train_local[a] for a in sorted(fit)]).mean(0))
    for name in ('全局拟合均值.npy','局部拟合均值.npy','源图划分.json'):
        assert digest(out / name) == digest(PREVIOUS / name)
    frozen_features = read(out / '特征冻结结束.json'); checks = read(out / '特征核验.json')
    assert frozen_features['local_cache_sha256'] == checks['local_cache_sha256'] == digest(out / '验证局部区域特征.npz')
    assert frozen_features['feature_check_sha256'] == digest(out / '特征核验.json')
    with np.load(out / '验证局部区域特征.npz') as cache:
        local = {a: cache[a] for a in cache.files}
    assert set(local) == set(store.data) == {e.area for e in episodes}
    assert all(v.shape == (25,4,768) and np.isfinite(v).all() for v in local.values())
    # Independent reconstruction of every validation patch's local feature representation.
    from transformers import CLIPVisionModelWithProjection
    from data.process_masa import preprocess_patch
    encoder = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / 'models/Sat2Cap'), local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    max_feature_error = 0.0
    with torch.inference_mode():
        for area in sorted(local):
            images = torch.stack([preprocess_patch(MASA / 'patches/val' / area / f'patch_{j}.jpg') for j in range(25)]).to(device)
            result = encoder(images)
            np.testing.assert_allclose(result.image_embeds.cpu().numpy(), store.data[area], rtol=2e-3, atol=2e-3)
            grid = encoder.vision_model.post_layernorm(result.last_hidden_state[:,1:]).reshape(25,7,7,768)
            rebuilt = torch.stack((grid[:,:3,:3].mean((1,2)), grid[:,:3,3:].mean((1,2)),
                                   grid[:,3:,:3].mean((1,2)), grid[:,3:,3:].mean((1,2))), dim=1).cpu().numpy()
            max_feature_error = max(max_feature_error,float(np.abs(rebuilt-local[area]).max()))
            np.testing.assert_array_equal(rebuilt,local[area])
    del encoder
    collected = {}; nav_count = nav_steps = 0
    for plan in reg['checkpoint_provenance']:
        folder = out / plan['folder']; config = read(folder / '配置.json')
        assert config == plan and config['conditions'] == list(variants_for(plan['arm']))
        assert digest(folder / 'model.pt') == digest(ROOT / plan['source_checkpoint']) == plan['checkpoint_sha256']
        assert (folder / 'model.pt').stat().st_mtime_ns < (out / '预登记.json').stat().st_mtime_ns
        weights = torch.load(folder / 'model.pt', map_location=device, weights_only=True)
        assert all(torch.isfinite(v).all() for v in weights.values())
        model = make_policy(plan['architecture']).to(device).eval(); model.load_state_dict(weights)
        for condition in plan['conditions']:
            path = folder / f'导航_{condition}_轨迹.jsonl'; rows = lines(path)
            saved = read(folder / f'导航_{condition}_结果.json')
            assert len(rows) == len({r['episode_id'] for r in rows}) == 100
            assert {r['episode_id'] for r in rows} == set(ep_by_id)
            assert digest(path) == saved['audit']['records_sha256']
            assert saved['audit']['checkpoint_sha256'] == plan['checkpoint_sha256']
            assert saved['audit']['checkpoint_replay'] and saved['audit']['environment_replay']
            assert path.stat().st_mtime_ns > (out / '特征冻结结束.json').stat().st_mtime_ns
            for row in rows:
                nav_steps += check_record(row, ep_by_id[row['episode_id']], model, plan['architecture'],condition,
                                          store,local,mean,local_mean,wrong,device)
            check_metrics(metrics(rows),saved['metrics'])
            for a,value in saved['by_source'].items():
                check_metrics(metrics([r for r in rows if r['area']==a]),value)
            for d,value in saved['by_distance'].items():
                check_metrics(metrics([r for r in rows if r['distance']==int(d)]),value)
            if plan['arm']=='HistoricalBoundary':
                old=lines(HISTORICAL / f"Boundary_s{plan['seed']}/val轨迹.jsonl")
                assert all(all(row[k]==value for k,value in before.items()) for before,row in zip(old,rows))
            collected[plan['arm'],plan['seed'],condition]=saved; nav_count+=100
        print(json.dumps(dict(audited=plan['folder'],neural_episodes=nav_count)),flush=True)
    rule_results=read(out / '规则结果.json')
    for name in ('Frontier','FixedRegion'):
        rows=lines(out / f'{name}_轨迹.jsonl')
        assert len(rows)==len({r['episode_id'] for r in rows})==100
        for row in rows:
            check_record(row,ep_by_id[row['episode_id']],rule=name)
        check_metrics(metrics(rows),rule_results[name])
    summary=read(out / '对照汇总.json')
    main={arm:[collected[arm,s,'NoTarget' if arm.endswith('_NoTarget') else 'Full'] for s in range(3)] for arm in MAIN_ARMS}
    for arm in MAIN_ARMS:
        for c in variants_for(arm):
            runs=[collected[arm,s,c] for s in range(3)]; expected=summary['averages'][arm][c]
            assert expected['sr_by_seed']==[r['metrics']['sr'] for r in runs]
            assert expected['successes_by_seed']==[r['metrics']['successes'] for r in runs]
            close(expected['sr_mean'],np.mean([r['metrics']['sr'] for r in runs]))
            close(expected['sg_mean'],np.mean([r['metrics']['mean_sg_all_episodes'] for r in runs]))
            assert expected['repeat_by_seed']==[r['metrics']['repeat_visit_rate_micro'] for r in runs]
            assert expected['out_of_bounds_by_seed']==[r['metrics']['out_of_bounds_rate'] for r in runs]
            assert expected['source_count']==4 and expected['planned_each']==100
            for group,key in (('by_distance','by_distance'),('by_source','by_source')):
                for item,value in expected[group].items():
                    for metric in ('sr','mean_sg_all_episodes'):
                        close(value[metric],np.mean([r[key][item][metric] for r in runs]))
    comparisons=summary['comparisons']
    check_comparison(main['Spatial256_Full'],main['Small256_Full'],comparisons['SpatialFull_minus_SmallFull'])
    check_comparison(main['Spatial256_NoTarget'],main['Small256_NoTarget'],comparisons['SpatialNoTarget_minus_SmallNoTarget'])
    for arm in MAIN_ARMS[:-1]:
        check_comparison(main[arm],main['HistoricalBoundary'],comparisons[f'{arm}_minus_HistoricalBoundary'])
    for a in ARCHITECTURES:
        full=main[f'{a}_Full']
        for c in ('NoTarget','MeanCue','WrongCue'):
            other=main[f'{a}_NoTarget'] if c=='NoTarget' else [collected[f'{a}_Full',s,c] for s in range(3)]
            value=summary['target_contrasts'][a][c]; check_comparison(full,other,value)
            assert summary['target_contribution_checks'][a][c]==bool(value['gain']>=.05-1e-12 and value['positive_seeds']>=2 and value['lower_metric_change']<=1e-12)
    strongest=max(rule_results,key=lambda n:rule_results[n]['sr'])
    assert strongest==summary['strongest_rule']
    assert rule_results==summary['rules']
    for arm in MAIN_ARMS:
        verify_engineering(main[arm],rule_results[strongest],summary['engineering_checks'][arm])
    selected=reproduce_selection(main,rule_results[strongest],summary['default_selection'])
    visual=summary['default_selection']['full_winner']; arch=visual.removesuffix('_Full')
    visual_pass=all(summary['engineering_checks'][visual].values()) and all(summary['target_contribution_checks'][arch].values())
    assert summary['visual_S2_arm']==visual and summary['local_visual_S2_numeric_and_target_checks_passed']==visual_pass
    assert nav_count==2700
    report=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),checkpoints=15,
        neural_episodes=nav_count,neural_actions=nav_steps,rule_episodes=200,val_local_features_reextracted=100,
        local_feature_max_abs_error=max_feature_error,source_bootstrap_contrasts_recomputed=14,
        source_and_checkpoint_hashes_verified=True,fit_only_means_verified=True,
        default_selection_independently_recomputed=True,selected_default=selected,
        formal_S2_passed=bool(visual_pass),independent_confirmation=False,
        artifacts_sha256={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()})
    write_new(out / '独立复核.json',report)
    config,conclusion=publish_default(out,reg,summary,report)
    print(json.dumps({k:v for k,v in report.items() if k!='artifacts_sha256'},ensure_ascii=False,indent=2))
    print(json.dumps(dict(default_config=str(DEFAULT_CONFIG),selected_default=config['selected_arm'],changed=conclusion['changed']),ensure_ascii=False))


if __name__=='__main__':
    main()
