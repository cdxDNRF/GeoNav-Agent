"""Frozen checkpoint confirmation on complete val100, with preregistered selection."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.spatial_relation import make_policy, quadrant_features
from eval.evaluate import verify_task_file, metrics
from train.curiosity_controlled import SRC, write_new
from train.dyncur_tiny import EmbeddingStore, digest
from train.local_capacity import read, lines, navigation_result, rules, compare, STANDARD
from train.navigation_spatial import ROOT, MASA
from train.target_controlled import wrong_cue_plan

PREVIOUS = MASA / '训练结果/PBRS完整导航与局部匹配对照_v1'
HISTORICAL = MASA / '训练结果/合法动作协作验证_v1'
OUTPUT = MASA / '评测结果/本地S2固定配置复验_v1'
DOC = ROOT / '选题报告相关/本地S2固定配置复验方案_v1.md'
DEFAULT_CONFIG = ROOT / 'project/local_policy_default.json'
ARCHITECTURES = ('Small256', 'Spatial256')
MAIN_ARMS = ('Small256_Full', 'Spatial256_Full', 'Small256_NoTarget', 'Spatial256_NoTarget', 'HistoricalBoundary')
SEEDS = (0, 1, 2)


def specifications():
    plans = []
    for seed in SEEDS:
        for architecture in ARCHITECTURES:
            for trained in ('Full', 'NoTarget'):
                arm = f'{architecture}_{trained}'
                plans.append(dict(arm=arm, architecture=architecture, trained=trained, seed=seed,
                                  folder=f'{arm}_s{seed}', source=PREVIOUS / f'{arm}_s{seed}/model.pt',
                                  conditions=['Full', 'MeanCue', 'WrongCue'] if trained == 'Full' else ['NoTarget']))
        plans.append(dict(arm='HistoricalBoundary', architecture='Small256', trained='Full', seed=seed,
                          folder=f'HistoricalBoundary_s{seed}', source=HISTORICAL / f'Boundary_s{seed}/model.pt', conditions=['Full']))
    return plans


def extract_validation_features(store, out, device):
    from transformers import CLIPVisionModelWithProjection
    from data.process_masa import preprocess_patch
    model = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / 'models/Sat2Cap'), local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    local = {}; maximum = 0.0
    with torch.inference_mode():
        for area in sorted(store.data):
            images = torch.stack([preprocess_patch(MASA / 'patches/val' / area / f'patch_{j}.jpg') for j in range(25)]).to(device)
            result = model(images)
            values = result.image_embeds.float().cpu().numpy()
            maximum = max(maximum, float(np.abs(values - store.data[area]).max()))
            if not np.allclose(values, store.data[area], atol=2e-3, rtol=2e-3):
                raise ValueError('validation global features differ from frozen cache')
            tokens = model.vision_model.post_layernorm(result.last_hidden_state[:, 1:])
            local[area] = quadrant_features(tokens).float().cpu().numpy()
            if not np.isfinite(local[area]).all():
                raise ValueError('non-finite validation local features')
    np.savez(out / '验证局部区域特征.npz', **local)
    write_new(out / '特征核验.json', dict(sources=len(local), patches=25*len(local), frozen=True,
              global_cache_max_abs_error=maximum, local_cache_sha256=digest(out / '验证局部区域特征.npz'),
              encoder_sha256=digest(ROOT / 'models/Sat2Cap/model.safetensors'),
              quadrant_recipe='post_layernorm non-CLS 7x7 tokens; nonoverlapping 3/4 row and column quadrants',
              torch=torch.__version__, transformers=__import__('transformers').__version__))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return local


def variants_for(arm):
    if arm == 'HistoricalBoundary':
        return ('Full',)
    return ('Full', 'MeanCue', 'WrongCue') if arm.endswith('_Full') else ('NoTarget',)


def aggregate(runs):
    if len(runs) != 3:
        raise ValueError('selection requires all three preregistered checkpoint seeds')
    ms = [r['metrics'] for r in runs]
    if len({m['episodes'] for m in ms}) != 1:
        raise ValueError('planned episode denominators differ')
    return dict(sr_mean=float(np.mean([m['sr'] for m in ms])), sr_by_seed=[m['sr'] for m in ms],
                successes_by_seed=[m['successes'] for m in ms], planned_each=ms[0]['episodes'],
                sg_mean=float(np.mean([m['mean_sg_all_episodes'] for m in ms])),
                repeat_by_seed=[m['repeat_visit_rate_micro'] for m in ms],
                out_of_bounds_by_seed=[m['out_of_bounds_rate'] for m in ms],
                source_count=len(runs[0]['by_source']),
                by_distance={str(d): {metric: float(np.mean([r['by_distance'][str(d)][metric] for r in runs]))
                                     for metric in ('sr', 'mean_sg_all_episodes')} for d in range(4, 9)},
                by_source={area: {metric: float(np.mean([r['by_source'][area][metric] for r in runs]))
                                  for metric in ('sr', 'mean_sg_all_episodes')} for area in runs[0]['by_source']})


def engineering_checks(runs, strongest_rule):
    m = aggregate(runs); rule = strongest_rule
    return dict(all_100_tasks_completed=all(r['metrics']['episodes'] == 100 for r in runs),
                mean_SR_60pct=m['sr_mean'] >= .60-1e-12, mean_SG_1p8=m['sg_mean'] <= 1.8+1e-12,
                rule_SR_gain_5pp=m['sr_mean'] >= rule['sr']+.05-1e-12,
                SG_no_worse_than_rule=m['sg_mean'] <= rule['mean_sg_all_episodes']+1e-12,
                each_distance_SR_30pct=all(v['sr'] >= .30-1e-12 for v in m['by_distance'].values()),
                C7_C8_SR_40pct=all(m['by_distance'][str(d)]['sr'] >= .40-1e-12 for d in (7, 8)),
                two_seeds_above_rule=sum(v > rule['sr']+1e-12 for v in m['sr_by_seed']) >= 2)


def choose_default(main, strongest_rule):
    """Hierarchical choice, never select a best seed or mean-masked diagnostic."""
    def contrast(a, b):
        return compare(main[a], main[b], 'sr', 'mean_sg_all_episodes')
    full = 'Spatial256_Full' if contrast('Spatial256_Full', 'Small256_Full')['observational_candidate'] else 'Small256_Full'
    no_target = 'Spatial256_NoTarget' if contrast('Spatial256_NoTarget', 'Small256_NoTarget')['observational_candidate'] else 'Small256_NoTarget'
    target_choice = contrast(no_target, full)
    engineering_candidate = no_target if target_choice['observational_candidate'] else full
    versus_old = contrast(engineering_candidate, 'HistoricalBoundary')
    checks = engineering_checks(main[engineering_candidate], strongest_rule)
    selected = engineering_candidate if versus_old['observational_candidate'] and all(checks.values()) else 'HistoricalBoundary'
    return dict(full_winner=full, no_target_winner=no_target, no_target_vs_full=target_choice,
                engineering_candidate=engineering_candidate, candidate_vs_incumbent=versus_old,
                engineering_candidate_checks=checks, proposed_default=selected,
                changed=selected != 'HistoricalBoundary', reference_seed=0,
                scope='Known development validation only; independent audit required before configuration is written.')


def summarize(results, rule_results):
    def runs(arm, condition=None):
        c = condition or ('NoTarget' if arm.endswith('_NoTarget') else 'Full')
        return [results[arm, seed, c] for seed in SEEDS]
    main = {arm: runs(arm) for arm in MAIN_ARMS}
    means = {arm: {c: aggregate(runs(arm, c)) for c in variants_for(arm)} for arm in MAIN_ARMS}
    pair = lambda a, b: compare(a, b, 'sr', 'mean_sg_all_episodes')
    comparisons = {
        'SpatialFull_minus_SmallFull': pair(main['Spatial256_Full'], main['Small256_Full']),
        'SpatialNoTarget_minus_SmallNoTarget': pair(main['Spatial256_NoTarget'], main['Small256_NoTarget'])}
    for arm in MAIN_ARMS[:-1]:
        comparisons[f'{arm}_minus_HistoricalBoundary'] = pair(main[arm], main['HistoricalBoundary'])
    contrasts = {}; target_passed = {}
    for a in ARCHITECTURES:
        full = main[f'{a}_Full']
        contrasts[a] = {c: pair(full, main[f'{a}_NoTarget'] if c == 'NoTarget' else runs(f'{a}_Full', c))
                        for c in ('NoTarget', 'MeanCue', 'WrongCue')}
        target_passed[a] = {c: v['gain'] >= .05-1e-12 and v['positive_seeds'] >= 2 and v['lower_metric_change'] <= 1e-12
                            for c, v in contrasts[a].items()}
    strongest = max(rule_results, key=lambda n: rule_results[n]['sr'])
    checks = {arm: engineering_checks(main[arm], rule_results[strongest]) for arm in MAIN_ARMS}
    choice = choose_default(main, rule_results[strongest])
    visual_arm = choice['full_winner']; architecture = visual_arm.removesuffix('_Full')
    return dict(averages=means, comparisons=comparisons, target_contrasts=contrasts,
                target_contribution_checks=target_passed, engineering_checks=checks,
                strongest_rule=strongest, rules=rule_results, default_selection=choice,
                visual_S2_arm=visual_arm,
                local_visual_S2_numeric_and_target_checks_passed=all(checks[visual_arm].values()) and all(target_passed[architecture].values()),
                formal_S2_passed=False, audit_pending=True,
                planned_neural_episodes=2700, planned_rule_episodes=200,
                data_scope='Known validation; 4 source maps and 86 unique routes; no independent map confirmation.',
                seed_scope='Three frozen training seed checkpoints, deterministic inference; historical checkpoints are continuation seeds.')


def make_report(out, summary):
    labels = {'Small256_Full': 'PBRS＋筛选', 'Spatial256_Full': '再加局部匹配',
              'Small256_NoTarget': 'Small从头无目标', 'Spatial256_NoTarget': 'Spatial从头无目标',
              'HistoricalBoundary': '历史默认Boundary'}
    rows = ['# 本地S2固定配置复验 v1', '',
        '完整val100、5×5/B10、三个固定训练种子；无新增训练或参数调整。验证集4张源图、86条不同路线，非独立确认。', '',
        '| 方法 | 三种子SR | 平均SR | 平均SG | 三种子重访率 |', '|---|---|---:|---:|---|']
    for arm in MAIN_ARMS:
        condition = 'NoTarget' if arm.endswith('_NoTarget') else 'Full'
        m = summary['averages'][arm][condition]
        rows.append(f"| {labels[arm]} | {' / '.join(f'{s:.0%}' for s in m['sr_by_seed'])} | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} | {' / '.join(f'{s:.2%}' for s in m['repeat_by_seed'])} |")
    effect = summary['comparisons']['SpatialFull_minus_SmallFull']; lo, hi = effect['source_interval']['interval95']
    rows += ['', f"局部匹配相对同批基线：SR{effect['gain']*100:+.2f}点，{effect['positive_seeds']}/3种子为正，SG差{effect['lower_metric_change']:+.3f}，源图95%区间[{lo*100:+.2f},{hi*100:+.2f}]点。", '',
        '| 架构 | 真实目标SR | 从头无目标SR | 均值遮蔽SR | 错误目标SR |', '|---|---:|---:|---:|---:|']
    for a in ARCHITECTURES:
        values = [summary['averages'][f'{a}_Full'][c]['sr_mean'] for c in ('Full', 'MeanCue', 'WrongCue')]
        no = summary['averages'][f'{a}_NoTarget']['NoTarget']['sr_mean']
        rows.append(f'| {a} | {values[0]:.2%} | {no:.2%} | {values[1]:.2%} | {values[2]:.2%} |')
    rows += ['', '规则：' + '；'.join(f"{n} SR={v['sr']:.2%}/SG={v['mean_sg_all_episodes']:.3f}" for n, v in summary['rules'].items()) + '。', '',
             '| C | Small Full | Spatial Full | Small NoTarget | Spatial NoTarget | 历史默认 |', '|---|---:|---:|---:|---:|---:|']
    for d in range(4, 9):
        values = [summary['averages'][a]['NoTarget' if a.endswith('_NoTarget') else 'Full']['by_distance'][str(d)]['sr'] for a in MAIN_ARMS]
        rows.append(f'| {d} | ' + ' | '.join(f'{v:.2%}' for v in values) + ' |')
    choice = summary['default_selection']
    rows += ['', f"预登记规则建议默认：**{labels[choice['proposed_default']]}**；变更={'是' if choice['changed'] else '否'}，独立审计通过后才写入配置。",
             f"本地视觉S2数值＋目标检查：{'通过，待审计' if summary['local_visual_S2_numeric_and_target_checks_passed'] else '未通过'}。", '',
             '无目标仍保留当前图和公开状态。遮蔽只作为消融，不能按其最高分临时替换默认。',
             '源图区间只有4图；重复路线和三种子不能当成独立地图。历史权重的训练数据/热启动不同，与新模型差值不作单因素解释。',
             '默认选择按冻结的分层顺序完成，不选最佳种子，不做集成；参考种子固定0。',
             '无test、大网格或跨数据集实验。最终状态见独立复核及默认选择结论。']
    write_new(out / '对照汇总.json', summary)
    with (out / '复验报告.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(rows) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); out = args.output_dir; device = torch.device(args.device)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=False); began = time.monotonic()
    try:
        episodes, manifest = verify_task_file(MASA, MASA / '任务清单_v2/episodes_val.jsonl', 'val')
        train_eps, train_manifest = verify_task_file(MASA, MASA / '任务清单_v2/episodes_train.jsonl', 'train')
        assert len(episodes) == 100 and Counter(e.dist for e in episodes) == {d: 20 for d in range(4, 9)}
        assert not {e.source_tile for e in episodes} & {e.source_tile for e in train_eps}
        assert not set(manifest['area_sha256'].values()) & set(train_manifest['area_sha256'].values())
        store = EmbeddingStore(MASA / 'papr_val_sat_embeds_grid_5.npy')
        assert set(store.data) == {e.area for e in episodes}
        old_audit = read(PREVIOUS / '独立复核.json'); old_acceptance = read(PREVIOUS / '验收结论.json')
        assert old_audit['status'] == 'passed' and old_acceptance['audit_sha256'] == digest(PREVIOUS / '独立复核.json')
        for name in ('全局拟合均值.npy', '局部拟合均值.npy', '源图划分.json'):
            assert digest(PREVIOUS / name) == old_audit['artifacts_sha256'][name]
            (out / name).write_bytes((PREVIOUS / name).read_bytes())
        mean = np.load(out / '全局拟合均值.npy'); local_mean = np.load(out / '局部拟合均值.npy')
        write_new(out / '导航任务.json', [asdict(e) for e in episodes])
        wrong = wrong_cue_plan(episodes); write_new(out / '错误目标计划.json', wrong)
        plans = specifications(); provenance = []
        for plan in plans:
            source = plan['source']; expected = (read(source.parent / '验证结果.json')['audit']['checkpoint_sha256']
                if plan['arm'] == 'HistoricalBoundary' else old_audit['artifacts_sha256'][source.relative_to(PREVIOUS).as_posix()])
            assert digest(source) == expected
            folder = out / plan['folder']; folder.mkdir()
            (folder / 'model.pt').write_bytes(source.read_bytes())
            config = {k: v for k, v in plan.items() if k != 'source'}
            config.update(source_checkpoint=source.relative_to(ROOT).as_posix(), checkpoint_sha256=expected)
            write_new(folder / '配置.json', config); provenance.append(config)
        sources = {p.relative_to(SRC).as_posix(): p for p in SRC.rglob('*.py')
                   if p.relative_to(SRC).parts[0] in ('agents', 'train', 'eval', 'env', 'data', 'tests')}
        for name, path in sources.items():
            target = out / '源码快照' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        (out / '冻结方案.md').write_bytes(DOC.read_bytes()); (out / '冻结标准.md').write_bytes(STANDARD.read_bytes())
        inputs = ('papr_val_sat_embeds_grid_5.npy', 'papr_train_sat_embeds_grid_5.npy', 'metadata.csv',
                  '任务清单_v2/episodes_val.jsonl', '任务清单_v2/manifest_val.json',
                  '任务清单_v2/episodes_train.jsonl', '任务清单_v2/manifest_train.json')
        reg = dict(version='local-s2-confirmation-v1', utc=datetime.now(timezone.utc).isoformat(), models=15,
            main_arms=list(MAIN_ARMS), seeds=list(SEEDS), checkpoint_provenance=provenance,
            planned_neural_episodes=2700, planned_rule_episodes=200, training_steps=0, cloud_calls=0,
            source_sha256={n: digest(p) for n, p in sources.items()}, data_sha256={n: digest(MASA / n) for n in inputs},
            frozen_inputs_sha256={n: digest(out / n) for n in ('全局拟合均值.npy', '局部拟合均值.npy', '源图划分.json', '导航任务.json', '错误目标计划.json')},
            encoder_sha256=digest(ROOT / 'models/Sat2Cap/model.safetensors'), encoder_config_sha256=digest(ROOT / 'models/Sat2Cap/config.json'),
            previous_audit_sha256=digest(PREVIOUS / '独立复核.json'), previous_acceptance_sha256=digest(PREVIOUS / '验收结论.json'),
            document_sha256=digest(DOC), standard_sha256=digest(STANDARD),
            val_area_sha256=manifest['area_sha256'], train_area_sha256=train_manifest['area_sha256'],
            source_count=4, unique_routes=len({(e.area, e.start, e.goal) for e in episodes}),
            wrong_cue_distance_exceptions=sum(not p['matched_distance'] for p in wrong.values()),
            inference='argmax; reset hidden per episode; three frozen trained seed checkpoints',
            selection_order=['SpatialFull vs SmallFull', 'SpatialNoTarget vs SmallNoTarget', 'NoTarget winner vs Full winner', 'candidate vs incumbent and engineering thresholds'],
            superiority_gate=dict(SR_gain=.02,positive_seeds=2,SG_no_worse=True), reference_seed=0,
            test_used=False, known_validation=True, runtime=dict(device=str(device), torch=torch.__version__, python=sys.version))
        write_new(out / '预登记.json', reg)
        print(json.dumps(dict(registered=str(out), models=15, neural_tasks=2700,
                              wrong_distance_exceptions=reg['wrong_cue_distance_exceptions'])), flush=True)
        local = extract_validation_features(store, out, device)
        write_new(out / '特征冻结结束.json', dict(local_cache_sha256=digest(out / '验证局部区域特征.npz'),
                  feature_check_sha256=digest(out / '特征核验.json'), model_evaluation_started=False))
        results = {}; completed_jobs = 0
        rule_results = rules(out, episodes)
        for plan in plans:
            folder = out / plan['folder']
            weights = torch.load(folder / 'model.pt', map_location=device, weights_only=True)
            model = make_policy(plan['architecture']).to(device); replay = make_policy(plan['architecture']).to(device)
            model.load_state_dict(weights); replay.load_state_dict(weights)
            for variant in plan['conditions']:
                result = navigation_result(folder, variant, model, replay, plan['architecture'], episodes,
                                           store, local, mean, local_mean, wrong, device)
                results[plan['arm'], plan['seed'], variant] = result; completed_jobs += 1
                if plan['arm'] == 'HistoricalBoundary':
                    prior = lines(HISTORICAL / f"Boundary_s{plan['seed']}/val轨迹.jsonl")
                    current = lines(folder / '导航_Full_轨迹.jsonl')
                    assert len(prior) == len(current) == 100
                    assert all(all(row[k] == value for k, value in old.items()) for old, row in zip(prior, current))
                print(json.dumps(dict(job=completed_jobs, planned_jobs=27, run=folder.name, condition=variant,
                                      SR=result['metrics']['sr'], SG=result['metrics']['mean_sg_all_episodes'])), flush=True)
            del model, replay, weights
        for name, expected in reg['source_sha256'].items():
            assert digest(SRC / name) == digest(out / '源码快照' / name) == expected
        for name, expected in reg['data_sha256'].items():
            assert digest(MASA / name) == expected
        for name, expected in reg['frozen_inputs_sha256'].items():
            assert digest(out / name) == expected
        for plan in provenance:
            assert digest(ROOT / plan['source_checkpoint']) == digest(out / plan['folder'] / 'model.pt') == plan['checkpoint_sha256']
        assert digest(DOC) == reg['document_sha256'] and digest(STANDARD) == reg['standard_sha256']
        assert digest(ROOT / 'models/Sat2Cap/model.safetensors') == reg['encoder_sha256']
        assert digest(ROOT / 'models/Sat2Cap/config.json') == reg['encoder_config_sha256']
        _, checked = verify_task_file(MASA, MASA / '任务清单_v2/episodes_val.jsonl', 'val')
        assert checked['area_sha256'] == reg['val_area_sha256']
        summary = summarize(results, rule_results); make_report(out, summary)
        write_new(out / '执行状态.json', dict(status='completed', jobs=completed_jobs,
                  neural_navigation_episodes=2700, rule_episodes=200, audit='pending', training_steps=0,
                  elapsed_seconds=time.monotonic()-began))
        print(json.dumps(dict(completed=True, proposed_default=summary['default_selection']['proposed_default'])), flush=True)
    except Exception as exc:
        write_new(out / '执行异常.json', dict(type=type(exc).__name__, error=str(exc))); raise


if __name__ == '__main__':
    main()
