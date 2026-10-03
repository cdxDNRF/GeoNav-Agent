"""Evidence-gated closeout for the frozen-M0 spatial-area confirmation.

This script only reads the frozen evaluation and writes new report/figure
artifacts after both stages, their independent audits, and the final verdict
are present and hash-consistent. All created outputs use exclusive-create.
"""
from datetime import date
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import math
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics


ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT / 'DATA/processed_data/MasaRoads/空间隔离连续区域_v2'
DATA = PREP / '工程数据'
OUT = ROOT / 'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
ENGINEERING = OUT / '工程接口'
CONFIRMATION = OUT / '独立确认'
AUDIT_REVISION = OUT / '审计修订登记_v2.json'
AUDIT_REVISION_SEAL = OUT / '审计修订封存_v2.json'
ORIGINAL_ENGINEERING_FAILURE = ENGINEERING / '原独立复核失败_v1.json'
AUDIT_REVISION_SOURCE = ROOT / 'project/src/eval/audit_spatial_area_confirmation_v2.py'
ORIGINAL_AUDITOR_SOURCE = ROOT / 'project/src/eval/audit_spatial_area_confirmation.py'
AUDIT_REVISION_V3 = OUT / '审计修订登记_v3.json'
AUDIT_REVISION_SEAL_V3 = OUT / '审计修订封存_v3.json'
ORIGINAL_ENGINEERING_FAILURE_V2 = ENGINEERING / '原独立复核失败_v2.json'
SUMMARY_DIAGNOSTIC_V2 = ENGINEERING / '汇总差异诊断_v2.json'
AUDIT_REVISION_SOURCE_V3 = ROOT / 'project/src/eval/audit_spatial_area_confirmation_v3.py'
PLOTS = ROOT / '绘图/数据结果图'
PLOT_DATA = ROOT / '绘图/绘图数据/冻结M0空间隔离导航确认_v1.json'
CAPTIONS = ROOT / '绘图/冻结M0空间隔离导航确认图注.md'
REPORT = OUT / '新空间隔离区域M0导航确认报告.md'
README = OUT / 'README.md'

SEEDS = (0, 1, 2)
CONDITIONS = ('CueFull', 'Baseline', 'CueMean', 'CueWrong')
RULES = ('Frontier', 'FixedRegion')
ARMS = (*CONDITIONS, *RULES)
STRATA = ('long_distance', 'seam_target', 'interior_target')
STRATUM_LABELS = {
    'long_distance': '长距离锚点 C12–C16',
    'seam_target': '接缝目标 C8–C12',
    'interior_target': '内部目标 C8–C12',
}
ARM_LABELS = {
    'CueFull': '完整目标图像',
    'Baseline': 'Baseline（禁用cue）',
    'CueMean': '均值目标图像',
    'CueWrong': '错误目标图像',
    'Frontier': 'Frontier规则',
    'FixedRegion': 'FixedRegion规则',
}
ARM_COLORS = {
    'CueFull': '#0072B2',
    'Baseline': '#7C8792',
    'CueMean': '#009E73',
    'CueWrong': '#D55E00',
    'Frontier': '#CC79A7',
    'FixedRegion': '#E69F00',
}
SEED_MARKERS = {0: 'o', 1: '^', 2: 's'}
BOOTSTRAP_SEED = 7317
PROBE_KINDS = ('cross_source_adjacent', 'same_source_adjacent', 'nonadjacent_matched_target')
PROBE_LABELS = {
    'cross_source_adjacent': '跨源邻接',
    'same_source_adjacent': '同源邻接',
    'nonadjacent_matched_target': '两格非邻接',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def write_exclusive(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode('utf-8')
    with path.open('xb') as handle:
        handle.write(content)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def pct(value, digits=2):
    return '—' if value is None else f'{100 * value:.{digits}f}%'


def signed_pct(value, digits=2):
    return '—' if value is None else f'{100 * value:+.{digits}f}个百分点'


def num(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def fmt_bool(value):
    return '通过' if value else '未通过'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def stage_checks(folder, expected_records, expected_probes, revision_v2_receipt_sha256,
                 revision_v3_receipt_sha256, original_auditor_sha256):
    audit = read(folder / '独立复核.json')
    state = read(folder / '执行状态.json')
    summary_path = folder / '对照汇总.json'
    probe_path = folder / '探针对照/探针汇总.json'
    need(audit['passed'], f'{folder.name}: independent audit did not pass')
    need(audit.get('audit_revision') == 3
         and audit.get('audit_only_rule_reset_correction') is True
         and audit.get('audit_only_timing_metadata_correction') is True
         and audit.get('audit_revision_receipt_sha256') == revision_v3_receipt_sha256
         and audit.get('reset_revision_receipt_sha256') == revision_v2_receipt_sha256
         and audit.get('original_frozen_auditor_sha256') == original_auditor_sha256,
         f'{folder.name}: stage audit revision stamp is missing or mismatched')
    need(state['completed'], f'{folder.name}: stage is not complete')
    need(audit['records'] == expected_records, f'{folder.name}: unexpected audited record count')
    need(audit['probe_records'] == expected_probes, f'{folder.name}: unexpected audited probe count')
    need(state['episodes'] == expected_records and state['probe_records'] == expected_probes,
         f'{folder.name}: execution state record totals changed')
    need(audit['summary_sha256'] == digest(summary_path), f'{folder.name}: summary/audit hash mismatch')
    need(audit['probe_summary_sha256'] == digest(probe_path), f'{folder.name}: probe/audit hash mismatch')
    return audit, state, read(summary_path), read(probe_path)


def check_frozen_bindings(reg):
    for field in ('protected_sha256', 'source_sha256', 'input_sha256',
                  'encoder_sha256', 'frozen_output_sha256'):
        for name, expected in reg[field].items():
            path = ROOT / name
            need(path.is_file() and digest(path) == expected,
                 f'frozen input changed or missing: {name}')


def verify_audit_revision():
    """Verify the audit-only repair receipt and every hash it froze."""
    need(all(path.is_file() for path in (AUDIT_REVISION, AUDIT_REVISION_SEAL,
                                         ORIGINAL_ENGINEERING_FAILURE, AUDIT_REVISION_SOURCE)),
         'versioned audit revision receipt, seal, source, and original failure record are required')
    receipt = read(AUDIT_REVISION)
    seal = read(AUDIT_REVISION_SEAL)
    failure = read(ORIGINAL_ENGINEERING_FAILURE)
    receipt_hash = digest(AUDIT_REVISION)
    need(seal == {'receipt_sha256': receipt_hash}, 'audit revision seal mismatch')
    need(receipt['revision'] == 2 and receipt['audit_only'] is True
         and receipt['neural_policy_or_rule_actions_changed'] is False
         and receipt['model_threshold_or_gate_changed'] is False
         and receipt['navigation_records_reused'] == 4200
         and receipt['probe_records_reused'] == 4320
         and receipt['confirmation_not_started'] is True,
         'audit revision scope or reused record totals changed')
    need(failure['passed'] is False and 'NoneType' in failure['exception']
         and "reset" in failure['exception']
         and failure['navigation_or_probe_records_changed'] is False
         and failure['confirmation_started'] is False
         and failure['original_auditor_preserved'] is True
         and failure['repair_tests_passed'] is True
         and failure['repair_tests_run'] == 3,
         'preserved v1 engineering audit failure record changed')
    required_sources = {
        'project/src/eval/audit_spatial_area_confirmation.py',
        'project/src/eval/audit_spatial_area_confirmation_v2.py',
        'project/src/tests/test_spatial_area_confirmation_audit_v2.py',
    }
    need(required_sources.issubset(receipt['source_sha256']),
         'audit revision receipt does not bind the v1/v2 auditor and repair tests')
    need(receipt['original_engineering_sha256'].get(
        rel(ORIGINAL_ENGINEERING_FAILURE)) == digest(ORIGINAL_ENGINEERING_FAILURE),
        'original failure record is not bound as an original engineering input')
    need(receipt['original_engineering_sha256'].get(
        rel(ENGINEERING / '对照汇总.json')) == digest(ENGINEERING / '对照汇总.json'),
        'original engineering summary hash is absent or changed')
    need(receipt['original_engineering_sha256'].get(
        rel(ENGINEERING / '执行状态.json')) == digest(ENGINEERING / '执行状态.json'),
        'original engineering state hash is absent or changed')
    for field in ('source_sha256', 'original_engineering_sha256'):
        for name, expected in receipt[field].items():
            path = ROOT / name
            need(path.is_file() and digest(path) == expected,
                 f'audit revision {field} binding changed or missing: {name}')
    original_auditor_sha256 = receipt['source_sha256'][rel(ORIGINAL_AUDITOR_SOURCE)]
    need(original_auditor_sha256 == digest(ORIGINAL_AUDITOR_SOURCE),
         'original frozen auditor source hash mismatch')
    source_paths = [AUDIT_REVISION, AUDIT_REVISION_SEAL, ORIGINAL_ENGINEERING_FAILURE,
                    AUDIT_REVISION_SOURCE, ORIGINAL_AUDITOR_SOURCE]
    source_paths.extend(ROOT / name for name in receipt['source_sha256'])
    source_paths.extend(ROOT / name for name in receipt['original_engineering_sha256'])
    return dict(receipt=receipt, seal=seal, receipt_sha256=receipt_hash,
                failure=failure, original_auditor_sha256=original_auditor_sha256,
                source_paths=source_paths)


def verify_audit_revision_v3(revision_v2):
    """Verify the timing-metadata-only v3 receipt layered on the v2 audit fix."""
    need(all(path.is_file() for path in (AUDIT_REVISION_V3, AUDIT_REVISION_SEAL_V3,
                                         ORIGINAL_ENGINEERING_FAILURE_V2,
                                         SUMMARY_DIAGNOSTIC_V2, AUDIT_REVISION_SOURCE_V3)),
         'v3 audit receipt, seal, diagnostic, original v2 failure, and source are required')
    receipt = read(AUDIT_REVISION_V3)
    seal = read(AUDIT_REVISION_SEAL_V3)
    failure = read(ORIGINAL_ENGINEERING_FAILURE_V2)
    diagnostic = read(SUMMARY_DIAGNOSTIC_V2)
    receipt_hash = digest(AUDIT_REVISION_V3)
    need(seal == {'receipt_sha256': receipt_hash}, 'v3 audit revision seal mismatch')
    need(receipt['revision'] == 3 and receipt['audit_only'] is True
         and receipt['neural_rule_actions_or_results_changed'] is False
         and receipt['model_threshold_or_gate_changed'] is False
         and receipt['original_navigation_records_reused'] == 4200
         and receipt['original_probe_records_reused'] == 4320,
         'v3 audit revision scope or reused record totals changed')
    need(receipt['prior_revision_receipt_sha256'] == revision_v2['receipt_sha256'],
         'v3 receipt does not bind the sealed v2 reset correction')
    need(receipt['difference_diagnostic_sha256'] == digest(SUMMARY_DIAGNOSTIC_V2),
         'v3 receipt does not bind the summary difference diagnostic')
    need(failure['passed'] is False and failure['all_neural_and_rule_replays_passed'] is True
         and 'seconds' in failure['exception']
         and failure['statistics_changed'] is False
         and failure['source_or_results_changed'] is False
         and failure['confirmation_started'] is False
         and failure['repair_tests_passed'] is True
         and failure['repair_tests_run'] == 3,
         'preserved v2 audit failure record changed')
    summary_path = ENGINEERING / '对照汇总.json'
    need(diagnostic['stage'] == 'engineering'
         and diagnostic['exact_difference_count'] == 1
         and diagnostic['difference_count_after_metadata_exclusion'] == 0
         and diagnostic['excluded_metadata_keys'] == ['seconds']
         and diagnostic['first_differences'][0]['path'] == '$.seconds'
         and diagnostic['first_differences'][0]['kind'] == 'only_stored'
         and diagnostic['first_differences_after_metadata_exclusion'] == []
         and diagnostic['trajectory_records'] == 4200
         and diagnostic['stored_summary_sha256'] == digest(summary_path)
         and diagnostic['action_replay_performed'] is False
         and diagnostic['model_checkpoints_loaded'] is False
         and diagnostic['read_only_saved_record_reconstruction'] is True,
         'v3 diagnostic does not isolate only the nonreproducible seconds field')
    prior_engineering_hashes = revision_v2['receipt']['original_engineering_sha256']
    need(diagnostic['trajectory_sha256'], 'v3 diagnostic contains no engineering trajectory hashes')
    for name, expected in diagnostic['trajectory_sha256'].items():
        need(prior_engineering_hashes.get(name) == expected
             and digest(ROOT / name) == expected,
             f'v3 diagnostic trajectory differs from preserved engineering input: {name}')
    required_sources = {
        'project/src/eval/audit_spatial_area_confirmation_v3.py',
        'project/src/tests/test_spatial_area_confirmation_audit_v3.py',
    }
    need(required_sources.issubset(receipt['source_sha256']),
         'v3 receipt does not bind the auditor and regression tests')
    required_inputs = {
        rel(AUDIT_REVISION), rel(AUDIT_REVISION_SEAL), rel(SUMMARY_DIAGNOSTIC_V2),
        rel(ORIGINAL_ENGINEERING_FAILURE_V2), rel(summary_path),
        rel(OUT / '预登记.json'), rel(OUT / '预登记封存.json'),
    }
    need(required_inputs.issubset(receipt['input_sha256']),
         'v3 receipt omits a prior receipt or original engineering input')
    for field in ('source_sha256', 'input_sha256'):
        for name, expected in receipt[field].items():
            path = ROOT / name
            need(path.is_file() and digest(path) == expected,
                 f'v3 audit revision {field} binding changed or missing: {name}')
    source_paths = [AUDIT_REVISION_V3, AUDIT_REVISION_SEAL_V3,
                    ORIGINAL_ENGINEERING_FAILURE_V2, SUMMARY_DIAGNOSTIC_V2,
                    AUDIT_REVISION_SOURCE_V3]
    source_paths.extend(ROOT / name for name in receipt['source_sha256'])
    source_paths.extend(ROOT / name for name in receipt['input_sha256'])
    return dict(receipt=receipt, seal=seal, receipt_sha256=receipt_hash,
                failure=failure, diagnostic=diagnostic, source_paths=source_paths)


def readiness():
    """Fail closed unless the audited final verdict binds both completed stages."""
    verdict_path = OUT / '验收结论.json'
    reg_path = OUT / '预登记.json'
    seal_path = OUT / '预登记封存.json'
    protocol_path = OUT / '执行协议.md'
    default_path = ROOT / 'project/local_policy_default.json'
    need(verdict_path.is_file() and reg_path.is_file() and seal_path.is_file(),
         'audited final verdict and preregistration must exist before staging')
    reg = read(reg_path)
    verdict = read(verdict_path)
    registration_seal = read(seal_path)
    expected_seal = dict(registration_sha256=digest(reg_path),
                         protocol_sha256=digest(protocol_path),
                         default_sha256=digest(default_path),
                         navigation_records=14700, probe_records=15120)
    need(registration_seal == expected_seal,
         'the five-field immutable preregistration receipt does not match')
    need(reg['default_sha256'] == expected_seal['default_sha256'],
         'preregistered default does not match sealed default')
    check_frozen_bindings(reg)
    audit_revision_v2 = verify_audit_revision()
    audit_revision_v3 = verify_audit_revision_v3(audit_revision_v2)
    eng_audit, eng_state, eng_summary, eng_probe = stage_checks(
        ENGINEERING, 4200, 4320, audit_revision_v2['receipt_sha256'],
        audit_revision_v3['receipt_sha256'], audit_revision_v2['original_auditor_sha256'])
    conf_audit, conf_state, conf_summary, conf_probe = stage_checks(
        CONFIRMATION, 10500, 10800, audit_revision_v2['receipt_sha256'],
        audit_revision_v3['receipt_sha256'], audit_revision_v2['original_auditor_sha256'])

    need(verdict['completed'] and verdict['engineering_interface_passed']
         and verdict['independent_audit_passed'], 'final verdict is incomplete or unaudited')
    need(verdict['navigation_records'] == 14700 and verdict['probe_records'] == 15120,
         'final verdict record totals do not match the preregistered plan')
    need(verdict['summary_sha256'] == digest(CONFIRMATION / '对照汇总.json'),
         'final verdict does not bind the confirmation summary')
    need(verdict['probe_summary_sha256'] == digest(CONFIRMATION / '探针对照/探针汇总.json'),
         'final verdict does not bind the confirmation probe summary')
    need(verdict['audit_sha256'] == digest(CONFIRMATION / '独立复核.json'),
         'final verdict does not bind the independent confirmation audit')
    need(verdict['default_sha256'] == reg['default_sha256'], 'default configuration binding changed')
    need(verdict['default_changed'] is False and verdict['new_training_steps'] == 0
         and verdict['cloud_calls'] == 0, 'unexpected model, cloud, or default change')
    need(verdict['pretrained_encoder_unknown_geography_verified'] is False,
         'verdict overstates encoder pretraining geography')
    need(conf_summary['maps'] == 10 and conf_summary['planned_total'] == 10500
         and conf_summary['cell_size_m'] == 300
         and conf_summary['projected_area_km2_each'] == 9,
         'confirmation summary geometry or record plan changed')
    need(eng_summary['maps'] == 4 and eng_summary['planned_total'] == 4200,
         'engineering summary map or record plan changed')
    region_manifest = read(DATA / '数据清单.json')
    regions = region_manifest['regions']
    need(len(regions) == 14
         and sum(row['split'] == 'dev' for row in regions) == 4
         and sum(row['split'] == 'test' for row in regions) == 10
         and all(row['projected_area_km2'] == 9 and row['cell_size_m'] == 300 for row in regions),
         'prepared region count, roles, or geometry changed')
    need(verdict['confirmation_regions'] == 10 and verdict['confirmation_tasks_each'] == 750,
         'unexpected confirmation cohort size')
    need(verdict['spatial_area_combined_passed'] == (
        verdict['spatial_unknown_area_navigation_passed']
        and verdict['seam_head_reliability_passed']),
        'combined verdict is inconsistent with its separate gates')
    return dict(reg=reg, verdict=verdict, engineering_audit=eng_audit,
                engineering_state=eng_state, engineering=eng_summary,
                engineering_probe=eng_probe, confirmation_audit=conf_audit,
                confirmation_state=conf_state, confirmation=conf_summary,
                confirmation_probe=conf_probe, registration_seal=registration_seal,
                audit_revision=dict(v2=audit_revision_v2, v3=audit_revision_v3),
                source_paths=[reg_path, seal_path, verdict_path, protocol_path, default_path,
                              ENGINEERING / '独立复核.json', ENGINEERING / '执行状态.json',
                              ENGINEERING / '对照汇总.json', ENGINEERING / '探针对照/探针汇总.json',
                              CONFIRMATION / '独立复核.json', CONFIRMATION / '执行状态.json',
                              CONFIRMATION / '对照汇总.json', CONFIRMATION / '探针对照/探针汇总.json',
                              DATA / '数据清单.json', DATA / '任务分层.json', DATA / '邻接诊断探针.json',
                              PREP / '元数据/连续区域选择.json',
                              OUT / '导航任务.json', OUT / '错误目标计划.json',
                              OUT / '探针错误目标计划.json',
                              ROOT / 'project/src/eval/spatial_area_confirmation.py',
                              ROOT / 'project/src/documents/stage_figures.py',
                              *audit_revision_v2['source_paths'],
                              *audit_revision_v3['source_paths']])


def result_paths(stage_folder, arm, seed=None):
    if arm in CONDITIONS:
        return (stage_folder / '神经对照' / f'M0_s{seed}_{arm}_轨迹.jsonl',
                stage_folder / '神经对照' / f'M0_s{seed}_{arm}_结果.json')
    return (stage_folder / '规则基线' / f'{arm}_轨迹.jsonl',
            stage_folder / '规则基线' / f'{arm}_结果.json')


def read_trace(stage_folder, arm, seed=None):
    trace, sidecar = result_paths(stage_folder, arm, seed)
    meta = read(sidecar)
    need(meta['trajectory_sha256'] == digest(trace), f'trajectory sidecar hash mismatch: {trace.name}')
    rows = [json.loads(line) for line in trace.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    expected = 750 if stage_folder == CONFIRMATION else 300
    need(len(rows) == expected, f'unexpected trajectory denominator: {trace.name}')
    return rows, trace


def trace_value(rows, metric):
    if metric == 'sr':
        return sum(bool(row['success']) for row in rows) / len(rows)
    if metric == 'mean_sg_m':
        return float(np.mean([row['sg_m'] for row in rows]))
    if metric == 'mean_valid_travel_m':
        return float(np.mean([row['valid_travel_m'] for row in rows]))
    if metric == 'mean_failed_sg_m':
        failed = [row for row in rows if not row['success']]
        return None if not failed else float(np.mean([row['sg_m'] for row in failed]))
    raise ValueError(metric)


def read_confirmation_traces(data):
    """Use exact audited trajectories to show all three weights and all regions."""
    summary = data['confirmation']
    raw = {arm: [] for arm in ARMS}
    trace_paths = []
    for arm in ARMS:
        seeds = SEEDS if arm in CONDITIONS else (None,)
        for seed in seeds:
            rows, path = read_trace(CONFIRMATION, arm, seed)
            trace_paths.append(path)
            expected_rows = summary['arms'][arm]['metrics']['episodes']
            if arm in RULES:
                need(expected_rows == 750, f'{arm}: summary rule denominator changed')
            else:
                need(expected_rows == 2250, f'{arm}: summary neural denominator changed')
            raw[arm].append((seed, rows))

        pooled_rows = [row for _, rows in raw[arm] for row in rows]
        metric = summary['arms'][arm]['metrics']
        need(len(pooled_rows) == metric['episodes'], f'{arm}: trajectory/summary row count mismatch')
        need(abs(trace_value(pooled_rows, 'sr') - metric['sr']) < 1e-12,
             f'{arm}: trajectory/summary SR mismatch')
        need(abs(trace_value(pooled_rows, 'mean_sg_m') - metric['mean_sg_m']) < 1e-8,
             f'{arm}: trajectory/summary SG metres mismatch')
        need(abs(trace_value(pooled_rows, 'mean_valid_travel_m') - metric['mean_valid_travel_m']) < 1e-8,
             f'{arm}: trajectory/summary valid travel metres mismatch')
        failed_sg = trace_value(pooled_rows, 'mean_failed_sg_m')
        need((failed_sg is None and metric['mean_failed_sg_m'] is None)
             or (failed_sg is not None and abs(failed_sg - metric['mean_failed_sg_m']) < 1e-8),
             f'{arm}: trajectory/summary failed SG metres mismatch')

    areas = sorted({row['area'] for _, rows in raw['CueFull'] for row in rows})
    need(len(areas) == 10, 'expected ten independently evaluated confirmation regions')
    need(len(summary['arms']['CueFull']['by_source']) == 10,
         'summary source strata do not contain ten confirmation regions')
    values = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for cohort in STRATA:
            values[arm][cohort] = dict(by_area={}, by_seed={})
            cohort_rows = [row for _, rows in raw[arm] for row in rows if row['stratum'] == cohort]
            expected_arm_n = 750 if arm in CONDITIONS else 250
            need(len(cohort_rows) == expected_arm_n, f'{arm}/{cohort}: cohort denominator mismatch')
            stat = summary['arms'][arm]['by_stratum'][cohort]
            need(stat['episodes'] == expected_arm_n, f'{arm}/{cohort}: summary denominator mismatch')
            need(abs(trace_value(cohort_rows, 'sr') - stat['sr']) < 1e-12,
                 f'{arm}/{cohort}: summary SR does not match trajectories')
            need(abs(trace_value(cohort_rows, 'mean_sg_m') - stat['mean_sg_m']) < 1e-8,
                 f'{arm}/{cohort}: summary SG metres do not match trajectories')
            need(abs(trace_value(cohort_rows, 'mean_valid_travel_m') - stat['mean_valid_travel_m']) < 1e-8,
                 f'{arm}/{cohort}: summary valid travel metres do not match trajectories')
            failed_sg = trace_value(cohort_rows, 'mean_failed_sg_m')
            need((failed_sg is None and stat['mean_failed_sg_m'] is None)
                 or (failed_sg is not None and abs(failed_sg - stat['mean_failed_sg_m']) < 1e-8),
                 f'{arm}/{cohort}: summary failed SG metres do not match trajectories')
            for area in areas:
                area_rows = [row for row in cohort_rows if row['area'] == area]
                expected_area_n = 75 if arm in CONDITIONS else 25
                need(len(area_rows) == expected_area_n, f'{arm}/{cohort}/{area}: map denominator mismatch')
                values[arm][cohort]['by_area'][area] = dict(
                    episodes=len(area_rows),
                    sr=trace_value(area_rows, 'sr'),
                    mean_sg_m=trace_value(area_rows, 'mean_sg_m'))
            for seed, rows in raw[arm]:
                seed_rows = [row for row in rows if row['stratum'] == cohort]
                need(len(seed_rows) == 250, f'{arm}/{seed}/{cohort}: raw weight denominator mismatch')
                values[arm][cohort]['by_seed'][str(seed) if seed is not None else 'single_rule'] = dict(
                    episodes=len(seed_rows),
                    successes=sum(bool(row['success']) for row in seed_rows),
                    sr=trace_value(seed_rows, 'sr'),
                    mean_sg_m=trace_value(seed_rows, 'mean_sg_m'))
            values[arm][cohort]['pooled'] = dict(
                episodes=stat['episodes'], successes=stat['successes'], sr=stat['sr'],
                mean_valid_travel_m=stat['mean_valid_travel_m'],
                mean_sg_m=stat['mean_sg_m'], failed_episodes=stat['failed_episodes'],
                mean_failed_sg_m=stat['mean_failed_sg_m'])
    return values, areas, trace_paths


def probe_rates(probe_summary):
    """Validate probe aggregate counts against every frozen checkpoint."""
    need(set(probe_summary['arms']) == {'Full', 'Mean', 'Wrong'}, 'probe conditions changed')
    result = {}
    for condition in ('Full', 'Mean', 'Wrong'):
        result[condition] = {}
        aggregate = probe_summary['arms'][condition]['by_kind']
        for kind in PROBE_KINDS:
            need(kind in aggregate, f'missing probe kind {kind}')
            metric = aggregate[kind]
            by_seed = [probe_summary['by_seed'][str(seed)][condition]['by_kind'][kind]
                       for seed in SEEDS]
            need(metric['records'] == 1200, f'{condition}/{kind}: probe denominator changed')
            need(sum(row['records'] for row in by_seed) == metric['records'],
                 f'{condition}/{kind}: per-weight denominators do not sum')
            if kind == 'nonadjacent_matched_target':
                numerator_key, denom_key = 'raw_false_accepts', 'nonadjacent_denominator'
            else:
                numerator_key, denom_key = 'raw_correct_accepts', 'adjacent_denominator'
            need(sum(row[numerator_key] for row in by_seed) == metric[numerator_key],
                 f'{condition}/{kind}: per-weight numerator mismatch')
            need(sum(row[denom_key] for row in by_seed) == metric[denom_key],
                 f'{condition}/{kind}: per-weight denominator mismatch')
            result[condition][kind] = dict(
                records=metric['records'], raw_accepted=metric['raw_accepted'],
                raw_correct_accepts=metric['raw_correct_accepts'],
                adjacent_denominator=metric['adjacent_denominator'],
                raw_correct_acceptance=metric['raw_correct_acceptance'],
                raw_wrong_direction_accepts=metric['raw_wrong_direction_accepts'],
                raw_false_accepts=metric['raw_false_accepts'],
                nonadjacent_denominator=metric['nonadjacent_denominator'],
                raw_nonadjacent_false_acceptance=metric['raw_nonadjacent_false_acceptance'],
                usable_accepted=metric['usable_accepted'],
                per_seed=[dict(seed=seed, raw_correct_accepts=row['raw_correct_accepts'],
                               adjacent_denominator=row['adjacent_denominator'],
                               raw_correct_acceptance=row['raw_correct_acceptance'],
                               raw_false_accepts=row['raw_false_accepts'],
                               nonadjacent_denominator=row['nonadjacent_denominator'],
                               raw_nonadjacent_false_acceptance=row['raw_nonadjacent_false_acceptance'],
                               raw_wrong_direction_accepts=row['raw_wrong_direction_accepts'],
                               usable_accepted=row['usable_accepted'])
                           for seed, row in zip(SEEDS, by_seed)])
    return result


def format_stratum_table(data, values, cohort):
    rows = [f'| 条件 | 成功数/分母 | SR | 平均实际移动距离（米） | 平均SG终点距离（米） | 失败数 | 失败平均SG（米） |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        v = values[arm][cohort]['pooled']
        rows.append(f"| {ARM_LABELS[arm]} | {v['successes']}/{v['episodes']} | {pct(v['sr'])} | "
                    f"{num(v['mean_valid_travel_m'], 1)} | {num(v['mean_sg_m'], 1)} | "
                    f"{v['failed_episodes']} | {num(v['mean_failed_sg_m'], 1)} |")
    return '\n'.join(rows)


def format_distance_table(summary):
    distances = sorted({int(distance) for arm in ARMS
                        for distance in summary['arms'][arm]['by_distance']})
    rows = ['| 初始距离 | ' + ' | '.join(ARM_LABELS[arm] for arm in ARMS) + ' |',
            '|---|' + '|'.join('---:' for _ in ARMS) + '|']
    for distance in distances:
        cells = []
        for arm in ARMS:
            metric = summary['arms'][arm]['by_distance'][str(distance)]
            cells.append(f"{metric['successes']}/{metric['episodes']}，{pct(metric['sr'])}，"
                         f"SG {num(metric['mean_sg_m'], 0)}米")
        rows.append(f"| C{distance} | " + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def format_probe_table(rates):
    rows = ['| 探针条件 | 类型 | 原始接受数 | 正确接受/分母 | 正确接受率 | 邻接错方向接受 | 非邻接误接受/分母 | 非邻接误接受率 | 门控后接受 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in ('Full', 'Mean', 'Wrong'):
        for kind in PROBE_KINDS:
            v = rates[condition][kind]
            correct = (f"{v['raw_correct_accepts']}/{v['adjacent_denominator']}"
                       if v['adjacent_denominator'] else '—')
            false = f"{v['raw_false_accepts']}/{v['nonadjacent_denominator']}" if v['nonadjacent_denominator'] else '—'
            wrong_direction = (str(v['raw_wrong_direction_accepts'])
                                if v['adjacent_denominator'] else '—')
            rows.append(f"| {condition} | {PROBE_LABELS[kind]} | {v['raw_accepted']} | {correct} | "
                        f"{pct(v['raw_correct_acceptance'])} | {wrong_direction} | {false} | "
                        f"{pct(v['raw_nonadjacent_false_acceptance'])} | {v['usable_accepted']} |")
    return '\n'.join(rows)


def failed_navigation_checks(verdict):
    failures = []
    if not verdict['primary_SR60']:
        failures.append('CueFull长距离SR未达到60%')
    for condition, checks in verdict['navigation_checks'].items():
        for key, passed in checks.items():
            if not passed:
                failures.append(f'{condition}：{key}')
    return failures


def navigation_first_sentence(verdict, summary):
    sr = summary['arms']['CueFull']['by_stratum']['long_distance']['sr']
    failures = failed_navigation_checks(verdict)
    if verdict['spatial_unknown_area_navigation_passed']:
        return (f"预登记的空间未知区域导航门槛全部通过，CueFull长距离主队列SR为{pct(sr)}；"
                '这一结论只针对本轮10个空间隔离区域与冻结M0闭环。')
    reason = '、'.join(failures) if failures else '完整判定没有通过'
    return (f"预登记的空间未知区域导航门槛未全部通过：CueFull长距离主队列SR为{pct(sr)}，"
            f"失败项包括{reason}；本轮不报告导航确认通过。")


def seam_first_sentence(verdict, probe_summary):
    cross = probe_summary['arms']['Full']['by_kind']['cross_source_adjacent']
    nonadj = probe_summary['arms']['Full']['by_kind']['nonadjacent_matched_target']
    status = '通过' if verdict['seam_head_reliability_passed'] else '未通过'
    return (f"Full原始头部接缝可靠性门槛{status}：跨源正确接受率为"
            f"{pct(cross['raw_correct_acceptance'])}，两格非邻接误接受率为"
            f"{pct(nonadj['raw_nonadjacent_false_acceptance'])}。")


def make_captions(data, values, rates):
    verdict, nav, probe = data['verdict'], data['confirmation'], data['confirmation_probe']
    nav_sentence = navigation_first_sentence(verdict, nav)
    seam_sentence = seam_first_sentence(verdict, probe)
    main = nav['arms']['CueFull']['by_stratum']['long_distance']
    seam = probe['arms']['Full']['by_kind']['cross_source_adjacent']
    same = probe['arms']['Full']['by_kind']['same_source_adjacent']
    nonadj = probe['arms']['Full']['by_kind']['nonadjacent_matched_target']
    caption45 = (nav_sentence +
        f"长距离主队列为250道C12–C16任务×3个冻结权重（{main['successes']}/{main['episodes']}）；"
        '另外分别展示250道接缝目标和250道内部目标任务，三类目标都在10个确认区取样。'
        '六种条件为完整目标图像、Baseline（禁用cue）、均值目标图像、错误目标图像、Frontier规则和FixedRegion规则；'
        '柱为全部区域及全部权重的均值，彩色小点为各确认区，空心圆/三角/方块分别为权重0/1/2的结果，'
        '区域点和权重点都不是置信区间。SR为导航成功率；SG为含成功记零的平均终点曼哈顿距离，单位米。'
        '60%门槛只用于完整目标图像的长距离队列；其余预登记配对SR、SG及区域bootstrap门槛见报告。')
    caption46 = (seam_first_sentence(verdict, probe) +
        f"Full在400个不同跨源有向对×3权重上正确接受{seam['raw_correct_accepts']}/"
        f"{seam['adjacent_denominator']}，400个同源邻接对×3权重正确接受{same['raw_correct_accepts']}/"
        f"{same['adjacent_denominator']}；在400个不同两格非邻接对×3权重上误接受"
        f"{nonadj['raw_false_accepts']}/{nonadj['nonadjacent_denominator']}。"
        '三个率使用分开的0–100%坐标；虚线90%和5%是预登记要求，只判定Full跨源正确接受、Full非邻接误接受及至少2/3权重同时满足，'
        '同源邻接正确接受率只作对照诊断。'
        '不是Mean/Wrong的新门槛。点列逐一显示完整目标、均值目标和错误目标三种原始头部判断；空心圆/三角/方块表示全部三个冻结权重，'
        '大菱形是三权重合并计数。raw指0.50五类头部阈值、合法动作与访问过滤前的接受；此图不是闭环SR或闭环成功路线识别率。')
    return caption45, caption46


def make_figure45(values, verdict, summary, caption):
    graphics.style()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    x_centers = np.arange(len(STRATA), dtype=float)
    bar_width = 0.105
    group_offsets = (np.arange(len(ARMS)) - (len(ARMS)-1)/2) * 0.132
    source_jitter = np.linspace(-0.037, 0.037, 10)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.75)
        ax.set_xlim(-.58, 2.58)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    areas = sorted(values['CueFull'][STRATA[0]]['by_area'])
    for cohort_index, cohort in enumerate(STRATA):
        for arm_index, arm in enumerate(ARMS):
            x = x_centers[cohort_index] + group_offsets[arm_index]
            pooled = values[arm][cohort]['pooled']
            sr = pooled['sr'] * 100
            sg = pooled['mean_sg_m']
            axes[0].bar(x, sr, width=bar_width, color=ARM_COLORS[arm],
                        edgecolor='#34424F', linewidth=.55, zorder=2)
            axes[1].bar(x, sg, width=bar_width, color=ARM_COLORS[arm],
                        edgecolor='#34424F', linewidth=.55, zorder=2)
            for area_index, area in enumerate(areas):
                point = values[arm][cohort]['by_area'][area]
                axes[0].scatter(x + source_jitter[area_index], point['sr'] * 100,
                                s=12, color=ARM_COLORS[arm], alpha=.68,
                                edgecolor='white', linewidth=.35, zorder=4)
                axes[1].scatter(x + source_jitter[area_index], point['mean_sg_m'],
                                s=12, color=ARM_COLORS[arm], alpha=.68,
                                edgecolor='white', linewidth=.35, zorder=4)
            if arm in CONDITIONS:
                for seed_index, seed in enumerate(SEEDS):
                    point = values[arm][cohort]['by_seed'][str(seed)]
                    jitter = (seed_index - 1) * .027
                    axes[0].scatter(x + jitter, point['sr'] * 100, s=30,
                                    marker=SEED_MARKERS[seed], facecolor='white',
                                    edgecolor='#202B33', linewidth=.9, zorder=5)
                    axes[1].scatter(x + jitter, point['mean_sg_m'], s=30,
                                    marker=SEED_MARKERS[seed], facecolor='white',
                                    edgecolor='#202B33', linewidth=.9, zorder=5)
            else:
                point = values[arm][cohort]['by_seed']['single_rule']
                axes[0].scatter(x, point['sr'] * 100, s=38, marker='D', facecolor='white',
                                edgecolor='#202B33', linewidth=.9, zorder=5)
                axes[1].scatter(x, point['mean_sg_m'], s=38, marker='D', facecolor='white',
                                edgecolor='#202B33', linewidth=.9, zorder=5)

    axes[0].set_ylim(0, 105)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    axes[0].set_ylabel('SR（导航成功率，%）')
    axes[0].set_title('(a) 三类确认任务上的闭环导航成功率', loc='left')
    full_long_x = group_offsets[ARMS.index('CueFull')]
    axes[0].plot([full_long_x - .065, full_long_x + .065], [60, 60],
                 color='#34424F', linestyle=(0, (4, 3)), linewidth=1.0, zorder=3)
    axes[0].text(full_long_x, 103.5, 'Full门槛60%', ha='center', va='top', fontsize=8.3,
                 color='#34424F', zorder=8,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=1.2))
    all_sg = [values[a][s]['pooled']['mean_sg_m'] for a in ARMS for s in STRATA]
    upper = max(1000, math.ceil(max(all_sg) / 500) * 500 + 500)
    axes[1].set_ylim(0, upper)
    axes[1].set_yticks(np.arange(0, upper + 1, 500 if upper <= 5000 else 1000))
    axes[1].set_ylabel('SG（平均终点距离，米；低为好）')
    axes[1].set_title('(b) 同三类队列的SG距离', loc='left')
    axes[1].set_xticks(x_centers, ['长距离\nC12–C16', '接缝目标\nC8–C12', '内部目标\nC8–C12'])
    axes[1].tick_params(axis='x', length=0, pad=5)
    method_handles = [Patch(facecolor=ARM_COLORS[a], edgecolor='#34424F', label=ARM_LABELS[a]) for a in ARMS]
    fig.legend(handles=method_handles, loc='upper center', bbox_to_anchor=(.5, .987),
               ncol=3, frameon=False, fontsize=9.1, columnspacing=1.2, handlelength=1.4)
    seed_handles = [Line2D([], [], marker=SEED_MARKERS[s], linestyle='', markerfacecolor='white',
                           markeredgecolor='#202B33', label=f'冻结权重 {s}') for s in SEEDS]
    seed_handles.append(Line2D([], [], marker='D', linestyle='', markerfacecolor='white',
                               markeredgecolor='#202B33', label='规则单次运行'))
    fig.legend(handles=seed_handles, loc='upper center', bbox_to_anchor=(.5, .895),
               ncol=4, frameon=False, fontsize=8.5, handletextpad=.35, columnspacing=1.0)
    fig.subplots_adjust(left=.105, right=.985, top=.835, bottom=.135, hspace=.30)
    passed = '通过' if verdict['spatial_unknown_area_navigation_passed'] else '未全部通过'
    fig.text(.5, .027,
             f"10个确认区；每区每层25题。导航门槛{passed}。彩色小点为10区SR/SG；空心符号为权重0/1/2的跨区均值；均不表示区间。",
             ha='center', fontsize=8.8)
    return fig


def make_figure46(rates, verdict, caption):
    graphics.style()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0), sharey=True)
    conditions = ('Full', 'Mean', 'Wrong')
    colors = {'Full': '#0072B2', 'Mean': '#009E73', 'Wrong': '#D55E00'}
    seed_jitter = (-.08, 0.0, .08)
    panels = (
        ('cross_source_adjacent', 'raw_correct_acceptance', 'raw_correct_accepts',
         'adjacent_denominator', '(a) 跨源邻接正确接受'),
        ('same_source_adjacent', 'raw_correct_acceptance', 'raw_correct_accepts',
         'adjacent_denominator', '(b) 同源邻接正确接受'),
        ('nonadjacent_matched_target', 'raw_nonadjacent_false_acceptance', 'raw_false_accepts',
         'nonadjacent_denominator', '(c) 两格非邻接误接受'),
    )
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.75)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', length=0, pad=5)
    for panel_index, (kind, rate_key, numerator_key, denominator_key, title) in enumerate(panels):
        ax = axes[panel_index]
        ax.set_title(title, loc='left')
        for i, condition in enumerate(conditions):
            metric = rates[condition][kind]
            ax.scatter(i, metric[rate_key] * 100, marker='D', s=54,
                       color=colors[condition], edgecolor='#243746', linewidth=.7, zorder=5)
            for seed_index, seed_row in enumerate(metric['per_seed']):
                ax.scatter(i + seed_jitter[seed_index], seed_row[rate_key] * 100,
                           marker=SEED_MARKERS[seed_row['seed']], s=30, facecolor='white',
                           edgecolor=colors[condition], linewidth=1.0, zorder=6)
        labels = []
        for condition in conditions:
            metric = rates[condition][kind]
            labels.append(f"{condition}\n{ARM_LABELS['CueFull'] if condition == 'Full' else ('均值目标' if condition == 'Mean' else '错误目标')}\n"
                          f"{metric[numerator_key]}/{metric[denominator_key]}")
        ax.set_xticks(np.arange(3), labels)
    axes[0].plot([-.28, .28], [90, 90], color='#34424F',
                 linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    axes[2].plot([-.28, .28], [5, 5], color='#34424F',
                 linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    axes[0].text(.31, 91.5, 'Full门槛 90%', ha='left', va='bottom', fontsize=8.0,
                 color='#34424F')
    axes[2].text(.31, 6.5, 'Full门槛 5%', ha='left', va='bottom', fontsize=8.0,
                 color='#34424F')
    axes[0].set_ylabel('原始头部正确接受率（%）')
    axes[1].set_ylabel('原始头部正确接受率（%）')
    axes[2].set_ylabel('原始头部误接受率（%）')
    fig.subplots_adjust(left=.07, right=.995, bottom=.285, top=.88, wspace=.34)
    seed_handles = [Line2D([], [], marker=SEED_MARKERS[s], linestyle='', markerfacecolor='white',
                           markeredgecolor='#34424F', label=f'权重 {s}') for s in SEEDS]
    seed_handles.append(Line2D([], [], marker='D', linestyle='', markerfacecolor='#7C8792',
                               markeredgecolor='#243746', label='三权重合并计数'))
    fig.legend(handles=seed_handles, loc='upper center', bbox_to_anchor=(.5, .985),
               ncol=4, frameon=False, fontsize=8.8, handletextpad=.35, columnspacing=1.1)
    status = '通过' if verdict['seam_head_reliability_passed'] else '未通过'
    fig.text(.5, .035,
             f'Full接缝头部门槛{status}；每种探针分母1200（400对×3冻结权重）。同源接受为诊断，Mean/Wrong不适用90%/5%门槛。',
             ha='center', fontsize=8.8)
    return fig


def save_figure_exclusive(fig, stem, caption, sources, protocol):
    fig.canvas.draw()
    files = []
    for fmt in ('pdf', 'svg', 'png'):
        buffer = BytesIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig.savefig(buffer, format=fmt, dpi=300, bbox_inches='tight')
        missing = [str(w.message) for w in caught if 'Glyph' in str(w.message)]
        need(not missing, f'{stem}: missing-font glyph: {missing[0] if missing else ""}')
        content = buffer.getvalue()
        if fmt == 'pdf':
            need(b'/FontFile' in content, f'{stem}: PDF has no embedded font program')
            need(b'/Subtype /Image' not in content, f'{stem}: PDF unexpectedly embeds a bitmap')
        elif fmt == 'svg':
            root = ET.fromstring(content)
            image_tags = [node for node in root.iter() if node.tag.split('}')[-1] == 'image']
            text_tags = [node for node in root.iter() if node.tag.split('}')[-1] == 'text']
            need(not image_tags and text_tags, f'{stem}: SVG must be vector with editable text')
        else:
            with Image.open(BytesIO(content)) as image:
                xdpi, ydpi = image.info.get('dpi', (0, 0))
                need(abs(xdpi - 300) < 1.5 and abs(ydpi - 300) < 1.5,
                     f'{stem}: PNG must be 300 dpi (found {xdpi}, {ydpi})')
        path = PLOTS / f'{stem}.{fmt}'
        write_exclusive(path, content)
        files.append(dict(path=rel(path), sha256=sha256(content).hexdigest(), format=fmt))
    plt.close(fig)
    return dict(id=stem, caption=caption, protocol=protocol, sources=sources, files=files,
                reproduction_script=rel(Path(__file__)))


def navigation_gate_text(data):
    verdict, summary = data['verdict'], data['confirmation']
    main = summary['arms']['CueFull']['by_stratum']['long_distance']
    checks = verdict['navigation_checks']
    rows = ['| 对照（Full − 对照） | 主队列SR差 | 正向权重 | SG变化（米） | 95%区域bootstrap SR差区间 | 每项门槛 |',
            '|---|---:|---:|---:|---:|---|']
    failures = failed_navigation_checks(verdict)
    for condition in ('Baseline', 'CueMean', 'CueWrong'):
        primary = summary['effects']['long_distance'][condition]
        pooled = summary['effects']['all'][condition]
        check = checks[condition]
        rows.append(f"| {ARM_LABELS[condition]} | {signed_pct(primary['sr_gain'])} | "
                    f"{primary['positive_seeds']}/3 | {num(primary['sg_change'] * 300, 1)} | "
                    f"[{signed_pct(primary['source95_SR'][0])}, {signed_pct(primary['source95_SR'][1])}] | "
                    f"SR≥5pp {fmt_bool(check['SR_gain5pp'])}；正向≥2 {fmt_bool(check['two_positive_weights'])}；"
                    f"主SG不差 {fmt_bool(check['primary_SG_no_worse'])}；全体SG不差 {fmt_bool(check['pooled_SG_no_worse'])}；"
                    f"区间下界>0 {fmt_bool(check['source95_SR_lower_positive'])} |")
    all_sr = summary['arms']['CueFull']['by_stratum']['long_distance']['sr']
    state = '通过' if verdict['spatial_unknown_area_navigation_passed'] else '未通过'
    lead = (f"导航判定：**{state}**。CueFull长距离SR为{pct(all_sr)}（{main['successes']}/{main['episodes']}），"
            '项目门槛为≥60%；该门槛与三项目标对照的配对要求同时满足才构成导航通过。')
    if failures:
        lead += ' 本次未通过项：' + '、'.join(failures) + '。'
    else:
        lead += ' 所有预登记导航子门槛均满足。'
    return lead, '\n'.join(rows)


def probe_gate_text(data, rates):
    verdict, probe = data['verdict'], data['confirmation_probe']
    full_cross = rates['Full']['cross_source_adjacent']
    full_nonadj = rates['Full']['nonadjacent_matched_target']
    passed = verdict['seam_head_reliability_passed']
    lead = (f"接缝头部判定：**{'通过' if passed else '未通过'}**。Full原始头部跨源正确接受率"
            f"{pct(full_cross['raw_correct_acceptance'])}（{full_cross['raw_correct_accepts']}/"
            f"{full_cross['adjacent_denominator']}，要求≥90%），两格非邻接误接受率"
            f"{pct(full_nonadj['raw_nonadjacent_false_acceptance'])}（{full_nonadj['raw_false_accepts']}/"
            f"{full_nonadj['nonadjacent_denominator']}，要求≤5%）；还要求至少2/3权重同时满足两项。")
    if not passed:
        checks = verdict['seam_checks']
        failed = [key for key, value in checks.items() if not value]
        lead += ' 未通过子项：' + '、'.join(failed) + '。'
    else:
        lead += ' 全部接缝预登记子门槛满足。'
    per_seed = ['| 权重 | Full跨源正确接受 | Full两格非邻接误接受 | 两项同时满足 |',
                '|---:|---:|---:|---|']
    for seed in SEEDS:
        cross = full_cross['per_seed'][seed]
        nonadj = full_nonadj['per_seed'][seed]
        both = (cross['raw_correct_acceptance'] >= .90 - 1e-12
                and nonadj['raw_nonadjacent_false_acceptance'] <= .05 + 1e-12)
        per_seed.append(f"| {seed} | {cross['raw_correct_accepts']}/{cross['adjacent_denominator']} "
                        f"（{pct(cross['raw_correct_acceptance'])}） | "
                        f"{nonadj['raw_false_accepts']}/{nonadj['nonadjacent_denominator']} "
                        f"（{pct(nonadj['raw_nonadjacent_false_acceptance'])}） | {fmt_bool(both)} |")
    return lead, '\n'.join(per_seed)


def make_report(data, values, rates, captions):
    verdict = data['verdict']
    audit_revision_v2 = data['audit_revision']['v2']
    audit_revision_v3 = data['audit_revision']['v3']
    nav_lead, nav_table = navigation_gate_text(data)
    seam_lead, seed_table = probe_gate_text(data, rates)
    eng = data['engineering']
    conf = data['confirmation']
    combined = verdict['spatial_area_combined_passed']
    verdict_label = '组合门槛通过' if combined else '组合门槛未通过'
    overall_rows = ['| 条件 | 成功数/分母 | SR | 平均实际移动距离（米） | 平均SG终点距离（米） | 失败数 | 失败平均SG（米） |',
                    '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        metric = conf['arms'][arm]['metrics']
        overall_rows.append(f"| {ARM_LABELS[arm]} | {metric['successes']}/{metric['episodes']} | "
                            f"{pct(metric['sr'])} | {num(metric['mean_valid_travel_m'], 1)} | "
                            f"{num(metric['mean_sg_m'], 1)} | {metric['failed_episodes']} | "
                            f"{num(metric['mean_failed_sg_m'], 1)} |")
    caption45, caption46 = captions
    plot45 = Path(os.path.relpath(PLOTS / '45_空间隔离区域三分层M0导航结果.png', OUT)).as_posix()
    plot46 = Path(os.path.relpath(PLOTS / '46_跨源接缝原始头部接受率.png', OUT)).as_posix()
    prep_manifest = read(DATA / '数据清单.json')
    prepared_regions = prep_manifest['regions']
    dev_regions = sum(region['split'] == 'dev' for region in prepared_regions)
    test_regions = sum(region['split'] == 'test' for region in prepared_regions)
    need(len(prepared_regions) == 14 and dev_regions == 4 and test_regions == 10
         and all(region['projected_area_km2'] == 9 for region in prepared_regions),
         'report geometry differs from the frozen data manifest')
    wrong_plan = read(OUT / '错误目标计划.json')
    distance_exceptions = sum(not row['matched_distance'] for row in wrong_plan.values())
    need(distance_exceptions == 69, 'frozen navigation wrong-target exception count changed')
    probe_wrong = read(OUT / '探针错误目标计划.json')
    need(all(row.get('shared_matched_triple') is True for row in probe_wrong.values()),
         'wrong-probe targets are not shared within each matched triple')
    source_scope = verdict['scope']
    engineering_table = ['| 条件 | SR | 平均实际移动距离（米） | 平均SG终点距离（米） | 失败数 | 失败平均SG（米） | 题数 |',
                         '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        metric = eng['arms'][arm]['metrics']
        engineering_table.append(f"| {ARM_LABELS[arm]} | {pct(metric['sr'])} | "
                                 f"{num(metric['mean_valid_travel_m'], 1)} | {num(metric['mean_sg_m'], 1)} | "
                                 f"{metric['failed_episodes']} | {num(metric['mean_failed_sg_m'], 1)} | "
                                 f"{metric['episodes']} |")
    return f'''# 新空间隔离区域M0导航确认报告

**{verdict_label}：导航确认{'通过' if verdict['spatial_unknown_area_navigation_passed'] else '未通过'}；接缝头部可靠性{'通过' if verdict['seam_head_reliability_passed'] else '未通过'}。** 原S2/S3/S4判定与本地默认保持不变。

## 范围与固定预算

本轮只确认冻结M0在本项目本地导航训练/开发足迹之外的新区域上的闭环表现，以及独立接缝探针的原始头部可靠性。使用Massachusetts Roads发布影像，同属Massachusetts影像发布系列；工程与确认区域各为单区9 km²、300米网格、10×10/B20，14区合计126 km²是分离足迹面积之和，不是一张连续地图。区域规则为相对本地旧151个Masa足迹及新区域彼此空间隔离；这不证明Sat2Cap编码器预训练没有见过这些地点，也不是跨数据集、跨成像域或全球未见保证。

| 阶段 | 区域数 | 唯一导航任务 | 神经轨迹 | 规则轨迹 | 探针头部判断 |
|---|---:|---:|---:|---:|---:|
| 工程接口 | {eng['maps']} | 300 | {eng['planned_neural']} | {eng['planned_rules']} | {data['engineering_audit']['probe_records']} |
| 独立确认 | {conf['maps']} | 750 | {conf['planned_neural']} | {conf['planned_rules']} | {data['confirmation_audit']['probe_records']} |
| 合计 | 14 | 1050 | 12600 | 2100 | 15120 |

工程阶段4区、300任务、4200闭环记录及4320探针均通过独立复核；其SR不作选型门槛。确认阶段10区、750条不重复路线、10500闭环记录及10800探针均通过独立复核。全轮共14700条导航终局记录、15120个头部判断。每区75个任务：25长距离C12–C16、25接缝目标C8–C12、25内部目标C8–C12；后两层距离较短，始终分开报告。

### 工程关观察值（不用于选型）

| 条件 | SR | 平均SG（米） | 题数 |
|---|---:|---:|---:|
{chr(10).join(engineering_table[2:])}

固定模型来源为此前的main、Small256 NoTarget探索器及EdgeTargetCue头的三份冻结权重和原均值。没有新训练或云端调用，0.50原始概率阈值、合法动作/未访问格门控、Frontier与FixedRegion规则、默认配置均保持。Full会使用给定目标图像；Baseline禁用cue；Mean将cue特征均值遮蔽；Wrong使用预先冻结的错误目标。三份权重全部报告，不挑最高种子、不集成。

## 确认集总体结果

下表对每个神经条件合并3份冻结权重，因此分母为2250；两条规则各执行一次，分母为750。混合750题的SR只作描述，不代替长距离主队列或完整通过判定。平均实际移动距离是每题沿合法移动累积的路径长度，越界原地消耗一步但不增加距离；SG是终点曼哈顿距离，成功记0，单位已换算为米。两者一个反映走过的路径，一个反映终点距目标的距离。失败平均SG只对失败题求均值。

{chr(10).join(overall_rows)}

## 三个预登记队列

长距离门槛只检验CueFull在250道C12–C16任务×3权重上的SR≥60%。Baseline、CueMean、CueWrong分别要求：长距离配对SR提高至少5个百分点、至少2/3权重有正增益、主队列和完整750题混合队列SG均不变差、10个区域成组4000次bootstrap的SR增益95%区间下界大于0。全部条件同时成立才报告导航确认通过。

配对表中的SG变化按Full减去对照计算，负值表示Full的平均终点更近。

{nav_lead}

{nav_table}

以下分层表保留三种队列与全部六个条件的失败结果，分母分别为神经条件750条（250题×3权重）和规则条件250条。失败SG单独计数/不被SR成功行抵消；SG对任何门槛的用法以预登记配对比较为准。

### 长距离锚点 C12–C16

{format_stratum_table(data, values, 'long_distance')}

### 接缝目标 C8–C12

{format_stratum_table(data, values, 'seam_target')}

### 内部目标 C8–C12

{format_stratum_table(data, values, 'interior_target')}

## 初始距离分档（描述性）

C8–C16按真实初始曼哈顿距离逐档展示；神经条件每格合并3权重，规则各为一次运行。C12同时可来自三种预登记队列，故该表仅检查距离分布，不替代分层后的长距离主队列或任何通过门槛。

{format_distance_table(conf)}

![图45：空间隔离区域三分层M0导航结果]({plot45})

**图45图注。** {caption45}

## 原始头部接缝探针

预登记Full门槛按原始五分类top方向且概率≥0.50定义接受；跨源正确接受还要求方向正确，分母为400个不同跨源有向对×3权重=1200。两格非邻接误接受是任一方向被raw接受，分母为400个不同非邻接对×3权重=1200。接缝门槛为Full正确接受≥90%、Full非邻接误接受≤5%，并要求至少2/3权重各自同时满足；区间只作描述，不重新调门槛。Mean与Wrong为固定诊断条件，不按Full门槛宣告通过或失败。

{seam_lead}

| 条件 | 类型 | 原始接受数 | 正确接受/分母 | 正确接受率 | 错方向接受 | 非邻接误接受/分母 | 非邻接误接受率 | 门控后接受 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(format_probe_table(rates).splitlines()[2:])}

{seed_table}

raw接受统计发生在合法动作/访问过滤前；“门控后接受”另按探针执行时的门控统计。两种数字口径不同。Mean/Wrong的正确接受仍以原目标方向为标签，同时错误目标在每个跨源/同源/非邻接三元组内共用一个预先抽取目标；抽取不要求距离匹配，真实匹配标志和错误给定目标方向均保留。闭环错误目标导航中有{distance_exceptions}条距离无法匹配的例外，全部保留在分母内，不能单独据此声称目标贡献。

![图46：跨源接缝原始头部接受率]({plot46})

**图46图注。** {caption46}

## 审计、来源与解释边界

- 工程阶段独立复核：{data['engineering_audit']['passed']}；4200条闭环、4320次探针。确认阶段独立复核：{data['confirmation_audit']['passed']}；10500条闭环、10800次探针。最终结论绑定两份汇总、探针汇总与确认审计的SHA-256。
- **审计修订来源。** 原v1独立复核在12组神经回放完成后，规则回放因向无状态规则对象调用`reset`而中止；v2审计封装只为规则重放加入无操作`reset`兼容。v2随后完成回放，但审计对照仅在不可重建的汇总元数据`seconds`字段上相差一项，所有其他统计一致；v3只对该字段验证有限正数并继续比较其余汇总。原始[v1失败记录](工程接口/原独立复核失败_v1.json)、[v2失败记录](工程接口/原独立复核失败_v2.json)、[差异诊断](工程接口/汇总差异诊断_v2.json)、两版审计源码/测试、[v2收据](审计修订登记_v2.json)和[v3收据](审计修订登记_v3.json)及旧工程输入哈希均在收据链中核验。修订仅更正审计流程，没有重跑导航或探针、产生新SR、改动作/模型/阈值/门槛；v2与v3各3项修复回归检查通过，原40项协议检查保持。
- 预登记导航门槛：SR主队列{'通过' if verdict['primary_SR60'] else '未通过'}，目标控制的数值门槛{'全部通过' if verdict['target_evidence_passed'] else '未全部通过'}。接缝头部原始可靠性{'通过' if verdict['seam_head_reliability_passed'] else '未通过'}。任一分关不通过，组合确认即未通过。
- 本轮0训练步、0云调用、未修改默认或策略阈值。原S2/S3/S4验收结论不由本轮自动更改；未将此结果写成多Agent提升。
- 空间隔离只支持本地M0导航/视觉头任务所用旧151足迹之外的项目级空间确认。预训练编码器的地理训练范围未知；所有新影像仍来自同一Massachusetts Roads发布家族。没有验证跨视角、跨模态、真实异时、任意角度、真实异步或实地飞行。
- 探针正确/错误接受率不是导航SR，单一成功率也不证明普遍的接缝识别可靠性。10个区域构成本轮空间样本；3权重和每格探针都不是额外的独立区域。
- 完整区域、距离、探针来源图块及四方向明细分别保留于[闭环汇总](独立确认/对照汇总.json)和[探针汇总](独立确认/探针对照/探针汇总.json)，SHA-256由最终判定绑定。

`source_scope`: {source_scope}

完整逐区域、距离、探针来源图块及四方向明细分别保留于[闭环汇总](独立确认/对照汇总.json)和[探针汇总](独立确认/探针对照/探针汇总.json)，SHA-256由最终判定绑定。重绘需要使用评测专用环境中的D:\\PYTHON\\python.exe，且输出文件必须尚不存在。
'''


def make_readme(data, figure_items, source_hashes):
    verdict = data['verdict']
    nav = '通过' if verdict['spatial_unknown_area_navigation_passed'] else '未通过'
    seam = '通过' if verdict['seam_head_reliability_passed'] else '未通过'
    combined = '通过' if verdict['spatial_area_combined_passed'] else '未通过'
    files = []
    for item in figure_items:
        base = item['id']
        files.append(f"- [{base}.png](../../../../../绘图/数据结果图/{base}.png) / "
                     f"[{base}.pdf](../../../../../绘图/数据结果图/{base}.pdf) / "
                     f"[{base}.svg](../../../../../绘图/数据结果图/{base}.svg)")
    return f'''# 冻结M0空间隔离导航确认 v1

**组合确认{combined}**：导航分关{nav}；接缝头部可靠性{seam}。本轮总计14700条导航终局记录、15120个探针判断；4个工程区与10个独立确认区，每个9km²。原S2/S3/S4结论及默认保持原样。

## 成果

- [新空间隔离区域M0导航确认报告.md](新空间隔离区域M0导航确认报告.md)：各分层完整SR/SG、全部导航及接缝门槛、失败项和范围解释。
- 绘图45：三分层闭环SR与米制SG，按区域展示离散值并显示三个冻结权重。
- 绘图46：Full/Mean/Wrong的跨源正确接受与非邻接误接受，分列预登记90%/5%Full门槛。
- 图注：[冻结M0空间隔离导航确认图注.md](../../../../../绘图/冻结M0空间隔离导航确认图注.md)
- 机器可读清单：[绘图数据/冻结M0空间隔离导航确认_v1.json](../../../../../绘图/绘图数据/冻结M0空间隔离导航确认_v1.json)

{chr(10).join(files)}

## 固定边界

继续使用先前冻结的main/Small256 NoTarget/EdgeTargetCue三组权重、原均值和0.50阈值；没有训练、云端调用、默认升级或策略改动。空间隔离是对本地M0任务区域的确认，不能推出Sat2Cap预训练未见、不同影像域泛化或真实航飞有效。探针可靠性与闭环SR分开报告。

## 溯源

来源文件SHA-256记录在图表清单中（共{len(source_hashes)}个文件），并由最终判定和两阶段独立审计绑定。重建脚本：`{rel(Path(__file__))}`；运行前必须存在并通过最终审计判定，脚本对所有既有成果使用独占创建，拒绝覆盖。
'''


def outputs_absent():
    expected = [REPORT, README, CAPTIONS, PLOT_DATA]
    for stem in ('45_空间隔离区域三分层M0导航结果', '46_跨源接缝原始头部接受率'):
        expected.extend(PLOTS / f'{stem}.{fmt}' for fmt in ('png', 'pdf', 'svg'))
    existing = [str(path) for path in expected if path.exists()]
    need(not existing, 'exclusive-create refusal; output already exists: ' + ', '.join(existing))


def main():
    # No output path is opened until audited readiness and every source binding pass.
    outputs_absent()
    data = readiness()
    values, areas, traces = read_confirmation_traces(data)
    rates = probe_rates(data['confirmation_probe'])
    captions = make_captions(data, values, rates)
    captions_text = '# 冻结M0空间隔离导航确认图注\n\n'
    captions_text += '## 图45：空间隔离区域三分层M0导航结果\n\n' + captions[0] + '\n\n'
    captions_text += '## 图46：跨源接缝原始头部接受率\n\n' + captions[1] + '\n'

    source_paths = data['source_paths'] + traces
    source_hashes = {rel(path): digest(path) for path in sorted(set(source_paths))}
    figure45_stem = '45_空间隔离区域三分层M0导航结果'
    figure46_stem = '46_跨源接缝原始头部接受率'
    protocol = ('冻结M0独立确认：10个区域；250长距离C12–C16 + 250接缝目标C8–C12 '
                '+ 250内部目标C8–C12；三份冻结权重和两条规则按预登记执行。')
    item45 = save_figure_exclusive(
        make_figure45(values, data['verdict'], data['confirmation'], captions[0]),
        figure45_stem, captions[0], list(source_hashes), protocol)
    item46 = save_figure_exclusive(
        make_figure46(rates, data['verdict'], captions[1]),
        figure46_stem, captions[1], list(source_hashes),
        '冻结接缝探针：400跨源相邻、400同源相邻、400两格非邻接不同探针对；每类×3权重；仅原始头部可靠性。')

    report_text = make_report(data, values, rates, captions)
    readme_text = make_readme(data, (item45, item46), source_hashes)
    write_exclusive(REPORT, report_text)
    write_exclusive(README, readme_text)
    write_exclusive(CAPTIONS, captions_text)

    output_hashes = {}
    for item in (item45, item46):
        for file in item['files']:
            path = ROOT / file['path']
            need(digest(path) == file['sha256'], f'figure changed after exclusive creation: {path.name}')
            output_hashes[file['path']] = file['sha256']
    for path in (REPORT, README, CAPTIONS):
        output_hashes[rel(path)] = digest(path)
    manifest = dict(
        date=date.today().isoformat(),
        run='冻结M0空间隔离导航确认_v1',
        verdict=dict(path=rel(OUT / '验收结论.json'), sha256=digest(OUT / '验收结论.json'),
                     completed=data['verdict']['completed'],
                     spatial_unknown_area_navigation_passed=data['verdict']['spatial_unknown_area_navigation_passed'],
                     seam_head_reliability_passed=data['verdict']['seam_head_reliability_passed'],
                     spatial_area_combined_passed=data['verdict']['spatial_area_combined_passed'],
                     default_changed=data['verdict']['default_changed'],
                     new_training_steps=data['verdict']['new_training_steps'],
                     cloud_calls=data['verdict']['cloud_calls']),
        counts=dict(engineering_regions=4, confirmation_regions=10, square_km_each=9,
                    navigation_records=14700, probe_records=15120,
                    confirmation_unique_tasks=750, confirmation_probes_per_kind=400,
                    confirmation_probe_weighted_denominator_per_kind=1200),
        source_scope=data['verdict']['scope'],
        registration_seal=data['registration_seal'],
        audit_revision=dict(final_revision=3,
            v2=dict(receipt_sha256=data['audit_revision']['v2']['receipt_sha256'],
                    seal=data['audit_revision']['v2']['seal'],
                    receipt=data['audit_revision']['v2']['receipt'],
                    preserved_original_failure=data['audit_revision']['v2']['failure']),
            v3=dict(receipt_sha256=data['audit_revision']['v3']['receipt_sha256'],
                    seal=data['audit_revision']['v3']['seal'],
                    receipt=data['audit_revision']['v3']['receipt'],
                    preserved_original_failure=data['audit_revision']['v3']['failure'],
                    summary_difference_diagnostic=data['audit_revision']['v3']['diagnostic']),
            engineering_stage_stamp=dict(
                revision=data['engineering_audit']['audit_revision'],
                v3_receipt_sha256=data['engineering_audit']['audit_revision_receipt_sha256'],
                v2_receipt_sha256=data['engineering_audit']['reset_revision_receipt_sha256']),
            confirmation_stage_stamp=dict(
                revision=data['confirmation_audit']['audit_revision'],
                v3_receipt_sha256=data['confirmation_audit']['audit_revision_receipt_sha256'],
                v2_receipt_sha256=data['confirmation_audit']['reset_revision_receipt_sha256'])),
        source_sha256=source_hashes,
        frozen_bindings={field: data['reg'][field] for field in
                         ('protected_sha256', 'source_sha256', 'input_sha256',
                          'encoder_sha256', 'frozen_output_sha256')},
        confirmation_summary=dict(
            primary_SR=data['confirmation']['primary_SR'],
            primary_SR60=data['confirmation']['primary_SR60'],
            target_checks=data['confirmation']['target_checks'],
            target_numeric_passed=data['confirmation']['target_numeric_passed'],
            seam_checks=data['confirmation_probe']['seam_checks'],
            seam_numeric_passed=data['confirmation_probe']['seam_numeric_passed']),
        figure_data=dict(
            full_audited_navigation_summary=data['confirmation'],
            full_audited_probe_summary=data['confirmation_probe'],
            arms={arm: {cohort: values[arm][cohort] for cohort in STRATA} for arm in ARMS},
            probe_rates=rates),
        figures=[item45, item46],
        other_outputs_sha256={key: value for key, value in output_hashes.items()
                              if key not in {f['path'] for item in (item45, item46) for f in item['files']}},
        reproduction_script=rel(Path(__file__)),
        script_sha256=digest(Path(__file__)))
    write_exclusive(PLOT_DATA, json_bytes(manifest))
    print(json.dumps(dict(report=rel(REPORT), readme=rel(README), manifest=rel(PLOT_DATA),
                          figures=[f['id'] for f in (item45, item46)],
                          navigation_passed=data['verdict']['spatial_unknown_area_navigation_passed'],
                          seam_passed=data['verdict']['seam_head_reliability_passed'],
                          combined_passed=data['verdict']['spatial_area_combined_passed']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
