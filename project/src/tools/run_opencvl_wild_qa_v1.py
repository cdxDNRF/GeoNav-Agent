"""Generate CV-050 bounded <=24-pair fit_train QA and Stage-1 single-step baseline bundle.

Strictly enforces:
- <= 24 pairs (48 PNGs) decoded exclusively from development_fit_train;
- 0 pixel decodes on development_validation, isolated_confirmation_blind, blocked,
  quarantined, or PL domains;
- 0 evaluation or label inspection on isolated_confirmation_blind (28 pairs);
- Full SHA-256 verification of upstream CV-049 frozen artifacts before execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from project.src.data.opencvl_wild_qa_v1 import (
    ALLOWED_SE_AERIAL_SIZES,
    ALLOWED_SE_GROUND_SIZES,
    DEFAULT_RESOLUTION_M,
    INTERIOR_MARGIN_THRESHOLD_M,
    LARGE_DISCREPANCY_THRESHOLD_M,
    QA_MAX_PAIRS,
    SOFT_TARGET_SIGMA_M,
    STRATUM_NAMES,
    build_statistical_baselines,
    evaluate_baseline_suite,
    inspect_png_image_quality,
    select_fit_train_qa_pairs,
    summarize_split_discrepancy,
)

ROOT = Path(__file__).resolve().parents[3]
RAW_BASE = ROOT / 'DATA/raw_data/OpenCVL/OpenCVL/mapillary/in_the_wild_test'
CV049_DIR = ROOT / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1'
OUT_DIR = ROOT / 'DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1'

UPSTREAM_REQUIRED_BINDINGS = (
    'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/协议冻结_v1.json',
    'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/公开洁净划分清单_v1.json',
    'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/核验/评估器专用冻结台账_禁止入模.json',
)

CV050_CODE_PATHS = (
    'project/src/data/opencvl_wild_qa_v1.py',
    'project/src/tools/run_opencvl_wild_qa_v1.py',
    'project/src/eval/audit_opencvl_wild_qa_v1.py',
    'project/src/tests/test_opencvl_wild_qa_v1.py',
)

ARTIFACT_FILENAMES = (
    'README.md',
    '抽检24对画质与分歧核验_v1.json',
    '单步定位统计基线对照_v1.json',
    '单步视觉实验冻结协议_v1.json',
)


def file_sha256(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_cv049_integrity(root: Path = ROOT) -> tuple[dict, dict, dict]:
    """Verify upstream CV-049 frozen SHA-256 digests before touching any data."""
    cv049_dir = root / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1'
    binding = load_json(cv049_dir / '核验/源码与输入绑定_v1.json')
    for rel, expected_sha in binding['protected_upstream_sha256'].items():
        if file_sha256(root / rel) != expected_sha:
            raise ValueError(f'CV-049 protected upstream digest mismatch: {rel}')
    for entry in binding['generator_sources'] + binding['generated_artifacts']:
        p = root / entry['path']
        if not p.is_file() or p.stat().st_size != entry['bytes'] or file_sha256(p) != entry['sha256']:
            raise ValueError(f'CV-049 frozen digest mismatch: {entry["path"]}')
    ledger = load_json(cv049_dir / '核验/评估器专用冻结台账_禁止入模.json')
    protocol = load_json(cv049_dir / '协议冻结_v1.json')
    return binding, ledger, protocol


def extract_allowed_dev_records(ledger: dict) -> tuple[list[dict], list[dict]]:
    """Return ONLY development_fit_train and development_validation SE records."""
    fit_train = [r for r in ledger['se_records'] if r.get('role') == 'development_fit_train']
    dev_val = [r for r in ledger['se_records'] if r.get('role') == 'development_validation']
    if len(fit_train) != 435 or len(dev_val) != 116:
        raise ValueError(f'Unexpected development split counts: fit_train={len(fit_train)}, dev_val={len(dev_val)}')
    return fit_train, dev_val


def run_bounded_fit_train_qa(
    fit_train: list[dict],
    dev_val: list[dict],
    raw_base: Path = RAW_BASE,
) -> dict:
    """Decode <=24 pairs (48 PNGs) from development_fit_train and summarize dev discrepancies."""
    fit_summary = summarize_split_discrepancy(fit_train, 'development_fit_train')
    val_summary = summarize_split_discrepancy(dev_val, 'development_validation')

    selected = select_fit_train_qa_pairs(fit_train, max_pairs=QA_MAX_PAIRS)
    rec_by_id = {r['image_id']: r for r in fit_train}

    decode_count = 0
    rows = []
    stratum_counts = {}
    ground_size_counts = {}

    for item in selected:
        sid = item['image_id']
        rec = rec_by_id[sid]
        if rec.get('role') != 'development_fit_train':
            raise ValueError(f'Attempted QA decode outside development_fit_train: {sid}')

        g_path = raw_base / item['ground_rel']
        a_path = raw_base / item['aerial_rel']
        g_info = inspect_png_image_quality(g_path)
        decode_count += 1
        a_info = inspect_png_image_quality(a_path)
        decode_count += 1

        g_size = (g_info['header']['width'], g_info['header']['height'])
        a_size = (a_info['header']['width'], a_info['header']['height'])
        exp_g = (rec['ground_asset']['width'], rec['ground_asset']['height'])
        exp_a = (rec['aerial_asset']['width'], rec['aerial_asset']['height'])
        if g_size not in ALLOWED_SE_GROUND_SIZES or g_size != exp_g:
            raise ValueError(f'Unexpected ground PNG size for {sid}: {g_size} vs ledger {exp_g}')
        if a_size not in ALLOWED_SE_AERIAL_SIZES or a_size != exp_a:
            raise ValueError(f'Unexpected aerial PNG size for {sid}: {a_size} vs ledger {exp_a}')
        if not g_info['quality']['nonblank'] or not a_info['quality']['nonblank']:
            raise ValueError(f'Nonblank quality gate failed for {sid}')

        stratum_counts[item['qa_stratum']] = stratum_counts.get(item['qa_stratum'], 0) + 1
        gk = f'{g_size[0]}x{g_size[1]}'
        ground_size_counts[gk] = ground_size_counts.get(gk, 0) + 1

        rows.append({
            'qa_rank': item['qa_rank'],
            'qa_stratum': item['qa_stratum'],
            'image_id': sid,
            'role': 'development_fit_train',
            'macro_id': item['macro_id'],
            'subcluster_id': item['subcluster_id'],
            'ground_rel': item['ground_rel'],
            'aerial_rel': item['aerial_rel'],
            'geometry': item['geometry'],
            'ground_qa': g_info,
            'aerial_qa': a_info,
        })

    if decode_count != QA_MAX_PAIRS * 2:
        raise ValueError(f'Expected {QA_MAX_PAIRS * 2} image decodes, got {decode_count}')

    return {
        'schema': 'cv050-fit-train-qa-verification/v1',
        'task_id': 'CV-050',
        'passed': True,
        'sampled_pairs': len(rows),
        'image_decodes': decode_count,
        'decode_quota_max_pairs': QA_MAX_PAIRS,
        'decode_accounting': {
            'fit_train_decodes': decode_count,
            'development_validation_decodes': 0,
            'isolated_confirmation_blind_decodes': 0,
            'blocked_or_quarantined_decodes': 0,
            'pl_decodes': 0,
            'non_fit_train_decodes': 0,
        },
        'all_nonblank': all(r['ground_qa']['quality']['nonblank'] and r['aerial_qa']['quality']['nonblank'] for r in rows),
        'allowed_ground_sizes': [list(s) for s in sorted(ALLOWED_SE_GROUND_SIZES)],
        'allowed_aerial_sizes': [list(s) for s in sorted(ALLOWED_SE_AERIAL_SIZES)],
        'sampled_stratum_counts': stratum_counts,
        'sampled_ground_size_counts': ground_size_counts,
        'discrepancy_summary': {
            'development_fit_train': fit_summary,
            'development_validation': val_summary,
            'blind_set_inspected': False,
        },
        'sampled_pair_reports': rows,
    }


def build_stage1_baseline_table(fit_train: list[dict], dev_val: list[dict]) -> dict:
    """Fit statistical priors exclusively on fit_train and evaluate across dev splits."""
    baselines = build_statistical_baselines(fit_train)
    fit_eval = evaluate_baseline_suite(fit_train, baselines)
    val_eval = evaluate_baseline_suite(dev_val, baselines)
    return {
        'schema': 'cv050-stage1-statistical-baselines/v1',
        'task_id': 'CV-050',
        'fitted_on': 'development_fit_train',
        'fit_train_count': len(fit_train),
        'dev_val_count': len(dev_val),
        'blind_set_evaluated': False,
        'blind_set_note': 'isolated_confirmation_blind (28 pairs) stays sealed; only static AlwaysCell12=39.29% from CV-049 exists.',
        'metric_parameters': {
            'soft_target_sigma_m': SOFT_TARGET_SIGMA_M,
            'interior_margin_threshold_m': INTERIOR_MARGIN_THRESHOLD_M,
            'large_discrepancy_threshold_m': LARGE_DISCREPANCY_THRESHOLD_M,
            'resolution_m': DEFAULT_RESOLUTION_M,
            'strata': list(STRATUM_NAMES),
        },
        'fitted_prior_specifications': baselines,
        'evaluations': {
            'development_fit_train': fit_eval,
            'development_validation': val_eval,
        },
    }


def build_cv051_frozen_protocol(baseline_report: dict, qa_report: dict) -> dict:
    """Freeze the pre-registered CV-051 single-step visual baseline experiment protocol."""
    val_all = baseline_report['evaluations']['development_validation']
    c12_val = val_all['AlwaysCell12Prior']['metrics']
    emp_val = val_all['TrainEmpiricalPrior']['metrics']

    return {
        'schema': 'cv051-single-step-visual-protocol/v1',
        'frozen_by_task': 'CV-050',
        'target_task': 'CV-051',
        'task_definition': 'Single-Step Cross-View 25-Cell Target Localization: (ground RGB, 625x625 aerial RGB) -> 25-cell probability distribution.',
        'ground_aspect_ratio_shift_disclosure': {
            'fit_train_sizes': qa_report['discrepancy_summary']['development_fit_train']['ground_size_histogram'],
            'dev_val_sizes': qa_report['discrepancy_summary']['development_validation']['ground_size_histogram'],
            'preprocessing_mandate': (
                'All visual encoders must apply an aspect-neutral geometric normalization '
                '(either centered square crop to min(W,H) followed by bicubic resize to 224x224, '
                'or zero-pad letterbox evaluated identically across fit_train and dev_val), '
                'and must report validation metrics stratified by native ground size (952x714 vs 952x532). '
                'Raw pixel width/height metadata is strictly forbidden as model input.'
            ),
            'aerial_native_tile': '625x625 PNG untouched center-preserved (zero padding and shifted crops forbidden).',
        },
        'label_and_loss_specification': {
            'primary_pose_namespace': 'mast3rpnp',
            'gaussian_soft_target_sigma_m': SOFT_TARGET_SIGMA_M,
            'primary_training_loss': 'KL(q_sigma5m || p_pred) or soft cross-entropy over 25 cell centers',
            'discrepancy_rationale': {
                'fit_train_disagree_chebyshev_le_1': qa_report['discrepancy_summary']['development_fit_train']['disagree_topology']['chebyshev_le_1_rate'],
                'fit_train_disagree_median_margin_m': qa_report['discrepancy_summary']['development_fit_train']['edge_margin_comparison_m']['disagree_pairs']['quantiles_m']['median'],
                'dev_val_disagree_chebyshev_le_1': qa_report['discrepancy_summary']['development_validation']['disagree_topology']['chebyshev_le_1_rate'],
            },
        },
        'admitted_local_backbones_for_cv051': {
            'zero_shot_and_frozen_pair': {
                'ground_encoder': 'models/clip_vit_b32_public_v1',
                'aerial_encoder': 'models/Sat2Cap',
                'adapter_module': 'project/src/agents/frozen_crossview_encoder_v2.py',
                'execution_constraint': 'CPU single live slot (max_live_encoders=1), frozen SHA-checked weights',
            },
            'probe_models': [
                'Frozen Zero-Shot CLIP(ground CLS) x Sat2Cap(5x5 cell-pooled aerial tokens) cosine baseline',
                'Lightweight trainable 25-cell spatial readout head trained strictly on development_fit_train (435 pairs)',
            ],
        },
        'evaluation_strata': list(STRATUM_NAMES) + ['ground_size_952x714', 'ground_size_952x532'],
        'dev_val_statistical_floors_to_beat': {
            'AlwaysCell12Prior': {
                'all_pairs_top1': c12_val['all_pairs']['top1_acc'],
                'all_pairs_top3': c12_val['all_pairs']['top3_acc'],
                'all_pairs_expected_planar_error_m': c12_val['all_pairs']['expected_planar_error_m'],
                'agree_top1': c12_val['opensfm_cell_agree']['top1_acc'],
                'interior_top1': c12_val['interior_margin_ge_2m']['top1_acc'],
            },
            'TrainEmpiricalPrior': {
                'all_pairs_top1': emp_val['all_pairs']['top1_acc'],
                'all_pairs_top3': emp_val['all_pairs']['top3_acc'],
                'all_pairs_expected_planar_error_m': emp_val['all_pairs']['expected_planar_error_m'],
                'all_pairs_soft_target_kl_nats': emp_val['all_pairs']['soft_target_kl_nats'],
                'agree_top1': emp_val['opensfm_cell_agree']['top1_acc'],
                'interior_top1': emp_val['interior_margin_ge_2m']['top1_acc'],
            },
        },
        'promotion_and_stopping_criteria': [
            'Gate G1 (Spatial Signal over Chance/Fixed Prior on Top-1): beat AlwaysCell12Prior Top-1 strictly on development_validation all_pairs (> 42.24%) AND opensfm_cell_agree AND interior_margin_ge_2m.',
            'Gate G2 (Distributional Advantage over TrainEmpiricalPrior): strictly improve BOTH soft_target_kl_nats (< TrainEmpiricalPrior KL) AND expected_planar_error_m (< TrainEmpiricalPrior error_m) AND top3_acc (> TrainEmpiricalPrior top3_acc) on development_validation.',
            'Gate G3 (Aspect-Ratio Invariance): no catastrophic collapse on either 952x532 or 952x714 aspect slices.',
            'Gate G4 (Image-Ablation Control): paired visual model must strictly beat ground-disabled / shuffled-aerial / mean-aerial controls by >= +3 percentage points or lower expected planar error on dev_val before requesting blind confirmation.',
            'Stopping Rule: if frozen + lightweight spatial head variants fail G1..G4 on dev_val, record honest negative result and DO NOT touch isolated_confirmation_blind (28 pairs).',
        ],
        'strict_prohibitions': [
            'Do not inspect, tune thresholds on, or decode isolated_confirmation_blind (28 pairs) during CV-051 development.',
            'Do not admit blocked_known_overlap (351), quarantined_cv045_vicinity (12), or PL (419) into training or validation.',
            'Do not tune soft-target sigma or grid boundaries after viewing development_validation scores.',
        ],
    }


def render_bundle_readme(qa_report: dict, baseline_report: dict) -> str:
    ft_d = qa_report['discrepancy_summary']['development_fit_train']
    dv_d = qa_report['discrepancy_summary']['development_validation']
    ft_ev = baseline_report['evaluations']['development_fit_train']
    dv_ev = baseline_report['evaluations']['development_validation']
    return f"""# OpenCVL 野测开发集分歧抽检与单步基线对照 v1（`CV-050`）

本目录保存 `CV-050` 对 `development_fit_train`（`435` 对）执行的受限 `24` 对（`48` 图）画质/分歧抽检核验、开发集（`fit_train 435 + dev_val 116`）位姿分歧几何分析、4 个零图像统计基线分层对照以及面向 `CV-051` 的单步视觉实验冻结协议。**全程 `development_validation`（116 对）零像素解码、`isolated_confirmation_blind`（28 对）零读取零评测、`blocked/quarantined/PL` 零解码**。

## 核心实证结论

1. **抽检 24 对（48 图）100% 通过 `nonblank` 与原生尺寸核验**：
   - 解码总数 `{qa_report['image_decodes']}` 张（`development_fit_train` 独占），主/副实现通道均值与标准差完全一致（最大差 `< 1e-6`）；
   - 同时发现 SE 地面图包含 `952×714` (4:3) 与 `952×532` (16:9) 两种合法原生画幅，且 `fit_train`（59.5% 为 `952×714`）与 `dev_val`（87.9% 为 `952×532`）存在明显画幅迁移，已纳入 `CV-051` 协议强约束。
2. **`opensfm != mast3rpnp` 分歧主要由 20m 网格边界量化效应驱动**：
   - `fit_train` 109 对分歧中 `{ft_d['disagree_topology']['chebyshev_le_1_count']}/109`（`{ft_d['disagree_topology']['chebyshev_le_1_rate']*100:.2f}%`）落在 8 邻域（`Chebyshev <= 1`），分歧对距 20m 网格线中位边距仅 `{ft_d['edge_margin_comparison_m']['disagree_pairs']['quantiles_m']['median']:.2f} m`（一致对 `{ft_d['edge_margin_comparison_m']['agree_pairs']['quantiles_m']['median']:.2f} m`）；
   - `dev_val` 50 对分歧中 `{dv_d['disagree_topology']['chebyshev_le_1_count']}/50`（`{dv_d['disagree_topology']['chebyshev_le_1_rate']*100:.2f}%`）落在 8 邻域。

## 单步 25 格统计基线对照总表（`all_pairs`）

| 基线模型（零图像输入） | `fit_train` Top-1 | `fit_train` Top-3 | `fit_train` KL | `dev_val` Top-1 | `dev_val` Top-3 | `dev_val` Cheb-1 | `dev_val` 期望误差 | `dev_val` KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Uniform25Prior` (`4%` 随机均值) | `{ft_ev['Uniform25Prior']['metrics']['all_pairs']['true_cell_prob_mean']*100:.2f}%`* | — | `{ft_ev['Uniform25Prior']['metrics']['all_pairs']['soft_target_kl_nats']:.3f}` | `{dv_ev['Uniform25Prior']['metrics']['all_pairs']['true_cell_prob_mean']*100:.2f}%`* | — | — | `{dv_ev['Uniform25Prior']['metrics']['all_pairs']['expected_planar_error_m']:.2f} m` | `{dv_ev['Uniform25Prior']['metrics']['all_pairs']['soft_target_kl_nats']:.3f}` |
| `UniformCenter9Prior` (`11.11%` 随机均值) | `{ft_ev['UniformCenter9Prior']['metrics']['all_pairs']['true_cell_prob_mean']*100:.2f}%`* | — | `{ft_ev['UniformCenter9Prior']['metrics']['all_pairs']['soft_target_kl_nats']:.3f}` | `{dv_ev['UniformCenter9Prior']['metrics']['all_pairs']['true_cell_prob_mean']*100:.2f}%`* | — | — | `{dv_ev['UniformCenter9Prior']['metrics']['all_pairs']['expected_planar_error_m']:.2f} m` | `{dv_ev['UniformCenter9Prior']['metrics']['all_pairs']['soft_target_kl_nats']:.3f}` |
| `AlwaysCell12Prior`（必猜中心 `12`） | `{ft_ev['AlwaysCell12Prior']['metrics']['all_pairs']['top1_acc']*100:.2f}%` | `{ft_ev['AlwaysCell12Prior']['metrics']['all_pairs']['top3_acc']*100:.2f}%` | `{ft_ev['AlwaysCell12Prior']['metrics']['all_pairs']['soft_target_kl_nats']:.3f}` | `{dv_ev['AlwaysCell12Prior']['metrics']['all_pairs']['top1_acc']*100:.2f}%` | `{dv_ev['AlwaysCell12Prior']['metrics']['all_pairs']['top3_acc']*100:.2f}%` | `{dv_ev['AlwaysCell12Prior']['metrics']['all_pairs']['chebyshev1_acc']*100:.2f}%` | `{dv_ev['AlwaysCell12Prior']['metrics']['all_pairs']['expected_planar_error_m']:.2f} m` | `{dv_ev['AlwaysCell12Prior']['metrics']['all_pairs']['soft_target_kl_nats']:.3f}` |
| `TrainEmpiricalPrior`（`fit_train` 拟合） | `{ft_ev['TrainEmpiricalPrior']['metrics']['all_pairs']['top1_acc']*100:.2f}%` | `{ft_ev['TrainEmpiricalPrior']['metrics']['all_pairs']['top3_acc']*100:.2f}%` | `{ft_ev['TrainEmpiricalPrior']['metrics']['all_pairs']['soft_target_kl_nats']:.3f}` | `{dv_ev['TrainEmpiricalPrior']['metrics']['all_pairs']['top1_acc']*100:.2f}%` | `{dv_ev['TrainEmpiricalPrior']['metrics']['all_pairs']['top3_acc']*100:.2f}%` | `{dv_ev['TrainEmpiricalPrior']['metrics']['all_pairs']['chebyshev1_acc']*100:.2f}%` | `{dv_ev['TrainEmpiricalPrior']['metrics']['all_pairs']['expected_planar_error_m']:.2f} m` | `{dv_ev['TrainEmpiricalPrior']['metrics']['all_pairs']['soft_target_kl_nats']:.3f}` |

注：`Uniform25Prior` / `UniformCenter9Prior` 在 `true_cell_prob_mean` 口径下为无信息随机猜测均值（`4.00%` / `11.11%`），其确定性排序详情见 [单步定位统计基线对照_v1.json](单步定位统计基线对照_v1.json)。

## 文件清单

- [抽检24对画质与分歧核验_v1.json](抽检24对画质与分歧核验_v1.json)：24 对（48 图）PNG 头部、RGB 哈希、双实现画质门限与全量开发集边界分歧几何统计。
- [单步定位统计基线对照_v1.json](单步定位统计基线对照_v1.json)：4 个零图像统计基线在 `fit_train`（435 对）与 `dev_val`（116 对）及 4 个分层子集上的完整对照指标。
- [单步视觉实验冻结协议_v1.json](单步视觉实验冻结协议_v1.json)：面向 `CV-051` 的候选视觉骨干、双画幅预处理规范、高斯软标签、晋级门槛与停止条件。
- [核验/源码与输入绑定_v1.json](核验/源码与输入绑定_v1.json) · [核验/另实现复核_v1.json](核验/另实现复核_v1.json)：上游 `CV-049`、生成源码与独立复核哈希绑定。
"""


def update_source_binding(out_dir: Path = OUT_DIR, root: Path = ROOT) -> dict:
    """Write or refresh 核验/源码与输入绑定_v1.json including optional audit report."""
    (out_dir / '核验').mkdir(parents=True, exist_ok=True)
    cv049_dir = root / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1'
    cv049_binding = load_json(cv049_dir / '核验/源码与输入绑定_v1.json')

    upstream_sha = {
        rel: file_sha256(root / rel)
        for rel in UPSTREAM_REQUIRED_BINDINGS
    }
    upstream_sha['DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/核验/源码与输入绑定_v1.json'] = file_sha256(
        cv049_dir / '核验/源码与输入绑定_v1.json'
    )

    generated = []
    for fname in ARTIFACT_FILENAMES:
        p = out_dir / fname
        if p.is_file():
            generated.append({
                'path': p.relative_to(root).as_posix(),
                'sha256': file_sha256(p),
                'bytes': p.stat().st_size,
            })

    audit_path = out_dir / '核验/另实现复核_v1.json'
    audit_entry = (
        {
            'path': audit_path.relative_to(root).as_posix(),
            'sha256': file_sha256(audit_path),
            'bytes': audit_path.stat().st_size,
        }
        if audit_path.is_file()
        else None
    )

    binding = {
        'schema': 'cv050-source-input-binding/v1',
        'task': 'CV-050',
        'upstream_cv049_binding_schema': cv049_binding['schema'],
        'upstream_frozen_sha256': upstream_sha,
        'generator_sources': [
            {
                'path': p,
                'sha256': file_sha256(root / p),
                'bytes': (root / p).stat().st_size,
            }
            for p in CV050_CODE_PATHS
        ],
        'generated_artifacts': generated,
        'alternate_audit_artifact': audit_entry,
        'execution_accounting': {
            'sampled_pairs': QA_MAX_PAIRS,
            'image_decodes': QA_MAX_PAIRS * 2,
            'fit_train_decodes': QA_MAX_PAIRS * 2,
            'dev_val_decodes': 0,
            'blind_decodes': 0,
            'model_calls': 0,
            'training': 0,
            'confirmation_consumed': 0,
        },
    }
    bpath = out_dir / '核验/源码与输入绑定_v1.json'
    bpath.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return binding


def build_and_save_bundle(*, save: bool, root: Path = ROOT, out_dir: Path = OUT_DIR) -> dict:
    _, ledger, _ = verify_cv049_integrity(root)
    fit_train, dev_val = extract_allowed_dev_records(ledger)
    qa_report = run_bounded_fit_train_qa(fit_train, dev_val, root / 'DATA/raw_data/OpenCVL/OpenCVL/mapillary/in_the_wild_test')
    baseline_report = build_stage1_baseline_table(fit_train, dev_val)
    cv051_protocol = build_cv051_frozen_protocol(baseline_report, qa_report)
    readme_text = render_bundle_readme(qa_report, baseline_report)

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / '核验').mkdir(parents=True, exist_ok=True)
        (out_dir / 'README.md').write_text(readme_text, encoding='utf-8')
        files_to_write = {
            '抽检24对画质与分歧核验_v1.json': qa_report,
            '单步定位统计基线对照_v1.json': baseline_report,
            '单步视觉实验冻结协议_v1.json': cv051_protocol,
        }
        for name, obj in files_to_write.items():
            (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        update_source_binding(out_dir=out_dir, root=root)

    val_emp = baseline_report['evaluations']['development_validation']['TrainEmpiricalPrior']['metrics']['all_pairs']
    val_c12 = baseline_report['evaluations']['development_validation']['AlwaysCell12Prior']['metrics']['all_pairs']
    return {
        'task': 'CV-050',
        'passed': True,
        'saved': save,
        'sampled_pairs': qa_report['sampled_pairs'],
        'image_decodes': qa_report['image_decodes'],
        'non_fit_train_decodes': qa_report['decode_accounting']['non_fit_train_decodes'],
        'all_nonblank': qa_report['all_nonblank'],
        'sampled_ground_size_counts': qa_report['sampled_ground_size_counts'],
        'dev_val_always_cell12': val_c12,
        'dev_val_train_empirical': val_emp,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true', help='Write CV-050 QA and baseline bundle to DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/')
    args = parser.parse_args(argv)
    summary = build_and_save_bundle(save=args.save)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
