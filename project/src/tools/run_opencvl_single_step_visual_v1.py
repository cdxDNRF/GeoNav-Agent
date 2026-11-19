"""Generate CV-051 Stage-2 single-step visual localization & ablation bundle.

Strictly enforces:
- Verification of upstream CV-049/CV-050 frozen digests and local CLIP+Sat2Cap SHA-256s;
- Single live CPU encoder slot (max_live_encoders=1) decoding ONLY development_fit_train
  (435 pairs) and development_validation (116 pairs);
- Zero image decodes and zero evaluation on isolated_confirmation_blind (28 pairs);
- All subspaces, priors, scalers, CV choices, and weights fitted solely on fit_train.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from project.src.agents.opencvl_single_step_visual_v1 import (
    STAGE2_STRATA,
    extract_stage2_dual_encoder_features,
    fit_and_evaluate_stage2_models,
    verify_encoder_assets,
)
from project.src.tools.run_opencvl_wild_qa_v1 import (
    extract_allowed_dev_records,
    file_sha256,
    load_json,
    verify_cv049_integrity,
)

ROOT = Path(__file__).resolve().parents[3]
RAW_BASE = ROOT / 'DATA/raw_data/OpenCVL/OpenCVL/mapillary/in_the_wild_test'
OUT_DIR = ROOT / 'DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1'
CACHE_FILENAME = '开发集冻结双塔特征缓存_v1.npz'

CV051_CODE_PATHS = (
    'project/src/agents/opencvl_single_step_visual_v1.py',
    'project/src/tools/run_opencvl_single_step_visual_v1.py',
    'project/src/eval/audit_opencvl_single_step_visual_v1.py',
    'project/src/tests/test_opencvl_single_step_visual_v1.py',
)

ARTIFACT_FILENAMES = (
    'README.md',
    CACHE_FILENAME,
    '单步视觉定位模型与消融结果_v1.json',
    '四阶段科研路线与阶段二判定_v1.json',
)


def verify_cv050_integrity(root: Path = ROOT) -> tuple[dict, dict, dict, dict]:
    """Verify both CV-049 and CV-050 frozen SHA-256 bindings before Stage 2 runs."""
    _, ledger, cv049_proto = verify_cv049_integrity(root)
    cv050_dir = root / 'DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1'
    b50 = load_json(cv050_dir / '核验/源码与输入绑定_v1.json')
    for rel, expected in b50['upstream_frozen_sha256'].items():
        if file_sha256(root / rel) != expected:
            raise ValueError(f'CV-050 upstream binding mismatch: {rel}')
    items = list(b50['generator_sources']) + list(b50['generated_artifacts'])
    if b50.get('alternate_audit_artifact') is not None:
        items.append(b50['alternate_audit_artifact'])
    for item in items:
        p = root / item['path']
        if not p.is_file() or p.stat().st_size != item['bytes'] or file_sha256(p) != item['sha256']:
            raise ValueError(f'CV-050 frozen file mismatch: {item["path"]}')
    cv051_proto = load_json(cv050_dir / '单步视觉实验冻结协议_v1.json')
    return b50, ledger, cv049_proto, cv051_proto


def build_four_stage_roadmap(stage2_result: dict, extraction_audit: dict) -> dict:
    """Build structured 4-stage OpenCVL research roadmap & Stage-2 gate verdict."""
    primary = stage2_result['primary_stage2_gate']
    dev_val = stage2_result['evaluations']['development_validation']
    fit_train = stage2_result['evaluations']['development_fit_train']

    def compact_row(split_eval: dict, name: str) -> dict:
        m = split_eval[name]['all_pairs']
        return {
            'top1_acc': m['top1_acc'],
            'top3_acc': m['top3_acc'],
            'top5_acc': m['top5_acc'],
            'chebyshev1_acc': m['chebyshev1_acc'],
            'expected_planar_error_m': m['expected_planar_error_m'],
            'soft_target_kl_nats': m['soft_target_kl_nats'],
        }

    model_order = [
        'Uniform25Prior',
        'UniformCenter9Prior',
        'AlwaysCell12Prior',
        'TrainEmpiricalPrior',
        'M1_ZeroShot_FullTile_Cosine',
        'M1b_ZeroShot_Center9Crop_Cosine',
        'M2a_SpatialHead_PureVisual_NoPrior',
        'M2b_SpatialHead_PriorCalibrated_Full',
        'M2c_MultiScale_PriorCalibrated_CV051b',
        'Control_GroundZeroed',
        'Control_AerialShuffled',
        'Control_MeanVisual',
    ]

    return {
        'schema': 'opencvl-four-stage-research-roadmap/v1',
        'task_id': 'CV-051',
        'blind_set_evaluated': False,
        'blind_pairs_sealed': 28,
        'cv045_reserved_pairs_touched': 0,
        'extraction_accounting': extraction_audit,
        'stages': [
            {
                'stage': 'Stage 1',
                'tasks': 'CV-048 ~ CV-050',
                'status': 'done',
                'pass_gate': 'Isolated SE clusters (435/116/28), 24-pair fit_train QA nonblank, statistical floors locked',
            },
            {
                'stage': 'Stage 2',
                'tasks': 'CV-051',
                'status': 'completed_in_this_run',
                'pass_gate': 'Strictly satisfy G1-G4 on development_validation (116 pairs)',
                'result_passed': primary['stage2_passed'],
                'decision': primary['decision'],
            },
            {
                'stage': 'Stage 3',
                'tasks': 'CV-052 (conditional)',
                'status': 'ready_for_one_shot' if primary['stage2_passed'] else 'not_triggered_by_stopping_rule',
                'pass_gate': 'One-shot blind 28-pair Top-1 > 39.29% + B=3/5 outer navigation beats CenterCross',
            },
            {
                'stage': 'Stage 4',
                'tasks': 'CV-053 / FINAL-001',
                'status': 'pending' if primary['stage2_passed'] else 'next_recommended',
                'pass_gate': 'Integrate aerial positive findings & cross-view empirical evidence into V5.0 & final report',
            },
        ],
        'cv_selected_reference_model': stage2_result['cv_selected_reference_model'],
        'primary_stage2_gate': primary,
        'gate_evaluations': stage2_result['gate_evaluations'],
        'summary_comparison_all_pairs': {
            'development_validation': {m: compact_row(dev_val, m) for m in model_order},
            'development_fit_train': {m: compact_row(fit_train, m) for m in model_order},
        },
    }


def render_stage2_readme(results_doc: dict, roadmap_doc: dict) -> str:
    dev_val = results_doc['evaluations']['development_validation']
    primary = roadmap_doc['primary_stage2_gate']
    verdict = '通过 G1–G4，准入阶段三单次盲测' if primary['stage2_passed'] else '未全部通过 G1–G4，按停止条件定性收束（严禁触碰 28 对盲测集）'
    lines = [
        '# OpenCVL 阶段二单步视觉定位基线与消融结果（`CV-051`）',
        '',
        f'- **阶段二总判定**：`{primary["decision"]}`（{verdict}）',
        f'- **主评估候选模型（5 折 `fit_train` CV 选定）**：`{roadmap_doc["cv_selected_reference_model"]}`',
        f'- **盲测保护**：`blind_set_evaluated = {roadmap_doc["blind_set_evaluated"]}`（`{roadmap_doc["blind_pairs_sealed"]}` 对盲测确认集 0 解码/0 评测）',
        '',
        '## `development_validation`（116 对）核心对照总表（`all_pairs`）',
        '',
        '| 模型 / 对照组 | Top-1 | Top-3 | Top-5 | Cheb-1 | 期望误差 (m) | Soft-KL ($\\sigma=5\\text{m}$) | `952×714` Top-1 | `952×532` Top-1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in (
        'AlwaysCell12Prior',
        'TrainEmpiricalPrior',
        'M1_ZeroShot_FullTile_Cosine',
        'M1b_ZeroShot_Center9Crop_Cosine',
        'M2a_SpatialHead_PureVisual_NoPrior',
        'M2b_SpatialHead_PriorCalibrated_Full',
        'M2c_MultiScale_PriorCalibrated_CV051b',
        'Control_GroundZeroed',
        'Control_AerialShuffled',
        'Control_MeanVisual',
    ):
        all_m = dev_val[name]['all_pairs']
        m714 = dev_val[name]['ground_size_952x714']
        m532 = dev_val[name]['ground_size_952x532']
        lines.append(
            f'| `{name}` | `{all_m["top1_acc"]*100:.2f}%` | `{all_m["top3_acc"]*100:.2f}%` | '
            f'`{all_m["top5_acc"]*100:.2f}%` | `{all_m["chebyshev1_acc"]*100:.2f}%` | '
            f'`{all_m["expected_planar_error_m"]:.2f}` | `{all_m["soft_target_kl_nats"]:.3f}` | '
            f'`{m714["top1_acc"]*100:.2f}%` | `{m532["top1_acc"]*100:.2f}%` |'
        )
    lines.extend([
        '',
        '## 门禁明细（`G1–G4`）',
        f'- `G1_top1_beats_AlwaysCell12`: `{primary["G1_top1_beats_AlwaysCell12"]}`',
        f'- `G2_distributional_and_error_advantage`: `{primary["G2_distributional_and_error_advantage"]}`',
        f'- `G3_aspect_ratio_no_collapse`: `{primary["G3_aspect_ratio_no_collapse"]}`',
        f'- `G4_visual_ablation_gain`: `{primary["G4_visual_ablation_gain"]}`',
    ])
    return '\n'.join(lines) + '\n'


def update_source_binding(out_dir: Path = OUT_DIR, root: Path = ROOT, extraction_audit: dict | None = None) -> dict:
    """Write or update 核验/源码与输入绑定_v1.json for CV-051."""
    (out_dir / '核验').mkdir(parents=True, exist_ok=True)
    cv050_dir = root / 'DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1'
    cv050_binding = load_json(cv050_dir / '核验/源码与输入绑定_v1.json')
    cv050_upstream = {
        'DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/单步视觉实验冻结协议_v1.json': file_sha256(
            cv050_dir / '单步视觉实验冻结协议_v1.json'
        ),
        'DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/单步定位统计基线对照_v1.json': file_sha256(
            cv050_dir / '单步定位统计基线对照_v1.json'
        ),
        'DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/核验/源码与输入绑定_v1.json': file_sha256(
            cv050_dir / '核验/源码与输入绑定_v1.json'
        ),
        'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/核验/评估器专用冻结台账_禁止入模.json': file_sha256(
            root / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/核验/评估器专用冻结台账_禁止入模.json'
        ),
    }

    encoder_assets = verify_encoder_assets(root)
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

    prev_binding_path = out_dir / '核验/源码与输入绑定_v1.json'
    if extraction_audit is None and prev_binding_path.is_file():
        extraction_audit = load_json(prev_binding_path).get('extraction_accounting')

    binding = {
        'schema': 'cv051-source-input-binding/v1',
        'task': 'CV-051',
        'upstream_cv050_binding_schema': cv050_binding['schema'],
        'upstream_frozen_sha256': cv050_upstream,
        'encoder_asset_sha256': encoder_assets,
        'generator_sources': [
            {
                'path': p,
                'sha256': file_sha256(root / p),
                'bytes': (root / p).stat().st_size if (root / p).is_file() else 0,
            }
            for p in CV051_CODE_PATHS
            if (root / p).is_file()
        ],
        'generated_artifacts': generated,
        'alternate_audit_artifact': audit_entry,
        'extraction_accounting': extraction_audit,
        'blind_set_evaluated': False,
        'blind_decodes': 0,
    }
    prev_binding_path.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return binding


def build_and_save_stage2_bundle(
    *,
    save: bool,
    reuse_cache: bool = False,
    root: Path = ROOT,
    out_dir: Path = OUT_DIR,
) -> dict:
    _, ledger, _, _ = verify_cv050_integrity(root)
    fit_train, dev_val = extract_allowed_dev_records(ledger)
    dev_entries = fit_train + dev_val

    t0 = time.perf_counter()
    cache_path = out_dir / CACHE_FILENAME
    binding_path = out_dir / '核验/源码与输入绑定_v1.json'
    if reuse_cache and cache_path.is_file() and binding_path.is_file():
        prev = load_json(binding_path)
        expected_cache = next((a for a in prev.get('generated_artifacts', []) if a['path'].endswith(CACHE_FILENAME)), None)
        if expected_cache is None or file_sha256(cache_path) != expected_cache['sha256']:
            raise ValueError('Existing feature cache failed SHA-256 verification')
        with np.load(cache_path, allow_pickle=False) as zf:
            feats = {k: zf[k] for k in zf.files}
        extraction_audit = dict(prev['extraction_accounting'])
        extraction_audit['cache_reused'] = True
    else:
        feats, extraction_audit = extract_stage2_dual_encoder_features(
            dev_entries,
            root / 'DATA/raw_data/OpenCVL/OpenCVL/mapillary/in_the_wild_test',
            root,
            cache_path=cache_path if save else None,
        )
        extraction_audit['cache_reused'] = False
    extract_sec = time.perf_counter() - t0
    extraction_audit['extraction_wall_sec'] = round(extract_sec, 3)

    t1 = time.perf_counter()
    stage2_res = fit_and_evaluate_stage2_models(fit_train, dev_val, feats)
    fit_sec = time.perf_counter() - t1

    results_doc = {
        'schema': 'cv051-stage2-visual-evaluation/v1',
        'task_id': 'CV-051',
        'fit_train_count': len(fit_train),
        'dev_val_count': len(dev_val),
        'blind_set_evaluated': False,
        'blind_pairs_sealed': 28,
        'strata': list(STAGE2_STRATA),
        'fitting_wall_sec': round(fit_sec, 3),
        'extraction_accounting': extraction_audit,
        'cv_selected_reference_model': stage2_res['cv_selected_reference_model'],
        'model_hyperparams': stage2_res['model_hyperparams'],
        'evaluations': stage2_res['evaluations'],
        'gate_evaluations': stage2_res['gate_evaluations'],
        'primary_stage2_gate': stage2_res['primary_stage2_gate'],
        'predictions': stage2_res['predictions'],
    }
    roadmap_doc = build_four_stage_roadmap(stage2_res, extraction_audit)
    readme_text = render_stage2_readme(results_doc, roadmap_doc)

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / '核验').mkdir(parents=True, exist_ok=True)
        (out_dir / 'README.md').write_text(readme_text, encoding='utf-8')
        (out_dir / '单步视觉定位模型与消融结果_v1.json').write_text(
            json.dumps(results_doc, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        (out_dir / '四阶段科研路线与阶段二判定_v1.json').write_text(
            json.dumps(roadmap_doc, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        update_source_binding(out_dir=out_dir, root=root, extraction_audit=extraction_audit)

    return {
        'task': 'CV-051',
        'passed_execution': True,
        'saved': save,
        'extraction_wall_sec': extraction_audit['extraction_wall_sec'],
        'fitting_wall_sec': round(fit_sec, 3),
        'cv_selected_reference_model': stage2_res['cv_selected_reference_model'],
        'primary_stage2_gate': stage2_res['primary_stage2_gate'],
        'dev_val_summary': roadmap_doc['summary_comparison_all_pairs']['development_validation'],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true', help='Save frozen Stage-2 visual bundle to DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1/')
    parser.add_argument('--reuse-cache', action='store_true', help='Reuse SHA-verified 开发集冻结双塔特征缓存_v1.npz instead of re-encoding')
    args = parser.parse_args(argv)
    summary = build_and_save_stage2_bundle(save=args.save, reuse_cache=args.reuse_cache)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
