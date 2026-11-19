"""Independent alternate audit for CV-051 Stage-2 visual localization & ablations.

Zero PyTorch import and zero image pixel decoding inside the auditor:
- Verifies upstream CV-049/CV-050 hashes, local CLIP+Sat2Cap weight hashes, and
  all CV-051 source/artifact/cache SHA-256 digests;
- Verifies the 551-pair .npz feature cache shapes, finiteness, unit L2 norms,
  single-slot tensor invariance (max_live_encoders=1), and exact 2D patch-to-cell
  overlap matrices within <= 1e-12;
- Independently recomputes all 6-stratum metrics and G1-G4 gate checks across
  development_fit_train (435) and development_validation (116) within <= 1e-9,
  proving zero blind-set evaluation or ID leakage.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from project.src.eval.audit_opencvl_wild_qa_v1 import (
    alt_geometry,
    read_json,
    sha,
)

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / 'DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1'
CACHE_FILENAME = '开发集冻结双塔特征缓存_v1.npz'


def alt_overlap_matrix(src_grid: int, dst_grid: int) -> np.ndarray:
    """Alternate overlap calculation using explicit edge intervals."""
    s_edges = [i / src_grid for i in range(src_grid + 1)]
    d_edges = [i / dst_grid for i in range(dst_grid + 1)]
    out = np.zeros((dst_grid * dst_grid, src_grid * src_grid), dtype=np.float64)
    cell_area = (1.0 / dst_grid) ** 2
    for dr in range(dst_grid):
        for dc in range(dst_grid):
            di = dr * dst_grid + dc
            for sr in range(src_grid):
                oy = min(d_edges[dr + 1], s_edges[sr + 1]) - max(d_edges[dr], s_edges[sr])
                if oy <= 0.0:
                    continue
                for sc in range(src_grid):
                    ox = min(d_edges[dc + 1], s_edges[sc + 1]) - max(d_edges[dc], s_edges[sc])
                    if ox > 0.0:
                        out[di, sr * src_grid + sc] = (oy * ox) / cell_area
    return out


def _alt_eval_6strata(recs: list[dict], probs_map: dict, ranks_map: dict) -> dict[str, dict]:
    geos = [alt_geometry(r) for r in recs]
    by_id_probs = [probs_map[r['image_id']] for r in recs]
    by_id_ranks = [ranks_map[r['image_id']] for r in recs]
    # Recompute sample-level exact metrics independently without relying on qa module evaluators.
    import math

    centers = [((k % 5) * 125.0 + 62.5, (k // 5) * 125.0 + 62.5) for k in range(25)]
    inv_2v = 0.5 / 25.0
    clip = 1e-12
    buckets: dict[str, list[dict]] = {
        'all_pairs': [],
        'opensfm_cell_agree': [],
        'interior_margin_ge_2m': [],
        'large_discrepancy_gt_10m': [],
        'ground_size_952x714': [],
        'ground_size_952x532': [],
    }
    for rec, g, probs, rank in zip(recs, geos, by_id_probs, by_id_ranks):
        if abs(sum(probs) - 1.0) > 1e-5 or len(probs) != 25 or set(rank) != set(range(25)):
            raise ValueError(f'Invalid prediction vector for {rec["image_id"]}')
        top1 = rank[0]
        r1, c1 = divmod(top1, 5)
        rm, cm = divmod(g['cell_m'], 5)
        dr, dc = abs(r1 - rm), abs(c1 - cm)
        dists = [math.hypot(cx - g['xm'], cy - g['ym']) * g['res'] for cx, cy in centers]
        raw_q = [math.exp(-d * d * inv_2v) for d in dists]
        zq = sum(raw_q)
        soft_q = [w / zq for w in raw_q]
        row = {
            'top1': int(g['cell_m'] == top1),
            'top3': int(g['cell_m'] in set(rank[:3])),
            'top5': int(g['cell_m'] in set(rank[:5])),
            'cheb1': int(max(dr, dc) <= 1),
            'l1_1': int((dr + dc) <= 1),
            'prob': probs[g['cell_m']],
            'center_err': dists[top1],
            'exp_err': sum(p * d for p, d in zip(probs, dists)),
            'ce': -sum(q * math.log(max(p, clip)) for q, p in zip(soft_q, probs)),
            'kl': sum(q * math.log(max(q, clip) / max(p, clip)) for q, p in zip(soft_q, probs) if q > 0.0),
        }
        buckets['all_pairs'].append(row)
        if g['agree']:
            buckets['opensfm_cell_agree'].append(row)
        if g['margin_m'] >= 2.0:
            buckets['interior_margin_ge_2m'].append(row)
        if g['diff_m'] > 10.0:
            buckets['large_discrepancy_gt_10m'].append(row)
        g_size = f'{rec["ground_asset"]["width"]}x{rec["ground_asset"]["height"]}'
        if g_size == '952x714':
            buckets['ground_size_952x714'].append(row)
        elif g_size == '952x532':
            buckets['ground_size_952x532'].append(row)

    out = {}
    for k, rows in buckets.items():
        n = len(rows)
        if n == 0:
            out[k] = {'count': 0}
            continue
        out[k] = {
            'count': n,
            'top1_acc': sum(r['top1'] for r in rows) / n,
            'top3_acc': sum(r['top3'] for r in rows) / n,
            'top5_acc': sum(r['top5'] for r in rows) / n,
            'chebyshev1_acc': sum(r['cheb1'] for r in rows) / n,
            'l1_1_acc': sum(r['l1_1'] for r in rows) / n,
            'true_cell_prob_mean': sum(r['prob'] for r in rows) / n,
            'top1_center_error_m': sum(r['center_err'] for r in rows) / n,
            'expected_planar_error_m': sum(r['exp_err'] for r in rows) / n,
            'soft_target_ce_nats': sum(r['ce'] for r in rows) / n,
            'soft_target_kl_nats': sum(r['kl'] for r in rows) / n,
        }
    return out


def _alt_eval_gates(val_eval: dict, cand_name: str) -> dict[str, bool]:
    cand = val_eval[cand_name]
    c12 = val_eval['AlwaysCell12Prior']
    emp = val_eval['TrainEmpiricalPrior']
    g1 = (
        cand['all_pairs']['top1_acc'] > c12['all_pairs']['top1_acc'] + 1e-12
        and cand['opensfm_cell_agree']['top1_acc'] > c12['opensfm_cell_agree']['top1_acc'] + 1e-12
        and cand['interior_margin_ge_2m']['top1_acc'] > c12['interior_margin_ge_2m']['top1_acc'] + 1e-12
    )
    g2 = (
        cand['all_pairs']['soft_target_kl_nats'] < emp['all_pairs']['soft_target_kl_nats'] - 1e-12
        and cand['all_pairs']['expected_planar_error_m'] < c12['all_pairs']['expected_planar_error_m'] - 1e-12
        and cand['all_pairs']['top3_acc'] > c12['all_pairs']['top3_acc'] + 1e-12
    )
    g3 = (
        cand['ground_size_952x714']['top1_acc'] >= c12['ground_size_952x714']['top1_acc'] - 0.05
        and cand['ground_size_952x714']['soft_target_kl_nats'] <= emp['ground_size_952x714']['soft_target_kl_nats'] + 0.05
        and cand['ground_size_952x532']['top1_acc'] >= c12['ground_size_952x532']['top1_acc'] - 0.05
        and cand['ground_size_952x532']['soft_target_kl_nats'] <= emp['ground_size_952x532']['soft_target_kl_nats'] + 0.05
    )
    g4 = True
    for cname in ('Control_GroundZeroed', 'Control_AerialShuffled', 'Control_MeanVisual'):
        ctrl = val_eval[cname]['all_pairs']
        ok = (
            cand['all_pairs']['top1_acc'] - ctrl['top1_acc'] >= 0.03 - 1e-12
            or ctrl['expected_planar_error_m'] - cand['all_pairs']['expected_planar_error_m'] >= 0.5 - 1e-12
        )
        g4 = g4 and ok
    return {
        'G1': g1,
        'G2': g2,
        'G3': g3,
        'G4': g4,
        'stage2_passed': g1 and g2 and g3 and g4,
    }


def audit_stage2_bundle(out_dir: Path = OUT_DIR, root: Path = ROOT) -> dict:
    """Run independent alternate audit on the frozen CV-051 Stage-2 visual bundle."""
    if 'torch' in sys.modules:
        raise ValueError('PyTorch must not be imported during Stage-2 audit')

    binding = read_json(out_dir / '核验/源码与输入绑定_v1.json')
    for rel, expected in binding['upstream_frozen_sha256'].items():
        if sha(root / rel) != expected:
            raise ValueError(f'Upstream SHA mismatch: {rel}')
    for asset in binding['encoder_asset_sha256']:
        p = root / asset['path']
        if sha(p) != asset['sha256'] or p.stat().st_size != asset['bytes']:
            raise ValueError(f'Encoder SHA mismatch: {asset["path"]}')
    for entry in binding['generator_sources'] + binding['generated_artifacts']:
        p = root / entry['path']
        if sha(p) != entry['sha256'] or p.stat().st_size != entry['bytes']:
            raise ValueError(f'CV-051 file digest mismatch: {entry["path"]}')
    if binding.get('alternate_audit_artifact') is not None:
        aa = binding['alternate_audit_artifact']
        ap = root / aa['path']
        if sha(ap) != aa['sha256'] or ap.stat().st_size != aa['bytes']:
            raise ValueError(f'Alternate audit artifact digest mismatch: {aa["path"]}')

    ledger = read_json(root / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/核验/评估器专用冻结台账_禁止入模.json')
    pub = read_json(root / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/公开洁净划分清单_v1.json')
    fit_train = [r for r in ledger['se_records'] if r['role'] == 'development_fit_train']
    dev_val = [r for r in ledger['se_records'] if r['role'] == 'development_validation']
    allowed_ids = {r['image_id'] for r in fit_train + dev_val}
    blind_ids = {item['image_id'] for item in pub['splits']['isolated_confirmation_blind']}

    results_doc = read_json(out_dir / '单步视觉定位模型与消融结果_v1.json')
    roadmap_doc = read_json(out_dir / '四阶段科研路线与阶段二判定_v1.json')
    if (
        results_doc['blind_set_evaluated'] is not False
        or roadmap_doc['blind_set_evaluated'] is not False
        or binding['blind_set_evaluated'] is not False
        or binding['blind_decodes'] != 0
    ):
        raise ValueError('Blind confirmation set must remain uninspected and unevaluated')

    ext = results_doc['extraction_accounting']
    if ext['max_live_encoders'] != 1 or ext['blind_decodes'] != 0 or ext['training_during_extraction'] != 0:
        raise ValueError('Extraction execution constraint violated')
    for chk in ext['frozen_tensor_checks']:
        if not chk['unchanged'] or chk['before_tensor_sha256'] != chk['after_tensor_sha256']:
            raise ValueError(f'Frozen tensor invariance failed: {chk}')

    # Scan JSON texts and .npz IDs for zero blind/forbidden leakage
    res_raw = (out_dir / '单步视觉定位模型与消融结果_v1.json').read_text(encoding='utf-8')
    map_raw = (out_dir / '四阶段科研路线与阶段二判定_v1.json').read_text(encoding='utf-8')
    for bid in blind_ids:
        if bid in res_raw or bid in map_raw:
            raise ValueError(f'Blind ID {bid} leaked into Stage-2 JSON deliverables')

    max_overlap_diff = 0.0
    max_norm_err = 0.0
    with np.load(out_dir / CACHE_FILENAME, allow_pickle=False) as zf:
        c_ids = [str(x) for x in zf['image_ids']]
        c_roles = [str(x) for x in zf['roles']]
        if len(c_ids) != 551 or set(c_ids) != allowed_ids or any(r not in {'development_fit_train', 'development_validation'} for r in c_roles):
            raise ValueError('Feature cache IDs or roles violated Stage-2 isolation')
        expected_shapes = {
            'ground_cls_512': (551, 512),
            'ground_bands_512': (551, 3, 512),
            'aerial_cls_512': (551, 512),
            'aerial_full25_512': (551, 25, 512),
            'aerial_center_cls_512': (551, 512),
            'aerial_center9_512': (551, 9, 512),
        }
        for key, exp_shape in expected_shapes.items():
            arr = zf[key]
            if tuple(arr.shape) != exp_shape or not np.isfinite(arr).all():
                raise ValueError(f'Invalid feature tensor {key}: shape={arr.shape}')
            norms = np.linalg.norm(arr.astype(np.float64), axis=-1)
            max_norm_err = max(max_norm_err, float(np.max(np.abs(norms - 1.0))))
        if max_norm_err > 1e-5:
            raise ValueError(f'Non-unit L2 feature vector norm deviation: {max_norm_err}')
        max_overlap_diff = max(
            float(np.max(np.abs(zf['w25_from_49'] - alt_overlap_matrix(7, 5)))),
            float(np.max(np.abs(zf['w9_from_49'] - alt_overlap_matrix(7, 3)))),
        )
        if max_overlap_diff > 1e-12:
            raise ValueError(f'2D overlap matrix diff > 1e-12: {max_overlap_diff}')

    # Recompute 6-stratum metrics independently from saved per-sample predictions
    max_metric_diff = 0.0
    recomputed_splits = {}
    for split_name, recs in (('development_fit_train', fit_train), ('development_validation', dev_val)):
        saved_eval = results_doc['evaluations'][split_name]
        preds = results_doc['predictions'][split_name]
        # Verify exact match for visual models + ablation controls via predictions
        recomputed_models = {}
        for mname, pred_obj in preds.items():
            alt_m = _alt_eval_6strata(recs, pred_obj['probs'], pred_obj['rankings'])
            recomputed_models[mname] = alt_m
            saved_m = saved_eval[mname]
            for sname, sdict in saved_m.items():
                adict = alt_m[sname]
                if adict['count'] != sdict['count']:
                    raise ValueError(f'Stratum count mismatch: {split_name}/{mname}/{sname}')
                if adict['count'] == 0:
                    continue
                for k, aval in adict.items():
                    if k == 'count':
                        continue
                    diff = abs(float(aval) - float(sdict[k]))
                    max_metric_diff = max(max_metric_diff, diff)
                    if diff > 1e-9:
                        raise ValueError(f'Metric mismatch {split_name}/{mname}/{sname}/{k}: {diff}')

        # Cross-check TrainEmpiricalPrior and AlwaysCell12Prior independently
        fit_counts = [0] * 25
        for r in fit_train:
            fit_counts[int(r['mast3rpnp']['index'])] += 1
        dist_sq = [((k % 5) - 2) ** 2 + ((k // 5) - 2) ** 2 for k in range(25)]
        emp_probs = [(fit_counts[k] + 0.5) / (len(fit_train) + 12.5) for k in range(25)]
        emp_rank = sorted(range(25), key=lambda k: (-fit_counts[k], dist_sq[k], k))
        c12_probs = [(1.0 - 1e-3) * (1.0 if k == 12 else 0.0) + 1e-3 / 25.0 for k in range(25)]
        c12_rank = sorted(range(25), key=lambda k: (dist_sq[k], k))
        for bname, bprobs, brank in (
            ('AlwaysCell12Prior', c12_probs, c12_rank),
            ('TrainEmpiricalPrior', emp_probs, emp_rank),
        ):
            p_map = {r['image_id']: bprobs for r in recs}
            r_map = {r['image_id']: brank for r in recs}
            alt_m = _alt_eval_6strata(recs, p_map, r_map)
            recomputed_models[bname] = alt_m
            for sname, sdict in saved_eval[bname].items():
                adict = alt_m[sname]
                if adict['count'] != sdict['count']:
                    raise ValueError(f'Baseline stratum count mismatch: {split_name}/{bname}/{sname}')
                if adict['count'] == 0:
                    continue
                for k, aval in adict.items():
                    if k == 'count':
                        continue
                    diff = abs(float(aval) - float(sdict[k]))
                    max_metric_diff = max(max_metric_diff, diff)
                    if diff > 1e-9:
                        raise ValueError(f'Baseline metric mismatch {split_name}/{bname}/{sname}/{k}: {diff}')
        recomputed_splits[split_name] = recomputed_models

    # Verify G1-G4 gate decisions independently
    for cname, saved_gate in results_doc['gate_evaluations'].items():
        alt_g = _alt_eval_gates(recomputed_splits['development_validation'], cname)
        if (
            alt_g['G1'] != saved_gate['G1_top1_beats_AlwaysCell12']
            or alt_g['G2'] != saved_gate['G2_distributional_and_error_advantage']
            or alt_g['G3'] != saved_gate['G3_aspect_ratio_no_collapse']
            or alt_g['G4'] != saved_gate['G4_visual_ablation_gain']
            or alt_g['stage2_passed'] != saved_gate['stage2_passed']
        ):
            raise ValueError(f'Gate verdict mismatch on {cname}')

    return {
        'schema': 'cv051-alternate-audit/v1',
        'task': 'CV-051',
        'passed': True,
        'checks': {
            'zero_torch_import': True,
            'upstream_cv049_cv050_hashes_match': True,
            'encoder_asset_hashes_match': True,
            'cv051_artifact_and_cache_hashes_match': True,
            'max_live_encoders_eq_1_and_tensors_unchanged': True,
            'cache_551_dev_only_and_unit_norms': True,
            'overlap_matrices_within_1e12': True,
            'six_stratum_metrics_within_1e9': True,
            'gates_g1_to_g4_match': True,
            'blind_confirmation_sealed_and_unleaked': True,
        },
        'verified_counts': {
            'fit_train_pairs': len(fit_train),
            'dev_val_pairs': len(dev_val),
            'cached_pairs': len(c_ids),
            'blind_pairs_sealed': len(blind_ids),
            'total_encoder_forwards': ext['total_forwards'],
        },
        'max_overlap_diff': max_overlap_diff,
        'max_feature_norm_err': max_norm_err,
        'max_metric_diff': max_metric_diff,
        'auditor_image_decodes': 0,
        'blind_decodes': 0,
        'blind_set_evaluated': False,
        'independent_personnel_review': False,
        'notes': 'same Antigravity session, independent interval-edge overlap & 6-stratum G1-G4 audit without torch',
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true', help='Write 核验/另实现复核_v1.json and update 核验/源码与输入绑定_v1.json')
    args = parser.parse_args(argv)
    result = audit_stage2_bundle(OUT_DIR, ROOT)
    if args.save:
        from project.src.tools.run_opencvl_single_step_visual_v1 import update_source_binding

        target = OUT_DIR / '核验/另实现复核_v1.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        update_source_binding(OUT_DIR, ROOT)
        result = audit_stage2_bundle(OUT_DIR, ROOT)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
