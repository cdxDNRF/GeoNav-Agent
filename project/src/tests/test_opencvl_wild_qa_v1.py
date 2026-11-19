"""Unit and bundle-audit tests for CV-050 OpenCVL fit_train QA and Stage-1 baselines."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from project.src.data.opencvl_wild_qa_v1 import (
    QA_MAX_PAIRS,
    build_statistical_baselines,
    compute_pair_discrepancy_geometry,
    evaluate_baseline_suite,
    gaussian_soft_target_distribution,
    inspect_png_image_quality,
    select_fit_train_qa_pairs,
    summarize_split_discrepancy,
)
from project.src.eval.audit_opencvl_wild_qa_v1 import OUT_DIR, ROOT, audit_qa_bundle


def make_se_record(
    sid: str,
    *,
    role: str = 'development_fit_train',
    macro_id: str = 'M01',
    xm: float = 312.5,
    ym: float = 312.5,
    xo: float = 312.5,
    yo: float = 312.5,
    width: int = 952,
    height: int = 714,
    blind_do_not_inspect: bool = False,
    policy_admitted: bool | None = None,
) -> dict:
    def cell_of(x: float, y: float) -> dict:
        col = min(4, max(0, int(x // 125.0)))
        row = min(4, max(0, int(y // 125.0)))
        return {'x': x, 'y': y, 'row': row, 'col': col, 'index': row * 5 + col}

    m = cell_of(xm, ym)
    o = cell_of(xo, yo)
    rec = {
        'image_id': sid,
        'role': role,
        'macro_id': macro_id,
        'subcluster_id': 'C01',
        'resolution': 0.16,
        'mast3rpnp': m,
        'mapilopensfm': o,
        'opensfm_cell_agree': m['index'] == o['index'],
        'ground_asset': {'rel': f'SE/ground/{sid}.png', 'width': width, 'height': height, 'bytes': 1000},
        'aerial_asset': {'rel': f'SE/aerial/{sid}.png', 'width': 625, 'height': 625, 'bytes': 1000},
    }
    if blind_do_not_inspect:
        rec['blind_do_not_inspect'] = True
    if policy_admitted is not None:
        rec['policy_admitted'] = policy_admitted
    return rec


def make_synthetic_fit_train_pool() -> list[dict]:
    rows = []
    # 12 large-discrepancy disagree pairs (> 15 m, crossing from cell 12 to cell 7 or 2)
    for i in range(12):
        rows.append(make_se_record(
            f'100{i:02d}',
            macro_id='M01',
            xm=312.5,
            ym=260.0,
            xo=312.5,
            yo=260.0 - (110.0 + i * 2.0),
            width=952,
            height=714 if i % 2 == 0 else 532,
        ))
    # 12 boundary-straddling small-discrepancy (<= 5m) pairs across macros
    macros_b = ['M01', 'M03', 'M04', 'M05', 'M12']
    for i in range(12):
        rows.append(make_se_record(
            f'200{i:02d}',
            macro_id=macros_b[i % len(macros_b)],
            xm=312.5,
            ym=252.0 + (i % 3),
            xo=312.5,
            yo=245.0,
        ))
    # 8 interior consensus pairs across core macros (margin >= 3m)
    macros_c = ['M01', 'M03', 'M04', 'M05']
    for i in range(8):
        rows.append(make_se_record(
            f'300{i:02d}',
            macro_id=macros_c[i % len(macros_c)],
            xm=312.5,
            ym=312.5,
            xo=318.0,
            yo=310.0,
        ))
    return rows


class OpenCVLWildQATests(unittest.TestCase):
    def test_geometry_discrepancy_and_margin(self):
        rec = make_se_record('1', xm=255.0, ym=312.5, xo=245.0, yo=312.5)
        geo = compute_pair_discrepancy_geometry(rec)
        # x=255 is 5px (0.8m) from internal grid line 250; cell 12 vs 11 -> L1=1, Chebyshev=1
        self.assertAlmostEqual(geo['min_internal_edge_dist_px'], 5.0)
        self.assertAlmostEqual(geo['cell_edge_margin_m'], 0.8)
        self.assertAlmostEqual(geo['planar_diff_m'], 1.6)
        self.assertEqual((geo['cell_l1'], geo['cell_chebyshev']), (1, 1))
        self.assertFalse(geo['in_opensfm_cell_agree'])
        self.assertFalse(geo['in_interior_margin_ge_2m'])
        self.assertFalse(geo['in_large_discrepancy_gt_10m'])

    def test_isolation_guards_block_blind_and_non_dev_roles(self):
        for bad_role in (
            'isolated_confirmation_blind',
            'blocked_known_overlap',
            'quarantined_cv045_vicinity',
            'pl_pending_visual_dedup',
        ):
            rec = make_se_record('99', role=bad_role)
            with self.assertRaises(ValueError):
                compute_pair_discrepancy_geometry(rec)
            with self.assertRaises(ValueError):
                summarize_split_discrepancy([rec], bad_role)

        blind_flagged = make_se_record('98', role='development_fit_train', blind_do_not_inspect=True)
        with self.assertRaises(ValueError):
            compute_pair_discrepancy_geometry(blind_flagged)

        dev_val = [make_se_record('77', role='development_validation')]
        with self.assertRaises(ValueError):
            select_fit_train_qa_pairs(dev_val, max_pairs=1)
        with self.assertRaises(ValueError):
            build_statistical_baselines(dev_val)

    def test_qa_selector_quota_bounds_and_stratified_determinism(self):
        pool = make_synthetic_fit_train_pool()
        with self.assertRaises(ValueError):
            select_fit_train_qa_pairs(pool, max_pairs=0)
        with self.assertRaises(ValueError):
            select_fit_train_qa_pairs(pool, max_pairs=QA_MAX_PAIRS + 1)

        s1 = select_fit_train_qa_pairs(pool, max_pairs=24)
        s2 = select_fit_train_qa_pairs(list(reversed(pool)), max_pairs=24)
        self.assertEqual(len(s1), 24)
        self.assertEqual([x['image_id'] for x in s1], [x['image_id'] for x in s2])
        self.assertEqual(len({x['image_id'] for x in s1}), 24)

        strata = [x['qa_stratum'] for x in s1]
        self.assertEqual(strata.count('large_discrepancy_disagree'), 10)
        self.assertEqual(strata.count('boundary_straddling_disagree'), 10)
        self.assertEqual(strata.count('consensus_control'), 4)

    def test_png_quality_inspection_in_workspace_temp(self):
        temp_root = ROOT / 'temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmpdir:
            good_path = Path(tmpdir) / 'good.png'
            arr = np.zeros((32, 48, 3), dtype=np.uint8)
            arr[:, :24, 0] = 200
            arr[:, 24:, 1] = 120
            arr[16:, :, 2] = 60
            Image.fromarray(arr, mode='RGB').save(good_path)
            res = inspect_png_image_quality(good_path)
            self.assertTrue(res['quality']['nonblank'])
            self.assertLessEqual(res['max_moment_diff'], 1e-6)
            self.assertEqual((res['header']['width'], res['header']['height']), (48, 32))

            black_path = Path(tmpdir) / 'black.png'
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode='RGB').save(black_path)
            black_res = inspect_png_image_quality(black_path)
            self.assertFalse(black_res['quality']['nonblank'])

    def test_statistical_baselines_and_soft_target_distribution(self):
        soft = gaussian_soft_target_distribution(312.5, 312.5, resolution=0.16, sigma_m=5.0)
        self.assertEqual(len(soft), 25)
        self.assertAlmostEqual(sum(soft), 1.0, places=10)
        self.assertEqual(soft.index(max(soft)), 12)

        fit_rows = make_synthetic_fit_train_pool()
        baselines = build_statistical_baselines(fit_rows)
        for spec in baselines.values():
            self.assertAlmostEqual(sum(spec['probs']), 1.0, places=9)
            self.assertEqual(set(spec['ranking']), set(range(25)))

        val_rows = [
            make_se_record('v1', role='development_validation', xm=312.5, ym=312.5, xo=312.5, yo=312.5),
            make_se_record('v2', role='development_validation', xm=312.5, ym=190.0, xo=312.5, yo=190.0),
        ]
        ev = evaluate_baseline_suite(val_rows, baselines)
        self.assertAlmostEqual(ev['AlwaysCell12Prior']['metrics']['all_pairs']['top1_acc'], 0.5)
        self.assertAlmostEqual(ev['Uniform25Prior']['metrics']['all_pairs']['true_cell_prob_mean'], 0.04)

    def test_frozen_cv050_bundle_passes_independent_audit(self):
        binding_file = OUT_DIR / '核验/源码与输入绑定_v1.json'
        if not binding_file.is_file():
            self.skipTest('CV-050 bundle has not been saved yet')
        audit_res = audit_qa_bundle(OUT_DIR, ROOT)
        self.assertTrue(audit_res['passed'])
        self.assertTrue(all(audit_res['checks'].values()))
        self.assertEqual(audit_res['verified_counts']['sampled_qa_pairs'], 24)
        self.assertEqual(audit_res['auditor_image_decodes'], 0)
        self.assertLessEqual(audit_res['max_baseline_metric_diff'], 1e-9)


if __name__ == '__main__':
    unittest.main()
