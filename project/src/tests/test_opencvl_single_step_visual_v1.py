"""Unit and bundle-audit tests for CV-051 Stage-2 OpenCVL single-step visual baseline."""
import unittest

import numpy as np

from project.src.agents.opencvl_single_step_visual_v1 import (
    CENTER9_CELLS,
    _assert_stage2_entry,
    build_grid_overlap_matrix,
    evaluate_stage2_gates,
    evaluate_stage2_predictions_6strata,
    fit_and_evaluate_stage2_models,
    preprocess_aerial_full_and_center9,
    preprocess_ground_aspect_neutral,
)
from project.src.eval.audit_opencvl_single_step_visual_v1 import (
    OUT_DIR,
    ROOT,
    alt_overlap_matrix,
    audit_stage2_bundle,
)
from project.src.tests.test_opencvl_wild_qa_v1 import make_se_record


def _l2_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class OpenCVLSingleStepVisualTests(unittest.TestCase):
    def test_grid_overlap_matrices(self):
        for dst in (5, 3):
            w = build_grid_overlap_matrix(7, dst)
            alt = alt_overlap_matrix(7, dst)
            self.assertEqual(w.shape, (dst * dst, 49))
            self.assertTrue(np.all(w >= 0.0))
            self.assertLessEqual(float(np.max(np.abs(w - alt))), 1e-12)
            self.assertLessEqual(float(np.max(np.abs(w.sum(axis=1) - 1.0))), 1e-12)
        # Central cell should map cleanly to central patch 24 with symmetry
        w5 = build_grid_overlap_matrix(7, 5)
        self.assertGreater(w5[12, 24], 0.35)
        self.assertAlmostEqual(w5[0, 0], w5[24, 48], places=12)

    def test_aspect_neutral_and_center9_preprocessing(self):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        rng = np.random.default_rng(7)
        for h in (714, 532):
            img_a = np.zeros((h, 952, 3), dtype=np.uint8)
            img_b = img_a.copy()
            s = min(952, h)
            left = (952 - s) // 2
            center_patch = rng.integers(0, 255, size=(s, s, 3), dtype=np.uint8)
            img_a[:, left : left + s, :] = center_patch
            img_b[:, left : left + s, :] = center_patch
            # Modify outer margins outside centered square; output must be bitwise identical.
            img_b[:, :left, :] = 255
            img_b[:, left + s :, :] = 128
            out_a = preprocess_ground_aspect_neutral(img_a, mean, std)
            out_b = preprocess_ground_aspect_neutral(img_b, mean, std)
            self.assertEqual(out_a.shape, (3, 224, 224))
            self.assertEqual(out_a.dtype, np.float32)
            np.testing.assert_array_equal(out_a, out_b)

        with self.assertRaises(ValueError):
            preprocess_ground_aspect_neutral(np.zeros((600, 800, 3), dtype=np.uint8), mean, std)

        aerial = np.zeros((625, 625, 3), dtype=np.uint8)
        aerial[125:500, 125:500, 1] = 200
        full_t, c9_t = preprocess_aerial_full_and_center9(aerial)
        self.assertEqual(full_t.shape, (3, 224, 224))
        self.assertEqual(c9_t.shape, (3, 224, 224))
        # Center-9 contains pure green=200 whereas full tile includes black border
        self.assertGreater(float(c9_t[1].mean()), float(full_t[1].mean()))

    def test_stage2_role_isolation_guards(self):
        for bad_role in (
            'isolated_confirmation_blind',
            'quarantined_cv045_vicinity',
            'blocked_known_overlap',
            'pl_pending_visual_dedup',
        ):
            bad = make_se_record('x1', role=bad_role)
            bad['country'] = 'SE'
            with self.assertRaises(ValueError):
                _assert_stage2_entry(bad)
            with self.assertRaises(ValueError):
                evaluate_stage2_predictions_6strata([bad], {'x1': [1.0 / 25.0] * 25}, {'x1': list(range(25))})

        pl_country = make_se_record('x2', role='development_fit_train')
        pl_country['country'] = 'PL'
        with self.assertRaises(ValueError):
            _assert_stage2_entry(pl_country)

        blind_flag = make_se_record('x3', role='development_validation', blind_do_not_inspect=True)
        blind_flag['country'] = 'SE'
        with self.assertRaises(ValueError):
            _assert_stage2_entry(blind_flag)

    def test_synthetic_visual_signal_recovery_and_ablation_separation(self):
        rng = np.random.default_rng(42)
        n_tr, n_va = 60, 40
        fit_entries = []
        dev_entries = []
        all_cells = []

        def make_entry_at_cell(sid: str, role: str, macro: str, cell: int, height: int):
            r, c = divmod(cell, 5)
            cx, cy = c * 125.0 + 62.5, r * 125.0 + 62.5
            rec = make_se_record(sid, role=role, macro_id=macro, xm=cx, ym=cy, xo=cx, yo=cy, height=height)
            rec['country'] = 'SE'
            return rec

        for i in range(n_tr):
            cell = CENTER9_CELLS[i % len(CENTER9_CELLS)]
            fit_entries.append(make_entry_at_cell(f'tr_{i}', 'development_fit_train', f'M{(i % 3) + 1:02d}', cell, 714 if i % 2 == 0 else 532))
            all_cells.append(cell)
        for i in range(n_va):
            cell = CENTER9_CELLS[i % len(CENTER9_CELLS)]
            dev_entries.append(make_entry_at_cell(f'va_{i}', 'development_validation', 'M02', cell, 532 if i < 30 else 714))
            all_cells.append(cell)

        n_tot = n_tr + n_va
        g_cls = np.zeros((n_tot, 512), dtype=np.float32)
        g_bands = np.zeros((n_tot, 3, 512), dtype=np.float32)
        a_cls = _l2_rows(rng.normal(size=(n_tot, 512))).astype(np.float32)
        a_25 = _l2_rows(rng.normal(size=(n_tot, 25, 512))).astype(np.float32)
        a_c_cls = _l2_rows(rng.normal(size=(n_tot, 512))).astype(np.float32)
        a_c9 = _l2_rows(rng.normal(size=(n_tot, 9, 512))).astype(np.float32)

        # Inject clean visual match between ground CLS/bands and the true target cell
        for i, cell in enumerate(all_cells):
            pos9 = CENTER9_CELLS.index(cell)
            base_dir = _l2_rows(rng.normal(size=(1, 512)))[0].astype(np.float32)
            a_25[i, cell] = base_dir
            a_c9[i, pos9] = base_dir
            g_cls[i] = _l2_rows((base_dir + 0.05 * rng.normal(size=512))[None, :])[0].astype(np.float32)
            for b in range(3):
                g_bands[i, b] = _l2_rows((base_dir + 0.05 * rng.normal(size=512))[None, :])[0].astype(np.float32)

        feats = {
            'image_ids': np.array([e['image_id'] for e in fit_entries + dev_entries]),
            'roles': np.array([e['role'] for e in fit_entries + dev_entries]),
            'ground_cls_512': g_cls,
            'ground_bands_512': g_bands,
            'aerial_cls_512': a_cls,
            'aerial_full25_512': a_25,
            'aerial_center_cls_512': a_c_cls,
            'aerial_center9_512': a_c9,
        }

        res = fit_and_evaluate_stage2_models(fit_entries, dev_entries, feats)
        va = res['evaluations']['development_validation']
        m1_acc = va['M1_ZeroShot_FullTile_Cosine']['all_pairs']['top1_acc']
        m2c_acc = va['M2c_MultiScale_PriorCalibrated_CV051b']['all_pairs']['top1_acc']
        self.assertGreater(m1_acc, 0.90)
        self.assertGreater(m2c_acc, 0.90)
        for ctrl in ('Control_GroundZeroed', 'Control_AerialShuffled', 'Control_MeanVisual'):
            self.assertGreater(m2c_acc - va[ctrl]['all_pairs']['top1_acc'], 0.40)

    def test_six_stratum_evaluation_and_g1_g4_gates(self):
        recs = [
            make_se_record('a1', role='development_validation', xm=187.5, ym=187.5, xo=187.5, yo=187.5, height=714),
            make_se_record('a2', role='development_validation', xm=437.5, ym=437.5, xo=437.5, yo=437.5, height=532),
        ]
        for r in recs:
            r['country'] = 'SE'
        good_probs = {'a1': [1.0 if k == 6 else 0.0 for k in range(25)], 'a2': [1.0 if k == 18 else 0.0 for k in range(25)]}
        good_ranks = {
            'a1': [6] + [k for k in range(25) if k != 6],
            'a2': [18] + [k for k in range(25) if k != 18],
        }
        m_good = evaluate_stage2_predictions_6strata(recs, good_probs, good_ranks)
        self.assertEqual(m_good['ground_size_952x714']['count'], 1)
        self.assertEqual(m_good['ground_size_952x532']['count'], 1)
        self.assertAlmostEqual(m_good['all_pairs']['top1_acc'], 1.0)

        p12 = {'a1': [1.0 if k == 12 else 0.0 for k in range(25)], 'a2': [1.0 if k == 12 else 0.0 for k in range(25)]}
        r12 = {sid: [12] + [k for k in range(25) if k != 12] for sid in ('a1', 'a2')}
        m_prior = evaluate_stage2_predictions_6strata(recs, p12, r12)
        val_map = {
            'AlwaysCell12Prior': m_prior,
            'TrainEmpiricalPrior': m_prior,
            'M2c': m_good,
            'Control_GroundZeroed': m_prior,
            'Control_AerialShuffled': m_prior,
            'Control_MeanVisual': m_prior,
        }
        gate_pass = evaluate_stage2_gates(val_map, 'M2c')
        self.assertTrue(gate_pass['stage2_passed'])
        self.assertEqual(gate_pass['decision'], 'PROMOTE_TO_STAGE3_BLIND_CONFIRMATION')

        val_fail = dict(val_map, M2c=m_prior)
        gate_fail = evaluate_stage2_gates(val_fail, 'M2c')
        self.assertFalse(gate_fail['stage2_passed'])
        self.assertEqual(gate_fail['decision'], 'STOP_AT_STAGE2_NEGATIVE_AND_PROCEED_TO_STAGE4_CLOSURE')

    def test_frozen_cv051_bundle_passes_independent_audit(self):
        binding_path = OUT_DIR / '核验/源码与输入绑定_v1.json'
        if not binding_path.is_file():
            self.skipTest('CV-051 bundle has not been saved yet')
        audit = audit_stage2_bundle(OUT_DIR, ROOT)
        self.assertTrue(audit['passed'])
        self.assertTrue(all(audit['checks'].values()))
        self.assertEqual(audit['verified_counts']['cached_pairs'], 551)
        self.assertEqual(audit['blind_decodes'], 0)
        self.assertLessEqual(audit['max_overlap_diff'], 1e-12)
        self.assertLessEqual(audit['max_metric_diff'], 1e-9)


if __name__ == '__main__':
    unittest.main()
