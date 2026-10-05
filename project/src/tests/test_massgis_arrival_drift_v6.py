import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "eval" / "diagnose_massgis_arrival_drift_v6.py"
SPEC = importlib.util.spec_from_file_location("massgis_arrival_drift_v6", MODULE)
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


def manifests():
    m = {10: {"grid_size": 10, "regions": []}, 15: {"grid_size": 15, "regions": []}}
    for k in (10, 15):
        cells = []
        for r in range(k):
            for c in range(k):
                x0 = 45000 + (c if k == 15 else c + 5) * 300
                y0 = 878000 + (14 - r) * 300
                shared = k == 15 and c >= 5 or k == 10
                shared_col = c - 5 if k == 15 else c
                cells.append({"cell": r * k + c, "bounds_m": [x0, y0, x0 + 300, y0 + 300],
                              "file_sha256": f"shared-{r}-{shared_col}" if shared else f"extra-{r}-{c}"})
        m[k]["regions"].append({"area": "img_1", "region_id": "mass2005_test", "bounds_m":
                                [45000, 878000, 49500, 882500] if k == 15 else [46500, 879500, 49500, 882500],
                                "cells": cells})
    return m


class DriftBoundaryTests(unittest.TestCase):
    def test_bounds_map_physical_cells_and_offset_sign(self):
        maps, transforms = DIAG.derive_maps(manifests())
        self.assertEqual(len(transforms["img_1"]["map"]), 100)
        self.assertEqual(transforms["img_1"]["region_id"], "mass2005_test")
        self.assertEqual(transforms["img_1"]["map"][17], 27)
        self.assertEqual(transforms["img_1"]["map"][0], 5)
        self.assertEqual((transforms["img_1"]["dx_m"], transforms["img_1"]["dy_m"]), (1500, 1500))
        self.assertEqual(maps[10]["img_1"]["byid"][17]["file_sha256"],
                         maps[15]["img_1"]["byid"][27]["file_sha256"])

    def test_common_pair_index_is_returned_as_pair_id_dictionary(self):
        left=[{"pair_id":"physical-a","cohort":"common","grid_size":10}]
        right=[{"pair_id":"physical-a","cohort":"common","grid_size":15}]
        pairs=DIAG.pair_task_rows(left,right)
        self.assertEqual(list(pairs),["physical-a"])
        self.assertEqual(pairs["physical-a"][0]["grid_size"],10)
        self.assertEqual(pairs["physical-a"][1]["grid_size"],15)
        with self.assertRaises(ValueError):
            DIAG.pair_task_rows(left,[{"pair_id":"different","cohort":"common"}])

    def test_first_pair_analysis_return_contract_is_summary_rows_and_pair_map(self):
        tree=ast.parse(MODULE.read_text(encoding="utf-8-sig"))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="first_pair_analysis")
        returns=[n.value for n in ast.walk(function) if isinstance(n,ast.Return)]
        self.assertEqual(len(returns),1)
        self.assertIsInstance(returns[0],ast.Tuple)
        self.assertEqual(len(returns[0].elts),3)

    def test_training_trace_hash_uses_bound_artifacts_index(self):
        key="Small256_NoTarget_s0/训练轨迹.npz"
        self.assertEqual(DIAG.registered_artifact_hash({"artifacts_sha256":{key:"abc"}},key),"abc")
        self.assertIsNone(DIAG.registered_artifact_hash({"files_sha256":{key:"wrong"}},key))
        audit=DIAG.jread(DIAG.TRAIN/"独立复核.json")
        self.assertIn(key,audit["artifacts_sha256"])

    def test_normalized_coordinate_delta_formula_sign(self):
        dr, dc = DIAG.normalized_coordinate_delta(1, 7)
        self.assertAlmostEqual(dr, -5 / 126)
        self.assertAlmostEqual(dc, 10 / 126)
        self.assertAlmostEqual(DIAG.normalized_coordinate_delta(5, 9)[1], 0.0)
        self.assertLess(DIAG.normalized_coordinate_delta(9, 2)[0], 0)

    def test_input_rebuild_dimension_and_bucket_shift(self):
        mean = np.ones(512, dtype=np.float32)
        current = np.zeros(512, dtype=np.float32)
        x10 = DIAG.policy_input(mean, current, (0, 5), 20, (5,), 10)
        x15 = DIAG.policy_input(mean, current, (0, 10), 20, (10,), 15)
        self.assertEqual(x10.shape, (1052,))
        self.assertEqual(x15.shape, (1052,))
        self.assertAlmostEqual(float(x10[1024]), 0.0)
        self.assertAlmostEqual(float(x10[1025]), 5 / 9)
        self.assertAlmostEqual(float(x10[1026]), 1.0)
        self.assertAlmostEqual(float(x15[1024]), 0.0)
        self.assertAlmostEqual(float(x15[1025]), 10 / 14)
        self.assertAlmostEqual(float(x15[1026]), 1.0)
        self.assertNotEqual(int(np.argmax(x10[1027:])), int(np.argmax(x15[1027:])))

    def test_visit_prefix_validation_rejects_bad_alignment(self):
        mean = np.zeros(512, dtype=np.float32)
        current = np.zeros(512, dtype=np.float32)
        with self.assertRaises(ValueError):
            DIAG.policy_input(mean, current, (1, 2), 19, (12,), 10)
        with self.assertRaises(ValueError):
            DIAG.policy_input(mean, current, (1, 2), 20, (13,), 10)

    def test_action_quality_uses_only_posthoc_manhattan_change(self):
        self.assertEqual(DIAG.action_quality((2, 2), (3, 2), "down", 10), "toward")
        self.assertEqual(DIAG.action_quality((2, 2), (3, 2), "up", 10), "away")
        self.assertEqual(DIAG.action_quality((0, 2), (3, 2), "up", 10), "side_or_wall")
        self.assertEqual(DIAG.action_quality((0, 0), (3, 3), "up", 10), "side_or_wall")

    def test_source_has_no_model_runtime_or_forward_api(self):
        tree = ast.parse(MODULE.read_text(encoding="utf-8-sig"))
        imports = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr.lower())
                elif isinstance(node.func, ast.Name):
                    calls.add(node.func.id.lower())
        self.assertFalse({"torch", "torchvision", "transformers"} & imports)
        self.assertFalse({"forward", "predict", "load_area_policy", "act"} & calls)


if __name__ == "__main__":
    unittest.main()

