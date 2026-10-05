"""DEV-001门槛导航器与校准逻辑的合成单元测试（不发任何模型或网络请求）。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from agents.massgis_threshold_navigator_v1 import thresholded_cue_choice


def probs(correct=0.9, second=0.05):
    """五类概率：correct指向方向0，second给方向1，not_adjacent占其余。"""
    second = min(second, (1.0 - correct) / 2)
    rest = (1.0 - correct - second) / 3.0
    return np.array([correct, second, rest, rest, rest], np.float64)


class ThresholdChoice(unittest.TestCase):
    def test_accepts_when_confident(self):
        cue, reason = thresholded_cue_choice(probs(0.9), 0.5, (1, 1), [12], 5)
        self.assertEqual((cue, reason), ('up', 'accepted'))

    def test_abstains_low_confidence_at_high_threshold(self):
        cue, reason = thresholded_cue_choice(probs(0.6), 0.95, (1, 1), [12], 5)
        self.assertIsNone(cue)
        self.assertEqual(reason, 'low_confidence')

    def test_not_adjacent_class_never_accepts(self):
        values = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
        cue, reason = thresholded_cue_choice(values, 0.5, (1, 1), [], 5)
        self.assertIsNone(cue)
        self.assertEqual(reason, 'not_adjacent')

    def test_boundary_destination_abstains(self):
        cue, reason = thresholded_cue_choice(probs(0.99), 0.5, (0, 0), [0], 5)  # up出界
        self.assertIsNone(cue)
        self.assertEqual(reason, 'illegal_top_direction')

    def test_visited_destination_abstains(self):
        # 位置(1,1)=格6，up到达格1；把1标记为已访问应弃权
        cue, reason = thresholded_cue_choice(probs(0.99), 0.5, (1, 1), [12, 1], 5)
        self.assertIsNone(cue)
        self.assertEqual(reason, 'visited_top_destination')

    def test_threshold_bounds_enforced(self):
        with self.assertRaises(ValueError):
            thresholded_cue_choice(probs(0.99), 0.45, (1, 1), [], 5)
        with self.assertRaises(ValueError):
            thresholded_cue_choice(probs(0.99), 1.05, (1, 1), [], 5)

    def test_null_threshold_abstains(self):
        cue, reason = thresholded_cue_choice(probs(0.99), None, (1, 1), [], 5)
        self.assertIsNone(cue)
        self.assertEqual(reason, 'uncalibrated_abstain')

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            thresholded_cue_choice(np.array([0.5, 0.5, 0.5, 0.5, 0.5]), 0.5, (1, 1), [], 5)

    def test_threshold_050_matches_scaled_choice_semantics(self):
        # 与冻结scaled_cue_choice在0.50处语义一致：接受/拒绝原因分布相同
        cases = [(probs(0.9), 'accepted'), (probs(0.3), 'low_confidence')]
        for values, expected in cases:
            cue, reason = thresholded_cue_choice(values, 0.5, (1, 1), [12], 5)
            self.assertEqual(reason, expected)


class CalibrationSpec(unittest.TestCase):
    def test_frozen_grid_and_targets(self):
        from eval.massgis_threshold_calibration_v1 import TAU_GRID, PRECISION_TARGET, RECALL_FLOOR
        self.assertEqual(TAU_GRID[0], 0.50)
        self.assertEqual(TAU_GRID[-1], 0.95)
        self.assertEqual(len(TAU_GRID), 10)
        self.assertEqual(PRECISION_TARGET, 0.90)
        self.assertEqual(RECALL_FLOOR, 0.90)


if __name__ == '__main__':
    unittest.main(verbosity=2)
