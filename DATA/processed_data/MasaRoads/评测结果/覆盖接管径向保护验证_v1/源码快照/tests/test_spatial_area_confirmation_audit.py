"""Synthetic checks for independent spatial-confirmation audit equations."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.edge_cue import EdgeTargetCueHead, edge_features, image_profiles
from agents.spatial_relation import make_policy
from agents.target_cue import cue_features
from eval.audit_spatial_area_confirmation import (
    _manual_gru,
    expected_probe_wrong,
    expected_task_bank,
    joint_probabilities,
    manual_cue_features,
    manual_edge_features,
    manual_head_raw,
    manual_profile,
)


class SpatialAreaAuditMathTests(unittest.TestCase):
    def test_native_profile_and_seam_equations(self):
        rng = np.random.default_rng(419)
        current_rgb = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
        target_rgb = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
        current = manual_profile(current_rgb)
        target = manual_profile(target_rgb)
        np.testing.assert_array_equal(current, image_profiles(current_rgb))
        np.testing.assert_array_equal(target, image_profiles(target_rgb))
        independent_edge = manual_edge_features(target, current)
        reference_edge = edge_features(target, current)
        np.testing.assert_array_equal(independent_edge, reference_edge)
        self.assertEqual(independent_edge.tobytes(), reference_edge.tobytes())

    def test_cue_vector_and_decoupled_joint_probabilities(self):
        rng = np.random.default_rng(733)
        target = rng.normal(size=512).astype(np.float32)
        current = rng.normal(size=512).astype(np.float32)
        target_local = rng.normal(size=(4, 768)).astype(np.float32)
        current_local = rng.normal(size=(4, 768)).astype(np.float32)
        independent = manual_cue_features(target, current, target_local, current_local)
        reference = cue_features(target, current, target_local, current_local)
        np.testing.assert_array_equal(independent, reference)
        self.assertEqual(independent.shape, (1041,))
        up = joint_probabilities(np.asarray([8, 0, 0, 0, 8], dtype=np.float32))
        far = joint_probabilities(np.asarray([0, 0, 0, 0, -8], dtype=np.float32))
        self.assertAlmostEqual(float(up.sum()), 1.0, places=6)
        self.assertGreater(float(up[0]), .99)
        self.assertGreater(float(far[4]), .99)

    def test_manual_head_and_gru_match_frozen_architecture_equations(self):
        torch.manual_seed(91)
        agent = SimpleNamespace(device=torch.device('cpu'), explorer=make_policy('Small256').eval(),
                                head=EdgeTargetCueHead().eval())
        cue = np.random.default_rng(82).normal(size=1061).astype(np.float32)
        raw_manual = manual_head_raw(agent, cue)
        raw_module = agent.head.raw_logits(torch.from_numpy(cue)[None])[0].detach().numpy()
        np.testing.assert_array_equal(raw_manual, raw_module)

        state = np.random.default_rng(512).normal(size=1052).astype(np.float32)
        state[1024], state[1025] = 4/9, 6/9
        hidden = torch.randn(1, 256)
        manual_logits, manual_hidden = _manual_gru(agent, state, hidden.clone())
        module_logits, _, _, module_hidden = agent.explorer.step(torch.from_numpy(state)[None], hidden.clone())
        np.testing.assert_allclose(manual_logits, module_logits[0].detach().numpy(), rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(manual_hidden.detach().numpy(), module_hidden.detach().numpy(), rtol=2e-6, atol=2e-7)

    def test_regenerated_task_bank_and_probe_false_target_exclusions(self):
        region = dict(area='img_2000', split='dev', source_tile='MasaRoadsArea/dev_00')
        tasks, strata = expected_task_bank([region])
        self.assertEqual(len(tasks), 75)
        self.assertEqual(len({(t['start'], t['goal']) for t in tasks}), 75)
        self.assertEqual({s['stratum'] for s in strata.values()}, {'long_distance', 'seam_target', 'interior_target'})

        probes = [
            dict(probe_id=f'p{i}', area='img_2000', pair_id=7, current_cell=current, target_cell=45,
                 distance=abs(current//10-4)+abs(current%10-5))
            for i, current in enumerate((44, 55, 43))
        ]
        plan = expected_probe_wrong(probes)
        self.assertEqual(plan, expected_probe_wrong(probes))
        cues = {item['cue_cell'] for item in plan.values()}
        self.assertEqual(len(cues), 1)
        self.assertNotIn(45, cues)
        self.assertTrue(cues.isdisjoint({44, 55, 43}))


if __name__ == '__main__':
    unittest.main()
