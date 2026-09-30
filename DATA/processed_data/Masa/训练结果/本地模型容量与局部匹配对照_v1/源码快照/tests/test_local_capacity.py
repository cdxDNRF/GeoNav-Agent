import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.spatial_relation import make_policy, quadrant_features, cosine_relations
from env.environment import Observation
from train.dyncur_tiny import set_seed
from train.target_pairs import paired_records
from train.local_capacity import (bank_features, train, direction_result, compare,
                                 observation_features, summarize, ARCHITECTURES, CONDITIONS)


class LocalCapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(18)
        self.values = rng.normal(size=(25, 512)).astype(np.float32)
        self.local = {'img_0': rng.normal(size=(25, 4, 768)).astype(np.float32)}
        self.store = SimpleNamespace(data={'img_0': self.values}, patch=lambda area, cell: self.values[cell])
        self.bank = paired_records(['img_0'], 8, 42)
        self.mean = self.values.mean(0)
        self.local_mean = self.local['img_0'].mean(0)

    def features(self, condition):
        return bank_features(self.bank, self.store, self.local, self.mean, self.local_mean, 'Spatial256', condition)

    def test_quadrants_keep_spatial_order_and_cosine_is_scale_invariant(self):
        grid = torch.zeros(1, 7, 7, 768)
        grid[:, :3, :3] = 1; grid[:, :3, 3:] = 2
        grid[:, 3:, :3] = 3; grid[:, 3:, 3:] = 4
        self.assertEqual(quadrant_features(grid.reshape(1, 49, 768))[0, :, 0].tolist(), [1, 2, 3, 4])
        a, b = self.local['img_0'][:2]
        np.testing.assert_allclose(cosine_relations(a, b), cosine_relations(a * 2, b * 3), atol=1e-6)

    def test_spatial_initial_policy_matches_small_with_extra_channels_ignored(self):
        set_seed(0); small = make_policy('Small256')
        set_seed(0); spatial = make_policy('Spatial256')
        x = torch.tensor(self.features('Full')[0][:, 0, 0])
        with torch.no_grad():
            left = small.step(x[:, :1052])[0]; right = spatial.step(x)[0]
        self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))
        for name, tensor in small.state_dict().items():
            expected = spatial.state_dict()[name]
            self.assertTrue(torch.equal(tensor, expected[:, :1052] if name == 'input.0.weight' else expected))
        self.assertEqual(sum(p.numel() for p in spatial.parameters()) - sum(p.numel() for p in small.parameters()), 4096)

    def test_target_mask_removes_both_global_and_local_goal_information(self):
        full, labels = self.features('Full')
        masked, _ = self.features('NoTarget')
        np.testing.assert_array_equal(masked[:, 0], masked[:, 1])
        np.testing.assert_array_equal(masked[..., 512:1052], full[..., 512:1052])
        record = self.bank[0]
        record['goals'].reverse()
        changed, _ = self.features('NoTarget')
        np.testing.assert_array_equal(changed, masked)

    def test_swapping_goal_updates_relation_block_without_changing_public_history(self):
        full, labels = self.features('Full')
        swapped, swapped_labels = self.features('SwapCue')
        np.testing.assert_array_equal(full[:, ::-1, :, :512], swapped[..., :512])
        np.testing.assert_allclose(full[:, ::-1, :, 1052:], swapped[..., 1052:], atol=1e-7)
        np.testing.assert_array_equal(full[..., 512:1052], swapped[..., 512:1052])
        np.testing.assert_array_equal(labels, swapped_labels)

    def test_navigation_feature_builder_agrees_with_pair_training_input(self):
        features, _ = self.features('Full')
        rec = self.bank[0]
        current, goal = rec['history'][-1], rec['goals'][0]
        obs = Observation(b'current', b'target', divmod(current, 5), 5, 8, tuple(rec['history']))
        actual = observation_features(obs, self.values[goal], self.values[current], self.local['img_0'][goal], self.local['img_0'][current], 'Spatial256')
        np.testing.assert_array_equal(actual, features[0, 0, 2])

    def test_actual_training_learns_relation_channels_and_checkpoint_replays(self):
        x, y = [torch.tensor(v) for v in self.features('Full')]
        model = make_policy('Spatial256')
        initial = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            trained = train(model, x, y, folder, 0, torch.device('cpu'), epochs=2)
            self.assertGreater(float(trained.input[0].weight[:, 1052:].abs().sum()), 0)
            self.assertTrue(all(torch.equal(initial[n], w) for n, w in trained.state_dict().items() if n.startswith(('critic.', 'next_state.'))))
            replay = make_policy('Spatial256')
            replay.load_state_dict(torch.load(folder / 'model.pt', weights_only=True))
            result = direction_result(folder, 'toy', trained, replay, x, y, self.bank)
            self.assertTrue(result['audit']['checkpoint_replay'])
            log = [json.loads(s) for s in (folder / '训练日志.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(log[-1]['optimizer_steps'], 2)
            self.assertEqual(log[-1]['target_uses'], 32)

    def test_candidate_gate_keeps_effect_and_SG_but_CI_is_diagnostic(self):
        def item(sr, sg):
            return {'metrics': {'sr': sr, 'mean_sg_all_episodes': sg},
                    'by_source': {'a': {'sr': sr}, 'b': {'sr': sr}}}
        base = [item(.4, 2)] * 3
        self.assertTrue(compare([item(.43, 1.9)] * 3, base, 'sr', 'mean_sg_all_episodes')['observational_candidate'])
        self.assertFalse(compare([item(.41, 1.9)] * 3, base, 'sr', 'mean_sg_all_episodes')['observational_candidate'])
        self.assertFalse(compare([item(.43, 2.1)] * 3, base, 'sr', 'mean_sg_all_episodes')['observational_candidate'])

    def test_report_uses_navigation_metrics_contract(self):
        direction = {'metrics': {'direction_accuracy': .4, 'cross_entropy': 1.2},
                     'by_source': {'a': {'direction_accuracy': .4}, 'b': {'direction_accuracy': .4}}}
        nav = {'metrics': {'sr': .3, 'mean_sg_all_episodes': 2, 'repeat_visit_rate_micro': .05},
               'by_source': {'a': {'sr': .3}, 'b': {'sr': .3}}}
        results = {(a, s, c): {'fit': direction, 'held': direction, 'nav': nav} for a in ARCHITECTURES for s in range(3) for c in CONDITIONS}
        with tempfile.TemporaryDirectory() as temporary:
            result = summarize(Path(temporary), results, {'Frontier': {'sr': .5, 'mean_sg_all_episodes': 2}})
            self.assertEqual(result['models'], 18)
            self.assertFalse(result['formal_S2_passed'])


if __name__ == '__main__':
    unittest.main()
