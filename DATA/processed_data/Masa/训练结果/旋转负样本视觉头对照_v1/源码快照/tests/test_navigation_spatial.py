"""Protect public inputs, source isolation and recurrent PPO evidence contracts."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.spatial_relation import make_policy
from env.environment import Observation
from train.local_capacity import observation_features, lines
from train.local_capacity import navigation_run
from train.navigation_spatial import NavigationWorld, train
from train.dyncur_tiny import set_seed
from eval.audit_navigation_spatial import check_record, check_training
from test_foundation import Fixture


class NavigationTrainingTests(Fixture):
    def setUp(self):
        super().setUp()
        torch.set_num_threads(1)
        rng = np.random.default_rng(81)
        self.store = SimpleNamespace(data={a: rng.normal(size=(25, 512)).astype(np.float32) for a in ('img_0', 'img_1')})
        self.local = {a: rng.normal(size=(25, 4, 768)).astype(np.float32) for a in self.store.data}
        self.mean = self.store.data['img_0'].mean(0)
        self.local_mean = self.local['img_0'].mean(0)

    def world(self, architecture='Spatial256', condition='Full', count=1):
        return NavigationWorld(self.store, self.local, ['img_0'], self.mean, self.local_mean, architecture, condition, count, 0)

    def test_held_maps_are_absent_from_training_tables_and_samples(self):
        world = self.world(count=16)
        self.assertEqual(world.areas, ['img_0'])
        self.assertEqual(len(world.table), 1)
        self.assertEqual(len(world.local), 1)
        for _ in range(20):
            world.reset(np.ones(16, bool))
            self.assertTrue((world.area == 0).all())
            self.assertTrue(((world.initial_distance >= 4) & (world.initial_distance <= 8)).all())
        with self.assertRaises(ValueError):
            NavigationWorld(self.store, self.local, ['img_0', 'img_0'], self.mean, self.local_mean, 'Small256', 'Full', 1, 0)

    def test_vector_inputs_rewards_and_terminals_match_real_navigation(self):
        for architecture in ('Small256', 'Spatial256'):
            for condition in ('Full', 'NoTarget'):
                world = self.world(architecture, condition)
                env = self.env()
                ep = self.ep(start=0, goal=24)
                obs = env.reset(ep)
                world.position[:] = ep.start; world.goal[:] = ep.goal; world.steps[:] = 0
                world.initial_distance[:] = 8
                world.visits[:] = 0; world.visits[0, ep.start] = 1
                # Arrival on the last allowed step; includes a necessary backtrack in this test path.
                for action in ('right', 'left', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'):
                    target = self.mean if condition == 'NoTarget' else self.store.data['img_0'][24]
                    local_target = self.local_mean if condition == 'NoTarget' else self.local['img_0'][24]
                    current = int(world.position[0])
                    expected = observation_features(obs, target, self.store.data['img_0'][current], local_target,
                                                    self.local['img_0'][current], architecture)
                    np.testing.assert_array_equal(world.observe()[0], expected)
                    info = world.step(np.array([('up', 'right', 'down', 'left').index(action)]))
                    obs, _, actual = env.step(action)
                    self.assertEqual(int(world.position[0]), obs.position[0]*5 + obs.position[1])
                    self.assertEqual(bool(info['done'][0]), env.done)
                    self.assertEqual(bool(info['revisited'][0]), actual.revisited)
                self.assertTrue(info['success'][0])
                self.assertAlmostEqual(float(info['external'][0]), 1.1, places=6)

    def test_no_target_masks_both_global_and_local_channels(self):
        world = self.world(condition='NoTarget', count=2)
        world.position[:] = 7; world.steps[:] = 3; world.visits[:] = 0
        world.visits[:, 7] = 1; world.goal[:] = [0, 24]
        before = world.observe()
        np.testing.assert_array_equal(before[0], before[1])
        world.goal[:] = [22, 4]
        np.testing.assert_array_equal(before, world.observe())

    def test_unseen_images_and_goal_coordinates_cannot_change_fixed_public_input(self):
        world = self.world(count=1)
        world.position[:] = 0; world.goal[:] = 24
        before = world.observe().copy()
        world.table[:, 1:24] += 999
        world.local[:, 1:24] -= 555
        np.testing.assert_array_equal(before, world.observe())
        # Relocate the supplied target while preserving its visual features.
        world.table[:, 23] = world.table[:, 24]
        world.local[:, 23] = world.local[:, 24]
        world.goal[:] = 23
        np.testing.assert_array_equal(before, world.observe())

    def test_real_PPO_updates_actor_critic_and_local_channels_but_freezes_unused_head(self):
        set_seed(0)
        model = make_policy('Spatial256')
        initial = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            trained = train(model, self.world(count=16), folder, 0, torch.device('cpu'), updates=2)
            current = trained.state_dict()
            for prefix in ('actor.', 'critic.', 'gru.'):
                self.assertTrue(any(not torch.equal(v, current[n]) for n, v in initial.items() if n.startswith(prefix)))
            self.assertTrue(all(torch.equal(v, current[n]) for n, v in initial.items() if n.startswith('next_state.')))
            self.assertGreater(float(current['input.0.weight'][:, 1052:].abs().sum()), 0)
            logs = lines(folder / '训练日志.jsonl')
            self.assertEqual(logs[-1]['steps'], 320)
            self.assertEqual(logs[-1]['optimizer_steps'], 4)
            self.assertLessEqual(max(x['replay_logprob_max_error'] for x in logs), 1e-5)
            with np.load(folder / '训练轨迹.npz') as trace:
                self.assertEqual(trace['action'].shape, (20, 16))
                self.assertEqual(int(trace['wall'].sum()), 0)
                self.assertEqual(int(trace['success'].sum()), sum(x['successes'] for x in logs))
                self.assertTrue((trace['success'] == (trace['done'] & (trace['after'] == trace['goal']))).all())
            (folder / '配置.json').write_text(json.dumps({'seed': 0}), encoding='utf-8')
            checked = check_training(folder, ['img_0'], updates=2, count=16)
            self.assertEqual(checked['steps'], 320)
            replay = make_policy('Spatial256')
            replay.load_state_dict(torch.load(folder / 'model.pt', weights_only=True))
            x = torch.tensor(self.world().observe())
            with torch.no_grad():
                self.assertTrue(torch.equal(trained.step(x)[0], replay.step(x)[0]))

    def test_independent_navigation_audit_rejects_input_and_success_corruption(self):
        set_seed(0)
        model = make_policy('Spatial256').eval()
        ep = self.ep(start=0, goal=24)
        store = SimpleNamespace(data=self.store.data, patch=lambda area, cell: self.store.data[area][cell])
        record = navigation_run(model, 'Spatial256', 'Full', ep, self.env(), store, self.local,
                                self.mean, self.local_mean, {}, torch.device('cpu'))
        args = (asdict(ep), model, 'Spatial256', 'Full', store, self.local,
                self.mean, self.local_mean, {}, torch.device('cpu'))
        self.assertEqual(check_record(record, *args), record['steps'])
        changed = copy.deepcopy(record)
        changed['decisions'][0]['features_sha256'] = 'invalid'
        with self.assertRaises(AssertionError):
            check_record(changed, *args)
        changed = copy.deepcopy(record)
        changed['success'] = not changed['success']
        with self.assertRaises(AssertionError):
            check_record(changed, *args)


if __name__ == '__main__':
    unittest.main()
