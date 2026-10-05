"""Meaningful scope, public-input, geometry and termination regression tests."""
from copy import deepcopy
from dataclasses import fields
import inspect
import random
import unittest
import numpy as np
from agents.area_policy import area_policy_features, resolve_area_config, load_area_policy
from agents.scaled_edge_navigator import scaled_policy_features, scaled_cue_choice
from agents.parameterized_coverage import choose_area_coverage
from agents.coverage_radial_guard import choose_guarded_coverage_action
from env.area_protocol import GRID10, GRID15, get_protocol, validate_manifest
from env.parameterized_area import AreaEpisode, ParameterizedAreaEnv
from eval import area_compatibility as p
from eval.audit_area_compatibility import independent_plan, adjacency


class AreaCompatibilityTests(unittest.TestCase):
    def ep(self, start=0, goal=1):
        return AreaEpisode('fixture', 'dev', 'img_9900', start, goal,
            abs(start//15-goal//15)+abs(start%15-goal%15), 20, 15, GRID15, 'synthetic_engineering_only')

    def test_physical_area_increases_at_same_cell_scale(self):
        self.assertEqual(get_protocol(GRID10).area_km2, 9)
        self.assertEqual(get_protocol(GRID15).area_km2, 20.25)
        self.assertEqual(get_protocol(GRID15).cell_size_m, 300)

    def test_manifest_rejects_density_or_budget_changes(self):
        manifest = p.read(p.FIXTURE / '数据清单.json')
        for key, value in [('cell_size_m', 200), ('mosaic_pixels', 3000), ('budget', 30), ('pixel_size_m', True)]:
            changed = dict(manifest, **{key: value})
            with self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_unreachable_initial_goal_rejected(self):
        with self.assertRaises(ValueError):
            self.ep(0, 224).validate()

    def test_last_budget_step_can_succeed_and_oob_costs_budget(self):
        env = ParameterizedAreaEnv(p.FIXTURE)
        env.reset(self.ep())
        for _ in range(19):
            obs, done, info = env.step('up')
            self.assertTrue(info.out_of_bounds)
            self.assertFalse(done)
        self.assertEqual(obs.remaining_budget, 1)
        _, done, _ = env.step('right')
        self.assertTrue(done)
        result = env.evaluator_result()
        self.assertEqual((result['success'], result['steps'], result['sg_m'], result['valid_travel_m']), (True, 20, 0, 300))

    def test_terminal_only_truth_and_no_actions_after_terminal(self):
        env = ParameterizedAreaEnv(p.FIXTURE)
        env.reset(self.ep())
        with self.assertRaises(RuntimeError):
            env.evaluator_result()
        env.step('right')
        with self.assertRaises(RuntimeError):
            env.step('left')

    def test_observation_excludes_evaluator_fields(self):
        env = ParameterizedAreaEnv(p.FIXTURE)
        obs = env.reset(self.ep())
        self.assertFalse({f.name for f in fields(obs)} & {'goal', 'source_tile', 'area', 'dist', 'protocol'})
        self.assertEqual(obs.grid_size, 15)

    def test_source_binding_and_episode_reset(self):
        env = ParameterizedAreaEnv(p.FIXTURE)
        ep = self.ep()
        with self.assertRaises(ValueError):
            env.reset(AreaEpisode(**{**ep.__dict__, 'source_tile': 'wrong_source'}))
        env.reset(ep)
        env.step('down')
        obs = env.reset(ep)
        self.assertEqual((obs.position, obs.visited, obs.remaining_budget), ((0, 0), (0,), 20))

    def test_grid10_input_vectors_are_exactly_legacy(self):
        rng = np.random.default_rng(9407)
        target, current = rng.normal(size=(2, 512)).astype(np.float32)
        visited = [0]
        for remaining in range(20, 0, -1):
            pos = divmod(visited[-1], 10)
            np.testing.assert_array_equal(area_policy_features(target, current, pos, remaining, visited, 10, 20),
                scaled_policy_features(target, current, pos, remaining, visited, 10, 20))
            visited.append(dict(adjacency(visited[-1], 10))['right'] if pos[1] < 9 else visited[-1]-1)

    def test_grid15_uses_same_input_width_and_bounded_pooling(self):
        zero = np.zeros(512, np.float32)
        values = area_policy_features(zero, zero, (14, 14), 1, [224]*20, 15, 20)
        self.assertEqual(values.shape, (1052,))
        np.testing.assert_array_equal(values[1024:1026], [1, 1])
        self.assertEqual(values[-1], 1)
        self.assertEqual(np.count_nonzero(values[-25:]), 1)

    def test_stale_public_prefix_rejected(self):
        zero = np.zeros(512, np.float32)
        with self.assertRaises(ValueError):
            area_policy_features(zero, zero, (14, 14), 20, [0], 15, 20)

    def test_planner_signature_cannot_accept_goal_or_source(self):
        parameters = inspect.signature(choose_area_coverage).parameters
        self.assertFalse(set(parameters) & {'goal', 'target', 'source', 'stratum', 'dist'})
        with self.assertRaises(TypeError):
            choose_area_coverage((0, 0), (0,), 20, 'down', None, grid_size=15, goal=224)

    def test_parameterized_grid10_planner_matches_frozen_implementation(self):
        rng = random.Random(9409)
        for start in (0, 9, 44, 90, 99):
            visited = [start]
            for remaining in range(20, 0, -1):
                pos = divmod(visited[-1], 10)
                for action, _ in adjacency(visited[-1], 10):
                    self.assertEqual(choose_area_coverage(pos, visited, remaining, action, None, grid_size=10),
                                     choose_guarded_coverage_action(pos, visited, remaining, action, None))
                visited.append(rng.choice([x for _, x in adjacency(visited[-1], 10)]))

    def test_grid15_planner_supports_ids_above_99(self):
        for action in ('up', 'left'):
            actual = choose_area_coverage((14, 14), (224,), 20, action, None, grid_size=15)
            self.assertEqual(actual, independent_plan((14, 14), (224,), 20, action, None, 15))
            self.assertTrue(all(0 <= x < 225 for row in actual['paths_by_first'].values() for x in row['cells']))

    def test_accepted_inward_cue_preserves_priority(self):
        result = choose_area_coverage((14, 14), (223, 224), 19, 'up', 'left', grid_size=15)
        self.assertEqual(result['action'], 'left')
        self.assertFalse(result['radial_guard_blocked'])

    def test_terminal_planner_does_not_credit_extra_observation(self):
        result = choose_area_coverage((14, 14), (224,)*20, 1, 'left', None, grid_size=15)
        self.assertEqual(result['horizon'], 1)
        self.assertTrue(all(v['gain'] == 0 for v in result['paths_by_first'].values()))

    def test_grid15_cue_rejects_illegal_or_visited_destination(self):
        values = [.05, .8, .05, .05, .05]
        self.assertEqual(scaled_cue_choice(values, .5, (14, 14), (224,), 15)[1], 'illegal_top_direction')
        self.assertEqual(scaled_cue_choice(values, .5, (14, 13), (223, 224), 15)[1], 'visited_top_destination')

    def test_only_confirmed_protocol_is_default(self):
        config, _ = resolve_area_config(p.ROOT, GRID10)
        self.assertEqual(config['status'], 'confirmed_default')
        self.assertEqual(config['reference_seed'], 0)
        with self.assertRaises(ValueError):
            resolve_area_config(p.ROOT, GRID15)
        config, _ = resolve_area_config(p.ROOT, GRID15, allow_experimental=True)
        self.assertEqual(config['status'], 'engineering_candidate_only')

    def test_frozen_loader_keeps_weights_and_scope(self):
        agent = load_area_policy(p.ROOT, GRID10)
        self.assertEqual(agent.policy, 'Coverage3Radial')
        self.assertFalse(any(parameter.requires_grad for parameter in agent.explorer.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in agent.head.parameters()))
        with self.assertRaises(ValueError):
            load_area_policy(p.ROOT, GRID15)


if __name__ == '__main__':
    unittest.main()
