"""Budget, information permissions, cue priority and damage protection."""
import inspect
import unittest
from agents.coverage_lookahead import choose_coverage_action
from eval.audit_coverage_lookahead import independent_coverage
from eval.coverage_lookahead_validation import gated, all_gates, STRATA


class CoverageLookaheadTests(unittest.TestCase):
    def test_only_public_geometry_and_original_proposals_are_accepted(self):
        self.assertEqual(tuple(inspect.signature(choose_coverage_action).parameters),
            ('position', 'visited', 'remaining_budget', 'explorer_action', 'cue_action'))
        with self.assertRaises(TypeError):
            choose_coverage_action((0, 3), (3,), 20, 'right', None, goal=99)

    def test_accepted_cue_preserves_target_action(self):
        result = choose_coverage_action((0, 3), (3,), 20, 'right', 'left')
        self.assertEqual(result['action'], 'left')
        self.assertFalse(result['changed'])
        self.assertEqual(result['paths_by_first'], {})

    def test_terminal_position_is_not_given_an_extra_cue_move(self):
        result = choose_coverage_action((0, 3), (3,) * 20, 1, 'right', None)
        self.assertEqual(result['horizon'], 1)
        self.assertTrue(all(v['gain'] == 0 for v in result['paths_by_first'].values()))
        self.assertEqual(result['action'], 'right')

    def test_small_one_cell_advantage_does_not_override_actor(self):
        result = choose_coverage_action((0, 3), (3,) * 19, 2, 'right', None)
        self.assertEqual(result['action'], 'right')
        self.assertFalse(result['changed'])

    def test_border_example_has_two_cell_advantage(self):
        result = choose_coverage_action((0, 3), (3,), 20, 'right', None)
        self.assertEqual(result['action'], 'down')
        self.assertEqual(result['opportunity_gain'], 2)
        self.assertEqual(result['baseline_gain'], 7)
        self.assertEqual(result['selected_gain'], 9)

    def test_no_advantage_keeps_original_direction(self):
        result = choose_coverage_action((3, 3), (33,), 20, 'right', None)
        self.assertEqual(result['action'], 'right')
        self.assertEqual(result['opportunity_gain'], 0)

    def test_bad_public_prefix_and_illegal_proposals_rejected(self):
        for args in [((0, 3), (4,), 20, 'right', None), ((0, 3), (3,), 19, 'right', None),
                     ((0, 3), (3,), 20, 'up', None), ((0, 3), (3,), 20, 'right', 'up')]:
            with self.assertRaises(ValueError):
                choose_coverage_action(*args)

    def test_geometry_has_no_episode_memory_or_input_mutation(self):
        visited = [3]
        expected = choose_coverage_action((0, 3), visited, 20, 'right', None)
        choose_coverage_action((8, 8), (88,), 20, 'left', None)
        self.assertEqual(choose_coverage_action((0, 3), visited, 20, 'right', None), expected)
        self.assertEqual(visited, [3])

    def test_independent_set_and_bitmask_planners_agree_at_boundaries(self):
        for cell in (0, 3, 9, 33, 44, 90, 99):
            position = divmod(cell, 10)
            actions = ['right' if cell % 10 < 9 else 'left', 'down' if cell // 10 < 9 else 'up']
            for remaining in (1, 2, 3, 20):
                visits = (cell,) * (21 - remaining)
                for action in actions:
                    self.assertEqual(choose_coverage_action(position, visits, remaining, action, None),
                                     independent_coverage(position, visits, remaining, action, None))

    def test_mixed_gain_cannot_hide_damage_to_a_stratum(self):
        effect = dict(sr_gain=.03, positive_seeds=3, sg_change=-.1)
        effects = {k: dict(effect) for k in ('all', *STRATA)}
        self.assertTrue(all_gates(gated(effects)))
        effects['long_distance']['sr_gain'] = -.021
        self.assertFalse(all_gates(gated(effects)))
        effects['long_distance'] = dict(effect)
        effects['interior_target']['sg_change'] = .01
        self.assertFalse(all_gates(gated(effects)))


if __name__ == '__main__':
    unittest.main()
