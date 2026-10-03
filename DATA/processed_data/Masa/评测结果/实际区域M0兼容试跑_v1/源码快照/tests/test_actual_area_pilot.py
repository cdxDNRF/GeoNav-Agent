"""Pilot denominators, posthoc seams, physical distance and rule compatibility."""
from dataclasses import replace
from pathlib import Path
import unittest
from env.environment import Observation
from env.actual_area import AreaEpisode
from env.scaled_grid import fixed_region_action
from agents.exploration import FrontierPolicy
from eval.actual_area_pilot import (DATA, read, validate_cohort, step_diagnostic, destination, diagnosis, target_checks)
from eval.audit_actual_area_pilot import independent_diagnostic, independent_rule


class AreaPilotTests(unittest.TestCase):
    def test_fixed75_tasks_and_wrong_target_exceptions(self):
        es = [AreaEpisode(**r) for r in read(DATA / '导航任务.json')]
        wrong = read(DATA / '错误目标计划.json')
        validate_cohort(es, wrong)
        with self.assertRaises(ValueError):
            validate_cohort(es[:-1], wrong)
        with self.assertRaises(ValueError):
            validate_cohort([replace(es[0], goal=es[0].start)] + es[1:], wrong)
        altered = {k: dict(v) for k, v in wrong.items()}
        altered[es[0].episode_id]['cue_cell'] = es[0].goal
        with self.assertRaises(ValueError):
            validate_cohort(es, altered)

    def test_seam_and_true_hit_from_independent_raster_order(self):
        regions = read(DATA / '数据清单.json')['regions']
        provenance = read(DATA / '图块来源.json')
        for cell, goal, action, crosses in ((4, 5, 'right', True), (44, 54, 'down', True), (3, 4, 'right', False)):
            d = step_diagnostic(cell, goal, goal, action, True, 1, regions[0]['area'], provenance)
            self.assertEqual(d, independent_diagnostic(cell, goal, goal, action, True, 1, regions[0]))
            self.assertEqual(d['crossed_source_boundary'], crosses)
            self.assertEqual(d['true_adjacency_cross_source'], crosses)
            self.assertTrue(d['accepted_immediate_true_hit'])
            self.assertEqual(d['effective_move_m'], 300)

    def test_terminal_adjacency_has_no_actionable_opportunity(self):
        r = read(DATA / '数据清单.json')['regions'][0]
        d = step_diagnostic(4, 5, 5, 'right', False, 0, r['area'], read(DATA / '图块来源.json'))
        self.assertFalse(d['actionable_true_adjacency'])
        self.assertFalse(d['true_adjacency_cross_source'])

    def test_boundary_move_consumes_no_physical_distance(self):
        self.assertEqual(destination(0, 'up'), (0, True))
        self.assertEqual(destination(99, 'right'), (99, True))
        self.assertEqual(destination(99, 'left'), (98, False))
        provenance = read(DATA / '图块来源.json')
        d = step_diagnostic(0, 99, 99, 'up', False, 20, 'img_1000', provenance)
        self.assertEqual(d['effective_move_m'], 0)
        self.assertFalse(d['crossed_source_boundary'])

    def test_zero_seam_denominator_is_not_perfect_acceptance(self):
        d = step_diagnostic(0, 99, 99, 'right', False, 20, 'img_1000', read(DATA / '图块来源.json'))
        row = dict(success=False, sg=17, decisions=[dict(reason='not_adjacent')], evaluation_diagnostics=[d])
        result = diagnosis([row])
        self.assertIsNone(result['adjacency']['cross_source']['acceptance_rate'])
        self.assertEqual(result['failure_categories']['never_within2'], 1)

    def test_last_only_and_missed_adjacency_are_separate(self):
        p = read(DATA / '图块来源.json')
        a = step_diagnostic(3, 5, 5, 'right', False, 1, 'img_1000', p)
        b = step_diagnostic(4, 5, 5, 'left', False, 1, 'img_1000', p)
        rows = [dict(success=False, sg=1, decisions=[dict(reason='not_adjacent')], evaluation_diagnostics=[a]),
                dict(success=False, sg=2, decisions=[dict(reason='low_confidence')], evaluation_diagnostics=[b])]
        result = diagnosis(rows)
        self.assertEqual(result['failure_categories']['adjacent_only_terminal'], 1)
        self.assertEqual(result['failure_categories']['missed_actionable_adjacency'], 1)

    def test_target_gate_is_not_SR60(self):
        self.assertTrue(all(target_checks(dict(sr_gain=.05, positive_seeds=2, sg_change=0)).values()))
        self.assertFalse(all(target_checks(dict(sr_gain=.04, positive_seeds=3, sg_change=-1)).values()))
        self.assertFalse(all(target_checks(dict(sr_gain=.10, positive_seeds=1, sg_change=-1)).values()))
        self.assertFalse(all(target_checks(dict(sr_gain=.10, positive_seeds=3, sg_change=.01)).values()))

    def test_original_rules_generic10_including_all_visited_wall(self):
        frontier = FrontierPolicy()
        fixtures = [(0, (0,)), (44, tuple(range(45))), (99, tuple(range(100))),
                    (0, tuple(range(100))), (55, (55, 45, 54, 56, 65))]
        for cell, visited in fixtures:
            obs = Observation(b'current', b'target', divmod(cell, 10), 10, 3, visited)
            self.assertEqual(frontier.act(obs), independent_rule('Frontier', obs))
            self.assertEqual(fixed_region_action(obs), independent_rule('FixedRegion', obs))
            self.assertFalse(destination(cell, frontier.act(obs))[1])
        obs = Observation(b'current', b'target', (0, 0), 10, 3, tuple(range(100)))
        self.assertEqual(fixed_region_action(obs), 'up')
        self.assertTrue(destination(0, fixed_region_action(obs))[1])


if __name__ == '__main__':
    unittest.main()
