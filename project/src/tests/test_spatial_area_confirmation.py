"""Meaningful gates, fixed cohorts and probe-control permission checks."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import numpy as np
from eval import spatial_area_confirmation as run


class SpatialConfirmationTests(unittest.TestCase):
    def inputs(self):
        return [run.read(run.DATA / name) for name in
                ('导航任务.json', '任务分层.json', '错误目标计划.json', '邻接诊断探针.json')]

    def test_all_fixed_routes_and_wrong_exceptions(self):
        args = self.inputs()
        run.validate_cohort(*args)
        bad = deepcopy(args)
        bad[0][-1] = bad[0][0]
        with self.assertRaises(ValueError):
            run.validate_cohort(*bad)
        bad = deepcopy(args)
        name = bad[0][0]['episode_id']
        bad[2][name]['cue_cell'] = bad[0][0]['goal']
        with self.assertRaises(ValueError):
            run.validate_cohort(*bad)

    def test_probe_triples_share_nonself_wrong_target(self):
        probes = self.inputs()[3]
        plan = run.make_probe_wrong(probes)
        self.assertEqual(plan, run.make_probe_wrong(probes))
        groups = {}
        for p in probes:
            groups.setdefault((p['area'], p['pair_id']), []).append(p)
        self.assertEqual(len(groups), 560)
        for group in groups.values():
            chosen = {plan[p['probe_id']]['cue_cell'] for p in group}
            self.assertEqual(len(chosen), 1)
            self.assertFalse(chosen & {group[0]['target_cell'], *(p['current_cell'] for p in group)})

    def test_probe_count_alone_does_not_accept_wrong_geometry(self):
        args = self.inputs()
        args[3][0]['expected_class'] = 'left'
        with self.assertRaises(ValueError):
            run.validate_cohort(*args)

    def test_modified_preregistration_cannot_pass_file_bindings(self):
        with patch.object(run, 'read', return_value=dict(registration_sha256='frozen')):
            with patch.object(run, 'digest', return_value='changed'):
                with self.assertRaisesRegex(ValueError, 'preregistration seal changed'):
                    run.check_bindings({})

    def test_primary_gate_needs_source_interval_and_both_SG(self):
        primary = dict(sr_gain=.05, positive_seeds=2, sg_change=0, source95_SR=[.01, .1])
        pooled = dict(sg_change=0)
        self.assertTrue(all(run.target_checks(primary, pooled).values()))
        self.assertFalse(all(run.target_checks(dict(primary, source95_SR=[0, .1]), pooled).values()))
        self.assertFalse(all(run.target_checks(primary, dict(sg_change=.1)).values()))
        self.assertFalse(all(run.target_checks(dict(primary, positive_seeds=1), pooled).values()))

    def test_whole_region_interval_retains_three_weight_dependency(self):
        def stats(sign):
            return dict(metrics=dict(sr=.5 + .1*sign, mean_sg_all_episodes=1),
                        by_source={s:dict(sr=.5+.1*sign, mean_sg_all_episodes=1) for s in ['a', 'b']})
        eff = run.effect([stats(1)]*3, [stats(0)]*3)
        self.assertEqual(eff['source_count'], 2)
        np.testing.assert_allclose(eff['source95_SR'], [.1, .1], atol=1e-15)

    def test_probe_raw_wrong_direction_is_not_correct(self):
        row = dict(expected_class='up', raw_accepted=True, raw_correct=False, top_class='right',
                   usable_reason='illegal_top_direction', given_expected_class='right')
        m = run.probe_metrics([row])
        self.assertEqual(m['raw_correct_acceptance'], 0)
        self.assertEqual(m['raw_wrong_direction_accepts'], 1)
        self.assertEqual(m['usable_accepted'], 0)
        self.assertEqual(m['given_target_correct_accepts'], 1)
        self.assertIsNone(m['raw_nonadjacent_false_acceptance'])

    def test_nonadjacent_raw_accept_survives_usability_rejection(self):
        row = dict(expected_class='not_adjacent', raw_accepted=True, raw_correct=False, top_class='up',
                   usable_reason='illegal_top_direction', given_expected_class='not_adjacent')
        self.assertEqual(run.probe_metrics([row])['raw_nonadjacent_false_acceptance'], 1)

    def test_given_target_class_does_not_wrap_row_boundary(self):
        self.assertEqual(run.given_class(9, 10), 'not_adjacent')
        self.assertEqual(run.given_class(4, 5), 'right')
        self.assertEqual(run.given_class(44, 54), 'down')

    def test_engineering_not_accepted_from_scores_or_incomplete_audit(self):
        with patch.object(run, 'read', return_value=dict(passed=True, completed=True, records=4199)):
            with self.assertRaises((ValueError, KeyError)):
                run.check_engineering()


if __name__ == '__main__':
    unittest.main()
