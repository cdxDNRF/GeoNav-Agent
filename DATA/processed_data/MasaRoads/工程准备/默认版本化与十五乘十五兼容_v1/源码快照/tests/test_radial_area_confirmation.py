"""Cohort, gate and audit-boundary tests before consuming reserved areas."""
from copy import deepcopy
import unittest
from eval import radial_area_confirmation as p
from eval.audit_radial_area_confirmation import independent_checks


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.strata, cls.wrong = (p.read(p.DATA / n) for n in ('导航任务.json', '任务分层.json', '错误目标计划.json'))

    def test_reserved_complete_cohort(self):
        p.validate_cohort(self.tasks, self.strata, self.wrong)

    def test_duplicate_route_rejected(self):
        tasks = deepcopy(self.tasks)
        tasks[1] = dict(tasks[0], episode_id=tasks[1]['episode_id'])
        with self.assertRaises(ValueError):
            p.validate_cohort(tasks, self.strata, self.wrong)

    def test_unreserved_area_rejected(self):
        tasks = deepcopy(self.tasks)
        tasks[0]['area'] = 'img_2000'
        with self.assertRaises(ValueError):
            p.validate_cohort(tasks, self.strata, self.wrong)

    def test_wrong_target_endpoint_rejected(self):
        wrong = deepcopy(self.wrong)
        wrong[self.tasks[0]['episode_id']]['cue_cell'] = self.tasks[0]['goal']
        with self.assertRaises(ValueError):
            p.validate_cohort(self.tasks, self.strata, wrong)

    def test_stratum_distance_not_interchangeable(self):
        strata = deepcopy(self.strata)
        strata[self.tasks[0]['episode_id']]['stratum'] = 'interior_target'
        with self.assertRaises(ValueError):
            p.validate_cohort(self.tasks, strata, self.wrong)

    def effects(self):
        return {s: dict(sr_gain=.03, positive_seeds=2, sg_change=-.1, source95_SR=[.01, .04]) for s in ('all', *p.STRATA)}

    def test_ci_required_even_if_point_gain_passes(self):
        e = self.effects()
        self.assertTrue(p.all_checks(p.checks(e)))
        e['all']['source95_SR'][0] = 0
        self.assertFalse(p.all_checks(p.checks(e)))

    def test_each_stratum_protected_and_auditor_agrees(self):
        for s in p.STRATA:
            for field, value in [('sr_gain', -.021), ('sg_change', .001)]:
                e = self.effects()
                e[s][field] = value
                self.assertEqual(p.checks(e), independent_checks(e))
                self.assertFalse(p.all_checks(p.checks(e)))

    def test_target_controls_have_separate_higher_gate(self):
        e = self.effects()['all']
        self.assertFalse(all(p.target_checks(e).values()))
        e['sr_gain'] = .05
        self.assertTrue(all(p.target_checks(e).values()))
        e['positive_seeds'] = 1
        self.assertFalse(all(p.target_checks(e).values()))

    def test_new_output_paths_do_not_alias_historical_batches(self):
        self.assertNotEqual(p.OUT, p.PREP)
        self.assertNotEqual(p.OUT, p.DEV)
        self.assertNotEqual(p.OUT, p.prior.OUT)
        for seed in p.SEEDS:
            for policy in ('M0', 'Coverage3Radial'):
                self.assertTrue(p.trajectory_path(policy, seed, 'CueFull').is_relative_to(p.OUT / '主对照'))


if __name__ == '__main__':
    unittest.main()
