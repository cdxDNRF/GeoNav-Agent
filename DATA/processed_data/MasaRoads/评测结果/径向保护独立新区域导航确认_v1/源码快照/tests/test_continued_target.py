"""Evidence gate boundary regressions: no passing on only SR or one weight."""
import unittest
from eval.continued_target_evidence import target_checks,cohort,PRIOR,read


class ContinuedTargetTests(unittest.TestCase):
    def test_exact_threshold_is_allowed(self):
        self.assertTrue(all(target_checks(dict(sr_gain=.05,positive_seeds=2,sg_change=0)).values()))

    def test_high_sr_does_not_hide_sg_damage(self):
        x=target_checks(dict(sr_gain=.30,positive_seeds=3,sg_change=.001))
        self.assertFalse(x['SG_no_worse'])

    def test_one_good_weight_is_insufficient(self):
        self.assertFalse(target_checks(dict(sr_gain=.051,positive_seeds=1,sg_change=-1))['two_positive_weights'])

    def test_gain_below_gate_does_not_pass(self):
        self.assertFalse(target_checks(dict(sr_gain=.0499,positive_seeds=3,sg_change=-1))['SR_gain5pp'])

    def test_missing_or_duplicated_planned_task_rejected(self):
        rows=read(PRIOR/'导航任务.json');self.assertEqual(len(cohort(rows)),500)
        with self.assertRaises(ValueError):cohort(rows[:-1])
        with self.assertRaises(ValueError):cohort([rows[0],*rows[:-1]])


if __name__=='__main__':unittest.main()
