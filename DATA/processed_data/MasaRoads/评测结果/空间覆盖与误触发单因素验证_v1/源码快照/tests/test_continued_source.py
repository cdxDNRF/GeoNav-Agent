"""Independent candidate gate and protected grid5 protocol regression tests."""
from copy import deepcopy
import unittest
from eval.continued_source_confirmation import primary_checks,validate_cohort,planned,PLAN,read


class ContinuedSourceTests(unittest.TestCase):
    def example(self):
        return dict(sr_gain=.02,positive_seeds=2,sg_change=-.1,source95_SR=[.001,.04]),dict(sr_gain=-.02,sg_change=0),dict(sr_mean=.7,sg_mean=1)

    def test_preregistered_boundaries(self):self.assertTrue(all(primary_checks(*self.example()).values()))

    def test_cross_zero_source_interval_blocks_stable_upgrade(self):
        a,b,c=self.example();a['source95_SR'][0]=0
        self.assertFalse(primary_checks(a,b,c)['source95_SR_grid10_positive'])

    def test_density_gain_cannot_hide_old_protocol_damage(self):
        a,b,c=self.example();a['sr_gain']=.1;b['sr_gain']=-.021
        self.assertFalse(primary_checks(a,b,c)['SR_grid5_drop_at_most2pp'])

    def test_sg_and_weight_requirements(self):
        a,b,c=self.example();a['positive_seeds']=1;b['sg_change']=.001
        x=primary_checks(a,b,c);self.assertFalse(x['two_positive_weights_grid10']);self.assertFalse(x['SG_grid5_no_worse'])

    def test_no_source_or_distance_omission_on_either_grid(self):
        sources=read(PLAN)['sources']['confirmation']
        for k in (5,10):
            es=planned(k,sources);validate_cohort(es,k)
            with self.assertRaises(ValueError):validate_cohort(es[:-1],k)
            with self.assertRaises(ValueError):validate_cohort(es,15-k)


if __name__=='__main__':unittest.main()
