"""Pure geometry/plan/control-metric tests, no real image/model execution."""
import copy
import unittest
import numpy as np
from project.src.eval import full_reference_anchor_v1 as run


def fixture():
    cases=run.plan();rows=[]
    for c in cases:
        xy=c['truth_evaluator_only'] if c['arm']=='Full' else [1.,1.]
        rows.append(dict(pair=c['pair'],arm=c['arm'],xy=xy,raw_score=.1))
    return cases,rows


class Tests(unittest.TestCase):
    def test_exact16_and_fixed_wrong_photos(self):
        p=run.plan();self.assertEqual(len(p),16)
        for c in p:self.assertEqual(c['wrong_pair'],{0:1,1:0,2:3,3:2}[c['pair']])
    def test_Full_position_and_goal_control_advantage(self):
        c,r=fixture();s=run.evaluate(c,r);self.assertTrue(s['full_reference_goal_advantage_supported'])
        self.assertEqual(s['metrics']['Full']['within5m'],4);self.assertEqual(s['contrasts']['Wrong']['joint_net'],4)
    def test_mean_or_disabled_as_good_as_Full_blocks_goal_claim(self):
        c,r=fixture()
        for x in r:
            if x['arm']=='Disabled':x['xy']=next(k['truth_evaluator_only'] for k in c if k['pair']==x['pair'])
        self.assertFalse(run.evaluate(c,r)['full_reference_goal_advantage_supported'])
    def test_source2or3_failure_blocks_even3of4(self):
        c,r=fixture();next(x for x in r if x['pair']==2 and x['arm']=='Full')['xy']=[1.,1.]
        s=run.evaluate(c,r);self.assertTrue(s['gates']['Full_at_least3of4_within5m']);self.assertFalse(s['full_reference_goal_advantage_supported'])
    def test_missing_duplicate_and_invalid_points_rejected(self):
        c,r=fixture()
        with self.assertRaises(ValueError):run.evaluate(c,r[:-1])
        with self.assertRaises(ValueError):run.evaluate(c,r[:-1]+r[:1])
        for point in ([625.,1.],[float('nan'),1.]):
            z=copy.deepcopy(r);z[0]['xy']=point
            with self.assertRaises(ValueError):run.evaluate(c,z)
    def test_heat_coordinates_ties_and_independent_peak(self):
        h=np.zeros((512,512),np.float32);m=np.zeros((256,256),np.float32);h[10,20]=1;h[11,1]=1;m[6,7]=.7
        r=run.peaks(h,m);self.assertEqual(r['heat_index'],10*512+20);self.assertEqual(r['xy'],[20.5*625/512,10.5*625/512]);self.assertTrue(r['alternate_peak_passed'])
        h[0,0]=float('nan')
        with self.assertRaises(ValueError):run.peaks(h,m)
    def test_resource_exact_cap_and_no_hidden_warmup(self):
        self.assertFalse(run.guard(4*1024**3,180,8*1024**2));self.assertTrue(run.guard(4*1024**3+1,1,1))
        self.assertTrue(run.guard(1,180.1,1));self.assertTrue(run.guard(1,1,8*1024**2+1))
    def test_errors_in_meters_not_resized_pixel_units(self):
        c,r=fixture();r[0]['xy']=[c[0]['truth_evaluator_only'][0]+10,c[0]['truth_evaluator_only'][1]]
        self.assertAlmostEqual(run.evaluate(c,r)['details'][0]['error_m'],1.6)


if __name__=='__main__':unittest.main()
