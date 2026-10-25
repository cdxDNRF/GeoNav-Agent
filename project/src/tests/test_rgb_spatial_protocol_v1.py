"""In-memory geometry/calibration/permission checks; no real data or models."""
import math
import unittest
from project.src.data.rgb_spatial_supervision_protocol_v1 import (
    tile_bounds,contains,patch_point,supervision,inside_probability,
    validate_features,make_folds,train_schedule,calibrate,macro_joint,
)


def sources():
    return {n:dict(city=n,width=w,height=w,mpp=m,xy=xy,raw_xy=raw)
            for n,w,m,xy,raw in [('a',2200,1/22,[1379,900],[859,1046]),
                                 ('b',1000,.1,[585,531],[390,476]),
                                 ('c',625,.06702412868632708,[345.5,342.5],[149.5,276.5])]}


class Tests(unittest.TestCase):
    def test_half_open_native_partition_no_gaps_or_clipping(self):
        w=101;h=77
        for y in range(h):
            for x in range(w):self.assertEqual(sum(contains(tile_bounds(w,h,c),(x,y)) for c in range(25)),1)
        self.assertFalse(any(contains(tile_bounds(w,h,c),(w,h)) for c in range(25)))
        self.assertEqual(tile_bounds(625,625,24),[500,500,625,625])

    def test_metric_mapping_does_not_assume_common100m(self):
        a=tile_bounds(2200,2200,0);b=tile_bounds(625,625,0)
        self.assertAlmostEqual((a[2]-a[0])/22,20)
        self.assertAlmostEqual((b[2]-b[0])*.06702412868632708,8.37801608579)
        self.assertEqual(patch_point([0,0,14,14],0),[1,1])

    def test_soft_spatial_label_and_empty_class(self):
        box=[0,0,14,14];p=supervision(box,[7,7],1)
        self.assertEqual(len(p),50);self.assertAlmostEqual(sum(p),1);self.assertEqual(p[-1],0)
        self.assertEqual(max(range(49),key=lambda i:p[i]),24)
        for xy in (None,[14,7],[-1,7]):self.assertEqual(supervision(box,xy,1),[0.]*49+[1.])

    def test_inside_score_stable_and_disabled_prior(self):
        self.assertAlmostEqual(inside_probability([0.]*49,math.log(49)),.5)
        self.assertAlmostEqual(inside_probability([1000.]*49,1000+math.log(49)),.5)

    def test_feature_packet_rejects_metadata_and_bank(self):
        unit=[1.]+[0.]*511;p=dict(given_target_feature=unit,current_patch_features=[unit]*49)
        self.assertTrue(validate_features(p))
        self.assertTrue(validate_features(dict(p,given_target_feature=[0.]*512)))
        for name in ('camera_xy','sourceID','K','history','unvisited_features'):
            with self.assertRaises(ValueError):validate_features(dict(p,**{name:0}))
        with self.assertRaises(ValueError):validate_features(dict(p,current_patch_features=[unit]*50))

    def test_fold_roles_wrong_donor_and_known_negative_disclosure(self):
        folds=make_folds(sources());self.assertEqual(len(folds),3)
        for f in folds:
            self.assertNotIn(f['fold'],f['fit_sources'])
            self.assertNotEqual(f['wrong_target_donor'],f['unrelated_reference'])
            self.assertFalse(f['all_unrelated_reference_sources_unseen'])
            self.assertEqual(len(f['fit_cases']),100);self.assertEqual(len(f['development_cases']),50)
            self.assertEqual([sum(c['relation']==k for c in f['development_cases']) for k in ('inside','outside','unrelated')],[1,24,25])

    def test_same_balanced_schedule_for_two_arms_no_held_inputs(self):
        for f in make_folds(sources()):
            lookup={c['case_id']:c for c in f['fit_cases']}
            for seed in (0,1,2):
                batches=train_schedule(f,seed);self.assertEqual(batches,train_schedule(f,seed))
                self.assertEqual(len(batches),200)
                for batch in batches:
                    rows=[lookup[i] for i in batch]
                    self.assertEqual([sum(r['relation']==k for r in rows) for k in ('inside','outside','unrelated')],[2,2,2])
                    self.assertTrue(all(r['query']!=f['fold'] and r['reference']!=f['fold'] for r in rows))

    def test_calibration_tie_prefers_reject_all_and_no_dev(self):
        cases=[dict(split='fit',query='a',relation=k,camera_xy=[0,0] if k=='inside' else None,mpp=1) for k in ('inside','outside','unrelated')]
        out=[dict(score=.5,point=[20,20]) for _ in cases]
        d=calibrate(cases,out);self.assertGreater(d['tau'],.5)
        self.assertAlmostEqual(d['fit_macro_joint'],2/3)
        self.assertEqual(d['fit_macro_joint_exact'],[2,3])
        cases[0]['split']='development'
        with self.assertRaises(ValueError):calibrate(cases,out)

    def test_false_localization_not_counted_as_positive_acceptance(self):
        cases=[dict(split='fit',query='a',relation=k,camera_xy=[0,0] if k=='inside' else None,mpp=1) for k in ('inside','outside','unrelated')]
        outputs=[dict(score=.9,point=[10,0]),dict(score=.1,point=[0,0]),dict(score=.1,point=[0,0])]
        self.assertAlmostEqual(macro_joint(cases,outputs,.5),2/3)
        outputs[0]['point']=[3,4];self.assertEqual(macro_joint(cases,outputs,.5),1)


if __name__=='__main__':unittest.main()
