"""Perturbation geometry, channel updates, control invariance and prospective gates."""
from io import BytesIO
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.target_perturbations import transform,VARIANTS
from eval.target_robustness import summarize
from agents.precomputed_frozen_edge import PrecomputedFrozenEdgeNavigator
from agents.edge_cue import EdgeTargetCueHead,image_profiles
from env.environment import Observation


class Explorer(nn.Module):
    def step(self,x,hidden=None):return torch.tensor([[0.,1.,3.,2.]]),torch.zeros(1),torch.zeros(1,512),torch.zeros(1,256)


class TargetRobustnessTests(unittest.TestCase):
    def test_clean_is_lossless_and_does_not_mutate_source(self):
        a=np.random.default_rng(7).integers(0,256,(300,300,3),dtype=np.uint8);before=a.copy()
        for v in VARIANTS:self.assertEqual(transform(a,v).size,(300,300))
        np.testing.assert_array_equal(np.asarray(transform(a,'Clean')),before);np.testing.assert_array_equal(a,before)

    def test_rotation_is_clockwise_exact_without_interpolation(self):
        a=np.random.default_rng(7).integers(0,256,(300,300,3),dtype=np.uint8)
        np.testing.assert_array_equal(np.asarray(transform(a,'Rot90CW')),np.rot90(a,-1))

    def test_ring_only_changes_eight_pixel_border(self):
        a=np.random.default_rng(7).integers(0,256,(300,300,3),dtype=np.uint8);b=np.asarray(transform(a,'Ring8'))
        np.testing.assert_array_equal(b[8:-8,8:-8],a[8:-8,8:-8])
        self.assertTrue((b[:8]==128).all() and (b[-8:]==128).all() and (b[:,:8]==128).all() and (b[:,-8:]==128).all())

    def test_brightness_and_crop_have_fixed_documented_strength(self):
        a=np.full((300,300,3),255,np.uint8);a[8:-8,8:-8]=100
        b=np.asarray(transform(a,'Bright080'));self.assertTrue((b[:8]==204).all() and (b[8:-8,8:-8]==80).all())
        self.assertTrue((np.asarray(transform(a,'Crop8'))==100).all())

    def test_blur_and_jpeg_preserve_rgb_shape_and_change_nonconstant_pixels(self):
        a=np.zeros((300,300,3),np.uint8);a[:,151:]=255
        for v in ('Blur1','JPEG25'):
            b=np.asarray(transform(a,v));self.assertEqual(b.shape,a.shape);self.assertEqual(b.dtype,np.uint8);self.assertFalse(np.array_equal(a,b))
        with self.assertRaises(ValueError):transform(a,'unknown_strength')

    def test_no_target_control_updates_cue_input_but_keeps_explorer_proposal(self):
        torch.set_num_threads(1);torch.manual_seed(7);rng=np.random.default_rng(7)
        def payload(rgb):
            out=BytesIO();Image.fromarray(rgb).save(out,format='PNG');return out.getvalue()
        a=rng.integers(0,256,(300,300,3),dtype=np.uint8);b=np.asarray(transform(a,'Rot90CW'))
        means=[np.zeros(s,np.float32) for s in [(512,),(512,),(4,768),(4,3,64,3)]]
        agent=PrecomputedFrozenEdgeNavigator(Explorer(),EdgeTargetCueHead(),'cpu',*means,.5,'Edge','Baseline')
        x=[rng.normal(size=s).astype(np.float32) for s in [(512,),(4,768),(512,),(4,768)]]
        obs=Observation(payload(a),payload(a),(2,2),5,10,(12,));original=agent.act_with_profiles(obs,*x,image_profiles(a),image_profiles(a));agent.reset()
        updated=agent.act_with_profiles(Observation(payload(a),payload(b),(2,2),5,10,(12,)),*x[:2],x[2]*2,x[3]*2,image_profiles(a),image_profiles(b))
        self.assertEqual(original['action'],updated['action']);self.assertEqual(original['explorer_features_sha256'],updated['explorer_features_sha256'])
        self.assertNotEqual(original['target_image_sha256'],updated['target_image_sha256']);self.assertNotEqual(original['cue_features_sha256'],updated['cue_features_sha256'])

    def runs(self,full=.80,sg=.4):
        def r(sr,sg):return dict(metrics=dict(episodes=500,successes=round(sr*500),sr=sr,mean_sg_all_episodes=sg,repeat_visit_rate_micro=.02,out_of_bounds_rate=0.),
            by_source={f'img_{i}':dict(sr=sr,mean_sg_all_episodes=sg) for i in range(20)},
            by_distance={str(d):dict(sr=sr,mean_sg_all_episodes=sg) for d in range(4,9)})
        runs={(v,s,c):r(.85 if v=='Clean' and c=='CueFull' else full if c=='CueFull' else .70, .4 if v=='Clean' else sg) for v in VARIANTS for s in range(3) for c in ('CueFull','Baseline')}
        rules={'Frontier':r(.45,2.4),'FixedRegion':r(.30,2.5)}
        return runs,rules

    def test_retention_thresholds_inclusive_and_failed_stress_is_reported(self):
        r,rule=self.runs(full=.80,sg=.7);v=summarize(r,rule)
        self.assertTrue(v['variants']['Rot90CW']['performance_retained'])
        r,rule=self.runs(full=.799,sg=.701);v=summarize(r,rule)
        self.assertFalse(v['variants']['Rot90CW']['performance_retained']);self.assertFalse(v['all_six_fixed_stresses_retained_numeric'])

    def test_high_exploration_score_cannot_replace_target_benefit(self):
        r,rule=self.runs(full=.70,sg=.4);v=summarize(r,rule)['variants']['Ring8']
        self.assertTrue(v['engineering_numeric_passed']);self.assertFalse(v['target_gain_retained']);self.assertFalse(v['SR_gain_CI95_positive'])
        self.assertEqual(v['versus_NoTarget']['source_interval']['source_count'],20)


if __name__=='__main__':unittest.main()
