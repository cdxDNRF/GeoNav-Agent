"""Raw scan confidence, tie symmetry, observed-pixel boundary and clean protection gates."""
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.rotation_candidate import TargetView,RotationCandidateNavigator,rotated_payload,select_view,ANGLES
from agents.edge_cue import EdgeTargetCueHead,image_profiles
from env.environment import Observation
from eval.orientation_confirmation import summarize,VARIANTS,STRATEGIES


class Explorer(nn.Module):
    def step(self,x,hidden=None):return torch.tensor([[0.,3.,1.,2.]]),torch.zeros(1),torch.zeros(1,512),torch.zeros(1,256)


class RotationTests(unittest.TestCase):
    def test_four_rotations_are_lossless_closed_under_composition(self):
        pixels=np.random.default_rng(9).integers(0,256,(300,300,3),dtype=np.uint8);out=BytesIO();Image.fromarray(pixels).save(out,format='PNG');p=out.getvalue()
        for a in ANGLES:
            for b in ANGLES:self.assertEqual(rotated_payload(rotated_payload(p,a),b),rotated_payload(p,(a+b)%360))
        with self.assertRaises(ValueError):rotated_payload(p,45)

    def test_selection_preserves_raw_probability_without_renormalizing_angles(self):
        p=np.array([[.49,0,0,0,.51],[.4,0,0,0,.6],[.3,0,0,0,.7],[.2,0,0,0,.8]])
        i,score=select_view(p,['d','c','b','a']);self.assertEqual(i,0);self.assertAlmostEqual(score,.49)

    def test_payload_sha_tie_is_independent_of_candidate_traversal_order(self):
        p=np.tile([.6,.1,.1,.1,.1],(4,1));keys=['d','a','c','b'];i,_=select_view(p,keys);self.assertEqual(keys[i],'a')
        order=[3,2,1,0];j,_=select_view(p[order],[keys[k] for k in order]);self.assertEqual([keys[k] for k in order][j],'a')

    def setup(self):
        torch.set_num_threads(1);torch.manual_seed(7);rng=np.random.default_rng(7)
        rgb=rng.integers(0,256,(300,300,3),dtype=np.uint8);f=BytesIO();Image.fromarray(rgb).save(f,format='PNG');p=f.getvalue()
        means=[np.zeros(s,np.float32) for s in [(512,),(512,),(4,768),(4,3,64,3)]]
        a=RotationCandidateNavigator(Explorer(),EdgeTargetCueHead(),'cpu',*means,.5,'Edge','CueFull')
        obs=Observation(p,p,(2,2),5,10,(12,));g=rng.normal(size=512).astype(np.float32);l=rng.normal(size=(4,768)).astype(np.float32)
        views=[TargetView(angle,rotated_payload(p,angle),rng.normal(size=512).astype(np.float32),rng.normal(size=(4,768)).astype(np.float32),image_profiles(rotated_payload(p,angle))) for angle in ANGLES]
        return a,obs,g,l,image_profiles(p),views

    def test_given_rotation_candidate_reordering_does_not_change_decision(self):
        a,obs,g,l,cp,views=self.setup();x=a.act_candidates(obs,g,l,cp,views);a.reset()
        # A 90-degree given target has exactly the same observed candidate pixel set.
        rotated=replace(obs,target_image=rotated_payload(obs.target_image,90))
        rotated_views=[replace(v,relative_clockwise=(v.relative_clockwise-90)%360) for v in reversed(views)]
        y=a.act_candidates(rotated,g,l,cp,rotated_views)
        for k in ('action','probabilities','cue_action','reason','selected_target_sha256','cue_features_sha256'):self.assertEqual(x[k],y[k])

    def test_candidate_pixels_are_checked_when_view_set_changes_for_same_target(self):
        a,obs,g,l,cp,views=self.setup();a.act_candidates(obs,g,l,cp,views)
        views[0]=replace(views[0],payload=views[1].payload)
        with self.assertRaises(ValueError):a.act_candidates(obs,g,l,cp,views)
        with self.assertRaises(TypeError):a.act_candidates(obs,g,l,cp,views,true_angle=90)

    def test_no_legal_candidate_reranking_after_highest_direction_is_invalid(self):
        from agents.target_cue import choose_cue
        p=np.array([[.8,.05,.05,.05,.05],[.02,.02,.7,.02,.24],[.1,.1,.1,.1,.6],[.1,.1,.1,.1,.6]])
        i,_=select_view(p,['a','b','c','d']);cue,reason=choose_cue(p[i],.5,(0,2),(2,))
        self.assertIsNone(cue);self.assertEqual(reason,'illegal_top_direction')

    def results(self,clean_candidate=.9):
        def r(sr,sg):return dict(metrics=dict(episodes=140,successes=round(140*sr),sr=sr,mean_sg_all_episodes=sg,repeat_visit_rate_micro=.02,out_of_bounds_rate=0),
            by_source={f'img_{i}':dict(sr=sr,mean_sg_all_episodes=sg) for i in range(28)},
            by_distance={str(d):dict(sr=sr,mean_sg_all_episodes=sg) for d in range(4,9)})
        out={}
        for v in VARIANTS:
            for s in range(3):
                for a in STRATEGIES:
                    sr=clean_candidate if a=='Candidate4' else .9 if a=='Original' and v=='Clean' else .7
                    out[v,s,a]=r(sr,.3 if a=='Candidate4' or v=='Clean' and a=='Original' else .8)
        return out,{'Frontier':r(.5,2.),'FixedRegion':r(.3,2.5)}

    def test_improved_rotation_cannot_release_when_clean_damage_exceeds_two_points(self):
        r,rules=self.results(.88);self.assertTrue(summarize(r,rules,'development')['release_numeric_passed'])
        r,rules=self.results(.879);v=summarize(r,rules,'development')
        self.assertTrue(v['release_checks']['rotation_gain5pp']);self.assertFalse(v['release_checks']['clean_loss_at_most2pp']);self.assertFalse(v['release_numeric_passed'])

    def test_probabilities_must_be_four_finite_joint_vectors(self):
        with self.assertRaises(ValueError):select_view(np.zeros((4,5)),['a','b','c','d'])
        with self.assertRaises(ValueError):select_view(np.full((4,5),np.nan),['a','b','c','d'])


if __name__=='__main__':unittest.main()
