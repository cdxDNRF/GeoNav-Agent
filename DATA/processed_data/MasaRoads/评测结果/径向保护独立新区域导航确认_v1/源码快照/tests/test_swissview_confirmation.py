"""Cross-dataset protocol sizes, leakage boundary and original-policy equivalence."""
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
from data.swissview_grid import tasks,task_scope,PILOT_IDS,FORMAL_IDS
from agents.frozen_edge_navigator import FrozenEdgeNavigator
from agents.precomputed_frozen_edge import PrecomputedFrozenEdgeNavigator
from agents.edge_cue import EdgeTargetCueHead,image_profiles
from env.environment import Observation
from eval.swissview_confirmation import summarize


class Explorer(nn.Module):
    def step(self,x,hidden=None):return torch.tensor([[0.,1.,3.,2.]]),torch.zeros(1),torch.zeros(1,512),torch.zeros(1,256)


class SwissViewTests(unittest.TestCase):
    def sources(self,n):return [dict(area=f'img_{i}',source_tile=f'SwissView100/swisstopo_{i:02d}.jpg') for i in range(n)]

    def test_formal20maps500tasks_and_pilot20_disjoint_ids(self):
        self.assertFalse(set(PILOT_IDS)&set(FORMAL_IDS))
        self.assertEqual(task_scope(tasks(self.sources(20),5),'formal')['episodes'],500)
        self.assertEqual(task_scope(tasks(self.sources(4),1),'pilot')['episodes'],20)

    def test_missing_duplicate_or_collapsed_map_cannot_inflate_size(self):
        eps=tasks(self.sources(20),5)
        for bad in [eps[:-1],eps[:-1]+[eps[0]],[replace(e,source_tile='one_file') for e in eps]]:
            with self.assertRaises(ValueError):task_scope(bad,'formal')

    def test_per_map_distance_balance_is_required(self):
        eps=tasks(self.sources(20),5);eps[0]=replace(eps[5],episode_id=eps[0].episode_id)
        with self.assertRaises(ValueError):task_scope(eps,'formal')

    def test_sampling_per_map_does_not_depend_on_traversal_order(self):
        a=tasks(self.sources(4),1);b=tasks(list(reversed(self.sources(4))),1)
        self.assertEqual({e.episode_id:e for e in a},{e.episode_id:e for e in b})

    def setup_pair(self,condition='CueFull',arm='Edge'):
        torch.set_num_threads(1);torch.manual_seed(7);rng=np.random.default_rng(7)
        def payload(v):
            f=BytesIO();Image.fromarray(np.full((300,300,3),v,np.uint8)).save(f,format='PNG');return f.getvalue()
        obs=Observation(payload(77),payload(104),(2,2),5,10,(12,))
        means=[rng.normal(size=s).astype(np.float32) for s in [(512,),(512,),(4,768),(4,3,64,3)]]
        head=EdgeTargetCueHead().eval();a=FrozenEdgeNavigator(Explorer(),head,'cpu',*means,.5 if arm=='Edge' else None,arm,condition)
        b=PrecomputedFrozenEdgeNavigator(Explorer(),head,'cpu',*means,.5 if arm=='Edge' else None,arm,condition)
        features=[rng.normal(size=s).astype(np.float32) for s in [(512,),(4,768),(512,),(4,768)]]
        return a,b,obs,features

    def test_given_profiles_exactly_preserve_raw_profile_equations(self):
        for c in ('Baseline','CueFull','CueMean','CueWrong'):
            a,b,obs,x=self.setup_pair(c)
            for _ in range(2):self.assertEqual(a.act(obs,*x),b.act_with_profiles(obs,*x,image_profiles(obs.current_image),image_profiles(obs.target_image)))
            b.reset();self.assertEqual(b.profiles,{});self.assertIsNone(b.hidden)

    def test_zero_target_channels_still_abstain_with_original_none_threshold(self):
        a,b,obs,x=self.setup_pair(arm='ZeroEdge')
        self.assertEqual(a.act(obs,*x),b.act_with_profiles(obs,*x,image_profiles(obs.current_image),image_profiles(obs.target_image)))
        self.assertIsNone(b.threshold)

    def test_profile_api_rejects_privileged_bank_or_invalid_values(self):
        _,b,obs,x=self.setup_pair();p=image_profiles(obs.current_image)
        with self.assertRaises(TypeError):b.act_with_profiles(obs,*x,p,p,goal=7)
        with self.assertRaises(ValueError):b.act_with_profiles(obs,*x,np.full(p.shape,np.nan),p)

    def test_target_evidence_is_separate_from_engineering_and_CI_units(self):
        def r(sr,sg):return dict(metrics=dict(episodes=20,successes=int(sr*20),sr=sr,mean_sg_all_episodes=sg,repeat_visit_rate_micro=.2,out_of_bounds_rate=0),
            by_source={f'img_{i}':dict(sr=sr,mean_sg_all_episodes=sg) for i in range(4)},
            by_distance={str(d):dict(sr=sr,mean_sg_all_episodes=sg) for d in range(4,9)})
        rows={(a,s,c):r(.5,1.) for s in range(3) for a,cs in [('Edge',('Baseline','CueFull','CueMean','CueWrong')),('ZeroEdge',('CueFull',))] for c in cs}
        v=summarize(rows,{'Frontier':r(.4,2.),'FixedRegion':r(.1,3.)},'pilot')
        self.assertTrue(v['engineering_numeric_passed']);self.assertFalse(v['target_numeric_passed']);self.assertFalse(v['formal_cross_dataset_passed'])
        self.assertEqual(v['effects']['Frontier']['source_interval']['source_count'],4)


if __name__=='__main__':unittest.main()
