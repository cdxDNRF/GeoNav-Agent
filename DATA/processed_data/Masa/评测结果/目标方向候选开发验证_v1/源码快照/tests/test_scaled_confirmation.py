"""Formal scale/sample gates and given-image profile execution equivalence."""
from dataclasses import replace
from pathlib import Path
from io import BytesIO
import sys
import unittest
import numpy as np
from PIL import Image
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.scaled_masa import scaled_tasks
from eval.scaled_edge_confirmation import formal_task_info,engineering_checks,summarize
from agents.scaled_edge_navigator import ScaledEdgeNavigator
from agents.precomputed_scaled_edge import PrecomputedScaledEdgeNavigator
from agents.edge_cue import EdgeTargetCueHead,image_profiles
from env.environment import Observation


def run(sr=.3,sg=4.5,n=500):
    return dict(metrics=dict(episodes=n,successes=round(sr*n),sr=sr,mean_sg_all_episodes=sg,repeat_visit_rate_micro=.2),
        by_source={f'raw{i}':dict(sr=sr,mean_sg_all_episodes=sg) for i in range(20)},
        by_distance={str(d):dict(sr=sr,mean_sg_all_episodes=sg) for d in range(12,17)})


class Explorer(nn.Module):
    def step(self,x,hidden=None):return torch.tensor([[0.,1.,3.,2.]]),torch.zeros(1),torch.zeros(1,512),torch.zeros(1,256)


class ScaledConfirmationTests(unittest.TestCase):
    def sources(self):return [dict(split='test' if i<10 else 'dev',area=f'img_{i}',source_tile=f'raw{i}') for i in range(20)]

    def test_formal_task_count_requires500_balanced_draws_on20_sources(self):
        eps=scaled_tasks(self.sources(),5);v=formal_task_info(eps)
        self.assertEqual(v['tasks'],500);self.assertEqual(v['sources'],20);self.assertEqual(v['unique_test_sources'],10);self.assertEqual(v['unique_dev_sources'],10)

    def test_source_and_episode_duplicates_do_not_inflate_formal_size(self):
        eps=scaled_tasks(self.sources(),5)
        with self.assertRaises(ValueError):formal_task_info(eps[:-1])
        with self.assertRaises(ValueError):formal_task_info(eps[:-1]+[eps[0]])
        eps=[replace(e,source_tile='same_map') for e in eps]
        with self.assertRaises(ValueError):formal_task_info(eps)

    def test_formal_each_source_C_balance_is_required(self):
        eps=scaled_tasks(self.sources(),5);eps[0]=replace(eps[5],episode_id=eps[0].episode_id)
        with self.assertRaises(ValueError):formal_task_info(eps)

    def test_formal_effect_thresholds_are_inclusive_and_revisit_is_diagnostic(self):
        checks=engineering_checks([run() for _ in range(3)],run(.25,5.))
        self.assertTrue(all(checks.values()))
        self.assertFalse(engineering_checks([run(sg=4.501) for _ in range(3)],run(.25,5.))['SG_4p5'])

    def test_missing_records_or_fewer_sources_cannot_pass_formal(self):
        self.assertFalse(engineering_checks([run(n=499) for _ in range(3)],run(.25,5.))['all_500_normal_terminals'])
        x=[run() for _ in range(3)]
        for r in x:r['by_source'].pop('raw19')
        self.assertFalse(engineering_checks(x,run(.25,5.))['source_count_20'])

    def test_visual_target_evidence_stays_separate_from_engineering(self):
        records={}
        for s in range(3):
            for c in ('Baseline','CueFull','CueMean','CueWrong'):records['Edge',s,c]=run()
            records['ZeroEdge',s,'CueFull']=run()
        summary=summarize(records,{'Frontier':run(.25,5.),'FixedRegion':run(.1,6.)},{f'raw{i}':'test' if i<10 else 'dev' for i in range(20)})
        self.assertTrue(summary['engineering_numeric_passed']);self.assertFalse(summary['target_numeric_passed']);self.assertFalse(summary['formal_S4_passed'])
        self.assertEqual(summary['effects']['Frontier']['source_interval']['source_count'],20)

    def setup_agents(self,condition='CueFull',arm='Edge'):
        torch.set_num_threads(1);torch.manual_seed(7);rng=np.random.default_rng(7)
        def payload(v):
            out=BytesIO();Image.fromarray(np.full((300,300,3),v,np.uint8)).save(out,format='PNG');return out.getvalue()
        obs=Observation(payload(77),payload(155),(2,3),10,20,(23,))
        means=[np.zeros(512,np.float32),np.ones(512,np.float32),np.ones((4,768),np.float32),image_profiles(payload(100))]
        head=EdgeTargetCueHead();explorer=Explorer();threshold=.5 if arm=='Edge' else None
        a=ScaledEdgeNavigator(explorer,head,'cpu',*means,threshold,arm,condition)
        b=PrecomputedScaledEdgeNavigator(explorer,head,'cpu',*means,threshold,arm,condition)
        x=[rng.normal(size=512).astype(np.float32),rng.normal(size=(4,768)).astype(np.float32),rng.normal(size=512).astype(np.float32),rng.normal(size=(4,768)).astype(np.float32)]
        return a,b,obs,x

    def test_precomputed_given_profiles_match_canonical_with_all_masks(self):
        for arm in ('Edge','ZeroEdge'):
            for c in ('Baseline','CueFull','CueMean','CueWrong'):
                a,b,obs,x=self.setup_agents(c,arm)
                expected=a.act(obs,*x);actual=b.act_with_profiles(obs,*x,image_profiles(obs.current_image),image_profiles(obs.target_image))
                self.assertEqual(actual,expected);b.reset();self.assertEqual(b.profiles,{})

    def test_invalid_profiles_and_extra_privileged_fields_are_rejected(self):
        a,b,obs,x=self.setup_agents();p=image_profiles(obs.current_image)
        with self.assertRaises(ValueError):b.act_with_profiles(obs,*x,np.zeros((4,64,3)),p)
        bad=p.copy();bad[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):b.act_with_profiles(obs,*x,bad,p)
        with self.assertRaises(TypeError):b.act_with_profiles(obs,*x,p,p,goal_coordinate=(9,9))


if __name__=='__main__':unittest.main()
