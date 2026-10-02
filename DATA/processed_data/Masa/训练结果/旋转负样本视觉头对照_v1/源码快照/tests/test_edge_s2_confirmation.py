"""Protocol-critical split routing, public cue masking, frozen replay and S2 gates."""
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.edge_cue import EdgeTargetCueHead,edge_features,image_profiles
from agents.frozen_edge_navigator import FrozenEdgeNavigator,load_frozen_edge_default
from agents.spatial_relation import make_policy
from agents.target_cue import cue_features
from env.environment import Observation,image_payload
from env.episode import Episode
from eval.edge_s2_confirmation import given_target_payload,summarize


def payload(value):
    out=BytesIO();Image.fromarray(np.full((300,300,3),value,np.uint8)).save(out,format='PNG');return out.getvalue()


class Explorer(nn.Module):
    def step(self,x,hidden=None):
        hidden=torch.zeros(1,1) if hidden is None else hidden
        return torch.tensor([[0.,3.,1.,2.]]),torch.zeros(1),torch.zeros(1,512),hidden+1


class ConfidentCue(nn.Module):
    def forward(self,x):return torch.tensor([[10.,-10.,-10.,-10.,-10.]])


class EdgeS2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        rng=np.random.default_rng(42)
        self.current=rng.normal(size=512).astype(np.float32);self.target=rng.normal(size=512).astype(np.float32)
        self.cl=rng.normal(size=(4,768)).astype(np.float32);self.tl=rng.normal(size=(4,768)).astype(np.float32)
        self.means=[rng.normal(size=512).astype(np.float32),rng.normal(size=512).astype(np.float32),
            rng.normal(size=(4,768)).astype(np.float32),image_profiles(np.full((300,300,3),50,np.uint8))]
        self.obs=Observation(payload(90),payload(170),(1,1),5,10,(6,))

    def agent(self,condition='CueFull',arm='Edge',head=None,explorer=None,threshold=.5):
        return FrozenEdgeNavigator(explorer or Explorer(),head or EdgeTargetCueHead(),torch.device('cpu'),
            *self.means,threshold,arm,condition)

    def call(self,a,obs=None,target=None,local=None):
        return a.act(obs or self.obs,self.current,self.cl,self.target if target is None else target,self.tl if local is None else local)

    def test_wrong_target_pixels_use_episode_split_with_colliding_area_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split,value in [('val',31),('train',211)]:
                p=root/'patches'/split/'img_0';p.mkdir(parents=True)
                Image.fromarray(np.full((300,300,3),value,np.uint8)).save(p/'patch_3.jpg')
            ep=Episode('probe','val','img_0',0,4,4)
            wrong={'probe':dict(cue_cell=3,matched_distance=False)}
            actual=given_target_payload(ep,'CueWrong',wrong,root)
            self.assertEqual(actual,image_payload(root/'patches/val/img_0/patch_3.jpg'))
            self.assertNotEqual(actual,image_payload(root/'patches/train/img_0/patch_3.jpg'))

    def test_full_public_wrapper_matches_existing_image_feature_recipe(self):
        a=self.agent();result=self.call(a)
        semantic=cue_features(self.target,self.current,self.tl,self.cl)
        seam=edge_features(image_profiles(self.obs.target_image),image_profiles(self.obs.current_image))
        expected=np.concatenate((semantic,seam)).astype(np.float32)
        self.assertEqual(result['cue_features_sha256'],sha256(expected.tobytes()).hexdigest())
        with torch.no_grad():np.testing.assert_array_equal(result['probabilities'],a.head(torch.tensor(expected)[None]).softmax(-1)[0].numpy())

    def test_mean_target_masks_global_local_and_edge_information(self):
        a=self.agent('CueMean');first=self.call(a);a.reset()
        second=self.call(a,replace(self.obs,target_image=payload(2)),self.target*17,self.tl*-4)
        self.assertEqual(first['cue_features_sha256'],second['cue_features_sha256'])
        self.assertEqual(first['probabilities'],second['probabilities'])
        self.assertEqual(first['action'],second['action'])
        expected=np.concatenate((cue_features(self.means[1],self.current,self.means[2],self.cl),
                                 edge_features(self.means[3],image_profiles(self.obs.current_image)))).astype(np.float32)
        self.assertEqual(first['cue_features_sha256'],sha256(expected.tobytes()).hexdigest())

    def test_baseline_disables_even_a_high_confidence_legal_cue(self):
        baseline=self.call(self.agent('Baseline',head=ConfidentCue()))
        full=self.call(self.agent(head=ConfidentCue()))
        self.assertEqual((baseline['cue_action'],baseline['action'],baseline['reason']),(None,'right','uncalibrated_abstain'))
        self.assertEqual((full['cue_action'],full['action']),('up','up'))

    def test_zero_edge_has_matched_features_and_original_none_threshold(self):
        a=self.agent(arm='ZeroEdge',threshold=None);result=self.call(a)
        expected=np.concatenate((cue_features(self.target,self.current,self.tl,self.cl),np.zeros(20,np.float32))).astype(np.float32)
        self.assertEqual(result['cue_features_sha256'],sha256(expected.tobytes()).hexdigest())
        self.assertIsNone(result['cue_action']);self.assertEqual(result['action'],result['explorer_action'])

    def test_per_episode_reset_reproduces_recurrent_actions(self):
        a=self.agent(explorer=make_policy('Small256'))
        first=self.call(a);self.call(a);a.reset();second=self.call(a)
        self.assertEqual(first,second);self.assertEqual(second['step'],1)

    def test_S2_gate_requires_all_target_comparisons_and_all_distance_bands(self):
        def run(sr,sg=1.):
            m=dict(episodes=100,successes=round(sr*100),sr=sr,mean_sg_all_episodes=sg,
                repeat_visit_rate_micro=.2,out_of_bounds_rate=0.)
            return dict(metrics=m,by_distance={str(d):dict(sr=sr,mean_sg_all_episodes=sg) for d in range(4,9)},
                by_source={f'img_{i}':dict(sr=sr,mean_sg_all_episodes=sg) for i in range(4)})
        records={}
        for s in range(3):
            for c in ('Baseline','CueFull','CueMean','CueWrong'):records['Edge',s,c]=run(.7 if c=='CueFull' else .6)
            records['ZeroEdge',s,'CueFull']=run(.6)
        rule=dict(sr=.5,mean_sg_all_episodes=2.)
        result=summarize(records,{'Frontier':rule,'FixedRegion':dict(sr=.3,mean_sg_all_episodes=3.)})
        self.assertTrue(result['formal_S2_numeric_checks_passed'])
        # v2 diagnostics do not silently become v1 rejection gates.
        self.assertFalse(result['diagnostics']['v1_revisit_10pct'])
        for s in range(3):records['Edge',s,'CueWrong']=run(.7)
        self.assertFalse(summarize(records,{'Frontier':rule})['formal_S2_numeric_checks_passed'])
        for s in range(3):
            records['Edge',s,'CueWrong']=run(.6)
            records['Edge',s,'CueFull']['by_distance']['4']['sr']=.2
        self.assertFalse(summarize(records,{'Frontier':rule})['formal_S2_numeric_checks_passed'])

    def test_default_loader_checks_frozen_hash_and_registered_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);torch.save(make_policy('Small256').state_dict(),root/'explorer.pt')
            torch.save(EdgeTargetCueHead().state_dict(),root/'head.pt')
            names=('全局拟合均值.npy','头部拟合全局均值.npy','头部拟合局部均值.npy','头部拟合边缘均值.npy')
            for name,value in zip(names,self.means):np.save(root/name,value)
            def rec(name):return dict(path=name,sha256=sha256((root/name).read_bytes()).hexdigest())
            config=dict(version='local-policy-default-v2',selected_arm='EdgeTargetCue',
                protocol=dict(grid_size=5,budget=10,decode='argmax',boundary_filter=True),
                checkpoints=[dict(seed=s,**rec('explorer.pt')) for s in range(3)],
                cue_heads=[dict(seed=s,**rec('head.pt')) for s in range(3)],
                means={n:rec(n) for n in names},thresholds={str(s):.5 for s in range(3)})
            a=load_frozen_edge_default(config,root,2,'cpu');self.assertEqual(a.threshold,.5)
            with self.assertRaises(ValueError):load_frozen_edge_default(config,root,3,'cpu')
            config['cue_heads'][2]['sha256']='0'*64
            with self.assertRaises(ValueError):load_frozen_edge_default(config,root,2,'cpu')


if __name__=='__main__':unittest.main()
