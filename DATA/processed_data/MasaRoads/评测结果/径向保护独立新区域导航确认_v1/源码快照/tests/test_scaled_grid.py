"""Protocol-critical density-scale behavior and frozen-input regressions."""
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.scaled_edge_navigator import scaled_policy_features,scaled_cue_choice
from agents.target_cue import choose_cue
from agents.governor import HierarchicalSearchGovernor,COARSE_REGIONS
from env.environment import Observation
from env.episode import ACTIONS
from env.scaled_grid import ScaledEpisode,ScaledGridEnv,fixed_region_action
from data.scaled_masa import scaled_crop,scaled_tasks,wrong_plan
from train.dyncur_tiny import policy_features
from eval.scaled_edge_pilot import summary_for


class ScaledGridTests(unittest.TestCase):
    def test_grid5_inputs_exactly_match_legacy_with_repeated_visits(self):
        rng=np.random.default_rng(42);t=rng.normal(size=512).astype(np.float32);c=rng.normal(size=512).astype(np.float32)
        for pos in ((0,0),(2,3),(4,4)):
            for b in (10,3,1):
                v=(0,24,0,13,13,13,13)
                np.testing.assert_array_equal(scaled_policy_features(t,c,pos,b,v,5,10),policy_features(t,c,pos,b,v))

    def test_grid10_visit_pool_is_fixed25_slots_with_declared_aliasing(self):
        z=np.zeros(512,np.float32);x=scaled_policy_features(z,z,(9,9),20,(0,1,10,11,99),10,20)
        self.assertEqual(x.shape,(1052,));np.testing.assert_array_equal(x[1024:1027],[1,1,1])
        self.assertEqual(x[1027],1.);self.assertAlmostEqual(x[-1],1/3)
        a=scaled_policy_features(z,z,(9,9),20,(0,99),10,20)
        b=scaled_policy_features(z,z,(9,9),20,(11,99),10,20)
        np.testing.assert_array_equal(a,b) # Explicitly lost within2x2 location information.

    def test_invalid_state_protocol_is_rejected(self):
        z=np.zeros(512,np.float32)
        for k,b,pos,v in [(10,10,(0,0),(0,)),(5,20,(0,0),(0,)),(10,20,(10,0),(0,)),(10,20,(0,0),(100,))]:
            with self.assertRaises(ValueError):scaled_policy_features(z,z,pos,1,v,k,b)

    def test_grid10_cue_keeps_top_only_and_correct_fine_cell_boundaries(self):
        right=np.array([.1,.7,.05,.05,.1])
        self.assertEqual(scaled_cue_choice(right,.5,(0,8),(8,),10),('right','accepted'))
        self.assertEqual(scaled_cue_choice(right,.5,(0,9),(9,),10),(None,'illegal_top_direction'))
        self.assertEqual(scaled_cue_choice(right,.5,(0,8),(8,9),10),(None,'visited_top_destination'))
        up=np.array([.6,.3,.03,.03,.04])
        self.assertEqual(scaled_cue_choice(up,.5,(0,8),(8,),10),(None,'illegal_top_direction'))

    def test_grid5_cue_decisions_match_legacy(self):
        values=[np.array([.7,.1,.05,.05,.1]),np.array([.05,.6,.1,.1,.15]),np.array([.1,.1,.1,.1,.6]),np.ones(5)/5]
        for pos in ((0,0),(2,3),(4,4)):
            for v in values:
                for threshold in (.5,None):
                    visited=(0,1,24,13)
                    self.assertEqual(scaled_cue_choice(v,threshold,pos,visited,5),choose_cue(v,threshold,pos,visited))

    def test_grid5_fixed_region_matches_original_across_public_histories(self):
        for cell in range(25):
            for visited in ((cell,),tuple(range(cell+1)),(24,13,0,cell,cell)):
                obs=Observation(b'',b'',divmod(cell,5),5,10,visited)
                self.assertEqual(fixed_region_action(obs),HierarchicalSearchGovernor().choose(obs,tuple(ACTIONS),COARSE_REGIONS).selected_action)

    def test_scaled_episode_validates_distance_and_prevents_old_grid_mixup(self):
        ep=ScaledEpisode('x','test','img_0',0,99,18);ep.validate()
        for bad in (replace(ep,goal=100),replace(ep,dist=17),replace(ep,grid_size=5),replace(ep,budget=32),replace(ep,area='../img_0')):
            with self.assertRaises(ValueError):bad.validate()

    def test_crop_is_cell_local_and_does_not_use_neighbor_pixels_for_upsampling(self):
        a=np.zeros((1500,1500,3),np.uint8);a[300:450,600:750]=[33,79,151]
        b=np.full_like(a,231);b[300:450,600:750]=a[300:450,600:750]
        actual=np.asarray(scaled_crop(Image.fromarray(a),24));expected=np.asarray(scaled_crop(Image.fromarray(b),24))
        np.testing.assert_array_equal(actual,expected);self.assertEqual(actual.shape,(300,300,3))

    def test_sampler_does_not_depend_on_source_order_and_wrong_plan_excludes_truth(self):
        sources=[dict(split='test',area=f'img_{i}',source_tile=f'raw{i}.png') for i in range(2)]
        a=scaled_tasks(sources);b=scaled_tasks(list(reversed(sources)))
        self.assertEqual({e.episode_id:e for e in a},{e.episode_id:e for e in b})
        wrong=wrong_plan(a)
        for e in a:self.assertNotIn(wrong[e.episode_id]['cue_cell'],(e.start,e.goal))

    def data(self,root):
        p=Path(root)/'patches/test/img_0';p.mkdir(parents=True)
        im=Image.new('RGB',(300,300),(60,70,80))
        for j in range(100):im.save(p/f'patch_{j}.jpg')

    def test_wall_consumes_step_and_twentieth_step_success_has_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.data(tmp);env=ScaledGridEnv(tmp);env.reset(ScaledEpisode('x','test','img_0',0,99,18))
            for action in ['up','left']+['down']*9+['right']*9:obs,done,info=env.step(action)
            row=env.evaluator_result();self.assertTrue(row['success']);self.assertEqual(row['steps'],20)
            self.assertEqual(row['out_of_bounds'],2);self.assertEqual(row['revisits'],2);self.assertEqual(row['sg'],0)
            with self.assertRaises(RuntimeError):env.step('up')

    def test_environment_rejects_patch_modified_after_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.data(tmp);env=ScaledGridEnv(tmp);env.reset(ScaledEpisode('x','test','img_0',0,99,18))
            Image.new('RGB',(300,300),(180,180,180)).save(Path(tmp)/'patches/test/img_0/patch_1.jpg')
            with self.assertRaises(ValueError):env.step('right')

    def test_pilot_effect_pass_does_not_claim_formal_S4_size_and_revisit_is_diagnostic(self):
        def run(sr,sg):
            return dict(metrics=dict(episodes=20,successes=round(sr*20),sr=sr,mean_sg_all_episodes=sg,repeat_visit_rate_micro=.2),
                by_source={f's{i}':dict(sr=sr,mean_sg_all_episodes=sg) for i in range(4)},
                by_distance={str(d):dict(sr=sr,mean_sg_all_episodes=sg) for d in range(12,17)})
        records={}
        for s in range(3):
            for c in ('Baseline','CueFull','CueMean','CueWrong'):records['Edge',s,c]=run(.4 if c=='CueFull' else .25,4.)
            records['ZeroEdge',s,'CueFull']=run(.25,4.)
        rules={'Frontier':run(.3,5.),'FixedRegion':run(.2,6.)}
        summary=summary_for(records,rules)
        self.assertTrue(summary['allow_formal_candidate_numeric']);self.assertFalse(summary['formal_S4_passed']);self.assertFalse(summary['formal_size_met'])
        for s in range(3):records['Edge',s,'CueFull']=run(.4,4.6)
        self.assertFalse(summary_for(records,rules)['numeric_scale_passed'])


if __name__=='__main__':unittest.main()
