"""Meaningful checks for calibration leakage, abstention and actual recurrent navigation."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.target_cue import cue_features,choose_cue,TargetCueHead,FEATURE_WIDTH
from agents.spatial_relation import make_policy
from env.episode import Episode
from test_foundation import Fixture
from train.trusted_cue import (head_split,pair_bank,pair_inputs,calibrate,cue_metrics,precision_interval,
    train_head,navigation_run,opportunity_diagnostic)
from train.dyncur_tiny import set_seed


class CueCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(91)
        self.values=rng.normal(size=(25,512)).astype(np.float32)
        self.local={'img_0':rng.normal(size=(25,4,768)).astype(np.float32)}
        self.store=SimpleNamespace(data={'img_0':self.values},patch=lambda area,cell:self.values[cell])
        self.mean=self.values.mean(0);self.local_mean=self.local['img_0'].mean(0)

    def test_source_split_and_natural_pair_labels_are_exhaustive(self):
        fit,cal=head_split([f'img_{j}' for j in range(109)])
        self.assertEqual((len(fit),len(cal)),(87,22));self.assertFalse(set(fit)&set(cal))
        self.assertEqual((fit,cal),head_split([f'img_{j}' for j in reversed(range(109))]))
        bank=pair_bank(['img_0'])
        self.assertEqual(len(bank),600);self.assertEqual(len({(r['current'],r['target']) for r in bank}),600)
        self.assertEqual([sum(r['label']==c for r in bank) for c in range(5)],[20,20,20,20,520])
        for r in bank:
            self.assertNotIn(r['wrong'],(r['current'],r['target']))
            if r['label']<4:
                dr,dc=((-1,0),(0,1),(1,0),(0,-1))[r['label']]
                self.assertEqual(r['target'],(r['current']//5+dr)*5+r['current']%5+dc)

    def test_all_target_channels_change_together_and_mask_retains_true_labels(self):
        bank=[r for r in pair_bank(['img_0']) if r['current']==0]
        masked=pair_inputs(bank,self.store,self.local,self.mean,self.local_mean,'MeanCue')
        self.assertEqual(masked.shape,(24,FEATURE_WIDTH))
        np.testing.assert_array_equal(masked,np.repeat(masked[:1],24,axis=0))
        full=pair_inputs(bank,self.store,self.local,self.mean,self.local_mean,'Full')
        wrong=pair_inputs(bank,self.store,self.local,self.mean,self.local_mean,'WrongCue')
        np.testing.assert_array_equal(full[:,512:1024],wrong[:,512:1024])
        for i,r in enumerate(bank):
            expect=cue_features(self.values[r['wrong']],self.values[r['current']],self.local['img_0'][r['wrong']],self.local['img_0'][r['current']])
            np.testing.assert_array_equal(wrong[i],expect)
        self.assertFalse(np.array_equal(full[:,:512],wrong[:,:512]))
        self.assertFalse(np.array_equal(full[:,1024:],wrong[:,1024:]))

    def test_raw_probability_gate_never_renormalizes_or_selects_second_direction(self):
        probabilities=np.array([.91,.03,.02,.01,.03])
        self.assertEqual(choose_cue(probabilities,.9,(0,0),[0]),(None,'illegal_top_direction'))
        self.assertEqual(choose_cue(probabilities,None,(2,2),[12]),(None,'uncalibrated_abstain'))
        self.assertEqual(choose_cue(probabilities,.95,(2,2),[12]),(None,'low_confidence'))
        self.assertEqual(choose_cue(probabilities,.9,(2,2),[12,7]),(None,'visited_top_destination'))
        self.assertEqual(choose_cue(probabilities,.9,(2,2),[12]),('up','accepted'))
        self.assertEqual(choose_cue([.01,.01,.01,.01,.96],.5,(2,2),[12]),(None,'not_adjacent'))
        with self.assertRaises(ValueError):choose_cue([1,1,0,0,0],.9,(0,0),[0])

    def test_calibration_rejects_confident_false_neighbors_and_uses_lowest_passing_threshold(self):
        bank=pair_bank([f'img_{j}' for j in range(7)])
        p=np.full((len(bank),5),.0125)
        for i,r in enumerate(bank):
            if r['label']<4:p[i,r['label']]=.95
            else:p[i]=[.6,.1,.1,.1,.1]
        chosen=calibrate(p,bank)
        self.assertEqual(chosen['threshold'],.7)
        self.assertFalse(chosen['grid'][0]['passed'])
        selected=cue_metrics(p,bank,.7)
        self.assertEqual(selected['accepted_precision'],1)
        self.assertEqual(selected['adjacent_recall'],1)
        self.assertEqual(selected['sources_with_acceptances'],7)
        p[:]=[.01,.01,.01,.01,.96]
        self.assertIsNone(calibrate(p,bank)['threshold'])
        metrics=cue_metrics(p,bank,None)
        self.assertEqual(metrics['accepted'],0);self.assertIsNone(metrics['accepted_precision'])
        self.assertAlmostEqual(metrics['class_accuracy'],520/600)

    def test_bootstrap_keeps_sources_with_no_accepted_cue(self):
        result=precision_interval(np.ones(10,bool),np.array([True]+[False]*9),[f'img_{j}' for j in range(10)])
        self.assertEqual(result['source_count'],10)
        self.assertGreater(result['zero_accept_resamples'],0)
        self.assertEqual(result['interval95'][0],0)

    def test_features_are_scale_invariant_and_real_training_updates_five_class_head(self):
        bank=pair_bank(['img_0'])[:16]
        features=pair_inputs(bank,self.store,self.local,self.mean,self.local_mean,'Full')
        a=cue_features(self.values[1],self.values[0],self.local['img_0'][1],self.local['img_0'][0])
        b=cue_features(self.values[1]*2,self.values[0]*3,self.local['img_0'][1]*4,self.local['img_0'][0]*5)
        np.testing.assert_allclose(a,b,atol=1e-6)
        set_seed(0);initial=copy.deepcopy(TargetCueHead().state_dict())
        with tempfile.TemporaryDirectory() as temporary,patch('train.trusted_cue.EPOCHS',2):
            model=train_head(features,np.array([r['label'] for r in bank]),Path(temporary),0,torch.device('cpu'))
            self.assertEqual(sum(p.numel() for p in model.parameters()),134277)
            self.assertTrue(any(not torch.equal(v,model.state_dict()[n]) for n,v in initial.items()))
            self.assertEqual(model(torch.tensor(features)).shape,(16,5))


class CueNavigationTests(Fixture):
    def setUp(self):
        super().setUp();torch.set_num_threads(1)
        rng=np.random.default_rng(52)
        values=rng.normal(size=(25,512)).astype(np.float32)
        self.store=SimpleNamespace(data={'img_0':values},patch=lambda area,cell:values[cell])
        self.local={'img_0':rng.normal(size=(25,4,768)).astype(np.float32)}
        self.mean=values.mean(0);self.local_mean=self.local['img_0'].mean(0)

    def run_condition(self,condition,threshold,explorer,head):
        ep=self.ep(start=0,goal=8)
        with patch('train.trusted_cue.MASA',self.root):
            return navigation_run(explorer,head,threshold,condition,ep,self.store,self.local,self.mean,self.mean,
                                  self.local_mean,{ep.episode_id:{'cue_cell':24}},torch.device('cpu'))

    def test_null_gate_is_identical_across_goal_controls_and_baseline(self):
        set_seed(0);explorer=make_policy('Small256').eval();head=TargetCueHead().eval()
        baseline=self.run_condition('Baseline',None,explorer,head)
        for c in ('CueFull','CueMean','CueWrong'):
            result=self.run_condition(c,None,explorer,head)
            self.assertEqual(result['trajectory'],baseline['trajectory'])
            self.assertEqual(result['success'],baseline['success'])
            self.assertTrue(all(d['cue_action'] is None for d in result['decisions']))

    def test_explorer_hidden_state_advances_even_when_cue_changes_action(self):
        class TrackedExplorer(torch.nn.Module):
            def __init__(self):super().__init__();self.carries=[]
            def step(self,x,hidden):
                value=0 if hidden is None else int(hidden[0,0])
                self.carries.append(value)
                row,col=round(float(x[0,1024])*4),round(float(x[0,1025])*4)
                logits=torch.tensor([[0.,10.,0.,0.]])
                legal=torch.tensor([[row>0,col<4,row<4,col>0]])
                logits.masked_fill_(~legal,torch.finfo(logits.dtype).min)
                return logits,None,None,torch.tensor([[value+1.]])
        class DownCue(torch.nn.Module):
            def forward(self,x):return torch.tensor([[0.,0.,10.,0.,0.]])
        explorer=TrackedExplorer();result=self.run_condition('CueFull',.9,explorer,DownCue())
        self.assertEqual(explorer.carries,list(range(result['steps'])))
        self.assertTrue(any(d['cue_action'] and d['action']!=d['explorer_action'] for d in result['decisions']))
        self.assertEqual(result['out_of_bounds'],0)

    def test_opportunity_diagnostic_excludes_post_budget_terminal_position(self):
        ep=self.ep(start=0,goal=8)
        record={'episode_id':ep.episode_id,'success':False,'distance':ep.dist,
                'trajectory':[{'patch_id':0},{'patch_id':1},{'patch_id':7}]}
        # Final position7 is adjacent to8, but there is no next action available.
        result=opportunity_diagnostic([record],[ep])
        self.assertEqual(result['all']['failed_with_adjacent_opportunity'],0)
        record['trajectory']=[{'patch_id':0},{'patch_id':7},{'patch_id':12}]
        result=opportunity_diagnostic([record],[ep])
        self.assertEqual(result['all']['opportunity_only_at_final_action'],1)
        with self.assertRaises(ValueError):opportunity_diagnostic([], [ep])


if __name__=='__main__':unittest.main()
