"""Goal labels remain trainer-side; changing cue never changes evaluation goal."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_foundation import Fixture
from agents.boundary_policy import BoundaryPolicy, boundary_mask
from env.episode import manhattan
from train.curiosity_controlled import Dynamics, train_arm
from train.target_pairs import paired_records, materialize, closer_actions, paired_logits, supervised_goal_loss, paired_diagnostic
from train.target_controlled import train_target, wrong_cue_plan, evaluate_cue


class TargetTests(Fixture):
    def store(self):
        class Store:
            data={'img_0':np.random.default_rng(12).normal(size=(25,512)).astype(np.float32)}
            def patch(self,area,cell):return self.data[area][cell]
        return Store()

    def test_pair_history_legal_same_distance_and_disjoint_directions(self):
        records=paired_records(['img_0'],64,19)
        self.assertEqual(records,paired_records(['img_0'],64,19))
        for record in records:
            h=record['history'];a,b=record['goals']
            self.assertNotIn(a,h);self.assertNotIn(b,h)
            self.assertTrue(all(manhattan(x,y)==1 for x,y in zip(h,h[1:])))
            self.assertEqual(manhattan(h[-1],a),manhattan(h[-1],b))
            self.assertFalse((closer_actions(h[-1],a)&closer_actions(h[-1],b)).any())

    def test_only_target_features_differ_and_remain_constant_over_history(self):
        store=self.store();records=paired_records(store.data,16,21)
        x,y=materialize(records,store)
        self.assertEqual(x.shape,(16,2,3,1052))
        np.testing.assert_array_equal(x[:,0,:,512:],x[:,1,:,512:])
        for i,r in enumerate(records):
            for j,g in enumerate(r['goals']):
                for t in range(3):np.testing.assert_array_equal(x[i,j,t,:512],store.patch(r['area'],g))
        np.testing.assert_allclose(y.sum(-1),1)
        legal=boundary_mask(torch.tensor(x[:,:,2])).numpy()
        self.assertFalse((y[~legal]>0).any())

    def test_swapped_goal_order_swaps_logits_without_history_contamination(self):
        store=self.store();x,_=materialize(paired_records(store.data,4,21),store)
        model=BoundaryPolicy();x=torch.tensor(x)
        a=paired_logits(model,x);b=paired_logits(model,x.flip(1))
        self.assertTrue(torch.allclose(a,b.flip(1),atol=1e-6))

    def test_auxiliary_loss_has_finite_gradient_through_earlier_observations(self):
        store=self.store();x,y=materialize(paired_records(store.data,4,21),store)
        x=torch.tensor(x,requires_grad=True);y=torch.tensor(y)
        loss=supervised_goal_loss(BoundaryPolicy(),x,y);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[:,:,0,:512].abs().sum()),0)

    def test_goal_blind_policy_cannot_get_both_disjoint_targets_right(self):
        class Blind(BoundaryPolicy):
            def step(self,features,hidden=None):
                features=features.clone();features[:,:512]=0
                return super().step(features,hidden)
        store=self.store();x,y=materialize(paired_records(store.data,16,21),store)
        result=paired_diagnostic(Blind(),torch.tensor(x),torch.tensor(y))
        self.assertEqual(result['action_change_rate'],0)
        self.assertEqual(result['both_correct_rate'],0)

    def test_zero_weight_training_exactly_reproduces_existing_PPO(self):
        torch.set_num_threads(1)
        store=self.store();x,y=materialize(paired_records(store.data,64,21),store)
        initial=BoundaryPolicy();dynamics=Dynamics(initial).requires_grad_(False)
        old=self.root/'old';old.mkdir();new=self.root/'new';new.mkdir()
        a=train_arm(initial,dynamics,.1,store,old,'PBRS',2,'cpu',updates=1)
        b=train_target(initial,store,torch.tensor(x),torch.tensor(y),new,2,0,'cpu',updates=1)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

    def test_nonzero_supervision_changes_weights_with_same_training_budget(self):
        torch.set_num_threads(1)
        store=self.store();x,y=materialize(paired_records(store.data,64,21),store)
        initial=BoundaryPolicy();a=self.root/'a';a.mkdir();b=self.root/'b';b.mkdir()
        control=train_target(initial,store,torch.tensor(x),torch.tensor(y),a,0,0,'cpu',updates=1)
        candidate=train_target(initial,store,torch.tensor(x),torch.tensor(y),b,0,.5,'cpu',updates=1)
        self.assertTrue(any(not torch.equal(v,control.state_dict()[k]) for k,v in candidate.state_dict().items()))

    def test_wrong_cue_plan_is_reproducible_and_does_not_change_task_goals(self):
        eps=[replace(self.ep(goal=g),episode_id=f'ep{g}') for g in (4,9,14,19,24)]
        plan=wrong_cue_plan(eps)
        self.assertEqual(plan,wrong_cue_plan(list(reversed(eps))))
        for ep in eps:
            self.assertNotIn(plan[ep.episode_id]['cue_cell'],(ep.goal,ep.start))
            if plan[ep.episode_id]['matched_distance']:
                self.assertEqual(manhattan(ep.start,plan[ep.episode_id]['cue_cell']),ep.dist)
        store=self.store();model=BoundaryPolicy();directory=self.root/'eval';directory.mkdir()
        torch.save(model.state_dict(),directory/'model.pt')
        result=evaluate_cue(model,self.root,eps,store,np.zeros(512,np.float32),plan,directory,'错误目标','cpu')
        import json
        rows=[json.loads(x) for x in (directory/'val轨迹.jsonl').read_text(encoding='utf-8').splitlines()]
        for ep,row in zip(eps,rows):
            final=row['trajectory'][-1]['patch_id']
            self.assertEqual(row['sg'],manhattan(final,ep.goal))
            self.assertEqual(row['success'],final==ep.goal)
        self.assertTrue(result['audit']['neural_replay'])
        self.assertEqual(result['completed'],5)


if __name__=='__main__':unittest.main()
