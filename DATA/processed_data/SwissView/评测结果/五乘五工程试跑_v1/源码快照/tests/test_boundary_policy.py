"""Boundary policy checks against actual environment and recurrent PPO paths."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_foundation import Fixture
from agents.boundary_policy import BoundaryPolicy, boundary_mask
from train.dyncur_tiny import TinyPolicy
from train.curiosity_controlled import FeatureWorld, Dynamics, train_arm, recurrent_logits, evaluate_and_audit
from train.boundary_controlled import preparation_checks


class BoundaryTests(Fixture):
    def features(self):
        x=torch.zeros(25,1052)
        x[:,1024]=torch.arange(25)//5/4
        x[:,1025]=torch.arange(25)%5/4
        return x

    def store(self):
        class Store:
            data={'img_0':np.random.default_rng(71).normal(size=(25,512)).astype(np.float32)}
            def patch(self, area, cell):
                return self.data[area][cell]
        return Store()

    def test_all_cells_actions_match_actual_environment(self):
        from env.episode import ACTIONS
        mask=boundary_mask(self.features())
        env=self.env()
        for cell in range(25):
            for action_index,action in enumerate(ACTIONS):
                env.reset(self.ep(start=cell,goal=(cell+12)%25))
                _,_,info=env.step(action)
                self.assertEqual(bool(mask[cell,action_index]),not info.out_of_bounds)
        self.assertTrue(bool(mask.any(-1).all()))

    def test_filter_reads_only_public_position(self):
        x=self.features()
        changed=torch.randn_like(x)
        changed[:,1024:1026]=x[:,1024:1026]
        self.assertTrue(torch.equal(boundary_mask(x),boundary_mask(changed)))

    def test_legal_probabilities_renormalized_and_invalid_gradient_zero(self):
        model=BoundaryPolicy()
        x=self.features()
        pi,_,_,_=model.step(x)
        dist=Categorical(logits=pi)
        self.assertTrue(bool((dist.probs[~boundary_mask(x)]==0).all()))
        self.assertTrue(torch.allclose(dist.probs.sum(-1),torch.ones(25)))
        chosen=dist.sample()
        (-dist.log_prob(chosen).mean()-.01*dist.entropy().mean()).backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
        # At the top-left cell, the masked action logits have no policy gradient.
        raw=torch.randn(1,4,requires_grad=True)
        masked=raw.masked_fill(~boundary_mask(x[:1]),torch.finfo(raw.dtype).min)
        loss=-Categorical(logits=masked).log_prob(torch.tensor([1])).sum()
        loss.backward()
        self.assertEqual(raw.grad[0,0].item(),0)
        self.assertEqual(raw.grad[0,3].item(),0)

    def test_disabled_identical_and_filter_preserves_hidden_and_value(self):
        reference=TinyPolicy()
        model=BoundaryPolicy(enabled=False)
        model.load_state_dict(reference.state_dict())
        x=self.features()
        a=reference.step(x);b=model.step(x)
        self.assertTrue(all(torch.equal(u,v) for u,v in zip(a,b)))
        model.enabled=True
        c=model.step(x)
        self.assertTrue(all(torch.equal(u,v) for u,v in zip(a[1:],c[1:])))
        self.assertTrue(torch.equal(a[0][boundary_mask(x)],c[0][boundary_mask(x)]))

    def test_recurrent_rollout_and_PPO_replay_use_identical_masks(self):
        model=BoundaryPolicy()
        features=torch.stack([self.features()[:4],self.features()[5:9],self.features()[20:24]])
        starts=torch.tensor([[True]*4,[False]*4,[True,False,False,False]])
        hidden=torch.zeros(4,256)
        old=[];actions=[]
        with torch.no_grad():
            for x,begins in zip(features,starts):
                logits,_,_,hidden=model.step(x,hidden*(~begins)[:,None])
                dist=Categorical(logits=logits);action=dist.sample()
                old.append(dist.log_prob(action));actions.append(action)
        pi,_=recurrent_logits(model,features,starts,torch.zeros(4,256))
        ratio=(Categorical(logits=pi).log_prob(torch.stack(actions))-torch.stack(old)).exp()
        self.assertTrue(torch.allclose(ratio,torch.ones_like(ratio),atol=1e-6))

    def test_actual_training_cannot_sample_boundary_moves(self):
        torch.set_num_threads(1)
        model=BoundaryPolicy()
        dyn=Dynamics(TinyPolicy()).requires_grad_(False)
        original=FeatureWorld.step
        count=[0]
        def checked(world,actions):
            info=original(world,actions)
            self.assertFalse(bool(info['wall'].any()))
            count[0]+=len(actions)
            return info
        directory=self.root/'train';directory.mkdir()
        with patch.object(FeatureWorld,'step',checked):
            trained=train_arm(model,dyn,.1,self.store(),directory,'PBRS',0,'cpu',updates=1)
        self.assertEqual(count[0],640)
        self.assertTrue(trained.enabled)

    def test_actual_evaluator_replays_masked_policy(self):
        model=BoundaryPolicy()
        eps=[replace(self.ep(goal=g),episode_id=f'example_{g}') for g in (4,9,14,19,24)]
        directory=self.root/'eval';directory.mkdir()
        torch.save(model.state_dict(),directory/'model.pt')
        result=evaluate_and_audit(model,self.root,eps,self.store(),directory,'cpu')
        self.assertTrue(result['audit']['neural_replay'])
        self.assertEqual(result['metrics']['out_of_bounds_rate'],0)
        self.assertEqual(result['completed'],5)

    def test_SR60_alone_does_not_pass_preparation(self):
        def run(sr,repeat=.2):
            return dict(completed=100,planned=100,
                metrics=dict(sr=sr,mean_sg_all_episodes=1.5,repeat_visit_rate_micro=repeat,out_of_bounds_rate=0),
                by_distance={str(d):dict(sr=sr,mean_sg_all_episodes=1.5) for d in range(4,9)})
        full=[run(.6)]*3
        checks=preparation_checks(full,full,full,dict(sr_mean=.53,sg_mean=2.11))
        self.assertTrue(checks['mean_SR_at_least_60pct'])
        self.assertFalse(checks['repeat_each_at_most_10pct'])
        self.assertFalse(checks['target_contribution'])
        self.assertFalse(all(checks.values()))


if __name__=='__main__':
    unittest.main()
