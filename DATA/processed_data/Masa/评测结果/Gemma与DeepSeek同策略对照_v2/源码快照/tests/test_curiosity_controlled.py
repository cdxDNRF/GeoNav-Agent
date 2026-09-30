"""Regression checks for actual reward/rollout failure modes, on synthetic maps."""
from pathlib import Path
import sys
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_foundation import Fixture
from env.episode import ACTIONS
from train.dyncur_tiny import TinyPolicy, TrainConfig, _transition_reward, policy_features
from train.curiosity_controlled import (Dynamics, FeatureWorld, training_reward,
    normalize_error, gae, recurrent_logits, train_arm, compare)


class ControlledTests(Fixture):
    def make_store(self):
        class Store:
            data={"img_0":np.random.default_rng(42).normal(size=(25,512)).astype(np.float32)}
        return Store()

    def test_arrival_reward_and_training_success_use_same_cell_ids(self):
        ext,pbrs,success=training_reward(np.array([19,19]),np.array([24,18]),np.array([24,24]),np.array([True,True]))
        np.testing.assert_allclose(ext,[1.1,-.1],atol=1e-6)
        self.assertEqual(success.tolist(),[True,False])
        np.testing.assert_allclose(pbrs,[.0625,.0625],atol=1e-6)
        cfg=TrainConfig("ppo_pbrs",0,1,1,.0003,.5,.99,0,.1,"cpu")
        self.assertAlmostEqual(_transition_reward(self.ep(),(3,4),(4,4),True,cfg),1.1625)

    def test_potential_telescopes_at_budget_terminal(self):
        # Two paths with the same start/budget should have the same discounted shaping sum.
        def shaping(path):
            return sum((.99**i)*training_reward(np.array([p]),np.array([n]),np.array([24]),np.array([i==len(path)-2]))[1][0]
                       for i,(p,n) in enumerate(zip(path,path[1:])))
        self.assertAlmostEqual(shaping([0,1,2,3]),shaping([0,5,0,0]),places=6)

    def test_vector_environment_and_features_match_real_environment(self):
        store=self.make_store()
        for trial in range(12):
            env=self.env(); ep=self.ep(start=trial,goal=24,budget=10)
            obs=env.reset(ep)
            world=FeatureWorld(store,1,trial)
            world.position[0]=ep.start;world.goal[0]=ep.goal;world.steps[0]=0
            world.visits[:]=0;world.visits[0,ep.start]=1
            rng=np.random.default_rng(trial)
            while not env.done:
                expect=policy_features(store.data['img_0'][ep.goal],store.data['img_0'][world.position[0]],obs.position,obs.remaining_budget,obs.visited)
                np.testing.assert_allclose(world.observe()[0],expect,atol=1e-7)
                action=int(rng.integers(4)); info=world.step(np.array([action]))
                obs,_,actual=env.step(tuple(ACTIONS)[action])
                self.assertEqual(int(world.position[0]),obs.position[0]*5+obs.position[1])
                self.assertEqual(bool(info['done'][0]),env.done)
                self.assertEqual(bool(info['wall'][0]),actual.out_of_bounds)
                self.assertEqual(bool(info['revisited'][0]),actual.revisited)

    def test_gae_stops_at_episode_boundary_and_bootstraps_rollout(self):
        rewards=torch.tensor([[1.],[2.],[3.]])
        adv,returns=gae(rewards,torch.zeros_like(rewards),torch.tensor([[False],[True],[False]]),torch.tensor([4.]),1,1)
        self.assertEqual(returns[:,0].tolist(),[3.,2.,7.])

    def test_dynamic_prediction_conditioned_on_action_and_frozen(self):
        model=Dynamics(TinyPolicy())
        h=torch.zeros(2,256)
        self.assertFalse(torch.equal(model.predict(h,torch.tensor([0,1]))[0],model.predict(h,torch.tensor([0,1]))[1]))
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        model.requires_grad_(False)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_normalization_is_bounded_without_mutating_scale(self):
        np.testing.assert_equal(normalize_error(np.array([0,.1,10]),.1),[0,1,3])
        with self.assertRaises(ValueError):normalize_error(np.ones(2),0)

    def test_recurrent_gradients_include_history_and_reset_at_new_episode(self):
        model=TinyPolicy();x=torch.randn(3,2,1052,requires_grad=True)
        starts=torch.tensor([[True,True],[False,False],[False,True]])
        pi,_=recurrent_logits(model,x,starts,torch.zeros(2,256))
        pi[-1].sum().backward()
        self.assertGreater(float(x.grad[0,0].abs().sum()),0)
        self.assertEqual(float(x.grad[0,1].abs().sum()),0)

    def test_real_minibatch_training_preserves_frozen_dynamics(self):
        torch.set_num_threads(1)
        model=TinyPolicy();dynamic=Dynamics(model).requires_grad_(False)
        before=copy.deepcopy(dynamic.state_dict()); directory=self.root/'training';directory.mkdir()
        trained=train_arm(model,dynamic,.1,self.make_store(),directory,'Curiosity',0,'cpu',updates=1)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in dynamic.state_dict().items()))
        self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in trained.state_dict().items()))
        self.assertTrue((directory/'训练日志.jsonl').exists())


if __name__=='__main__':unittest.main()
