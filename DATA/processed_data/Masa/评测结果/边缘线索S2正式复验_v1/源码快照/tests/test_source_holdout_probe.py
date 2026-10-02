"""Source isolation, public-only interventions and independent label replay."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.boundary_policy import BoundaryPolicy
from train.target_pairs import materialize
from train.source_holdout_probe import (source_split,pair_key,make_bank,fit_mean,input_condition,
    loss_for,train_probe,predict,save_predictions,uniform_reference,contrast,source_interval)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        class Store:
            data={f'img_{i}':np.random.default_rng(i).normal(size=(25,512)).astype(np.float32) for i in range(6)}
            def patch(self,area,cell):return self.data[area][cell]
        self.store=Store()
        self.sources={a:a+'.png' for a in self.store.data}

    def test_source_split_is_reproducible_and_rejects_duplicate_sources(self):
        a=source_split(self.store.data,self.sources,2)
        b=source_split(reversed(list(self.store.data)),self.sources,2)
        self.assertEqual(a,b);self.assertFalse(set(a[0])&set(a[1]))
        bad=self.sources.copy();bad['img_1']=bad['img_0']
        with self.assertRaises(ValueError):source_split(self.store.data,bad,2)

    def test_balanced_banks_exclude_reversed_targets_and_training_pairs(self):
        train=make_bank(['img_0','img_1'],16,5)
        diag=make_bank(['img_0','img_1'],8,6,exclude=train)
        self.assertEqual(len({pair_key(r) for r in train}),32)
        self.assertFalse({pair_key(r) for r in train}&{pair_key(r) for r in diag})
        for area in ('img_0','img_1'):self.assertEqual(sum(r['area']==area for r in diag),8)
        reversed_row=copy.deepcopy(train[0]);reversed_row['goals'].reverse()
        self.assertEqual(pair_key(train[0]),pair_key(reversed_row))

    def test_normalizer_never_uses_held_features(self):
        before=fit_mean(self.store,['img_0','img_1'])
        self.store.data['img_2'][:]=1e7
        after=fit_mean(self.store,['img_0','img_1'])
        np.testing.assert_array_equal(before,after)

    def tensors(self):
        records=make_bank(['img_0'],8,7)
        x,y=[torch.tensor(v) for v in materialize(records,self.store)]
        return records,x,y,torch.tensor(fit_mean(self.store,['img_0']))

    def test_input_adapters_preserve_public_state_and_do_not_mutate_original(self):
        _,x,_,mean=self.tensors();before=x.clone()
        for mode in ('NoTarget','StateOnly','MeanCue','SwapCue'):
            changed=input_condition(x,mode,mean)
            self.assertTrue(torch.equal(changed[...,1024:],x[...,1024:]))
            if mode!='StateOnly':self.assertTrue(torch.equal(changed[...,512:1024],x[...,512:1024]))
        self.assertTrue(torch.equal(x,before))
        swap=input_condition(x,'SwapCue',mean)
        self.assertTrue(torch.equal(swap[:,:,:,0:512],x.flip(1)[:,:,:,0:512]))

    def test_visual_changes_cannot_change_state_only_input(self):
        _,x,_,mean=self.tensors();other=x.clone();other[...,:1024]=torch.randn_like(other[...,:1024])*100
        self.assertTrue(torch.equal(input_condition(x,'StateOnly',mean),input_condition(other,'StateOnly',mean)))
        other=x.clone();other[...,:512]=torch.randn_like(other[...,:512])
        self.assertTrue(torch.equal(input_condition(x,'NoTarget',mean),input_condition(other,'NoTarget',mean)))

    def test_no_target_models_cannot_solve_both_disjoint_goals(self):
        bank,x,y,mean=self.tensors();model=BoundaryPolicy()
        for mode in ('NoTarget','StateOnly','MeanCue'):
            rows=predict(model,x,y,mean,mode,bank)
            self.assertTrue(all(r['actions'][0]==r['actions'][1] for r in rows))
            self.assertFalse(any(all(r['correct']) for r in rows))

    def test_uniform_reference_matches_enumerated_action_pairs(self):
        bank,x,_,_=self.tensors()
        from agents.boundary_policy import boundary_mask
        legal=boundary_mask(x[:,:,2]).numpy()
        marginals=[];joints=[]
        for r,mask in zip(bank,legal):
            matches=[(bool(r['labels'][0][a]),bool(r['labels'][1][b]))
                     for a in np.flatnonzero(mask[0]) for b in np.flatnonzero(mask[1])]
            marginals.append(np.mean(matches));joints.append(np.mean([all(m) for m in matches]))
        result=uniform_reference(bank)
        self.assertAlmostEqual(result['direction_accuracy'],np.mean(marginals))
        self.assertAlmostEqual(result['both_correct_rate'],np.mean(joints))

    def test_training_updates_policy_and_saved_model_replays_predictions(self):
        bank,x,y,mean=self.tensors();initial=BoundaryPolicy();state=copy.deepcopy(initial.state_dict())
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            trained=train_probe(initial,x,y,mean,'Full',0,folder,epochs=2)
            self.assertTrue(any(not torch.equal(v,state[k]) for k,v in trained.state_dict().items() if k.startswith('actor')))
            self.assertTrue(all(torch.equal(v,state[k]) for k,v in trained.state_dict().items() if k.startswith(('critic','next_state'))))
            result=save_predictions(folder,'synthetic',trained,x,y,mean,'Full',bank,'cpu')
            self.assertTrue(result['audit']['checkpoint_replay'])
            self.assertEqual(result['metrics']['invalid_actions'],0)
            self.assertEqual(result['metrics']['targets'],16)

    def test_contrast_requires_gain_and_CE_with_source_grouped_interval(self):
        def row(acc,ce):return {'metrics':{'direction_accuracy':acc,'cross_entropy':ce},
            'by_source':{f's{i}':{'direction_accuracy':acc} for i in range(4)}}
        control=[row(.5,1.)]*3
        self.assertFalse(contrast([row(.6,1.2)]*3,control)['passed'])
        self.assertFalse(contrast([row(.5,1.)]*3,control)['passed'])
        good=[row(.6,.8)]*3
        self.assertTrue(contrast(good,control)['passed'])
        ci=source_interval(good,control)
        self.assertEqual(ci['source_count'],4)
        np.testing.assert_allclose(ci['interval95'],[.1,.1])


if __name__=='__main__':unittest.main()
