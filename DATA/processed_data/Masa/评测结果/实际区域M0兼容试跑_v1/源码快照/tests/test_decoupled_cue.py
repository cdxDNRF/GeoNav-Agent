"""Guard joint arrival confidence, masked supervision and matched-budget training."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.decoupled_cue import DecoupledTargetCueHead, decoupled_loss
from agents.target_cue import TargetCueHead, choose_cue
from train.decoupled_cue import component_metrics, train_head
from train.dyncur_tiny import set_seed


class DecoupledCueTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_same_capacity_and_paired_initialization(self):
        set_seed(7);reference=TargetCueHead().state_dict()
        set_seed(7);model=DecoupledTargetCueHead()
        self.assertEqual(sum(p.numel() for p in model.parameters()),134277)
        self.assertEqual(set(reference),set(model.state_dict()))
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,reference[key],rtol=0,atol=0)

    def fixed_model(self,raw):
        model=DecoupledTargetCueHead()
        with torch.no_grad():
            model.network[-1].weight.zero_()
            model.network[-1].bias.copy_(torch.tensor(raw,dtype=torch.float32))
        return model

    def test_joint_confidence_rejects_confident_direction_when_not_adjacent(self):
        model=self.fixed_model([0,10,0,0,-4])
        values=model(torch.zeros(1,1041)).softmax(-1)[0].detach().numpy()
        conditional=values[:4]/values[:4].sum()
        self.assertGreater(conditional[1],.99)
        self.assertLess(values[1],.02)
        self.assertEqual(choose_cue(values,.5,(2,2),(12,)),(None,'not_adjacent'))

    def test_high_adjacency_and_correct_direction_can_override(self):
        values=self.fixed_model([0,10,0,0,10])(torch.zeros(1,1041)).softmax(-1)[0].detach().numpy()
        self.assertGreater(values[1],.99)
        self.assertEqual(choose_cue(values,.9,(2,2),(12,)),('right','accepted'))
        self.assertEqual(choose_cue(values,.9,(2,2),(12,13)),(None,'visited_top_destination'))

    def test_uniform_directions_do_not_inherit_high_adjacency_confidence(self):
        values=self.fixed_model([0,0,0,0,10])(torch.zeros(1,1041)).softmax(-1)[0].detach().numpy()
        self.assertGreater(values[:4].sum(),.99)
        self.assertLess(values.max(),.26)
        self.assertEqual(choose_cue(values,.5,(2,2),(12,)),(None,'low_confidence'))

    def test_nonadjacent_samples_have_no_direction_gradient(self):
        raw=torch.zeros(3,5,requires_grad=True)
        total,binary,directional=decoupled_loss(raw,torch.tensor([4,4,4]))
        total.backward()
        torch.testing.assert_close(raw.grad[:,:4],torch.zeros(3,4),rtol=0,atol=0)
        self.assertTrue((raw.grad[:,4]>0).all())
        self.assertEqual(float(directional),0)
        self.assertAlmostEqual(float(total),float(binary))

    def test_positive_and_negative_supervision_remain_separate_and_finite(self):
        raw=torch.tensor([[0.,0.,0.,0.,-1000.],[0.,0.,0.,0.,1000.]],requires_grad=True)
        loss,binary,directional=decoupled_loss(raw,torch.tensor([1,4]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertLess(float(raw.grad[0,1]),0)
        self.assertLess(float(raw.grad[0,4]),0)
        self.assertGreater(float(raw.grad[1,4]),0)
        torch.testing.assert_close(raw.grad[1,:4],torch.zeros(4),rtol=0,atol=0)
        self.assertAlmostEqual(float(directional),float(np.log(4)),places=6)

    def test_actual_training_counts_adjacent_direction_uses_and_updates_both_outputs(self):
        features=np.random.default_rng(8).normal(size=(20,1041)).astype(np.float32)
        labels=np.array([0,1,2,3]+[4]*16,dtype=np.int64)
        set_seed(0);initial=DecoupledTargetCueHead().state_dict()
        with tempfile.TemporaryDirectory() as temp,patch('train.decoupled_cue.EPOCHS',2):
            folder=Path(temp)
            model=train_head(features,labels,folder,0,torch.device('cpu'))
            import json
            resource=json.loads((folder/'训练资源.json').read_text(encoding='utf-8'))
            self.assertEqual((resource['optimizer_steps'],resource['pair_uses'],resource['adjacent_pair_uses']),(2,40,8))
            self.assertFalse(torch.equal(initial['network.3.weight'][:4],model.state_dict()['network.3.weight'][:4]))
            self.assertFalse(torch.equal(initial['network.3.weight'][4],model.state_dict()['network.3.weight'][4]))
            values=model(torch.tensor(features)).softmax(-1)
            torch.testing.assert_close(values.sum(-1),torch.ones(20))

    def test_component_diagnostic_does_not_confuse_direction_with_arrival(self):
        values=np.array([[.09,.01,.01,.01,.88],[.01,.01,.01,.01,.96]])
        result=component_metrics(values,[dict(label=0),dict(label=4)])
        self.assertEqual(result['conditional_direction_accuracy'],1)
        self.assertEqual(result['adjacent_recall_at_half'],0)
        self.assertIsNone(result['adjacent_precision_at_half'])
        self.assertEqual(result['false_negative'],1)


if __name__=='__main__':unittest.main()
