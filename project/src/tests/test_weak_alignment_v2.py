"""CV-008 in-memory engineering tests; no image/model/network calls."""
import inspect
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'project/src'))
from agents.weak_alignment_head_v1 import WeakAlignmentHead
from train.weak_scene_alignment_v1 import engineering_checks,loss_for
from eval.weak_alignment_eval_v1 import eval_tasks
from tools.run_weak_alignment_v2 import read,PREV,MM,OUT


class EngineeringTests(unittest.TestCase):
    def test_identity_and_exact_parameter_budget(self):
        h = WeakAlignmentHead();g = torch.nn.functional.normalize(torch.ones(512),dim=-1)
        self.assertEqual(sum(p.numel() for p in h.parameters()),8192)
        torch.testing.assert_close(h(g),g,atol=1e-6,rtol=0)

    def test_gradient_resume_and_optimizer_scope(self):
        self.assertTrue(engineering_checks()['resume_parameters_and_optimizer_bitwise_equal'])

    def test_metadata_and_bank_rejected(self):
        h = WeakAlignmentHead()
        for bad in [{'goal':0},torch.zeros(25,25,512),torch.zeros(512),torch.full((512,),float('nan'))]:
            with self.assertRaises(ValueError):h(bad)
        with self.assertRaises(TypeError):h(torch.ones(512),source_id='x')
        self.assertEqual(list(inspect.signature(h.forward).parameters),['ground'])

    def test_fixed_split_and_quarantine(self):
        bank=read(PREV/'唯一训练查询与图块引用.json');design=read(PREV/'训练设计冻结.json')
        self.assertEqual(len(bank['queries']),14);self.assertEqual(len(bank['cells']),1134)
        self.assertEqual({q['scene'] for q in bank['queries'] if q['role']=='fit'},set(design['training']['source_groups']))
        self.assertNotIn('GrandeDixence',{c['scene'] for c in bank['cells']})

    def test_exact_window_mapping_and_wrong_source(self):
        bank=read(PREV/'唯一训练查询与图块引用.json');design=read(PREV/'训练设计冻结.json')
        tasks=eval_tasks(bank,read(PREV/'派生素材清单.json')['records'],read(MM/'冻结执行任务.json')['pairs'],design['evaluation']['weak_dev_groups'])
        self.assertEqual(len(tasks)*6,846)
        for t in tasks[:125]:
            q=bank['queries'][t['query']]
            self.assertEqual(bank['cells'][t['candidates'][t['truth']]]['cell_sha256'],q['positive_cell_sha256'])
            self.assertNotEqual(q['scene'],bank['queries'][t['wrong_query']]['scene'])

    def test_real_cell_sha_contract_before_forward(self):
        from tools.run_weak_alignment_v2 import pixel_array
        bank=read(PREV/'唯一训练查询与图块引用.json')
        processor=read(ROOT/'models/clip_vit_b32_public_v1/preprocessor_config.json')
        for item in [bank['cells'][0],bank['cells'][-1]]:
            self.assertEqual(pixel_array(item,'aerial',processor).shape,(3,224,224))

    def test_recovery_counts_and_successful_feature_coverage(self):
        from tools.run_weak_alignment_v2 import FAILED
        state=read(FAILED/'执行状态.json')
        self.assertEqual((state['attempted_encoder_inputs'],state['successful_encoder_calls']),(14,2))
        self.assertEqual(state['training_steps'],0)
        self.assertEqual(np.load(FAILED/'特征/ground.npy',allow_pickle=False).shape,(14,512))

    def test_encoder_budget_includes_engineering(self):
        self.assertEqual(sum((n+7)//8 for n in [14,1134]),144)
        self.assertEqual(sum([14,1134]),1148)

    def test_feature_gradients_and_bad_labels_rejected(self):
        h=WeakAlignmentHead();g=torch.nn.functional.normalize(torch.ones(1,512),dim=-1)
        with self.assertRaises(ValueError):loss_for(h,g,torch.zeros(1,25,512),torch.tensor([0]))
        with self.assertRaises(ValueError):loss_for(h,g.requires_grad_(),torch.zeros(1,81,512),torch.tensor([0]))


if __name__=='__main__':unittest.main()
