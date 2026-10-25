"""Zero-real-forward tests for mapped initialization and single encoder slot."""
import unittest
from unittest.mock import patch
from project.src.agents.frozen_crossview_encoder_v2 import construct_mapped_model,FrozenCrossViewEncoder


class Tests(unittest.TestCase):
    def test_mapped_parameter_storage_and_position_buffer(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self,config):
                super().__init__();self.weight=torch.nn.Parameter(torch.empty(2,3))
                self.emb=torch.nn.Module();self.emb.register_buffer('position_ids',torch.arange(50).expand(1,-1),persistent=False)
        state={'weight':torch.ones(2,3)}
        model=construct_mapped_model(None,state,Tiny)
        self.assertEqual(model.weight.data_ptr(),state['weight'].data_ptr())
        self.assertEqual(model.emb.position_ids.flatten().tolist(),list(range(50)))
        self.assertEqual(model.emb.position_ids.device.type,'cpu')
        self.assertFalse(model.training);self.assertFalse(model.weight.requires_grad)

    def test_strict_missing_weight_rejected(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self,c):super().__init__();self.weight=torch.nn.Parameter(torch.empty(2))
        with self.assertRaises(RuntimeError):construct_mapped_model(None,{},Tiny)

    def test_unknown_meta_buffer_rejected(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self,c):
                super().__init__();self.register_buffer('unknown',torch.empty(2),persistent=False)
        with self.assertRaises(ValueError):construct_mapped_model(None,{},Tiny)

    def test_releasing_slot_preserves_hash_and_clears_model(self):
        import torch
        from project.src.agents.frozen_crossview_encoder_v1 import tensor_digest
        encoder=FrozenCrossViewEncoder.__new__(FrozenCrossViewEncoder)
        model=torch.nn.Linear(2,2);model.requires_grad_(False);model.eval()
        encoder.models={'ground':model};encoder.before={'ground':tensor_digest(model)};encoder.released_checks=[]
        encoder._release_slot()
        self.assertEqual(encoder.models,{})
        self.assertTrue(encoder.released_checks[0]['unchanged'])


if __name__=='__main__':unittest.main()
