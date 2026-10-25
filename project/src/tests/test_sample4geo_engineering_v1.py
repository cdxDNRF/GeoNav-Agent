import unittest
import numpy as np
import torch
from project.src.agents.sample4geo_engineering_v1 import preprocess, strict_restore, tensor_receipts


class EngineeringTests(unittest.TestCase):
    def test_shapes_and_rgb(self):
        rgb=np.full((27,43,3),(255,0,128),dtype=np.uint8)
        for v,s in [('ground',(1,3,140,768)),('aerial',(1,3,384,384))]:
            x,r=preprocess(rgb,v)
            self.assertEqual(tuple(x.shape),s)
            expected=(np.array([255,0,128])/255-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
            np.testing.assert_allclose(x.numpy()[0,:,0,0],expected,atol=3e-7,rtol=0)

    def test_input_rejection(self):
        for x,v in [(np.zeros((2,2,3)),'ground'),(np.zeros((2,2),np.uint8),'ground'),
                    (np.zeros((2,2,3),np.uint8),'unknown')]:
            with self.assertRaises(ValueError):preprocess(x,v)

    def test_strict_restore_and_frozen_parameters(self):
        m=torch.nn.Linear(2,3)
        s={k:torch.ones_like(x) for k,x in m.state_dict().items()}
        r=strict_restore(m,s)
        self.assertEqual(r,tensor_receipts(s))
        self.assertFalse(m.training)
        self.assertTrue(all(not x.requires_grad for x in m.parameters()))
        for bad in [{},dict(s,extra=torch.ones(1)),dict(s,weight=torch.ones(2,3)),dict(s,bias=torch.ones(3,dtype=torch.float64))]:
            with self.assertRaises(ValueError):strict_restore(m,bad)

    def test_unsafe_or_nonfinite_state_rejected(self):
        for s in [{'x':torch.tensor([float('nan')])},{'x':torch.tensor([1.0],dtype=torch.float64)}, {'x':'not a tensor'}]:
            with self.assertRaises(ValueError):tensor_receipts(s)


if __name__=='__main__':unittest.main()
