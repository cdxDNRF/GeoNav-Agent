"""Synthetic preflight: zero encoder calls, zero head forwards/optimizer steps."""
import io
import unittest
import numpy as np
import torch
from project.src.agents import rgb_spatial_runtime_v1 as rt
from project.src.train import rgb_spatial_executor_v1 as ex
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame


def frame(w,h,color=(100,120,140)):
    a=np.empty((h,w,3),np.uint8);a[:]=color;return RGBFrame(w,h,a.tobytes())


class Tests(unittest.TestCase):
    def test_ground_letterbox_dimensions_and_zero_normalized_padding(self):
        t,m=rt.prepare(frame(952,532),'ground')
        self.assertEqual(m['resized_wh'],[224,125]);self.assertEqual(m['padding_ltrb'],[0,49,0,50])
        self.assertEqual(tuple(t.shape),(1,3,224,224));self.assertEqual(float(t[:,:,:49,:].abs().max()),0)

    def test_ground_keeps_both_image_edges(self):
        a=np.zeros((100,200,3),np.uint8);a[:,:100,0]=255;a[:,100:,2]=255
        t,m=rt.prepare(RGBFrame(200,100,a.tobytes()),'ground')
        self.assertGreater(float(t[0,0,112,0]),1);self.assertGreater(float(t[0,2,112,-1]),1)

    def test_aerial_direct_current_crop_resize(self):
        t,m=rt.prepare(frame(125,125),'aerial');self.assertEqual(m['padding_ltrb'],[0,0,0,0]);self.assertTrue(torch.isfinite(t).all())

    def test_head_count_and_common_initialization_without_forward(self):
        a=rt.Spatial64(0);b=rt.Spatial64(0)
        self.assertEqual(sum(p.numel() for p in a.parameters()),65537);self.assertTrue(torch.equal(a.Wg,a.Wa))
        self.assertEqual(rt.state_hash(a.state_dict()),rt.state_hash(b.state_dict()));self.assertEqual(a.forward_calls,0)

    def test_public_packet_rejects_hidden_fields_before_forward(self):
        unit=torch.zeros((1,512));unit[:,0]=1;p=dict(given_target_feature=unit,current_patch_features=unit[:,None,:].expand(-1,49,-1))
        rt.validate_tensor_packet(p)
        for k in ('GPS','heading','goal_cell','sourceID','parent_bank','labels'):
            with self.assertRaises(ValueError):rt.validate_tensor_packet(dict(p,**{k:0}))

    def test_controls_mean_does_not_use_held_and_disabled_ignores_target(self):
        a=np.eye(3,512,dtype=np.float32);cache={('ground',n):a[i] for i,n in enumerate(('fit0','fit1','held'))};fold=dict(fit_sources=['fit0','fit1'])
        m=ex.mean_for_fold(cache,fold);self.assertEqual(float(m[2]),0)
        for condition in ('Disabled','Mean'):
            self.assertTrue(np.array_equal(ex.target_control(a[0],condition,m,a[1]),ex.target_control(a[2],condition,m,a[1])))

    def test_checkpoint_zero_step_roundtrip_and_mismatched_context_rejected(self):
        h=rt.Spatial64(0);o=rt.optimizer(h);context=dict(protocol='synthetic',cache='none')
        buffer=io.BytesIO();torch.save(ex.checkpoint(h,o,0,context),buffer);buffer.seek(0);v=torch.load(buffer,weights_only=True,map_location='cpu')
        h2,o2,step=ex.restore(v,context);self.assertEqual(step,0)
        self.assertEqual(rt.state_hash(h.state_dict()),rt.state_hash(h2.state_dict()))
        with self.assertRaises(ValueError):ex.restore(v,dict(protocol='other',cache='none'))

    def test_formal_executor_rejects_incomplete_cache_before_update(self):
        h=rt.Spatial64(0);o=rt.optimizer(h);fold=dict(fit_cases=[dict(case_id='synthetic/a/b/0',query='a',reference='b',cell=0)])
        with self.assertRaises(ValueError):ex.fixed_train(h,o,fold,0,{},lambda *a:None)
        self.assertEqual(h.forward_calls,0)

    def test_nonfinite_or_grad_carrying_features_rejected(self):
        g=torch.zeros((1,512));g[:,0]=1;a=g[:,None,:].repeat(1,49,1)
        with self.assertRaises(ValueError):rt.validate_tensor_packet(dict(given_target_feature=g.requires_grad_(),current_patch_features=a))
        g=g.detach();a[0,0,0]=float('nan')
        with self.assertRaises(ValueError):rt.validate_tensor_packet(dict(given_target_feature=g,current_patch_features=a))


if __name__=='__main__':unittest.main()
