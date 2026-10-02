"""Geometry orientation, target interventions, genuine learning and offline release."""
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.edge_cue import image_profiles,edge_features,EdgeTargetCueHead
from agents.decoupled_cue import DecoupledTargetCueHead
from eval.audit_edge_cue import manual_profile,manual_seam,manual_joint
from train.edge_cue import paired_features,train_head,navigation_gate
from train.dyncur_tiny import set_seed
from train.trusted_cue import pair_bank


class EdgeCueTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_contiguous_synthetic_scene_has_correct_direction_for_all_four_neighbors(self):
        y,x=np.mgrid[:900,:900]
        rgb=np.stack((np.sin(.009*x+.004*y),np.cos(.003*x-.005*y),np.sin(.004*x+.009*y)),axis=-1)
        scene=np.round((rgb+1)*127).astype(np.uint8)
        current=image_profiles(scene[300:600,300:600])
        slices=((slice(0,300),slice(300,600)),(slice(300,600),slice(600,900)),
                (slice(600,900),slice(300,600)),(slice(300,600),slice(0,300)))
        for direction,region in enumerate(slices):
            target=image_profiles(scene[region]);features=edge_features(target,current).reshape(4,5)
            self.assertEqual(int(features[:,0].argmin()),direction)
            self.assertLess(float(features[direction,0]),.02)

    def test_public_image_bytes_preserve_profiles_and_side_orientation(self):
        rgb=np.zeros((300,300,3),np.uint8);rgb[:,:,0]=np.arange(300,dtype=np.int64)[:,None]%256
        image=Image.fromarray(rgb);stream=BytesIO();image.save(stream,format='PNG')
        array=image_profiles(rgb);np.testing.assert_array_equal(array,image_profiles(stream.getvalue()))
        cuts=np.linspace(0,300,65,dtype=np.int64)
        self.assertAlmostEqual(float(array[1,0,0,0]),float(rgb[cuts[0]:cuts[1],-1,0].mean()/255),places=7)
        np.testing.assert_array_equal(array,manual_profile(rgb))

    def test_flat_boundary_correlation_is_zero_and_finite(self):
        profiles=image_profiles(np.full((300,300,3),128,np.uint8))
        values=edge_features(profiles,profiles).reshape(4,5)
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values,np.zeros((4,5),np.float32))

    def test_matched_capacity_zero_projection_preserves_old_joint_predictions(self):
        set_seed(2);old=DecoupledTargetCueHead().eval();state=old.state_dict()
        set_seed(2);new=EdgeTargetCueHead().eval();new.load_state_dict(state,strict=False)
        self.assertEqual(sum(p.numel() for p in new.parameters()),136837)
        base=torch.randn(8,1041);features=torch.cat((base,torch.randn(8,20)),dim=-1)
        torch.testing.assert_close(new(features),old(base),rtol=0,atol=0)

    def test_real_training_zero_arm_keeps_projection_zero_and_edge_arm_updates_it(self):
        rng=np.random.default_rng(18);base=rng.normal(size=(20,1041)).astype(np.float32)
        edges=rng.normal(size=(20,20)).astype(np.float32);labels=np.array([0,1,2,3]+[4]*16,dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp,patch('train.edge_cue.EPOCHS',2):
            root=Path(temp);initial=root/'initial.pt';torch.save(DecoupledTargetCueHead().state_dict(),initial)
            control=root/'Zero';candidate=root/'Edge';control.mkdir();candidate.mkdir()
            a=train_head(np.concatenate((base,np.zeros_like(edges)),axis=1),labels,control,0,torch.device('cpu'),initial)
            b=train_head(np.concatenate((base,edges),axis=1),labels,candidate,0,torch.device('cpu'),initial)
            self.assertEqual(int(torch.count_nonzero(a.edge_projection.weight)),0)
            self.assertGreater(int(torch.count_nonzero(b.edge_projection.weight)),0)
            left=torch.load(control/'初始化.pt',weights_only=True);right=torch.load(candidate/'初始化.pt',weights_only=True)
            for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
            import json
            record=json.loads((candidate/'训练资源.json').read_text(encoding='utf-8'))
            self.assertEqual((record['optimizer_steps'],record['pair_uses'],record['adjacent_pair_uses']),(2,40,8))

    def test_target_interventions_include_edges_and_zero_control(self):
        rng=np.random.default_rng(23);values=rng.normal(size=(25,512)).astype(np.float32)
        store=SimpleNamespace(patch=lambda area,cell:values[cell]);local={'img_0':rng.normal(size=(25,4,768)).astype(np.float32)}
        profiles={'img_0':rng.random((25,4,3,64,3),dtype=np.float32)};pm=profiles['img_0'].mean(0)
        bank=[r for r in pair_bank(['img_0']) if r['current']==0]
        args=(bank,store,local,values.mean(0),local['img_0'].mean(0),profiles,pm)
        masked=paired_features(*args,'MeanCue','Edge')
        np.testing.assert_array_equal(masked,np.repeat(masked[:1],len(bank),axis=0))
        wrong=paired_features(*args,'WrongCue','Edge');full=paired_features(*args,'Full','Edge')
        np.testing.assert_array_equal(wrong[:,512:1024],full[:,512:1024])
        self.assertFalse(np.array_equal(wrong[:,1041:],full[:,1041:]))
        for i,row in enumerate(bank):np.testing.assert_array_equal(wrong[i,1041:],edge_features(profiles['img_0'][row['wrong']],profiles['img_0'][row['current']]))
        for condition in ('Full','MeanCue','WrongCue'):
            np.testing.assert_array_equal(paired_features(*args,condition,'ZeroEdge')[:,1041:],np.zeros((len(bank),20),np.float32))

    def test_predeclared_gate_requires_two_calibrated_seeds_with_held_evidence(self):
        calibration={str(s):dict(threshold=.9) for s in range(3)}
        probes={str(s):dict(accepted=120,sources_with_acceptances=6,accepted_precision=.9,precision_interval=dict(interval95=[.8,.97])) for s in range(3)}
        calibration['2']['threshold']=None
        self.assertTrue(navigation_gate(calibration,probes)['released'])
        probes['1']['accepted_precision']=.84
        self.assertFalse(navigation_gate(calibration,probes)['released'])
        probes['1']['accepted_precision']=.9;probes['1']['accepted']=99
        self.assertFalse(navigation_gate(calibration,probes)['released'])
        probes['1']['accepted']=120;probes['1']['precision_interval']['interval95'][0]=.74
        self.assertFalse(navigation_gate(calibration,probes)['released'])

    def test_independent_seam_and_joint_formulas_match_for_broadcasted_inputs(self):
        rng=np.random.default_rng(37);target=rng.random((4,3,64,3),dtype=np.float32)
        current=rng.random((3,4,3,64,3),dtype=np.float32)
        np.testing.assert_array_equal(edge_features(target,current),manual_seam(target,current))
        model=EdgeTargetCueHead().eval();features=torch.tensor(rng.normal(size=(3,1061)).astype(np.float32))
        with torch.no_grad():expected=model(features).softmax(-1).numpy()
        np.testing.assert_allclose(expected,manual_joint(model,features),atol=2e-6,rtol=0)


if __name__=='__main__':unittest.main()
