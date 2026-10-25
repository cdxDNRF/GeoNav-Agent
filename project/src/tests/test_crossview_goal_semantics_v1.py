"""Meaningful semantics/input, full-frame geometry and unchanged square tests."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from data.crossview_goal_semantics_v1 import public_target,fullframe_geometry,fullframe_pixels
from data.mmgag_views_v1 import ground_pixels
CFG={'size':224,'crop_size':224,'image_mean':[.48145466,.4578275,.40821073],'image_std':[.26862954,.26130258,.27577711]}
class GoalTests(unittest.TestCase):
    def pixels(self,h=12,w=16):
        a=np.full((h,w,3),127,np.uint8);a.setflags(write=False);return a
    def test_capture_point_definition_exposes_no_answer(self):
        x=public_target({'target_rgb':self.pixels(),'modality':'ground','localization_target':'capture_point'})
        self.assertEqual(set(x.__dataclass_fields__),{'target_rgb','modality','localization_target'})
    def test_hidden_fields_and_landmark_without_annotations_rejected(self):
        payload={'target_rgb':self.pixels(),'modality':'ground','localization_target':'capture_point'}
        for key in ['goal_index','lat','lon','source_id','map_features','filename']:
            with self.assertRaises(ValueError):public_target({**payload,key:1})
        with self.assertRaises(ValueError):public_target({**payload,'localization_target':'visible_landmark'})
    def test_mutable_pixels_path_and_whole_bank_rejected(self):
        for a in [np.zeros((4,4,3),np.uint8),'photo.jpg',np.zeros((25,4,4,3),np.uint8)]:
            with self.assertRaises(ValueError):public_target({'target_rgb':a,'modality':'ground','localization_target':'capture_point'})
    def test_full_frame_keeps_all_edges_and_padding_zero(self):
        rgb=np.zeros((100,200,3),np.uint8);rgb[:,0]=[255,0,0];rgb[:,-1]=[0,255,0];rgb[0]=[0,0,255];rgb[-1]=[255,255,0]
        g=fullframe_geometry(rgb);self.assertEqual(g['content_box'],[0,56,224,168]);self.assertEqual(g['source_region'],[0,0,200,100])
        a=fullframe_pixels(rgb,CFG);self.assertEqual(a.dtype,np.float32);self.assertEqual(a.shape,(3,224,224));self.assertFalse(a[:,:56].any());self.assertFalse(a[:,168:].any())
        self.assertGreater(a[0,112,0],a[1,112,0]);self.assertGreater(a[1,112,-1],a[0,112,-1]);self.assertGreater(a[2,56,112],a[0,56,112])
    def test_square_exactly_matches_existing_processor(self):
        a=np.random.default_rng(3).integers(0,256,(256,256,3),dtype=np.uint8)
        np.testing.assert_array_equal(fullframe_pixels(a,CFG),ground_pixels(a,CFG))
    def test_portrait_and_odd_padding_are_deterministic(self):
        a=self.pixels(4032,2268);g=fullframe_geometry(a);self.assertEqual(g['resized_size'],[126,224]);self.assertEqual(g['content_box'],[49,0,175,224])
        b=self.pixels(7,11);g=fullframe_geometry(b);self.assertEqual(g['content_box'],[0,41,224,183])
    def test_no_new_encoder_size_or_invalid_normalization(self):
        with self.assertRaises(ValueError):fullframe_pixels(self.pixels(),{**CFG,'size':256})
        with self.assertRaises(ValueError):fullframe_pixels(self.pixels(),{**CFG,'image_std':[1,0,1]})
if __name__=='__main__':unittest.main()
