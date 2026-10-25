"""Offline label/patch boundaries and deterministic grouped split; no disk fixtures."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from data.weak_scene_pairs_v1 import raw_path,crop_box,split_scenes,pixel_pair,materialize_pair
class WeakPairTests(unittest.TestCase):
    def test_all25_boxes_row_column_and_area(self):
        for j in range(25):
            x,y,xx,yy=crop_box(j);self.assertEqual((xx-x,yy-y),(300,300));self.assertEqual((y//300)*5+x//300,j)
    def test_invalid_label_indices_rejected(self):
        for j in [True,-1,25,'12',12.0]:
            with self.assertRaises(ValueError):crop_box(j)
    def test_raw_paths_cannot_access_MMGAG_or_outside(self):
        root=Path(__file__).resolve().parents[3]
        for p in ['../private.jpg','DATA/raw_data/MM-GAG/ground_images_jpg/x.jpg','DATA/raw_data/SwissView/data/SwissViewMonuments/../../../../secret.jpg']:
            with self.assertRaises(ValueError):raw_path(root,p)
    def test_group_split_is_stable_and_disjoint(self):
        names=[f'group{i}' for i in range(14)];a=split_scenes(names);b=split_scenes(names[::-1]);self.assertEqual(a,b)
        self.assertEqual(len(a['fit']),9);self.assertEqual(len(a['development']),5);self.assertFalse(set(a['fit'])&set(a['development']))
        with self.assertRaises(ValueError):split_scenes(names[:13])
    def test_pixel_adapter_returns_only_readonly_single_images(self):
        g=Image.new('RGB',(20,30),(1,2,3));a=Image.new('RGB',(1500,1500),(0,0,0));d=ImageDraw.Draw(a);d.rectangle((900,900,1199,1199),fill=(0,255,0))
        v=pixel_pair(g,a,18);self.assertEqual(set(v),{'target_rgb','current_rgb'});self.assertEqual(v['current_rgb'].shape,(300,300,3));self.assertTrue((v['current_rgb'][:,:,1]==255).all())
        self.assertFalse(v['current_rgb'].flags.writeable);self.assertFalse(v['target_rgb'].flags.writeable)
    def test_wrong_image_size_and_quarantined_pair_rejected(self):
        with self.assertRaises(ValueError):pixel_pair(Image.new('RGB',(2,2)),Image.new('RGB',(100,100)),0)
        with self.assertRaises(ValueError):materialize_pair('.',{'status':'quarantined','label_semantics':None,'capture_point_verified':False})
    def test_unproven_capture_point_label_cannot_be_promoted(self):
        with self.assertRaises(ValueError):materialize_pair('.',{'status':'weak_scene_engineering_only','label_semantics':'capture_point','capture_point_verified':True})
if __name__=='__main__':unittest.main()
