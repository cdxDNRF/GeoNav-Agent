import dataclasses,importlib.util,sys,unittest,tempfile,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('mmgag_views_test',ROOT/'project/src/data/mmgag_views_v1.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class ViewTests(unittest.TestCase):
    def test_all_goal_windows_keep_hidden_global_center(self):
        for g in range(25):
            cells=m.window_positions(g);self.assertEqual(cells[g],(5,5));self.assertEqual(len(set(cells)),25)
            self.assertTrue(all(0<=r<11 and 0<=c<11 for r,c in cells))
    def test_directions_are_image_rows_and_columns(self):
        cells=m.window_positions(12)
        self.assertEqual(cells[7],(4,5));self.assertEqual(cells[13],(5,6));self.assertEqual(cells[17],(6,5));self.assertEqual(cells[11],(5,4))
    def test_filename_is_numeric(self):
        self.assertEqual(tuple(map(int,m.PATCH.fullmatch('sat_0_10.jpg').groups())),(0,10))
        self.assertIsNone(m.PATCH.fullmatch('stitched_image.jpg'))
    def test_bad_goal_rejected(self):
        for g in [-1,25,1.2,True]:
            with self.assertRaises(ValueError):m.window_positions(g)
    def test_exif_orientation_and_metadata_removed(self):
        im=Image.new('RGB',(3,2));im.putpixel((0,0),(255,0,0));exif=Image.Exif();exif[274]=6;im.info['exif']=exif.tobytes();im.info['filename']='private-source';im.info['GPS']='private'
        a=m.clean_rgb(im,ground=True);self.assertEqual(a.shape,(3,2,3));self.assertEqual(tuple(a[0,1]),(255,0,0));self.assertFalse(a.flags.writeable)
        self.assertFalse(Image.fromarray(a).info)
    def test_public_input_cannot_accept_truth_or_map(self):
        rgb=m.clean_rgb(Image.new('RGB',(2,2)))
        obj=m.public_request(rgb,rgb,row=1,col=2,grid_size=5,remaining_steps=9,visited=((1,2),))
        self.assertEqual({f.name for f in dataclasses.fields(obj)},{'current_rgb','target_rgb','target_modality','row','col','grid_size','remaining_steps','visited'})
        with self.assertRaises(TypeError):m.public_request(rgb,rgb,row=1,col=2,grid_size=5,remaining_steps=9,visited=((1,2),),goal_index=12)
    def test_channel_schema_contains_no_fake_local_features(self):
        rgb=m.clean_rgb(Image.new('RGB',(2,2)))
        self.assertFalse(hasattr(m.public_request(rgb,rgb,row=0,col=0,grid_size=5,remaining_steps=10,visited=()),'edge_features'))
    def test_normalization_and_bad_features(self):
        self.assertTrue(np.allclose(m.normalize_features([[3,4]]),[[.6,.8]]))
        for x in [[[0,0]],[[float('nan'),1]]]:
            with self.assertRaises(ValueError):m.normalize_features(x)
    def test_aerial_normalization(self):
        rgb=m.clean_rgb(Image.new('RGB',(256,256),(255,0,0)))
        a=m.aerial_pixels(rgb);self.assertEqual(a.shape,(3,224,224));self.assertAlmostEqual(float(a[0,0,0]),(1-.3670)/.2209,places=5)
    def test_real_numeric_filename_window_and_duplicate_rejection(self):
        root=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1/工程临时';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as td:
            p=Path(td).resolve();self.assertTrue(p.is_relative_to(root.resolve()))
            for r in range(11):
                for c in range(11):(p/f'sat_{r}_{c}.jpg').touch()
            (p/'stitched_image.jpg').touch()
            window=m.numeric_window(p,24,'column')
            self.assertEqual(window[24].name,'sat_5_5.jpg');self.assertEqual(window[0].name,'sat_1_1.jpg')
            (p/'sat_00_1.jpg').touch()
            with self.assertRaises(ValueError):m.tile_index(p)
    def test_capacity_aware_groups_never_split_or_read_scores(self):
        spec2=importlib.util.spec_from_file_location('cv_prepare_test',ROOT/'project/src/tools/prepare_crossview_probe_v1.py');prep=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(prep)
        groups=[[f'g{i}_{j}' for j in range(4 if i<3 else 1)] for i in range(30)]
        hashes={s:hashlib.sha256(s.encode()).hexdigest() for g in groups for s in g}
        dev,hold=prep.select_groups(groups,hashes)
        self.assertEqual((len(dev),len(hold)),(10,20));self.assertGreaterEqual(sum(min(4,len(g)) for g in dev),16)
        self.assertFalse(set(s for g in dev for s in g)&set(s for g in hold for s in g))
if __name__=='__main__':unittest.main()
