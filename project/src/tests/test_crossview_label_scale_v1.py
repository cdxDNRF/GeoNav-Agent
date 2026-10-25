"""Boundary tests for frozen label/actual-crop diagnostics, no image files/models."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.crossview_label_scale_v1 import mapping,crop_geometry,gps
class LabelScaleTests(unittest.TestCase):
    def test_all25_labels_invert_to_raw_center(self):
        for g in range(25):
            m=mapping(g);self.assertEqual(len(set(m)),25);self.assertEqual(m[g],(5,5))
            self.assertTrue(all(0<=r<11 and 0<=c<11 for r,c in m))
            gr,gc=divmod(g,5)
            self.assertEqual([(r+gr-5)*5+c+gc-5 for r,c in m],list(range(25)))
    def test_invalid_goal_cannot_silently_relabel(self):
        for x in [-1,25,True,1.0,'1']:
            with self.assertRaises(ValueError):mapping(x)
    def test_portrait_and_landscape_crop_integer_rounding(self):
        for w,h in [(3024,4032),(4032,3024)]:
            x=crop_geometry(w,h);self.assertEqual(sorted(x['resized_size']),[224,298]);self.assertAlmostEqual(x['retained_area_fraction'],224/298)
        for w,h in [(2268,4032),(4032,2268)]:
            x=crop_geometry(w,h);self.assertEqual(sorted(x['resized_size']),[224,398]);self.assertAlmostEqual(x['retained_area_fraction'],224/398)
    def test_square_and_invalid_dimensions(self):
        self.assertEqual(crop_geometry(256,256)['source_crop_box'],[0,0,256,256]);self.assertEqual(crop_geometry(256,256)['retained_area_fraction'],1)
        with self.assertRaises(ValueError):crop_geometry(0,100)
    def test_absent_gps_is_unknown_not_zero_coordinate(self):
        class Empty:
            def get_ifd(self,n):return {}
        self.assertIsNone(gps(Empty()))
        class Equator:
            def get_ifd(self,n):return {1:'N',2:[0,0,0],3:'E',4:[0,0,0]}
        self.assertEqual(gps(Equator()),[0,0])
if __name__=='__main__':unittest.main()
