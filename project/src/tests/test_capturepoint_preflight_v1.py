import sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from tools.preflight_capturepoint_sources_v1 import axis_scores,PROBES,mapped,clean_pair
class PixelBoundaries(unittest.TestCase):
    def test_asymmetric_row_and_column_identification(self):
        rng=np.random.default_rng(11);tiles=rng.integers(0,256,size=(81,16,16,3),dtype=np.uint8)
        row=np.stack([tiles[(y-1)*9+x-1] for y,x in PROBES])
        col=np.stack([tiles[(x-1)*9+y-1] for y,x in PROBES])
        self.assertTrue(axis_scores(tiles,row)['passed']);self.assertEqual(axis_scores(tiles,col)['best'],'column')
        self.assertFalse(axis_scores(tiles,col)['passed'])
    def test_uniform_ambiguous_axis_is_rejected(self):
        self.assertFalse(axis_scores(np.zeros((81,16,16,3),np.uint8),np.zeros((8,16,16,3),np.uint8))['passed'])
    def test_all_probes_and_transforms_stay_inside_unique81(self):
        for name in ['row','column','flip_row','flip_column','rotate180','rotate90','rotate270','antitranspose']:
            for y,x in PROBES:self.assertTrue(all(1<=v<=9 for v in mapped(name,y,x)))
    def test_pair_has_only_readonly_rgb(self):
        im=Image.new('RGB',(7,9));im.info['secret']='not public'
        p=clean_pair(im,im);self.assertEqual(set(p),{'target_rgb','current_rgb'})
        self.assertTrue(all(a.dtype==np.uint8 and not a.flags.writeable for a in p.values()))
if __name__=='__main__':unittest.main()
