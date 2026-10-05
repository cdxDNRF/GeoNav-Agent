from pathlib import Path
from io import BytesIO
import sys,unittest
import numpy as np
from PIL import Image,PngImagePlugin
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.massgis_edge_bridge_v1 import native_profiles,native_edge_features
from agents.edge_cue import image_profiles
from data import massgis_native_v1 as m
def payload(a,metadata=False):
    stream=BytesIO();info=PngImagePlugin.PngInfo()
    if metadata:info.add_text('private','evaluator source identity')
    Image.fromarray(a).save(stream,format='PNG',pnginfo=info);return stream.getvalue()

class Tests(unittest.TestCase):
    def test_equivalent_native_replication(self):
        old=np.random.default_rng(17).integers(0,256,(300,300,3),dtype=np.uint8)
        native=old.repeat(2,0).repeat(2,1)
        self.assertTrue(np.array_equal(native_profiles(payload(native)),image_profiles(old)))
    def test_no_metadata_feature(self):
        a=np.full((600,600,3),70,np.uint8)
        self.assertTrue(np.array_equal(native_profiles(payload(a)),native_profiles(payload(a,True))))
    def test_wrong_size(self):
        with self.assertRaises(ValueError):native_profiles(payload(np.zeros((300,300,3),np.uint8)))
    def test_finite_features(self):
        a=np.zeros((600,600,3),np.uint8);b=np.full_like(a,200)
        f=native_edge_features(payload(a),payload(b));self.assertEqual(f.shape,(20,));self.assertTrue(np.isfinite(f).all())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    m.write(m.QA/'输入适配单元测试.json',dict(successful=result.wasSuccessful(),tests_run=result.testsRun,new_SR=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
