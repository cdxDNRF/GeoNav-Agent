"""Synthetic, offline checks of the acquisition/QC invariants."""
from pathlib import Path
from io import BytesIO
import sys
import unittest
import struct
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_confirmation_pool_v1 as m
from eval.audit_massgis_confirmation_pool_v1 import blank_count,raw_stream,distance
from data.massgis_native_v2 import codestream

class Frozen(unittest.TestCase):
    def test_integer_source_threshold(self):
        self.assertTrue(m.quality_pass(1,100,[dict(count=1,pixels=100)]))
        self.assertFalse(m.quality_pass(2,100,[dict(count=0,pixels=100)]))
    def test_support_defect_not_diluted(self):
        self.assertFalse(m.quality_pass(0,10000,[dict(count=2,pixels=100)]))
    def test_cell_threshold(self):
        self.assertTrue(m.quality_pass(0,100,[],[dict(blank_count=2,pixels=100)]))
        self.assertFalse(m.quality_pass(0,100,[],[dict(blank_count=3,pixels=100)]))
    def test_independent_blank_equation(self):
        x=np.array([[[0,0,0],[255,255,255],[0,0,1],[254,255,255]]],np.uint8)
        self.assertEqual(blank_count(x),2);self.assertEqual(int(m.base.blank(x).sum()),2)
    def test_nir_not_alpha(self):
        x=np.array([[[1,2,3,0],[4,5,6,255]]],np.uint8)
        np.testing.assert_array_equal(m.base.rgb_first_three(x),x[:,:,:3])
    def test_unchanged_codestream(self):
        payload=b'\xff\x4f\x00\x00';box=struct.pack('>I4s',12,b'jp2c')+payload
        prefix=struct.pack('>I4s',12,b'abcd')+b'1234'
        self.assertEqual(codestream(prefix+box),payload);self.assertEqual(raw_stream(prefix+box),payload)
    def test_to_eof_codestream(self):
        data=struct.pack('>I4s',0,b'jp2c')+b'raw';self.assertEqual(raw_stream(data),codestream(data))
    def test_attempt_cap(self):
        with self.assertRaises(ValueError):m.reserve([dict(url='u',name=str(i)) for i in range(240)],[],'v','get')
    def test_per_url_cap(self):
        with self.assertRaises(ValueError):m.reserve([dict(url='u',name=str(i)) for i in range(3)],[],'u','get')
    def test_unsettled_charged(self):
        starts=[dict(url='old',name='x')];self.assertEqual(m.reserve(starts,[],'u','get'),24*1024**2)
    def test_settled_not_doublecharged(self):
        self.assertEqual(m.reserve([dict(url='old',name='x')],[dict(name='x',body_bytes=7)],'u','head'),7)
    def test_byte_cap(self):
        with self.assertRaises(ValueError):m.reserve([],[dict(name='x',body_bytes=3*1024**3-1)],'u','get')
    def test_geographic_distance(self):
        for a,b in [([0,0,4,4],[7,8,9,10]),([0,0,4,4],[4,0,8,4]),([0,0,4,4],[1,1,3,3])]:
            self.assertEqual(distance(a,b),m.base.old.gap(a,b))
    def test_north_up_native_slice(self):
        ys,xs=m.base.slice_pixels([0,3000,300,3300],[0,0,4500,4500])
        self.assertEqual((ys.start,ys.stop,xs.start,xs.stop),(2400,3000,0,600))
    def test_png_native_exact(self):
        x=np.arange(108,dtype=np.uint8).reshape(6,6,3);buf=BytesIO();Image.fromarray(x).save(buf,format='PNG')
        with Image.open(BytesIO(buf.getvalue())) as im:np.testing.assert_array_equal(np.array(im),x)
    def test_frozen_queue_and_development_exclusion(self):
        regs=m.read(m.POOL/'元数据/隔离区域见证队列.json')['candidates'];history=m.read(m.POOL/'元数据/历史排除边界.json')['regions']
        history=history+[dict(bounds_m=[233000,894000,241000,902000])]
        self.assertEqual(len(m.geometry(regs,history)),20)
        bad=[dict(r) for r in regs];bad[1]=dict(bad[1],bounds_m=bad[0]['bounds_m'],sources=bad[0]['sources'])
        with self.assertRaises(ValueError):m.geometry(bad,history)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Frozen)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    m.write(m.TESTS,dict(successful=result.wasSuccessful(),tests_run=result.testsRun,network_requests=0,policy_calls=0,new_SR=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
