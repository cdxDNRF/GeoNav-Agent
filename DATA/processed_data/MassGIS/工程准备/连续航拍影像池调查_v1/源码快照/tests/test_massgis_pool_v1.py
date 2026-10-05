"""Meaningful request-origin and spatial-exclusion checks; no network."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_pool_v1 as m

class Tests(unittest.TestCase):
    def test_official(self):m.valid_url('https://gis-prod.digital.mass.gov/geoserver/wfs?service=WFS')
    def test_suffix_attack(self):
        with self.assertRaises(ValueError):m.valid_url('https://mass.gov.evil.example/index')
    def test_credentials(self):
        with self.assertRaises(ValueError):m.valid_url('https://secret@mass.gov/index')
    def test_http(self):
        with self.assertRaises(ValueError):m.valid_url('http://mass.gov/index')
    def test_render(self):
        with self.assertRaises(ValueError):m.valid_url('https://gis-prod.digital.mass.gov/exportImage')
    def test_overlap(self):self.assertEqual(m.gap([0,0,4500,4500],[100,100,5000,5000]),0)
    def test_boundary(self):self.assertEqual(m.gap([0,0,4500,4500],[7500,0,12000,4500]),3000)
    def test_corner(self):self.assertEqual(m.gap([0,0,4500,4500],[7500,8500,12000,13000]),5000)
    def test_invalid(self):
        for b in ([0,0,0,1],[0,0,float('nan'),1]):
            with self.assertRaises(ValueError):m.gap(b,[0,0,1,1])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    m.write(m.ROOT/'选题报告相关/连续影像池调查测试_v1.json',dict(successful=result.wasSuccessful(),tests_run=result.testsRun,network_requests=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)
