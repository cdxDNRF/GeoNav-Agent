import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_geometry_v1 as g
from data import massgis_pool_v1 as m

def feature(x,y,sheet):
    b=[x,y,x+4000,y+4000]
    return {'id':str(sheet),'bbox':b,'geometry':{'type':'Polygon','coordinates':[[[x,y],[x,y+4000],[x+4000,y+4000],[x+4000,y],[x,y]]]},
        'properties':{'sheet_id':sheet,'url_lossy':'https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/images/coq2005_hm_jp2_lossy/'+str(sheet)+'.zip'}}
def index():
    return {'crs':{'properties':{'name':'urn:ogc:def:crs:EPSG::26986'}},'numberMatched':4,'numberReturned':4,
        'features':[feature(x,y,i) for i,(x,y) in enumerate([(0,0),(4000,0),(0,4000),(4000,4000)])]}
H={'regions':[{'bounds_m':[-20000,-20000,-10000,-10000]}]}

class Tests(unittest.TestCase):
    def test_complete(self):
        r=g.analyze(index(),H);self.assertEqual(r['witness_count'],1);self.assertEqual(r['witness'][0]['bounds_m'],[0,0,4500,4500])
        self.assertIsNone(r['data_gate_passed']);self.assertEqual(r['formal_quality_passed_regions'],0)
    def test_missing(self):
        d=index();d['features'].pop();d['numberMatched']=d['numberReturned']=3
        self.assertEqual(g.analyze(d,H)['witness_count'],0)
    def test_history(self):self.assertEqual(g.analyze(index(),{'regions':[{'bounds_m':[7500,0,10000,4000]}]})['witness_count'],1)
    def test_near_history(self):self.assertEqual(g.analyze(index(),{'regions':[{'bounds_m':[7499,0,10000,4000]}]})['witness_count'],0)
    def test_precision(self):
        f=feature(.016,0,1)
        with self.assertRaises(ValueError):g.tile(f)
    def test_no_bbox_only(self):
        f=feature(0,0,1);f['geometry']['coordinates'][0][1]=[1,4000]
        with self.assertRaises(ValueError):g.tile(f)
    def test_pagination(self):
        d=index();d['numberMatched']=5
        with self.assertRaises(ValueError):g.analyze(d,H)
    def test_crs(self):
        d=index();d['crs']['properties']['name']='EPSG:4326'
        with self.assertRaises(ValueError):g.analyze(d,H)
    def test_duplicate(self):
        d=index();d['features'][1]=copy.deepcopy(d['features'][0])
        with self.assertRaises(ValueError):g.analyze(d,H)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    m.write(m.OUT/'核验/几何单元测试.json',{'successful':result.wasSuccessful(),'tests_run':result.testsRun,'network_requests':0})
    raise SystemExit(0 if result.wasSuccessful() else 1)
