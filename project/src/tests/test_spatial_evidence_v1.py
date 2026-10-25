"""Saved CV041 viewer regressions. No model/encoding/training/navigation."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
from PIL import Image
from project.src.platform_core.spatial_evidence_reader_v1 import SpatialEvidenceReader,EvidenceUnavailable,BATCH
from project.src.tools.spatial_evidence_transfer_v1 import verified_spatial_assets

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'平台/实验工作台_v4/空间小头对照证据查阅_v1'


class Tests(unittest.TestCase):
    def setUp(self):self.reader=SpatialEvidenceReader(ROOT)
    def test_catalog_denominators_and_all_reject_reference(self):
        c=self.reader.catalog();self.assertEqual(len(c['cases']),150);self.assertEqual(c['development_outputs'],2850)
        self.assertEqual(c['metrics']['Spatial/Full']['positive_joint'],0);self.assertAlmostEqual(c['always_reject_macro_joint'],2/3)
        self.assertEqual(c['fixed_location_upper_bound']['within5m'],3);self.assertFalse(c['navigation_admitted'])
    def test_cities_methods_seeds_conditions_and_relations(self):
        c=self.reader.catalog()
        for city in (s['id'] for s in c['sources']):
            for rel in ('inside','outside','unrelated'):
                case=next(v for v in c['cases'] if v['city']==city and v['relation']==rel)
                for method in ('Spatial','TrainedMean','RawCos'):
                    for seed in (('none',) if method=='RawCos' else ('0','1','2')):
                        for arm in (('NoTarget',) if method=='TrainedMean' else ('Full','Disabled','Mean','Wrong')):
                            d=self.reader.detail(str(case['id']),method,seed,arm)
                            self.assertEqual(d['relation'],rel);self.assertEqual(d['truth_xy'] is not None,rel=='inside');self.assertEqual(len(d['heat']),7)
                            self.assertTrue(d['bounds'][0]<=d['prediction'][0]<d['bounds'][2]);self.assertEqual(d['new_model_calls'],0)
    def test_wrong_actual_photo_original_goal_and_crop_sizes(self):
        c=self.reader.catalog()
        for city in ('amsterdam','krakow','stockholm'):
            case=next(v for v in c['cases'] if v['city']==city and v['relation']=='inside');i=str(case['id'])
            true=self.reader.image(i,'Spatial','0','Full','target');wrong=self.reader.image(i,'Spatial','0','Wrong','target')
            self.assertNotEqual(true,wrong);self.assertEqual(true,self.reader.image(i,'Spatial','0','Wrong','original'))
            d=self.reader.detail(i,'Spatial','0','Full')
            with Image.open(BytesIO(self.reader.image(i,'Spatial','0','Full','current'))) as im:self.assertEqual(im.size,(d['bounds'][2]-d['bounds'][0],d['bounds'][3]-d['bounds'][1]))
    def test_no_target_photo_explanatory_and_raw_seed_not_duplicated(self):
        self.assertTrue(self.reader.detail('13','TrainedMean','0','NoTarget')['target_is_explanatory'])
        self.assertIsNone(self.reader.detail('13','RawCos','none','Full')['seed'])
        with self.assertRaises(EvidenceUnavailable):self.reader.detail('13','RawCos','0','Full')
    def test_invalid_selection_paths_and_method_combinations(self):
        for case in ('-1','150','../0','０',None):
            with self.assertRaises(EvidenceUnavailable):self.reader.detail(case,'Spatial','0','Full')
        for args in [('0','bogus','0','Full'),('0','Spatial','none','Full'),('0','TrainedMean','0','Full')]:
            with self.assertRaises(EvidenceUnavailable):self.reader.detail(*args)
        with self.assertRaises(EvidenceUnavailable):self.reader.image('0','Spatial','0','Full','../file')
    def test_missing_tampered_prediction_summary_and_image_fail_closed(self):
        with self.assertRaises(EvidenceUnavailable):SpatialEvidenceReader(OUT/'missing-data-root').catalog()
        original=Path.read_bytes
        for name,call in [('主对照汇总.json',lambda:self.reader.catalog()),('amsterdam__Spatial__0__development__Full.json',lambda:self.reader.detail('13','Spatial','0','Full')),('amsterdam_ground.png',lambda:self.reader.image('13','Spatial','0','Full','target'))]:
            def changed(p,selected=name):
                raw=original(p);return raw+b' ' if p.name==selected else raw
            with patch.object(Path,'read_bytes',changed):
                with self.assertRaises(EvidenceUnavailable):call()
    def test_minimal_portable_assets_no_model_feature_or_fit_payload(self):
        a=verified_spatial_assets(ROOT);self.assertEqual(len(a),70)
        self.assertEqual(sum('/记录/' in k for k in a),57)
        self.assertTrue(all(not k.startswith(('models/','DATA/raw_data/')) and not k.endswith(('.pt','.npz')) and '__fit_' not in k for k in a))
    def test_export_is_original_record_and_provenance_not_new_run(self):
        d=self.reader.export('13','Spatial','0','Full');self.assertEqual(d['format'],'geonav-spatial-evidence-v1')
        self.assertFalse(d['record']['accepted']);self.assertTrue(d['record']['provenance']['seal_sha256']);self.assertEqual(len(d['record']['logits']),50)
    @unittest.skipUnless((OUT/'测试').is_dir(),'await approved QA test workspace')
    def test_http_readonly_tokens_existing_pages_and_same_origin(self):
        from project.src.platform_core.service_v11 import Workbench
        from project.src.webapp.workbench_server_v20 import make_server
        with tempfile.TemporaryDirectory(prefix='http-',dir=OUT/'测试') as store:
            w=Workbench(ROOT,store,bindings=False);server=make_server(w,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}'
            try:
                for page in ('/','/scenes','/ground-live','/ground-walk','/perception','/spatial-evidence'):
                    with urllib.request.urlopen(url+page) as f:self.assertIn(b'/spatial-evidence',f.read())
                with urllib.request.urlopen(url+'/api/meta') as f:m=json.load(f);self.assertEqual(m['version'],'geonav-workbench-v4.13');self.assertTrue(m['scene_token']);self.assertTrue(m['workflow_token'])
                with urllib.request.urlopen(url+'/api/spatial-evidence/catalog') as f:self.assertEqual(json.load(f)['base_cases'],150)
                q='?case=13&method=Spatial&seed=0&arm=Full'
                with urllib.request.urlopen(url+'/api/spatial-evidence/export'+q) as f:self.assertIn('attachment',f.headers['Content-Disposition']);self.assertEqual(json.load(f)['record']['id'],13)
                with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(urllib.request.Request(url+'/api/spatial-evidence/catalog',headers={'Origin':'https://untrusted.invalid'}))
                self.assertEqual(e.exception.code,403)
                with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(urllib.request.Request(url+'/api/spatial-evidence/detail',data=b'{}',headers={'Content-Type':'application/json'}))
                self.assertEqual(e.exception.code,405)
                with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(url+'/api/spatial-evidence/detail?case=../0&method=Spatial&seed=0&arm=Full')
            finally:server.shutdown();server.server_close();w.shutdown();thread.join(3)


if __name__=='__main__':unittest.main()
