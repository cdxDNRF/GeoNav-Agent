"""Review regressions: no model, network or navigation required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from project.src.tests import test_live_platform_v2 as live_tests
from project.src.tests import test_cloud_platform_v2 as cloud_tests
from project.src.webapp.live_v3 import LiveManager
from project.src.webapp.server_v3 import make_server
from project.src.webapp.cloud_v3 import CloudConfig, CloudPolicy
try:
    from project.src.webapp.binding_v3 import execution_bindings, validate_bindings, source_closure
except ImportError:
    execution_bindings = validate_bindings = source_closure = None

ROOT = Path(__file__).resolve().parents[3]

class LiveV3Tests(live_tests.LiveTests):
    def setUp(self):
        p=patch.object(live_tests,'LiveManager',LiveManager);p.start();self.addCleanup(p.stop)
        p=patch('project.src.webapp.server_v2.make_server',make_server);p.start();self.addCleanup(p.stop)
        super().setUp()

class CloudV3Tests(cloud_tests.CloudTests):
    def setUp(self):
        p=patch.object(cloud_tests,'CloudPolicy',CloudPolicy);p.start();self.addCleanup(p.stop)
        super().setUp()

class ReviewTests(unittest.TestCase):
    def test_bad_paid_responses_keep_usage_and_digest(self):
        obs=SimpleNamespace(position=(0,0),grid_size=5,remaining_budget=10,visited=(0,),
            legal_actions=('right','down'),current_image=b'current',target_image=b'target')
        for content,reason in [('not json','stop'),('{"action":"left"}','stop'),('{"action":"down"}','length')]:
            response={'choices':[{'message':{'content':content},'finish_reason':reason}],
                'usage':{'total_tokens':41,'prompt_tokens':30,'completion_tokens':11,'private':'ignored'}}
            p=CloudPolicy(CloudConfig('https://example.com/v1','test'),transport=lambda *_:response)
            d=p.act(obs,{'action':'right'})['cloud']
            self.assertTrue(d['fallback']);self.assertEqual(d['usage']['total_tokens'],41)
            self.assertEqual(len(d['response_sha256']),64);self.assertNotIn('private',d['usage'])

    def test_real_inputs_and_scientific_dependency_closure_bound(self):
        self.assertIsNotNone(execution_bindings)
        inputs,sources=execution_bindings(ROOT)
        prefix='DATA/processed_data/MassGIS/工程准备/同产品开发导航兼容_v1/'
        for suffix in ['元数据/原冻结模型配置.json','核验/预登记.json','核验/特征完成.json',
                       '工程数据/grid5/数据清单.json','工程数据/grid10/数据清单.json']:
            self.assertIn(prefix+suffix,inputs)
        for module in ['agents/frozen_edge_navigator.py','agents/massgis_navigator_v1.py',
                       'agents/area_policy.py','train/dyncur_tiny.py','env/massgis_native_area_v2.py']:
            self.assertIn('project/src/'+module,sources)
        validate_bindings(ROOT,inputs,sources)

    def test_bound_dependency_mutation_rejected(self):
        self.assertIsNotNone(validate_bindings)
        with tempfile.TemporaryDirectory(prefix='测试临时_',dir=ROOT/'平台/实时导航平台_v2/核验') as temp:
            root=Path(temp);p=root/'dependency.py';p.write_text('one',encoding='utf-8')
            import hashlib
            values={'dependency.py':hashlib.sha256(p.read_bytes()).hexdigest()}
            validate_bindings(root,{},values)
            p.write_text('two',encoding='utf-8')
            with self.assertRaises(ValueError):validate_bindings(root,{},values)

if __name__=='__main__':unittest.main()
