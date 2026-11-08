"""Synthetic policy/transport only; D-scoped temporary audit journals."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from project.src.tests.test_controller_hub_v1 import SceneOwner,OWNER,ROOT
from project.src.platform_core.scene_workspace_v3 import SceneWorkspace
from project.src.platform_core.controller_hub_v2 import ControllerHub
from project.src.platform_core.api_controller_v1 import API,RequestLedger
from project.src.platform_core.api_comparison_v1 import APIComparisons
from project.src.webapp.cloud_v3 import CloudConfig

TEMP=ROOT/'平台/实验工作台_v4/单模型API统一接入与同题对照_v1/测试'


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=TEMP);self.store=Path(self.tmp.name).resolve();assert self.store.is_relative_to(TEMP.resolve())
        w=SceneOwner();self.w=w;w.store=self.store;w.scenes=SceneWorkspace(w);w.config=CloudConfig('https://example.test/v1','synthetic','SECRET_SENTINEL',timeout=1,max_tokens=32)
        w.api_ledger=RequestLedger(self.store);w.api_transport=self.transport;w.scene_policies=lambda did:['M0','Frontier','Cloud'];w.api_local_factory=self.factory;w.controllers=ControllerHub(w)
        self.payloads=[];self.local_calls=[];self.response=dict(choices=[dict(finish_reason='stop',message=dict(content='{"action":"right","evidence":"observed road"}'))],usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15))
    def tearDown(self):self.tmp.cleanup()
    def factory(self,definition):
        outer=self
        class Runtime:
            def decide(self):outer.local_calls.append(self.obs);return dict(action='down',reason='synthetic M0 fallback',explorer_logits=[0,0,1,0])
        return Runtime()
    def transport(self,config,payload):self.payloads.append(payload);return self.response
    def create(self):return self.w.controllers.create(dict(name='synthetic API',dataset_id='masa-s2',task_key='t0',controller=API,seed=0),OWNER)
    def test_browse_and_create_make_no_network_calls(self):
        self.assertIn(API,self.w.controllers.datasets()[1]['controllers']);v=self.create();self.assertEqual(v['api_requests'],0);self.assertEqual(self.payloads,[])
    def test_payload_has_only_images_and_public_state_not_local_advice(self):
        v=self.create();v=self.w.controllers.step(v['id'],OWNER,kind='aerial');p=self.payloads[0]
        text=json.dumps(p);self.assertNotIn('explorer_logits',text);self.assertNotIn('synthetic M0',text);self.assertNotIn('goal_index',text);self.assertNotIn('source',text)
        contents=p['messages'][1]['content'];self.assertEqual(sum(x['type']=='image_url' for x in contents),2)
        state=json.loads(contents[-1]['text']);self.assertEqual(set(state),{'position','grid_size','remaining_budget','visited','legal_actions'})
        self.assertEqual(v['trajectory'][-1]['origin'],'api');self.assertEqual(v['api_requests'],1);self.assertEqual(v['api_usage']['usage']['total_tokens'],15)
    def test_no_credentials_in_saved_journal(self):
        self.response['choices'][0]['message']['content']='{"action":"right","evidence":"SECRET_SENTINEL"}'
        v=self.create();self.w.controllers.step(v['id'],OWNER,kind='aerial')
        text=''.join(p.read_text('utf-8') for p in self.store.rglob('*.json'));self.assertNotIn('SECRET_SENTINEL',text)
    def test_bad_schema_uses_local_fallback_and_three_errors_break_group(self):
        self.response['choices'][0]['message']['content']='{"action":"right","extra":1}'
        v=self.create();rid=v['id']
        for _ in range(3):v=self.w.controllers.step(rid,OWNER,kind='aerial')
        self.assertEqual(v['api_fallbacks'],3);self.assertEqual(v['trajectory'][-1]['origin'],'fallback')
        with self.assertRaises(ValueError):self.w.controllers.step(rid,OWNER,kind='aerial')
        self.assertEqual(len(self.payloads),3);self.assertEqual(len(self.local_calls),3)
    def test_transport_error_is_one_attempt_no_retry(self):
        def fail(*a):self.payloads.append('failed');raise TimeoutError()
        self.w.api_transport=fail;v=self.create();v=self.w.controllers.step(v['id'],OWNER,kind='aerial')
        self.assertEqual(len(self.payloads),1);self.assertEqual(v['api_fallbacks'],1);self.assertEqual(v['api_usage']['requests'],1)
    def test_pending_request_blocks_new_calls(self):
        v=self.create();raw=self.w.scenes.runs[v['id']];obs=self.w.scenes._observation(raw,0)
        self.w.api_ledger.reserve(v['api_group_id'],v['id'],obs,self.w.config)
        with self.assertRaises(ValueError):self.w.controllers.step(v['id'],OWNER,kind='aerial')
        self.assertEqual(self.payloads,[])
    def test_human_history_sync_does_not_call_api(self):
        v=self.create();rid=v['id'];self.w.controllers.mode(rid,OWNER,'manual','aerial');self.w.controllers.step(rid,OWNER,'down','aerial');self.w.controllers.mode(rid,OWNER,API,'aerial')
        v=self.w.controllers.step(rid,OWNER,kind='aerial');self.assertEqual(len(self.local_calls),2);self.assertEqual(len(self.payloads),1);self.assertEqual(v['controller_class'],'mixed')
    def test_config_changes_rejected_before_api(self):
        v=self.create();self.w.config=CloudConfig('https://example.test/v1','changed',timeout=1,max_tokens=32)
        with self.assertRaises(ValueError):self.w.controllers.step(v['id'],OWNER,kind='aerial')
        self.assertEqual(self.payloads,[])
    def test_ground_api_rejected_without_requests(self):
        with self.assertRaises(ValueError):self.w.controllers.create(dict(name='x',dataset_id='camera-development',task_key='x',controller=API,seed=0),OWNER)
        self.assertEqual(self.payloads,[])
    def test_save_reload_preserves_audit_and_classification(self):
        v=self.create();rid=v['id'];self.w.controllers.step(rid,OWNER,kind='aerial');self.w.controllers.finish(rid,OWNER,'aerial');v=self.w.controllers.save_run(rid,OWNER,'aerial')
        self.assertEqual(v['controller_class'],'api');self.assertEqual(v['result']['api_actions'],1);before={str(p):p.read_bytes() for p in self.store.rglob('*.json')}
        self.w.scenes.release(OWNER);v=self.w.controllers.state(rid,OWNER,'aerial');self.assertTrue(v['readonly']);self.assertEqual(v['api_usage']['requests'],1)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.store.rglob('*.json')})


class ComparisonTests(unittest.TestCase):
    factory=APITests.factory
    transport=APITests.transport
    def setUp(self):
        APITests.setUp(self);self.w.datasets['masa-s2']['tasks']=[dict(self.w.base,key='t'+str(i),episode_id='t'+str(i)) for i in range(10)]
        self.c=APIComparisons(self.w);ref=dict(tasks=['t'+str(i) for i in range(10)],baseline={},reference_sha256={},input_sha256={},seed=0,budget=10)
        for k in ref['tasks']:ref['baseline'][k]={c:dict(result=dict(success=c=='M0',SG_cells=0 if c=='M0' else 2)) for c in ['M0','frontier']}
        self.p=patch.object(self.c,'reference',return_value=ref);self.p.start()
    def tearDown(self):self.p.stop();APITests.tearDown(self)
    def create_group(self):return self.c.create(dict(name='synthetic fixed comparison'),OWNER)
    def test_group_reuses_baselines_without_model_execution(self):
        v=self.create_group();self.assertEqual(self.local_calls,[]);self.assertEqual(self.payloads,[]);self.assertTrue(v['arms']['M0']['reused']);self.assertEqual(v['arms']['M0']['SR'],1)
        v=self.c.step(v['id'],OWNER);self.assertEqual(len(self.local_calls),1);self.assertEqual(v['api_usage']['requests'],1);self.assertEqual(len(v['rows']),10)
    def test_group_failure_keeps_partial_trace_and_plan(self):
        self.response['choices'][0]['message']['content']='{"type":"wrong"}';v=self.create_group();gid=v['id']
        for _ in range(3):self.c.step(gid,OWNER)
        with self.assertRaises(ValueError):self.c.step(gid,OWNER)
        v=self.c.state(gid,OWNER);self.assertEqual(v['status'],'failed');self.assertEqual(v['rows'][0]['partial']['step'],3);self.assertEqual(v['api_usage']['requests'],3);self.assertEqual(v['arms'][API]['planned'],10);self.assertIsNone(v['arms'][API]['SG_cells'])
    def test_export_does_not_execute_api(self):
        v=self.create_group();self.c.step(v['id'],OWNER);before=len(self.payloads)
        self.assertEqual(len(self.c.export_csv(v['id'],OWNER).splitlines()),11);self.assertEqual(len(self.payloads),before)


if __name__=='__main__':unittest.main()
