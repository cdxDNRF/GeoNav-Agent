"""In-memory boundary checks; no directories, models, real data or network."""
import copy
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from project.src.platform_core.controller_hub_v1 import ControllerHub,MASA,CAMERA
from project.src.platform_core.controller_batch_v1 import ControllerBatches
from project.src.platform_core.scene_workspace_v2 import SceneWorkspace
from project.src.platform_core.scene_workspace_v1 import signature

ROOT=Path(__file__).resolve().parents[3];OWNER='a'*32


class SceneOwner:
    def __init__(self):
        self.root=ROOT;self.store=ROOT;self.sources={};self.calls=[]
        self.base=dict(key='t0',episode_id='t0',area='synthetic',split='val',grid_size=5,budget=10,start=0,goal=24)
        self.datasets={MASA:dict(id=MASA,grid=5,budget=10,tasks=[self.base])}
        self.capturepoint_sessions=Mock();self.scenes=SceneWorkspace(self)
        self.scenes.directory=ROOT/'__never_created_controller_test__'
    def _owner(self,value):
        if value!=OWNER:raise ValueError('owner')
        return value
    def scene_dataset(self,did):return self.datasets[did]
    def scene_bindings(self,*args):return {}
    def scene_validate(self,*args):pass
    def scene_policies(self,*args):return ['M0','Frontier']
    def scene_image(self,*args):
        from PIL import Image
        stream=BytesIO();Image.new('RGB',(4,4),(12,34,56)).save(stream,format='PNG');return stream.getvalue()
    def scene_runtime(self,definition,policy):
        parent=self
        class Runtime:
            def decide(self):
                obs=self.obs;parent.calls.append((policy,obs))
                return dict(action='right',reason='synthetic',explorer_logits=[0.,1.,0.,0.])
        return Runtime()


class HubTests(unittest.TestCase):
    def setUp(self):self.owner=SceneOwner();self.h=ControllerHub(self.owner)
    def create(self,controller='manual'):
        return self.h.create(dict(name='synthetic',dataset_id=MASA,task_key='t0',controller=controller,seed=0),OWNER)
    def test_no_truth_in_active_public_view(self):
        v=self.create();self.assertIsNone(v['result']);self.assertNotIn('goal',v['definition']);self.assertNotIn('goal',v['task']);self.assertEqual(v['model_calls'],0)
    def test_ground_m0_rejected_before_factory(self):
        with self.assertRaises(ValueError):self.h.create(dict(name='x',dataset_id=CAMERA,task_key='t0',controller='M0',seed=0),OWNER)
        self.assertEqual(self.owner.calls,[]);self.owner.capturepoint_sessions.create.assert_not_called()
    def test_manual_and_model_provenance_and_causal_sync(self):
        v=self.create();rid=v['id'];self.h.step(rid,OWNER,'down','aerial');self.h.mode(rid,OWNER,'M0','aerial');v=self.h.step(rid,OWNER,kind='aerial')
        self.assertEqual(v['controller_class'],'mixed');self.assertEqual(v['model_calls'],2)
        self.assertEqual([o.position for _,o in self.owner.calls],[(0,0),(1,0)])
        self.assertEqual([o.visited for _,o in self.owner.calls],[(0,),(0,5)])
        for _,obs in self.owner.calls:
            self.assertNotIn('goal',vars(obs));self.assertNotIn('area',vars(obs));self.assertNotIn('distance',vars(obs))
    def test_algorithm_lock_and_external_action_guard(self):
        v=self.create('M0');rid=v['id']
        with self.assertRaises(ValueError):self.h.mode(rid,OWNER,'frontier','aerial')
        with self.assertRaises(ValueError):self.h.step(rid,OWNER,'left','aerial')
        self.assertEqual(self.owner.calls,[]);self.assertEqual(self.h.state(rid,OWNER,'aerial')['step'],0)
    def test_last_step_arrival_and_grid_units(self):
        v=self.create();rid=v['id']
        for a in ['up']*2+['down']*4+['right']*4:v=self.h.step(rid,OWNER,a,'aerial')
        self.assertEqual(v['step'],10);self.assertTrue(v['result']['success']);self.assertIsNone(v['result']['SG_m']);self.assertEqual(v['result']['moved_cells'],8)
    def test_terminal_export_guard(self):
        v=self.create()
        with self.assertRaises(ValueError):self.h.export(v['id'],OWNER,'aerial')
        self.h.finish(v['id'],OWNER,'aerial');self.assertEqual(self.h.export(v['id'],OWNER,'aerial')['status'],'cancelled')
    def test_modified_legacy_scene_rejected(self):
        v=self.create();self.owner.scenes.runs[v['id']]['definition']['task']['goal']=23
        with self.assertRaises(ValueError):self.h.state(v['id'],OWNER,'aerial')
    def test_comparison_rejects_different_seed_and_no_meter_invention(self):
        a=dict(readonly=True,result=dict(success=False,SG_cells=2,moved_m=None),dataset_id=MASA,task=dict(task_key='t0',task_id='t0'),seed=0,definition=dict(input_sha256={}),controller_class='model')
        b=copy.deepcopy(a);b['result'].update(success=True,SG_cells=0)
        with patch.object(self.h,'state',side_effect=[a,b]):
            v=self.h.compare('a','b',OWNER);self.assertTrue(v['recovered']);self.assertIsNone(v['moved_m_delta'])
        b['seed']=1
        with patch.object(self.h,'state',side_effect=[a,b]):
            with self.assertRaises(ValueError):self.h.compare('a','b',OWNER)


class MemoryPath:
    def __init__(self,files,name):self.files=files;self.name=name
    def __truediv__(self,name):return MemoryPath(self.files,self.name+'/'+name)
    def read_text(self,*args):return self.files[self.name]
    def mkdir(self,**kwargs):pass


class BatchHub:
    def __init__(self):
        self.owner=SimpleNamespace(root=ROOT,_owner=lambda o:None,scenes=SimpleNamespace(runs={}),sources={},
            datasets={MASA:dict(tasks=[dict(key='t'+str(i)) for i in range(10)])},masa=SimpleNamespace(bindings=lambda *a:{}))
        self.saved={};self.created=0;self.factory_fail=False
    def task(self,dataset,key):
        if dataset!=MASA or key not in [t['key'] for t in self.owner.datasets[MASA]['tasks']]:raise ValueError('task')
    def create(self,spec,owner):
        if self.factory_fail:raise ValueError('factory unavailable')
        self.created+=1;rid=str(self.created);v=dict(id=rid,step=0,model_calls=0,status='ready',task=spec['task_key'],controller=spec['controller'])
        self.owner.scenes.runs[rid]=v;return copy.deepcopy(v)
    def state(self,rid,owner,kind):return copy.deepcopy(self.owner.scenes.runs.get(rid) or self.saved[rid])
    def step(self,rid,owner,kind):
        v=self.owner.scenes.runs[rid];v['step']+=1;v['model_calls']+=int(v['controller']=='M0')
        if v['step']==2:v.update(status='completed',result=dict(success=v['controller']=='M0',SG_cells=0 if v['controller']=='M0' else 2,coverage=.12,moved_cells=2))
        return copy.deepcopy(v)
    def save_run(self,rid,owner,kind):self.saved[rid]=copy.deepcopy(self.owner.scenes.runs[rid]);return copy.deepcopy(self.saved[rid])
    def detach_saved(self,rid,owner):self.owner.scenes.runs.pop(rid)
    def finish(self,rid,owner,kind):self.owner.scenes.runs[rid]['status']='cancelled';return self.state(rid,owner,kind)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.h=BatchHub();self.b=ControllerBatches(self.h,ROOT);self.files={}
        self.b._folder=lambda bid:MemoryPath(self.files,bid)
        def write(path,value):self.files[path.name]=json.dumps(value)
        self.patches=[patch('project.src.platform_core.controller_batch_v1.atomic',write),patch('project.src.platform_core.controller_batch_v1.durable_snapshot',write)]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in self.patches:p.stop()
    def create(self,tasks=None):return self.b.create(dict(name='in-memory',dataset_id=MASA,task_keys=tasks or ['t0'],controllers=['M0','frontier'],seed=0),OWNER)
    def test_creation_is_canonical_and_no_forward(self):
        v=self.create(['t3','t1']);self.assertEqual(v['definition']['task_keys'],['t1','t3']);self.assertEqual(self.h.created,0);self.assertIsNone(v['metrics']['arms']['M0']['SG_cells'])
    def test_bounds_and_ground_rejected(self):
        for tasks in [[],['t0']*2,['t'+str(i) for i in range(11)]]:
            with self.assertRaises(ValueError):self.b.create(dict(name='x',dataset_id=MASA,task_keys=tasks,controllers=['M0'],seed=0),OWNER)
        with self.assertRaises(ValueError):self.b.create(dict(name='x',dataset_id=CAMERA,task_keys=['t0'],controllers=['M0'],seed=0),OWNER)
    def test_planned_denominator_pair_and_completed_eviction(self):
        v=self.create(['t0','t1']);bid=v['id']
        for _ in range(2):v=self.b.step(bid,OWNER)
        self.assertEqual(v['metrics']['arms']['M0']['SR'],.5);self.assertIsNone(v['metrics']['arms']['M0']['SG_cells']);self.assertFalse(self.h.owner.scenes.runs)
        for _ in range(6):v=self.b.step(bid,OWNER)
        self.assertEqual(v['status'],'completed');self.assertEqual(v['metrics']['paired']['damaged'],2);self.assertEqual(v['action_attempts'],8);self.assertEqual(v['model_calls'],4)
    def test_restart_requires_explicit_consent_and_counts_old_attempt(self):
        bid=self.create()['id'];self.b.step(bid,OWNER);self.b.release(OWNER);self.h.owner.scenes.runs.clear();v=self.b.state(bid,OWNER)
        self.assertTrue(v['needs_restart'])
        with self.assertRaises(ValueError):self.b.step(bid,OWNER)
        self.assertEqual(self.h.created,1);v=self.b.step(bid,OWNER,True)
        self.assertEqual(v['action_attempts'],2);self.assertEqual(v['rows'][0]['attempts'][0]['steps'],1);self.assertEqual(v['rows'][0]['step'],1)
    def test_completed_rows_do_not_rerun_after_restart(self):
        bid=self.create()['id'];self.b.step(bid,OWNER);self.b.step(bid,OWNER);self.b.live.clear();v=self.b.step(bid,OWNER)
        self.assertEqual(self.h.created,2);self.assertEqual(v['rows'][0]['status'],'completed');self.assertEqual(v['current']['controller'],'frontier')
    def test_expired_runtime_requires_retry_instead_of_silent_repeat(self):
        bid=self.create()['id'];self.b.step(bid,OWNER);self.h.owner.scenes.runs.clear()
        self.assertTrue(self.b.state(bid,OWNER)['needs_restart'])
        with self.assertRaises(ValueError):self.b.step(bid,OWNER)
    def test_cross_page_control_guard(self):
        bid=self.create()['id'];self.b.step(bid,OWNER)
        with self.assertRaises(ValueError):self.b.step(bid,'b'*32)
        self.assertFalse(self.b.state(bid,'b'*32)['controllable'])
    def test_factory_failure_stops_queue_without_action(self):
        bid=self.create()['id'];self.h.factory_fail=True
        with self.assertRaises(ValueError):self.b.step(bid,OWNER)
        v=self.b.state(bid,OWNER);self.assertEqual(v['status'],'failed');self.assertEqual(v['action_attempts'],0)
    def test_progress_tamper_rejected(self):
        bid=self.create()['id'];path=bid+'/progress.json';s=json.loads(self.files[path]);s['rows'][0]['step']=9;self.files[path]=json.dumps(s)
        with self.assertRaises(ValueError):self.b.state(bid,OWNER)
    def test_export_never_runs_and_csv_has_grid_units(self):
        bid=self.create()['id'];value=self.b.export(bid,OWNER,'csv');self.assertIn('SG_cells',value);self.assertEqual(self.h.created,0)
    def test_failed_commit_does_not_double_count_policy_calls(self):
        bid=self.create()['id'];self.b.step(bid,OWNER)
        with patch.object(self.h,'save_run',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.b.step(bid,OWNER)
        v=self.b.state(bid,OWNER);self.assertEqual(v['model_calls'],2);self.assertEqual(v['action_attempts'],2);self.assertEqual(v['status'],'failed')
    def test_cancel_does_not_masquerade_as_completion(self):
        bid=self.create()['id'];self.b.step(bid,OWNER);v=self.b.cancel(bid,OWNER)
        self.assertEqual(v['status'],'cancelled');self.assertEqual(v['metrics']['arms']['M0']['completed'],0)


class CliTests(unittest.TestCase):
    def test_lock_conflict_precedes_workbench_initialization(self):
        from project.src.platform_core import controller_cli_v2 as cli
        with patch.object(cli,'StoreLock',side_effect=ValueError('store owned')),patch('project.src.platform_core.service_v14.Workbench') as w:
            with self.assertRaisesRegex(ValueError,'store owned'):cli.main(['run','--store',str(ROOT),'--task-key','synthetic'])
            w.assert_not_called()


class HttpTests(unittest.TestCase):
    def setUp(self):
        from project.src.webapp import workbench_server_v24 as server
        class Parent:
            def allowed(self):return self.origin_ok
            def send(self,blob,mime='application/json',status=200,download=None):self.response=(blob,mime,status,download)
            def json(self,value,status=200,download=None):self.send(json.dumps(value).encode(),'application/json',status,download)
        w=SimpleNamespace(root=ROOT,sources={},_owner=lambda v:v,controllers=Mock(),controller_batches=Mock(),capturepoints=SimpleNamespace(sources={}),
            capturepoint_sessions=Mock(sources={}),ground_live=SimpleNamespace(sources={}),ground_walk=SimpleNamespace(sources={}))
        with patch.object(server.base,'make_server',return_value=SimpleNamespace(RequestHandlerClass=Parent)),patch.object(server.secrets,'token_urlsafe',return_value='test-token'):
            srv=server.make_server(w,0)
        self.handler=object.__new__(srv.RequestHandlerClass);self.handler.origin_ok=True;self.w=w
    def request(self,path,body=None,authorized=True):
        h=self.handler;h.path=path;h.headers={'X-GeoNav-Session':OWNER}
        if authorized:h.headers['X-GeoNav-Scene-Token']='test-token'
        if body is None:h.do_GET()
        else:
            blob=json.dumps(body).encode();h.headers.update({'Content-Type':'application/json','Content-Length':str(len(blob))});h.rfile=BytesIO(blob);h.do_POST()
        blob,mime,status,download=h.response
        return (json.loads(blob) if mime.startswith('application/json') else blob),status,mime,download
    def test_active_legacy_export_is_rejected(self):
        self.w.capturepoint_sessions.state_kind.return_value=dict(status='ready')
        _,status,_,_=self.request('/api/capturepoint-sessions/export?id='+'1'*32)
        self.assertEqual(status,400)
    def test_terminal_legacy_export_still_works(self):
        self.w.capturepoint_sessions.state_kind.return_value=dict(status='completed')
        _,status,_,download=self.request('/api/capturepoint-sessions/export?id='+'1'*32)
        self.assertEqual(status,200);self.assertTrue(download.endswith('.json'))
    def test_no_token_or_extra_truth_fields_rejected_before_execution(self):
        _,status,_,_=self.request('/api/controllers/create',{},False);self.assertEqual(status,403)
        _,status,_,_=self.request('/api/controllers/step',dict(id='1'*32,kind='aerial',action=None,goal=24));self.assertEqual(status,400)
        self.w.controllers.create.assert_not_called();self.w.controllers.step.assert_not_called()
    def test_preview_jpeg_content_type(self):
        self.w.controllers.preview.return_value=b'\xff\xd8synthetic'
        _,status,mime,_=self.request('/api/controllers/preview-image?dataset_id=masa-s2&task_key=t0&cell=target')
        self.assertEqual((status,mime),(200,'image/jpeg'))
    def test_release_drops_only_owner_batch_and_controller_memory(self):
        self.w.controllers.release.return_value=dict(released=True)
        _,status,_,_=self.request('/api/controllers/release',{})
        self.assertEqual(status,200);self.w.controller_batches.release.assert_called_once_with(OWNER);self.w.controllers.release.assert_called_once_with(OWNER)
    def test_batch_restart_flag_must_be_boolean(self):
        _,status,_,_=self.request('/api/controller-batches/step',dict(id='1'*32,restart_incomplete='yes'))
        self.assertEqual(status,400);self.w.controller_batches.step.assert_not_called()


if __name__=='__main__':unittest.main()
