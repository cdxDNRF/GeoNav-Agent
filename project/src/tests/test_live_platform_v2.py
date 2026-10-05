"""Synthetic live session checks; no historical evaluator is re-run."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

try:
    from project.src.webapp.live_v2 import LiveManager
except ImportError:
    LiveManager = None

ROOT = Path(__file__).resolve().parents[3]
TEMP = ROOT/'平台/实时导航平台_v2/核验'


class FakeRuntime:
    def __init__(self, root, task, policy, seed, cloud=None):
        self.task = task; self.cell = task['start']; self.visited = [self.cell]
        self.remaining = task['budget']; self.done = False; self.calls = 0
        self.obs = self.observe()

    def observe(self):
        return SimpleNamespace(position=divmod(self.cell, 5), grid_size=5,
            remaining_budget=self.remaining, visited=tuple(self.visited),
            current_image=b'current', target_image=b'target', legal_actions=('right','down'))

    def decide(self):
        self.calls += 1
        return dict(action='right', explorer_action='right', explorer_logits=[0,1,0,0],
            probabilities=[0,1,0,0,0], reason='accepted')

    def advance(self, action):
        self.cell += 1; self.visited.append(self.cell); self.remaining -= 1
        self.done = self.cell == self.task['goal'] or self.remaining == 0
        self.obs = self.observe()

    def result(self):
        return dict(success=self.cell==self.task['goal'], sg=abs(self.cell-self.task['goal']),
            sg_m=300*abs(self.cell-self.task['goal']), steps=len(self.visited)-1,
            valid_travel_m=300*(len(self.visited)-1), termination='goal_reached')

    def image(self, cell):
        return b'image'+str(cell).encode()


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(LiveManager, '实时模块尚未实现')
        TEMP.mkdir(parents=True, exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(prefix='测试临时_', dir=TEMP)
        self.addCleanup(self.folder.cleanup); self.root=Path(self.folder.name)
        self.task=dict(key='g5:0', grid_size=5,budget=2,start=0,goal=2,
            episode_id='toy',area='img_6100',stratum='hidden',dist=2)
        self.manager=LiveManager(self.root,tasks=[self.task],factory=FakeRuntime,bindings={})

    def test_public_state_and_image_permissions(self):
        state=self.manager.create('g5:0','M0',0); sid=state['id']
        self.assertNotIn('goal',state);self.assertNotIn('task',state)
        self.assertEqual(self.manager.image(sid,'target'),b'target')
        with self.assertRaises(ValueError):self.manager.image(sid,'2')
        with self.assertRaises(ValueError):self.manager.export(sid)
        self.assertEqual(state['encoding_mode'],'sealed_cache_live_policy')

    def test_actual_decision_then_final_budget_success(self):
        state=self.manager.create('g5:0','M0',0);sid=state['id']
        a=self.manager.step(sid);self.assertEqual(a['steps'],1)
        b=self.manager.step(sid);self.assertEqual(b['status'],'completed')
        self.assertTrue(b['result']['success']);self.assertEqual(b['result']['sg'],0)
        with self.assertRaises(ValueError):self.manager.step(sid)
        exported=self.manager.export(sid);self.assertEqual(len(exported['decisions']),2)
        self.assertTrue(Path(self.root/exported['output_path']).is_file())

    def test_cancel_is_not_a_navigation_action(self):
        sid=self.manager.create('g5:0','M0',0)['id']
        state=self.manager.cancel(sid);self.assertEqual(state['status'],'cancelled')
        self.assertIsNone(state['result']);self.assertEqual(state['steps'],0)
        with self.assertRaises(ValueError):self.manager.step(sid)
        self.assertEqual(self.manager.export(sid)['decisions'],[])

    def test_sessions_and_concurrent_steps_are_isolated(self):
        a=self.manager.create('g5:0','M0',0)['id'];b=self.manager.create('g5:0','M0',0)['id']
        with ThreadPoolExecutor(2) as pool:list(pool.map(lambda _:self.manager.step(a),range(2)))
        self.assertEqual(self.manager.state(a)['steps'],2)
        self.assertEqual(self.manager.state(b)['steps'],0)

    def test_only_registered_tasks_and_policies(self):
        with self.assertRaises(ValueError):self.manager.create('../evil','M0',0)
        with self.assertRaises(ValueError):self.manager.create('g5:0','Coverage3Radial',0)
        with self.assertRaises(ValueError):self.manager.create('g5:0','M0',9)

    def test_http_mutation_requires_same_origin_token(self):
        from project.src.webapp.server_v2 import make_server
        server=make_server(self.manager,port=0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        url=f'http://127.0.0.1:{server.server_port}'
        with urlopen(url+'/api/meta') as response:meta=json.load(response)
        body=json.dumps({'task':'g5:0','policy':'M0','seed':0}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(url+'/api/create',data=body,headers={'Content-Type':'application/json'}))
        self.assertEqual(caught.exception.code,403)
        headers={'Content-Type':'application/json','X-GeoNav-Token':meta['token'],'Origin':url}
        with urlopen(Request(url+'/api/create',data=body,headers=headers)) as response:state=json.load(response)
        self.assertEqual(state['status'],'ready')
        headers['Origin']='https://other.example'
        with self.assertRaises(HTTPError) as caught:urlopen(Request(url+'/api/cancel',data=json.dumps({'id':state['id']}).encode(),headers=headers))
        self.assertEqual(caught.exception.code,403)


if __name__=='__main__':unittest.main()
