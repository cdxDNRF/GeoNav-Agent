"""Synthetic formal references and public permission/termination checks."""
from pathlib import Path
from dataclasses import replace, fields
from io import BytesIO
from copy import deepcopy
import hashlib
import inspect
import json
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, PngImagePlugin
from env.massgis_confirm_area_v1 import confirmation_spec, ConfirmEpisode, ConfirmAreaEnv
from env.massgis_native_area_v2 import NativeEpisode
from eval import massgis_confirmation_protocol_v1 as p
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_area_compatibility import AreaCompatibilityTests


class ConfirmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = (p.ROOT/'选题报告相关').resolve()
        cls.temp = Path(tempfile.mkdtemp(prefix='MassGIS正式协议测试临时_', dir=cls.parent)).resolve()
        for k in (10, 15):
            directory = cls.temp/str(k); directory.mkdir(); folder = directory/'images'; folder.mkdir()
            cells = []
            for j in range(k*k):
                f = folder/f'patch_{j}.png'; meta = PngImagePlugin.PngInfo(); meta.add_text('private', 'evaluator-only')
                Image.new('RGB', (600, 600), (j % 256, 30, 90)).save(f, pnginfo=meta)
                cells.append(dict(cell=j, path=f.relative_to(p.ROOT).as_posix(),
                                  file_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
            s = confirmation_spec(k)
            manifest = dict(protocol=s.name, grid_size=k, budget=20, cell_size_m=300,
                native_cell_pixels=600, epsg=26986, pixel_size_m=.5, mosaic_pixels=k*600,
                patch_format='png', confirmation_only=True,
                regions=[dict(split='test', area='img_9900', source_tile='synthetic', cells=cells)])
            (directory/'数据清单.json').write_text(json.dumps(manifest), encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        if cls.temp.parent != cls.parent or not cls.temp.name.startswith('MassGIS正式协议测试临时_'):
            raise ValueError('Unsafe cleanup')
        shutil.rmtree(cls.temp)

    def ep(self, k=15):
        return ConfirmEpisode('synthetic', 'test', 'img_9900', 0, 1, 1, 20, k,
                              confirmation_spec(k).name, 'synthetic')

    def env(self, k=15):
        return ConfirmAreaEnv(self.temp/str(k), p.ROOT)

    def test_protocol_identity_and_scale(self):
        self.assertEqual(confirmation_spec(10).area_km2, 9)
        self.assertEqual(confirmation_spec(15).area_km2, 20.25)
        for k in (5, True, 10.0, 20):
            with self.assertRaises(ValueError): confirmation_spec(k)

    def test_reachable_only_and_strict_types(self):
        for changes in (dict(goal=224, dist=28), dict(budget=21), dict(dist=True), dict(goal=0), dict(split='dev')):
            with self.assertRaises(ValueError): replace(self.ep(), **changes).validate()

    def test_old_development_episode_not_relabelled(self):
        old = NativeEpisode('dev', 'dev', 'img_9900', 0, 1, 1, 20, 15,
                            'massgis-native-grid15-dev-v2', 'synthetic')
        with self.assertRaises(ValueError): self.env().reset(old)

    def test_public_images_strip_metadata_and_truth(self):
        for k in (10, 15):
            obs = self.env(k).reset(self.ep(k))
            with Image.open(BytesIO(obs.current_image)) as im:
                self.assertEqual(im.size, (600, 600)); self.assertEqual(im.info, {})
            self.assertFalse({f.name for f in fields(obs)} & {'goal','dist','source_tile','area','pair_id','stratum'})

    def test_last_action_success_oob_movement(self):
        for k in (10, 15):
            env = self.env(k); env.reset(self.ep(k))
            for _ in range(19): self.assertTrue(env.step('up')[2].out_of_bounds)
            self.assertTrue(env.step('right')[1]); r = env.evaluator_result()
            self.assertEqual((r['success'],r['out_of_bounds'],r['steps'],r['sg_m'],r['valid_travel_m']), (True,19,20,0,300))

    def test_terminal_and_reset(self):
        env = self.env(); env.reset(self.ep())
        with self.assertRaises(RuntimeError): env.evaluator_result()
        env.step('right')
        with self.assertRaises(RuntimeError): env.step('left')
        obs = env.reset(self.ep()); self.assertEqual((obs.visited,obs.remaining_budget), ((0,),20))
        with self.assertRaises(ValueError): env.step('stop')

    def test_source_identity(self):
        with self.assertRaises(ValueError): self.env().reset(replace(self.ep(), source_tile='different'))

    def test_content_binding_and_path_escape(self):
        file = self.temp/'15/数据清单.json'; old = file.read_bytes(); original = json.loads(old)
        try:
            for field, value in [('path','../outside.png'), ('file_sha256','0'*64)]:
                changed = deepcopy(original); changed['regions'][0]['cells'][0][field] = value
                file.write_text(json.dumps(changed), encoding='utf-8')
                with self.assertRaises(ValueError): self.env().reset(self.ep())
        finally: file.write_bytes(old)

    def test_manifest_geometry_and_duplicate_reference(self):
        file = self.temp/'15/数据清单.json'; old = file.read_bytes(); original = json.loads(old)
        try:
            changed = deepcopy(original); changed['native_cell_pixels'] = 300
            file.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaises(ValueError): self.env()
            changed = deepcopy(original); changed['regions'][0]['cells'][1]['path'] = changed['regions'][0]['cells'][0]['path']
            file.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaises(ValueError): self.env()
        finally: file.write_bytes(old)

    def test_shared_physical_routes_and_far_quota(self):
        cells = [dict(crosses_source_seam=(j//15==1 or j%15==13)) for j in range(225)]
        a,b = p.task_queue('img_9900','synthetic',cells,0)
        self.assertEqual((len(a),len(b)),(75,100)); ids=p.mapping(10)
        for x,y in zip(a,b):
            self.assertEqual((y['start'],y['goal'],y['pair_id']), (ids[x['start']],ids[x['goal']],x['pair_id']))
        self.assertEqual(len({(x['start'],x['goal']) for x in b}),100)
        for group in ('short','middle','long','far'):
            self.assertEqual(sum(t['stratum']==group for t in b),25)
            self.assertEqual(sum(t['target_mixed_source'] for t in b if t['stratum']==group),8)
        self.assertTrue(all(17<=t['dist']<=20 for t in b[75:]))

    def test_wrong_target_and_probe_complete(self):
        t=dict(episode_id='corner',start=0,goal=99,dist=18,grid_size=10)
        w=p.wrong_targets([t])['corner'];self.assertNotIn(w['cue_cell'],(0,99));self.assertFalse(w['matched_distance'])
        self.assertEqual(len(p.probe_pairs(10,0)),1404);self.assertEqual(len(p.probe_pairs(15,0)),3304)

    def test_wrapper_public_signature_and_frozen_equations(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from agents.massgis_confirm_navigator_v1 import ConfirmNavigator
        from agents.massgis_navigator_v1 import NativeNavigator
        from env.environment import Observation
        self.assertFalse(set(inspect.signature(ConfirmNavigator.act).parameters) & {'goal','bank','episode','source','dist'})
        class Explorer(torch.nn.Module):
            def step(self,x,h):return torch.zeros((1,4)),None,None,h
        class Head(torch.nn.Module):
            def forward(self,x):return torch.tensor([[0.,0.,0.,0.,5.]])
        z=np.zeros(512,np.float32);l=np.zeros((4,768),np.float32);profile=np.zeros((4,3,64,3),np.float32)
        for k in (10,15):
            for policy in ('M0','Coverage3Radial'):
                for condition in ('CueFull','Baseline','CueMean','CueWrong'):
                    legacy=SimpleNamespace(explorer=Explorer(),head=Head(),device='cpu',em=z,hm=z,lm=l,pm=profile)
                    new=ConfirmNavigator(legacy,k,policy,condition);old=NativeNavigator(legacy,k,policy,condition)
                    obs=Observation(b'current',b'target',(k-1,k-1),k,20,(k*k-1,))
                    args=(obs,z,l,z,l,profile,profile)
                    self.assertEqual(new.act_with_profiles(*args),old.act_with_profiles(*args))


if __name__ == '__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (ConfirmTests,AreaCompatibilityTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    p.write(p.TEST,dict(successful=result.wasSuccessful(),tests_run=result.testsRun,new_tests=12,
                        old_protocol_tests=18,formal_model_forwards=0,formal_navigation=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)
