"""Synthetic native environment and sampler tests, plus sealed18 regressions."""
from pathlib import Path
from io import BytesIO
from dataclasses import replace,fields
import inspect
import json
import shutil
import tempfile
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,PngImagePlugin
from env.massgis_native_area_v2 import native_spec,NativeEpisode,NativeAreaEnv
from eval import massgis_navigation_compat_v1 as m
from agents.massgis_navigator_v1 import NativeNavigator
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_area_compatibility import AreaCompatibilityTests

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=(m.ROOT/'选题报告相关').resolve();cls.temp=Path(tempfile.mkdtemp(prefix='MassGIS导航测试临时_',dir=cls.parent)).resolve()
        for k in (5,10,15):
            s=native_spec(k);r=cls.temp/str(k);folder=r/'patches/dev/img_6100';folder.mkdir(parents=True)
            (r/'数据清单.json').write_text(json.dumps(dict(protocol=s.name,grid_size=k,budget=s.budget,cell_size_m=300,native_cell_pixels=600,
             epsg=26986,pixel_size_m=.5,mosaic_pixels=k*600,patch_format='png',development_only=True,
             regions=[dict(split='dev',area='img_6100',source_tile='synthetic')])),encoding='utf-8')
            for j in range(k*k):
                info=PngImagePlugin.PngInfo();info.add_text('private','hidden evaluator coordinates')
                Image.new('RGB',(600,600),(j%256,40,80)).save(folder/f'patch_{j}.png',pnginfo=info)
    @classmethod
    def tearDownClass(cls):
        if cls.temp.parent!=cls.parent or not cls.temp.name.startswith('MassGIS导航测试临时_'):raise ValueError('Unsafe cleanup')
        shutil.rmtree(cls.temp)
    def ep(self,k=10):
        s=native_spec(k);return NativeEpisode('synthetic','dev','img_6100',0,1,1,s.budget,k,s.name,'synthetic')
    def test_sizes_and_physical_scale(self):
        for k in (5,10,15):self.assertEqual(native_spec(k).area_km2,(k*.3)**2);self.assertEqual(native_spec(k).native_cell_pixels,600)
    def test_unknown_grid_rejected(self):
        for k in (True,7,20):
            with self.assertRaises(ValueError):native_spec(k)
    def test_test_split_rejected(self):
        with self.assertRaises(ValueError):replace(self.ep(),split='test').validate()
    def test_old_protocol_not_aliased(self):
        with self.assertRaises(ValueError):replace(self.ep(),protocol='masa-roads-grid10-spatial-v1').validate()
    def test_budget_and_distance_truth(self):
        for changes in (dict(budget=21),dict(dist=2),dict(goal=0)):
            with self.assertRaises(ValueError):replace(self.ep(),**changes).validate()
    def test_public_native_and_no_truth(self):
        for k in (5,10,15):
            obs=NativeAreaEnv(self.temp/str(k)).reset(self.ep(k))
            with Image.open(BytesIO(obs.current_image)) as im:self.assertEqual(im.size,(600,600));self.assertEqual(im.info,{})
            self.assertFalse({f.name for f in fields(obs)}&{'goal','dist','source_tile','area'})
    def test_last_step_and_oob(self):
        for k in (5,10,15):
            env=NativeAreaEnv(self.temp/str(k));env.reset(self.ep(k))
            for _ in range(native_spec(k).budget-1):self.assertTrue(env.step('up')[2].out_of_bounds)
            self.assertTrue(env.step('right')[1]);r=env.evaluator_result();self.assertTrue(r['success']);self.assertEqual((r['sg_m'],r['valid_travel_m']),(0,300))
    def test_no_stop(self):
        env=NativeAreaEnv(self.temp/'10');env.reset(self.ep())
        with self.assertRaises(ValueError):env.step('stop')
    def test_reset_clears_public_memory(self):
        env=NativeAreaEnv(self.temp/'10');env.reset(self.ep());env.step('down');obs=env.reset(self.ep())
        self.assertEqual((obs.position,obs.visited,obs.remaining_budget),((0,0),(0,),20))
    def test_source_binding(self):
        with self.assertRaises(ValueError):NativeAreaEnv(self.temp/'10').reset(replace(self.ep(),source_tile='wrong'))
    def test_terminal_truth_only(self):
        env=NativeAreaEnv(self.temp/'10');env.reset(self.ep())
        with self.assertRaises(RuntimeError):env.evaluator_result()
    def test_mapper_keeps_native_scale(self):
        self.assertEqual(m.mapping(5)[:5],[10,11,12,13,14]);self.assertEqual(m.mapping(10)[-1],149);self.assertEqual(m.mapping(15),list(range(225)))
    def test_tasks_distance_seam_and_unique(self):
        cells=m.read(m.PREVIOUS/'元数据/区域像素与逐格来源.json')['quality']['cells']
        for k in (5,10):
            tasks=m.seeded_tasks(k,[cells[j] for j in m.mapping(k)])
            self.assertEqual(len(tasks),25 if k==5 else 75);self.assertEqual(len({(t['start'],t['goal']) for t in tasks}),len(tasks))
            for stratum in {t['stratum'] for t in tasks}:self.assertEqual(sum(t['target_mixed_source'] for t in tasks if t['stratum']==stratum),8)
            for t in tasks:self.assertEqual(t['dist'],abs(t['start']//k-t['goal']//k)+abs(t['start']%k-t['goal']%k))
    def test_wrong_target_excludes_endpoint(self):
        tasks=[dict(episode_id='corner',grid_size=5,start=0,goal=24,dist=8)]
        w=m.wrong_plan(tasks)['corner'];self.assertNotIn(w['cue_cell'],(0,24));self.assertFalse(w['matched_distance'])
    def test_probe_completeness(self):
        pairs=sum((m.probes(k) for k in (5,10,15)),[]);self.assertEqual(len(pairs),5012)
        self.assertEqual(sum(t['kind']=='adjacent' for t in pairs),1280);self.assertEqual(sum(t['kind']=='distance2' for t in pairs),2332)
    def test_policy_signature_is_two_images_only(self):
        self.assertFalse(set(inspect.signature(NativeNavigator.act).parameters)&{'goal','source','dist','episode','bank','store'})

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (NativeTests,AreaCompatibilityTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    m.write(m.TESTS,dict(successful=result.wasSuccessful(),tests_run=result.testsRun,new_native_tests=16,old_protocol_tests=18,network_requests=0,real_model_forwards=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)
