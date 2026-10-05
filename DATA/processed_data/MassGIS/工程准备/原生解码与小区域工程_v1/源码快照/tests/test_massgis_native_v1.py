"""No network; synthetic fixtures are not real navigation evidence."""
from pathlib import Path
import tempfile
import shutil
import unittest
import json
import sys
from io import BytesIO
import numpy as np
from PIL import Image,PngImagePlugin
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_native_v1 as m
from env.massgis_area_v1 import MassGISAreaEnv,MassGISEpisode,PROTOCOL

class Pure(unittest.TestCase):
    def test_all_reset_endpoints_reachable(self):
        for start in range(225):
            goal=m.adjacent_goal(start)
            self.assertTrue(0<=goal<225)
            self.assertEqual(abs(start//15-goal//15)+abs(start%15-goal%15),1)
    def test_nir_not_alpha(self):
        a=np.array([[[90,80,70,0],[11,22,33,255]]],np.uint8)
        self.assertTrue(np.array_equal(m.rgb_first_three(a),np.array([[[90,80,70],[11,22,33]]],np.uint8)))
    def test_three_channels_rejected(self):
        with self.assertRaises(ValueError):m.rgb_first_three(np.zeros((2,2,3),np.uint8))
    def test_dark_pixel_not_blank(self):self.assertEqual(int(m.blank(np.array([[[0,0,1],[0,0,0],[255,255,255]]],np.uint8)).sum()),2)
    def test_local_defect_not_diluted(self):
        x=np.zeros((80,80),bool);x[:5,:5]=True
        self.assertLess(float(x.mean()),.01);self.assertGreater(m.blocks(x,[0,30,50],30)[0]['count']/900,.01)
    def test_source_intersection(self):self.assertEqual(m.intersection([0,0,4000,4000],[3500,3500,8000,8000]),[3500,3500,4000,4000])
    def test_north_up_slice(self):
        ys,xs=m.slice_pixels([0,3000,500,3500],[0,0,4000,4000]);self.assertEqual((ys.start,ys.stop,xs.start,xs.stop),(1000,2000,0,1000))
    def test_subpixel_crop_rejected(self):
        with self.assertRaises(ValueError):m.slice_pixels([.1,3000,500,3500],[0,0,4000,4000])

class Interface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=(m.ROOT/'选题报告相关').resolve()
        cls.root=Path(tempfile.mkdtemp(prefix='MassGIS工程测试临时_',dir=cls.base)).resolve()
        cls.folder=cls.root/'patches/dev/img_6000';cls.folder.mkdir(parents=True)
        manifest=dict(protocol=PROTOCOL,grid_size=15,budget=20,cell_size_m=300,native_cell_pixels=600,
            epsg=26986,pixel_size_m=.5,mosaic_pixels=9000,patch_format='png',engineering_only=True,
            regions=[dict(split='dev',area='img_6000',source_tile='synthetic/interface')])
        (cls.root/'数据清单.json').write_text(json.dumps(manifest),encoding='utf-8')
        for i in range(225):
            im=Image.new('RGB',(600,600),(100,i%256,70));info=PngImagePlugin.PngInfo();info.add_text('private','hidden source identity')
            im.save(cls.folder/f'patch_{i}.png',pnginfo=info)
    @classmethod
    def tearDownClass(cls):
        if cls.root.parent!=cls.base or not cls.root.name.startswith('MassGIS工程测试临时_'):raise ValueError('Unsafe synthetic cleanup target')
        shutil.rmtree(cls.root)
    def ep(self,start=0,goal=1):
        d=abs(start//15-goal//15)+abs(start%15-goal%15)
        return MassGISEpisode('synthetic_case','dev','img_6000',start,goal,d,20,15,PROTOCOL,'synthetic/interface')
    def test_public_native_image(self):
        obs=MassGISAreaEnv(self.root).reset(self.ep())
        with Image.open(BytesIO(obs.current_image)) as im:self.assertEqual(im.size,(600,600));self.assertEqual(im.info,{})
        self.assertFalse(hasattr(obs,'goal'));self.assertFalse(hasattr(obs,'source_tile'))
    def test_outside_spends_step(self):
        env=MassGISAreaEnv(self.root);env.reset(self.ep());obs,done,info=env.step('up')
        self.assertEqual(obs.position,(0,0));self.assertEqual(obs.remaining_budget,19);self.assertTrue(info.out_of_bounds)
    def test_final_step_success(self):
        env=MassGISAreaEnv(self.root);env.reset(self.ep())
        for _ in range(19):env.step('up')
        _,done,_=env.step('right');self.assertTrue(done);r=env.evaluator_result()
        self.assertTrue(r['success']);self.assertEqual(r['sg_m'],0);self.assertEqual(r['valid_travel_m'],300)
    def test_stop_rejected(self):
        env=MassGISAreaEnv(self.root);env.reset(self.ep())
        with self.assertRaises(ValueError):env.step('stop')
    def test_goal_unreachable(self):
        with self.assertRaises(ValueError):self.ep(0,224).validate()
    def test_missing_patch(self):
        p=self.folder/'patch_224.png';tmp=self.folder/'held.png';p.rename(tmp)
        try:
            with self.assertRaises(ValueError):MassGISAreaEnv(self.root).reset(self.ep())
        finally:tmp.rename(p)
    def test_wrong_native_size(self):
        p=self.folder/'patch_224.png';before=p.read_bytes();Image.new('RGB',(300,300)).save(p)
        try:
            with self.assertRaises(ValueError):MassGISAreaEnv(self.root).reset(self.ep())
        finally:p.write_bytes(before)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Pure,Interface)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    m.write(m.TESTS,dict(successful=result.wasSuccessful(),tests_run=result.testsRun,network_requests=0,
        real_navigation_actions=0,synthetic_interface_only=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
