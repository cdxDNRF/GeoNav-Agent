"""Geometry corruption, georeference rejection, protocol and terminal boundaries."""
from dataclasses import fields
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from PIL import Image, TiffImagePlugin
import tifffile
from data.continuous_area import geoinfo, validate_group, blank_fraction, tasks
from env.actual_area import AreaEpisode, ActualAreaGridEnv, PROTOCOL
from env.scaled_grid import ScaledEpisode

ROOT=Path(__file__).resolve().parents[3]


class ActualAreaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent=(ROOT/'选题报告相关').resolve()
        cls.tmp=tempfile.TemporaryDirectory(prefix='实际区域测试临时_',dir=parent)
        cls.path=Path(cls.tmp.name).resolve()
        def cleanup():
            resolved=cls.path.resolve()
            if resolved.parent!=parent or not resolved.name.startswith('实际区域测试临时_'):
                raise ValueError('unsafe test cleanup target')
            cls.tmp.cleanup()
        cls.addClassCleanup(cleanup)
        cls.grid=cls.path/'grid';folder=cls.grid/'patches/dev/img_1000';folder.mkdir(parents=True)
        Image.new('RGB',(300,300),(110,120,130)).save(cls.path/'fixture.jpg',quality=75)
        data=(cls.path/'fixture.jpg').read_bytes()
        for cell in range(100):(folder/f'patch_{cell}.jpg').write_bytes(data)
        manifest=dict(protocol=PROTOCOL,pixel_size_m=1,cell_size_m=300,native_cell_pixels=300,mosaic_pixels=3000,grid_size=10,
                      regions=[dict(area='img_1000',source_tile='fixture/area')])
        (cls.grid/'数据清单.json').write_text(json.dumps(manifest),encoding='utf-8')

    def write_geo(self,name,scale=1,epsg=26986,unit=9001,raster=1,missing=False,rotation=False):
        info=TiffImagePlugin.ImageFileDirectory_v2()
        if not missing:
            info[33550]=(float(scale),float(scale),0.);info.tagtype[33550]=12
            info[33922]=(0.,0.,0.,200000.,900000.,0.);info.tagtype[33922]=12
            info[34735]=(1,1,0,4,1024,0,1,1,1025,0,1,raster,3072,0,1,epsg,3076,0,1,unit);info.tagtype[34735]=3
        if rotation:
            info[34264]=tuple(float(i==j) for i in range(4) for j in range(4));info.tagtype[34264]=12
        p=self.path/name;Image.new('RGB',(1500,1500),(90,100,110)).save(p,compression='tiff_adobe_deflate',tiffinfo=info);return p

    def test_geotiff_roundtrip_and_independent_decoder(self):
        p=self.write_geo('valid.tiff');g=geoinfo(p)
        self.assertEqual(g['bounds_m'],[200000.,898500.,201500.,900000.])
        a=tifffile.imread(p);self.assertEqual(a.shape,(1500,1500,3));self.assertTrue(np.all(a==[90,100,110]))

    def test_missing_or_rotated_geotiff_rejected(self):
        for p in [self.write_geo('missing.tiff',missing=True),self.write_geo('rotated.tiff',rotation=True)]:
            with self.assertRaises(ValueError):geoinfo(p)

    def test_wrong_crs_unit_pixel_type_or_scale_rejected(self):
        for i,kwargs in enumerate([dict(epsg=4326),dict(unit=9002),dict(raster=2),dict(scale=.5)]):
            p=self.write_geo(f'wrong{i}.tiff',**kwargs)
            with self.assertRaises(ValueError):geoinfo(p)

    def group(self):
        return [dict(id=str(i),split='train',col=i%2,north_row=1-i//2,x=200000.+1500*(i%2),y=900000.-1500*(i//2)) for i in range(4)]

    def test_contiguous_quadrants_and_submillimetre_tolerance(self):
        g=self.group();g[3]['x']+=.0002;self.assertLess(max(abs(x) for p in validate_group(g) for x in p),.001)

    def test_gap_wrong_order_reused_or_test_tile_rejected(self):
        for mode in ('gap','order','duplicate','test'):
            g=self.group()
            if mode=='gap':g[3]['x']+=1
            elif mode=='order':g[1],g[2]=g[2],g[1]
            elif mode=='duplicate':g[3]['id']=g[0]['id']
            else:g[3]['split']='test'
            with self.assertRaises(ValueError):validate_group(g)

    def test_white_and_black_blank_cells(self):
        self.assertEqual(blank_fraction(np.full((10,10,3),255,np.uint8)),1)
        self.assertEqual(blank_fraction(np.zeros((10,10,3),np.uint8)),1)
        self.assertEqual(blank_fraction(np.full((10,10,3),110,np.uint8)),0)

    def test_protocol_and_manifest_mix_rejected(self):
        env=ActualAreaGridEnv(self.grid)
        old=ScaledEpisode('old','dev','img_1000',0,99,18,source_tile='fixture/area')
        with self.assertRaises(ValueError):env.reset(old)
        wrong=AreaEpisode('new','dev','img_1000',0,99,18,source_tile='wrong/area')
        with self.assertRaises(ValueError):env.reset(wrong)
        with self.assertRaises(ValueError):AreaEpisode('new','dev','img_1000',0,99,18,protocol=old.protocol,source_tile='fixture/area').validate()

    def test_last_budget_arrival_and_outside_costs(self):
        env=ActualAreaGridEnv(self.grid);ep=AreaEpisode('last','dev','img_1000',0,99,18,source_tile='fixture/area')
        obs=env.reset(ep)
        for i,action in enumerate(['up','up']+['right']*9+['down']*9):
            obs,done,info=env.step(action);self.assertEqual(obs.remaining_budget,19-i)
            self.assertEqual(info.out_of_bounds,i<2)
            self.assertEqual(done,i==19)
        r=env.evaluator_result();self.assertTrue(r['success']);self.assertEqual(r['steps'],20)
        self.assertEqual(r['sg_m'],0);self.assertEqual(r['valid_travel_m'],5400)

    def test_budget_failure_distance_and_no_postterminal_action(self):
        env=ActualAreaGridEnv(self.grid);env.reset(AreaEpisode('failure','dev','img_1000',0,99,18,source_tile='fixture/area'))
        for _ in range(20):env.step('up')
        r=env.evaluator_result();self.assertFalse(r['success']);self.assertEqual(r['sg_m'],5400);self.assertEqual(r['valid_travel_m'],0)
        with self.assertRaises(RuntimeError):env.step('right')

    def test_public_observation_has_no_map_truth(self):
        env=ActualAreaGridEnv(self.grid);obs=env.reset(AreaEpisode('public','dev','img_1000',0,99,18,source_tile='fixture/area'))
        names={f.name for f in fields(obs)}
        self.assertFalse(names & {'goal','distance','source_tile','area','epsg','global_map','coordinates'})
        from io import BytesIO
        with Image.open(BytesIO(obs.current_image)) as im:self.assertFalse(im.getexif());self.assertEqual(im.size,(300,300))

    def test_fixed_long_distance_tasks_cross_original_tiles(self):
        rs=[dict(area=f'img_{1000+i}',source_tile=f'MasaArea/region_{i:02d}') for i in range(3)]
        es=tasks(rs);self.assertEqual(len(es),75);self.assertEqual(len({e.episode_id for e in es}),75)
        for e in es:
            quadrant=lambda cell:(cell//10//5)*2+cell%10//5
            self.assertNotEqual(quadrant(e.start),quadrant(e.goal));self.assertEqual(e.protocol,PROTOCOL)
        self.assertEqual(es,tasks(rs))


if __name__=='__main__':unittest.main()
