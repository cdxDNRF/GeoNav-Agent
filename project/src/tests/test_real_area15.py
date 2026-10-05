from pathlib import Path
from io import BytesIO
from hashlib import sha256
import json
import tempfile
import unittest
from unittest.mock import patch
import sys
import numpy as np
from PIL import Image,TiffImagePlugin
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import real_area15 as d


def tiff_bytes(color=(30,80,120),x=100000,y=200000):
    tags=TiffImagePlugin.ImageFileDirectory_v2()
    tags[33550]=(1.,1.,0.); tags[33922]=(0.,0.,0.,float(x),float(y),0.)
    tags[34735]=(1,1,0,4,1024,0,1,1,1025,0,1,1,3072,0,1,26986,3076,0,1,9001)
    output=BytesIO(); Image.new('RGB',(1500,1500),color).save(output,format='TIFF',tiffinfo=tags)
    return output.getvalue()


class DataTests(unittest.TestCase):
    def folder(self):
        temp=tempfile.TemporaryDirectory(prefix='十五乘十五数据测试临时_',dir=d.ROOT/'选题报告相关')
        self.addCleanup(temp.cleanup); path=Path(temp.name).resolve()
        self.assertEqual(path.parent,(d.ROOT/'选题报告相关').resolve()); return path

    def test_source_blank_and_cell_gates(self):
        rgb=np.full((1500,1500,3),80,np.uint8); rgb[:30,:30]=0
        self.assertTrue(d.quality(rgb)['passed'])
        rgb[:60,:60]=0
        self.assertFalse(d.quality(rgb)['passed'])
        self.assertLess(d.quality(rgb)['source_blank_fraction'],.01)

    def test_native_geometry_requires_no_gap(self):
        rows=[dict(id=str(i),x=100000+(i%3)*1500,y=200000-(i//3)*1500) for i in range(9)]
        self.assertEqual(d.validate_group(rows),[100000,195500,104500,200000])
        rows[8]['x']+=1
        with self.assertRaises(ValueError): d.validate_group(rows)

    def test_actual_rectangle_buffer(self):
        self.assertEqual(d.bounds_gap([0,0,4500,4500],[7500,0,12000,4500]),3000)
        self.assertEqual(d.bounds_gap([0,0,4500,4500],[4500,0,9000,4500]),0)

    def test_crashed_reservation_and_attempts_survive_restart(self):
        folder=self.folder(); path=folder/'events.jsonl'
        limits=dict(d.LIMITS,body_bytes_per_file=10,body_bytes_total=20)
        j=d.Journal(path,limits); key,_=j.start('a','https://example.com/a'); j.event(key,'receiving',3)
        resumed=d.Journal(path,limits)
        self.assertEqual(resumed.charged(),10)
        resumed.start('a','https://example.com/a')
        with self.assertRaises(RuntimeError): resumed.start('a','https://example.com/a')
        with self.assertRaises(RuntimeError): resumed.start('b','https://example.com/b')

    def test_terminal_bytes_and_failed_body_accounted(self):
        path=self.folder()/'events.jsonl'; limits=dict(d.LIMITS,body_bytes_per_file=10,body_bytes_total=20)
        j=d.Journal(path,limits); a,_=j.start('a','url'); j.event(a,'failed',4)
        self.assertEqual(j.charged(),4)
        b,_=j.start('b','url'); j.event(b,'completed',6)
        self.assertEqual(d.Journal(path,limits).charged(),10)

    def test_sampler_same_distance_three_strata(self):
        region=dict(area='img_8000',split='dev',source_tile='fixture')
        episodes,strata=d.tasks([region]); self.assertEqual(len(episodes),75)
        from collections import Counter
        self.assertEqual(Counter(s['stratum'] for s in strata.values()),dict(long_distance=25,seam_target=25,interior_target=25))
        self.assertEqual(len({(e.start,e.goal) for e in episodes}),75)
        for e in episodes: self.assertEqual(d.l1(e.start,e.goal),e.dist); e.validate()

    def test_sampler_covers_both_seams_and_sides(self):
        _,strata=d.tasks([dict(area='img_8000',split='dev',source_tile='fixture')])
        cases={(s['seam_orientation'],s['seam_boundary'],s['seam_goal_side']) for s in strata.values() if s['stratum']=='seam_target'}
        self.assertEqual(len(cases),8)

    def test_wrong_targets_preserve_distance(self):
        episodes,_=d.tasks([dict(area='img_8000',split='dev',source_tile='fixture')]); wrong=d.wrong_plan(episodes)
        for e in episodes:
            g=wrong[e.episode_id]['cue_cell']; self.assertNotIn(g,(e.start,e.goal)); self.assertEqual(d.l1(e.start,g),e.dist)

    def test_probes_have_correct_source_adjacency(self):
        rows=d.probes([dict(area='img_8000',split='dev')]); self.assertEqual(len(rows),360)
        index=lambda cell: (cell//15//5)*3+(cell%15//5)
        for p in rows:
            self.assertEqual(d.l1(p['current_cell'],p['target_cell']),p['distance'])
            if p['kind']=='cross_source_adjacent': self.assertNotEqual(index(p['current_cell']),index(p['target_cell']))
            if p['kind']=='same_source_adjacent': self.assertEqual(index(p['current_cell']),index(p['target_cell']))

    def test_mosaic_preserves_nine_native_source_blocks(self):
        folder=self.folder(); rows=[]
        for i in range(9):
            p=folder/f'{i}.tiff'; p.write_bytes(tiff_bytes((i+10,30,80),100000+(i%3)*1500,200000-(i//3)*1500))
            rows.append(dict(id=str(i),x=100000+(i%3)*1500,y=200000-(i//3)*1500,raw_path=d.rel(p),raw_sha256=d.digest(p)))
        im=d.mosaic_image(dict(sources=rows)); self.assertEqual(im.size,(4500,4500))
        for i in range(9): self.assertEqual(im.getpixel(((i%3)*1500+500,(i//3)*1500+500)),(i+10,30,80))

    def test_frozen_prefix_binding_before_quality(self):
        p=self.folder()/'source.tiff'; p.write_bytes(tiff_bytes())
        record=dict(name='10001000_15.tiff',url='fixture',upstream_split='train',prefix_bytes=65536,
                    prefix_sha256='incorrect',expected_geo=dict(bounds_m=[100000,198500,101500,200000]))
        with self.assertRaises(ValueError): d.decode_source(p,record)
        record['prefix_sha256']=sha256(p.read_bytes()[:65536]).hexdigest()
        row=d.decode_source(p,record); self.assertTrue(row['quality']['passed'])

    def test_saved_file_requires_registered_attempt(self):
        folder=self.folder(); (folder/'tiff').mkdir(); (folder/'tiff'/'10001000_15.tiff').write_bytes(b'unregistered')
        with patch.object(d,'RAW',folder):
            with self.assertRaises(ValueError): d.download(dict(name='10001000_15.tiff',url='fixture'),d.Journal(folder/'journal.jsonl'))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DataTests))
    d.write(d.TESTS,dict(successful=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                       source_sha256={d.rel(Path(__file__)):d.digest(Path(__file__))}))
    sys.exit(0 if result.wasSuccessful() else 1)
