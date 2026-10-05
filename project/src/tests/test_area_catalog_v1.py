"""Geography, provenance, resumed request budgets and exact packing checks."""
from pathlib import Path
from io import BytesIO
from PIL import Image, TiffImagePlugin
import itertools
import json
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import area_catalog_v1 as c


def tile(name,collection='mass_roads',split='train'):
    return dict(name=name,collection=collection,upstream_split=split,
                url=c.BASE+f'{collection}/{split}/sat/{name}')


def exclusions():
    # Each old coordinate supports the same lattice; all are far from fixture.
    old=[dict(id=f'{8000+i:04d}8000_15',x=(8000+i)*100,y=800000,
              bounds_m=[(8000+i)*100,798500,(8000+i)*100+1500,800000]) for i in range(151)]
    consumed=[dict(bounds_m=[900000+i*10000,900000,903000+i*10000,903000],
                   sources=[dict(id=f'used{i}_{j}') for j in range(4)]) for i in range(24)]
    return old,consumed


class CatalogTests(unittest.TestCase):
    def test_upstream_identity_and_duplicate_links(self):
        url=c.BASE+'mass_buildings/test/sat/index.html'
        p=c.Links(url,'mass_buildings','test')
        p.feed('<a href="10001000_15.tiff">x</a><a href="10001000_15.tiff">again</a>')
        self.assertEqual(len(p.records),1)
        self.assertEqual(p.records['10001000_15.tiff']['upstream_split'],'test')

    def test_external_and_other_split_links_rejected(self):
        for href in ['https://example.com/10001000_15.tiff','../mask/10001000_15.tiff',c.BASE+'mass_roads/test/sat/10001000_15.tiff']:
            p=c.Links(c.BASE+'mass_roads/train/sat/index.html','mass_roads','train')
            with self.assertRaises(ValueError): p.feed(f'<a href="{href}">x</a>')

    def test_coordinate_aliases_preserve_provenance(self):
        r=c.merge([tile('10001000_15.tiff','mass_buildings','test'),tile('10001000_15.tiff')])[(1000,1000)]
        self.assertEqual(r['collection'],'mass_roads')
        self.assertEqual(len(r['aliases']),2)
        self.assertFalse(r['pixels_equal_across_aliases_verified'])

    def test_complete_grid_and_history_exclusion(self):
        names=[f'{1000+15*x:04d}{1000-15*y:04d}_15.tiff' for y in range(3) for x in range(3)]
        tiles=c.merge([tile(n) for n in names]); old,consumed=exclusions()
        groups,_=c.propose(tiles,old,consumed,{})
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['provisional_bounds_m'],[100000,95500,104500,100000])
        consumed[0]['bounds_m']=[104500,95500,107500,100000]
        self.assertEqual(c.propose(tiles,old,consumed,{})[0],[])

    def test_cached_bad_quality_is_not_ignored(self):
        names=[f'{1000+15*x:04d}{1000-15*y:04d}_15.tiff' for y in range(3) for x in range(3)]
        old,consumed=exclusions()
        groups,counts=c.propose(c.merge([tile(n) for n in names]),old,consumed,{names[0][:-5]:dict(quality=dict(passed=False))})
        self.assertEqual(groups,[]); self.assertEqual(counts['excluded_cached_quality'],1)

    def test_incomplete_history_cannot_become_fresh(self):
        old,consumed=exclusions()
        with self.assertRaises(ValueError): c.propose({},old[:-1],consumed,{})

    def test_full_rectangle_gap_and_boundary(self):
        self.assertEqual(c.gap([0,0,4500,4500],[7500,0,12000,4500]),3000)
        self.assertEqual(c.gap([0,0,4500,4500],[4000,4000,8500,8500]),0)
        with self.assertRaises(ValueError): c.gap([0,0,float('nan'),1],[0,0,1,1])

    def test_packing_matches_exhaustive_subsets(self):
        rng=random.Random(1903)
        for n in range(1,10):
            regions=[]
            for _ in range(n):
                x,y=rng.randrange(12)*1500,rng.randrange(12)*1500
                regions.append(dict(provisional_bounds_m=[x,y,x+4500,y+4500]))
            expected=max(len(s) for k in range(n+1) for s in itertools.combinations(range(n),k)
                         if all(c.gap(regions[a]['provisional_bounds_m'],regions[b]['provisional_bounds_m'])>=2999.999
                                for a,b in itertools.combinations(s,2)))
            actual=c.pack(regions)
            self.assertTrue(actual['exact']); self.assertEqual(actual['count'],expected)
        self.assertEqual(c.pack([])['count'],0)

    def test_real_tags_without_pixel_decoding(self):
        buf=BytesIO(); tags=TiffImagePlugin.ImageFileDirectory_v2()
        tags[33550]=(1.,1.,0.); tags[33922]=(0.,0.,0.,100000.,100000.,0.)
        tags[34735]=(1,1,0,4,1024,0,1,1,1025,0,1,1,3072,0,1,26986,3076,0,1,9001)
        Image.new('RGB',(1500,1500)).save(buf,format='TIFF',tiffinfo=tags)
        prefix=buf.getvalue()[:65536]
        with patch.object(Image.Image,'load',side_effect=AssertionError('pixel decoding forbidden')):
            result=c.geotags(prefix)
        self.assertEqual(result['bounds_m'],[100000.,98500.,101500.,100000.])

    def test_incomplete_headers_do_not_pass_group(self):
        candidate=dict(group_names=[str(i) for i in range(9)])
        self.assertIsNone(c.actual_group(candidate,{},[]))
        heads={str(i):dict(status='passed',geo=dict(x=1500*(i%3),y=-1500*(i//3)),pixel_quality_cached_verified=False) for i in range(9)}
        result=c.actual_group(candidate,heads,[[100000,100000,101000,101000]])
        self.assertTrue(result['coordinates_TIFF_verified']); self.assertFalse(result['all_pixels_quality_verified'])
        heads['8']['geo']['x']+=1
        self.assertIsNone(c.actual_group(candidate,heads,[]))

    def test_interrupted_attempts_consume_budget(self):
        with tempfile.TemporaryDirectory(prefix='目录调查测试临时_',dir=c.ROOT/'选题报告相关') as folder:
            parent=Path(folder).resolve()
            self.assertEqual(parent.parent,(c.ROOT/'选题报告相关').resolve())
            (parent/'元数据').mkdir()
            url=c.BASE+'mass_roads/train/sat/10001000_15.tiff'
            with patch.object(c,'OUT',parent):
                c.log(dict(url=url,kind='header',attempt=1,status='started',received_bytes=0))
                c.log(dict(url=url,kind='header',attempt=2,status='started',received_bytes=0))
                with patch.object(c,'urlopen',side_effect=AssertionError('no extra request allowed')):
                    with self.assertRaises(RuntimeError): c.fetch(url,parent/'never.bin','header')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CatalogTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if c.TESTS.exists(): raise FileExistsError('do not overwrite prior test evidence')
    c.write(c.TESTS,dict(successful=result.wasSuccessful(),tests_run=result.testsRun,
                        failures=len(result.failures),errors=len(result.errors),source_sha256={c.rel(Path(__file__)):c.digest(Path(__file__))}))
    sys.exit(0 if result.wasSuccessful() else 1)
