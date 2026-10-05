from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import area_catalog_links_v2 as v
from data import area_catalog_v1 as c


class LinkTests(unittest.TestCase):
    def test_official_duplicate_slash(self):
        href=c.BASE+'mass_buildings/train/sat//22678915_15.tiff'
        self.assertEqual(v.normalize(href,'mass_buildings','train'),href.replace('/sat//','/sat/'))
    def test_normal_single_relative(self):
        self.assertEqual(v.normalize('22678915_15.tiff','mass_buildings','valid'),c.BASE+'mass_buildings/valid/sat/22678915_15.tiff')
    def test_origin_and_collection_remain_strict(self):
        for href in ['https://evil.example/22678915_15.tiff',c.BASE+'mass_roads/train/sat//22678915_15.tiff',
                     c.BASE+'mass_buildings/test/sat//22678915_15.tiff',c.BASE+'mass_buildings/train/sat/../sat//22678915_15.tiff',
                     c.BASE+'mass_buildings/train/sat//22678915_15.tiff?x=1']:
            with self.assertRaises(ValueError): v.normalize(href,'mass_buildings','train')
    def test_original_href_kept(self):
        p=v.Parser('mass_buildings','train'); href=c.BASE+'mass_buildings/train/sat//22678915_15.tiff'
        p.feed(f'<a href="{href}">x</a>')
        self.assertEqual(p.records['22678915_15.tiff']['original_href'],href)
        self.assertEqual(p.changed,1)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LinkTests))
    c.write(c.ROOT/'选题报告相关/十五乘十五目录链接适配单元测试_v2.json',
            dict(successful=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
    sys.exit(0 if result.wasSuccessful() else 1)
