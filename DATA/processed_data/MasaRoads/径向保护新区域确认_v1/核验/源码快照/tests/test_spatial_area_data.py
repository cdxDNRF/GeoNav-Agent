"""Full-footprint isolation, seam strata/probes, public protocol and boundaries."""
from collections import Counter
from dataclasses import fields
from pathlib import Path
from io import BytesIO
import json
import tempfile
import unittest
from PIL import Image
from data.spatial_continuous import (CatalogParser, rectangle_gap, clear_of, tasks, probes, wrong_plan, l1)
from env.spatial_area import SpatialAreaEpisode, SpatialAreaGridEnv, PROTOCOL
from env.scaled_grid import ScaledEpisode
from env.actual_area import AreaEpisode

ROOT = Path(__file__).resolve().parents[3]


class SpatialDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regions = [dict(area=f'img_{2000+i}', split='dev' if i < 4 else 'test',
                           source_tile=f'MasaRoadsArea/{i}') for i in range(14)]
        cls.episodes, cls.strata = tasks(cls.regions)
        parent = (ROOT / '选题报告相关').resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='空间隔离区域测试临时_', dir=parent)
        cls.path = Path(cls.temp.name).resolve()
        def cleanup():
            target = cls.path.resolve()
            if target.parent != parent or not target.name.startswith('空间隔离区域测试临时_'):
                raise ValueError('unsafe synthetic-fixture cleanup')
            cls.temp.cleanup()
        cls.addClassCleanup(cleanup)
        cls.grid = cls.path / 'grid'
        im = Image.new('RGB', (300, 300), (88, 109, 122))
        buffer = BytesIO()
        im.save(buffer, format='JPEG', quality=75)
        for split, area in (('dev', 'img_2000'), ('test', 'img_2004')):
            folder = cls.grid / 'patches' / split / area
            folder.mkdir(parents=True)
            for cell in range(100):
                (folder / f'patch_{cell}.jpg').write_bytes(buffer.getvalue())
        manifest = dict(protocol=PROTOCOL, epsg=26986, pixel_size_m=1, cell_size_m=300, native_cell_pixels=300,
                        mosaic_pixels=3000, grid_size=10, regions=[cls.regions[0], cls.regions[4]])
        (cls.grid / '数据清单.json').write_text(json.dumps(manifest), encoding='utf-8')

    def test_whole_footprint_gap_not_center_distance(self):
        a = [0, 0, 3000, 3000]
        self.assertEqual(rectangle_gap(a, [3100, 0, 4600, 1500]), 100)
        self.assertFalse(clear_of(a, [[3100, 0, 4600, 1500]]))
        self.assertFalse(clear_of(a, [[3000, 0, 4500, 1500]]))
        self.assertTrue(clear_of(a, [[6000, 0, 7500, 1500]]))

    def test_diagonal_and_overlap_gap(self):
        a = [0, 0, 3000, 3000]
        self.assertAlmostEqual(rectangle_gap(a, [6000, 6000, 9000, 9000]), 3000*2**.5)
        self.assertEqual(rectangle_gap(a, [1000, 1000, 4000, 4000]), 0)
        self.assertTrue(clear_of(a, []))

    def test_official_catalog_url_and_identity(self):
        p = CatalogParser()
        p.feed('<a href="http://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/10078660_15.tiff">input</a>')
        self.assertEqual(p.urls['10078660_15.tiff'], 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/10078660_15.tiff')
        for href in ('https://example.com/10078660_15.tiff', 'https://www.cs.toronto.edu/other/10078660_15.tiff'):
            with self.assertRaises(ValueError):
                CatalogParser().feed('<a href="'+href+'">x</a>')

    def test_1050_unique_routes_and_three_prespecified_strata(self):
        self.assertEqual(len(self.episodes), 1050)
        self.assertEqual(len({e.episode_id for e in self.episodes}), 1050)
        self.assertEqual(Counter(e.split for e in self.episodes), {'dev': 300, 'test': 750})
        for region in self.regions:
            es = [e for e in self.episodes if e.area == region['area']]
            self.assertEqual(len({(e.start, e.goal) for e in es}), 75)
            self.assertEqual(Counter(self.strata[e.episode_id]['stratum'] for e in es),
                             {'long_distance': 25, 'seam_target': 25, 'interior_target': 25})
            for layer in ('seam_target', 'interior_target'):
                self.assertEqual(Counter(e.dist for e in es if self.strata[e.episode_id]['stratum'] == layer), {d: 5 for d in range(8, 13)})

    def test_target_seam_geometry_and_both_directions(self):
        labels = []
        for e in self.episodes:
            meta = self.strata[e.episode_id]
            r, c = divmod(e.goal, 10)
            if meta['stratum'] == 'seam_target':
                self.assertTrue((r in (4, 5)) != (c in (4, 5)))
                labels.append((meta['seam_orientation'], meta['seam_goal_side']))
            elif meta['stratum'] == 'interior_target':
                self.assertFalse(r in (4, 5) or c in (4, 5))
        self.assertEqual(set(labels), {('horizontal', 4), ('horizontal', 5), ('vertical', 4), ('vertical', 5)})
        self.assertTrue(all(self.strata[e.episode_id]['evaluation_only'] for e in self.episodes))

    def test_deterministic_sampling_and_nonzero_seed_binding(self):
        again, meta = tasks(self.regions[:1])
        self.assertEqual(again, self.episodes[:75])
        self.assertEqual(meta, {e.episode_id: self.strata[e.episode_id] for e in again})

    def test_probe_geometry_matching_and_no_navigation_SR(self):
        ps = probes(self.regions)
        self.assertEqual(len(ps), 1680)
        self.assertEqual(len({p['probe_id'] for p in ps}), 1680)
        quadrant = lambda cell: 2*(cell//10//5)+cell%10//5
        for p in ps:
            self.assertTrue(0 <= p['current_cell'] < 100 and 0 <= p['target_cell'] < 100)
            self.assertFalse(p['counts_as_navigation'])
            self.assertTrue(p['evaluation_only'])
            cross = quadrant(p['current_cell']) != quadrant(p['target_cell'])
            if p['kind'] == 'cross_source_adjacent':
                self.assertTrue(cross); self.assertEqual(p['distance'], 1)
            elif p['kind'] == 'same_source_adjacent':
                self.assertFalse(cross); self.assertEqual(p['distance'], 1)
            else:
                self.assertEqual(p['distance'], 2); self.assertEqual(p['expected_class'], 'not_adjacent')
        for region in self.regions:
            pp = [p for p in ps if p['area'] == region['area']]
            self.assertEqual(Counter(p['kind'] for p in pp), {'cross_source_adjacent': 40, 'same_source_adjacent': 40, 'nonadjacent_matched_target': 40})
            for pair in range(40):
                matched = [p for p in pp if p['pair_id'] == pair]
                self.assertEqual(len({p['target_cell'] for p in matched}), 1)

    def test_wrong_target_distance_or_explicit_exception(self):
        wrong = wrong_plan(self.episodes)
        self.assertEqual(wrong, wrong_plan(self.episodes))
        for e in self.episodes:
            w = wrong[e.episode_id]
            self.assertNotIn(w['cue_cell'], (e.start, e.goal))
            self.assertEqual(l1(e.start, w['cue_cell']) == e.dist, w['matched_distance'])

    def episode(self, split='test', source='MasaRoadsArea/4'):
        return SpatialAreaEpisode('fixture', split, 'img_2004' if split == 'test' else 'img_2000', 0, 99, 18,
                                  source_tile=source)

    def test_public_observation_and_new_protocol(self):
        env = SpatialAreaGridEnv(self.grid)
        obs = env.reset(self.episode())
        self.assertEqual(obs.grid_size, 10)
        self.assertEqual(obs.remaining_budget, 20)
        self.assertFalse({f.name for f in fields(obs)} & {'goal', 'source_tile', 'epsg', 'bounds', 'stratum', 'area'})
        with Image.open(BytesIO(obs.current_image)) as im:
            self.assertFalse(im.getexif())

    def test_old_protocol_and_wrong_source_rejected(self):
        env = SpatialAreaGridEnv(self.grid)
        for ep in (ScaledEpisode('old', 'test', 'img_2004', 0, 99, 18),
                   AreaEpisode('old', 'dev', 'img_2000', 0, 99, 18, source_tile='MasaRoadsArea/0'), self.episode(source='wrong')):
            with self.assertRaises(ValueError): env.reset(ep)

    def test_last_budget_hit_and_oob_zero_distance(self):
        env = SpatialAreaGridEnv(self.grid)
        env.reset(self.episode())
        for i, action in enumerate(['up', 'up']+['right']*9+['down']*9):
            obs, done, info = env.step(action)
            self.assertEqual(info.out_of_bounds, i < 2)
            self.assertEqual(done, i == 19)
            self.assertEqual(obs.remaining_budget, 19-i)
        r = env.evaluator_result()
        self.assertTrue(r['success']); self.assertEqual(r['sg_m'], 0); self.assertEqual(r['valid_travel_m'], 5400)
        with self.assertRaises(RuntimeError): env.step('down')

    def test_failed_budget_sg_and_split_binding(self):
        env = SpatialAreaGridEnv(self.grid)
        env.reset(self.episode())
        for _ in range(20): env.step('up')
        r = env.evaluator_result()
        self.assertFalse(r['success']); self.assertEqual(r['sg_m'], 5400); self.assertEqual(r['valid_travel_m'], 0)
        dev = SpatialAreaEpisode('dev', 'dev', 'img_2000', 0, 99, 18, source_tile='MasaRoadsArea/0')
        env.reset(dev)


if __name__ == '__main__': unittest.main()
