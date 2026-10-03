"""Fresh geographic isolation, new-only quotas and fixed data-only contracts."""
from collections import Counter
from dataclasses import fields
from io import BytesIO
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from data import radial_confirmation_areas as fresh
from env.spatial_area import SpatialAreaEpisode,SpatialAreaGridEnv,PROTOCOL

ROOT=Path(__file__).resolve().parents[3]


class FreshAreaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regions=[dict(area=f'img_{3000+i}',split='test',source_tile=f'MasaRoadsFresh/test_{i:02d}') for i in range(10)]
        cls.episodes,cls.strata=fresh.tasks(cls.regions)
        parent=(ROOT/'选题报告相关').resolve()
        cls.temp=tempfile.TemporaryDirectory(prefix='径向新区域测试临时_',dir=parent)
        cls.path=Path(cls.temp.name).resolve()
        def cleanup():
            target=cls.path.resolve()
            if target.parent!=parent or not target.name.startswith('径向新区域测试临时_'):
                raise ValueError('fixture cleanup outside intended workspace')
            cls.temp.cleanup()
        cls.addClassCleanup(cleanup)
        cls.grid=cls.path/'grid';folder=cls.grid/'patches/test/img_3000';folder.mkdir(parents=True)
        b=BytesIO();Image.new('RGB',(300,300),(80,110,120)).save(b,format='JPEG',quality=75)
        for cell in range(100):(folder/f'patch_{cell}.jpg').write_bytes(b.getvalue())
        manifest=dict(protocol=PROTOCOL,epsg=26986,pixel_size_m=1,cell_size_m=300,native_cell_pixels=300,
            mosaic_pixels=3000,grid_size=10,regions=[cls.regions[0]])
        (cls.grid/'数据清单.json').write_text(json.dumps(manifest),encoding='utf-8')

    def test_fresh_filter_excludes_previously_consumed_whole_regions(self):
        names=['10008000_15.tiff','10158000_15.tiff','10007985_15.tiff','10157985_15.tiff']
        catalog={'urls':{n:'https://www.cs.toronto.edu/'+n for n in names}}
        old=[dict(id='09907900_15',x=99000,y=790000,bounds_m=[99000,788500,100500,790000])]
        consumed=[dict(bounds_m=[100000,797000,103000,800000],sources=[{'id':n[:-5]} for n in names])]
        self.assertEqual(len(fresh.fresh_candidates(catalog,old,[])),1)
        self.assertEqual(fresh.fresh_candidates(catalog,old,consumed),[])

    def test_geographic_exclusion_does_not_depend_on_matching_source_names(self):
        names=['10008000_15.tiff','10158000_15.tiff','10007985_15.tiff','10157985_15.tiff']
        catalog={'urls':{n:'https://www.cs.toronto.edu/'+n for n in names}}
        old=[dict(id='09907900_15',x=99000,y=790000,bounds_m=[99000,788500,100500,790000])]
        consumed=[dict(bounds_m=[103100,797000,106100,800000],sources=[{'id':'different'}])]
        self.assertEqual(fresh.fresh_candidates(catalog,old,consumed),[])

    def test_full_footprints_use_boundary_gap(self):
        self.assertFalse(fresh.clear_of([0,0,3000,3000],[[3100,0,6100,3000]]))
        self.assertTrue(fresh.clear_of([0,0,3000,3000],[[6000,0,9000,3000]]))

    def test_only_new_unique_downloads_consume_file_budget(self):
        reused={f'old{i}' for i in range(152)}
        cached=reused|{f'new{i}' for i in range(198)}
        self.assertEqual(fresh.new_download_count(cached,['old0','new198','new199'],reused),200)
        self.assertGreater(fresh.new_download_count(cached,['a','b','c','d'],reused),fresh.FILE_LIMIT)

    def test_resume_cannot_issue_a_third_request_for_a_url(self):
        acq=fresh.Acquisition.__new__(fresh.Acquisition)
        acq.cache={};acq.previous_attempts={'fixture.tiff':2};acq.catalog={'urls':{'fixture.tiff':'https://www.cs.toronto.edu/fixture.tiff'}}
        with patch.object(fresh,'urlopen',side_effect=AssertionError('network forbidden')) as request:
            with self.assertRaisesRegex(ValueError,'two attempts'):acq.fetch('fixture.tiff')
            request.assert_not_called()

    def test_all750_routes_and_balanced_strata_are_fixed(self):
        self.assertEqual(len(self.episodes),750)
        self.assertEqual(len({e.episode_id for e in self.episodes}),750)
        self.assertEqual(Counter(e.split for e in self.episodes),{'test':750})
        self.assertEqual(Counter(v['stratum'] for v in self.strata.values()),{'long_distance':250,'seam_target':250,'interior_target':250})
        for region in self.regions:
            es=[e for e in self.episodes if e.area==region['area']]
            self.assertEqual(len({(e.start,e.goal) for e in es}),75)
            for layer in ('long_distance','seam_target','interior_target'):
                self.assertEqual(Counter(e.dist for e in es if self.strata[e.episode_id]['stratum']==layer),
                    {d:5 for d in (range(12,17) if layer=='long_distance' else range(8,13))})

    def test_sampler_is_deterministic_and_geometry_labels_evaluation_only(self):
        again,labels=fresh.tasks(self.regions)
        self.assertEqual(again,self.episodes);self.assertEqual(labels,self.strata)
        self.assertTrue(all(x['evaluation_only'] for x in labels.values()))
        for e in again:
            r,c=divmod(e.goal,10);layer=labels[e.episode_id]['stratum']
            if layer=='seam_target':self.assertTrue((r in (4,5))!=(c in (4,5)))
            if layer=='interior_target':self.assertFalse(r in (4,5) or c in (4,5))

    def test_1200_matched_probes_never_count_as_navigation(self):
        probes=fresh.probes(self.regions)
        self.assertEqual(len(probes),1200)
        self.assertEqual(Counter(x['kind'] for x in probes),{'cross_source_adjacent':400,'same_source_adjacent':400,'nonadjacent_matched_target':400})
        self.assertTrue(all(x['evaluation_only'] and not x['counts_as_navigation'] for x in probes))

    def test_wrong_targets_preserve_distance_or_record_exception(self):
        plan=fresh.wrong_plan(self.episodes)
        for e in self.episodes:
            p=plan[e.episode_id];self.assertNotIn(p['cue_cell'],(e.start,e.goal))
            d=abs(e.start//10-p['cue_cell']//10)+abs(e.start%10-p['cue_cell']%10)
            self.assertEqual(d==e.dist,p['matched_distance'])

    def test_public_observation_carries_no_target_coordinate_or_region_label(self):
        env=SpatialAreaGridEnv(self.grid)
        obs=env.reset(self.episodes[0])
        self.assertFalse({f.name for f in fields(obs)}&{'goal','dist','distance','area','source_tile','stratum','epsg'})
        self.assertEqual(obs.remaining_budget,20)
        with Image.open(BytesIO(obs.current_image)) as im:self.assertFalse(im.getexif())

    def test_new_area_source_binding_rejects_wrong_region_identity(self):
        env=SpatialAreaGridEnv(self.grid)
        e=SpatialAreaEpisode('fixture','test','img_3000',0,99,18,source_tile='wrong')
        with self.assertRaises(ValueError):env.reset(e)

    def test_data_preparation_imports_no_neural_runtime(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('transformers',sys.modules)
        self.assertEqual((fresh.FILE_LIMIT,fresh.BYTE_LIMIT),(200,3*1024**3//2))
        self.assertIs(fresh.geoinfo,fresh.core.geoinfo)


if __name__=='__main__':unittest.main()
