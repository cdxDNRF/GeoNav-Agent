"""Native window and geometry regressions on synthetic sources only."""
from pathlib import Path
import copy
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import area_windows_v1 as d
from eval import audit_area_windows_v1 as audit


def fixture():
    names=[str(i) for i in range(16)]
    rows={n:dict(x=(i%4)*1500,y=6000-(i//4)*1500,quality=dict(passed=True),
                 epsg=26986,raster_type='PixelIsArea') for i,n in enumerate(names)}
    counts={n:np.full((5,5),int(n)+1,np.int64) for n in names}
    return names,rows,counts


def region(bounds,names):
    return dict(bounds_m=bounds,source_windows=[dict(name=n) for n in names])


class WindowTests(unittest.TestCase):
    def test_blank_integer_gates_preserve_boundary(self):
        count=np.full((5,5),900,np.int64)
        self.assertTrue(d.quality_ok(count))
        count[0,0]+=1; self.assertFalse(d.quality_ok(count))
        count=np.zeros((5,5),np.int64); count[0,0]=1800
        self.assertTrue(d.quality_ok(count))
        count[0,0]+=1; self.assertFalse(d.quality_ok(count))

    def test_only_exact_rgb_black_or_white_is_blank(self):
        rgb=np.full((300,300,3),80,np.uint8)
        rgb[0,0]=0; rgb[0,1]=255; rgb[0,2]=[0,0,1]; rgb[0,3]=[255,255,254]
        self.assertEqual(d.quality_counts(rgb).tolist(),[[2]])
        self.assertEqual(audit.independent_counts(rgb).tolist(),[[2]])

    def test_native_window_uses_actual_shifted_sources(self):
        names,rows,counts=fixture()
        r=d.candidate(names,4,300,900,rows,counts,[0,6000],'window4x4')
        self.assertEqual(r['bounds_m'],[300,600,4800,5100])
        self.assertEqual(r['cell_sources'][0],'0'); self.assertEqual(r['cell_sources'][-1],'15')
        self.assertEqual(r['source_windows'][0]['pixel_window'],[300,900,1500,1500])
        self.assertEqual(r['source_windows'][-1]['pixel_window'],[0,0,300,900])
        self.assertEqual(sum((s['pixel_window'][2]-s['pixel_window'][0])*(s['pixel_window'][3]-s['pixel_window'][1]) for s in r['source_windows']),4500**2)

    def test_cell_counts_equal_direct_native_crop(self):
        names,rows,counts=fixture()
        countgrid=np.concatenate([np.concatenate([counts[n] for n in names[r*4:r*4+4]],axis=1) for r in range(4)],axis=0)
        r=d.candidate(names,4,1200,600,rows,counts,[0,6000],'window4x4')
        self.assertEqual(r['cell_blank_counts'],countgrid[2:17,4:19].reshape(-1).tolist())

    def test_unaligned_or_oversized_window_rejected(self):
        names,rows,counts=fixture()
        for dx,dy in ((1,0),(1800,0),(0,-300)):
            with self.assertRaises(ValueError): d.candidate(names,4,dx,dy,rows,counts,[0,6000],'window4x4')

    def test_bad_source_or_parent_gap_cannot_be_hidden_by_crop(self):
        names,rows,counts=fixture(); rows['15']['quality']['passed']=False
        with self.assertRaises(ValueError): d.candidate(names,4,0,0,rows,counts,[0,6000],'window4x4')
        rows['15']['quality']['passed']=True; rows['15']['x']+=1
        with self.assertRaises(ValueError): d.candidate(names,4,0,0,rows,counts,[0,6000],'window4x4')

    def test_same_geographic_window_deduplicates_baseline_parent(self):
        names,rows,counts=fixture()
        full=d.candidate(names,4,0,0,rows,counts,[0,6000],'window4x4')
        small=[names[r*4+c] for r in range(3) for c in range(3)]
        base=d.candidate(small,3,0,0,rows,counts,[0,6000],'baseline3x3')
        union=d.deduplicate([full,base])
        self.assertEqual(len(union),1); self.assertEqual(len(union[0]['origins']),2)

    def test_duplicate_key_with_conflicting_provenance_rejected(self):
        names,rows,counts=fixture(); a=d.candidate(names,4,0,0,rows,counts,[0,6000],'window4x4')
        b=copy.deepcopy(a); b['cell_sources'][0]='other'
        with self.assertRaises(ValueError): d.deduplicate([a,b])

    def test_full_rectangle_buffer_and_diagonal_distance(self):
        a=[0,0,4500,4500]
        self.assertEqual(d.distance(a,[7500,0,12000,4500]),3000)
        self.assertEqual(audit.rectangle_distance(a,[4500,4500,9000,9000]),0)
        self.assertAlmostEqual(d.distance(a,[6500,6500,11000,11000]),2000*2**.5)

    def test_two_packers_agree_on_conflicts_and_source_reuse(self):
        regions=[region([0,0,4500,4500],['a']),region([7500,0,12000,4500],['b']),
                 region([15000,0,19500,4500],['b']),region([1000,0,5500,4500],['d'])]
        production=d.pack(regions,3); independent=audit.independent_maximum(regions,3)
        self.assertTrue(production['exact']); self.assertTrue(independent['exact'])
        self.assertEqual(production['count'],2); self.assertEqual(independent['count'],2)

    def test_timeout_never_claims_exact_maximum(self):
        result=d.pack([region([0,0,4500,4500],['a'])],0)
        self.assertFalse(result['exact']); self.assertIsNone(result['maximum'])

    def test_empty_pool_is_zero_not_one(self):
        self.assertEqual(d.pack([],3)['count'],0)
        self.assertEqual(audit.independent_maximum([],3)['maximum'],0)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WindowTests))
    if d.TESTS.exists(): raise RuntimeError('test evidence already exists')
    d.write(d.TESTS,dict(successful=result.wasSuccessful(),tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        utc=d.stamp(),scope='synthetic pixels/provenance/geometry/packing only',
        sources={d.rel(d.ROOT/'project/src'/name):d.digest(d.ROOT/'project/src'/name) for name in d.CODE}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
