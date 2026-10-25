"""Actual-run boundary checks using only synthetic RGB/numeric fixtures."""
import copy
import unittest
import numpy as np
from project.src.agents import observed_canvas_runtime_v1 as rt
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame
from project.src.platform_core.observed_canvas_v1 import synthetic_tile
from project.src.data import observed_canvas_protocol_v1 as protocol


def goal():return RGBFrame(1,1,b'\x01\x02\x03')
def packet(step):
    p=protocol.target_blind_path()[step]
    return dict(given_target_rgb=goal(),current_image=synthetic_tile(p['public_cell']),position=p['position'],grid_size=5,
                step_index=step,remaining_budget=10-step,legal_actions=('up','right','down','left'))


class Tests(unittest.TestCase):
    def test_replay_goal_and_no_hidden_parent_or_history(self):
        r=rt.Replay(goal())
        for extra in ('parent_image','goal_cell','source_id','history','true_xy'):
            with self.assertRaises(ValueError):r.observe(dict(packet(0),**{extra:None}))
        r.observe(packet(0));self.assertEqual(r.core.receipt()['unique_cells'],1)
    def test_initial_both_representations_identical(self):
        r=rt.Replay(goal());r.observe(packet(0));np.testing.assert_array_equal(r.input_array('Single'),r.input_array('Cumulative'))
    def test_prefix_changes_only_retention_and_common_masks(self):
        r=rt.Replay(goal());r.observe(packet(0));r.observe(packet(1),last_action='right')
        self.assertFalse(np.array_equal(r.input_array('Single'),r.input_array('Cumulative')))
        self.assertEqual(r.core.receipt()['visited_cells'],(10,11));self.assertEqual(r.input_array('Cumulative').shape,(1,3,512,512))
    def test_changed_goal_or_skipped_prefix_rejected(self):
        r=rt.Replay(goal());r.observe(packet(0))
        with self.assertRaises(ValueError):r.observe(packet(2),last_action='right')
        bad=packet(1);bad['given_target_rgb']=RGBFrame(1,1,b'\x09\x09\x09')
        with self.assertRaises(ValueError):r.observe(bad,last_action='right')
    def test_frozen_cohort_rejects_subset_reorder_or_altered_label(self):
        d=protocol.make_cases(protocol.read(protocol.BASE/'固定四对清单.json')['pairs'],protocol.read(protocol.BASE/'空间监督检查.json')['pairs'])['development']
        rt.require_frozen_cases(d,copy.deepcopy(d))
        for bad in (d[:-1],list(reversed(d)),d[:-1]+d[:1]):
            with self.assertRaises(ValueError):rt.require_frozen_cases(bad,d)
        bad=copy.deepcopy(d);bad[0]['labels_evaluator_only']['primary_relation']='inside'
        with self.assertRaises(ValueError):rt.require_frozen_cases(bad,d)
    def test_all_25_pixel_center_masks_partition_both_sizes(self):
        for size in (256,512):
            a=sum(rt.mask(size,[i]).astype(np.uint8) for i in range(25))
            self.assertTrue(np.all(a==1))
    def test_two_peak_algorithms_and_unvisited_global_peak(self):
        h=np.zeros((512,512),np.float32);m=np.zeros((256,256),np.float32);h[0,0]=100
        h[220,10]=.8;m[110,10]=.75
        p,c,receipt=rt.peaks_checked(h,m,[10,11],11)
        self.assertTrue(receipt['passed']);self.assertEqual(p['heat_index'],220*512+10);self.assertEqual(p['score'],.75)
        self.assertNotEqual(p['xy'],c['xy'])
    def test_equal_heat_ties_global_row_major(self):
        h=np.ones((512,512),np.float32);m=np.zeros((256,256),np.float32)
        self.assertEqual(rt.vector_peak(h,m,[11,10]),rt.alternate_peak(h,m,[10,11]))
    def test_nonfinite_or_invalid_ROI_rejected(self):
        h=np.zeros((512,512),np.float32);m=np.zeros((256,256),np.float32)
        for cells in ([],[True],[25],[10,10]):
            with self.assertRaises(ValueError):rt.vector_peak(h,m,cells)
        h[0,0]=np.nan
        with self.assertRaises(ValueError):rt.vector_peak(h,m,[10])
    def test_resource_limits_exact_boundaries(self):
        self.assertFalse(rt.resource_violation(4*1024**3,1800,16*1024**2))
        self.assertTrue(rt.resource_violation(4*1024**3+1,1,1));self.assertTrue(rt.resource_violation(1,1800.1,1));self.assertTrue(rt.resource_violation(1,1,16*1024**2+1))
    def test_close_erases_public_pixels(self):
        r=rt.Replay(goal());r.observe(packet(0));r.close();self.assertIsNone(r.target);self.assertEqual(r.core.tiles,{})
        with self.assertRaises(ValueError):r.input_array('Single')


if __name__=='__main__':unittest.main()
