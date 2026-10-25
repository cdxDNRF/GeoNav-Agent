"""Memory-only synthetic canvas checks; no real image/model or environment runs."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from project.src.platform_core.observed_canvas_v1 import ObservedCanvas,synthetic_packet,synthetic_tile,MAX_PIXEL_BYTES,MEAN,STD,array_sha
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame


def goal(v=7):return RGBFrame(2,1,bytes([v,2,3,v,4,5]))


class Tests(unittest.TestCase):
    def start(self,budget=10,max_bytes=MAX_PIXEL_BYTES):
        s=ObservedCanvas(max_pixel_bytes=max_bytes);s.begin(goal(),budget=budget);return s
    def test_native_unknown_exact_mean_and_mask(self):
        s=self.start();v=s.observe_public(synthetic_packet((2,3),0,10));a=s.snapshot()
        self.assertEqual(v['visited_cells'],(13,));self.assertEqual(v['known_fraction'],.04)
        self.assertEqual(int(a.native_seen.sum()),125*125)
        self.assertTrue(np.all(a.native_rgb[~a.native_seen]==np.asarray(MEAN,np.float32)))
        self.assertTrue(np.all(a.normalized_nchw[0,:,~a.visible_support]==0))
    def test_row_col_and_corner_pixels(self):
        s=self.start();s.observe_public(synthetic_packet((1,4),0,10));a=s.snapshot()
        np.testing.assert_array_equal(a.native_rgb[125,500],np.array([1,2,3],np.float32)/255)
        np.testing.assert_array_equal(a.native_rgb[249,624],np.array([251,252,253],np.float32)/255)
        self.assertFalse(a.native_seen[124,500]);self.assertFalse(a.native_seen[125,499])
    def test_new_cells_accumulate_and_revisit_no_coverage_gain(self):
        s=self.start();s.observe_public(synthetic_packet((1,1),0,10));s.observe_public(synthetic_packet((1,2),1,10),last_action='right')
        before=s.snapshot();s.observe_public(synthetic_packet((1,1),2,10),last_action='left');after=s.snapshot()
        self.assertEqual(after.visited_cells,(6,7));np.testing.assert_array_equal(before.native_rgb,after.native_rgb)
        np.testing.assert_array_equal(before.native_seen,after.native_seen)
    def test_boundary_stay_consumes_step_without_new_pixels(self):
        s=self.start();s.observe_public(synthetic_packet((0,0),0,10));v=s.observe_public(synthetic_packet((0,0),1,10),last_action='up')
        self.assertEqual(v['remaining_budget'],9);self.assertEqual(v['unique_cells'],1)
    def test_external_truth_history_paths_and_bank_rejected(self):
        s=self.start()
        for key in ('goal_index','goal_position','GPS','heading','source_id','history','visited','parent_image','path','candidate_bank','unvisited_features'):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):s.observe_public(dict(synthetic_packet((0,0),0,10),**{key:None}))
        self.assertEqual(s.tiles,{})
    def test_skipped_replayed_and_inconsistent_steps_atomic(self):
        s=self.start();s.observe_public(synthetic_packet((2,2),0,10));before=s.snapshot()
        invalid=[synthetic_packet((2,3),2,10),synthetic_packet((2,2),0,10),synthetic_packet((1,1),1,10),dict(synthetic_packet((2,3),1,10),remaining_budget=10)]
        for packet in invalid:
            with self.assertRaises(ValueError):s.observe_public(packet,last_action='right')
            self.assertEqual(s.receipt()['step_index'],0)
        np.testing.assert_array_equal(s.snapshot().native_rgb,before.native_rgb)
    def test_terminal_last_observation_then_no_extra_step(self):
        s=self.start(budget=1);s.observe_public(synthetic_packet((2,2),0,1));r=s.observe_public(synthetic_packet((2,3),1,1),last_action='right')
        self.assertEqual(r['remaining_budget'],0)
        with self.assertRaises(ValueError):s.observe_public(synthetic_packet((2,4),2,2),last_action='right')
    def test_pixel_cap_refuses_without_eviction_or_partial_commit(self):
        s=self.start(max_bytes=125*125*3);s.observe_public(synthetic_packet((2,2),0,10))
        with self.assertRaises(ValueError):s.observe_public(synthetic_packet((2,3),1,10),last_action='right')
        self.assertEqual(s.receipt()['visited_cells'],(12,));self.assertEqual(s.receipt()['step_index'],0)
    def test_new_episode_and_invalid_begin_clear_old_history(self):
        s=self.start();s.observe_public(synthetic_packet((2,2),0,10));s.begin(goal(99),budget=2)
        with self.assertRaises(ValueError):s.snapshot()
        s.observe_public(synthetic_packet((0,0),0,2));a=s.snapshot()
        self.assertEqual(a.given_target,goal(99));self.assertEqual(a.visited_cells,(0,));self.assertFalse(a.native_seen[300,300])
        with self.assertRaises(ValueError):s.begin(goal(),grid_size=10)
        self.assertFalse(s.active);self.assertEqual(s.tiles,{})
    def test_snapshot_detached_from_session_state(self):
        s=self.start();s.observe_public(synthetic_packet((2,2),0,10));a=s.snapshot();h=array_sha(a.native_rgb)
        a.native_rgb.setflags(write=True);a.native_rgb[:]=0
        self.assertEqual(array_sha(s.snapshot().native_rgb),h)
        self.assertIsInstance(a.visited_cells,tuple)
    def test_static_revisit_changed_pixels_rejected(self):
        s=self.start();s.observe_public(synthetic_packet((0,0),0,10))
        with self.assertRaises(ValueError):s.observe_public(synthetic_packet((0,0),1,10,tile=synthetic_tile(1)),last_action='up')
        self.assertEqual(s.receipt()['step_index'],0)
    def test_parent_image_or_wrong_grid_or_action_permission_denied(self):
        s=self.start()
        with self.assertRaises(ValueError):s.observe_public(synthetic_packet((0,0),0,10,tile=RGBFrame(625,625,b'\0'*(625*625*3))))
        a=synthetic_packet((2,2),0,10);a['legal_actions']=('right',);s.observe_public(a)
        with self.assertRaises(ValueError):s.observe_public(synthetic_packet((1,2),1,10),last_action='up')
        with self.assertRaises(ValueError):s.observe_public(synthetic_packet((1,2),1,10),last_action={})
    def test_close_clears_pixels_and_target(self):
        s=self.start();s.observe_public(synthetic_packet((0,0),0,10));s.close()
        self.assertEqual(s.tiles,{});self.assertIsNone(s.target)
        with self.assertRaises(ValueError):s.snapshot()
    def test_full_snake_synthetic_stream_stays_bounded(self):
        s=self.start(budget=24);coords=[(r,c) for r in range(5) for c in (range(5) if r%2==0 else range(4,-1,-1))]
        old=None
        for i,pos in enumerate(coords):
            action=None if old is None else 'down' if pos[0]>old[0] else 'right' if pos[1]>old[1] else 'left'
            v=s.observe_public(synthetic_packet(pos,i,24),last_action=action);old=pos
        self.assertEqual(v['unique_cells'],25);self.assertEqual(v['pixel_bytes'],MAX_PIXEL_BYTES)
        self.assertTrue(s.snapshot().native_seen.all())
    def test_single_tile_matches_bound_legacy_preprocessor_without_torch(self):
        from project.src.agents.ccvpe_runtime_v1 import prepare_local_rgb
        class Tensor:
            def __init__(self,a):self.a=a
            def permute(self,*axes):return Tensor(self.a.transpose(axes))
            def unsqueeze(self,axis):return Tensor(np.expand_dims(self.a,axis))
            def contiguous(self):return Tensor(np.ascontiguousarray(self.a))
        fake=SimpleNamespace(from_numpy=lambda a:Tensor(a))
        with patch.dict(sys.modules,{'torch':fake}):
            for cell in range(25):
                s=self.start();frame=synthetic_tile(cell);s.observe_public(synthetic_packet(divmod(cell,5),0,10))
                with Image.frombytes('RGB',(125,125),frame.pixels) as im:old=prepare_local_rgb(im,cell).a
                np.testing.assert_array_equal(s.snapshot().normalized_nchw,old)


if __name__=='__main__':unittest.main()
