"""Synthetic input/geometry/role tests without images or models."""
import unittest
from project.src.data import crossview_input_admission_v1 as d
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame


class Tests(unittest.TestCase):
    def test_geo_components_transitive_and_adjacent_frames_not_independent(self):
        rows=[dict(category='a',ground_latlon=[0,0]),dict(category='a',ground_latlon=[0,.003]),dict(category='a',ground_latlon=[0,.006]),dict(category='b',ground_latlon=[2,2])]
        self.assertEqual(d.connected_groups(rows)['groups'],[[0,1,2],[3]])
    def test_same_sequence_group_even_if_far_and_distinct_sequence_namespaces(self):
        rows=[dict(category='zod/seq',sequence_idx='0',ground_latlon=[0,0]),dict(category='zod/seq',sequence_idx='0',ground_latlon=[1,1]),dict(category='zod/drive',sequence_idx='0',ground_latlon=[2,2])]
        self.assertEqual(d.connected_groups(rows)['groups'],[[0,1],[2]])
    def test_spatial_sign_units_and_no_survey_claim(self):
        r=dict(ground_image='g',aerial_image='a',ground_latlon=[0,0],aerial_latlon=[0,0],dx=0,dy=0,resolution=.16)
        v=d.spatial_fields(r);self.assertTrue(v['GPS_offset_agree_within1m']);self.assertFalse(v['independent_camera_survey_accuracy_verified'])
        r['resolution']=0;self.assertFalse(d.spatial_fields(r)['valid'])
    def test_calibration_shape_is_insufficient_and_resize_pad_adjustment(self):
        K=[[400,0,480],[0,400,190],[0,0,1]]
        self.assertFalse(d.K_status(K,[960,380],False)['usable_calibration'])
        self.assertTrue(d.K_status(K,[960,380],True)['usable_calibration'])
        self.assertFalse(d.K_status(K,[960,380],'false')['usable_calibration'])
        self.assertEqual(d.transformed_K(K,.5,.5,10,20),[[200,0,250],[0,200,115],[0,0,1]])
    def test_image_only_has_no_hidden_pose_or_bank(self):
        f=RGBFrame(1,1,b'\x01\x02\x03');v=dict(given_target_rgb=f,current_aerial_rgb=f,position=[0,0],grid_size=5,step_index=0,remaining_budget=10,legal_actions=['right'])
        self.assertTrue(d.validate_public_packet(v))
        for k in ('GPS','heading','true_xy','goal_cell','source_id','parent_image','history'):
            with self.assertRaises(ValueError):d.validate_public_packet(dict(v,**{k:None}))
    def test_role_and_pixel_gaps_do_not_become_training_from_record_count(self):
        r=d.training_readiness(camera_point_label=True,geometry_checked=True,pixels_checked=False,role_authorized=False,split_isolated=True,profile_supported=True)
        self.assertFalse(r['admitted']);self.assertEqual(set(r['missing']),{'pixels_checked','role_authorized'})
        self.assertTrue(d.training_readiness(camera_point_label=True,geometry_checked=True,pixels_checked=True,role_authorized=True,split_isolated=True,profile_supported=True)['admitted'])


if __name__=='__main__':unittest.main()
