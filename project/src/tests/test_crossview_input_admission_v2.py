"""In-memory schema/pose tests. No files, real images, model imports or calls."""
import copy
import unittest
from project.src.data.crossview_input_admission_v2 import (
    mapillary_pose, pose_increment_consistency, normalized_mapillary_rows,
)


def sample():
    # ~11.1m east: corrected dx increases; positive dy is north/up.
    return dict(mapil_image='ground/a.png', aerial_image='aerial/a.png',
                ground_latlon=[0, 0], ground_latlon_mapilraw=[0, 0],
                ground_latlon_mapilopensfm=[0, 0.0001],
                dx_mapilraw=-10, dy_mapilraw=2, heading_mapilraw=30,
                dx_mapilopensfm=101.19508, dy_mapilopensfm=2,
                heading_mapilopensfm=35, resolution=0.1,
                pose_mapilopensfm_available=True)


class PoseTests(unittest.TestCase):
    def test_explicit_pose_namespace_and_no_aerial_GPS_invention(self):
        r = sample()
        a, b = (mapillary_pose(r, variant=v) for v in ('raw', 'opensfm'))
        self.assertEqual(a['ground_latlon'], [0, 0])
        self.assertEqual(b['ground_latlon'], [0, 0.0001])
        self.assertEqual((b['dx'], b['heading']), (101.19508, 35))
        self.assertNotIn('aerial_latlon', b)
        self.assertFalse(b['absolute_camera_survey_verified'])

    def test_no_implicit_or_unavailable_pose_fallback(self):
        r = sample()
        with self.assertRaises(TypeError): mapillary_pose(r)
        with self.assertRaises(ValueError): mapillary_pose(r, variant='auto')
        r['pose_mapilopensfm_available'] = False
        with self.assertRaises(ValueError): mapillary_pose(r, variant='opensfm')
        self.assertEqual(mapillary_pose(r, variant='raw')['dx'], -10)
        member = dict(member=dict(name='OpenCVL_sample_data/mapillary/cities/SE/s/labels.json'), records=[r])
        self.assertFalse(normalized_mapillary_rows([member], 'raw')[0]['consistency']['available'])

    def test_wrong_raw_alias_cannot_silently_mix_coordinates(self):
        r = sample(); r['ground_latlon'] = r['ground_latlon_mapilopensfm']
        with self.assertRaises(ValueError): mapillary_pose(r, variant='opensfm')

    def test_increment_sign_and_units(self):
        r = sample(); self.assertLess(pose_increment_consistency(r)['residual_m'], 1e-5)
        r['dx_mapilopensfm'] = -121.19508
        self.assertGreater(pose_increment_consistency(r)['residual_m'], 22)

    def test_invalid_numeric_fields_rejected(self):
        for field, value in [('resolution', 0), ('dx_mapilraw', float('nan')),
                             ('heading_mapilraw', True)]:
            r = sample(); r[field] = value
            with self.assertRaises(ValueError): mapillary_pose(r, variant='raw')

    def test_city_only_no_test_member_or_mutation(self):
        r = sample(); original = copy.deepcopy(r)
        member = dict(member=dict(name='OpenCVL_sample_data/mapillary/cities/SE/s/labels.json'), records=[r])
        result = normalized_mapillary_rows([member], 'opensfm')
        self.assertEqual(r, original)
        self.assertFalse(result[0]['K_status']['usable_calibration'])
        member['member']['name'] = 'OpenCVL_sample_data/mapillary/in_the_wild_test/labels.json'
        with self.assertRaises(ValueError): normalized_mapillary_rows([member], 'raw')


if __name__ == '__main__': unittest.main()
