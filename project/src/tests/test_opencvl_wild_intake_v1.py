"""In-memory synthetic boundary and separation tests for OpenCVL wild intake v1."""
import struct
import unittest
import zlib

from project.src.data.opencvl_wild_intake_v1 import (
    cell_from_offsets,
    connected_components_by_gap,
    dual_pose_summary,
    partition_se_clean_macros,
    summarize_role_stats,
)


def make_label(sid='1001', lat=55.6, lon=13.0, *, dx_m=0, dy_m=0, dx_o=0, dy_o=0, res=0.16, include_latlon=True):
    return {
        'mapil_image_id': sid,
        'mapil_image': f'ground/{sid}.png',
        'aerial_image': f'aerial/{sid}.png',
        'country': 'SE' if include_latlon else 'PL',
        'resolution': res,
        'dx_mast3rpnp': dx_m,
        'dy_mast3rpnp': dy_m,
        'heading_mast3rpnp': 0.0,
        'dx_mapilopensfm': dx_o,
        'dy_mapilopensfm': dy_o,
        'heading_mapilopensfm': 0.0,
        'max_offset_m': 20.0,
        'aerial_latlon': [lat, lon] if include_latlon else None,
        'mapillary_creator_id': 'c1',
    }


def make_clean_node(sid, lat, lon, min_known_gap_m=120_000.0, idx=12):
    return {
        'source_key': f'mapillary:{sid}',
        'image_id': sid,
        'aerial_latlon': [lat, lon],
        'radius_m': 70.71,
        'min_known_gap_m': min_known_gap_m,
        'mast3rpnp': {'index': idx},
        'opensfm_cell_agree': True,
        'in_center9': True,
        'creator_id': 'c1',
    }


class OpenCVLWildIntakeTests(unittest.TestCase):
    def test_cell_boundaries_and_boolean_rejection(self):
        c12 = cell_from_offsets(0.0, 0.0, 0.0, 0.16, 625, 625)
        self.assertEqual((c12['inside'], c12['index']), (True, 12))
        edge_out = cell_from_offsets(312.5, 0.0, 0.0, 0.16, 625, 625)
        self.assertFalse(edge_out['inside'])
        with self.assertRaises(ValueError):
            cell_from_offsets(True, 0.0, 0.0, 0.16, 625, 625)

    def test_dual_pose_handles_se_and_pl_explicitly(self):
        se = dual_pose_summary(make_label('1234', dx_m=10, dy_m=10, dx_o=150, dy_o=150), 625, 625, require_latlon=True)
        self.assertEqual(se['mast3rpnp']['index'], 12)
        self.assertFalse(se['opensfm_cell_agree'])
        self.assertTrue(se['in_center9'])

        pl = dual_pose_summary(make_label('2345', res=0.1, include_latlon=False), 1000, 1000, require_latlon=False)
        self.assertIsNone(pl['aerial_latlon'])
        self.assertTrue(pl['opensfm_cell_agree'])

        with self.assertRaises(ValueError):
            dual_pose_summary(make_label('2345', include_latlon=True), 1000, 1000, require_latlon=False)

    def test_pair_path_id_mismatch_and_traversal_rejected(self):
        bad = make_label('1234')
        bad['aerial_image'] = 'aerial/9999.png'
        with self.assertRaises(ValueError):
            dual_pose_summary(bad, 625, 625, require_latlon=True)
        bad2 = make_label('1234')
        bad2['mapil_image'] = '../ground/1234.png'
        with self.assertRaises(ValueError):
            dual_pose_summary(bad2, 625, 625, require_latlon=True)

    def test_connected_components_deterministic_and_symmetric(self):
        pts = [
            make_clean_node('1', 55.600, 13.000),
            make_clean_node('2', 55.602, 13.002),
            make_clean_node('3', 58.400, 15.600),
        ]
        groups = connected_components_by_gap(pts, 500.0, subtract_radii=True)
        self.assertEqual([[x['image_id'] for x in g] for g in groups], [['1', '2'], ['3']])

    def test_macro_partition_and_cross_split_clearance_guard(self):
        known = [
            {'source_key': 'cv045_1', 'center_latlon': [59.85, 17.63], 'radius_m': 70.71, 'is_cv045': True},
            {'source_key': 'hist_1', 'center_latlon': [57.70, 11.97], 'radius_m': 500.0, 'is_cv045': False},
        ]
        clean = [
            make_clean_node('10', 55.60, 13.00, 200_000.0),  # largest unseen -> fit
            make_clean_node('11', 55.601, 13.001, 200_000.0),
            make_clean_node('12', 55.602, 13.002, 200_000.0),
            make_clean_node('20', 58.40, 15.60, 100_000.0),  # second unseen -> val
            make_clean_node('21', 58.401, 15.601, 100_000.0),
            make_clean_node('30', 63.82, 20.26, 100_000.0),  # third unseen -> blind conf
            make_clean_node('40', 59.856, 17.63, 700.0),     # near cv045 -> quarantine
            make_clean_node('50', 57.71, 11.98, 900.0),      # near hist -> fit
        ]
        roles, _ = partition_se_clean_macros(clean, known)
        self.assertEqual(roles['mapillary:10'], 'development_fit_train')
        self.assertEqual(roles['mapillary:50'], 'development_fit_train')
        self.assertEqual(roles['mapillary:20'], 'development_validation')
        self.assertEqual(roles['mapillary:30'], 'isolated_confirmation_blind')
        self.assertEqual(roles['mapillary:40'], 'quarantined_cv045_vicinity')

    def test_role_stats_always_cell12_floor(self):
        rows = [make_clean_node('1', 55.0, 13.0, idx=12), make_clean_node('2', 55.0, 13.0, idx=7)]
        stats = summarize_role_stats(rows)
        self.assertEqual(stats['count'], 2)
        self.assertEqual(stats['always_cell12_count'], 1)
        self.assertEqual(stats['always_cell12_rate'], 0.5)

    def test_crc_formula_consistency(self):
        chunk = b'IHDR' + struct.pack('>IIBBBBB', 625, 625, 8, 2, 0, 0, 0)
        crc = zlib.crc32(chunk) & 0xffffffff
        self.assertEqual(len(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + chunk + struct.pack('>I', crc)), 33)


if __name__ == '__main__':
    unittest.main()
