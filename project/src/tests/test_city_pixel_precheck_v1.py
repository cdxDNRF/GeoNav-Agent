"""In-memory-only tests; no directories/network/real images/model calls."""
import struct
import unittest
from unittest.mock import MagicMock,patch
import zlib
import numpy as np
from project.src.data.city_pixel_precheck_v1 import png_header,pixel_quality,label_geometry
from project.src.tests.test_crossview_input_admission_v2 import sample


def header(w=952,h=532):
    payload=struct.pack('>IIBBBBB',w,h,8,2,0,0,0)
    chunk=b'IHDR'+payload
    return b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+chunk+struct.pack('>I',zlib.crc32(chunk)&0xffffffff)


class Tests(unittest.TestCase):
    def test_png_identity_dimensions_and_crc(self):
        self.assertEqual(png_header(header())['width'],952)
        for b in (b'not png',header()[:-1]+b'\x00',header(5000,400)):
            with self.assertRaises(ValueError):png_header(b)

    def test_nonblank_and_saturation(self):
        self.assertFalse(pixel_quality(np.zeros((10,10,3),np.uint8))['nonblank'])
        self.assertFalse(pixel_quality(np.full((10,10,3),255,np.uint8))['nonblank'])
        a=np.arange(300,dtype=np.uint16).astype(np.uint8).reshape(10,10,3)
        self.assertTrue(pixel_quality(a)['nonblank'])

    def test_label_axes_and_pixel_center(self):
        r=sample();d=label_geometry(r,[952,532],[625,625])
        self.assertEqual(d['variants']['raw']['x'],302.5)
        self.assertEqual(d['variants']['raw']['y'],310.5)
        self.assertTrue(d['both_pose_points_inside'])
        self.assertFalse(d['absolute_raster_georeference_verified'])

    def test_outside_point_not_clipped_or_relabelled(self):
        r=sample();r['dx_mapilopensfm']=400
        d=label_geometry(r,[952,532],[625,625])
        self.assertFalse(d['both_pose_points_inside'])
        self.assertEqual(d['variants']['opensfm']['x'],712.5)

    def test_matching_dimensions_do_not_prove_K_chain(self):
        r=sample();r['K']=[[400,0,476],[0,400,266],[0,0,1]]
        d=label_geometry(r,[952,532],[625,625])
        self.assertTrue(d['principal_matches_actual_image_center_within1px'])
        self.assertFalse(d['K_status']['usable_calibration'])
        self.assertFalse(d['K_pixel_chain_provenance_verified'])
        d=label_geometry(r,[480,270],[625,625])
        self.assertFalse(d['principal_matches_actual_image_center_within1px'])

    def test_pose_namespaces_remain_explicit(self):
        r=sample();r['ground_latlon']=r['ground_latlon_mapilopensfm']
        with self.assertRaises(ValueError):label_geometry(r,[952,532],[625,625])

    def test_full_response_or_encoded_range_is_never_read(self):
        from project.src.tools.city_pixel_acquisition_v1 import ExactRange
        for status,headers in [(200,{}),(206,{'Content-Range':'bytes 0-7/100','Content-Encoding':'gzip'})]:
            with patch('project.src.tools.city_pixel_acquisition_v1.log'):
                fetch=ExactRange('https://example.invalid/?x=1',100)
                response=MagicMock();response.__enter__.return_value=response
                response.status=status;response.headers=headers
                fetch.opener=MagicMock();fetch.opener.open.return_value=response
                with self.assertRaises(ValueError):fetch.get(0,8)
                response.read.assert_not_called()

    def test_exact_range_and_incomplete_range_accounting(self):
        from project.src.tools.city_pixel_acquisition_v1 import ExactRange
        for chunks,success in [([b'abcd',b'efgh'],True),([b'abcd',b''],False)]:
            with patch('project.src.tools.city_pixel_acquisition_v1.log'):
                fetch=ExactRange('https://example.invalid/?x=1',100)
                response=MagicMock();response.__enter__.return_value=response
                response.status=206;response.headers={'Content-Range':'bytes 0-7/100'}
                response.read.side_effect=chunks;fetch.opener=MagicMock();fetch.opener.open.return_value=response
                if success:self.assertEqual(fetch.get(0,8),b'abcdefgh')
                else:
                    with self.assertRaises(ValueError):fetch.get(0,8)
                self.assertEqual(fetch.used,8 if success else 4)
                self.assertEqual(fetch.opener.open.call_count,1)

    def test_network_cap_rejected_before_request(self):
        from project.src.tools.city_pixel_acquisition_v1 import ExactRange
        with patch('project.src.tools.city_pixel_acquisition_v1.log'):
            fetch=ExactRange('https://example.invalid/?x=1',100);fetch.used=16*1024**2
            fetch.opener=MagicMock()
            with self.assertRaises(ValueError):fetch.get(0,1)
            fetch.opener.open.assert_not_called()


if __name__=='__main__':unittest.main()
