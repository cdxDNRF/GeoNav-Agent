"""In-memory intake failure/boundary cases; no temp directories or image decode."""
import copy
import hashlib
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
import zlib
from project.src.data.capturepoint_intake_v1 import safe_relative,image_header,LocalAssets,camera_geometry,inspect_records
from project.src.tools.inspect_capturepoint_intake_v1 import official_records,normalize_official_labels


def png(w=100,h=100):
    data=struct.pack('>IIBBBBB',w,h,8,2,0,0,0);chunk=b'IHDR'+data
    return b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+chunk+struct.pack('>I',zlib.crc32(chunk)&0xffffffff)


def record(key='a',role='fit_preparation',lat=52.):
    return dict(record_id=key,source_key='mapillary:'+key,role=role,profile='image_only',pose_variant='opensfm',
        corrected_pose_available=True,dx=0,dy=0,resolution=1.,camera_latlon=[lat,5.],
        ground=dict(path=key+'_g.png'),aerial=dict(path=key+'_a.png'),sequence_key=None)


class MemoryAssets:
    def __init__(self):self.calls=0
    def __call__(self,asset):
        self.calls+=1;path=safe_relative(asset['path']);digest=hashlib.sha256(path.encode()).hexdigest()
        if asset.get('sha256',digest)!=digest:raise ValueError('SHA256 mismatch')
        return dict(path=path,sha256=digest,width=100,height=100,bytes=33,pixels_decoded=False)


def known(row,**kwargs):
    return dict(source_key=row['source_key'],camera_latlon=row['camera_latlon'],radius_m=71.,image_hashes=[],sequence_key=None,**kwargs)


class IntakeTests(unittest.TestCase):
    def test_relative_paths_and_video_exclusion(self):
        self.assertEqual(safe_relative('images/1.png'),'images/1.png')
        for path in ['../1.png','C:/1.png','/1.png','a\\1.png','视频/1.png','.env','https://x/1.png']:
            with self.assertRaises(ValueError):safe_relative(path)

    def test_png_header_crc_dimensions_no_decode(self):
        self.assertEqual(image_header(png(103,107))['width'],103)
        self.assertFalse(image_header(png())['pixels_decoded'])
        damaged=bytearray(png());damaged[-1]^=1
        with self.assertRaises(ValueError):image_header(damaged)
        with self.assertRaises(ValueError):image_header(png(5000,100))

    def test_jpeg_header_and_truncation(self):
        sof=b'\x08'+struct.pack('>HH',480,640)+b'\x01\x01\x11\x00'
        data=b'\xff\xd8\xff\xc0'+struct.pack('>H',len(sof)+2)+sof
        self.assertEqual((image_header(data)['width'],image_header(data)['height']),(640,480))
        with self.assertRaises(ValueError):image_header(data[:-2])

    def test_native_half_open_and_no_raw_fallback(self):
        r=record();r['dx']=50
        g=camera_geometry(r,100,100);self.assertFalse(g['inside_reference']);self.assertEqual(g['x'],100.)
        r['corrected_pose_available']=False
        with self.assertRaises(ValueError):camera_geometry(r,100,100)

    def test_known_confirmation_rejected_before_asset_read(self):
        r=record(role='confirmation');loader=MemoryAssets();out=inspect_records([r],[known(r,role='development')],loader)
        self.assertFalse(out['data_preparation_ready']);self.assertEqual(loader.calls,0)

    def test_protected_identity_rejected_before_asset_read(self):
        r=record();loader=MemoryAssets();out=inspect_records([r],[known(r,protected=True)],loader)
        self.assertFalse(out['local_checks_passed']);self.assertEqual(loader.calls,0)

    def test_frozen_development_not_silently_reassigned(self):
        r=record();loader=MemoryAssets();out=inspect_records([r],[known(r,role='development')],loader)
        self.assertFalse(out['data_preparation_ready']);self.assertEqual(loader.calls,0)

    def test_duplicate_image_and_sequence_link_split_groups(self):
        a=record();b=record('b','development',53.);b['ground']=copy.deepcopy(a['ground'])
        out=inspect_records([a,b],[],MemoryAssets());self.assertEqual(out['spatial_groups'],1);self.assertFalse(out['role_isolation_ok'])
        b['ground']['path']='other.png';a['sequence_key']=b['sequence_key']='drive-1'
        out=inspect_records([a,b],[],MemoryAssets());self.assertFalse(out['role_isolation_ok'])

    def test_geo_chain_closure_not_only_direct_endpoints(self):
        rows=[record('a',lat=52.),record('b',lat=52.004),record('c','development',52.008)]
        out=inspect_records(rows,[],MemoryAssets());self.assertEqual(out['spatial_groups'],1);self.assertFalse(out['role_isolation_ok'])

    def test_nearby_renamed_source_cannot_be_new_confirmation(self):
        old=record('old');new=record('renamed','confirmation',52.001)
        out=inspect_records([new],[known(old)],MemoryAssets());self.assertFalse(out['data_preparation_ready'])

    def test_outside_is_explicit_negative_and_permissions_stay_false(self):
        r=record();r['dx']=-200;out=inspect_records([r],[],MemoryAssets())
        self.assertTrue(out['data_preparation_ready']);self.assertEqual(out['outside_reference_records'],1)
        self.assertEqual(out['in_window_records'],0);self.assertFalse(out['training_authorized']);self.assertFalse(out['formal_confirmation_admitted'])

    def test_local_asset_cached_expected_hash_still_checked(self):
        assets=LocalAssets(Path.cwd());p=(Path.cwd()/'not-created.png').resolve()
        assets.cache[p]=dict(path='not-created.png',sha256='0'*64,width=100,height=100,bytes=33)
        with self.assertRaises(ValueError):assets(dict(path='not-created.png',sha256='1'*64))
        self.assertEqual(assets.used,0)

    def test_missing_or_bad_asset_is_not_silently_skipped(self):
        r=record();r['ground']['sha256']='wrong';out=inspect_records([r],[],MemoryAssets())
        self.assertEqual(out['record_count'],1);self.assertFalse(out['data_preparation_ready']);self.assertTrue(out['records'][0]['errors'])

    def test_unsupported_profile_duplicate_and_record_caps(self):
        r=record();r['profile']='calibrated_photo';self.assertFalse(inspect_records([r],[],MemoryAssets())['local_checks_passed'])
        with self.assertRaises(ValueError):inspect_records([record(),record()],[],MemoryAssets())
        with self.assertRaises(ValueError):inspect_records([record(str(i)) for i in range(65)],[],MemoryAssets())

    def test_official_wrappers_and_ambiguous_ground_fields(self):
        self.assertEqual(official_records({'labels':[{}]}),[{}])
        with self.assertRaises(ValueError):official_records({'labels':[],'data':[]})
        root=Path.cwd()/'virtual-not-created';path=root/'mapillary/cities/NL/demo/labels.json'
        row=dict(mapil_image_id='123',mapil_image='ground/123.png',aerial_image='aerial/123.png',
            pose_mapilopensfm_available=True,dx_mapilopensfm=0,dy_mapilopensfm=0,resolution=.1,ground_latlon_mapilopensfm=[52.,5.])
        with patch.object(Path,'is_file',return_value=False):
            out=normalize_official_labels([row],path,root,'development',['123']);self.assertEqual(len(out),1)
            self.assertEqual(out[0]['ground']['path'],'mapillary/cities/NL/demo/ground/123.png')
            row['ground_image']='different.png'
            with self.assertRaises(ValueError):normalize_official_labels([row],path,root,'development')

    def test_incomplete_selection_and_unavailable_pose_rejected(self):
        root=Path.cwd()/'virtual-not-created';path=root/'labels.json'
        row=dict(mapil_image_id='123',mapil_image='g.png',aerial_image='a.png',pose_mapilopensfm_available=False)
        with self.assertRaises(ValueError):normalize_official_labels([row],path,root,'development')
        with self.assertRaises(ValueError):normalize_official_labels([],path,root,'development',['456'])


if __name__=='__main__':unittest.main()
