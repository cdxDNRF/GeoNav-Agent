"""Synthetic-only CV028 acquisition/coordinate failure boundaries."""
import ast
import io
from pathlib import Path
import struct
import unittest
import zipfile
import zlib
from project.src.tools import opencvl_sample_preflight_v1 as p


def example_zip(name='safe/labels.json',body=b'{"dx":1}'):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:z.writestr(name,body)
    data=stream.getvalue();e=struct.unpack('<4s4H2LH',data[-22:]);directory=data[e[6]:e[6]+e[5]]
    return data,directory


class Tests(unittest.TestCase):
    def test_central_directory_and_payload_crc(self):
        data,cd=example_zip();m=p.parse_directory(cd)[0]
        a=struct.unpack_from('<4s5H3I2H',data,m['local_offset']);start=m['local_offset']+30+a[9]+a[10]
        body=p.unpack_payload(data[start:start+m['compressed_bytes']],m,1024)
        self.assertEqual(body,b'{"dx":1}')
        with self.assertRaises(ValueError):p.unpack_payload(b'wrong',m,1024)

    def test_portable_rename_does_not_reject_upstream_timestamp(self):
        _,cd=example_zip('safe/2022-05-18T07:05:25.png')
        self.assertIn(':',p.parse_directory(cd)[0]['name'])

    def test_traversal_absolute_and_windows_drive_rejected(self):
        for name in ('../escape.png','/escape.png','C:/escape.png','safe/../../escape','safe\\escape'):
            with self.assertRaises(ValueError):p.archive_name(name)

    def test_truncated_directory_rejected(self):
        _,cd=example_zip()
        with self.assertRaises(ValueError):p.parse_directory(cd[:-1])
        with self.assertRaises(ValueError):p.parse_directory(b'not a central header')

    def test_range_identity_fails_before_read(self):
        p.validate_range(206,'bytes 10-19/100',10,10,100)
        with self.assertRaises(ValueError):p.validate_range(200,None,10,10,100)
        with self.assertRaises(ValueError):p.validate_range(206,'bytes 0-9/100',10,10,100)
        with self.assertRaises(ValueError):p.validate_range(206,'bytes 10-19/101',10,10,100)

    def test_zip_bomb_budget_and_crc_rejected(self):
        data=b'a'*1000;comp=zlib.compress(data)[2:-4]
        m=dict(flags=0,compression=8,bytes=1000,compressed_bytes=len(comp),crc32=f'{zlib.crc32(data):08x}')
        with self.assertRaises(ValueError):p.unpack_payload(comp,m,100)
        with self.assertRaises(ValueError):p.unpack_payload(comp,dict(m,crc32='00000000'),2000)

    def test_north_east_geographic_signs(self):
        east,north=p.offset_en([60.00001,10.00001],[60,10])
        self.assertGreater(east,0);self.assertGreater(north,0)
        self.assertEqual(p.geo_distance([60,10],[60,10]),0)

    def test_odd_pixel_grid_covers_parent_once(self):
        rects=p.local_rectangles(625,625)
        self.assertEqual(len(rects),25)
        self.assertEqual(sum((r-l)*(b-t) for l,t,r,b in rects),625**2)
        for x,y in ((0,0),(124.999,124.999),(125,125),(312.5,312.5),(624.999,624.999)):
            self.assertEqual(sum(l<=x<r and t<=y<b for l,t,r,b in rects),1)
        self.assertEqual(sum(l<=625<r and t<=625<b for l,t,r,b in rects),0)

    def test_no_model_or_navigation_import(self):
        tree=ast.parse(Path(p.__file__).read_text(encoding='utf-8'));mods=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):mods.extend(a.name for a in n.names)
            if isinstance(n,ast.ImportFrom):mods.append(n.module)
        self.assertFalse(any(m.split('.')[0] in {'torch','transformers','timm','ollama'} or '.agents.' in m for m in mods))


if __name__=='__main__':unittest.main()
