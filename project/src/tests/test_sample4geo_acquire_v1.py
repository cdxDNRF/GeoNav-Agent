import io
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from project.src.tools.sample4geo_acquire_v1 import (
    MEMBER, checked_range, directory_location, select_member, data_start, unpack_member,
)


class AcquisitionTests(unittest.TestCase):
    def archive(self, filename=MEMBER, duplicate=False):
        b = io.BytesIO()
        with zipfile.ZipFile(b, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr(filename, b'known-checkpoint-synthetic'*200)
            if duplicate:
                z.writestr(filename, b'other')
        return b.getvalue()

    def test_ranges_fail_closed(self):
        for status, headers in [(200, {}), (206, {'Content-Range': 'bytes 0-9/20'}),
                                (206, {'Content-Range': 'bytes 0-9/10', 'Content-Length': '11'}),
                                (206, {'Content-Range': 'bytes 0-9/10', 'Content-Encoding': 'gzip'})]:
            with self.assertRaises(ValueError):
                checked_range(status, headers, 0, 9, 10)
        checked_range(206, {'Content-Range': 'bytes 0-9/10', 'Content-Length': '10'}, 0, 9, 10)

    def test_exact_directory_and_offsets(self):
        raw = self.archive()
        off, size, end = directory_location(raw, len(raw))
        x = select_member(raw[off:off+size], end)
        begin, nl, el = data_start(raw[x.header_offset:x.header_offset+30], x)
        self.assertEqual(raw[begin-nl-el:begin-el].decode(), MEMBER)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            self.assertEqual(x.CRC, z.getinfo(MEMBER).CRC)

    def test_wrong_or_duplicate_member(self):
        for raw in [self.archive('../wrong.pth'), self.archive(duplicate=True)]:
            off, size, end = directory_location(raw, len(raw))
            with self.assertRaises(ValueError):
                select_member(raw[off:off+size], end)

    def test_eocd_rejects_trailing_and_spanning(self):
        raw = self.archive()
        with self.assertRaises(ValueError):
            directory_location(raw+b'x', len(raw)+1)
        b = bytearray(raw)
        struct.pack_into('<H', b, len(b)-18, 1)
        with self.assertRaises(ValueError):
            directory_location(b, len(b))

    def test_cap_and_encryption(self):
        raw = self.archive()
        off, size, end = directory_location(raw, len(raw))
        with patch('project.src.tools.sample4geo_acquire_v1.CAP', 1):
            with self.assertRaises(ValueError):
                select_member(raw[off:off+size], end)
        b = bytearray(raw[off:off+size])
        struct.pack_into('<H', b, 8, 1)
        with self.assertRaises(ValueError):
            select_member(b, end)

    def test_stream_crc_and_no_overwrite(self):
        raw = self.archive()
        off, size, end = directory_location(raw, len(raw))
        x = select_member(raw[off:off+size], end)
        begin, _, _ = data_start(raw[x.header_offset:x.header_offset+30], x)
        root = Path(__file__).resolve().parents[3]/'DATA/processed_data/MM-GAG/单权重取得与工程_v1'
        with tempfile.TemporaryDirectory(prefix='test_tmp-', dir=root) as td:
            compressed = Path(td)/'part'
            target = Path(td)/'checkpoint'
            compressed.write_bytes(raw[begin:begin+x.compress_size])
            result = unpack_member(compressed, target, x)
            self.assertEqual(result['bytes'], x.file_size)
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                self.assertEqual(target.read_bytes(), z.read(MEMBER))
            with self.assertRaises(FileExistsError):
                unpack_member(compressed, target, x)
            x.CRC ^= 1
            with self.assertRaises(ValueError):
                unpack_member(compressed, Path(td)/'bad', x)


if __name__ == '__main__':
    unittest.main()
