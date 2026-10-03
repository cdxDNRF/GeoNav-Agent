"""Resource extension must preserve v1 and reuse old files without writing them."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import tempfile
import unittest
from data import spatial_continuous as original
from data import spatial_continuous_v2 as revised

ROOT = Path(__file__).resolve().parents[3]


class ResourceRevisionTests(unittest.TestCase):
    def test_original_runtime_not_mutated(self):
        self.assertEqual(original.FILE_LIMIT, 120)
        self.assertEqual(original.BYTE_LIMIT, 1024**3)
        self.assertEqual(revised.FILE_LIMIT, 200)
        self.assertEqual(revised.BYTE_LIMIT, 3*1024**3//2)
        self.assertEqual(original.OUT.name, '空间隔离连续区域_v1')
        self.assertEqual(revised.OUT.name, '空间隔离连续区域_v2')

    def test_same_frozen_geometry_quality_and_sampler_bodies(self):
        for name in ('rectangle_gap', 'clear_of', 'filename_candidates', 'tasks', 'probes', 'wrong_plan'):
            a, b = getattr(original, name), getattr(revised._core, name)
            self.assertEqual(a.__code__.co_code, b.__code__.co_code)
            self.assertEqual(a.__code__.co_consts, b.__code__.co_consts)
        self.assertEqual(original.GAP, revised._core.GAP)
        self.assertEqual(original.TOL, revised._core.TOL)

    def test_saved_source_cache_readonly_and_sha_checked(self):
        parent = (ROOT / '选题报告相关').resolve()
        temp = tempfile.TemporaryDirectory(prefix='空间隔离区域测试临时_', dir=parent)
        folder = Path(temp.name).resolve()
        try:
            p = folder / 'source.tiff'
            p.write_bytes(b'synthetic-cache-identity')
            source = dict(id='fixture', raw_path=p.relative_to(ROOT).as_posix(), raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            record = dict(name='fixture.tiff', status='completed', received_bytes=22, source=source)
            (folder / '下载记录.jsonl').write_text(json.dumps(record)+'\n', encoding='utf-8')
            with patch.object(revised._core, 'RAW', folder):
                acquisition = revised._core.Acquisition(dict(urls={}))
                before = p.read_bytes()
                self.assertEqual(acquisition.fetch('fixture.tiff'), source)
                self.assertEqual(p.read_bytes(), before)
                p.write_bytes(b'changed')
                with self.assertRaises(ValueError): revised._core.Acquisition(dict(urls={}))
        finally:
            target = folder.resolve()
            if target.parent != parent or not target.name.startswith('空间隔离区域测试临时_'):
                raise ValueError('unsafe cache-fixture cleanup')
            temp.cleanup()


if __name__ == '__main__': unittest.main()
