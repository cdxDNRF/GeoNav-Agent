"""Resource-only extension of frozen v1, in a separate runtime namespace.

The v1 source, raw120 files and stop judgment remain unchanged. Geometry,
quality and samplers use the exact frozen implementation. Only output paths,
document bindings and the total resource caps change. Old raw bytes are reused.
"""
from pathlib import Path
import argparse
import importlib.util
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'project/src'
spec = importlib.util.spec_from_file_location('data._spatial_resource_v2_runtime', SRC / 'data/spatial_continuous.py')
_core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _core
spec.loader.exec_module(_core)

RAW1, OUT1 = _core.RAW, _core.OUT
_core.RAW = ROOT / 'DATA/raw_data/MasaRoads/空间隔离连续区域_v2'
_core.OUT = ROOT / 'DATA/processed_data/MasaRoads/空间隔离连续区域_v2'
_core.META = _core.OUT / '元数据'
_core.QA = _core.OUT / '核验'
_core.DATA = _core.OUT / '工程数据'
_core.DOC = ROOT / '选题报告相关/空间隔离连续区域准备资源修订方案_v2.md'
_core.TESTS = ROOT / '选题报告相关/空间隔离连续区域准备测试_v2.json'
_core.FILE_LIMIT = 200
_core.BYTE_LIMIT = 3*1024**3//2
_core.CODE += ('data/spatial_continuous_v2.py', 'eval/audit_spatial_area_data_v2.py', 'tests/test_spatial_area_v2_resources.py')

RAW, OUT, META, QA, DATA = _core.RAW, _core.OUT, _core.META, _core.QA, _core.DATA
DOC, TESTS, FILE_LIMIT, BYTE_LIMIT = _core.DOC, _core.TESTS, _core.FILE_LIMIT, _core.BYTE_LIMIT
read, write, digest, rel = _core.read, _core.write, _core.digest, _core.rel
check_bindings = _core.check_bindings
rectangle_gap, clear_of, tasks, probes, wrong_plan = _core.rectangle_gap, _core.clear_of, _core.tasks, _core.probes, _core.wrong_plan


def check_reuse():
    frozen = read(META / 'v1只读复用冻结.json')
    binding = read(QA / '复用登记绑定.json')
    if digest(META / 'v1只读复用冻结.json') != binding['reuse_manifest_sha256']:
        raise ValueError('reuse manifest changed')
    for n, h in frozen['files_sha256'].items():
        if digest(ROOT / n) != h:
            raise ValueError('v1 readonly reuse drift: ' + n)


def prepare():
    stop = read(OUT1 / '核验/停止状态.json')
    if stop['qualified_regions'] != 12 or stop['downloaded_sources'] != 120 or not stop['resource_limit_reached']:
        raise ValueError('recorded v1 resource-only stop required')
    oldfrozen = read(OUT1 / '核验/停止输入冻结.json')
    for n, h in oldfrozen['files_sha256'].items():
        if digest(ROOT / n) != h:
            raise ValueError('v1 frozen stop artifact changed')
    _core.prepare()
    # This is a small log copy, not a copy or move of the large imagery.
    (RAW / '下载记录.jsonl').write_bytes((RAW1 / '下载记录.jsonl').read_bytes())
    files = dict(oldfrozen['files_sha256'])
    files[rel(OUT1 / '核验/停止输入冻结.json')] = digest(OUT1 / '核验/停止输入冻结.json')
    write(META / 'v1只读复用冻结.json', dict(files_sha256=files, reused_sources=120, old_received_bytes=stop['received_bytes'],
          original_v1_data_engineering_passed=False, geometry_or_quality_changed=False, model_score_selection=False,
          total_file_cap=FILE_LIMIT, total_network_byte_cap=BYTE_LIMIT, old_raw_files_copied=False, old_raw_files_moved=False))
    write(QA / '复用登记绑定.json', dict(reuse_manifest_sha256=digest(META / 'v1只读复用冻结.json'),
          old_stop_sha256=digest(OUT1 / '核验/停止状态.json'), resource_revision_only=True))
    check_reuse()


def select():
    check_reuse()
    _core.select()
    check_reuse()


def build():
    check_reuse()
    _core.build()
    check_reuse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'select', 'build'))
    {'prepare': prepare, 'select': select, 'build': build}[parser.parse_args().mode]()
