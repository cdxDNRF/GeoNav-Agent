"""Read the sealed consumed-area metadata object without changing frozen v1.

Only the known {regions, metadata...} container is unwrapped. The exclusion
footprints, 3km rule, cached-quality checks and candidate order are unchanged.
"""
from pathlib import Path
import importlib.util
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import area_compatibility as p


def unwrap_consumed(value):
    if not isinstance(value, dict) or not isinstance(value.get('regions'), list):
        raise ValueError('sealed consumed-area object with regions list required')
    regions = value['regions']
    if len(regions) != 14 or len({r['area'] for r in regions}) != 14:
        raise ValueError('all fourteen old consumed regions required')
    if any(len(r['bounds_m']) != 4 or len(r['sources']) != 4 for r in regions):
        raise ValueError('complete footprint/source records required')
    return regions


def run():
    p.check()
    metadata = p.prior.PREP / '元数据/已消费14连续区域.json'
    value = p.read(metadata)
    regions = unwrap_consumed(value)
    assert regions == value['regions']
    for invalid in (regions, dict(regions=regions[:13]), {}):
        try:
            unwrap_consumed(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('schema regression must reject incomplete exclusions')
    source = Path(__file__)
    snapshot = p.OUT / '源码快照/eval' / source.name
    with snapshot.open('xb') as f:
        f.write(source.read_bytes())
    registration = dict(revision=2, reason='consumed14 metadata is a sealed object containing regions, not a bare list',
        old_function_executed=False, old_frozen_code_changed=False, exclusion_or_order_changed=False,
        format_tests_passed=4, protected_input_sha256={p.rel(metadata): p.digest(metadata)},
        source_sha256={p.rel(q): p.digest(q) for q in (source, snapshot)}, consumed_regions=14)
    p.write(p.OUT / '目录格式适配登记_v2.json', registration)
    p.write(p.OUT / '目录格式适配封存_v2.json', dict(sha256=p.digest(p.OUT / '目录格式适配登记_v2.json')))
    spec = importlib.util.spec_from_file_location('eval._area_catalog_adapter_runtime', p.SRC / 'eval/area_compatibility.py')
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    original_read = core.read
    def adapted_read(path):
        parsed = original_read(path)
        return unwrap_consumed(parsed) if Path(path).resolve() == metadata.resolve() else parsed
    core.read = adapted_read
    core.feasibility()
    for mapping in (registration['source_sha256'], registration['protected_input_sha256']):
        for name, expected in mapping.items():
            assert p.digest(p.ROOT / name) == expected, name
    p.check()


if __name__ == '__main__':
    run()
