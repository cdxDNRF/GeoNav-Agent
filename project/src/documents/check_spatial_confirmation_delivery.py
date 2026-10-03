"""Verify final scientific exports and append one immutable delivery record."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import sys
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'project/src'))
from eval import spatial_area_confirmation as experiment
from eval import audit_spatial_area_confirmation_v3 as auditing
OUT = experiment.OUT
FIG = ROOT / '绘图'
MANIFEST = FIG / '绘图数据/冻结M0空间隔离导航确认_v2.json'
VISUAL = FIG / '审阅/冻结M0空间隔离导航确认视觉核验_v2.json'
FILE_QA = FIG / '审阅/冻结M0空间隔离导航确认文件核验_v2.json'
DELIVERY = OUT / '阶段交付核验.json'


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def write_once(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def check_hashes(mapping):
    for name, expected in mapping.items():
        assert digest(ROOT / name) == expected, name


def main():
    assert not DELIVERY.exists() and not FILE_QA.exists(), 'exclusive immutable delivery'
    reg = read(OUT / '预登记.json')
    experiment.check_bindings(reg)
    auditing.check_revision()
    state, verdict = read(OUT / '最终状态.json'), read(OUT / '验收结论.json')
    assert state['completed'] and verdict['completed']
    assert state['verdict_sha256'] == digest(OUT / '验收结论.json')
    assert verdict['default_sha256'] == digest(ROOT / 'project/local_policy_default.json')
    records = actions = probes = patches = 0
    for stage in experiment.STAGES:
        folder = experiment.stage_root(stage)
        audit = read(folder / '独立复核.json')
        assert audit['passed'] and audit['audit_revision'] == 3
        assert audit['audit_revision_receipt_sha256'] == digest(OUT / '审计修订登记_v3.json')
        assert audit['reset_revision_receipt_sha256'] == digest(OUT / '审计修订登记_v2.json')
        assert audit['summary_sha256'] == digest(folder / '对照汇总.json')
        assert audit['probe_summary_sha256'] == digest(folder / '探针对照/探针汇总.json')
        check_hashes(read(folder / '特征冻结结束.json')['files_sha256'])
        records += audit['records']
        actions += audit['actions']
        probes += audit['probe_records']
        patches += audit['reconstructed_native_patches']
        for path in (folder / '神经对照').glob('*结果.json'):
            assert read(path)['trajectory_sha256'] == digest(path.with_name(path.name.replace('结果.json', '轨迹.jsonl')))
        for path in (folder / '规则基线').glob('*结果.json'):
            assert read(path)['trajectory_sha256'] == digest(path.with_name(path.name.replace('结果.json', '轨迹.jsonl')))
        for path in (folder / '探针对照').glob('M0*结果.json'):
            assert read(path)['judgments_sha256'] == digest(path.with_name(path.name.replace('结果.json', '判断.jsonl')))
    assert (records, probes, patches) == (14700, 15120, 1400)
    assert not verdict['default_changed'] and verdict['new_training_steps'] == verdict['cloud_calls'] == 0
    ledger = read(OUT / '源文件模型使用状态.json')
    assert len(ledger['regions']) == 14 and all(r['navigation_model_used'] and r['cue_probe_used'] and not r['local_training_used'] for r in ledger['regions'])
    swiss = read(ROOT / '选题报告相关/SwissView源文件使用状态_2026-10-02_v2.json')
    assert swiss['navigation_evaluated_sources'] == 68 and swiss['not_yet_navigation_evaluated_sources'] == 32
    index = read(ROOT / '项目导航/实验索引.json')
    assert len(index['batches']) == len({b['path'] for b in index['batches']}) == 66
    entry = next(b for b in index['batches'] if b['path'] == rel(OUT))
    assert all((ROOT / p).is_file() for p in (*entry['reports'], *entry['evidence']))
    manifest = read(MANIFEST)
    check_hashes(manifest['source_sha256'])
    check_hashes(manifest['other_outputs_sha256'])
    assert manifest['script_sha256'] == digest(ROOT / manifest['reproduction_script'])
    original_manifest_path = FIG / '绘图数据/冻结M0空间隔离导航确认_v1.json'
    original_manifest = read(original_manifest_path)
    original_visual_path = FIG / '审阅/冻结M0空间隔离导航确认视觉核验_v1.json'
    assert read(original_visual_path)['passed'] is False
    assert manifest['layout_revision'] == 2
    for path in (original_manifest_path, original_visual_path):
        assert manifest['source_sha256'][rel(path)] == digest(path)
    for key in ('figure_data', 'confirmation_summary', 'frozen_bindings'):
        assert manifest[key] == original_manifest[key], key
    catalog = read(FIG / '图表来源清单.json')['figures']
    assert len(catalog) == len({item['id'] for item in catalog})
    details = []
    for item in manifest['figures']:
        assert next(f for f in catalog if f['id'] == item['id']) == item
        paths = {}
        for file in item['files']:
            path = ROOT / file['path']
            assert digest(path) == file['sha256']
            paths[path.suffix] = path
        with Image.open(paths['.png']) as im:
            assert min(im.info['dpi']) > 299
            size, dpi = im.size, im.info['dpi']
        tags = Counter(e.tag.rsplit('}', 1)[-1] for e in ET.parse(paths['.svg']).iter())
        assert tags['text'] > 0 and tags['image'] == 0
        with fitz.open(paths['.pdf']) as pdf:
            assert len(pdf) == 1 and not pdf[0].get_images()
            fonts = pdf[0].get_fonts(full=True)
            assert fonts and all(f[2] != 'Type3' and pdf.extract_font(f[0])[3] for f in fonts)
        details.append(dict(id=item['id'], files=item['files'], PNG_size=size, DPI=dpi,
                            PDF_pages=1, embedded_fonts=True, PDF_raster_images=0,
                            SVG_editable_texts=tags['text'], SVG_images=0))
    assert len(details) == 2
    ids = {int(p.name.split('_')[0]) for p in (FIG / '数据结果图').glob('*.png') if re.match(r'^\d{2}_', p.name)}
    assert ids == set(range(1, 47))
    visual = read(VISUAL)
    assert visual['passed'] and visual['final_plots'] == 2 and visual['PNG_and_rendered_PDF_reviewed']
    check_hashes(visual['files_sha256'])
    markdowns = [OUT / 'README_v2.md', OUT / '新空间隔离区域M0导航确认报告_v2.md',
                 ROOT / 'README.md', ROOT / '项目导航/当前进度.md', ROOT / '项目导航/实验索引.md', FIG / 'README.md',
                 FIG / '冻结M0空间隔离导航确认图注.md', FIG / '冻结M0空间隔离导航确认排版修订_v2.md']
    links = 0
    for path in markdowns:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text('utf-8')):
            link = link.strip('<>').split('#')[0]
            if not link or re.match(r'[a-z]+://', link):
                continue
            assert (path.parent / link).resolve().exists(), (str(path), link)
            links += 1
    write_once(FILE_QA, dict(passed=True, figures=details, index_entries=66, logical_result_figures=46,
                            local_links_verified=links, all_runtime_model_data_bindings_unchanged=True,
                            original_audit_failures_retained=True, audit_revision=3,
                            layout_revision=2, original_visual_QA_failure_retained=True))
    sources = [OUT / '新空间隔离区域M0导航确认报告_v2.md', OUT / 'README_v2.md', OUT / '验收结论.json',
               OUT / '最终状态.json', OUT / '源文件模型使用状态.json', MANIFEST, VISUAL, FILE_QA]
    sources.extend(experiment.stage_root(s) / '独立复核.json' for s in experiment.STAGES)
    sources.extend(ROOT / file['path'] for item in manifest['figures'] for file in item['files'])
    sources.extend(markdowns)
    sources.extend(ROOT / name for name in (
        '项目导航/实验索引.json', '绘图/图表来源清单.json',
        '项目导航/README.md', '项目导航/目录结构与维护.md',
        'project/src/documents/index_spatial_confirmation.py'))
    delivery = dict(completed=True, engineering_interface_passed=True,
                    spatial_unknown_area_navigation_passed=verdict['spatial_unknown_area_navigation_passed'],
                    seam_head_reliability_passed=verdict['seam_head_reliability_passed'],
                    combined_passed=verdict['spatial_area_combined_passed'],
                    navigation_records=records, replayed_actions=actions, probe_judgments=probes,
                    reextracted_patches=patches, regions=14, engineering_regions=4, confirmation_regions=10,
                    area_km2_each=9, cell_m=300, tests_before_registration=40, audit_repair_tests=6,
                    audit_revision=3, original_failures_and_sources_preserved=True,
                    default_changed=False, new_training_steps=0, cloud_calls=0, SwissView_registry_unchanged=True,
                    original_data_and_models_unchanged=True, confirmation_regions_consumed=10,
                    pretrained_encoder_unknown_geography_verified=False,
                    new_scientific_figures=2, figure_export_and_visual_QA_passed=True,
                    layout_revision=2, original_visual_QA_failure_retained=True,
                    index_entries=66, logical_result_figures=46,
                    source_sha256={rel(p): digest(p) for p in sources}, check_script_sha256=digest(Path(__file__)))
    write_once(DELIVERY, delivery)
    print({key: value for key, value in delivery.items() if key != 'source_sha256'}, flush=True)


if __name__ == '__main__':
    main()
