"""Verify prepared data and figure bindings, then append one delivery record.

This script never invokes a navigation policy, retrains models, fetches sources,
or rewrites earlier artifacts. Visual review is recorded separately after the
actual PNG, rendered PDF and fourteen-region contact sheet have been inspected.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'DATA/processed_data/MasaRoads/空间隔离连续区域_v2'
OLD = ROOT / 'DATA/processed_data/MasaRoads/空间隔离连续区域_v1'
FIG = ROOT / '绘图'
VISUAL = FIG / '审阅/空间隔离连续区域视觉核验_v2.json'
FILE_QA = FIG / '审阅/空间隔离连续区域文件核验_v2.json'
DELIVERY = RUN / '核验/阶段交付核验.json'


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def write_once(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def verify_hashes(mapping):
    for name, expected in mapping.items():
        assert digest(ROOT / name) == expected, name


def main():
    assert not FILE_QA.exists() and not DELIVERY.exists(), 'immutable delivery already exists'
    assert VISUAL.is_file(), 'inspect images and append visual review before delivery'
    reg = read(RUN / '核验/预登记.json')
    for field in ('protected_sha256', 'source_sha256', 'frozen_metadata_sha256'):
        verify_hashes(reg[field])
    assert len(reg['protected_sha256']) == 986
    assert reg['protocol_sha256'] == digest(RUN / '执行协议.md')
    assert reg['default_sha256'] == digest(ROOT / 'project/local_policy_default.json')
    frozen = read(RUN / '核验/输入冻结结束.json')
    verify_hashes(frozen['files_sha256'])
    assert frozen['completed'] and len(frozen['files_sha256']) == 1476
    reuse = read(RUN / '元数据/v1只读复用冻结.json')
    binding = read(RUN / '核验/复用登记绑定.json')
    verify_hashes(reuse['files_sha256'])
    assert len(reuse['files_sha256']) == 144 and reuse['reused_sources'] == 120
    assert binding['reuse_manifest_sha256'] == digest(RUN / '元数据/v1只读复用冻结.json')
    assert binding['old_stop_sha256'] == digest(OLD / '核验/停止状态.json')
    assert binding['resource_revision_only'] and not reuse['geometry_or_quality_changed']
    assert not reuse['old_raw_files_copied'] and not reuse['old_raw_files_moved']
    assert not reuse['original_v1_data_engineering_passed'] and not reuse['model_score_selection']

    verdict = read(RUN / '核验/验收结论.json')
    audit = read(RUN / '核验/独立复核.json')
    state = read(RUN / '核验/最终状态.json')
    tests = read(RUN / '测试记录.json')
    assert tests['successful'] and tests['tests_run'] == 26 and not tests['failures'] and not tests['errors']
    assert verdict['audit_sha256'] == state['audit_sha256'] == digest(RUN / '核验/独立复核.json')
    assert verdict['raw_and_derived_input_freeze_sha256'] == digest(RUN / '核验/输入冻结结束.json')
    assert state['report_sha256'] == digest(RUN / '空间隔离连续区域数据准备报告.md')
    assert state['verdict_sha256'] == digest(RUN / '核验/验收结论.json')
    assert audit['passed'] and state['completed'] and verdict['data_engineering_passed']
    assert verdict['spatial_isolation_from_localM0Masa151_passed'] and verdict['dev_test_region_gap_passed']
    assert audit['GeoTIFF_tags_and_pixels_independently_checked']
    assert audit['full_footprint_spatial_buffers_verified'] and audit['original_v1_inputs_unchanged']
    assert audit['protected_files_verified'] == 986 and audit['input_files_verified'] == 1476
    assert not verdict['unknown_area_navigation_passed'] and not verdict['seam_recognition_model_validated']
    assert not audit['pretrained_encoder_unknown_geography_verified']
    for item, fields in (
        (state, ('navigation_evaluation_started', 'new_SR_produced', 'navigation_model_calls',
                 'cloud_calls', 'new_training_steps', 'default_changed')),
        (audit, ('navigation_model_calls', 'new_navigation_records', 'new_SR_produced',
                 'cloud_calls', 'new_training_steps', 'default_changed')),
        (frozen, ('navigation_model_calls', 'new_navigation_records', 'training_steps', 'cloud_calls')),
    ):
        assert all(item[field] == 0 for field in fields), fields
    assert not any(RUN.rglob('*.pt')) and not any(RUN.rglob('*轨迹.jsonl'))

    manifest = read(RUN / '工程数据/数据清单.json')
    regions = manifest['regions']
    assert manifest['protocol'] == reg['protocol'] == 'masa-roads-grid10-spatial-v1'
    assert manifest['epsg'] == 26986 and manifest['grid_size'] == 10 and manifest['budget'] == 20
    assert manifest['pixel_size_m'] == 1 and manifest['cell_size_m'] == 300
    assert len(regions) == len({r['area'] for r in regions}) == 14
    assert Counter(r['split'] for r in regions) == {'dev': 4, 'test': 10}
    assert all(r['projected_area_km2'] == 9 for r in regions)
    assert len({s['id'] for r in regions for s in r['sources']}) == 56
    assert all(r['project_role'] == ('engineering' if r['split'] == 'dev' else 'frozen_confirmation') for r in regions)
    ledger = read(RUN / '元数据/本项目模型使用状态.json')
    assert ledger['default_sha256'] == reg['default_sha256']
    assert [(r['area'], r['split'], r['source_ids']) for r in ledger['regions']] == [
        (r['area'], r['split'], [s['id'] for s in r['sources']]) for r in regions]
    assert all(not r[f] for r in ledger['regions'] for f in
               ('navigation_model_used', 'cue_probe_used', 'local_training_used'))
    source_selection = read(RUN / '元数据/连续区域选择.json')
    assert source_selection['source_quality_selection_only'] and not source_selection['selection_by_model_score']
    assert len(source_selection['rejected_before_selection']) == 42
    assert source_selection['complete_isolated_filename_candidates'] == 507
    raw1 = ROOT / 'DATA/raw_data/MasaRoads/空间隔离连续区域_v1/tiff'
    raw2 = ROOT / 'DATA/raw_data/MasaRoads/空间隔离连续区域_v2/tiff'
    assert len(list(raw1.glob('*.tiff'))) == 120 and len(list(raw2.glob('*.tiff'))) == 32
    assert source_selection['downloaded_sources'] == audit['all_downloaded_sources'] == state['all_downloaded_sources'] == 152
    assert source_selection['received_bytes'] == audit['network_received_bytes'] == 1053215224
    assert audit['all_downloaded_sources'] <= reg['download_file_limit'] == 200
    assert audit['network_received_bytes'] <= reg['download_byte_limit'] == 1610612736
    assert math.isclose(audit['minimum_old_region_gap_m'], 50289.163842783404, abs_tol=1e-6)
    assert audit['minimum_new_region_pair_gap_m'] >= reg['min_region_gap_m'] - reg['coordinate_tolerance_m']

    tasks = read(RUN / '工程数据/导航任务.json')
    strata = read(RUN / '工程数据/任务分层.json')
    probes = read(RUN / '工程数据/邻接诊断探针.json')
    wrong = read(RUN / '工程数据/错误目标计划.json')
    assert len(tasks) == len({t['episode_id'] for t in tasks}) == len({(t['area'], t['start'], t['goal']) for t in tasks}) == 1050
    assert len(probes) == len({p['probe_id'] for p in probes}) == 1680
    assert set(strata) == set(wrong) == {t['episode_id'] for t in tasks}
    assert Counter(t['split'] for t in tasks) == {'dev': 300, 'test': 750}
    assert Counter(p['split'] for p in probes) == {'dev': 480, 'test': 1200}
    assert Counter((t['area'], strata[t['episode_id']]['stratum']) for t in tasks) == {
        (r['area'], name): 25 for r in regions for name in ('long_distance', 'seam_target', 'interior_target')}
    assert all(t['budget'] == 20 and t['grid_size'] == 10 and t['protocol'] == reg['protocol'] for t in tasks)
    assert all(p['evaluation_only'] and not p['counts_as_navigation'] for p in probes)
    confirm_cross = [p for p in probes if p['split'] == 'test' and p['kind'] == 'cross_source_adjacent']
    assert len(confirm_cross) == len({(p['area'], p['current_cell'], p['target_cell']) for p in confirm_cross}) == 400
    assert sum(t['split'] == 'test' and strata[t['episode_id']]['stratum'] == 'seam_target' for t in tasks) == 250
    assert sum(not item['matched_distance'] for item in wrong.values()) == audit['wrong_distance_exceptions'] == 69
    patches = list((RUN / '工程数据/patches').rglob('patch_*.jpg'))
    assert len(patches) == audit['exact_patches'] == state['derived_patches'] == 1400
    assert audit['reconstructed_mosaics'] == len(list((RUN / '工程数据/mosaics').glob('*.tiff'))) == 14
    swiss = read(ROOT / '选题报告相关/SwissView源文件使用状态_2026-10-02_v2.json')
    assert swiss['navigation_evaluated_sources'] == 68 and swiss['not_yet_navigation_evaluated_sources'] == 32

    index = read(ROOT / '项目导航/实验索引.json')
    assert len(index['batches']) == len({b['path'] for b in index['batches']}) == 65
    for folder in (OLD, RUN):
        entry = next(b for b in index['batches'] if b['path'] == rel(folder))
        assert all((ROOT / p).is_file() for p in (*entry['reports'], *entry['evidence']))
    figures = read(FIG / '绘图数据/空间隔离连续区域准备_最终_v2.json')
    verify_hashes(figures['source_sha256'])
    assert figures['script_sha256'] == digest(ROOT / 'project/src/documents/summarize_spatial_area_data.py')
    assert figures['script_sha256'] == digest(FIG / '审阅/空间隔离区域作图源码_v2.py')
    assert figures['refinement_script_sha256'] == digest(ROOT / 'project/src/documents/refine_spatial_coverage_figure.py')
    assert figures['refinement_script_sha256'] == digest(FIG / '审阅/空间隔离探针角色标注修订_v2.py')
    assert figures['initial_manifest_sha256'] == digest(FIG / '绘图数据/空间隔离连续区域准备_v2.json')
    catalog = read(FIG / '图表来源清单.json')
    assert len(catalog['figures']) == len({f['id'] for f in catalog['figures']})
    result_ids = {int(p.name.split('_')[0]) for p in (FIG / '数据结果图').glob('*.png') if re.match(r'^\d{2}_', p.name)}
    assert result_ids == set(range(1, 45))
    details = []
    for item in figures['figures']:
        assert next(f for f in catalog['figures'] if f['id'] == item['id']) == item
        assert all((ROOT / p).is_file() for p in item['sources'])
        assert all(digest(ROOT / f['path']) == f['sha256'] for f in item['files'])
        paths = {Path(f['path']).suffix: ROOT / f['path'] for f in item['files']}
        assert set(paths) == {'.png', '.pdf', '.svg'}
        with Image.open(paths['.png']) as im:
            assert min(im.info['dpi']) > 299
            size, dpi = im.size, im.info['dpi']
        tags = Counter(e.tag.rsplit('}', 1)[-1] for e in ET.parse(paths['.svg']).iter())
        assert tags['text'] > 0 and tags['image'] == 0
        with fitz.open(paths['.pdf']) as pdf:
            assert len(pdf) == 1 and not pdf[0].get_images()
            fonts = pdf[0].get_fonts(full=True)
            assert fonts and all(f[2] != 'Type3' and pdf.extract_font(f[0])[3] for f in fonts)
            text = pdf[0].get_text()
            if item['id'].startswith('43_'):
                assert all(label in text for label in ('50.29km', '3km', '1毫米容差', '不证明编码器预训练未见'))
            else:
                assert all(label in text for label in ('工程', '确认', '160', '400', '1680', '没有模型判断或新SR'))
            details.append(dict(id=item['id'], files=item['files'], PNG_size=size, DPI=dpi,
                                PDF_pages=1, embedded_fonts=True, PDF_raster_images=0,
                                SVG_editable_texts=tags['text'], SVG_images=0))
    assert len(details) == 2
    visual = read(VISUAL)
    verify_hashes(visual['files_sha256'])
    verify_hashes(visual['preview_sources_sha256'])
    assert visual['passed'] and visual['final_plots'] == 2 and visual['PNG_and_rendered_PDF_reviewed']
    assert visual['fourteen_region_contact_sheet_reviewed'] and visual['preview_maps'] == 14
    assert visual['bar_engineering_confirmation_roles_explicit'] and visual['no_apparent_label_clipping']
    assert visual['data_preparation_coverage_only'] and not visual['SR_or_recognition_claimed']

    markdowns = [RUN / '空间隔离连续区域数据准备报告.md', RUN / '下一项冻结M0空间隔离验证草案.md',
                 RUN / 'README.md', ROOT / 'README.md', ROOT / '项目导航/当前进度.md',
                 ROOT / '项目导航/实验索引.md', FIG / 'README.md', FIG / '空间隔离连续区域图注_v2.md']
    links_checked = 0
    for path in markdowns:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text('utf-8')):
            link = link.strip('<>').split('#')[0]
            if not link or re.match(r'[a-z]+://', link):
                continue
            assert (path.parent / link).resolve().exists(), (str(path), link)
            links_checked += 1
    fileqa = dict(passed=True, figures=details, initial_fig44_exports_retained=True,
                  new_scientific_figures=2, index_entries=65, catalog_entries=len(catalog['figures']),
                  legacy_top_catalog_partial=True, logical_result_figures=len(result_ids),
                  local_links_verified=links_checked, input_and_protected_hashes_verified=True,
                  preview_maps=14, model_inference_performed=False)
    write_once(FILE_QA, fileqa)
    sources = [RUN / name for name in (
        '空间隔离连续区域数据准备报告.md', '下一项冻结M0空间隔离验证草案.md',
        '核验/预登记.json', '核验/独立复核.json', '核验/验收结论.json', '核验/最终状态.json',
        '核验/输入冻结结束.json', '核验/复用登记绑定.json', '元数据/本项目模型使用状态.json',
        '元数据/v1只读复用冻结.json', '测试记录.json')]
    sources.extend((FIG / '绘图数据/空间隔离连续区域准备_最终_v2.json', FILE_QA, VISUAL))
    sources.extend(ROOT / f['path'] for item in figures['figures'] for f in item['files'])
    sources.extend(markdowns)
    final = dict(completed=True, data_engineering_passed=True, spatial_isolation_from_localM0Masa151_passed=True,
                 original_v1_preserved=True, v2_resource_revision_only=True, old_raw_files_readonly_reused=120,
                 newly_downloaded_sources=32, selected_raw_sources=56, total_downloaded_sources=152,
                 network_received_bytes=1053215224, isolated_regions=14, engineering_regions=4,
                 confirmation_regions=10, area_km2_each=9, published_pixel_m=1, cell_m=300,
                 exact_patches=1400, planned_unique_routes=1050, planned_probe_pairs=1680,
                 confirmation_routes=750, confirmation_cross_source_pairs=400,
                 minimum_old_region_gap_m=audit['minimum_old_region_gap_m'],
                 minimum_new_region_pair_gap_m=audit['minimum_new_region_pair_gap_m'], coordinate_tolerance_m=0.001,
                 tests_passed=26, protected_files_verified=986, frozen_input_files_verified=1476,
                 original_data_models_default_unchanged=True, SwissView_registry_unchanged=True,
                 navigation_model_calls=0, new_navigation_records=0, new_SR_produced=False,
                 new_training_steps=0, cloud_calls=0, default_changed=False,
                 unknown_area_navigation_passed=False, seam_recognition_validated=False,
                 pretrained_encoder_unknown_geography_verified=False, new_scientific_figures=2,
                 figures_export_and_visual_QA_passed=True, index_entries=65, logical_result_figures=44,
                 source_sha256={rel(p): digest(p) for p in sources}, check_script_sha256=digest(Path(__file__)),
                 next='freeze M0 engineering interface on4development regions, then confirm on10frozen regions')
    write_once(DELIVERY, final)
    print(json.dumps({k: v for k, v in final.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
