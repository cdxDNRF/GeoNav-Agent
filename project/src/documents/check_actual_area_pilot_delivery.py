"""Read-only final artifact checks; append the pilot delivery attestation once."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'DATA/processed_data/Masa/评测结果/实际区域M0兼容试跑_v1'
FIG = ROOT / '绘图'


def read(p): return json.loads(p.read_text('utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()
def write(p, value):
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    assert not (RUN / '阶段交付核验.json').exists(), 'immutable final delivery'
    reg = read(RUN / '预登记.json')
    for field in ('protected_sha256', 'source_sha256', 'input_sha256', 'encoder_sha256', 'frozen_output_sha256'):
        assert all(digest(ROOT / p) == h for p, h in reg[field].items()), field
    frozen = read(RUN / '特征冻结结束.json')
    assert all(digest(ROOT / p) == h for p, h in frozen['files_sha256'].items())
    v = read(RUN / '验收结论.json')
    a = read(RUN / '独立复核.json')
    s = read(RUN / '对照汇总.json')
    state = read(RUN / '最终状态.json')
    assert v['summary_sha256'] == digest(RUN / '对照汇总.json') and v['audit_sha256'] == digest(RUN / '独立复核.json')
    assert state['report_sha256'] == digest(RUN / '实际区域M0兼容试跑报告.md') and state['verdict_sha256'] == digest(RUN / '验收结论.json')
    assert a['passed'] and a['records'] == 1050 and a['actions'] == 19011 and state['completed']
    rows = []
    for folder in ('神经对照', '规则基线'):
        for p in sorted((RUN / folder).glob('*轨迹.jsonl')):
            batch = [json.loads(line) for line in p.read_text('utf-8').splitlines()]
            assert len(batch) == 75 and [r['episode_id'] for r in batch] == [r['episode_id'] for r in read(RUN / '导航任务.json')]
            assert digest(p) == read(p.with_name(p.name.replace('轨迹.jsonl', '结果.json')))['trajectory_sha256']
            rows.extend(batch)
    assert len(rows) == 1050 and sum(r['steps'] for r in rows) == 19011
    full = [r for r in rows if r['condition'] == 'CueFull']
    cross = [r for r in full if any(d['actionable_true_adjacency'] and d['true_adjacency_cross_source'] for d in r['evaluation_diagnostics'])]
    supplemental = read(RUN / '接缝诊断范围补充.json')
    assert supplemental['cross_source_unique_tasks'] == sorted({r['episode_id'] for r in cross})
    assert len(cross) == 3 and len({r['episode_id'] for r in cross}) == 1
    assert supplemental['full_failed_records'] == sum(not r['success'] for r in full) == 49
    assert abs(supplemental['mean_sg_m_failed_only'] - sum(r['sg_m'] for r in full if not r['success'])/49) < 1e-12
    assert sum(d['crossed_source_boundary'] for r in full for d in r['evaluation_diagnostics']) == 474
    assert all(digest(ROOT / p) == h for p, h in supplemental['source_sha256'].items())
    index = read(ROOT / '项目导航/实验索引.json')
    assert len(index['batches']) == len({b['path'] for b in index['batches']}) == 63
    batch = next(b for b in index['batches'] if b['path'] == rel(RUN))
    assert all((ROOT / p).is_file() for p in (*batch['reports'], *batch['evidence']))
    manifest = read(FIG / '绘图数据/实际区域M0兼容试跑_v2.json')
    assert all(digest(ROOT / p) == h for p, h in manifest['source_sha256'].items())
    assert manifest['script_sha256'] == digest(ROOT / 'project/src/documents/refine_actual_area_figure.py')
    assert manifest['initial_manifest_sha256'] == digest(FIG / '绘图数据/实际区域M0兼容试跑_v1.json')
    catalog = read(FIG / '图表来源清单.json')
    assert len(catalog['figures']) == len({f['id'] for f in catalog['figures']})
    # Earlier figures8..26 use individual manifests; the top catalog is partial.
    # Count logical result figures by their actual directory numbers, not by
    # assuming the catalog includes every older figure or revision.
    result_ids = {int(p.name.split('_')[0]) for p in (FIG / '数据结果图').glob('*.png')
                  if re.match(r'^\d{2}_', p.name)}
    assert result_ids == set(range(1, 43))
    details = []
    for item in manifest['figures']:
        assert next(f for f in catalog['figures'] if f['id'] == item['id']) == item
        assert all(digest(ROOT / f['path']) == f['sha256'] for f in item['files'])
        assert all((ROOT / p).is_file() for p in item['sources'])
        paths = {Path(f['path']).suffix: ROOT / f['path'] for f in item['files']}
        with Image.open(paths['.png']) as im:
            assert min(im.info['dpi']) > 299
            size, dpi = im.size, im.info['dpi']
        svg = ET.parse(paths['.svg'])
        tags = Counter(e.tag.rsplit('}', 1)[-1] for e in svg.iter())
        assert tags['text'] > 0 and tags['image'] == 0
        with fitz.open(paths['.pdf']) as doc:
            assert len(doc) == 1 and not doc[0].get_images()
            fonts = doc[0].get_fonts(full=True)
            assert fonts and all(f[2] != 'Type3' and len(doc.extract_font(f[0])[3]) > 0 for f in fonts)
            pdftext = doc[0].get_text()
            if item['id'].startswith('41_'):
                assert '合计75题' in pdftext and '316.0' in pdftext and '78.22' in pdftext
            else:
                assert '173/173' in pdftext and '3/3' in pdftext and '仅1题' in pdftext
            details.append(dict(id=item['id'], paths=item['files'], PNG_size=size, DPI=dpi,
                   PDF_pages=1, embedded_fonts=True, PDF_raster_images=0, SVG_editable_texts=tags['text'], SVG_images=0))
    # Check navigational/report local links without opening external URLs.
    mdpaths = [RUN / '实际区域M0兼容试跑报告.md', RUN / 'README.md', ROOT / 'README.md',
               ROOT / '项目导航/当前进度.md', FIG / 'README.md']
    checked = 0
    for p in mdpaths:
        for link in re.findall(r'\]\(([^)]+)\)', p.read_text('utf-8')):
            link = link.strip('<>').split('#')[0]
            if not link or re.match(r'[a-z]+://', link):
                continue
            target = (p.parent / link).resolve()
            assert target.exists(), (str(p), link)
            checked += 1
    fileqa = dict(passed=True, figures=details, initial_fig41_exports_retained=True, logical_new_figures=2,
                  index_entries=63, catalog_entries=len(catalog['figures']), legacy_top_catalog_partial=True,
                  scientific_result_figures=len(result_ids), local_links_verified=checked,
                  all_runtime_data_model_and_default_bindings_unchanged=True, no_new_training_or_cloud_calls=True,
                  supplemental_cross_seam_task_count_verified=1)
    write(FIG / '审阅/实际区域M0兼容试跑文件核验_v2.json', fileqa)
    sources = [RUN / n for n in ('实际区域M0兼容试跑报告.md', '验收结论.json', '独立复核.json', '对照汇总.json',
               '最终状态.json', '接缝诊断范围补充.json', '下一项空间隔离连续区域确认准备.md')]
    sources.extend((FIG / '绘图数据/实际区域M0兼容试跑_v2.json', FIG / '审阅/实际区域M0兼容试跑文件核验_v2.json'))
    sources.extend(ROOT / p for i in manifest['figures'] for p in [f['path'] for f in i['files']])
    visual = read(FIG / '审阅/实际区域M0兼容试跑视觉核验_v2.json')
    assert visual['passed'] and visual['final_plots'] == 2 and visual['PNG_and_rendered_PDF_reviewed']
    assert all(digest(ROOT / p) == h for p, h in visual['files_sha256'].items())
    sources.append(FIG / '审阅/实际区域M0兼容试跑视觉核验_v2.json')
    final = dict(completed=True, execution_compatibility_passed=True, target_evidence_passed=v['target_evidence_passed'],
           known_area_pilot_passed=v['known_area_pilot_passed'], maps=3, known_training_maps_only=True, projected_area_km2_each=9,
           native_cell_m=300, tasks_each_weight=75, unique_routes=73, weights=3, all_planned_records=1050,
           replayed_actions=19011, tests_passed=19, original_data_models_default_unchanged=True, new_training_steps=0,
           cloud_calls=0, SwissView_registry_unchanged=True, new_scientific_figures=2, figures_export_and_visual_QA_passed=True,
           index_entries=63, total_result_figures=42, cross_seam_goal_opportunities=3, cross_seam_unique_task_count=1,
           cross_seam_general_reliability_confirmed=False, formal_unknown_geography_passed=False, default_changed=False,
           source_sha256={rel(p): digest(p) for p in sources}, check_script_sha256=digest(Path(__file__)),
           next='prepare spatially separated external continuous imagery and preregister seam-balanced tasks')
    write(RUN / '阶段交付核验.json', final)
    print({k: v for k, v in final.items() if k != 'source_sha256'}, flush=True)


if __name__ == '__main__':
    main()
