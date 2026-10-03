"""Create a layout-only v2 of the audited spatial-confirmation figures.

The frozen v1 figures, report, manifest, and failed visual review remain
untouched. This script reuses the original audited data readers and plotting
functions, adjusts only figure geometry, then exclusively creates v2 files.
"""
from copy import deepcopy
from datetime import date
from hashlib import sha256
from io import BytesIO
import importlib.util
import json
import math
from pathlib import Path
import sys
import warnings
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
PLOTS = ROOT / '绘图/数据结果图'
PLOT_DATA_V1 = ROOT / '绘图/绘图数据/冻结M0空间隔离导航确认_v1.json'
PLOT_DATA_V2 = ROOT / '绘图/绘图数据/冻结M0空间隔离导航确认_v2.json'
CAPTIONS = ROOT / '绘图/冻结M0空间隔离导航确认图注.md'
QA_V1 = ROOT / '绘图/审阅/冻结M0空间隔离导航确认视觉核验_v1.json'
REPORT_V1 = RUN / '新空间隔离区域M0导航确认报告.md'
README_V1 = RUN / 'README.md'
REPORT_V2 = RUN / '新空间隔离区域M0导航确认报告_v2.md'
README_V2 = RUN / 'README_v2.md'
REVISION_NOTE_V2 = ROOT / '绘图/冻结M0空间隔离导航确认排版修订_v2.md'
SCRIPT_V1 = ROOT / 'project/src/documents/summarize_spatial_area_confirmation.py'

STEMS = (
    '45_空间隔离区域三分层M0导航结果',
    '46_跨源接缝原始头部接受率',
)
PLOT_FILES_V2 = {
    stem: {fmt: PLOTS / f'{stem}_v2.{fmt}' for fmt in ('png', 'pdf', 'svg')}
    for stem in STEMS
}
REQUIRED_NEW_OUTPUTS = [
    *[path for formats in PLOT_FILES_V2.values() for path in formats.values()],
    REPORT_V2, README_V2, REVISION_NOTE_V2, PLOT_DATA_V2,
]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(content):
    return sha256(content).hexdigest()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def exclusive_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode('utf-8')
    with path.open('xb') as handle:
        handle.write(content)


def load_original_script():
    spec = importlib.util.spec_from_file_location('spatial_confirmation_v1', SCRIPT_V1)
    need(spec is not None and spec.loader is not None, 'cannot load original report script')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_v1_and_visual_review(v1):
    need(digest(SCRIPT_V1) == v1['script_sha256'], 'original report script hash changed')
    need(v1['reproduction_script'] == rel(SCRIPT_V1), 'v1 manifest script path mismatch')

    for name, expected in v1['source_sha256'].items():
        path = ROOT / name
        need(path.is_file() and digest(path) == expected,
             f'v1 source binding mismatch: {name}')
    for name, expected in v1['other_outputs_sha256'].items():
        path = ROOT / name
        need(path.is_file() and digest(path) == expected,
             f'v1 text-output binding mismatch: {name}')
    for item in v1['figures']:
        for file in item['files']:
            path = ROOT / file['path']
            need(path.is_file() and digest(path) == file['sha256'],
                 f'v1 figure binding mismatch: {file["path"]}')

    qa = read_json(QA_V1)
    need(qa['passed'] is False and qa['values_or_verdict_changed'] is False
         and qa['requires_layout_revision'] == 2
         and qa['PNG_and_rendered_PDF_reviewed'] is True,
         'v1 visual-review record does not authorize a layout-only revision')
    need(len(qa['issues']) == 3
         and any('图45' in issue and '标题重叠' in issue for issue in qa['issues'])
         and any('图46' in issue and '标题重叠' in issue for issue in qa['issues'])
         and any('边界留白' in issue for issue in qa['issues']),
         'v1 visual-review issues differ from the recorded layout defects')
    for name, expected in qa['files_sha256'].items():
        path = ROOT / name
        need(path.is_file() and digest(path) == expected,
             f'v1 reviewed-file hash mismatch: {name}')
    return qa


def assert_data_unchanged(data, values, rates, v1):
    figure_data = v1['figure_data']
    need(data['confirmation'] == figure_data['full_audited_navigation_summary'],
         'reloaded navigation summary differs from v1 manifest')
    need(data['confirmation_probe'] == figure_data['full_audited_probe_summary'],
         'reloaded probe summary differs from v1 manifest')
    arms = {arm: {cohort: values[arm][cohort]
                  for cohort in ('long_distance', 'seam_target', 'interior_target')}
            for arm in ('CueFull', 'Baseline', 'CueMean', 'CueWrong', 'Frontier', 'FixedRegion')}
    need(arms == figure_data['arms'], 'recomputed plot values differ from v1 figure_data')
    need(rates == figure_data['probe_rates'], 'recomputed probe rates differ from v1 figure_data')


def boxes_intersect(a, b):
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)


def bbox_as_list(box):
    return [round(float(value), 3) for value in (box.x0, box.y0, box.x1, box.y1)]


def measure_title_legend_gaps(fig, label):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legends = [(i, legend.get_window_extent(renderer))
               for i, legend in enumerate(fig.legends)]
    titles = []
    for i, ax in enumerate(fig.axes):
        for title in (ax.title, ax._left_title, ax._right_title):
            if title.get_text():
                titles.append((i, title.get_window_extent(renderer)))
    need(legends and titles, f'{label}: expected figure legend and panel titles')
    overlaps = []
    for legend_index, legend_box in legends:
        for axis_index, title_box in titles:
            if boxes_intersect(legend_box, title_box):
                overlaps.append(f'legend {legend_index}/title {axis_index}')
    for (first_i, first), (second_i, second) in zip(legends, legends[1:]):
        if boxes_intersect(first, second):
            overlaps.append(f'legend {first_i}/legend {second_i}')
    need(not overlaps, f'{label}: layout bounding boxes overlap: {overlaps}')
    return {
        'figure_legends_px': {str(i): bbox_as_list(box) for i, box in legends},
        'panel_titles_px': {str(i): bbox_as_list(box) for i, box in titles},
        'legend_title_intersections': 0,
        'legend_legend_intersections': 0,
    }


def assert_fig46_marker_clearance(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    min_clearance = math.inf
    near_endpoint_count = 0
    plotted_values = []
    for ax in fig.axes:
        bbox = ax.get_window_extent(renderer)
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if not len(offsets):
                continue
            display = ax.transData.transform(offsets)
            sizes = collection.get_sizes()
            max_radius_px = (math.sqrt(float(max(sizes))) / 2 * dpi / 72) if len(sizes) else 0
            for (_, y), (_, center_y) in zip(offsets, display):
                value = float(y)
                plotted_values.append(value)
                if value <= 5 or value >= 95:
                    near_endpoint_count += 1
                clearance = min(center_y - bbox.y0, bbox.y1 - center_y) - max_radius_px
                min_clearance = min(min_clearance, clearance)
                need(clearance > 1,
                     'figure 46 marker does not clear the axes boundary by 1 px')
    need(near_endpoint_count > 0, 'figure 46 has no markers near the 0%/100% boundaries')
    need(plotted_values and min(plotted_values) >= 0 and max(plotted_values) <= 100,
         'figure 46 plotted percentages fall outside the honest 0–100 data range')
    return dict(near_endpoint_markers_checked=near_endpoint_count,
                plotted_min_pct=round(min(plotted_values), 6),
                plotted_max_pct=round(max(plotted_values), 6),
                conservative_marker_boundary_clearance_px=round(min_clearance, 3),
                y_limits=[-3, 103], y_ticks=[0, 20, 40, 60, 80, 100])


def revised_figures(original, values, rates, data, captions):
    fig45 = original.make_figure45(
        values, data['verdict'], data['confirmation'], captions[0])
    # Put both legend rows above the axes title area; preserve all artists/data.
    fig45.set_size_inches(9.2, 7.0)
    fig45.subplots_adjust(top=.775)
    layout45 = measure_title_legend_gaps(fig45, 'figure 45')

    fig46 = original.make_figure46(rates, data['verdict'], captions[1])
    fig46.set_size_inches(10.6, 4.8)
    for ax in fig46.axes:
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    fig46.subplots_adjust(top=.775)
    layout46 = measure_title_legend_gaps(fig46, 'figure 46')
    marker_check = assert_fig46_marker_clearance(fig46)
    return fig45, fig46, {'figure45': layout45, 'figure46': layout46,
                           'figure46_endpoint_clearance': marker_check}


def render_exclusive_payload(fig, path, stem):
    fmt = path.suffix.lstrip('.').lower()
    buffer = BytesIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.savefig(buffer, format=fmt, dpi=300, bbox_inches='tight')
    glyph_warnings = [str(w.message) for w in caught if 'Glyph' in str(w.message)]
    need(not glyph_warnings, f'{stem}: missing-font glyph: {glyph_warnings[:1]}')
    content = buffer.getvalue()
    if fmt == 'pdf':
        need(b'/FontFile' in content, f'{stem}: PDF has no embedded font program')
        need(b'/Subtype /Image' not in content, f'{stem}: PDF unexpectedly contains a bitmap')
    elif fmt == 'svg':
        svg_root = ET.fromstring(content)
        tags = [node.tag.split('}')[-1] for node in svg_root.iter()]
        need('text' in tags and 'image' not in tags,
             f'{stem}: SVG must retain editable text and contain no raster image')
    elif fmt == 'png':
        with Image.open(BytesIO(content)) as image:
            xdpi, ydpi = image.info.get('dpi', (0, 0))
            need(abs(xdpi - 300) < 1.5 and abs(ydpi - 300) < 1.5,
                 f'{stem}: PNG must be 300 dpi (found {xdpi}, {ydpi})')
    else:
        raise ValueError(f'unsupported output format: {fmt}')
    return content


def prepare_report_and_readme(v1, data):
    report = REPORT_V1.read_text(encoding='utf-8')
    readme = README_V1.read_text(encoding='utf-8')
    generated_report = ORIGINAL.make_report(data, VALUES, RATES, CAPTIONS_DATA)
    need(report == generated_report, 'v1 report does not reproduce from the original report script')

    report = report.replace(
        '../../../../../绘图/数据结果图/45_空间隔离区域三分层M0导航结果.png',
        '../../../../../绘图/数据结果图/45_空间隔离区域三分层M0导航结果_v2.png')
    report = report.replace(
        '../../../../../绘图/数据结果图/46_跨源接缝原始头部接受率.png',
        '../../../../../绘图/数据结果图/46_跨源接缝原始头部接受率_v2.png')
    need('_v2.png' in report and report.count('![图45') == 1 and report.count('![图46') == 1,
         'report does not link exactly once to both v2 figures')
    layout_notice = (
        '\n\n**排版修订 v2（仅布局）**：本版重排图45/46的图例、标题和边界留白；'
        '所有数值、统计口径、图注、门槛及验收结论与v1保持一致。'
        '图注的数据定义见[原图注文件](../../../../../绘图/冻结M0空间隔离导航确认图注.md)。\n')
    report = report.replace('# 新空间隔离区域M0导航确认报告\n',
                            '# 新空间隔离区域M0导航确认报告\n' + layout_notice, 1)

    readme = readme.replace('# 冻结M0空间隔离导航确认 v1',
                            '# 冻结M0空间隔离导航确认 v2（仅排版修订）', 1)
    readme = readme.replace('新空间隔离区域M0导航确认报告.md',
                            '新空间隔离区域M0导航确认报告_v2.md')
    readme = readme.replace('冻结M0空间隔离导航确认_v1.json',
                            '冻结M0空间隔离导航确认_v2.json')
    for stem in STEMS:
        for fmt in ('png', 'pdf', 'svg'):
            readme = readme.replace(f'{stem}.{fmt}', f'{stem}_v2.{fmt}')
    readme = readme.replace(
        '## 固定边界\n',
        '图45/46的排版修订说明：[冻结M0空间隔离导航确认排版修订_v2.md]'
        '(../../../../../绘图/冻结M0空间隔离导航确认排版修订_v2.md)。'
        '本版只改版面留白，原图注仍是数据口径来源。\n\n## 固定边界\n', 1)
    need('README_v2.md' not in readme and '报告_v2.md' in readme
         and '冻结M0空间隔离导航确认_v2.json' in readme
         and all(f'{stem}_v2.' in readme for stem in STEMS),
         'v2 README links are incomplete')
    return report, readme


def make_revision_note(layout_checks, v1_hash, qa_hash, new_script_hash):
    return f'''# 冻结M0空间隔离导航确认排版修订 v2

本次修订只处理v1视觉核验记录中的三项版面问题：图45的权重图例与(a)标题相叠，图46的权重图例与(b)标题相叠，以及图46接近0%/100%的点符号缺少边界留白。原始核验记录见[视觉核验v1](审阅/冻结M0空间隔离导航确认视觉核验_v1.json)，其绑定的4张原PNG及PDF渲染校样均已逐项校验SHA-256。

图45将画布增高并下移绘图区，为两行图例与面板标题留出净空；图46将画布增高并下移绘图区，纵轴范围改为−3%至103%，刻度仍为0%、20%、40%、60%、80%、100%，所以端点数据没有被裁切，且阅读刻度不变。脚本在保存前以渲染器像素坐标测量图例和面板标题包围框，确认无交叠；并检查图46端点符号边缘与坐标轴边框之间有超过1像素的净空（实测保守净空：{layout_checks['figure46_endpoint_clearance']['conservative_marker_boundary_clearance_px']}像素）。

图表由原报告脚本的只读`readiness()`、确认轨迹/探针读取器及原`make_figure45()`/`make_figure46()`重建；排版前对照v1清单逐项比较`figure_data`、`confirmation_summary`、冻结绑定、最终判定及源哈希。仅改变画布高度、绘图区上边界和图46纵轴显示留白；不改变任何输入、逐区/逐权重数据、图注口径、门槛、验收结论或报告中的结果数值。数据定义仍以[冻结M0空间隔离导航确认图注](冻结M0空间隔离导航确认图注.md)为准。

v1原文件和SHA-256保持不变，见[原v1清单](绘图数据/冻结M0空间隔离导航确认_v1.json)（SHA-256 `{v1_hash}`）。排版脚本为[`refine_spatial_confirmation_figures.py`](../project/src/documents/refine_spatial_confirmation_figures.py)（SHA-256 `{new_script_hash}`）；失败视觉核验文件SHA-256为`{qa_hash}`。新文件独立保存为`_v2`并在[修订清单](绘图数据/冻结M0空间隔离导航确认_v2.json)记录格式、像素净空和数据一致性检查。
'''


def main():
    existing = [str(path) for path in REQUIRED_NEW_OUTPUTS if path.exists()]
    need(not existing, 'exclusive-create refusal; v2 output already exists: ' + ', '.join(existing))

    v1 = read_json(PLOT_DATA_V1)
    qa = verify_v1_and_visual_review(v1)
    global ORIGINAL, DATA, VALUES, RATES, CAPTIONS_DATA
    ORIGINAL = load_original_script()
    DATA = ORIGINAL.readiness()
    VALUES, AREAS, TRACES = ORIGINAL.read_confirmation_traces(DATA)
    RATES = ORIGINAL.probe_rates(DATA['confirmation_probe'])
    assert_data_unchanged(DATA, VALUES, RATES, v1)
    CAPTIONS_DATA = ORIGINAL.make_captions(DATA, VALUES, RATES)

    figures = revised_figures(ORIGINAL, VALUES, RATES, DATA, CAPTIONS_DATA)
    fig45, fig46, layout_checks = figures
    stems_by_figure = (STEMS[0], STEMS[1])
    payloads = {}
    for fig, stem in zip((fig45, fig46), stems_by_figure):
        for fmt, path in PLOT_FILES_V2[stem].items():
            payloads[path] = render_exclusive_payload(fig, path, stem)
    plt.close(fig45)
    plt.close(fig46)

    report_text, readme_text = prepare_report_and_readme(v1, DATA)
    captions_data_text = CAPTIONS.read_text(encoding='utf-8')
    need(CAPTIONS_DATA[0] in captions_data_text and CAPTIONS_DATA[1] in captions_data_text,
         'v1 caption file does not contain the current unchanged data definitions')

    source_hashes = dict(v1['source_sha256'])
    evidence_paths = [
        PLOT_DATA_V1, SCRIPT_V1, REPORT_V1, README_V1, CAPTIONS, QA_V1,
        *[ROOT / file['path'] for item in v1['figures'] for file in item['files']],
        *[ROOT / name for name in qa['files_sha256']],
    ]
    for path in evidence_paths:
        need(path.is_file(), f'missing v1 layout evidence: {rel(path)}')
        source_hashes[rel(path)] = digest(path)
    source_hashes[rel(Path(__file__))] = digest(Path(__file__))

    figure_items = []
    for old, stem in zip(v1['figures'], STEMS):
        item = deepcopy(old)
        need(item['id'] == stem, f'unexpected v1 logical figure ID: {item["id"]}')
        item['files'] = [dict(path=rel(PLOT_FILES_V2[stem][fmt]),
                              sha256=digest_bytes(payloads[PLOT_FILES_V2[stem][fmt]]),
                              format=fmt)
                         for fmt in ('pdf', 'svg', 'png')]
        item['reproduction_script'] = rel(Path(__file__))
        item['layout_revision'] = 2
        item['sources'] = sorted(set(item['sources']) | {
            rel(PLOT_DATA_V1), rel(QA_V1), rel(REPORT_V1), rel(README_V1),
            *[rel(ROOT / name) for name in qa['files_sha256']],
        })
        figure_items.append(item)

    v1_hash = digest(PLOT_DATA_V1)
    qa_hash = digest(QA_V1)
    script_hash = digest(Path(__file__))
    revision_note = make_revision_note(layout_checks, v1_hash, qa_hash, script_hash)

    other_output_payloads = {
        rel(REPORT_V2): report_text.encode('utf-8'),
        rel(README_V2): readme_text.encode('utf-8'),
        rel(REVISION_NOTE_V2): revision_note.encode('utf-8'),
    }
    manifest = dict(v1)
    manifest['date'] = date.today().isoformat()
    manifest['run'] = '冻结M0空间隔离导航确认_v2_排版'
    manifest['layout_revision'] = 2
    manifest['previous_manifest'] = dict(path=rel(PLOT_DATA_V1), sha256=v1_hash)
    manifest['visual_review_failure'] = dict(path=rel(QA_V1), sha256=qa_hash,
                                             passed=False,
                                             issues=qa['issues'],
                                             reviewed_files_sha256=qa['files_sha256'])
    manifest['source_sha256'] = source_hashes
    manifest['figures'] = figure_items
    manifest['other_outputs_sha256'] = {
        name: digest_bytes(content) for name, content in other_output_payloads.items()
    }
    manifest['reproduction_script'] = rel(Path(__file__))
    manifest['script_sha256'] = script_hash
    manifest['layout_revision_details'] = dict(
        changes=['figure45 canvas/axes spacing only',
                 'figure46 canvas/axes spacing and display range -3..103 with original 0..100 ticks'],
        data_and_verdict_changed=False,
        v1_figure_data_equal=True,
        v1_confirmation_summary_preserved=True,
        v1_frozen_bindings_preserved=True,
        measured_geometry=layout_checks,
    )

    # All validation and rendering is complete before the first exclusive write.
    for path, content in payloads.items():
        exclusive_write(path, content)
    for name, content in other_output_payloads.items():
        exclusive_write(ROOT / name, content)
    exclusive_write(PLOT_DATA_V2, json_bytes(manifest))

    # Read-after-write checks bind exactly the bytes just staged.
    for path, content in payloads.items():
        need(digest(path) == digest_bytes(content), f'post-write hash mismatch: {rel(path)}')
    for name, content in other_output_payloads.items():
        need(digest(ROOT / name) == digest_bytes(content), f'post-write hash mismatch: {name}')
    need(digest(PLOT_DATA_V2) == digest_bytes(json_bytes(manifest)),
         'post-write manifest hash mismatch')
    need(manifest['figure_data'] == v1['figure_data']
         and manifest['confirmation_summary'] == v1['confirmation_summary']
         and manifest['frozen_bindings'] == v1['frozen_bindings'],
         'v2 manifest changed frozen v1 result objects')

    print(json.dumps(dict(
        layout_revision=2,
        report=rel(REPORT_V2), readme=rel(README_V2),
        revision_note=rel(REVISION_NOTE_V2), manifest=rel(PLOT_DATA_V2),
        script_sha256=script_hash,
        files={rel(path): digest_bytes(content) for path, content in payloads.items()},
        geometry=layout_checks,
        data_and_verdict_changed=False,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
