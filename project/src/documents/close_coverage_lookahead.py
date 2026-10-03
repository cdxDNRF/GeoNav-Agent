"""Update mutable navigation once, then verify and seal the completed batch."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'DATA/processed_data/MasaRoads/评测结果/空间覆盖与误触发单因素验证_v1'
FIG = ROOT / '绘图'
REPORT = RUN / '三步覆盖接管开发验证报告.md'
MANIFEST = FIG / '绘图数据/三步覆盖接管开发验证_v1.json'
VISUAL = FIG / '审阅/三步覆盖接管开发验证视觉核验_v1.json'
FILE_QA = FIG / '审阅/三步覆盖接管开发验证文件核验_v1.json'
DELIVERY = RUN / '阶段交付核验.json'


def read(p):
    return json.loads(p.read_text('utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rel(p):
    return p.relative_to(ROOT).as_posix()


def json_write(p, obj, exclusive=False):
    with p.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def result_text():
    s = read(RUN / '主对照/对照汇总.json')
    a, b = s['candidate']['metrics'], s['reference']['metrics']
    effect = s['effects']['all']
    return (f"同10个已消费区域、750题×3权重的后验开发：三步覆盖候选混合SR{a['sr']*100:.2f}%"
        f"（M0 {b['sr']*100:.2f}%，+{effect['sr_gain']*100:.2f}点，3/3权重正）；"
        f"混合SG{b['mean_sg_m']:.2f}→{a['mean_sg_m']:.2f}米，长距离SR72.80%→65.07%。"
        f"恢复{s['recovered']}、损伤{s['harmed']}，净增{s['recovered']-s['harmed']}条成功。"
        "SR收益保留，长距离及SG保护门槛未过；2250轨迹/36350动作重放与10项测试通过。"
        "未启动目标对照或新区域确认，默认M0保持。")


def index():
    verdict = read(RUN / '验收结论.json')
    assert verdict['completed'] and not verdict['development_main_passed']
    assert read(VISUAL)['passed'] and REPORT.is_file()
    manifest = read(MANIFEST)
    summary = result_text()
    p = ROOT / '项目导航/实验索引.json'
    value = read(p)
    assert len(value['batches']) == 66 and not any(b['path'] == rel(RUN) for b in value['batches'])
    value['batches'].append(dict(dataset='MasaRoads', kind='评测', name=RUN.name,
        category='覆盖与误触发后验开发', status='SR收益；长距离/SG保护未通过',
        scope='10已消费区域/750任务/3权重；0训练/0云端；无新源确认，Sat2Cap预训练地理未知',
        path=rel(RUN), reports=[rel(REPORT)], evidence=[rel(RUN / '验收结论.json'),
            rel(RUN / '主对照/独立复核.json'), rel(RUN / '离线诊断/独立算术复核.json')]))
    value['date'] = '2026-10-03'
    json_write(p, value)
    p = ROOT / '项目导航/实验索引.md'
    text = p.read_text('utf-8').replace('66批', '67批').replace('66个', '67个')
    text += f'\n\n## 空间覆盖与误触发单因素验证_v1\n\n{summary}\n\n[报告](../{rel(REPORT)}) / [判定](../{rel(RUN / "验收结论.json")}) / [复核](../{rel(RUN / "主对照/独立复核.json")})。\n'
    p.write_text(text, 'utf-8')
    p = ROOT / 'README.md'
    text = p.read_text('utf-8')
    start, end = text.index('更新至'), text.index('## 最新阶段')
    text = text[:start] + ('更新至2026年10月03日。环境、统一评测、单Agent主线及原S2/S3/S4、SwissView和实际面积空间隔离确认均已完成。'
        '最新三步覆盖候选获得混合SR收益，但长距离与SG保护门槛未通过，默认M0保持。先读最新阶段与当前进度，再看对应批次协议。\n\n') + text[end:]
    text = text.replace('## 最新阶段：冻结M0空间隔离导航确认完成', '## 前一阶段：冻结M0空间隔离导航确认完成', 1)
    position = text.index('## 前一阶段：冻结M0空间隔离导航确认完成')
    latest = (f'## 最新阶段：三步覆盖接管开发验证收尾\n\n{summary}\n\n'
        f'[报告]({rel(REPORT)}) / [判定]({rel(RUN / "验收结论.json")}) / '
        f'[独立重放]({rel(RUN / "主对照/独立复核.json")}) / [科研图47]({rel(FIG / "三步覆盖接管开发验证图注.md")})。\n\n')
    text = text[:position] + latest + text[position:]
    text = text.replace('全部66批', '全部67批').replace('66批实验', '67批实验').replace('46张实验与数据协议图', '47张实验与数据协议图')
    text = text.replace('新增实际面积试跑保持300米/格并扩大至每图9 km²，使用三张已知训练区域，未知地区正式确认尚未完成。',
        '新增实际面积试跑保持300米/格并扩大至每图9 km²；其后三批已完成空间隔离数据准备、冻结M0新区域确认及本轮后验开发。')
    text = text.replace('源文件隔离不证明严格地理无重叠或预训练编码器未见。任意角度、真实异时、跨视角/跨模态目标、未知地区的大范围验证和无人机实飞尚待验证。',
        '新区域与本地M0旧足迹保持至少3公里空间缓冲；Sat2Cap预训练地理覆盖未知。任意角度、真实异时、跨视角/跨模态及无人机实飞尚待验证。')
    p.write_text(text, 'utf-8')
    p = ROOT / '项目导航/当前进度.md'
    text = p.read_text('utf-8')
    start, end = text.index('截至'), text.index('## 最新：实际搜索面积')
    introduction = ('截至2026-10-03，原S2/S3/S4、SwissView同模态及实际面积空间隔离确认继续有效。'
        '本轮Coverage3仅为已消费10区的后验开发候选，SR有收益但保护门槛未过；默认M0保持。\n\n')
    latest = (f'## 最新：三步覆盖单因素开发收尾\n\n{summary}\n\n'
        f'[报告](../{rel(REPORT)}) / [判定](../{rel(RUN / "验收结论.json")}) / '
        f'[图47](../{rel(FIG / "三步覆盖接管开发验证图注.md")})。\n\n'
        '该因素按一次冻结验证收束；本批未在失败后调参或扩大源图。候选权重与代码保留，适用边界及SR/距离权衡见完整报告。\n\n')
    text = text[:start] + introduction + latest + text[end:]
    text = text.replace('## 最新：实际搜索面积的空间隔离确认', '## 前一轮：实际搜索面积的空间隔离确认', 1)
    old = ('优先事后诊断C8—12接缝与内部目标两类约51%的成功率，再决定一个有机制证据的改进因素。'
           '真值只用于诊断，不输入Agent。新因素需另行预登记；这14个区域已消费，不能再次声明为未参与实验的独立确认源。')
    new = ('覆盖偏置与误触发诊断已完成，三步覆盖候选已完成一次冻结验证，当前因素收束。'
           '后续先明确是否接受SR与长距离损伤的权衡，或提出新的保护机制，再另立前瞻协议；不能回写本次未通过判定。'
           '这14个区域已经消费，新独立确认必须使用其他空间区域。真值仅用于事后诊断。')
    assert old in text
    text = text.replace(old, new)
    p.write_text(text, 'utf-8')
    for name in ('项目导航/README.md', '项目导航/目录结构与维护.md'):
        p = ROOT / name
        p.write_text(p.read_text('utf-8').replace('66批', '67批').replace('66个', '67个'), 'utf-8')
    p = FIG / '图表来源清单.json'
    value = read(p)
    assert not any(f['id'] == manifest['figures'][0]['id'] for f in value['figures'])
    value['figures'].extend(manifest['figures'])
    json_write(p, value)
    p = FIG / 'README.md'
    text = p.read_text('utf-8')
    stem = manifest['figures'][0]['id']
    text += (f'\n\n## 图47：三步覆盖接管的SR与距离权衡\n\n{summary}\n\n'
        f'[PDF](数据结果图/{stem}.pdf) / [SVG](数据结果图/{stem}.svg) / [PNG](数据结果图/{stem}.png) / '
        '[图注](三步覆盖接管开发验证图注.md) / [数据与来源](绘图数据/三步覆盖接管开发验证_v1.json)。\n')
    p.write_text(text, 'utf-8')
    print(dict(index_entries=67, logical_figures=47, status='SR收益；保护门槛未通过'))


def check():
    assert not DELIVERY.exists() and not FILE_QA.exists(), 'immutable delivery already exists'
    verdict, state = read(RUN / '验收结论.json'), read(RUN / '最终状态.json')
    assert verdict['completed'] and state['completed']
    assert not verdict['development_main_passed'] and verdict['development_target_evidence_passed'] is None
    assert not verdict['controls_started'] and not state['new_area_confirmation_started']
    assert not verdict['default_changed'] and verdict['new_training_steps'] == verdict['cloud_calls'] == 0
    assert verdict['default_sha256'] == digest(ROOT / 'project/local_policy_default.json')
    assert state['verdict_sha256'] == digest(RUN / '验收结论.json')
    assert not list((RUN / '目标对照').glob('*轨迹.jsonl'))
    registration = read(RUN / '预登记.json')
    assert read(RUN / '预登记封存.json')['registration_sha256'] == digest(RUN / '预登记.json')
    for field in ('protected_sha256', 'source_sha256', 'frozen_sha256'):
        for name, expected in registration[field].items():
            assert digest(ROOT / name) == expected, name
    diagnostic = read(RUN / '离线诊断/独立算术复核.json')
    assert diagnostic['passed'] and diagnostic['navigation_records'] == 4500 and diagnostic['rechecked_actions'] == 78525
    for name, expected in diagnostic['files_sha256'].items():
        assert digest(RUN / '离线诊断' / name) == expected
    audit = read(RUN / '主对照/独立复核.json')
    assert audit['passed'] and (audit['records'], audit['actions']) == (2250, 36350)
    assert audit['summary_sha256'] == digest(RUN / '主对照/对照汇总.json')
    summary = read(RUN / '主对照/对照汇总.json')
    assert verdict['SR'] == summary['candidate']['metrics']['sr']
    assert verdict['original_SR'] == summary['reference']['metrics']['sr']
    assert (summary['recovered'], summary['harmed']) == (439, 267)
    assert summary['effects']['all']['positive_seeds'] == 3
    assert not summary['checks']['mixed_SG_no_worse']
    assert not summary['checks']['strata']['long_distance']['SR_loss_at_most2pp']
    tests = read(ROOT / '选题报告相关/三步覆盖接管测试_v2.json')
    assert tests['passed'] and tests['tests_run'] == 10
    for p in (RUN / '主对照').glob('*结果.json'):
        assert read(p)['trajectory_sha256'] == digest(p.with_name(p.name.replace('结果.json', '轨迹.jsonl')))
    visual, manifest = read(VISUAL), read(MANIFEST)
    assert visual['passed'] and visual['PNG_and_rendered_PDF_reviewed'] and visual['final_plots'] == 1
    for mapping in (visual['files_sha256'], manifest['source_sha256'], manifest['other_outputs_sha256']):
        for name, expected in mapping.items():
            assert digest(ROOT / name) == expected, name
    assert manifest['main_summary'] == summary and manifest['verdict'] == verdict
    assert manifest['script_sha256'] == digest(ROOT / manifest['reproduction_script'])
    figure = manifest['figures'][0]
    files = {}
    for item in figure['files']:
        p = ROOT / item['path']
        assert digest(p) == item['sha256']
        files[p.suffix] = p
    with Image.open(files['.png']) as im:
        assert min(im.info['dpi']) > 299
        size, dpi = im.size, im.info['dpi']
    with fitz.open(files['.pdf']) as pdf:
        assert len(pdf) == 1 and not pdf[0].get_images()
        fonts = pdf[0].get_fonts(full=True)
        assert fonts and all(f[2] != 'Type3' and pdf.extract_font(f[0])[3] for f in fonts)
    svg = Counter(n.tag.rsplit('}', 1)[-1] for n in ET.parse(files['.svg']).iter())
    assert svg['text'] > 0 and svg['image'] == 0
    assert figure == next(f for f in read(FIG / '图表来源清单.json')['figures'] if f['id'] == figure['id'])
    ids = {int(p.name.split('_')[0]) for p in (FIG / '数据结果图').glob('*.png') if re.match(r'^\d{2}_', p.name)}
    assert ids == set(range(1, 48))
    index_value = read(ROOT / '项目导航/实验索引.json')
    assert len(index_value['batches']) == len({b['path'] for b in index_value['batches']}) == 67
    entry = next(b for b in index_value['batches'] if b['path'] == rel(RUN))
    assert all((ROOT / n).is_file() for n in (*entry['reports'], *entry['evidence']))
    markdowns = [REPORT, RUN / 'README.md', ROOT / 'README.md', ROOT / '项目导航/当前进度.md',
        ROOT / '项目导航/实验索引.md', FIG / 'README.md', FIG / '三步覆盖接管开发验证图注.md']
    link_count = 0
    for p in markdowns:
        for link in re.findall(r'\]\(([^)]+)\)', p.read_text('utf-8')):
            link = link.strip('<>').split('#')[0]
            if not link or re.match(r'[a-z]+://', link):
                continue
            assert (p.parent / link).resolve().exists(), (str(p), link)
            link_count += 1
    json_write(FILE_QA, dict(passed=True, logical_figures=47, index_entries=67, local_links_verified=link_count,
        PNG_size=size, DPI=dpi, PDF_embedded_fonts=True, PDF_raster_images=0, SVG_editable_texts=svg['text'],
        all_registered_data_model_runtime_hashes_unchanged=True), exclusive=True)
    paths = list(markdowns) + [RUN / n for n in ('预登记.json', '预登记封存.json', '候选配置.json',
        '验收结论.json', '最终状态.json', '主对照/独立复核.json', '离线诊断/独立算术复核.json')]
    paths.extend((MANIFEST, VISUAL, FILE_QA, Path(__file__)))
    paths.extend(files.values())
    paths.extend(ROOT / n for n in ('项目导航/实验索引.json', '项目导航/README.md',
        '项目导航/目录结构与维护.md', '绘图/图表来源清单.json'))
    result = dict(completed=True, development_candidate_passed=False, SR_gain_preserved=True,
        independent_new_area_confirmation_started=False, default_changed=False, new_training_steps=0, cloud_calls=0,
        diagnostic_old_records=4500, diagnostic_old_actions=78525,
        new_navigation_records=2250, new_replayed_actions=36350, tests_passed=10,
        new_result_figure=1, index_entries=67, logical_figures=47,
        all_original_data_models_results_and_runtime_preserved=True,
        protected_files_verified=len(registration['protected_sha256']),
        source_sha256={rel(p): digest(p) for p in paths},
        state='one frozen candidate closed; SR improved with long-distance and SG damage')
    json_write(DELIVERY, result, exclusive=True)
    print({k: v for k, v in result.items() if k != 'source_sha256'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('index', 'check'))
    {'index': index, 'check': check}[parser.parse_args().mode]()
