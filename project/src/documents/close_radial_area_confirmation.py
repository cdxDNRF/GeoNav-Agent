"""Publish audited confirmation results, provenance, figure and navigation index."""
from pathlib import Path
from collections import Counter
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'project/src'))
from eval import radial_area_confirmation as p
from documents import stage_figures as graphics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import fitz
from PIL import Image

RUN, FIG = p.OUT, ROOT / '绘图'
REPORT = RUN / '径向保护独立新区域导航确认报告.md'
CAPTION = FIG / '径向保护独立新区域导航确认图注_v1.md'
MANIFEST = FIG / '绘图数据/径向保护独立新区域导航确认_v1.json'
STEM = '50_径向保护独立新区域的分层确认结果'
GROUPS = ('all', *p.STRATA)
LABELS = ('混合队列', '长距离\nC12—16', '接缝目标\nC8—12', '内部目标\nC8—12')


def m(summary, arm, group):
    return summary[arm]['metrics'] if group == 'all' else summary[arm]['by_stratum'][group]


def description(s, v):
    a, b = s['candidate']['metrics'], s['reference']['metrics']
    ci = s['effects']['all']['source95_SR']
    status = '全部独立升级证据通过' if v['eligible_for_default_upgrade'] else '独立默认升级未通过'
    return (f"10个未参与候选选型的空间隔离新区域、750不同任务×3冻结权重：M0 SR{b['sr']*100:.2f}%、"
        f"径向保护SR{a['sr']*100:.2f}%（差{(a['sr']-b['sr'])*100:+.2f}点，"
        f"区域95%差[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]点）；混合SG"
        f"{b['mean_sg_m']:.2f}→{a['mean_sg_m']:.2f}米。恢复{s['recovered']}、损伤{s['harmed']}，"
        f"净增{s['recovered']-s['harmed']}成功。{status}；默认M0保持，0训练/0云端。")


def report():
    v, s = p.read(RUN / '验收结论.json'), p.read(RUN / '主对照/对照汇总.json')
    audit, reg = p.read(RUN / '主对照/独立复核.json'), p.check_bindings()
    assert v['completed'] and audit['passed'] and audit['records'] == 4500
    assert audit['summary_sha256'] == p.digest(RUN / '主对照/对照汇总.json')
    assert not REPORT.exists() and not MANIFEST.exists()
    rows = ['# 径向保护独立新区域导航确认报告', '', description(s, v), '',
        '本轮使用已封存img_3000…img_3009，全部未参与此候选选型；每区9km²、每格300米、10×10/B20。'
        '长距离、接缝目标及内部目标每权重各250题。3组权重重复同750条路线，独立地理样本为10区域。', '',
        '候选、三权重、视觉头、均值及0.50全部冻结，与新运行的M0同题同信息/预算，逐题重置隐状态。'
        '首次动作前1000图块/编码器特征/profile/来源已重建并封存；本轮没有调参、训练、下载或云端调用。', '',
        '| 队列 | M0 SR | 候选SR | SR差（百分点） | M0 SG米 | 候选SG米 |',
        '|---|---:|---:|---:|---:|---:|']
    for group, label in zip(GROUPS, LABELS):
        a, b = m(s, 'candidate', group), m(s, 'reference', group)
        rows.append(f"| {label.replace(chr(10), '')} | {b['sr']*100:.2f}% | {a['sr']*100:.2f}% | {(a['sr']-b['sr'])*100:+.2f} | {b['mean_sg_m']:.2f} | {a['mean_sg_m']:.2f} |")
    rows += ['', 'SR分母为每方法2250个计划终局；SG是全部终局曼哈顿距离，成功记0，300米/格。实际有效移动距离独立报告，不等于SG。', '',
        '| 权重 | M0 SR | 候选SR | 候选SG米 | 覆盖接管动作 | 径向否决动作 |', '|---|---:|---:|---:|---:|---:|']
    seeds = {'reference': [], 'candidate': []}
    for seed in p.SEEDS:
        values = {}
        for arm, policy in [('reference', 'M0'), ('candidate', 'Coverage3Radial')]:
            path = p.trajectory_path(policy, seed, 'CueFull').with_name(f'{policy}_s{seed}_CueFull_结果.json')
            values[arm] = p.read(path)
            seeds[arm].append(values[arm])
        a, b = values['candidate'], values['reference']
        rows.append(f"| {seed} | {b['metrics']['sr']*100:.2f}% | {a['metrics']['sr']*100:.2f}% | {a['metrics']['mean_sg_m']:.2f} | {a['changed_actions']} | {a['guarded_actions']} |")
    rows += ['', '| 预登记主门槛 | 判定 |', '|---|---|']
    names = dict(mixed_SR_gain2pp='混合SR至少提高2点', two_positive_weights='至少2/3权重正收益',
        mixed_SG_no_worse='混合SG不变差', source95_SR_lower_positive='完整区域SR差95%下界>0')
    for key, label in names.items():
        rows.append(f"| {label} | {'通过' if s['checks'][key] else '未通过'} |")
    for group in p.STRATA:
        for key, label in [('SR_loss_at_most2pp', 'SR损伤不超过2点'), ('SG_no_worse', 'SG不变差')]:
            rows.append(f"| {group}：{label} | {'通过' if s['checks']['strata'][group][key] else '未通过'} |")
    rows += ['', '以上为合取门槛；混合SR收益不能抵消分层保护失败。SR差区间采用完整10区域配对bootstrap（4000次、seed7317），区域内平均三权重；不是把权重当作独立源图。', '',
        f"区域层面：{sum(e['sr'] > 0 for e in s['effects']['all']['source_effects'].values())}/10区域SR提高，"
        f"{sum(e['sg'] > 0 for e in s['effects']['all']['source_effects'].values())}/10区域平均SG增加。总体和分层达标不表示每区、每题都改善。", '',
        '| 指标 | M0 | 候选 |', '|---|---:|---:|']
    for field, label in [('failed_episodes', '失败条数'), ('mean_failed_sg_m', '失败集合平均SG米'),
                         ('mean_valid_travel_m', '全部轨迹平均有效移动米'), ('repeat_visit_rate_micro', '动作重访率')]:
        a, b = m(s, 'candidate', 'all')[field], m(s, 'reference', 'all')[field]
        if field == 'failed_episodes':
            rows.append(f'| {label} | {b} | {a} |')
        elif field == 'repeat_visit_rate_micro':
            rows.append(f'| {label} | {b*100:.2f}% | {a*100:.2f}% |')
        else:
            rows.append(f'| {label} | {b:.2f} | {a:.2f} |')
    rows += ['', '失败SG比较的是两组各自失败集合，成员可能不同，不能据此推断同一失败题变好或变坏。逐题恢复/损伤与各源/距离/权重明细见主对照汇总及配对文件。', '']
    if v['controls_started']:
        target = p.read(RUN / '目标对照/对照汇总.json')
        rows += ['| 候选条件 | SR | SG米 | 四项目标证据门槛 |', '|---|---:|---:|---|',
            f"| Full | {s['candidate']['metrics']['sr']*100:.2f}% | {s['candidate']['metrics']['mean_sg_m']:.2f} | — |"]
        for condition in p.CONTROLS:
            metrics = target['comparisons'][condition]['reference']['metrics']
            rows.append(f"| {condition} | {metrics['sr']*100:.2f}% | {metrics['mean_sg_m']:.2f} | {'全部通过' if all(target['checks'][condition].values()) else '未全部通过'} |")
        rows += ['', '每控制分别要求Full SR≥+5点、≥2/3正、SG不变差、完整区域SR差95%下界>0。', '',
            '[目标对照汇总](目标对照/对照汇总.json) / [目标对照独立复核](目标对照/独立复核.json)。', '']
    else:
        rows += ['主对照未通过全部门槛，按预登记没有启动6750条目标对照。本轮没有新的目标控制成绩；旧开发目标证据不能替代新区域证据。', '']
    actions = audit['actions'] + (p.read(RUN / '目标对照/独立复核.json')['actions'] if v['controls_started'] else 0)
    rows += [f"验证：19相关测试通过；6条旧区域/90动作接口重放；1000图块及特征独立重建；{v['records']}条新导航/{actions}动作完整重放，"
        '检查了显式GRU/目标头方程、独立集合覆盖规划、图像与特征绑定、终止条件和来源bootstrap。', '',
        '本轮10区已消费，使用状态另记于本批；数据准备的“尚未使用”账本保留历史快照，不可再把本轮区域当作新的独立确认集。'
        '原S2/S3/S4和原默认配置不回写。结论限同Massachusetts影像家族的新足迹，编码器预训练范围未知；不证明任意角度、异时、跨模态或真实无人机效果。', '',
        '主门槛失败则保留收益和失败证据、收束此冻结候选，不用本批回调规则再宣称独立通过。'
        '全部主与目标证据通过则具备当前协议的版本化默认升级依据；原默认文件仍保留。', '',
        '[验收结论](验收结论.json) / [主对照汇总](主对照/对照汇总.json) / [独立复核](主对照/独立复核.json) / [首动作前封存](首动作前输入封存.json)', '']
    p.write(REPORT, '\n'.join(rows))
    p.write(RUN / 'README.md', '# 径向保护独立新区域确认\n\n' + description(s, v) +
        '\n\n[报告](径向保护独立新区域导航确认报告.md) / [判定](验收结论.json) / [复核](主对照/独立复核.json)。\n')
    caption = (description(s, v) + '\n\n图50：SR为预算内成功率，SG为全部终点曼哈顿距离（米，成功0），两者分开展示。'
        '柱为三冻结权重平均，圆点/三角点为各权重成绩，点不是独立区域置信区间；完整区域配对SR差区间见报告。'
        '10新区域/750不同任务，三分层各250/权重，300米/格、10×10/B20。\n')
    p.write(CAPTION, caption)
    graphics.style()
    plt.rcParams.update({'font.size': 10, 'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1))
    colors = (graphics.GRAY, graphics.BLUE)
    for ax, field, scale, ylabel in zip(axes, ('sr', 'mean_sg_m'), (100, 1), ('成功率 SR（%）', '终点距离 SG（米）')):
        panel_upper = 0
        for j, arm in enumerate(('reference', 'candidate')):
            x = np.arange(4) + (j - .5) * .36
            means = [m(s, arm, group)[field] * scale for group in GROUPS]
            ax.bar(x, means, width=.33, color=colors[j], edgecolor='#34424F', linewidth=.7)
            values = np.asarray([[item['metrics'][field] if group == 'all' else item['by_stratum'][group][field]
                                  for group in GROUPS] for item in seeds[arm]]) * scale
            for seed, offset in enumerate((-.06, 0, .06)):
                ax.scatter(x + offset, values[seed], marker='o' if j == 0 else '^', s=20, facecolors='white', edgecolors='#253641', linewidths=.7, zorder=4)
            upper = values.max(0)
            panel_upper = max(panel_upper, float(values.max()), max(means))
            pad = 1.5 if scale == 100 else max(values.max() * .025, 10)
            for xx, yy, up in zip(x, means, upper):
                ax.text(xx, max(yy, up) + pad, f'{yy:.1f}' if scale == 100 else f'{yy:.0f}', ha='center', va='bottom', fontsize=10)
        ax.set_xticks(np.arange(4), LABELS)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#DBE1E6', linewidth=.6)
        ax.set_ylim(0, 108 if scale == 100 else panel_upper * 1.25)
    fig.legend([Patch(facecolor=c, edgecolor='#34424F') for c in colors], ['M0', 'Coverage3Radial'],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1))
    fig.subplots_adjust(left=.08, right=.99, bottom=.19, top=.87, wspace=.30)
    files = []
    for ext in ('png', 'pdf', 'svg'):
        path = FIG / '数据结果图' / f'{STEM}.{ext}'
        assert not path.exists()
        fig.savefig(path, dpi=300)
        files.append(dict(path=p.rel(path), sha256=p.digest(path)))
    plt.close(fig)
    source_paths = [Path(__file__), REPORT, CAPTION, RUN / '验收结论.json', RUN / '主对照/对照汇总.json',
                    RUN / '主对照/独立复核.json', RUN / '预登记.json', ROOT / 'project/src/documents/stage_figures.py']
    figure = dict(id=STEM, caption=caption, protocol='10 new regions; grid10/B20;300m;750routes;3 frozen weights',
        sources=[p.rel(path) for path in source_paths], files=files, reproduction_script=p.rel(Path(__file__)))
    p.write(MANIFEST, dict(figures=[figure], source_sha256={p.rel(path): p.digest(path) for path in source_paths},
                          summary=s, verdict=v, index_updated=False))
    with fitz.open(FIG / '数据结果图' / f'{STEM}.pdf') as pdf:
        pdf[0].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(FIG / '审阅' / f'{STEM}_PDF校样.png')
    p.write(FIG / '审阅/径向保护独立新区域导航确认作图设计_v1.md',
        '# 实验结果图设计\n\n两面板分组柱图：左SR（0—108%），右终点SG米（从0开始）。'
        'M0灰、候选蓝，固定左右位置并辅以圆/三角权重点；使用可复现Matplotlib和共享科研样式。'
        '全宽180mm插入时字号约8.6pt，不使用3D、渐变或无意义装饰。各权重点展示重复结果，区域CI只标在图注与报告，不混淆独立性。'
        '输出嵌入字体PDF、可编辑文字SVG、300dpi PNG；需直接查看PNG和独立PDF校样后登记视觉核验。\n')
    print(dict(report_created=True, navigation_records=v['records'], actions=actions), flush=True)


def close():
    p.check_bindings()
    v, s = p.read(RUN / '验收结论.json'), p.read(RUN / '主对照/对照汇总.json')
    visual = p.read(FIG / '审阅/径向保护独立新区域导航确认视觉核验_v1.json')
    assert visual['passed'] and visual['PNG_and_rendered_PDF_reviewed']
    manifest = p.read(MANIFEST)
    for mapping in (manifest['source_sha256'], visual['files_sha256']):
        for name, expected in mapping.items():
            assert p.digest(ROOT / name) == expected, name
    figure = manifest['figures'][0]
    for item in figure['files']:
        assert p.digest(ROOT / item['path']) == item['sha256']
    with fitz.open(FIG / '数据结果图' / f'{STEM}.pdf') as pdf:
        assert len(pdf) == 1 and not pdf[0].get_images()
        assert all(font[2] != 'Type3' and pdf.extract_font(font[0])[3] for font in pdf[0].get_fonts(full=True))
    svg = Counter(n.tag.rsplit('}', 1)[-1] for n in ET.parse(FIG / '数据结果图' / f'{STEM}.svg').iter())
    assert svg['text'] > 0 and svg['image'] == 0
    with Image.open(FIG / '数据结果图' / f'{STEM}.png') as image:
        assert min(image.info['dpi']) > 299
    text = description(s, v)
    idx_path = ROOT / '项目导航/实验索引.json'
    idx = p.read(idx_path)
    assert len(idx['batches']) == 69 and not any(b['path'] == p.rel(RUN) for b in idx['batches'])
    idx['batches'].append(dict(dataset='MasaRoads', kind='评测结果', name=RUN.name,
        category='径向保护独立新区域导航确认', status='升级证据通过' if v['eligible_for_default_upgrade'] else '独立默认升级未通过',
        scope='10 new footprints;750routes/3weights;300m/grid10/B20;frozen single candidate',
        path=p.rel(RUN), reports=[p.rel(REPORT)], evidence=[p.rel(RUN / n) for n in ('验收结论.json', '主对照/独立复核.json', '首动作前输入封存.json')]))
    idx['date'] = '2026-10-03'
    catalog = p.read(FIG / '图表来源清单.json')
    assert not any(f['id'] == STEM for f in catalog['figures'])
    catalog['figures'].append(figure)
    def replace_write(path, content):
        with path.open('w', encoding='utf-8', newline='\n') as f:
            f.write(content)
    replace_write(idx_path, json.dumps(idx, ensure_ascii=False, indent=2))
    replace_write(FIG / '图表来源清单.json', json.dumps(catalog, ensure_ascii=False, indent=2))
    root_text = (ROOT / 'README.md').read_text('utf-8')
    start, end = root_text.index('更新至'), root_text.index('本轮正式导航确认先登记')
    root_text = root_text[:start] + f'更新至2026年10月03日。{text}\n\n' + root_text[end:]
    latest = ('## 最新阶段：径向保护独立新区域导航确认完成\n\n' + text + '\n\n'
        f'[报告]({p.rel(REPORT)}) / [判定]({p.rel(RUN / "验收结论.json")}) / [完整重放]({p.rel(RUN / "主对照/独立复核.json")}) / [图50]({p.rel(CAPTION)})。\n\n')
    root_text = root_text.replace('## 最新阶段：径向保护独立新区域准备通过', latest + '## 前置数据：径向保护独立新区域准备通过', 1)
    root_text = root_text.replace('## 前置数据：径向保护独立新区域准备通过\n\n',
        '## 前置数据：径向保护独立新区域准备通过\n\n以下保留数据准备结束时的历史状态；本轮10区已导航消费，正式结论及使用登记见页首新批次。\n\n', 1)
    root_text = root_text.replace('69批', '70批').replace('49张实验与数据协议图', '50张实验与数据协议图')
    replace_write(ROOT / 'README.md', root_text)
    progress_path = ROOT / '项目导航/当前进度.md'
    progress = progress_path.read_text('utf-8')
    start, end = progress.index('截至'), progress.index('## 最新：')
    progress = progress[:start] + '截至2026-10-03，' + text + '\n\n' + progress[end:]
    progress = progress.replace('## 最新：径向保护新区域数据准备通过',
        '## 最新：径向保护独立新区域确认完成\n\n' + text + '\n\n' +
        f'[报告](../{p.rel(REPORT)}) / [判定](../{p.rel(RUN / "验收结论.json")}) / [图50](../{p.rel(CAPTION)})。\n\n'
        '## 前置数据：径向保护新区域准备通过', 1)
    progress = progress.replace('## 前置数据：径向保护新区域准备通过\n\n',
        '## 前置数据：径向保护新区域准备通过\n\n以下保留准备阶段历史状态，本轮10区已消费；本轮正式导航和使用登记见页首。\n\n', 1)
    start = progress.index('## 下一项方向')
    next_text = ('当前协议具备版本化升级依据；先保存新默认候选及适用协议，再扩展验证，原M0配置保留。' if v['eligible_for_default_upgrade'] else
        '本轮冻结候选确认收束，原M0保持；保留独立收益与分层损伤，不在已消费10区域回调参数后重称独立通过。下一项另立协议，优先围绕实测失败分层或更大连续足迹推进。')
    progress = progress[:start] + '## 下一项方向\n\n' + next_text + '\n'
    replace_write(progress_path, progress)
    for name in ('项目导航/README.md', '项目导航/目录结构与维护.md'):
        path = ROOT / name
        replace_write(path, path.read_text('utf-8').replace('69批', '70批').replace('69个', '70个'))
    path = ROOT / '项目导航/实验索引.md'
    replace_write(path, path.read_text('utf-8').replace('69批', '70批').replace('69个', '70个') +
        f'\n\n## 径向保护独立新区域导航确认_v1\n\n{text}\n\n[报告](../{p.rel(REPORT)}) / [判定](../{p.rel(RUN / "验收结论.json")})。\n')
    path = FIG / 'README.md'
    replace_write(path, path.read_text('utf-8') + f'\n\n## 图50：径向保护独立新区域导航确认\n\n{text}\n\n'
        f'[PDF](数据结果图/{STEM}.pdf) / [SVG](数据结果图/{STEM}.svg) / [PNG](数据结果图/{STEM}.png) / '
        '[图注](径向保护独立新区域导航确认图注_v1.md) / [来源](绘图数据/径向保护独立新区域导航确认_v1.json)。\n')
    # Bind every finished output without a self-referential receipt hash.
    output_paths = [q for q in RUN.rglob('*') if q.is_file()]
    output_paths.extend(ROOT / n for n in ('README.md', '项目导航/当前进度.md', '项目导航/实验索引.json',
        '项目导航/实验索引.md', '项目导航/README.md', '项目导航/目录结构与维护.md', '绘图/README.md', '绘图/图表来源清单.json'))
    output_paths.extend([Path(__file__), MANIFEST, CAPTION, FIG / '审阅/径向保护独立新区域导航确认视觉核验_v1.json'])
    output_paths.extend(ROOT / n for n in ('选题报告相关/径向保护独立新区域导航确认测试_v1.json',
        '选题报告相关/径向保护独立新区域导航确认测试启动失败_v1.json',
        '选题报告相关/径向保护独立新区域导航确认冻结方案_v1.md'))
    output_paths.extend(ROOT / item['path'] for item in figure['files'])
    links = 0
    for path in (ROOT / 'README.md', progress_path, REPORT, RUN / 'README.md', FIG / 'README.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text('utf-8')):
            target = target.strip('<>').split('#')[0]
            if target and not re.match(r'[a-z]+://', target):
                assert (path.parent / target).resolve().exists(), (path, target)
                links += 1
    p.check_bindings()
    p.write(RUN / '阶段交付核验.json', dict(completed=True, eligible_for_default_upgrade=v['eligible_for_default_upgrade'],
        new_navigation_records=v['records'], tests_passed=19, native_patches_reextracted=1000,
        protected_files_verified=len(p.read(RUN / '预登记.json')['protected_sha256']),
        index_entries=70, logical_figures=50, local_links_verified=links, default_changed=False,
        new_training_steps=0, cloud_calls=0, files_sha256={p.rel(q): p.digest(q) for q in output_paths}))
    print(dict(delivery_completed=True, index_entries=70, navigation_records=v['records']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('report', 'close'))
    {'report': report, 'close': close}[parser.parse_args().mode]()
