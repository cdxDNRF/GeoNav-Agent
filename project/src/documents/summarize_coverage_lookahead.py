"""Audited one-factor development report and one standalone scientific figure."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'project/src'))
from documents import stage_figures as graphics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

RUN = ROOT / 'DATA/processed_data/MasaRoads/评测结果/空间覆盖与误触发单因素验证_v1'
OLD = ROOT / 'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
FIG = ROOT / '绘图'
REPORT = RUN / '三步覆盖接管开发验证报告.md'
README = RUN / 'README.md'
CAPTION = FIG / '三步覆盖接管开发验证图注.md'
MANIFEST = FIG / '绘图数据/三步覆盖接管开发验证_v1.json'
STEM = '47_三步覆盖接管的成功率与距离权衡'
COHORTS = ('all', 'long_distance', 'seam_target', 'interior_target')
LABELS = ('混合队列', '长距离C12—16', '接缝C8—12', '内部C8—12')


def read(p):
    return json.loads(p.read_text('utf-8-sig'))


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def rel(p):
    return p.relative_to(ROOT).as_posix()


def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))


def metric(arm, cohort):
    return arm['metrics'] if cohort == 'all' else arm['by_stratum'][cohort]


def main():
    verdict, state = read(RUN / '验收结论.json'), read(RUN / '最终状态.json')
    audit, summary = read(RUN / '主对照/独立复核.json'), read(RUN / '主对照/对照汇总.json')
    diag = read(RUN / '离线诊断/诊断汇总.json')
    assert verdict['completed'] and state['completed'] and audit['passed']
    assert state['verdict_sha256'] == digest(RUN / '验收结论.json')
    assert audit['records'] == 2250 and audit['summary_sha256'] == digest(RUN / '主对照/对照汇总.json')
    assert verdict['default_sha256'] == digest(ROOT / 'project/local_policy_default.json')
    expected = [REPORT, README, CAPTION, MANIFEST]
    expected.extend(FIG / '数据结果图' / f'{STEM}.{ext}' for ext in ('png', 'pdf', 'svg'))
    assert not any(p.exists() for p in expected), 'exclusive report and figures'
    source_paths = [RUN / n for n in ('验收结论.json', '最终状态.json', '预登记.json', '预登记封存.json',
        '候选配置.json', '执行协议.md', '离线诊断/输入登记.json', '离线诊断/诊断汇总.json',
        '离线诊断/逐题诊断.json', '离线诊断/逐题配对.json', '离线诊断/独立算术复核.json',
        '主对照/对照汇总.json', '主对照/逐题配对.json', '主对照/独立复核.json')]
    source_paths.extend(ROOT / n for n in ('project/local_policy_default.json',
        '选题报告相关/三步覆盖接管测试_v1.json', '选题报告相关/三步覆盖接管测试_v2.json'))
    seeds = {'M0': [], 'Coverage3': []}
    for seed in range(3):
        for name, folder, stem in [('M0', OLD / '独立确认/神经对照', f'M0_s{seed}_CueFull'),
                                   ('Coverage3', RUN / '主对照', f'Coverage3_s{seed}_CueFull')]:
            result_path = folder / f'{stem}_结果.json'
            trace_path = folder / f'{stem}_轨迹.jsonl'
            result = read(result_path)
            assert result['trajectory_sha256'] == digest(trace_path)
            seeds[name].append(result)
            source_paths.extend((result_path, trace_path))
    if verdict['controls_started']:
        source_paths.extend(p for p in (RUN / '目标对照').glob('*') if p.is_file())
    candidate, reference = summary['candidate'], summary['reference']
    status = '通过开发门槛' if verdict['development_main_passed'] else '未通过开发门槛'
    table = ['| 队列 | M0 SR | 候选SR | SR差 | M0 SG米 | 候选SG米 | M0失败SG米 | 候选失败SG米 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort, label in zip(COHORTS, LABELS):
        a, b = metric(candidate, cohort), metric(reference, cohort)
        table.append(f"| {label} | {b['sr']*100:.2f}% | {a['sr']*100:.2f}% | {(a['sr']-b['sr'])*100:+.2f}点 | "
            f"{b['mean_sg_m']:.2f} | {a['mean_sg_m']:.2f} | {b['mean_failed_sg_m']:.2f} | {a['mean_failed_sg_m']:.2f} |")
    gates = summary['checks']
    gate_rows = [('混合SR至少+2个百分点', gates['mixed_SR_gain2pp']),
                 ('至少2/3权重正收益', gates['two_positive_weights']), ('混合SG不变差', gates['mixed_SG_no_worse'])]
    for s, label in zip(COHORTS[1:], LABELS[1:]):
        gate_rows.extend([(label + ' SR损伤不超过2点', gates['strata'][s]['SR_loss_at_most2pp']),
                          (label + ' SG不变差', gates['strata'][s]['SG_no_worse'])])
    gate_table = '| 冻结门槛 | 判定 |\n|---|---|\n' + '\n'.join(f"| {label} | {'通过' if ok else '未通过'} |" for label, ok in gate_rows)
    effect = summary['effects']['all']
    weight_table = '| 权重 | M0混合SR | 候选混合SR | SR差 |\n|---:|---:|---:|---:|\n'
    for seed in range(3):
        a, b = seeds['Coverage3'][seed]['metrics'], seeds['M0'][seed]['metrics']
        weight_table += f"| {seed} | {b['sr']*100:.2f}% | {a['sr']*100:.2f}% | {(a['sr']-b['sr'])*100:+.2f}点 |\n"
    caption = ('同10个已消费空间区域、750个固定任务×3冻结权重的后验开发对照。混合队列及三分层分别展示，'
        '每份权重混合750题、每层250题。柱为全部三权重合并值；空心符号为各权重结果，不是置信区间。'
        'SG是成功记0的终点曼哈顿距离，米制，较低为好。控制器只改变无已接受cue时的公开几何仲裁；'
        '模型、目标头、0.50、输入权限与B20保持。开发门槛按混合增益、分层损伤和SG共同判定，'
        '图中SR改善不能独自代表通过，也不是未见区域的独立确认。')
    graphics.style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    xs = np.arange(4)
    colors = {'M0': '#0072B2', 'Coverage3': '#D55E00'}
    titles = []
    for panel, (key, scale, ylabel, title) in enumerate([
            ('sr', 100, 'SR（%）', '(a) 混合与各分层成功率'),
            ('mean_sg_m', 1, '平均终点SG（米）', '(b) 同队列的终点距离')]):
        ax = axes[panel]
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.65)
        for j, (name, arm) in enumerate([('M0', reference), ('Coverage3', candidate)]):
            values = [metric(arm, c)[key] * scale for c in COHORTS]
            x = xs + (-.18 if j == 0 else .18)
            ax.bar(x, values, width=.33, color=colors[name], edgecolor='#243746', linewidth=.6, zorder=2)
            for seed, shape in enumerate(('o', '^', 's')):
                ys = [metric(seeds[name][seed], c)[key] * scale for c in COHORTS]
                ax.scatter(x + (seed - 1) * .045, ys, marker=shape, facecolor='white',
                           edgecolor='#243746', s=27, linewidth=.8, zorder=4)
            if panel == 0:
                for idx, value in enumerate(values):
                    top = max(metric(seeds[name][s], COHORTS[idx])[key] * scale for s in range(3))
                    ax.text(x[idx], top + 4, f'{value:.1f}', ha='center', fontsize=8.5)
        ax.set_xticks(xs, ['混合\n750题', '长距离\n250题', '接缝\n250题', '内部\n250题'])
        ax.tick_params(axis='x', length=0, pad=6)
        ax.set_ylabel(ylabel)
        titles.append(ax.set_title(title, loc='left'))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    axes[0].set_ylim(0, 110)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    max_sg = max(metric(seeds[name][s], c)['mean_sg_m'] for name in seeds for s in range(3) for c in COHORTS)
    axes[1].set_ylim(0, np.ceil((max_sg + 100) / 500) * 500)
    handles = [Patch(facecolor=colors[n], edgecolor='#243746', label='原M0' if n == 'M0' else '三步覆盖候选') for n in colors]
    handles.extend(Line2D([], [], marker=shape, linestyle='', markerfacecolor='white', markeredgecolor='#243746', label=f'权重{s}')
                   for s, shape in enumerate(('o', '^', 's')))
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .99), ncol=5, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.075, right=.985, top=.78, bottom=.20, wspace=.35)
    fig.text(.5, .035, f'10个已消费源；三冻结权重；{status}。完整恢复/损伤、SG及区域bootstrap见报告。', ha='center', fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lb = legend.get_window_extent(renderer)
    assert all(not lb.overlaps(t.get_window_extent(renderer)) for t in titles), 'legend/title overlap'
    files = []
    blobs = []
    for ext in ('png', 'pdf', 'svg'):
        buffer = BytesIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig.savefig(buffer, format=ext, dpi=300, bbox_inches='tight')
        assert not any('Glyph' in str(w.message) for w in caught), 'missing glyph'
        blobs.append((FIG / '数据结果图' / f'{STEM}.{ext}', buffer.getvalue()))
    plt.close(fig)
    for p, data in blobs:
        with p.open('xb') as f:
            f.write(data)
        files.append(dict(path=rel(p), sha256=digest(p)))
    d = diag['arms']['CueFull']['all']
    fail_rows = [r for r in read(RUN / '离线诊断/逐题诊断.json')['CueFull'] if not r['success']]
    opportunities = sum(r['actionable_neighborhood_cells'] for r in fail_rows) / len(fail_rows)
    no_adj = d['failed_episodes'] - d['failure_categories']['missed_actionable_adjacency']
    sr_ci = [100*v for v in effect['source95_SR']]
    sg_ci = [300*v for v in effect['source95_SG']]
    controls_text = ('条件性目标对照已执行，数值与独立复核见目标对照目录；新区域确认仍需另立冻结协议。'
        if verdict['controls_started'] else '主门槛未通过，按运行前协议未启动6750条条件性目标对照，也未下载或消费新的确认区域。')
    text = f'''# 三步覆盖接管开发验证报告

2026-10-03。**{status}**。本批固定混合队列SR为{candidate['metrics']['sr']*100:.2f}%，对原M0提高{effect['sr_gain']*100:.2f}个百分点；SG从{reference['metrics']['mean_sg_m']:.2f}米变为{candidate['metrics']['mean_sg_m']:.2f}米。完整数值与所有未通过项如下，原默认M0保持。

## 离线诊断与选型依据

先只读复核4500条M0 Full/Baseline旧记录、78525动作，没有产生新SR实验。939条Full失败中{no_adj}条未获得有剩余预算的目标邻接机会，4条获得过但未成功。失败轨迹平均邻接检验机会覆盖为{opportunities:.2f}格；覆盖只是几何机会，不代表目标已排除。

视觉线索在原M0配对中恢复498、损伤11条。356次接受但未立即命中，其中226次向真目标靠近、130次远离；首次非邻接线索分歧既见于60条成功恢复，也见于11条损伤，不能据此把所有非命中线索统一删除。未接受cue时方向头的合法方向进度率56.91%、均匀合法方向期望56.30%；这是后验诊断，不能当稳定远距离指引。

选择唯一候选Coverage3：无已接受cue时，每次真实观察枚举三步合法路径，新增邻接检验机会较原探索器首动作可实现的最佳路径至少多2格才接管；同分少重访、优先原动作和固定方向顺序。预算末步只计直接到达；每步重新规划，不提前读取未来图像或领取观察。目标头、原0.50、权重、均值及B20固定。

## 队列与全部结果

同10个已消费区域、750个固定任务，三份训练权重配对，新增2250条候选轨迹；原M0的2250条同题Full只按SHA复用。保留旧test字段供配对，但本批明确为后验开发。每权重混合750题、每层250题；合并后的分母分别2250/750。三个权重不是三个新增独立区域。

{chr(10).join(table)}

SG为含成功记0的终点曼哈顿距离。失败SG只对失败题求均值；实际移动距离与终点SG分开，见机器汇总。

{weight_table}

混合SR配对差95%区域bootstrap区间[{sr_ci[0]:+.2f}, {sr_ci[1]:+.2f}]个百分点，SG变化区间[{sg_ci[0]:+.2f}, {sg_ci[1]:+.2f}]米。4000次、seed7317，三权重先在区域内平均再整组抽取10区域。开发分数、区间及已查看数据均不能代替新源确认。

恢复{summary['recovered']}条原失败，损伤{summary['harmed']}条原成功，净增{summary['recovered']-summary['harmed']}条成功；所有配对保留，不只展示净增或某个权重。

## 冻结门槛与判定

{gate_table}

主要验收指标是固定混合SR，但同时保护各分层及SG。保留SR收益和距离代价，未在看到结果后改阈值、改分母或改门槛。{controls_text}

## 验证与范围

- 10项公开信息、cue优先、末步预算、边界、独立规划及分层损伤回归检查通过。最初模块式测试启动没有找到tests包；该次未执行实际测试，失败记录保留，随后按文件发现完成10项测试，两个记录均绑定。
- 离线诊断独立算术复核通过；新主评测2250轨迹/{audit['actions']}动作全部重放，通过原检查点及显式GRU/视觉头方程、独立集合式规划、预算、物理距离、配对和bootstrap核对。算术实现由同一开发会话运行，不冒称另一位人工审阅者。
- 本批0训练、0云端调用、默认保持，无新增地理区域；原S2/S3/S4及上一批空间隔离确认结论保留。Sat2Cap预训练地理覆盖未知；同Massachusetts发布影像，不是异时、跨模态或实飞结论。
- 本候选按单次冻结验证收束。新增因素与新确认须另立协议，现有10区不能再称未消费源。

[运行前协议](执行协议.md) / [判定](验收结论.json) / [主复核](主对照/独立复核.json) / [逐题配对](主对照/逐题配对.json) / [离线诊断](离线诊断/诊断汇总.json)。

![图47](../../../../../绘图/数据结果图/{STEM}.png)

图注：{caption}
'''
    write(REPORT, text)
    write(README, f'# 空间覆盖与误触发单因素验证 v1\n\n{status}；混合SR{candidate["metrics"]["sr"]*100:.2f}%，SG{candidate["metrics"]["mean_sg_m"]:.2f}米；默认保持。\n\n'
        '[完整报告](三步覆盖接管开发验证报告.md) / [判定](验收结论.json) / [独立复核](主对照/独立复核.json)。\n\n'
        '本批是已消费10区的后验开发；诊断4500旧记录，新主评测2250记录。条件性后续依据运行前协议，不覆盖旧结果。\n')
    write(CAPTION, '# 三步覆盖接管开发验证图注\n\n' + caption + '\n')
    source_paths.extend((Path(__file__), ROOT / 'project/src/documents/stage_figures.py'))
    item = dict(id=STEM, files=files, caption=caption, protocol='consumed10regions/750tasks/3weights; grid10/B20/300m; posthoc development',
                sources=[rel(p) for p in source_paths])
    manifest = dict(date='2026-10-03', source_sha256={rel(p): digest(p) for p in source_paths},
        reproduction_script=rel(Path(__file__)), script_sha256=digest(Path(__file__)),
        verdict=verdict, main_summary=summary, diagnostic_summary=diag, per_seed=seeds,
        figures=[item], other_outputs_sha256={rel(p): digest(p) for p in (REPORT, README, CAPTION)})
    write(MANIFEST, manifest)
    print(dict(report=rel(REPORT), figure=STEM, manifest=rel(MANIFEST), status=status), flush=True)


if __name__ == '__main__':
    main()
