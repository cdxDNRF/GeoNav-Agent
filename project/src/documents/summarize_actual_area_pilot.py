"""Evidence-bound native-area pilot report, figures and current navigation."""
from pathlib import Path
import hashlib
import json
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'DATA/processed_data/Masa/评测结果/实际区域M0兼容试跑_v1'
DATA = ROOT / 'DATA/processed_data/Masa/实际区域扩展_v1/工程数据'
FIG = ROOT / '绘图'
NAV = ROOT / '项目导航'
NAMES = ('CueFull', 'Baseline', 'CueMean', 'CueWrong', 'Frontier', 'FixedRegion')
LABELS = ('M0真实目标', '禁用目标线索', '目标均值', '错误目标图', 'Frontier', 'FixedRegion')


def read(p): return json.loads(p.read_text('utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()
def write(p, v):
    with p.open('x', encoding='utf-8') as out: json.dump(v, out, ensure_ascii=False, indent=2)
def text(p, v):
    with p.open('x', encoding='utf-8') as out: out.write(v)


def required():
    v = read(RUN / '验收结论.json')
    a = read(RUN / '独立复核.json')
    s = read(RUN / '对照汇总.json')
    r = read(RUN / '预登记.json')
    assert a['passed'] and v['audit_passed'] and v['execution_compatibility_passed']
    assert digest(RUN / '对照汇总.json') == v['summary_sha256'] and digest(RUN / '独立复核.json') == v['audit_sha256']
    assert digest(ROOT / 'project/local_policy_default.json') == v['default_sha256']
    assert a['records'] == 1050 and r['planned_total'] == 1050
    for field in ('protected_sha256', 'source_sha256', 'input_sha256', 'encoder_sha256', 'frozen_output_sha256'):
        assert all(digest(ROOT / p) == h for p, h in r[field].items())
    return v, a, s, r


def report(v, a, s, reg):
    arm = s['arms']['CueFull']
    m = arm['metrics']
    status = '工程兼容与目标证据均通过' if v['known_area_pilot_passed'] else '工程兼容通过，目标证据尚未全部通过'
    rows = []
    for name, label in zip(NAMES, LABELS):
        z = s['arms'][name]
        metric = z['metrics']
        seeds = ' / '.join(f'{x*100:.2f}%' for x in z['sr_by_seed']) if name in NAMES[:4] else '单次确定性执行'
        rows.append(f"| {label} | {metric['episodes']} | {seeds} | {metric['sr']*100:.2f}% | {metric['mean_sg_all_episodes']:.3f} | {metric['mean_sg_m']:.2f} | {metric['mean_valid_travel_m']:.2f} | {metric['repeat_visit_rate_micro']*100:.2f}% | {metric['out_of_bounds_rate']*100:.2f}% |")
    effects = []
    for name in NAMES[1:]:
        e = s['effects'][name]
        check = s['target_checks'].get(name)
        judge = ('通过' if all(check.values()) else '未通过') if check else '描述对照，无升级门槛'
        effects.append(f"| {name} | {e['sr_gain']*100:+.2f} | {e['positive_seeds']}/3 | {e['sg_change']*300:+.2f} | [{e['source95_SR'][0]*100:+.2f}, {e['source95_SR'][1]*100:+.2f}] | {judge} |")
    diag = arm['diagnostics']
    full_rows = [json.loads(line) for seed in range(3) for line in
                 (RUN / '神经对照' / f'M0_s{seed}_CueFull_轨迹.jsonl').read_text('utf-8').splitlines()]
    cross_tasks = sorted({row['episode_id'] for row in full_rows if any(d['actionable_true_adjacency'] and
                         d['true_adjacency_cross_source'] for d in row['evaluation_diagnostics'])})
    failed_rows = [row for row in full_rows if not row['success']]
    failed_sg_m = float(np.mean([row['sg_m'] for row in failed_rows])) if failed_rows else 0.
    write(RUN / '接缝诊断范围补充.json', dict(cross_source_unique_tasks=cross_tasks, cross_source_unique_task_count=len(cross_tasks),
          cross_source_weight_opportunities=diag['adjacency']['cross_source']['opportunities'],
          full_failed_records=len(failed_rows), mean_sg_m_failed_only=failed_sg_m,
          source_sha256={rel(RUN / '神经对照' / f'M0_s{seed}_CueFull_轨迹.jsonl'):
                        digest(RUN / '神经对照' / f'M0_s{seed}_CueFull_轨迹.jsonl') for seed in range(3)},
          posthoc_only=True, cross_seam_reliability_confirmed=False))
    seams = []
    for key, label in (('same_source', '同原图邻接'), ('cross_source', '跨原图边界邻接')):
        d = diag['adjacency'][key]
        rate = 'NA' if d['correct_acceptance_rate'] is None else f"{d['correct_acceptance_rate']*100:.2f}%"
        seams.append(f"| {label} | {d['opportunities']} | {d['accepted']} | {d['accepted_correct']} | {rate} |")
    fail_labels = {'never_within2': '预算内未到目标两格内', 'within2_never_adjacent': '到两格内但未邻接',
                   'adjacent_only_terminal': '预算耗尽时才邻接', 'missed_actionable_adjacency': '有预算邻接但未成功'}
    failures = '\n'.join(f"| {fail_labels[k]} | {diag['failure_categories'][k]} |" for k in fail_labels)
    maps = '\n'.join(f"| {source} | {x['episodes']} | {x['sr']*100:.2f}% | {x['mean_sg_m']:.2f} | {x['mean_valid_travel_m']:.2f} |" for source, x in arm['by_source'].items())
    distances = '\n'.join(f"| C{k} | {x['episodes']} | {x['sr']*100:.2f}% | {x['mean_sg_m']:.2f} |" for k, x in arm['by_distance'].items())
    accepted_n = diag['cue_accepts']
    acceptance = f"{diag['accepted_true_hits']}/{accepted_n}（{diag['accepted_true_hit_rate']*100:.2f}%）" if accepted_n else '0/0（NA）'
    nextstep = ('本轮足迹兼容试跑已完成，下一项应准备未参与训练/开发、空间隔离的连续区域，再按冻结M0开展确认。当前已知区域成绩不需要作为继续调参追分的理由。'
                if v['known_area_pilot_passed'] else '本轮固定结果后结束，先依据目标控制和失败分组提出一个有证据的机制假设；另行登记验证，不在本队列调整门槛。')
    text(RUN / '实际区域M0兼容试跑报告.md', f'''# 冻结M0在实际面积地图上的兼容试跑

**{status}：SR {m['sr']*100:.2f}%，全体SG {m['mean_sg_all_episodes']:.3f}格 / {m['mean_sg_m']:.2f}米，平均有效移动{m['mean_valid_travel_m']/1000:.3f}公里。** 三张已知训练区域各9 km²；本轮不是未知地理区域的正式泛化验收，也不替换原默认或重写S2/S3/S4结论。

## 冻结设计与全部结果

原三张3000×3000连续地图、1米/像素、300米/格、10×10/B20；原75题逐字节复用，73条不同起终点路线、两次重复抽样保留，C12—16每图各5题。原Small256 NoTarget探索器、EdgeTargetCue头、Sat2Cap、拟合均值和0.50阈值不训练、不调参；全部seed0/1/2、argmax、每步GRU、每题清空状态，没有F或云端协作。

四神经条件各75×3=225条（共900），两规则各75条（共150），全部1050题正常终局并留在分母。四控制无条件执行，没有按早期分数停止。规则保留原程序的10×10行为，FixedRegion允许越界原地耗步，不静默增加过滤；神经M0零越界。

| 条件 | 记录 | seed0 / 1 / 2 SR | 平均SR | 全体SG格 | 全体SG米 | 平均有效移动米 | 重访率 | 越界率 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

SG为全部终局的曼哈顿距离（成功为0），SG米=300×SG格，仍是网格距离，不是直线定位误差。真实目标的{len(failed_rows)}条失败单独平均SG为{failed_sg_m:.2f}米；全体316米等均值不能解释为每次失败都接近目标。有效移动米=300×合法实际移动次数；越界耗一步但物理移动0米。B20最多6公里、原10×10密度B20最多3公里，因此不能称物理航程与原密度实验相同。总有效移动等详细值在[汇总](对照汇总.json)；神经每条件总量包含三权重，规则各仅一次执行。

## 同题目标证据与规则差距

均值条件同时遮蔽global/local/edge目标通道；错图同时替换给定图像和三类特征，环境真目标保持。禁用cue只让原探索器执行。13/75题错误图无法匹配初始距离，例外在本轮开始前既已固定。

| Full相对参照 | SR差（百分点） | 正收益权重 | SG米差 | 三整地图配对SR差95%区间（点） | 原目标门槛 |
|---|---:|---:|---:|---|---|
{chr(10).join(effects)}

目标门槛为Full对每控制SR≥+5点、≥2/3权重为正且SG不更差，未设SR60%通用通行证。配对bootstrap以**三整张mosaic**为单位、图内平均三权重（seed5251、4000次）；只有三张已知训练地图，区间是描述统计，不能据此声称独立地理泛化。十二原源不当十二个独立地区，225权重记录也不当225张独立图。规则在配对中引用同75题一次执行，不是三次规则重复。

逐题恢复/损伤：{json.dumps(s['paired_recovery'], ensure_ascii=False)}。这些对照支持或反驳当前冻结策略是否利用给定目标图，不单独证明边缘特征的因果贡献；后者仍是之前单因素批次的结论。

## 接缝与失败阶段（仅评测端真值）

真实目标条件累计跨原1500米源边界{diag['crossings']}次，{diag['crossing_records']}/225条轨迹至少跨一次；这些穿越是地理相邻源图边界，不是额外传感器或时间变化。

| 有行动预算的真目标邻接 | 决策机会数 | cue接受数 | 接受且一步命中数 | 正确接受 / 机会 |
|---|---:|---:|---:|---:|
{chr(10).join(seams)}

**跨源邻接只有{diag['adjacency']['cross_source']['opportunities']}个权重机会，来自{len(cross_tasks)}道不同题**（{', '.join(cross_tasks) or '无'}）。本轮不能判定跨接缝识别普遍可靠；探索跨边界的474次穿越与识别目标的3次邻接不是同一个分母。该样本覆盖缺口应在下一批任务冻结前处理，而不是在本批结果出来后加题或挑成功例。

机会以行动前距离1格且有剩余预算计数，可能同轨迹重复；末步执行后才邻接不计机会。零机会组记录NA，不记100%。全体cue接受的一步真目标命中为{acceptance}；另{diag['accepted_not_true_hit']}次接受没有一步命中，这不能直接等同造成失败或净损伤。跨接缝和同源处是观察分组，不是随机干预；样本数/探索访问状态不同，不能自动把差异归因于接缝。

| Full最终失败类别 | 轨迹数（跨三权重） |
|---|---:|
{failures}

分类互斥，优先判有预算邻接，再判终局才邻接、两格内、从未两格内。真实位置/距离、源身份、跨接缝标签仅在评测端计算并存为`evaluation_diagnostics`，不进入Agent。Agent接收当前图、给定目标图、公开位置/预算/访问历史和仅这两张图的预计算特征；不持有全图或特征库。预算不足与探索路径可造成接近机会缺失，此诊断本身不证明唯一原因。

本批失败没有出现“有预算真邻接却未成功”；31条到两格内但未邻接、8条在终局才邻接、10条从未两格内，表明这些失败轨迹先缺少可执行目标邻接机会。这支持把后续探索/路径分配作为候选诊断方向，但未验证增加预算或某条新规则能恢复它们，也不说明视觉头在更复杂新地区已经足够。

## 地图与距离分组

| 完整地图 | 记录（3权重） | SR | SG米 | 平均有效移动米 |
|---|---:|---:|---:|---:|
{maps}

| 初始距离 | 记录（3权重） | SR | SG米 |
|---|---:|---:|---:|
{distances}

## 复核与下一项

19项测试通过（原环境11项、新试跑8项）。复核独立重建300个原生图块及JPEG，重提取global/local与edge profile逐项精确一致；重放900神经和150规则轨迹、{a['actions']}动作，另行重算策略输入/GRU与接受规则、图像SHA、均值/错图干预、规则BFS/区域顺序、手算移动/边界/首次到达/末步/SG米、接缝分组、失败类别、配对恢复损伤及地图区间。复核由本执行代理的独立实现完成，不声称不同人员审查。

{a['protected_files_verified']}份历史/原始/模型保护文件、{a['data_input_files_verified']}派生数据输入、{a['code_and_snapshots_verified']}源码与快照、三特征库SHA一致。零训练、零云端调用、无SwissView新源消耗；使用状态仍68已导航/32未导航。图块字节与同源旧5×5一致，三已知训练区域成绩可以验证协议适配，却不足以排除源图记忆或严格地理/预训练重叠。

{nextstep} 不能直接从原151张全部已纳入训练/验证/测试的图中再挑拼图并宣称未知地理确认；需要外部连续影像及坐标/分辨率/来源/空间隔离检查。任意角度、真实异时、跨视角/跨模态、真正多Agent与无人机实飞仍未完成。

[方案](执行协议.md) / [预登记](预登记.json) / [任务](导航任务.json) / [特征冻结](特征冻结结束.json) / [完整汇总](对照汇总.json) / [逐题配对](逐题配对.json) / [复核](独立复核.json) / [判定](验收结论.json)
''')
    text(RUN / 'README.md', '# 本批入口\n\n[兼容试跑报告](实际区域M0兼容试跑报告.md) / [完整汇总](对照汇总.json) / [复核](独立复核.json) / [判定](验收结论.json) / [冻结协议](执行协议.md)。三已知训练区域，不能当未知地理正式确认。\n')
    text(RUN / '下一项空间隔离连续区域确认准备.md', '''# 下一项：空间隔离连续区域的数据与协议准备（未启动）

本轮不继续在三已知工程地图上训练或选择阈值。先完成外部连续影像的可用性检查，再冻结M0确认。尚未下载新影像、调用云端或启动新导航。

候选影像需真实相邻坐标及明确来源/使用许可，保持约1米原生分辨率、每格300米（可记录可接受的分辨率偏差，但不能悄悄插值后称原生），覆盖至少3000×3000米。选择足够完整区域、以地理范围隔离开发与确认，并验证与已有Masa范围没有重叠；源文件不复用不替代空间隔离。若编码器预训练覆盖不能查明，明确保留这个边界。

先登记区域/尺度/空白质量/影像方向/接缝阈值与可用数，确认源一旦冻结不按导航结果换图。导航队列应预先覆盖源边界附近目标与同源目标，接缝识别不能只靠少数权重重复机会。保留三权重、同信息规则及目标控制、B20、SG格/米和有效移动米；正式样本规模及按区域成组的证据要求需在实际数据可用性检查后登记，不能从本批三图置信区间外推。

原151张Masa已用于训练/开发/测试，不能重新命名为未知区域；SwissView当前JPEG包缺少本批已核验的连续GeoTIFF坐标，不能凭文件编号拼接并称地理连续。实际面积兼容通过与未知地区正式确认分别记录。
''')


def plots(s, v):
    graphics.style()
    graphics.ITEMS.clear()
    colors = [graphics.BLUE, graphics.GRAY, graphics.ORANGE, graphics.TEAL, '#8A75A7', '#998363']
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 4.85), sharey=True)
    y = np.arange(6)
    for ax, field, scale, label, precision in ((axes[0], 'sr', 100, '(a) SR：成功率（%）', 2),
                                               (axes[1], 'mean_sg_m', 1, '(b) SG：全体终点距离（米）', 1),
                                               (axes[2], 'mean_valid_travel_m', .001, '(c) 平均有效移动（公里）', 2)):
        values = [s['arms'][name]['metrics'][field]*scale for name in NAMES]
        ax.barh(y, values, color=colors, height=.67, alpha=.95)
        for i, name in enumerate(NAMES[:4]):
            for seed, marker in enumerate(('o', '^', 's')):
                r = read(RUN / '神经对照' / f'M0_s{seed}_{name}_结果.json')
                ax.scatter(r['metrics'][field]*scale, i+(.13*(seed-1)), marker=marker, facecolor='white',
                           edgecolor='#26313B', s=19, linewidth=.7, zorder=4)
        for i, value in enumerate(values):
            ax.text(value+max(values)*.025, i, f'{value:.{precision}f}', va='center', fontsize=9)
        ax.set_xlim(0, 104 if field == 'sr' else max(values)*1.28)
        ax.set_title(label, loc='left', fontsize=10)
        ax.set_yticks(y, LABELS)
        ax.grid(axis='x', alpha=.7)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].legend(handles=[Line2D([], [], linestyle='', marker=z, markerfacecolor='white', markeredgecolor='#26313B',
                          markersize=4.5, label=f'权重{k}') for k, z in enumerate(('o', '^', 's'))],
                  loc='upper center', bbox_to_anchor=(1.73, -.12), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.155, right=.975, top=.87, bottom=.31, wspace=.29)
    fig.text(.54, .09, '三已知训练地图 × 75题；神经各三权重，规则各一次。白色标记为权重，非CI。', ha='center', fontsize=9)
    fig.text(.54, .035, '每图9 km²；300米/格、10×10/B20。SG含成功的0；有效移动计合法动作，越界0米。', ha='center', fontsize=9)
    caption = ('冻结M0在三已知Masa训练区域上的实际面积兼容试跑；每图3000×3000原生像素、9km²、300m/格，原75题/73不同路线。'
               '四神经条件各75×3=225，Frontier与原FixedRegion各75条，全部1050题完成。柱为同队列平均，白色圆/三角/方形为seed0/1/2；规则无权重点，不表示三重复。'
               'SG为所有终局的网格曼哈顿距离乘300米，并非直线定位误差；平均有效移动包含所有题，越界耗步但移动0米。'
               '已知区域工程和目标证据判定见报告，不能称未知地区正式泛化，也不与历史不同队列SR相减。')
    refs = [rel(RUN / p) for p in ('对照汇总.json', '验收结论.json', '独立复核.json')]
    graphics.save(fig, FIG / '数据结果图', '41_实际区域M0成功率与米制距离', caption, refs,
                  dict(maps=3, known_train=True, neural_records=900, rule_records=150, cell_size_m=300, area_km2_each=9))
    d = s['arms']['CueFull']['diagnostics']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
    groups = [d['adjacency'][key] for key in ('same_source', 'cross_source')]
    for i, group in enumerate(groups):
        rate = group['correct_acceptance_rate']
        if rate is None:
            axes[0].text(i, 2, 'NA：无机会', ha='center', fontsize=10)
        else:
            axes[0].bar(i, rate*100, color=(graphics.BLUE, graphics.TEAL)[i], width=.55)
            axes[0].text(i, rate*100+3, f"{group['accepted_correct']}/{group['opportunities']}\n{rate*100:.1f}%", ha='center', fontsize=10)
    axes[0].set_xticks([0, 1], ['同原图邻接', '跨原图边界邻接'])
    axes[0].set_ylim(0, 118)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel('接受且一步命中 / 有预算邻接机会（%）')
    axes[0].set_title('(a) Full目标线索的观察分组', loc='left')
    cats = ('never_within2', 'within2_never_adjacent', 'adjacent_only_terminal', 'missed_actionable_adjacency')
    labels = ('未到目标两格内', '到两格内但未邻接', '耗尽预算时才邻接', '有预算邻接但未成功')
    counts = [d['failure_categories'][key] for key in cats]
    axes[1].barh(np.arange(4), counts, height=.58, color=graphics.GRAY)
    for i, n in enumerate(counts): axes[1].text(n+.35, i, str(n), va='center', fontsize=10)
    axes[1].set_yticks(np.arange(4), labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(max(counts)*1.2, 2))
    axes[1].set_xlabel('失败轨迹数（225条权重记录内）')
    axes[1].set_title('(b) Full最终失败的互斥分类', loc='left')
    axes[0].grid(axis='y'); axes[1].grid(axis='x')
    for ax in axes: ax.set_axisbelow(True)
    fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.25, wspace=.85)
    fig.text(.5, .10, f"Full总失败{sum(counts)}条；有预算邻接机会可在同轨迹重复。终局才邻接不计机会。", ha='center', fontsize=9)
    fig.text(.5, .035, '跨源机会仅1题的3权重记录；真值只作评测诊断。分组差异不证明接缝因果效应。', ha='center', fontsize=9)
    caption = ('真实目标M0的事后接缝和失败诊断。左图按当前/目标是否来自同1500源图划分真邻接机会，只有行动前距目标1格且有预算才纳入；'
               '柱为接受并一步命中/机会数，标明分子/分母；同轨迹可能重复机会，零机会记NA。右图失败互斥分组，优先判有预算邻接，再终局才邻接、两格内、从未两格内。'
               '跨源机会仅同1题的3权重，不能支持可靠性普遍确认。225记录来自75题×3权重而非225独立地区；真实位置/距离/源身份未进入Agent，跨接缝分组不是随机干预，不能推断接缝因果效应。')
    graphics.save(fig, FIG / '数据结果图', '42_实际区域接缝线索与失败阶段', caption, refs,
                  dict(maps=3, tasks=75, weights=3, condition='CueFull', posthoc_only=True, causal_seam_claim=False))
    for item in graphics.ITEMS: item['reproduction_script'] = rel(Path(__file__))
    items = graphics.ITEMS
    write(FIG / '绘图数据/实际区域M0兼容试跑_v1.json', dict(date='2026-10-02', figures=items,
          script_sha256=digest(Path(__file__)), source_sha256={p: digest(ROOT / p) for i in items for p in i['sources']}, summary=s))
    text(FIG / '实际区域M0兼容试跑图注.md', '\n\n'.join('# '+i['id']+'\n\n'+i['caption'] for i in items)+'\n')
    catalog = read(FIG / '图表来源清单.json')
    assert not {i['id'] for i in items} & {i['id'] for i in catalog['figures']}
    catalog['figures'].extend(items)
    (FIG / '图表来源清单.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    text(FIG / '审阅/实际区域M0兼容试跑设计_v1.md', '图41并列同题SR、全体SG米和实际有效移动，不重复规则权重点；图42显示真邻接接受分子/分母和互斥失败类别。三已知训练地图/73不同路线范围明确，接缝分析为事后分组，不作因果和未知地理泛化声称。PNG300dpi、PDF嵌入字体、SVG文字可编辑；推荐插入宽度≥20cm。另保存导出和视觉核验。\n')
    (FIG / '审阅/实际区域M0兼容试跑_作图源码_v1.py').write_bytes(Path(__file__).read_bytes())


def navigation(s, v, a):
    m = s['arms']['CueFull']['metrics']
    target = '通过' if v['target_evidence_passed'] else '未全部通过'
    summary = (f"冻结M0在三张已知训练连续区域（每图9 km²、300米/格、10×10/B20）的75题×3权重中，SR {m['sr']*100:.2f}%、SG {m['mean_sg_all_episodes']:.3f}格 / {m['mean_sg_m']:.2f}米，平均有效移动{m['mean_valid_travel_m']/1000:.3f}公里。"
               f"四神经条件与两规则全部1050轨迹/{a['actions']}动作重放通过，三个目标控制{target}。本批支持已知区域兼容；未知地理正式确认未完成，原默认及S2/S3/S4保持。")
    links = f'[报告]({rel(RUN / "实际区域M0兼容试跑报告.md")}) / [判定]({rel(RUN / "验收结论.json")}) / [复核]({rel(RUN / "独立复核.json")}) / [科研图](绘图/实际区域M0兼容试跑图注.md)。'
    p = ROOT / 'README.md'
    value = p.read_text('utf-8')
    value = value.replace('实际搜索足迹的数据与协议工程已通过，下一项为冻结M0兼容试跑。', '实际搜索足迹的冻结M0兼容试跑已完成，下一项为未知连续区域的数据与隔离协议准备。', 1)
    value = value.replace('## 最新阶段：实际搜索足迹的数据与协议工程通过', '## 最新阶段：实际面积冻结M0兼容试跑完成\n\n'+summary+'\n\n'+links+'\n\n## 前置数据：实际搜索足迹的数据与协议工程通过', 1)
    value = value.replace('当前10×10是在同一原图足迹内增加切图密度，尚未扩大实际覆盖面积。', '原正式S4的10×10只提高原图内密度；新增实际面积试跑保持300米/格并扩大至每图9 km²，使用三张已知训练区域，未知地区正式确认尚未完成。')
    value = value.replace('跨视角/跨模态目标、更大实际区域和无人机实飞尚待验证。', '跨视角/跨模态目标、未知地区的大范围验证和无人机实飞尚待验证。')
    value = value.replace('后续使用新实际足迹数据单独登记冻结M0兼容试跑。', '新增实际足迹1050条兼容试跑完成，后续优先准备空间隔离连续新区域。')
    value = value.replace('62批实验', '63批实验').replace('40张实验与数据协议图', '42张实验与数据协议图')
    p.write_text(value, encoding='utf-8')
    p = NAV / '当前进度.md'
    value = p.read_text('utf-8').replace('实际足迹数据工程现已通过，下一项为冻结M0兼容试跑。', '实际足迹冻结M0兼容试跑已完成，下一项准备空间隔离的新连续区域。', 1)
    value = value.replace('10×10只增加原图内密度；真实异时、任意角度、跨视角/跨模态、更大实际范围和实飞未完成。', '原正式S4的10×10只增加密度；新增9 km²实际区域试跑已完成，但未知地理的实际范围正式确认、真实异时、任意角度、跨视角/跨模态和实飞未完成。')
    value += '\n## 实际面积扩展：冻结M0兼容试跑\n\n'+summary+'\n\n'+links.replace('](', '](../')+'\n'
    value += '\n本轮零训练、零云端调用，SwissView使用状态未改变。先准备未参与训练/开发且空间隔离的连续影像；源文件不复用不等同地理独立。\n'
    p.write_text(value, encoding='utf-8')
    idx = read(NAV / '实验索引.json')
    assert len(idx['batches']) == 62 and not any(x['path'] == rel(RUN) for x in idx['batches'])
    idx['batches'].append(dict(dataset='Masa', kind='评测结果', name=RUN.name, category='实际面积兼容试跑',
         status='工程通过；目标控制'+target+'；已知训练地区', scope=f'3已知地图/75×3/9km²各图；SR{m["sr"]*100:.2f}%/SG{m["mean_sg_m"]:.2f}m；1050全重放',
         path=rel(RUN), reports=[rel(RUN / '实际区域M0兼容试跑报告.md')], evidence=[rel(RUN / n) for n in ('验收结论.json', '独立复核.json', '最终状态.json')]))
    (NAV / '实验索引.json').write_text(json.dumps(idx, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    p = NAV / '实验索引.md'
    value = p.read_text('utf-8') + f'| Masa / 评测结果 | {RUN.name} | 工程与目标控制{target}；已知地区 | 3图75×3/9 km²各图；[报告](../{rel(RUN / "实际区域M0兼容试跑报告.md")}) / [复核](../{rel(RUN / "独立复核.json")}) / [原目录](../{rel(RUN)}) |\n'
    p.write_text(value, encoding='utf-8')
    for name in ('README.md', '目录结构与维护.md'):
        p = NAV / name
        p.write_text(p.read_text('utf-8').replace('62批', '63批'), encoding='utf-8')
    p = FIG / 'README.md'
    first, rest = p.read_text('utf-8').split('\n', 1)
    p.write_text(first+'\n\n## 最新：实际区域M0兼容试跑\n\n'+summary+'\n\n[图41：SR与米制距离](数据结果图/41_实际区域M0成功率与米制距离.pdf) / [图42：接缝与失败](数据结果图/42_实际区域接缝线索与失败阶段.pdf) / [图注](实际区域M0兼容试跑图注.md) / [数据与SHA](绘图数据/实际区域M0兼容试跑_v1.json)。同目录含300dpi PNG及可编辑SVG。\n'+rest, encoding='utf-8')


def main():
    assert not (RUN / '实际区域M0兼容试跑报告.md').exists(), 'immutable reporting already exists'
    assert not any((FIG / '数据结果图' / (n+'.'+ext)).exists() for n in ('41_实际区域M0成功率与米制距离', '42_实际区域接缝线索与失败阶段') for ext in ('png', 'pdf', 'svg'))
    v, a, s, r = required()
    report(v, a, s, r)
    plots(s, v)
    navigation(s, v, a)
    write(RUN / '最终状态.json', dict(completed=True, audit_passed=True, known_area_pilot_passed=v['known_area_pilot_passed'],
          formal_unknown_geography_passed=False, records=1050, new_training_steps=0, cloud_calls=0, default_changed=False,
          report_sha256=digest(RUN / '实际区域M0兼容试跑报告.md'), verdict_sha256=digest(RUN / '验收结论.json'),
          audit_sha256=digest(RUN / '独立复核.json'), next='prepare spatially separated external continuous regions'))
    print(dict(report=True, new_figures=2, index_entries=63, target_evidence_passed=v['target_evidence_passed']), flush=True)


if __name__ == '__main__':
    main()
