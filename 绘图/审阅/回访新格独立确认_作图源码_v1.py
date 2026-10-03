"""Evidence-bound reporting of the single unused-source F confirmation."""
from pathlib import Path
import hashlib
import json
import statistics
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'DATA/processed_data/SwissView/评测结果/回访新格筛选独立源图确认_v1'
FIG = ROOT / '绘图'
NAV = ROOT / '项目导航'
USAGE = ROOT / '选题报告相关/SwissView源文件使用状态_2026-10-02_v2.json'


def read(p): return json.loads(p.read_text('utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()
def write(p, value):
    with p.open('x', encoding='utf-8') as output: json.dump(value, output, ensure_ascii=False, indent=2)
def text(p, value):
    with p.open('x', encoding='utf-8') as output: output.write(value)


def required():
    verdict = read(RUN / '验收结论.json')
    audit = read(RUN / '独立复核.json')
    summary = read(RUN / '主对照汇总.json')
    reg = read(RUN / '预登记.json')
    assert verdict['audit_passed'] and audit['passed']
    assert verdict['summary_sha256'] == digest(RUN / '主对照汇总.json')
    assert verdict['audit_sha256'] == digest(RUN / '独立复核.json')
    assert digest(ROOT / 'project/local_policy_default.json') == reg['default_sha256']
    for arm in ('M0', 'F'):
        results = [read(RUN / '主对照' / f'{arm}_s{s}_CueFull_结果.json') for s in range(3)]
        for metric, field in [('sr', 'sr'), ('mean_sg_all_episodes', 'sg')]:
            values = [r['metrics'][metric] for r in results]
            assert values == summary['arms'][arm][field + '_by_seed']
            assert abs(statistics.mean(values) - summary['arms'][arm][field + '_mean']) < 1e-12
    target = read(RUN / '目标证据汇总.json') if verdict['target_controls_started'] else None
    return verdict, audit, summary, reg, target


def report(v, a, s, reg, target):
    status = '独立确认通过' if v['independent_candidate_confirmed'] else '独立升级未通过，保留原M0'
    e = s['effect']
    sr = lambda x: f"{x*100:.2f}%"
    seeds = lambda x: ' / '.join(sr(t) for t in x)
    rows = '\n'.join(f"| {name} | {seeds(s['arms'][name]['sr_by_seed'])} | {sr(s['arms'][name]['sr_mean'])} | {s['arms'][name]['sg_mean']:.3f} |" for name in ('M0', 'F'))
    gate_rows = '\n'.join(f"| {name} | {'通过' if value else '未通过'} |" for name, value in s['checks'].items())
    if target:
        controls = '\n'.join(f"| {name} | {sr(r['sr_mean'])} | {r['sg_mean']:.3f} |" for name, r in target['arms'].items())
        target_text = f"""主门槛通过后执行4500条控制。Full为{sr(s['arms']['F']['sr_mean'])}/SG{s['arms']['F']['sg_mean']:.3f}；控制共享相同F规则。

| F条件 | 平均SR | 全体SG |
|---|---:|---:|
{controls}

目标控制判定：{json.dumps(target['checks'], ensure_ascii=False)}。均值/错误目标同时处理global/local/edge通道，环境目标保持真实；错图例外在首次导航前固定。
"""
    else:
        target_text = '主门槛未全部通过，按预登记没有追加4500条目标对照。此前开发目标证据仍是历史结果，本批不能声称已完成新的目标贡献验收。'
    wrong = read(RUN / '错误目标计划.json')
    exceptions = sum(not x['matched_distance'] for x in wrong.values())
    final_decision = '下一项可使用已确认F的独立10×10运行配置；具体采用以本批默认采用记录为准。' if v['independent_candidate_confirmed'] else '本因素开发增益没有达到独立升级要求，固定结果后收束，不在此队列补调参数。原M0继续作为默认；下一项转向保持图块尺度的实际区域扩展数据准备。'
    text(RUN / '独立源图确认报告.md', f'''# 回访新格筛选F：新源文件确认

**{status}。** 本批固定SwissView100编号60—79的20张此前未导航源文件；500题×三原训练权重、10×10/B20。与上一批Masa开发250题属于不同队列，不能直接把两批SR连成累计提升曲线。

## 同题主对照与判定

| 方案 | seed0 / seed1 / seed2 SR | 平均SR | 全体SG |
|---|---|---:|---:|
{rows}

F−M0的SR差为{e['sr_gain']*100:+.2f}个百分点，{e['positive_seeds']}/3权重正收益，SG差为{e['sg_change']:+.3f}格。20源文件成组、源内平均三权重的配对SR差95%区间[{e['source95_SR'][0]*100:+.2f},{e['source95_SR'][1]*100:+.2f}]点；SG差区间[{e['source95_SG'][0]:+.3f},{e['source95_SG'][1]:+.3f}]格。bootstrap seed5251、4000次。

| 冻结升级条件 | 本批结果 |
|---|---|
{gate_rows}

F实际介入{s['triggered_records']}/1500条真实目标轨迹，改变{s['changed_actions']}个动作；恢复{s['recovery']['recovered']}条、损伤{s['recovery']['harmed']}条，净{s['recovery']['recovered']-s['recovery']['harmed']:+d}/1500。每权重恢复/损伤：{json.dumps(s['recovery_by_seed'], ensure_ascii=False)}。源与距离分组保存在主汇总和各权重结果，所有计划终局均纳入SR分母。

## 目标图证据

{target_text}

固定错误目标计划中{exceptions}/500题无法匹配初始距离；无结果驱动的替换。

## 数据隔离、复核和结论范围

运行前实际扫描{reg['source_usage']['historical_records']}条历史导航记录，48已用源与原登记一致。新20源与已用源的原始字节/解码RGB没有重复；任务/错图、候选函数、原权重/拟合均值/0.50阈值、协议和源码在推理前冻结。2000图块/特征在导航前冻结；从原RGB逐张重建JPEG，并独立精确重算global/local特征与edge profile。

13项相关测试通过（本批6项、原公开状态规则7项）。本批{a['records']}条新轨迹、{a['actions']}个动作全部从冻结权重与环境逐步复核；另行选择新格、检查GRU、目标干预、手算几何/首次到达/末步成功/SG、恢复损伤、分组指标与配对区间。{a['protected_files_verified']}份历史文件、{a['derived_input_files_verified']}份派生输入及冻结源码保持。测试启动器的同名模块冲突记录保留，修正入口后测试通过，未改变冻结候选。复核由本执行代理的独立实现完成，不声称不同人员审查。

本轮零训练、零云端调用。60—79号在评测后计入已用，无论是否升级；累计68源已导航、32尚未。旧源使用状态与各正式批次原件不回写。5×5默认没有经过F，也没有新增5×5成绩；历史S2/S3/S4及SwissView5×5结论继续有效。

源文件隔离不等同严格地理无重叠或编码器预训练未见。10×10仍是在原图足迹内增加密度；不能称实际面积扩展、真实异时、跨视角、任意角度或多Agent验证。

{final_decision}

[方案](执行协议.md) / [预登记](预登记.json) / [源图核查](源图使用核查.json) / [主对照](主对照汇总.json) / [恢复损伤](逐题恢复与损伤.json) / [复核](独立复核.json) / [判定](验收结论.json)
''')
    text(RUN / 'README.md', '# 本批入口\n\n[独立确认报告](独立源图确认报告.md) / [判定](验收结论.json) / [复核](独立复核.json) / [协议](执行协议.md)。\n')
    write(RUN / '最终状态.json', dict(completed=True, audit_passed=True, independent_candidate_confirmed=v['independent_candidate_confirmed'],
          new_navigation_records=a['records'], new_source_files=20, new_training_steps=0, cloud_calls=0,
          original_grid5_default_changed=False, report_sha256=digest(RUN / '独立源图确认报告.md'),
          verdict_sha256=digest(RUN / '验收结论.json'), audit_sha256=digest(RUN / '独立复核.json'),
          state='confirmed_pending_grid10_release' if v['independent_candidate_confirmed'] else 'factor_closed_default_M0_retained'))
    previous = read(ROOT / '选题报告相关/SwissView源文件使用状态_2026-10-02_v1.json')
    ids = sorted({int(x) for x in previous['navigation_evaluated_ids']} | set(range(60, 80)))
    previous.update(navigation_evaluated_ids=[f'{i:02d}' for i in ids], not_yet_navigation_evaluated_ids=[f'{i:02d}' for i in range(100) if i not in ids],
                    navigation_evaluated_sources=68, not_yet_navigation_evaluated_sources=32,
                    newly_consumed_confirmation_ids=[str(i) for i in range(60, 80)], latest_confirmation=rel(RUN),
                    latest_verdict_sha256=digest(RUN / '验收结论.json'), latest_final_state_sha256=digest(RUN / '最终状态.json'),
                    note='IDs60..79 are consumed regardless of the upgrade decision; old v1 registry preserved. Source-file isolation only.')
    write(USAGE, previous)


def plots(v, s, target):
    graphics.style(); graphics.ITEMS.clear()
    rows = [s['arms']['M0'], s['arms']['F']]
    labels = ['M0\n原默认', 'F\n回访新格筛选']
    e = s['effect']
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4)); graphics.frames(axes)
    for ax, field, scale, ylabel in [(axes[0], 'sr', 100, 'SR：成功率（%）'), (axes[1], 'sg', 1, 'SG：全体终点距离（格）')]:
        graphics.bars(ax, labels, [r[field+'_mean']*scale for r in rows], [graphics.GRAY, graphics.BLUE],
                      decimals=2 if scale == 100 else 3, percent=scale == 100, seeds=[[x*scale for x in r[field+'_by_seed']] for r in rows])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105 if scale == 100 else max(max(r['sg_by_seed']) for r in rows)*1.25 + .05)
    axes[0].set_title(f"(a) SR差{e['sr_gain']*100:+.2f}个百分点", loc='left')
    axes[1].set_title(f"(b) SG差{e['sg_change']:+.3f}格", loc='left')
    fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.27, wspace=.30)
    decision = '独立确认通过' if v['independent_candidate_confirmed'] else '未过升级门槛；原默认保持'
    fig.text(.5, .11, 'SwissView 60—79号20新源文件；500题×3权重；10×10/B20。标记为权重，非置信区间。', ha='center', fontsize=9)
    fig.text(.5, .04, f"源文件配对SR差95%区间：{e['source95_SR'][0]*100:+.2f}至{e['source95_SR'][1]*100:+.2f}点。{decision}。", ha='center', fontsize=9)
    caption = f"固定20新SwissView源文件的同题对照：M0 SR{s['arms']['M0']['sr_mean']*100:.2f}%/SG{s['arms']['M0']['sg_mean']:.3f}，F SR{s['arms']['F']['sr_mean']*100:.2f}%/SG{s['arms']['F']['sg_mean']:.3f}。SR差{e['sr_gain']*100:+.2f}点，{e['positive_seeds']}/3权重正；源文件配对95%区间[{e['source95_SR'][0]*100:+.2f},{e['source95_SR'][1]*100:+.2f}]点。{decision}。500题×3冻结权重、10×10/B20；圆/三角/方形为seed0/1/2，非CI；SG包括成功的0，网格密度增加不代表实际面积扩展。"
    graphics.save(fig, FIG / '数据结果图', '38_回访新格筛选新源确认', caption,
                  [rel(RUN / name) for name in ('主对照汇总.json', '验收结论.json', '独立复核.json')],
                  dict(sources=20, planned_each_weight=500, weights=3, grid=10, budget=20, source_files_independent=True, geographic_area_expanded=False))
    if target:
        rows = [s['arms']['F']] + [target['arms'][c] for c in ('Baseline', 'CueMean', 'CueWrong')]
        labels = ['真实目标', '禁用cue', '均值目标', '错误目标']
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4)); graphics.frames(axes)
        for ax, field, scale in [(axes[0], 'sr', 100), (axes[1], 'sg', 1)]:
            graphics.bars(ax, labels, [r[field+'_mean']*scale for r in rows], [graphics.BLUE, graphics.GRAY, graphics.ORANGE, graphics.TEAL],
                          decimals=2 if scale == 100 else 3, percent=scale == 100, seeds=[[x*scale for x in r[field+'_by_seed']] for r in rows])
            ax.set_ylim(0, 105 if scale == 100 else max(max(r['sg_by_seed']) for r in rows)*1.25 + .05)
            ax.set_ylabel('SR（%）' if scale == 100 else 'SG：全体终点距离（格）')
        axes[0].set_title('(a) 冻结F的目标图对照', loc='left'); axes[1].set_title('(b) 同规则终点距离', loc='left')
        fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.22, wspace=.30)
        fig.text(.5, .075, '相同20新源、500题×3权重；10×10/B20。标记为权重，非CI；环境真目标保持。', ha='center', fontsize=9)
        caption = '新源目标证据，四条件均使用相同F公开回访筛选；均值/错误同时处理全局、局部、边缘目标通道，评测环境目标不变。三权重标记不表示置信区间；分组区间及错误目标匹配例外见报告。'
        graphics.save(fig, FIG / '数据结果图', '39_回访新格筛选新源目标证据', caption,
                      [rel(RUN / '目标证据汇总.json'), rel(RUN / '主对照汇总.json')], dict(sources=20, planned_each_weight=500, weights=3, grid=10, budget=20))
    items = graphics.ITEMS
    for item in items: item['reproduction_script'] = rel(Path(__file__))
    write(FIG / '绘图数据/回访新格独立源图确认_v1.json', dict(date='2026-10-02', figures=items, script_sha256=digest(Path(__file__)),
          source_sha256={p: digest(ROOT / p) for item in items for p in item['sources']}, primary_summary=s, target_summary=target))
    text(FIG / '回访新格独立确认图注.md', '\n\n'.join('# ' + item['id'] + '\n\n' + item['caption'] for item in items) + '\n')
    catalog = read(FIG / '图表来源清单.json')
    assert not {i['id'] for i in items} & {i['id'] for i in catalog['figures']}
    catalog['figures'].extend(items)
    (FIG / '图表来源清单.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    text(FIG / '审阅/回访新格独立确认设计_v1.md', '科研图使用与既有图相同字体和颜色规范，轴从0起，PDF/SVG矢量导出、PNG300dpi。圆/三角/方形表示三权重，源文件区间在图注单独报告。源文件独立、相同足迹密度协议与正式地理扩展分别标明。建议插入宽度至少18cm；像素审阅另记。\n')
    (FIG / '审阅/回访新格独立确认_作图源码_v1.py').write_bytes(Path(__file__).read_bytes())
    return len(items)


def navigation(v, s, a, figure_count):
    outcome = '新源确认通过' if v['independent_candidate_confirmed'] else '新源升级未通过，原M0保持'
    e = s['effect']
    summary = f"SwissView固定60—79号20新源、500题×3权重，10×10/B20：M0 SR{s['arms']['M0']['sr_mean']*100:.2f}%/SG{s['arms']['M0']['sg_mean']:.3f}，F {s['arms']['F']['sr_mean']*100:.2f}%/{s['arms']['F']['sg_mean']:.3f}；SR差{e['sr_gain']*100:+.2f}点、{e['positive_seeds']}/3权重正，源文件95%差[{e['source95_SR'][0]*100:+.2f},{e['source95_SR'][1]*100:+.2f}]点。{outcome}。恢复{s['recovery']['recovered']}、损伤{s['recovery']['harmed']}；{a['records']}轨迹/{a['actions']}动作与2000图块/特征复核通过。"
    latest = '## 最新结果：回访新格筛选独立确认\n\n' + summary + '\n\n[报告](' + rel(RUN / '独立源图确认报告.md') + ') / [判定](' + rel(RUN / '验收结论.json') + ') / [科研图](绘图/回访新格独立确认图注.md)。本轮零训练、零云端；累计68源已导航、32尚未。\n\n'
    path = ROOT / 'README.md'; value = path.read_text('utf-8')
    value = value.replace('## 最新开发结果：', latest + '## 上一批开发结果：', 1)
    value = value.replace('F尚待未参与选型的新源确认。', 'F开发成绩的后续独立判定见本页最新结果。')
    value = value.replace('全部60批实验索引', '全部61批实验索引').replace('58批实验入口', '61批实验入口')
    value = value.replace('37张实验结果图', f'{37+figure_count}张实验结果图')
    value = value.replace('累计48源文件已跑导航、52尚未跑', '累计68源文件已跑导航、32尚未跑')
    value = value.replace('SwissView源文件使用状态_2026-10-02_v1.json', 'SwissView源文件使用状态_2026-10-02_v2.json')
    value = value.replace('下一项适合固定候选做新源确认。', '本轮已完成固定候选新源确认，判定见最新结果。')
    value = value.replace('本轮新增3000条本地CPU导航，无训练、无API、无新源文件；原默认保持。', f'最近独立确认新增{a["records"]}条本地CPU导航、20源文件，无训练、无API；后续优先实际区域扩展数据准备。')
    path.write_text(value, encoding='utf-8')
    path = NAV / '当前进度.md'; value = path.read_text('utf-8')
    value = value.replace('SwissView累计48已导航、52尚未，最新40—59号已消费', 'SwissView累计68已导航、32尚未，最新60—79号已消费')
    value = value.replace('SwissView源文件使用状态_2026-10-02_v1.json', 'SwissView源文件使用状态_2026-10-02_v2.json')
    value = value.replace('原默认保持，下一项固定候选做未参与选型的新源确认。', '开发时原默认保持；随后已完成新源确认，结论见下方。')
    value += '\n## 最新独立确认\n\n' + summary + '\n\n[新源报告](../' + rel(RUN / '独立源图确认报告.md') + ') / [判定](../' + rel(RUN / '验收结论.json') + ')。本轮冻结确认到此结束，下一项优先实际覆盖范围的数据准备。\n'
    path.write_text(value, encoding='utf-8')
    index = read(NAV / '实验索引.json'); assert len(index['batches']) == 60
    index['batches'].append(dict(dataset='SwissView', kind='评测结果', name=RUN.name, category='独立源文件确认', status=outcome,
                                 scope='20新源/500×3/10×10B20；零训练和云端', path=rel(RUN), reports=[rel(RUN / '独立源图确认报告.md')],
                                 evidence=[rel(RUN / name) for name in ('验收结论.json', '独立复核.json', '最终状态.json')]))
    (NAV / '实验索引.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path = NAV / '实验索引.md'; value = path.read_text('utf-8')
    value += f'| SwissView / 评测结果 | {RUN.name} | {outcome} | 20新源500×3；[报告](../{rel(RUN / "独立源图确认报告.md")}) / [复核](../{rel(RUN / "独立复核.json")}) / [原目录](../{rel(RUN)}) |\n'
    path.write_text(value, encoding='utf-8')
    for name in ('README.md', '目录结构与维护.md'):
        path = NAV / name; path.write_text(path.read_text('utf-8').replace('60批', '61批'), encoding='utf-8')
    path = FIG / 'README.md'; value = path.read_text('utf-8'); first, rest = value.split('\n', 1)
    value = rest.replace('新增候选未做独立确认，默认保持。', '本段为开发时结论；后续新源确认见最新图38。')
    path.write_text(first + '\n\n## 最新：回访新格筛选新源确认\n\n' + summary + '\n\n[图38 PDF](数据结果图/38_回访新格筛选新源确认.pdf) / [图注](回访新格独立确认图注.md) / [数据与SHA](绘图数据/回访新格独立源图确认_v1.json)。同目录含300dpi PNG、矢量PDF与可编辑SVG。\n' + value, encoding='utf-8')
    path = ROOT / '中期报告相关/README.md'; value = path.read_text('utf-8')
    value = value.replace('候选尚待独立确认，未替换默认，也未回写上述定稿材料。', '后续独立确认状态以当前进度为准；新增实验未回写上述定稿材料。')
    path.write_text(value, encoding='utf-8')


def main():
    assert not (RUN / '独立源图确认报告.md').exists(), 'immutable closeout exists'
    assert not any((FIG / '数据结果图' / f'38_回访新格筛选新源确认.{ext}').exists() for ext in ('png', 'pdf', 'svg'))
    v, a, s, reg, target = required()
    report(v, a, s, reg, target)
    count = plots(v, s, target)
    navigation(v, s, a, count)
    print(dict(report=True, figures=count, sources_consumed=20, independent_candidate_confirmed=v['independent_candidate_confirmed']), flush=True)


if __name__ == '__main__': main()
