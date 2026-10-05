"""Publish DATA-007 from saved evidence only; never run models or overwrite outputs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'DATA/processed_data/MassGIS/工程准备/同产品开发导航兼容_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def metric(rs):
    n = len(rs)
    return dict(planned=n, successes=sum(r['success'] for r in rs),
                SR=sum(r['success'] for r in rs)/n,
                SG_m=sum(r['sg_m'] for r in rs)/n,
                movement_m=sum(r['valid_travel_m'] for r in rs)/n,
                steps=sum(r['steps'] for r in rs),
                out_of_bounds=sum(r['out_of_bounds'] for r in rs))


def acceptance(rs):
    ds = [d for r in rs for d in r['evaluation_diagnostics']]
    acc = [d for d in ds if d['cue_accepted']]
    return dict(actions=len(ds), accepted=len(acc),
                correct_true=sum(d['accepted_true_hit'] for d in acc),
                correct_given=sum(d['accepted_given_hit'] for d in acc),
                precision_true=sum(d['accepted_true_hit'] for d in acc)/len(acc) if acc else None,
                precision_given=sum(d['accepted_given_hit'] for d in acc)/len(acc) if acc else None)


def probe(k, condition, selected, seed=None):
    labels = np.array([r['label'] for r in read(OUT/f'元数据/grid{k}视觉探针.json')])
    pos = (labels < 4) & selected
    accepted = correct = 0
    seeds = range(3) if seed is None else [seed]
    for s in seeds:
        p = np.load(OUT/f'特征/probe_g{k}_s{s}_{condition}.npy', allow_pickle=False)
        top = p.argmax(axis=1)
        take = (top < 4) & (p[np.arange(len(top)), top] >= .5) & selected
        accepted += int(take.sum())
        correct += int((take & (top == labels) & pos).sum())
    positives = int(pos.sum()) * len(seeds)
    return dict(pairs_each_seed=int(selected.sum()), accepted=accepted, correct=correct,
                adjacent_pairs=positives, precision=correct/accepted if accepted else None,
                recall=correct/positives if positives else None)


def pct(value):
    return '—' if value is None else f'{100*value:.2f}%'


def main():
    names = ['兼容补充分层结果.json', '同产品开发导航兼容报告.md',
             '下一项正式导航协议准备草案.md', '元数据/开发导航消费账本.json', '核验/交付生成绑定.json']
    if any((OUT/n).exists() for n in names):
        raise ValueError('Delivery already exists or is partial; no overwrite')
    verdict = read(OUT/'验收结论.json')
    audit = read(OUT/'核验/独立导航兼容复核_v2.json')
    reg = read(OUT/'核验/预登记.json')
    if not audit['passed'] or sha(OUT/'核验/独立导航兼容复核_v2.json') != verdict['audit_sha256']:
        raise ValueError('Completed, bound audit required')
    if not (verdict['technical_compatibility_passed'] and verdict['target_mechanism_development_gate_passed']):
        raise ValueError('This success-report template requires both frozen gates to pass')
    elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(reg['utc'])).total_seconds()
    if elapsed > reg['limits']['wall_seconds']:
        raise ValueError('Registered execution/audit wall limit exceeded')

    inputs = {OUT/'验收结论.json', OUT/'核验/独立导航兼容复核_v2.json', OUT/'核验/预登记.json'}
    neural = {}
    for k, policies in [(5, ['M0']), (10, ['M0', 'Coverage3Radial'])]:
        for policy in policies:
            for condition in ('CueFull', 'Baseline', 'CueMean', 'CueWrong'):
                files = [OUT/('主对照' if condition == 'CueFull' else '目标对照')/
                         f'g{k}_{policy}_s{s}_{condition}.jsonl' for s in range(3)]
                inputs.update(files)
                rs = [r for f in files for r in rows(f)]
                key = f'g{k}_{policy}_{condition}'
                v = metric(rs)
                for field in ('SR', 'SG_m', 'movement_m', 'successes', 'planned'):
                    if abs(v[field]-verdict['metrics'][key][field]) > 1e-9:
                        raise ValueError(f'Metric mismatch: {key}/{field}')
                neural[key] = dict(metrics=v, accepted_cues=acceptance(rs),
                                  by_distance={str(d): metric([r for r in rs if r['distance'] == d])
                                               for d in sorted({r['distance'] for r in rs})})
    rules = {}
    for f in sorted((OUT/'规则基线').glob('*.jsonl')):
        inputs.add(f)
        rules[f.stem] = metric(rows(f))
    probes = {}
    for k in (5, 10, 15):
        pf = OUT/f'元数据/grid{k}视觉探针.json'
        cf = OUT/f'元数据/grid{k}逐格来源.json'
        inputs.update([pf, cf])
        pairs = read(pf)
        cells = read(cf)['cells']
        mixed = np.array([cells[r['target']]['crosses_source_seam'] for r in pairs])
        for c in ('CueFull', 'CueMean'):
            inputs.update(OUT/f'特征/probe_g{k}_s{s}_{c}.npy' for s in range(3))
            full = probe(k, c, np.ones(len(pairs), bool))
            if full != verdict['probe_metrics'][f'g{k}_{c}']:
                raise ValueError('Probe metric mismatch')
            probes[f'g{k}_{c}'] = dict(total=full,
                seeds=[probe(k, c, np.ones(len(pairs), bool), s) for s in range(3)],
                target_seam={str(b): probe(k, c, mixed == b) for b in (False, True)})

    supplemental = dict(neural=neural, rules=rules, probes=probes,
        geographic_regions=1, independent_person_review=False,
        evaluation_only_truth=True, formal_pool_model_consumption=0)
    ledger = dict(date='2026-10-04', task='DATA-007', role='development_only',
        parent='DATA-005', already_consumed_support_bounds_m=[233000, 894000, 241000, 902000],
        native_feature_window_bounds_m=[233000, 894000, 237500, 898500],
        native_unique_images=225, copied_grid_images=350, geographic_regions=1,
        encoder_used=True, static_probe_grids=[5, 10, 15], navigation_grids=[5, 10],
        grid15_navigation_records=0, formal_regions_navigation_used=0, spare_regions_navigation_used=0,
        historical_ledger_overwritten=False, cannot_be_relabelled_unseen=True)
    report = [
        '# MassGIS同产品开发导航兼容报告\n',
        '2026-10-04 / DATA-007。工程兼容与10×10目标机制开发门槛均通过；只允许准备下一轮独立确认，未升级默认。',
        '本轮复用DATA-005已消费地区：原生0.5米RGB影像、600像素/格、300米/格。右上角5×5、10×10及完整15×15互相嵌套，面积分别2.25、9、20.25km²，但地理样本只有1区。只有5/10网格进行了导航；15网格只有接口及静态图对探针。正式10区和3备用区未参与模型选型、编码器提取或导航。',
        '冻结Sat2Cap、Small256 NoTarget、三组探索器/视觉头权重、拟合均值、0.50及Coverage3Radial。600→300 BOX只用于原边缘特征，原生PNG未重采样；编码器仍用原224 BICUBIC预处理。没有训练、下载、云端或外部助手调用。',
        '## 同题导航结果\n',
        'SR按计划分母，SG为全部终局曼哈顿距离×300米，成功记0；移动距离为实际有效移动，越界原地不增加米数但耗一步。以下是三权重合并均值，各权重任务数相同，不是独立地理重复。Baseline禁用目标线索接受，CueMean替换目标特征为冻结拟合均值，CueWrong为固定错误目标。',
        '| 网格/策略 | 目标条件 | 成功/计划 | SR | SG（米） | 平均有效移动（米） |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for key, value in neural.items():
        k, policy, condition = key.split('_')
        v = value['metrics']
        report.append(f"| {k[1:]}×{k[1:]}/{policy} | {condition} | {v['successes']}/{v['planned']} | {pct(v['SR'])} | {v['SG_m']:.2f} | {v['movement_m']:.2f} |")
    report += ['\n## 10×10真实目标分层\n',
        '| 策略 | 分组 | 成功/计划 | SR | SG（米） | 平均有效移动（米） |',
        '|---|---|---:|---:|---:|---:|']
    for policy in ('M0', 'Coverage3Radial'):
        v = verdict['metrics'][f'g10_{policy}_CueFull']
        groups = [(f'权重{s}', x) for s, x in enumerate(v['seeds'])]
        groups += [(f'距离层{g}', x) for g, x in v['distance_groups'].items()]
        groups += [('目标含接缝' if g == 'True' else '目标无接缝', x) for g, x in v['target_seam_groups'].items()]
        for g, x in groups:
            report.append(f"| {policy} | {g} | {x['successes']}/{x['planned']} | {pct(x['SR'])} | {x['SG_m']:.2f} | {x['movement_m']:.2f} |")
    report += ['\n## 配对恢复、损伤和目标证据\n',
        '| Coverage3Radial真实目标相对 | SR差（百分点） | 恢复 | 损伤 |',
        '|---|---:|---:|---:|']
    for key, x in verdict['comparisons'].items():
        report.append(f"| {key} | {100*x['SR_difference']:+.2f} | {x['restored']} | {x['harmed']} |")
    for control in ('Baseline', 'CueMean', 'CueWrong'):
        x = verdict['comparisons'][control]
        report.append(f"\n相对{control}的三个权重SR差为" + '、'.join(f'{100*d:+.2f}' for d in x['seed_differences']) + '个百分点；均达到≥5点、至少2/3权重正差的开发门槛。')
    report += ['\n## 静态视觉头与实际接受\n',
        'Raw探针直接使用五类top和0.50，不借合法/未访问动作门控提高精度。邻接方向类正确且达到阈值才算正确接受；不同网格的非邻接候选数不同，精度不能单独解释为方向能力下降。',
        '| 网格 | 条件 | 图对/权重 | 接受 | 正确接受 | 精度 | 邻接正确接受召回 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for key, x in probes.items():
        k, c = key.split('_'); v = x['total']
        report.append(f"| {k[1:]}×{k[1:]} | {c} | {v['pairs_each_seed']} | {v['accepted']} | {v['correct']} | {pct(v['precision'])} | {pct(v['recall'])} |")
    fp, mp = probes['g10_CueFull']['total'], probes['g10_CueMean']['total']
    report.append(f"\n10网格raw精度{pct(fp['precision'])}≥90%，召回{pct(fp['recall'])}≥50%，正确接受{fp['correct']}≥30；召回比均值目标提高{100*(fp['recall']-mp['recall']):.2f}个百分点≥10点，开发目标门槛通过。")
    report.append('15网格raw精度88.91%，低于10网格使用的90%参考线；本轮冻结门槛适用于10网格，不能将通过结论外推给15网格。应在正式协议中预先定义误触发、接缝与目标对照判定，不能看正式成绩后调阈值。静态图对数量增加会改变候选先验，尚不能据此认定闭环导航精度或SR。')
    report += ['\n| 实际导航真实目标 | 接受次数 | 命中真实目标 | 接受精度 |', '|---|---:|---:|---:|']
    for key in ('g5_M0_CueFull', 'g10_M0_CueFull', 'g10_Coverage3Radial_CueFull'):
        x = neural[key]['accepted_cues']
        report.append(f"| {key} | {x['accepted']} | {x['correct_true']} | {pct(x['precision_true'])} |")
    report += ['\n这些数字来自实际已访问状态，受到探索路径与合法/未访问门控影响，不能替代raw探针。完整距离C分层、各权重raw和接缝目标探针见[补充分层JSON](兼容补充分层结果.json)。',
        '\n## 规则基线与工程验收\n',
        '| 规则作业 | 成功/计划 | SR | SG（米） | 平均有效移动（米） | 越界次数 |',
        '|---|---:|---:|---:|---:|---:|']
    for key, v in rules.items():
        report.append(f"| {key} | {v['successes']}/{v['planned']} | {pct(v['SR'])} | {v['SG_m']:.2f} | {v['movement_m']:.2f} | {v['out_of_bounds']} |")
    report.append(f"\n34项原预检查（新16/旧18）和审计范围4项检查通过；{audit['records_replayed']}条轨迹/{audit['actions_checked']}动作、350原生PNG副本、30072头概率精确重算，包装与旧冻结实现逐动作/logits/概率/输入哈希一致。9张公开图重新编码及另实现整数BOX/状态/SG/移动/覆盖方程通过；6040历史文件SHA一致。独立实现由同一开发Agent完成，不是独立人员审查。2100神经轨迹越界为0。")
    report.append('预冻结C8最多4条不同路线，初稿测试失败保留；模型运行前将5网格配额修正为6/5/5/5/4。首次审计v1将神经零越界条件误用于旧FixedRegion，拒绝记录及冻结源码保留；审计v2仅修正规则基线检查范围，7次越界全部原地耗步、全200规则题保留，未改轨迹、权重、导航策略或验收门槛。')
    report += ['\n## 证据与适用范围\n',
        '- [冻结方案](../../../../../选题报告相关/MassGIS同产品开发导航兼容冻结方案_v1.md)、[首动作前封存](核验/首动作前封存.json)、[导航完成绑定](核验/导航完成.json)。',
        '- [判定](验收结论.json)、[独立复核v2](核验/独立导航兼容复核_v2.json)、[v1拒绝](核验/审计v1拒绝记录.json)、[v2预登记](核验/审计适配_v2预登记.json)。',
        '- [开发消费账本](元数据/开发导航消费账本.json)、[补充分层](兼容补充分层结果.json)、[下一项准备草案](下一项正式导航协议准备草案.md)。',
        '\n只证明单个已消费地区的同产品开发兼容；不输出地区置信区间。不能与旧MassachusettsRoads独立68.04%直接相减，不证明新产品独立泛化、15×15导航、多Agent、异时、任意角度、地面照片或无人机实飞。原S2/S3/S4与9km²默认保留。']
    draft = '''# EVAL-001正式导航协议准备草案

2026-10-04。DATA-007工程兼容与10×10目标机制开发门槛通过；本文件只交接协议准备，不是已冻结执行配置，不调用模型或消费正式区域。

## 固定范围

复用DATA-006事先保留的10个正式区域，3备用不参与选型。不根据模型成绩换区域；获取失败、数据变动或质量问题须保存拒绝并另登记，不能挑成功区。原生0.5米/600PNG/300米格、同模态北向影像，所有三权重、均值、0.50、M0/Coverage3Radial保持。完整数据只属于评测器；策略只收公开Observation和当前/给定目标两图特征。旧默认不自动扩到MassGIS。

在相同源产品上分别定义右上角10×10（原15格行0..9、列5..14，9km²）和完整15×15（20.25km²），统一B20。小窗口只原样裁取，不改变物理格尺度。另写正式环境/评测/审计版本：当前NativeAreaEnv只允许dev，不能把data-only清单改名就直接运行正式导航。全部权重、数据、任务、缓存和源码先绑定。

## 首动作前必须完成

1. 冻结10个区域的具体ID/边界/逐格来源及两网格关系，重验数据与历史隔离；10区域是地理样本数，双网格、三权重和控制不能算新增区域。
2. 固定任务采样种子、不同起终点、短/中/长距离及目标含接缝配额。15网格最大距离28，B20不可完成的任务必须事先决定排除还是独立压力队列；不能混入后再改分母。共同距离C4..16与更远可达目标C17..20分别报告，具体配额/队列大小在协议中事先确定。
3. 冻结同题M0/Coverage3Radial主对照，以及禁用/均值/错误目标对照、错目标距离匹配例外与无自动重试规则。正式评测消耗后不能再用于选阈值或候选。
4. 分开定义“接口正确”“新产品目标机制”“独立策略收益”“扩大面积后的表现”门槛。根据任务和行动预算预登记SR、SG米数、区域级配对置信区间、恢复/损伤及分层保护；不能把SR60%当通用通过线，不能用旧68.04%作为新产品基线。
5. 15网格开发raw精度88.91%为已知风险，10网格通过不能替代15网格机制验收。在正式首动作前确定raw误接受、实际cue正确率及接缝分层的验证/停止条件；若需诊断15网格闭环，另立已消费开发区的小试协议，不能让正式池承担调参。
6. 冻结主对照与目标对照的执行顺序、最多episode/探针/编码器数量、派生磁盘和时间上限、作业中断恢复办法及独立审计源码；完成合成接口与旧冻结回归。协议检查不需要新增训练。

## 交付与判断

按计划分母统计SR，全部终局SG米数与实际有效移动米数分别报告。区域级置信区间、全部三权重、距离与接缝分层、静态头和实际接受、恢复/损伤必须保留；单题或各权重不当独立区域。默认替换须另立版本并限已通过作用域；失败保留，不调门槛追过线。

本批未冻结正式题数、阈值门槛或请求资源，不允许直接凭这份草案启动15×15正式导航。下一Agent先读AGENTS、PROJECT_STATE、TASKS、HANDOFF，认领EVAL-001后完成上述配置。本轮不新增训练、多Agent、跨视角或实飞。
'''
    data = {'兼容补充分层结果.json': supplemental,
            '元数据/开发导航消费账本.json': ledger}
    for name, value in data.items():
        with (OUT/name).open('x', encoding='utf-8', newline='\n') as f:
            json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
    rendered = []
    for part in report:
        rendered.append(part)
        if not part.startswith('|'):
            rendered.append('')
    for name, text in [('同产品开发导航兼容报告.md', '\n'.join(rendered)+'\n'),
                       ('下一项正式导航协议准备草案.md', draft)]:
        with (OUT/name).open('x', encoding='utf-8', newline='\n') as f:
            f.write(text)
    binding = dict(generated_from_saved_results_only=True, model_forwards=0,
        elapsed_seconds_from_registration=elapsed,
        source_sha256={str(Path(__file__).relative_to(ROOT)).replace('\\', '/'): sha(Path(__file__))},
        input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(inputs)},
        output_sha256={n: sha(OUT/n) for n in names[:-1]})
    with (OUT/names[-1]).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(binding, f, ensure_ascii=False, indent=2); f.write('\n')
    print('Saved report, stratified results, usage ledger, and preparation draft; no model calls.')


if __name__ == '__main__':
    main()
