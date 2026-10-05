"""Build an immutable formal report from audited saved results only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval import massgis_confirmation_run_v1 as r


def pct(x):return '未验收' if x is None else f'{100*x:.2f}%'


def main():
    path=r.OUT/'正式双网格确认报告.md'
    if path.exists():raise ValueError('Report exists; no overwrite')
    verdict=r.read(r.OUT/'验收结论.json');summary=r.read(r.OUT/'主对照/汇总.json');final=verdict['per_grid']
    lines=['# MassGIS正式10×10/15×15独立确认报告','',
        '冻结原10个空间隔离地区、三权重及0.50，300米/格、B20，同模态北向原生航拍切图。10/15网格分别为9/20.25km²；没有新增训练、网络、云端或默认升级。',
        '', '地理样本只有10区；双网格、三权重、控制条件不增加地区数。10格750题，15格1000题，其中750题与10格物理端点对应，其余250题更远。不同混合队列SR不能直接作扩区差，配对结果单列。',
        '', '## 主结果与原门槛','',
        '| 网格/队列 | 策略 | 成功/计划 | SR | SG（米） | 平均有效移动（米） |',
        '|---|---|---:|---:|---:|---:|']
    for k in (10,15):
        for policy,name in [('M0','baseline'),('Coverage3Radial','candidate')]:
            m=summary[str(k)]['comparison'][name]
            lines.append(f"| {k}×{k}/完整混合 | {policy} | {m['successes']}/{m['planned']} | {pct(m['SR'])} | {m['SG_m']:.2f} | {m['movement_m']:.2f} |")
        if k==15:
            for policy,name in [('M0','baseline'),('Coverage3Radial','candidate')]:
                m=summary['15']['common_comparison'][name]
                lines.append(f"| 15×15/共同子队列 | {policy} | {m['successes']}/{m['planned']} | {pct(m['SR'])} | {m['SG_m']:.2f} | {m['movement_m']:.2f} |")
    lines+=['','SR按计划分母；SG是全部终局曼哈顿距离×300米，成功记0；有效移动另报。三权重平均不是默认seed0单次成绩。','']
    for k in (10,15):
        item=summary[str(k)];c=item['comparison'];ci=c['region95_SR_difference'];f=final[str(k)]
        lines += [f'### {k}×{k}判定','',
            f"Coverage相对M0 SR差{100*c['SR_difference']:+.2f}点，完整区域95%差[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]点；恢复{c['restored']}、损伤{c['harmed']}。主门槛{'通过' if item['main_gate_passed'] else '未通过'}。",
            '', '主门槛逐项：'+ '；'.join(key+('通过' if value else '未过') for key,value in item['gates'].items())+'。',
            '', '| 距离层 | M0 SR | Coverage SR | M0 SG（米） | Coverage SG（米） |', '|---|---:|---:|---:|---:|']
        for group,v in item['strata'].items():
            lines.append(f"| {group} | {pct(v['baseline']['SR'])} | {pct(v['candidate']['SR'])} | {v['baseline']['SG_m']:.2f} | {v['candidate']['SG_m']:.2f} |")
        lines+=['','各权重、地区及接缝对照见[主汇总](主对照/汇总.json)，全部距离C分层及规则详表见[补充分层](补充分层结果.json)。','',
            f"正式目标对照{'已按条件执行' if f['controls_run'] else '因主门槛未过而未启动'}；默认候选资格{'通过全部必要证据' if f['eligible_for_default_upgrade'] else '未通过全部必要证据'}。",
            '', '| 目标对照 | SR差（百分点） | 区域95%差（百分点） | 恢复 | 损伤 |', '|---|---:|---|---:|---:|']
        for control,v in f['targets'].items():
            cc=v['region95_SR_difference']
            lines.append(f"| {control} | {100*v['SR_difference']:+.2f} | [{100*cc[0]:+.2f},{100*cc[1]:+.2f}] | {v['restored']} | {v['harmed']} |")
        if not f['targets']:lines.append('| 未启动 | — | — | — | — |')
        raw=f['raw'];actual=f['actual_full']
        lines+=['',f"Raw五类top/0.50正确接受精度{pct(raw['precision'])}、邻接正确接受召回{pct(raw['recall'])}。实际导航接受精度{pct(actual['accepted_precision'])}；错误cue/全部执行动作{pct(actual['false_cue_action_rate'])}，接缝目标队列{pct(f['actual_seam']['false_cue_action_rate'])}。这些分母不同，不能把预算误触发占比解释为接受精度。",
            '', '可靠性逐项：'+'；'.join(key+('未执行对应对照' if value is None else '通过' if value else '未过') for key,value in f['reliability_gates'].items())+'。','']
    lines+=['## 同物理任务的面积扩展','', '| 策略 | 15格相对10格SR差（点） | 区域95%差（点） | SG差（米） |', '|---|---:|---|---:|']
    for policy,v in final['common_area_expansion'].items():
        ci=v['region95_SR_difference'];lines.append(f"| {policy} | {100*v['SR_difference']:+.2f} | [{100*ci[0]:+.2f},{100*ci[1]:+.2f}] | {v['SG_m_difference']:+.2f} |")
    main_a=r.read(r.QA/'主对照独立复核.json');end_a=r.read(r.QA/'最终独立复核.json');ledger=r.read(r.OUT/'元数据/正式池导航完成账本.json')
    lines+=['','## 验证、消费与适用边界','',
        f"2250不同原图特征、282480静态头预测已消费并核验；所有原生BOX profile独立重算，90图重新编码，全部概率精确一致。主10500轨迹/{main_a['actions']}动作独立重放；目标{ledger['control_records']}及规则3500轨迹/{end_a['target_and_rule_actions']}动作另作完整重放。",
        '', f"{end_a['protected_unchanged']}历史文件SHA一致；同一开发Agent另实现状态、指标、覆盖及区域bootstrap，不称独立人员审查。正式10区现已消费，不能再改名当未见确认；3备用未参与本批。所有原件保留，失败门槛不回写或放松。",
        '', '没有默认自动升级；原S2/S3/S4及旧MassachusettsRoads 9km²默认仍有效。本次新产品/队列不与旧68.04%直接作差。结论只限同模态航拍切图仿真，不能外推地面目标、异时、任意角度、实飞或算法多Agent。',
        '', '## 原件入口','',
        '- [原冻结协议](../../工程准备/正式导航协议冻结_v1/冻结协议.json) / [执行预登记](核验/执行预登记.json) / [首动作前封存](核验/首动作前封存.json)。',
        '- [判定](验收结论.json) / [特征复核](核验/特征与探针独立复核.json) / [主重放](核验/主对照独立复核.json) / [最终重放](核验/最终独立复核.json)。',
        '- [完成消费账本](元数据/正式池导航完成账本.json) / [主对照](主对照/) / [目标对照](目标对照/) / [规则基线](规则基线/)。','']
    supplemental={}
    for k in (10,15):
        rows=r.all_rows(k,'Coverage3Radial','CueFull')
        supplemental[str(k)]=dict(by_distance={str(d):r.metrics([x for x in rows if x['distance']==d]) for d in sorted({x['distance'] for x in rows})},
            rules={policy:r.metrics([row for rec in r.manifest(k)['regions'] for row in r.records(r.job_path(k,policy,0,'CueFull',rec['area']))]) for policy in ('Frontier','FixedRegion')})
    r.write(r.OUT/'补充分层结果.json',supplemental)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
    r.write(r.QA/'报告生成绑定.json',dict(source_sha256={r.rel(Path(__file__)):r.sha(Path(__file__))},
        verdict_sha256=r.sha(r.OUT/'验收结论.json'),report_sha256=r.sha(path),supplemental_sha256=r.sha(r.OUT/'补充分层结果.json'),new_model_calls=0))
    print('Formal report published from audited saved evidence.',flush=True)


if __name__=='__main__':main()
