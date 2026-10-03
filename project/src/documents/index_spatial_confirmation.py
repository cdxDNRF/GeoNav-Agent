"""Update mutable project navigation after an audited experiment is delivered."""
from pathlib import Path
from datetime import date
import json

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
FIG = ROOT / '绘图'


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    verdict = read(RUN / '验收结论.json')
    summary = read(RUN / '独立确认/对照汇总.json')
    probes = read(RUN / '独立确认/探针对照/探针汇总.json')
    audits = [read(RUN / folder / '独立复核.json') for folder in ('工程接口', '独立确认')]
    assert verdict['completed'] and all(a['passed'] and a['audit_revision'] == 3 for a in audits)
    manifest_path = FIG / '绘图数据/冻结M0空间隔离导航确认_v2.json'
    manifest = read(manifest_path)
    report = RUN / '新空间隔离区域M0导航确认报告_v2.md'
    assert report.is_file()
    nav = '通过' if verdict['spatial_unknown_area_navigation_passed'] else '未通过'
    seam = '通过' if verdict['seam_head_reliability_passed'] else '未通过'
    combined = '通过' if verdict['spatial_area_combined_passed'] else '未通过'
    full = summary['arms']['CueFull']
    table = '\n'.join(f"| {name} | {100*full['by_stratum'][key]['sr']:.2f}% | {full['by_stratum'][key]['mean_sg_m']:.2f} |"
                       for name, key in [('长距离C12—16（主队列）', 'long_distance'),
                                         ('接缝目标C8—12', 'seam_target'), ('内部目标C8—12', 'interior_target')])
    cross = probes['arms']['Full']['by_kind']['cross_source_adjacent']
    negative = probes['arms']['Full']['by_kind']['nonadjacent_matched_target']
    status = f'导航{nav}；接缝头部{seam}；组合{combined}'
    results = (f"10个空间隔离确认区域（每区9km²、300米/格、10×10/B20），750不同任务×3冻结权重。"
               f"长距离主队列SR{100*summary['primary_SR']:.2f}%，混合队列SR{100*full['metrics']['sr']:.2f}%/"
               f"SG{full['metrics']['mean_sg_m']:.2f}米。跨源正确接受{100*cross['raw_correct_acceptance']:.2f}%，"
               f"非邻接误接受{100*negative['raw_nonadjacent_false_acceptance']:.2f}%；{status}。"
               f"两阶段14700导航记录/{sum(a['actions'] for a in audits)}动作及15120探针判断复核通过；"
               "40冻结前测试及6审计修订回归测试通过。无训练、无云端、默认M0保持。")
    links = (f"[报告]({rel(report)}) / [判定]({rel(RUN / '验收结论.json')}) / "
             f"[确认复核]({rel(RUN / '独立确认/独立复核.json')}) / [图45—46]({rel(FIG / '冻结M0空间隔离导航确认排版修订_v2.md')})")
    index_path = ROOT / '项目导航/实验索引.json'
    index = read(index_path)
    assert len(index['batches']) == 65 and not any(b['path'] == rel(RUN) for b in index['batches'])
    index['batches'].append(dict(dataset='MasaRoads', kind='评测', name=RUN.name,
        category='实际区域独立确认', status=status, scope='4工程/10确认，9km²每区；14700导航/15120探针；本地M0足迹空间隔离，Sat2Cap预训练地理覆盖未知；原模型与默认保持',
        path=rel(RUN), reports=[rel(report)], evidence=[rel(RUN / '验收结论.json'),
        rel(RUN / '工程接口/独立复核.json'), rel(RUN / '独立确认/独立复核.json'), rel(RUN / '审计修订登记_v3.json')]))
    index['date'] = date.today().isoformat()
    write(index_path, index)
    md = ROOT / '项目导航/实验索引.md'
    text = md.read_text('utf-8')
    text = text.replace('65批', '66批').replace('65个', '66个')
    text += (f'\n\n## 冻结M0空间隔离导航确认_v1\n\n{results}\n\n'
             f"[报告](../{rel(report)}) / [判定](../{rel(RUN / '验收结论.json')}) / "
             f"[独立复核](../{rel(RUN / '独立确认/独立复核.json')})。\n")
    md.write_text(text, 'utf-8')
    rootmd = ROOT / 'README.md'
    text = rootmd.read_text('utf-8')
    start = text.index('更新至')
    end = text.index('## 最新阶段')
    text = text[:start] + (f'更新至{date.today().strftime("%Y年%m月%d日")}。接手项目先读本文件，再看对应批次协议。'
            '原S2/S3/S4及SwissView同模态迁移验收保持有效；实际搜索足迹的兼容试跑、'
            f'空间隔离数据准备及冻结M0独立确认均已完成，最新判定为{status}。\n\n') + text[end:]
    start = text.index('## 最新阶段')
    end = text.index('## 上一阶段', start)
    old_latest = text[start:end].replace('## 最新阶段：', '## 前置数据：')
    old_latest = old_latest.replace('\n\n', '\n\n以下为上一轮数据准备阶段的历史状态；导航确认已在本轮完成。\n\n', 1)
    latest = f'## 最新阶段：冻结M0空间隔离导航确认完成\n\n{results}\n\n{links}。\n\n'
    text = text[:start] + latest + old_latest + text[end:]
    text = text.replace('全部65批', '全部66批').replace('65批实验', '66批实验').replace('44张实验与数据协议图', '46张实验与数据协议图')
    text = text.replace('后续先冻结M0工程接口和10区确认。', '冻结M0工程接口与10区确认已完成，下一因素需另行预登记。')
    rootmd.write_text(text, 'utf-8')
    progress = ROOT / '项目导航/当前进度.md'
    text = progress.read_text('utf-8')
    first = text.index('截至')
    end = text.index('## 默认与验收')
    intro = (f'截至{date.today().isoformat()}，环境、统一评测、最小VLM及训练过的单Agent主线已完成；'
             f'原S2/S3/S4及SwissView验收继续有效。本轮实际面积与空间隔离确认已完成：{status}。'
             '默认M0保持，所有14新区域已用于本批导航，不能再次声明未消费。\n\n')
    local_links = links.replace('](', '](../')
    section = (f'## 最新：实际搜索面积的空间隔离确认\n\n{results}\n\n'
               f'| 确认队列（每权重250题） | SR | SG米 |\n|---|---:|---:|\n{table}\n\n'
               f'{local_links}。\n\n空间隔离仅针对本地M0训练/任务源；Sat2Cap预训练地理覆盖未知，'
               '本轮未验证任意角度、真实异时或实飞。新区域使用状态保存在本批，旧数据准备账本保持历史状态。\n\n')
    progress.write_text(text[:first] + intro + section + text[end:], 'utf-8')
    for path in (ROOT / '项目导航/README.md', ROOT / '项目导航/目录结构与维护.md'):
        path.write_text(path.read_text('utf-8').replace('65批', '66批').replace('65个', '66个'), 'utf-8')
    catalog_path = FIG / '图表来源清单.json'
    catalog = read(catalog_path)
    assert not any(f['id'].startswith(('45_', '46_')) for f in catalog['figures'])
    catalog['figures'].extend(manifest['figures'])
    write(catalog_path, catalog)
    figuresmd = FIG / 'README.md'
    text = figuresmd.read_text('utf-8')
    text += (f'\n\n## 最新：图45—46，冻结M0空间隔离导航确认\n\n{results}\n\n'
             '[图45 PDF](数据结果图/45_空间隔离区域三分层M0导航结果_v2.pdf) / '
             '[图46 PDF](数据结果图/46_跨源接缝原始头部接受率_v2.pdf) / '
             '[图注](冻结M0空间隔离导航确认图注.md) / '
             '[来源与数据](绘图数据/冻结M0空间隔离导航确认_v2.json) / '
             '[排版修订说明](冻结M0空间隔离导航确认排版修订_v2.md)。\n\n'
             'PNG300dpi，PDF嵌入字体、SVG可编辑文本；仅展示该批源与协议，不与旧异队列连接为提升曲线。\n')
    figuresmd.write_text(text, 'utf-8')
    print(dict(index_entries=66, result_figures=46, latest=status), flush=True)


if __name__ == '__main__':
    main()
