"""Preserve initial exports; resolve marker/mean-label spacing in figure41."""
from pathlib import Path
import json
import hashlib
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'DATA/processed_data/Masa/评测结果/实际区域M0兼容试跑_v1'
FIG = ROOT / '绘图'
NAME = '41_实际区域M0成功率与米制距离'


def read(p): return json.loads(p.read_text('utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()
def write(p, value):
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    assert not any((FIG / '数据结果图' / (NAME+'_v2.'+fmt)).exists() for fmt in ('png', 'pdf', 'svg'))
    original = read(FIG / '绘图数据/实际区域M0兼容试跑_v1.json')
    s = original['summary']
    assert digest(RUN / '对照汇总.json') == original['source_sha256'][rel(RUN / '对照汇总.json')]
    names = ('CueFull', 'Baseline', 'CueMean', 'CueWrong', 'Frontier', 'FixedRegion')
    labels = ('M0真实目标', '禁用目标线索', '目标均值', '错误目标图', 'Frontier', 'FixedRegion')
    colors = [graphics.BLUE, graphics.GRAY, graphics.ORANGE, graphics.TEAL, '#8A75A7', '#998363']
    graphics.style()
    graphics.ITEMS.clear()
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 4.85), sharey=True)
    y = np.arange(6)
    for ax, field, scale, label, precision in ((axes[0], 'sr', 100, '(a) SR：成功率（%）', 2),
                                             (axes[1], 'mean_sg_m', 1, '(b) SG：全体终点距离（米）', 1),
                                             (axes[2], 'mean_valid_travel_m', .001, '(c) 平均有效移动（公里）', 2)):
        values = [s['arms'][n]['metrics'][field]*scale for n in names]
        ax.barh(y, values, color=colors, height=.67, alpha=.95)
        label_positions = list(values)
        for i, name in enumerate(names[:4]):
            seed_values = []
            for seed, marker in enumerate(('o', '^', 's')):
                r = read(RUN / '神经对照' / f'M0_s{seed}_{name}_结果.json')
                value = r['metrics'][field]*scale
                seed_values.append(value)
                ax.scatter(value, i+.13*(seed-1), marker=marker, facecolor='white', edgecolor='#26313B', s=19,
                           linewidth=.7, zorder=4)
            label_positions[i] = max([values[i]]+seed_values)
        pad = max(values)*.035
        for i, (value, x) in enumerate(zip(values, label_positions)):
            ax.text(x+pad, i, f'{value:.{precision}f}', va='center', fontsize=9)
        ax.set_xlim(0, 104 if field == 'sr' else max(label_positions)*1.28)
        ax.set_title(label, loc='left', fontsize=10)
        ax.set_yticks(y, labels)
        ax.grid(axis='x', alpha=.7)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], linestyle='', marker=marker, markerfacecolor='white', markeredgecolor='#26313B',
                    markersize=4.5, label=f'权重{seed}') for seed, marker in enumerate(('o', '^', 's'))],
               loc='center', bbox_to_anchor=(.54, .20), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.155, right=.975, top=.87, bottom=.31, wspace=.29)
    fig.text(.54, .09, '三张已知训练地图，合计75题；神经各三权重，规则各一次。标记为权重，数字为均值。', ha='center', fontsize=9)
    fig.text(.54, .035, '每图9 km²；300米/格、10×10/B20。SG含成功的0；有效移动计合法动作，越界0米。', ha='center', fontsize=9)
    item = original['figures'][0]
    graphics.save(fig, FIG / '数据结果图', NAME+'_v2', item['caption'], item['sources'], item['protocol'])
    final_item = dict(graphics.ITEMS[0], id=NAME, reproduction_script=rel(Path(__file__)),
                      revision='v2: mean labels beyond weight markers; explicitly75total tasks; initial exports retained')
    final_figures = [final_item, original['figures'][1]]
    write(FIG / '绘图数据/实际区域M0兼容试跑_v2.json', dict(date='2026-10-02', figures=final_figures, summary=s,
          script_sha256=digest(Path(__file__)), source_sha256=original['source_sha256'],
          initial_manifest_sha256=digest(FIG / '绘图数据/实际区域M0兼容试跑_v1.json')))
    with (FIG / '实际区域M0兼容试跑图注_v2.md').open('x', encoding='utf-8') as f:
        f.write('\n\n'.join('# '+item['id']+'\n\n'+item['caption'] for item in final_figures)+'\n')
    catalog = read(FIG / '图表来源清单.json')
    assert sum(i['id'] == NAME for i in catalog['figures']) == 1
    catalog['figures'] = [final_item if i['id'] == NAME else i for i in catalog['figures']]
    (FIG / '图表来源清单.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for p in (ROOT / 'README.md', FIG / 'README.md', ROOT / '项目导航/当前进度.md'):
        v = p.read_text('utf-8').replace('实际区域M0兼容试跑图注.md', '实际区域M0兼容试跑图注_v2.md')
        v = v.replace('41_实际区域M0成功率与米制距离.pdf', '41_实际区域M0成功率与米制距离_v2.pdf')
        v = v.replace('绘图数据/实际区域M0兼容试跑_v1.json', '绘图数据/实际区域M0兼容试跑_v2.json')
        p.write_text(v, encoding='utf-8')
    with (FIG / '审阅/实际区域M0兼容试跑修订_v2.md').open('x', encoding='utf-8') as f:
        f.write('图41初稿及v1清单全部保留。v2将均值数字放在该行最右权重标记之后，消除圆/方/三角与均值的相叠；图下注明三地图合计75题，避免误读为每图75题。没有改变数据、轴起点、权重标记或图42。最终入口与图表来源清单指向v2；需继续做PNG/PDF目视核验。\n')
    (FIG / '审阅/实际区域M0兼容试跑_图41修订源码_v2.py').write_bytes(Path(__file__).read_bytes())
    print(dict(revision='v2', initial_exports_retained=True, data_changed=False), flush=True)


if __name__ == '__main__':
    main()
