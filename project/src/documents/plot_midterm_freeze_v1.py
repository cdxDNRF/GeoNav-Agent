"""Generate protocol-separated midterm research figures from sealed summaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '中期报告相关/中期成果冻结版_v1'
FIG = OUT / '图表'
INPUTS = {
    's2': 'DATA/processed_data/Masa/评测结果/边缘线索S2正式复验_v1/对照汇总.json',
    's3': 'DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1/对照汇总.json',
    's4': 'DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1/对照汇总.json',
    'swiss': 'DATA/processed_data/SwissView/评测结果/五乘五正式迁移_v1/对照汇总.json',
    'roads': 'DATA/processed_data/MasaRoads/评测结果/径向保护独立新区域导航确认_v1/主对照/对照汇总.json',
    'massgis': 'DATA/processed_data/MassGIS/评测结果/正式十乘十十五乘十五确认_v1/验收结论.json',
    'geo': 'DATA/processed_data/Masa/训练结果/GeoExplorer分布一致性成对开发_v1/开发评测/开发汇总.json',
}


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_values():
    data = {key: read(value) for key, value in INPUTS.items()}
    main = [
        dict(name='S2稳定', sr=data['s2']['averages']['Edge']['CueFull']['sr_mean'] * 100,
             sg=data['s2']['averages']['Edge']['CueFull']['sg_mean'], protocol='5×5 / B10', status='通过'),
        dict(name='S3独立源图', sr=data['s3']['averages']['Edge']['CueFull']['sr_mean'] * 100,
             sg=data['s3']['averages']['Edge']['CueFull']['sg_mean'], protocol='5×5 / B10', status='通过'),
        dict(name='S4密度扩展', sr=data['s4']['averages']['Edge']['CueFull']['sr_mean'] * 100,
             sg=data['s4']['averages']['Edge']['CueFull']['sg_mean'], protocol='10×10 / B20', status='通过'),
        dict(name='SwissView同模态', sr=data['swiss']['averages']['Edge']['CueFull']['sr_mean'] * 100,
             sg=data['swiss']['averages']['Edge']['CueFull']['sg_mean'], protocol='5×5 / B10', status='通过'),
        dict(name='9 km²连续区域', sr=data['roads']['candidate']['metrics']['sr'] * 100,
             sg=data['roads']['candidate']['metrics']['mean_sg_m'], protocol='10×10 / B20', status='通过', unit='米'),
    ]
    massgis = [
        dict(name='MassGIS 10×10', sr=data['massgis']['per_grid']['10']['actual_full']['SR'] * 100,
             sg=data['massgis']['per_grid']['10']['actual_full']['SG_m'], status='未过'),
        dict(name='MassGIS 15×15', sr=data['massgis']['per_grid']['15']['actual_full']['SR'] * 100,
             sg=data['massgis']['per_grid']['15']['actual_full']['SG_m'], status='未过'),
    ]
    geo = data['geo']
    baseline = [
        dict(name='原PPO更新', sr=geo['full']['LegacyUnmaskedEvaluate']['sr'] * 100,
             sg=geo['full']['LegacyUnmaskedEvaluate']['sg_cells_all_planned'], status='对照'),
        dict(name='分布一致性修正', sr=geo['full']['ConsistentMaskedEvaluate']['sr'] * 100,
             sg=geo['full']['ConsistentMaskedEvaluate']['sg_cells_all_planned'], status='未过'),
        dict(name='Frontier', sr=geo['frontier']['sr'] * 100,
             sg=geo['frontier']['sg_cells_all_planned'], status='规则基线'),
    ]
    return dict(main=main, massgis=massgis, baseline=baseline,
        target_contrasts=geo['target_contrasts'], gates=geo['gates'],
        sources={name: sha(ROOT / path) for name, path in INPUTS.items()})


def label_bars(ax, bars):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')


def plot(values):
    font = 'Microsoft YaHei'
    if not any('msyh' in path.lower() for path in font_manager.findSystemFonts()):
        font = 'Noto Sans CJK SC'
    plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'font.size': 8.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(11.6, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.04, .96])
    teal, orange, red, grey, blue = '#137C78', '#E69F00', '#C94C4C', '#8A98A8', '#386CB0'

    ax = fig.add_subplot(grid[0, :])
    names = [x['name'] for x in values['main']]
    sr = [x['sr'] for x in values['main']]
    bars = ax.bar(np.arange(len(names)), sr, color=teal, edgecolor='#075C58', linewidth=.9)
    label_bars(ax, bars)
    ax.set_ylim(0, 100);ax.set_ylabel('成功率 SR（%）');ax.set_xticks(np.arange(len(names)), names)
    ax.set_title('(a) 已通过协议：各协议独立验收，柱高不可解释为连续提升', loc='left', fontsize=10, fontweight='bold')
    ax.grid(axis='y', color='#D7DEE5', linewidth=.6);ax.set_axisbelow(True)
    for i, record in enumerate(values['main']):
        unit = record.get('unit', '格')
        ax.text(i, 5, f"{record['protocol']}\nSG {record['sg']:.3f}{unit}", ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')

    ax = fig.add_subplot(grid[1, 0])
    names = [x['name'] for x in values['massgis']];sr = [x['sr'] for x in values['massgis']]
    bars = ax.bar(np.arange(2), sr, color=[orange, red], edgecolor='#7A4D00', hatch=['//', 'xx'], linewidth=.9)
    label_bars(ax, bars);ax.set_ylim(0, 100);ax.set_ylabel('成功率 SR（%）');ax.set_xticks(np.arange(2), names)
    ax.set_title('(b) 新影像产品扩展：正式门槛未过', loc='left', fontsize=10, fontweight='bold')
    ax.grid(axis='y', color='#D7DEE5', linewidth=.6);ax.set_axisbelow(True)
    for i, record in enumerate(values['massgis']):
        ax.text(i, 3, f"SG {record['sg']:.0f} m\n未升级默认", ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')

    ax = fig.add_subplot(grid[1, 1])
    names = [x['name'] for x in values['baseline']];sr = [x['sr'] for x in values['baseline']]
    bars = ax.bar(np.arange(3), sr, color=[grey, orange, blue], edgecolor=['#607080', '#875D00', '#234E80'], hatch=['', '//', '..'], linewidth=.9)
    label_bars(ax, bars);ax.set_ylim(0, 100);ax.set_ylabel('成功率 SR（%）');ax.set_xticks(np.arange(3), names)
    ax.set_title('(c) 同题学习基线：修正有效，目标资格仍未过', loc='left', fontsize=10, fontweight='bold')
    ax.grid(axis='y', color='#D7DEE5', linewidth=.6);ax.set_axisbelow(True)
    ax.text(1, 4, 'Full−Mean +0.95点\nFull−Wrong +1.19点', ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')
    for spine in fig.axes:
        spine.spines['top'].set_visible(False);spine.spines['right'].set_visible(False)
    fig.suptitle('中期结果分账：正式通过、扩展失败与学习基线分别报告', fontsize=13, fontweight='bold')
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(FIG / f'02_协议分账与学习基线.{suffix}', dpi=320 if suffix == 'png' else None,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    output = FIG / '图表数据_v1.json'
    if output.exists():raise FileExistsError('figure data already exists')
    values = source_values()
    output.write_text(json.dumps(values, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plot(values)
    print(json.dumps(dict(passed=True, inputs=len(INPUTS), outputs=4), ensure_ascii=False))


if __name__ == '__main__':main()
