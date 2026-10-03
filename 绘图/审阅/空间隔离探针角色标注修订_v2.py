"""Keep initial44; make engineering/confirmation roles explicit within bars."""
from pathlib import Path
import sys
import json
import hashlib
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / '绘图'
RUN = ROOT / 'DATA/processed_data/MasaRoads/空间隔离连续区域_v2'
NAME = '44_接缝目标分层与诊断探针覆盖'
def read(p): return json.loads(p.read_text('utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()


def main():
    final_path = FIG / '绘图数据/空间隔离连续区域准备_最终_v2.json'
    assert not final_path.exists()
    initial = read(FIG / '绘图数据/空间隔离连续区域准备_v2.json')
    graphics.style(); graphics.ITEMS.clear()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8))
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), 3, 3, facecolor='#F1F4F6', edgecolor='none'))
    for j in range(11):
        ax.axvline(j*.3, color='#C6CFD7', linewidth=.45); ax.axhline(j*.3, color='#C6CFD7', linewidth=.45)
    ax.axvline(1.5, color='#35424F', linestyle='--', linewidth=1.25); ax.axhline(1.5, color='#35424F', linestyle='--', linewidth=1.25)
    bank = read(RUN / '工程数据/导航任务.json'); meta = read(RUN / '工程数据/任务分层.json')
    es = [e for e in bank if e['area'] == 'img_2000']
    for layer, color, marker, fill, size in (('interior_target', graphics.BLUE, 'o', False, 67),
                                          ('long_distance', graphics.GRAY, '^', True, 30), ('seam_target', graphics.ORANGE, 's', True, 35)):
        cells = sorted({e['goal'] for e in es if meta[e['episode_id']]['stratum'] == layer})
        ax.scatter([.3*(c%10)+.15 for c in cells], [.3*(c//10)+.15 for c in cells], marker=marker, s=size,
                   facecolors=color if fill else 'none', edgecolors=color, linewidths=.85, zorder=5)
    ax.set_xlim(0, 3); ax.set_ylim(3, 0); ax.set_aspect('equal')
    ax.set_xticks([0, 1.5, 3]); ax.set_yticks([0, 1.5, 3]); ax.set_xlabel('相对东向距离（km）', fontsize=9.5); ax.set_ylabel('相对南向距离（km）', fontsize=9.5)
    ax.set_title('(a) img_2000目标覆盖（几何示意）', loc='left', fontsize=10)
    axes[1].barh([0, 1, 2], [160]*3, color=graphics.BLUE, height=.60, edgecolor='white')
    axes[1].barh([0, 1, 2], [400]*3, left=[160]*3, color=graphics.ORANGE, height=.60, edgecolor='white')
    for i in range(3):
        axes[1].text(80, i, '工程\n160', ha='center', va='center', color='white', fontsize=9.5)
        axes[1].text(360, i, '确认\n400', ha='center', va='center', color='white', fontsize=9.5)
    axes[1].set_yticks([0, 1, 2], ['跨源真邻接', '同源真邻接', '两格非邻接']); axes[1].invert_yaxis()
    axes[1].set_xlim(0, 600); axes[1].set_xticks([0, 200, 400, 600]); axes[1].set_xlabel('冻结探针对数（非导航轨迹）', fontsize=9.5)
    axes[1].grid(axis='x', alpha=.5); axes[1].set_axisbelow(True); axes[1].set_title('(b) 1680探针的角色分配', loc='left', fontsize=10)
    fig.subplots_adjust(left=.07, right=.975, top=.87, bottom=.31, wspace=.82)
    fig.legend(handles=[Line2D([], [], marker='s', linestyle='', color=graphics.ORANGE, label='边界目标25题/区'),
                         Line2D([], [], marker='o', markerfacecolor='none', linestyle='', color=graphics.BLUE, label='非边界25题/区'),
                         Line2D([], [], marker='^', linestyle='', color=graphics.GRAY, label='长距离25题/区')],
               loc='center', bbox_to_anchor=(.48, .20), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .105, '每区75条不同路线；相同目标位置可重复，左图按位置去重绘制。虚线为1500米源边界。', ha='center', fontsize=9)
    fig.text(.5, .035, '每区40跨源 + 40同源 + 40非邻接探针；同一目标配对。只准备覆盖，没有模型判断或新SR。', ha='center', fontsize=9)
    old = initial['figures'][1]
    graphics.save(fig, FIG / '数据结果图', NAME+'_标注修订', old['caption'], old['sources'], old['protocol'])
    item = dict(graphics.ITEMS[0], id=NAME, reproduction_script=rel(Path(__file__)),
                revision='bar roles written explicitly; initial exports retained; same data and coordinates')
    final = dict(initial, figures=[initial['figures'][0], item], initial_manifest_sha256=digest(FIG / '绘图数据/空间隔离连续区域准备_v2.json'),
                 refinement_script_sha256=digest(Path(__file__)))
    with final_path.open('x', encoding='utf-8') as f: json.dump(final, f, ensure_ascii=False, indent=2)
    catalog = read(FIG / '图表来源清单.json')
    catalog['figures'] = [item if i['id'] == NAME else i for i in catalog['figures']]
    (FIG / '图表来源清单.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    p = FIG / 'README.md'; text = p.read_text('utf-8').replace(NAME+'.pdf', NAME+'_标注修订.pdf')
    text = text.replace('绘图数据/空间隔离连续区域准备_v2.json', '绘图数据/空间隔离连续区域准备_最终_v2.json'); p.write_text(text, encoding='utf-8')
    (FIG / '审阅/空间隔离探针角色标注修订_v2.py').write_bytes(Path(__file__).read_bytes())
    print(dict(role_labels_explicit=True, initial_exports_retained=True, data_unchanged=True), flush=True)


if __name__ == '__main__': main()
