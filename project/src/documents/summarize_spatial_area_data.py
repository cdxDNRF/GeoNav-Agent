"""Evidence-bound spatial-data closeout, source geometry figures and navigation."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'DATA/processed_data/MasaRoads/空间隔离连续区域_v2'
STOP = ROOT / 'DATA/processed_data/MasaRoads/空间隔离连续区域_v1'
QA, META, DATA = RUN / '核验', RUN / '元数据', RUN / '工程数据'
FIG, NAV = ROOT / '绘图', ROOT / '项目导航'


def read(p): return json.loads(p.read_text('utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()
def write(p, value):
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2)
def text(p, value):
    with p.open('x', encoding='utf-8') as f: f.write(value)


def required():
    v = read(QA / '验收结论.json')
    a = read(QA / '独立复核.json')
    m = read(DATA / '数据清单.json')
    s = read(META / '连续区域选择.json')
    assert v['data_engineering_passed'] and a['passed'] and not v['new_SR_produced']
    assert digest(QA / '独立复核.json') == v['audit_sha256']
    assert digest(QA / '输入冻结结束.json') == v['raw_and_derived_input_freeze_sha256']
    assert a['isolated_regions'] == 14 and a['exact_patches'] == 1400
    reg = read(QA / '预登记.json')
    assert digest(ROOT / 'project/local_policy_default.json') == reg['default_sha256']
    return v, a, m, s


def report(v, a, m, selection):
    table = '\n'.join(f"| {r['area']} | {'工程dev' if r['split']=='dev' else '冻结确认test'} | {'、'.join(s['id'] for s in r['sources'])} | {r['min_old_gap_m']/1000:.3f} |" for r in m['regions'])
    received = selection['received_bytes']/2**20
    size = sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())/2**20
    bank = read(DATA / '导航任务.json')
    strata = read(DATA / '任务分层.json')
    geometry = {split: {layer: sum(e['split'] == split and strata[e['episode_id']]['stratum'] == layer for e in bank)
                       for layer in ('long_distance', 'seam_target', 'interior_target')} for split in ('dev', 'test')}
    text(RUN / '空间隔离连续区域数据准备报告.md', f'''# 空间隔离连续区域：数据与协议准备已通过

**已构建14个单区9km²的连续区域：4工程区、10冻结确认区，1400图块；与旧151个Masa足迹最短间隔{a['minimum_old_region_gap_m']/1000:.3f}公里。** 新区域两两间隔至少3公里（允许1毫米标签舍入）。这里只验收数据/协议与接缝覆盖，没有新导航SR，原M0和S2/S3/S4判定保持。

## 来源、发布尺度与空间排除

影像来自[Mnih官方Massachusetts Roads入口](https://www.cs.toronto.edu/~vmnih/data/)的[train影像目录](https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html)，本地保存1108链接和源页面SHA。只用RGB影像，不读取道路分割标签，也不把作者的分割成绩当导航SR。原项目151张Masa对应Massachusetts Buildings这一较小发布集合；本批来自覆盖更广的Roads影像目录。

[Mnih(2013)博士论文第6章](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf)给出Roads的1500×1500、2.25km²规格，且明确发布前统一到1像素/平方米。因此本报告“1米发布尺度”“不新增插值”是相对发布影像，不能扩展成原传感器从未重采样。文件为RGB、EPSG26986/NAD83 Massachusetts Mainland、metre、PixelIsArea、north-up；四张坐标相邻的1500源图组成3000×3000发布像素。

来源页要求学术成果引用上述博士论文；没有把未明确的许可写成CC或Apache。URL、HTTP状态/失败、源文件字节/RGB SHA、作者页面与上游预处理说明全部保存。上游目录名train仅是原作者的划分，这56张选定影像从未进入本项目的本地训练或导航；它不是本项目训练集标记。

空间隔离按**完整闭合矩形足迹之间的最短距离**，不是中心点或名字。排除原151全部训练/开发/验证/测试足迹，14新区域之间（包括4工程与10确认）均要求≥3000米。实际旧/新最小间隔{a['minimum_old_region_gap_m']:.6f}米，新/新最小{a['minimum_new_region_pair_gap_m']:.6f}米；相邻标签舍入容差1毫米。源文件/像素无复用不是唯一依据；坐标缓冲与字节/RGB无重复分别核验。

| 区域 | 本项目角色 | UL、UR、LL、LR原始源 | 到旧151足迹最短公里 |
|---|---|---|---:|
{table}

每区9km²、300米/格、10×10/B20。合计126km²是14个隔离足迹之和，不能称一张126km²连续地图。确认10区合计90km²；当前仍是离散网格仿真，未开展真实航飞。来源与旧数据同属Massachusetts发布系列，不能称不同成像域、不同年份或全球预训练未见。

## 资源版本与确定性筛选

第一版触及120源上限，仅12/14区域合格，因此原v1判定为未通过，停止报告/方案/源文件/网络失败不覆盖。第二版在任何导航/模型响应之前只调整资源上限到200源、两轮合计1.5GiB；沿用相同坐标次序、3000米隔离、每源精确全白/全黑≤1%/每格≤2%、相同角色与任务种子。

最终累计下载152源，其中120源位于v1并只读复用，v2只增加32源；没有复制或移动大型旧影像。两轮接收共{received:.2f}MiB（含失败请求已收字节），在修订上限内。507个按文件名提出、通过旧足迹初筛的完整候选中，按真实GeoTIFF坐标/固定质量门槛接受前14个，42次候选因空白质量失败而记录排除；没有用图像风景、接缝强度、模型分数或SR挑选。

选定56源，全部已下载152源（包括未采用者）及日志保留。工程包约{size:.1f}MiB，含14张无损GeoTIFF、展示预览、1400 JPEG75图块、每格原源/像素窗口/投影边界与SHA；既有Masa/SwissView数据不搬移。源路径可能指向v1或v2，需按清单读取，不假设选定源均在v2/tiff内。

## 导航分层与独立接缝探针

每区75题无重复起终点路线：25长距离C12—16各5题；25边界目标和25非边界目标均C8—12各5题。水平/垂直源边界及两侧交替覆盖，双边界交点不进入边界分层；两分层初始距离配平。seed6203及协议/区域/分层/距离派生种子在下载之前冻结，69条错误目标无法匹配初始距离的例外逐题保存（wrong seed6211）。

| 本项目角色 | 长距离锚点 | 边界目标 | 非边界目标 | 导航题数 | 诊断探针 |
|---|---:|---:|---:|---:|---:|
| 4工程区 | {geometry['dev']['long_distance']} | {geometry['dev']['seam_target']} | {geometry['dev']['interior_target']} | 300 | 480 |
| 10确认区 | {geometry['test']['long_distance']} | {geometry['test']['seam_target']} | {geometry['test']['interior_target']} | 750 | 1200 |
| 全部14区 | 350 | 350 | 350 | 1050 | 1680 |

每区120探针：40个真实跨源相邻有向对（四方向各10）、40个相同目标的同源相邻对、40个相同目标的两格非邻接对。确认区有400个不同跨源有向对，比上一批仅1个导航题的3份权重机会有更明确的几何覆盖；实际探索仍可能到不了这些目标，所以“目标覆盖”“探针识别”“闭环SR”分开报告。

探针只准备当前/目标图对和评测标签，本轮没有任何目标头调用。后续探针接受/漏接/误接受不算导航SR，也不能用确认探针校准0.50阈值。不同距离范围和分层不能混成与原C12—16完全同队列的提升曲线；长距离锚点作为单独主分析候选。

## 环境、复核与适用边界

新`masa-roads-grid10-spatial-v1`明确工程dev/确认test，环境拒绝旧density/known-area Episode和错源绑定，公开Observation不新增目标位置、绝对坐标、源ID、全图或分层字段。四方向、越界原地耗一步/0米、首次到达（含末步）、每题独立历史；终局SG米=300×SG格、有效移动米=300×合法移动次数。绝对坐标和真实目标/接缝/探针标签仅在数据与评测端。

26项测试通过（原11、分层/空间/协议12、资源复用3）。独立实现用tifffile读取真实标签和像素，复算旧151足迹/RGB、152下载源及质量，重现第一批合格14区及全部拒绝，验证旧/新与新/新缓冲及源/RGB不复用；逐像素重建14地图和全部1400图块/JPEG、投影窗口/SHA；独立重建1050唯一任务、69错图例外、1680探针/四方向/配对目标，1050环境reset通过。

{a['protected_files_verified']}份既有文件、{a['input_files_verified']}输入与v1只读复用全部SHA保持。无策略动作、导航模型或目标头推理、训练、云端调用；本轮没有新SR或默认升级。SwissView原68已导航/32未导航状态未消费。SwissView发布说明及LV95坐标属瑞士，未把不同CRS的坐标数值直接比较；核心空间结论明确针对本地M0的151个Masa训练/评测足迹。

Sat2Cap的完整预训练足迹不能查明；仅可称对本地导航/视觉头任务训练及开发空间隔离，不能称对全球预训练模型未见。源区之间的3公里规则也不是统计独立性的充分证明；同州、同影像发布系列的环境相关性仍在。任意角度、真实异时、跨视角/跨模态、多Agent和实飞均未因此完成。

下一步先冻结M0在4工程区的接口兼容检查，随后对10确认区进行一次预登记评测和接缝探针诊断。具体预算与门槛见[下一项草案](下一项冻结M0空间隔离验证草案.md)，本轮未启动。

[方案](执行协议.md) / [预登记](核验/预登记.json) / [选择与拒绝](元数据/连续区域选择.json) / [输入](工程数据/数据清单.json) / [任务分层](工程数据/任务分层.json) / [探针](工程数据/邻接诊断探针.json) / [复核](核验/独立复核.json) / [判定](核验/验收结论.json)
''')
    text(RUN / '下一项冻结M0空间隔离验证草案.md', '''# 下一项草案：冻结M0工程接口与空间隔离确认（未启动）

使用已准备14区域、原Sat2Cap/Small256 NoTarget/EdgeTargetCue三权重、原均值与0.50门槛，不新增训练或云端因素。M0仍单智能体模块系统，不以这批数据宣称多Agent。

分两关且在首次模型调用前正式登记。4工程区只检验输入/特征/协议/合法动作/日志与重放；若发现实现问题保留日志，修复后另版本，不先打开确认结果来排错。工程结果不能用于更换确认源、目标采样、M0权重或阈值。

拟定闭环预算：4工程区300题×3权重×4条件=3600神经轨迹，两规则各300=600，共4200；10确认区750题×3权重×4条件=9000神经轨迹，两规则各750=1500，共10500。四条件Full、禁用cue、目标均值、错误目标；保留原Frontier及FixedRegion原行为/越界耗步。所有条件固定执行、同信息/任务/预算，合计14700；不因早期分数省略失败或追加策略。

拟定探针预算：14区1680探针×3权重×Full/Mean/Wrong=15120次头部判断；其中确认10区1200×3×3=10800，工程4区4320。探针不算导航SR、不能替代闭环，也不用于重新校准原阈值。须分别报告跨源/同源邻接的正确接受/漏接、非邻接误接受、四方向和每区域分母。

建议主分析为确认区250道长距离锚点（C12—16）×3权重；另外500题为距离配平的边界/非边界分层，各250×3。分别报告三个队列，不只展示混合750题SR。

建议正式确认门槛沿用原阶段的证据要求，而非仅总体SR60：长距离Full SR至少60%，相对每目标控制SR至少+5点、至少2/3权重正收益、全体SG不更差，并保留按10完整区域成组的配对区间；是否要求区间下界为正需在正式方案明确。该60%为项目预设，不是来源论文的导航成绩。分层和探针数值完整保留，接缝普遍可靠性应另明确正确接受与非邻接误接受门槛，不能从“准备400个跨源对”直接判通过。

SG同时给格/米与有效移动公里、失败单独SG、重访/OOB、末步成功、终止与接缝机会。原投影面积与6公里B20物理预算明确；不与旧不同队列SR相减。唯一确认后保留全部正负结果；原S2/S3/S4及M0默认不自动改写。具体实施/冻结/验收以之后正式方案为准，本草案无推理或新SR。
''')
    text(RUN / 'README.md', '# 本批入口\n\n[空间隔离数据准备报告](空间隔离连续区域数据准备报告.md) / [输入清单](工程数据/数据清单.json) / [复核](核验/独立复核.json) / [判定](核验/验收结论.json) / [下一项草案](下一项冻结M0空间隔离验证草案.md)。4工程/10确认区域已准备，无新导航SR。原始源按清单指向v1或v2位置，只读保留。\n')
    write(QA / '最终状态.json', dict(completed=True, data_engineering_passed=True, isolated_regions=14, engineering_regions=4,
          confirmation_regions=10, selected_raw_sources=56, all_downloaded_sources=152, derived_patches=1400,
          planned_tasks=1050, planned_probes=1680, navigation_evaluation_started=False, new_SR_produced=False,
          navigation_model_calls=0, cloud_calls=0, new_training_steps=0, default_changed=False,
          report_sha256=digest(RUN / '空间隔离连续区域数据准备报告.md'), audit_sha256=digest(QA / '独立复核.json'),
          verdict_sha256=digest(QA / '验收结论.json'), next='preregister frozen M0 engineering interface then10region confirmation'))
    write(META / '本项目模型使用状态.json', dict(regions=[dict(area=r['area'], source_tile=r['source_tile'], split=r['split'],
          source_ids=[s['id'] for s in r['sources']], metadata_and_pixel_QC_completed=True, navigation_model_used=False,
          cue_probe_used=False, local_training_used=False) for r in m['regions']], default_sha256=read(QA / '预登记.json')['default_sha256'],
          note='Project M0 usage only; pretrained encoder geography unknown; engineering4/confirmation10rolesfrozen'))


def plots(a, m):
    graphics.style(); graphics.ITEMS.clear()
    old = read(META / '旧151足迹.json')
    regions = m['regions']
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.45))
    for ax in axes:
        ax.set_xlabel('EPSG26986东向坐标（km）', fontsize=9.5)
        ax.set_ylabel('北向坐标（km）', fontsize=9.5)
        ax.set_aspect('equal'); ax.grid(alpha=.45); ax.set_axisbelow(True)
    for r in old:
        x0, y0, x1, y1 = np.asarray(r['bounds_m'])/1000
        axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=graphics.GRAY, edgecolor='white', linewidth=.2, alpha=.65))
    for i, r in enumerate(regions):
        x0, y0, x1, y1 = np.asarray(r['bounds_m'])/1000
        dev = r['split'] == 'dev'
        color, linestyle = (graphics.BLUE, '-') if dev else (graphics.ORANGE, '--')
        for ax in axes:
            ax.add_patch(Rectangle((x0, y0), 3, 3, facecolor=color, edgecolor=color, alpha=.55, linewidth=1, linestyle=linestyle))
        axes[1].text((x0+x1)/2, y1+.6, ('D'+str(i)) if dev else ('T'+str(i-4)), ha='center', va='bottom', fontsize=9)
    axes[0].set_xlim(94, 251); axes[0].set_ylim(860, 925)
    axes[1].set_xlim(98, 182); axes[1].set_ylim(862, 896)
    axes[0].set_title('(a) 与旧151足迹的真实空间关系', loc='left', fontsize=10)
    axes[1].set_title('(b) 新区域及工程/确认角色', loc='left', fontsize=10)
    axes[0].annotate('旧Masa：151足迹', xy=(236, 900), xytext=(211, 922), fontsize=9.5,
                     arrowprops=dict(arrowstyle='->', color=graphics.GRAY, lw=.8))
    axes[0].annotate('新14区', xy=(158, 884), xytext=(151, 908), fontsize=9.5,
                     arrowprops=dict(arrowstyle='->', color=graphics.ORANGE, lw=.8))
    fig.legend(handles=[Patch(facecolor=graphics.GRAY, label='旧151足迹'), Patch(facecolor=graphics.BLUE, edgecolor=graphics.BLUE, label='工程4区（D，实线）'),
                         Patch(facecolor=graphics.ORANGE, edgecolor=graphics.ORANGE, linestyle='--', label='确认10区（T，虚线）')],
               loc='center', bbox_to_anchor=(.52, .23), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.08, right=.985, top=.88, bottom=.33, wspace=.32)
    fig.text(.5, .115, f"完整足迹到旧区域最近{a['minimum_old_region_gap_m']/1000:.2f}km；新区域间至少3km（1毫米容差）。", ha='center', fontsize=9.5)
    fig.text(.5, .035, '每区9 km²；两面板均保持真实纵横比例，尺度不同。空间隔离不证明编码器预训练未见。', ha='center', fontsize=9)
    caption = '新增14个单区9km²区域与本地M0所用151旧足迹最短间隔50.289km，新区域两两间隔至少3km（坐标舍入容差1mm）。左图和右图按EPSG26986实测标签绘制，真实纵横比、不同视域；D表示4工程区，T表示10确认区，并使用实/虚线辅助颜色区分。单区由四相邻发布影像组成；14区不是一张连续126km²地图，空间隔离也不证明预训练编码器未见。'
    refs = [rel(DATA / '数据清单.json'), rel(META / '旧151足迹.json'), rel(QA / '独立复核.json'), rel(QA / '验收结论.json')]
    graphics.save(fig, FIG / '数据结果图', '43_新连续区域空间隔离与角色', caption, refs,
                  dict(regions=14, engineering=4, confirmation=10, old_footprints=151, min_gap_m=3000, navigation_SR=None))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8))
    ax = axes[0]
    for y in (0, 1.5):
        for x in (0, 1.5):
            ax.add_patch(Rectangle((x, y), 1.5, 1.5, facecolor='#F1F4F6', edgecolor='none'))
    for j in range(11):
        ax.axvline(j*.3, color='#C6CFD7', linewidth=.45); ax.axhline(j*.3, color='#C6CFD7', linewidth=.45)
    ax.axvline(1.5, color='#35424F', linestyle='--', linewidth=1.25)
    ax.axhline(1.5, color='#35424F', linestyle='--', linewidth=1.25)
    bank = read(DATA / '导航任务.json')
    meta = read(DATA / '任务分层.json')
    goals = [e for e in bank if e['area'] == regions[0]['area']]
    for layer, color, marker, filled, size in (('interior_target', graphics.BLUE, 'o', False, 67),
                                              ('long_distance', graphics.GRAY, '^', True, 30),
                                              ('seam_target', graphics.ORANGE, 's', True, 35)):
        cells = sorted({e['goal'] for e in goals if meta[e['episode_id']]['stratum'] == layer})
        ax.scatter([.3*(c%10)+.15 for c in cells], [.3*(c//10)+.15 for c in cells], s=size, marker=marker,
                   facecolors=color if filled else 'none', edgecolors=color, linewidths=.85, zorder=5)
    ax.set_xlim(0, 3); ax.set_ylim(3, 0); ax.set_aspect('equal')
    ax.set_xticks([0, 1.5, 3]); ax.set_yticks([0, 1.5, 3])
    ax.set_xlabel('相对东向距离（km）', fontsize=9.5); ax.set_ylabel('相对南向距离（km）', fontsize=9.5)
    ax.set_title('(a) img_2000目标覆盖（几何示意）', loc='left', fontsize=10)
    axes[1].barh([0, 1, 2], [160]*3, color=graphics.BLUE, height=.60, edgecolor='white', label='工程4区')
    axes[1].barh([0, 1, 2], [400]*3, left=[160]*3, color=graphics.ORANGE, height=.60, edgecolor='white', label='确认10区')
    for i in range(3):
        axes[1].text(80, i, '160', ha='center', va='center', color='white', fontsize=10)
        axes[1].text(360, i, '400', ha='center', va='center', color='white', fontsize=10)
    axes[1].set_yticks([0, 1, 2], ['跨源真邻接', '同源真邻接', '两格非邻接'])
    axes[1].invert_yaxis(); axes[1].set_xlim(0, 600)
    axes[1].set_xticks([0, 200, 400, 600]); axes[1].set_xlabel('冻结探针对数（非导航轨迹）', fontsize=9.5)
    axes[1].grid(axis='x', alpha=.5); axes[1].set_axisbelow(True)
    axes[1].set_title('(b) 1680探针的角色分配', loc='left', fontsize=10)
    fig.subplots_adjust(left=.07, right=.975, top=.87, bottom=.31, wspace=.82)
    fig.legend(handles=[Line2D([], [], marker='s', linestyle='', color=graphics.ORANGE, label='边界目标25题/区'),
                         Line2D([], [], marker='o', markerfacecolor='none', linestyle='', color=graphics.BLUE, label='非边界25题/区'),
                         Line2D([], [], marker='^', linestyle='', color=graphics.GRAY, label='长距离25题/区')],
               loc='center', bbox_to_anchor=(.48, .20), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .105, '每区75条不同路线；相同目标位置可重复，左图按位置去重绘制。虚线为1500米源边界。', ha='center', fontsize=9)
    fig.text(.5, .035, '每区40跨源 + 40同源 + 40非邻接探针；同一目标配对。只准备覆盖，没有模型判断或新SR。', ha='center', fontsize=9)
    caption = '确认10区已准备250道边界目标任务和400个不同跨源有向邻接对，覆盖不再只依赖同一题的三权重机会。每区75不同路线由25长距离C12—16、25边界目标C8—12、25非边界C8—12组成；左图为首工程区实际冻结目标位置的纯几何示意，重复目标去重，虚线表示原源边界。右图显示三类探针各560对，其中工程160/确认400，每一跨源对配相同目标的同源邻接和两格非邻接；图表示准备样本数，不是导航SR、模型接受率或识别验收。'
    refs = [rel(DATA / '导航任务.json'), rel(DATA / '任务分层.json'), rel(DATA / '邻接诊断探针.json'), rel(QA / '独立复核.json')]
    graphics.save(fig, FIG / '数据结果图', '44_接缝目标分层与诊断探针覆盖', caption, refs,
                  dict(tasks=1050, probes=1680, confirm_seam_targets=250, confirm_cross_pairs=400, neural_inference=False))
    for item in graphics.ITEMS: item['reproduction_script'] = rel(Path(__file__))
    write(FIG / '绘图数据/空间隔离连续区域准备_v2.json', dict(date='2026-10-02', figures=graphics.ITEMS,
          script_sha256=digest(Path(__file__)), source_sha256={p: digest(ROOT / p) for i in graphics.ITEMS for p in i['sources']},
          manifest=m, audit=a))
    text(FIG / '空间隔离连续区域图注_v2.md', '\n\n'.join('# '+i['id']+'\n\n'+i['caption'] for i in graphics.ITEMS)+'\n')
    catalog = read(FIG / '图表来源清单.json')
    assert not {i['id'] for i in graphics.ITEMS} & {i['id'] for i in catalog['figures']}
    catalog['figures'].extend(graphics.ITEMS)
    (FIG / '图表来源清单.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    text(FIG / '审阅/空间隔离区域作图设计_v2.md', '''# 科研配图设计与核验入口

支持性数据工程图，沿用figure-designer的物理坐标/分类计数、矢量、双编码及自包含图注规范。图43以真实EPSG足迹展示空间关系，两视域保持真实纵横比，工程实线D/确认虚线T区别；图44用实际冻结目标格的几何示意和三类探针计数，零起点计数轴，不画虚构SR或CI。

Matplotlib8.4英寸画布，中文9pt以上，建议插入宽度≥20cm以保留8pt以上；蓝/橙/灰与形状/线型辅助，不仅依赖颜色。PDF/SVG全矢量，无包裹位图；PNG300dpi。标题为面板定义，图注第一句陈述数据准备结论，并说明三权重机会与真实不同任务的区别。

需在导出后核对PDF嵌入字体、SVG文本/无image、PNG/PDF校样的图例与标签。源片像素内容另用14区域联系表审阅；联系表是栅格质量证据，不代替科研图的矢量导出。没有论文Introduction一致性检查要求，本图不是新的动机图。\n''')
    (FIG / '审阅/空间隔离区域作图源码_v2.py').write_bytes(Path(__file__).read_bytes())
    # Read-only previews for QA, generated after region/task selection is frozen.
    contact = Image.new('RGB', (2040, 2220), 'white')
    draw = ImageDraw.Draw(contact)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    for i, r in enumerate(regions):
        x, y = (i%4)*510, (i//4)*550
        with Image.open(DATA / 'previews' / (r['area']+'.jpg')) as im:
            thumbnail = im.copy(); thumbnail.thumbnail((490, 490)); contact.paste(thumbnail, (x+10, y+40))
        draw.text((x+10, y+6), ('工程 ' if r['split']=='dev' else '确认 ')+r['area'], fill='black', font=font)
    contact.save(FIG / '审阅/空间隔离区域14张预览联系表_v2.png')


def navigation(a):
    summary = (f"14个空间隔离连续区域的数据/协议已通过：4工程、10确认，每区9 km²/300米每格，1400图块、1050不同导航路线和1680诊断探针。"
               f"与旧151足迹最近{a['minimum_old_region_gap_m']/1000:.3f}公里，新区域间至少3公里；26测试及逐像素/任务/空间复核通过。"
               "当前没有新SR或模型调用，空间隔离不证明Sat2Cap预训练未见；下一项为冻结M0工程接口及10区确认。")
    links = f'[报告]({rel(RUN / "空间隔离连续区域数据准备报告.md")}) / [复核]({rel(QA / "独立复核.json")}) / [判定]({rel(QA / "验收结论.json")}) / [下一项草案]({rel(RUN / "下一项冻结M0空间隔离验证草案.md")}) / [图43—44](绘图/空间隔离连续区域图注_v2.md)。'
    p = ROOT / 'README.md'; v = p.read_text('utf-8')
    v = v.replace('下一项为未知连续区域的数据与隔离协议准备。', '空间隔离连续区域数据已通过，下一项为冻结M0接口及10区确认。', 1)
    v = v.replace('## 最新阶段：实际面积冻结M0兼容试跑完成', '## 最新阶段：新连续区域空间隔离数据准备通过\n\n'+summary+'\n\n'+links+'\n\n## 上一阶段：实际面积冻结M0兼容试跑完成', 1)
    v = v.replace('63批实验', '65批实验').replace('42张实验与数据协议图', '44张实验与数据协议图')
    v = v.replace('后续优先准备空间隔离连续新区域。', '新连续区域准备已通过，后续先冻结M0工程接口和10区确认。')
    v = v.replace('DATA/processed_data/{Masa,SwissView}', 'DATA/processed_data/{Masa,MasaRoads,SwissView}')
    p.write_text(v, encoding='utf-8')
    p = NAV / '当前进度.md'; v = p.read_text('utf-8').replace('下一项准备空间隔离的新连续区域。', '新连续区域数据准备已通过，下一项冻结M0工程接口及10区确认。', 1)
    v += '\n## 空间隔离新连续区域：数据准备通过\n\n'+summary+'\n\n'+links.replace('](', '](../')+'\n\n'
    v += 'v1在120源上限停止于12区域，原未通过记录保留；v2只调整资源上限、只读复用120源并新增32源，按相同固定条件凑齐14区域。确认前不把数据通过当未知地理导航通过。\n'
    p.write_text(v, encoding='utf-8')
    idx = read(NAV / '实验索引.json')
    assert len(idx['batches']) == 63
    for path, status, scope, reports in ((STOP, '资源上限停止；未通过', '120源/12合格区域；无导航SR', ['资源上限停止报告.md']),
                                        (RUN, '数据/协议与空间隔离通过；导航未启动', '14区域/56选定源/1050任务/1680探针；无新SR', ['空间隔离连续区域数据准备报告.md'])):
        idx['batches'].append(dict(dataset='MasaRoads', kind='数据准备', name=path.name, category='空间隔离数据', status=status,
            scope=scope, path=rel(path), reports=[rel(path / f) for f in reports],
            evidence=[rel(path / '核验/验收结论.json'), rel(path / ('核验/停止状态.json' if path == STOP else '核验/独立复核.json'))]))
    (NAV / '实验索引.json').write_text(json.dumps(idx, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    p = NAV / '实验索引.md'; v = p.read_text('utf-8')
    for path, outcome, filename in ((STOP, '资源上限停止，未通过', '资源上限停止报告.md'), (RUN, '空间隔离数据通过，无新SR', '空间隔离连续区域数据准备报告.md')):
        v += f'| MasaRoads / 数据准备 | {path.name} | {outcome} | [报告](../{rel(path / filename)}) / [判定](../{rel(path / "核验/验收结论.json")}) / [原目录](../{rel(path)}) |\n'
    p.write_text(v, encoding='utf-8')
    for name in ('README.md', '目录结构与维护.md'):
        p = NAV / name; p.write_text(p.read_text('utf-8').replace('63批', '65批'), encoding='utf-8')
    p = FIG / 'README.md'; first, rest = p.read_text('utf-8').split('\n', 1)
    p.write_text(first+'\n\n## 最新：新连续区域的空间隔离与接缝覆盖\n\n'+summary+'\n\n[图43：空间隔离](数据结果图/43_新连续区域空间隔离与角色.pdf) / [图44：覆盖与探针](数据结果图/44_接缝目标分层与诊断探针覆盖.pdf) / [图注](空间隔离连续区域图注_v2.md) / [数据与SHA](绘图数据/空间隔离连续区域准备_v2.json)。当前是数据准备，无新导航SR。\n'+rest, encoding='utf-8')


def main():
    assert not (RUN / '空间隔离连续区域数据准备报告.md').exists(), 'immutable closeout exists'
    assert not any((FIG / '数据结果图' / (name+'.'+ext)).exists() for name in ('43_新连续区域空间隔离与角色', '44_接缝目标分层与诊断探针覆盖') for ext in ('png', 'pdf', 'svg'))
    v, a, m, s = required()
    report(v, a, m, s); plots(a, m); navigation(a)
    print(dict(report=True, new_figures=2, index_entries=65, data_engineering_passed=True, new_SR=False), flush=True)


if __name__ == '__main__': main()
