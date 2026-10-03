"""Data-only fresh-area report, confirmation draft and one scientific figure."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import sys
import warnings

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'project/src'))
from documents import stage_figures as graphics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle,Patch
import numpy as np

RUN=ROOT/'DATA/processed_data/MasaRoads/径向保护新区域确认_v1'
DATA,META,QA=RUN/'工程数据',RUN/'元数据',RUN/'核验'
FIG=ROOT/'绘图'
CANDIDATE=ROOT/'DATA/processed_data/MasaRoads/评测结果/覆盖接管径向保护验证_v1'
REPORT=RUN/'新区域确认数据准备报告.md'
DRAFT=RUN/'下一项冻结径向保护新区域导航确认方案_v1.md'
CAPTION=FIG/'径向保护新区域确认准备图注.md'
MANIFEST=FIG/'绘图数据/径向保护新区域确认准备_v1.json'
STEM='49_径向保护新确认区域与任务覆盖'


def read(p):return json.loads(p.read_text('utf-8-sig'))
def digest(p):
    h=sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2))


def main():
    audit,verdict=read(QA/'独立复核.json'),read(QA/'验收结论.json')
    assert audit['passed'] and verdict['data_engineering_passed'] and not verdict['new_SR_produced']
    assert verdict['audit_sha256']==digest(QA/'独立复核.json')
    assert verdict['raw_and_derived_input_freeze_sha256']==digest(QA/'输入冻结结束.json')
    assert read(QA/'预登记.json')['default_sha256']==digest(ROOT/'project/local_policy_default.json')
    selection=read(META/'连续区域选择.json');regions=selection['regions']
    old=read(META/'旧151足迹.json');consumed=read(META/'已消费14连续区域.json')['regions']
    sources=read(META/'下载源坐标清单.json');usage=read(META/'本项目模型使用状态.json')
    assert len(regions)==10 and all(not r['navigation_called'] and not r['visual_head_called'] for r in usage['regions'])
    outputs=[REPORT,DRAFT,RUN/'README.md',CAPTION,MANIFEST]
    outputs.extend(FIG/'数据结果图'/f'{STEM}.{ext}' for ext in ('png','pdf','svg'))
    assert not any(p.exists() for p in outputs)
    paths=[p for p in META.iterdir() if p.is_file()]+[p for p in QA.glob('*.json')]
    paths.extend(p for p in DATA.glob('*.json'))
    paths.extend((RUN/'执行协议.md',RUN/'测试记录.json',CANDIDATE/'验收结论.json',CANDIDATE/'候选配置.json',
        CANDIDATE/'下一项空间隔离新区域确认草案.md',ROOT/'project/local_policy_default.json',
        Path(__file__),ROOT/'project/src/documents/stage_figures.py'))
    selected_ids={s['id'] for r in regions for s in r['sources']}
    reused_ids={s['id'] for s in read(META/'只读复用源坐标清单.json')}
    selected_reused=len(selected_ids&reused_ids)
    table='| 新区 | 源图（UL/UR/LL/LR） | 至旧151最短公里 | 至旧14区最短公里 |\n|---|---|---:|---:|\n'
    for r in regions:
        table+=f"| {r['area']} | {' / '.join(s['id'] for s in r['sources'])} | {r['min_old_gap_m']/1000:.3f} | {r['min_consumed_region_gap_m']/1000:.3f} |\n"
    caption=('数据准备阶段，仅元数据和固定像素质量筛选；无导航模型、编码器或视觉头预测。左图为EPSG:26986闭合足迹，'
        '包含原151影像、已消费14连续区域及10个预留确认新区；轴用投影公里，不是经纬度。'
        '新旧及新区之间的最短矩形距离独立核验均至少3公里。右图为冻结750条不同导航任务的三个分层（各250）'
        '和1200个匹配目标几何探针的三个类别（各400）；探针不计入导航SR。每新区9 km²、300米/格、10×10/B20，'
        '共40幅选中源图及1000图块。该图不包含新的模型成功率。')
    graphics.style();fig,axes=plt.subplots(1,2,figsize=(12.5,5.5),gridspec_kw={'width_ratios':[1.2,1]})
    colors=['#7C8792','#D55E00','#009E73']
    for j,group in enumerate((old,consumed,regions)):
        for r in group:
            x0,y0,x1,y1=np.asarray(r['bounds_m'])/1000
            axes[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=colors[j],edgecolor=colors[j],alpha=.4 if j==0 else .7,linewidth=.6))
    all_bounds=np.asarray([r['bounds_m'] for r in old+consumed+regions])/1000
    axes[0].set_xlim(all_bounds[:,0].min()-4,all_bounds[:,2].max()+4)
    axes[0].set_ylim(all_bounds[:,1].min()-4,all_bounds[:,3].max()+4)
    axes[0].set_aspect('equal',adjustable='box');axes[0].set_xlabel('东向投影坐标（公里）');axes[0].set_ylabel('北向投影坐标（公里）')
    axes[0].grid(alpha=.45);titles=[axes[0].set_title('(a) 历史足迹与预留新区',loc='left')]
    counts=[250,250,250,400,400,400];xs=np.arange(6)
    axes[1].bar(xs,counts,color=['#0072B2']*3+['#009E73']*3,edgecolor='#243746',linewidth=.6,zorder=3)
    for x,n in zip(xs,counts):axes[1].text(x,n+12,str(n),ha='center',fontsize=10)
    axes[1].set_xticks(xs,['长距离\nC12—16','接缝\nC8—12','内部\nC8—12','跨源\n邻接','同源\n邻接','非邻接\n控制'])
    axes[1].set_ylim(0,470);axes[1].set_ylabel('冻结条目数');axes[1].set_axisbelow(True);axes[1].grid(axis='y',alpha=.55)
    titles.append(axes[1].set_title('(b) 确认任务与几何探针',loc='left'))
    handles=[Patch(facecolor=c,label=label,alpha=.7) for c,label in zip(colors,['原151影像','已消费14区域','预留10新区'])]
    handles.append(Patch(facecolor='#0072B2',label='750导航任务'))
    legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=4,frameon=False)
    fig.subplots_adjust(left=.075,right=.985,top=.79,bottom=.23,wspace=.35)
    fig.text(.5,.055,f"最小间距：至旧151 {audit['minimum_old_region_gap_m']/1000:.2f} km；至旧14区 {audit['minimum_consumed14_region_gap_m']/1000:.2f} km；新区间 {audit['minimum_new_region_pair_gap_m']/1000:.2f} km。",ha='center',fontsize=9)
    fig.text(.5,.02,'数据与空间复核通过；真实导航尚未启动，不产生新SR。Sat2Cap预训练地理覆盖未知。',ha='center',fontsize=9)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    assert all(not legend.get_window_extent(renderer).overlaps(t.get_window_extent(renderer)) for t in titles)
    files=[]
    for ext in ('png','pdf','svg'):
        b=BytesIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');fig.savefig(b,format=ext,dpi=300,bbox_inches='tight')
        assert not any('Glyph' in str(w.message) for w in caught)
        p=FIG/'数据结果图'/f'{STEM}.{ext}'
        with p.open('xb') as f:f.write(b.getvalue())
        files.append(dict(path=rel(p),sha256=digest(p)))
    plt.close(fig)
    report=f'''# 径向保护新区域确认：数据准备报告

2026-10-03。**数据工程与空间隔离复核通过，尚无新SR。** 候选与默认M0保持，下一项为固定新区域上的导航确认。

## 数据来源与固定选择

作者Massachusetts Roads发布影像的冻结1108文件清单，发布像素1米，EPSG:26986。排除原151幅M0源影像及已消费14连续区域的整个足迹，另按固定文件坐标顺序、3公里缓冲和既定空白像素阈值取最早10个合格区域，不以模型分数挑区。公共下载来源为[作者数据目录](https://www.cs.toronto.edu/~vmnih/data/)。上游train标签不等于项目训练；Sat2Cap预训练地理范围未知。

只读复用原152下载源。新获取{selection['newly_downloaded_sources']}张，响应体{selection['received_bytes']/1024**2:.2f} MiB；新增上限200张/1.5 GiB（含失败重试）。选中40幅源图中{40-selected_reused}幅本轮新增、{selected_reused}幅来自从未用于模型预测的只读旧下载。旧原图未复制、移动或覆盖。候选组{selection['complete_isolated_filename_candidates']}，完整质量/几何拒绝数{len(selection['rejected_before_selection'])}，详见元数据记录。

{table}

最短闭合矩形间距：新区域至原151为{audit['minimum_old_region_gap_m']:.3f}米、至已消费14区为{audit['minimum_consumed14_region_gap_m']:.3f}米；新区域彼此为{audit['minimum_new_region_pair_gap_m']:.3f}米。区域源图连续性、GeoTIFF标签和逐像素内容都独立核验，不能只凭文件名或中心距离认定隔离。

## 固定输入与评测权限

10区×9 km²、每格300米、10×10/B20，共1000个原生300×300图块、10幅无融合/无重采样拼接影像。750个不同导航路线（每区每层25题，三层各250）；1200个几何匹配探针（跨源、同源、非邻接各400），探针不计SR。错目标初始距离不能匹配的例外{audit['wrong_distance_exceptions']}条，固定保留。

采样seed6203/错目标seed6211；JPEG质量75与旧输入逻辑一致。保存每格来源图、像素窗口、投影边界、原生/JPEG/文件SHA。数据使用账本全部为预留确认角色：编码器、头部、导航均未调用。只对真实任务reset750次检查接口，没有在真实新区执行动作。

## 核验与结论范围

12项新增检查通过。独立实现以tifffile解析全部源标签/RGB、重做固定次序选择、重建10幅拼接及1000个JPEG、复算750任务和1200探针，验证下载/重试/复用账本、空间缓冲与冻结哈希。算术实现由同一会话执行，不冒称另一位人工审阅者。原数据、模型、历史结果和默认SHA保持。

本阶段通过只说明有可追溯的独立确认输入，不能认为候选已在新区域提高SR，也不能替换默认。仍是同Massachusetts发布影像家族的本地切图仿真，不证明编码器未见当地，不是任意角度、异时、跨模态或实飞结果。

[数据清单](工程数据/数据清单.json) / [任务](工程数据/导航任务.json) / [选择与拒绝](元数据/连续区域选择.json) / [使用状态](元数据/本项目模型使用状态.json) / [独立复核](核验/独立复核.json) / [判定](核验/验收结论.json) / [冻结导航草案](下一项冻结径向保护新区域导航确认方案_v1.md)。

![图49](../../../../绘图/数据结果图/{STEM}.png)

图注：{caption}
'''
    write(REPORT,report)
    write(RUN/'README.md','# 径向保护新区域确认数据 v1\n\n数据/空间隔离通过；10预留新区，1000图块，750任务、1200几何探针。尚未导航、尚无新SR，默认M0保持。\n\n[报告](新区域确认数据准备报告.md) / [复核](核验/独立复核.json) / [判定](核验/验收结论.json) / [下一项草案](下一项冻结径向保护新区域导航确认方案_v1.md)。\n')
    draft=f'''# 下一项冻结导航确认方案 v1（待执行）

2026-10-03。本数据批次通过工程复核，正式导航尚未启动。本方案是下一项的完整草案，导航前还须建立独立评测批次、冻结代码/缓存/输入并验证接口；不得运行旧脚本覆盖既有输出。

输入只使用本轮10预留新区/750任务，三个队列各250，每格300米、10×10/B20。输入冻结SHA：`{digest(QA/'输入冻结结束.json')}`；数据审计SHA：`{digest(QA/'独立复核.json')}`。区域全部未被编码器/头部/导航使用；正式运行的消费状态另记，不修改本准备批次的冻结账本。

候选Coverage3Radial配置SHA：`{digest(CANDIDATE/'候选配置.json')}`。原三步/至少2格覆盖和单步起点径向否决不改，已接受cue优先。M0/候选使用同三权重、头部、均值和0.50，同预算/信息权限，逐题清空隐状态；无新增训练或云端调用。

主对照750题×3权重×M0/候选Full=4500轨迹。升级需全部通过：固定混合SR对M0≥+2点、至少2/3权重正、混合SG不变差；每层SR损伤≤2点且SG不变差；4000次/seed7317的10区域配对混合SR差95%下界>0。报告所有恢复/损伤、失败SG、有效移动米、保护和cue触发，不换分母或只保留长距离。

主数值及独立重放通过，才做候选Baseline/CueMean/CueWrong共6750条；最大新增11250导航。Full对每控制SR≥+5点、≥2/3正、SG不变差、区域95%差下界>0。保存逐步终局、显式神经方程/独立集合规划、输入图块和特征绑定、来源bootstrap。任一必要门槛失败即停止升级，不调参数、换区或松门槛。全部独立证据通过后才决定默认替换，当前默认M0保持。

特征提取只允许发生在此数据准备已封存之后，使用冻结Sat2Cap和相同预处理；缓存/模型/任务哈希在第一个动作前登记。真值坐标、初始距离、分层和源身份仅评测器拥有，控制器仍只接收五项公开输入。数据准备期间的1200几何探针没有视觉识别成绩；若将来诊断头部必须分开报告，不计导航SR。

结论限同Massachusetts影像家族的新地理足迹，预训练地理范围未知；不扩张为异时/任意角度/跨模态/真实无人机结论。
'''
    write(DRAFT,draft);write(CAPTION,'# 径向保护新区域确认准备图注\n\n'+caption+'\n')
    item=dict(id=STEM,files=files,caption=caption,protocol='fresh10reserved regions;750tasks;1200geometry probes;1000native patches; no model predictions or SR',sources=[rel(p) for p in paths])
    write(MANIFEST,dict(date='2026-10-03',source_sha256={rel(p):digest(p) for p in paths},
        reproduction_script=rel(Path(__file__)),script_sha256=digest(Path(__file__)),audit=audit,verdict=verdict,selection=selection,
        figures=[item],other_outputs_sha256={rel(p):digest(p) for p in (REPORT,DRAFT,RUN/'README.md',CAPTION)}))
    print(dict(report=rel(REPORT),figure=STEM,new_SR_produced=False,next_draft=rel(DRAFT)),flush=True)


if __name__=='__main__':main()
