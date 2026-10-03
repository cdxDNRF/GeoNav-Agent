"""Close out the footprint-data gate with source-bound scientific protocol figures."""
from pathlib import Path
import json
import hashlib
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'DATA/processed_data/Masa/实际区域扩展_v1'
DATA=RUN/'工程数据';QA=RUN/'核验';FIG=ROOT/'绘图';NAV=ROOT/'项目导航'


def read(p):return json.loads(p.read_text('utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def text(p,v):
    with p.open('x',encoding='utf-8') as f:f.write(v)


def required():
    v=read(QA/'验收结论.json');a=read(QA/'独立复核.json');m=read(DATA/'数据清单.json');s=read(RUN/'连续区域选择.json')
    assert v['data_engineering_passed'] and a['passed'] and not v['navigation_evaluation_started'] and not v['new_SR_produced']
    assert digest(QA/'独立复核.json')==v['audit_sha256'] and digest(QA/'输入冻结结束.json')==v['input_freeze_sha256']
    return v,a,m,s


def reports(v,a,m,s):
    rows='\n'.join(f"| {r['area']} | {'、'.join(x['id'] for x in r['sources'])} | 3000×3000 | 9 | 100 |" for r in m['regions'])
    rejected='；'.join('/'.join(x['rejected_ids']) for x in s['rejected_before_selection']) or '无'
    size=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())/2**20
    text(RUN/'实际区域扩展数据准备报告.md',f'''# 实际区域扩展：数据与协议工程已通过

**已准备三张连续3000×3000地图，保持原1米/像素和300米/格，单地图投影面积由2.25增至9平方公里。** 本轮是已知训练区域的数据与环境工程验收，尚未运行导航模型或产生新SR，原M0默认保持。

## 从密度扩展到实际足迹

| 协议 | 原始覆盖边长 | 原始图块尺度 | 输入图块 | 网格 | 面积 | 动作预算 | 有效移动上限 |
|---|---:|---:|---|---|---:|---:|---:|
| 既有原生5×5 | 1500米 | 300米/格 | 300×300 | 5×5 | 2.25 km² | B10 | 3 km |
| 既有10×10密度 | 1500米 | 150米/格 | 150裁片插值至300 | 10×10 | 2.25 km² | B20 | 3 km |
| 本轮10×10实际足迹 | 3000米 | 300米/格 | 原生300×300 | 10×10 | 9 km² | B20 | 6 km |

同一个B20并不表示相同物理航程；扩大足迹和改用原生图块后，不把后续SR变化单独归因于面积。面积按EPSG:26986投影坐标计算，不声称额外测得椭球面面积。

## 源文件与构建

本地151张Masa TIFF均通过GeoTIFF标签/尺寸/单位检查。151图中存在99个完整2×2组合、56个全train组合；按预先固定坐标顺序选择三组不复用源的工程区域。

| 工程地图 | UL、UR、LL、LR原始源图 | 原生像素 | 投影面积km² | 图块 |
|---|---|---|---:|---:|
{rows}

空白质量门槛为每源精确全白/全黑≤1%、每格≤2%；选择前排除含{rejected}的候选，记录在[区域选择](连续区域选择.json)。没有按导航结果或接缝评分挑图。源文件仍在原址，不搬移大型数据；派生工程包约{size:.1f} MiB，包含三张无损GeoTIFF、展示缩略图、300个JPEG75图块和来源/任务JSON。

十二源均来自原train，不占用原val/test像素足迹，三组不复用源。它们是已知训练区域，不能当作十二张新源或三张独立泛化地图；也不能由三组不复用源推断它们严格地理独立。原PNG与TIFF像素一致，所有300格JPEG与旧5×5同源图块逐字节一致。

原图实际投影位置决定拼接次序；没有图像融合、填洞或原生像素插值。全部151图的共同格网对齐允许1毫米标签舍入；本批选定区域的四图对齐残差最大{a['native_coordinate_max_alignment_residual_m']:.6f}米。展示缩略图仅供目视查看，不进入Agent输入。

## 任务、信息权限与复核

新协议`masa-local-grid10-area-v1`与旧密度协议分开：10×10/B20、四方向、越界原地耗步、首次到达（包含末步）、episode内记忆。每地图C12—16各5题，共75题，seed5017；这些距离必跨原1500源图边界。错误目标计划在导航之前冻结，匹配初始距离的例外保留。

新环境拒绝旧密度Episode/清单，终局另给SG米=300×SG格、有效移动米=300×合法移动次数。Agent仍只有公开Observation的当前/目标图、当前位置、预算和访问历史；GeoTIFF坐标、源身份、全图、目标索引及真距离留在数据和评测端。图像payload剥离路径/EXIF。

11项测试通过，涵盖错误地理标记、CRS/单位/分辨率、旋转、几何缺块/错放/复用、协议混用、边界耗步、末步成功和信息权限。独立实现使用tifffile与PIL交叉读标签与像素，重建三张mosaic/全部300格，核对同源旧图块、固定任务及错目标；75次环境reset只检查输入和协议，没有策略动作、模型调用或新SR。

接缝MAE和邻近行列差值完整保留在[质量诊断](核验/图像质量与接缝.json)，只描述像素差异，不证明视觉头跨接缝可靠。原代码/默认/正式判定及十二源数据保护；没有训练、云端调用或SwissView新源消耗（仍68已导航、32尚未）。

## 下一项

下一步是三地图75题×3权重的冻结M0兼容试跑，配合同信息规则与目标控制，检查跨原图接缝的线索是否可靠。草案见[下一项试跑方案](下一项导航兼容试跑草案.md)。它是已知区域的小型工程测试；正式实际范围扩展和未知地理区域确认仍需足够、空间隔离的连续影像。不能拿数据工程通过替代导航SR或原S2/S3/S4升级。

[方案](执行协议.md) / [坐标清单](源图坐标清单.json) / [图块来源](工程数据/图块来源.json) / [固定任务](工程数据/导航任务.json) / [复核](核验/独立复核.json) / [判定](核验/验收结论.json)

## 原始来源

原图规格与[Mnih博士论文6.1节](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf)的151张、每图1500×1500/2.25平方公里描述一致；[原数据页面](https://www.cs.toronto.edu/~vmnih/data/)提供来源和引用要求。像素到模型坐标的读取依据[OGC GeoTIFF规范](https://docs.ogc.org/is/19-008r4/19-008r4.html)，本地完整标签及像素复核作为当前工程证据。
''')
    text(RUN/'下一项导航兼容试跑草案.md','''# 下一项：冻结M0在实际面积工程地图上的兼容试跑（草案，未启动）

前提：本批数据/协议工程及视觉预览核验通过，全部图块/任务/默认SHA绑定。使用三张已知train区域的75固定题、10×10/B20、原Small256 NoTarget+EdgeTargetCue三权重/0.50阈值；不加入F、不训练、不调用云端。

拟固定M0真实、禁用cue、均值目标、错误目标四条件，各75×3=225条，总900神经轨迹；同信息的Frontier和FixedRegion各75条，总预算1050。后续运行前另登记新的输出版本、运行器、基线合法动作/10×10适配、编码器输入冻结和源码快照，不能在本数据报告目录覆盖旧日志。

工程门槛是全部计划合法终局、真实动作/图像/GRU/环境可重放，所有题留在分母。目标证据按原门槛：真实相对各目标控制SR至少+5点、至少2/3权重为正、SG不更差。保留按三整地图分组的区间和每源块/距离/接缝附近指标；三个已知地图不足以支持稳定泛化或正式升级。先定位是否发生跨源接缝误接受、原生300米图块识别失败或探索预算不足，再决定是否需要下一因素。

输出同时报告SG格/米、合法移动米、预算终止、重访、规则差距、错误目标匹配例外，以及目标/当前输入对应的原始源图。接缝标记、真实距离与目标位置只用于评测端事后分组，不进入Agent。

不把新SR与旧密度协议不同队列直接相减，不把B20声称成相同物理航程；数据工程通过也不替代导航通过。一次冻结试跑后保存全部正负结果，当前默认不因小队列自动升级。正式未知区域确认需另建空间隔离连续区域。
''')
    text(RUN/'README.md','# 本批入口\n\n[数据准备报告](实际区域扩展数据准备报告.md) / [工程输入](工程数据/数据清单.json) / [75题](工程数据/导航任务.json) / [复核](核验/独立复核.json) / [判定](核验/验收结论.json) / [下一项草案](下一项导航兼容试跑草案.md)。本批没有新SR。\n')
    write(QA/'最终状态.json',dict(completed=True,data_engineering_passed=True,regions=3,source_rasters=12,patches=300,tasks=75,
        known_training_geography=True,navigation_evaluation_started=False,new_SR_produced=False,navigation_model_calls=0,
        new_training_steps=0,cloud_calls=0,default_changed=False,report_sha256=digest(RUN/'实际区域扩展数据准备报告.md'),
        verdict_sha256=digest(QA/'验收结论.json'),audit_sha256=digest(QA/'独立复核.json'),next='separately register frozen M0 actual-area compatibility pilot'))


def plots(m):
    graphics.style();graphics.ITEMS.clear()
    fig,axes=plt.subplots(1,3,figsize=(8.4,4.4),sharex=True,sharey=True)
    configs=[('原生5×5',1.5,.3,graphics.GRAY,'300米/格；B10','移动上限3公里'),
             ('旧10×10密度',1.5,.15,graphics.ORANGE,'150米/格；B20','移动上限3公里'),
             ('新10×10实际足迹',3,.3,graphics.BLUE,'300米/格；B20','移动上限6公里')]
    for i,(title,side,cell,color,line1,line2) in enumerate(configs):
        ax=axes[i];ax.add_patch(Rectangle((0,0),side,side,facecolor=color,alpha=.15,edgecolor=color,lw=1.5))
        n=round(side/cell)
        for j in range(n+1):
            v=j*cell;ax.plot([v,v],[0,side],color=color,lw=.6);ax.plot([0,side],[v,v],color=color,lw=.6)
        ax.text(side/2,side/2,f'{side*side:g} km²',ha='center',va='center',fontsize=12,fontweight='bold',bbox=dict(fc='white',ec='none',alpha=.9,pad=2))
        ax.set_xlim(0,3.06);ax.set_ylim(3.06,0);ax.set_aspect('equal');ax.set_xticks([0,1.5,3]);ax.set_yticks([0,1.5,3])
        ax.set_title(title,fontsize=10.5,loc='left');ax.set_xlabel('相对东向距离（km）',fontsize=9)
        ax.text(.5,-.25,line1+'\n'+line2,transform=ax.transAxes,ha='center',va='top',fontsize=9)
    axes[0].set_ylabel('相对南向距离（km）',fontsize=9)
    fig.subplots_adjust(left=.075,right=.99,top=.87,bottom=.33,wspace=.19)
    fig.text(.5,.08,'三图采用相同物理坐标轴；框内为可搜索足迹。新地图保留原生1米/像素及300米/格。',ha='center',fontsize=9)
    fig.text(.5,.025,'数据工程通过；三已知训练区域，未运行导航。B为动作次数，物理移动预算随每格尺度变化。',ha='center',fontsize=9)
    caption='协议尺度对比：原5×5为1.5km边长/300m每格/2.25km²/B10；旧10×10只提高密度，为1.5km边长/150m每格/2.25km²/B20；新10×10用四张坐标相邻原图，为3km边长/原生300m每格/9km²/B20。坐标轴相同，面积为投影面积；合法移动上限分别3、3、6km，不能称物理预算相同。三张已知训练区域只完成数据工程，未产生新SR。'
    sources=[rel(DATA/'数据清单.json'),rel(QA/'独立复核.json'),rel(ROOT/'选题报告相关/实际区域扩展数据准备方案_v1.md')]
    graphics.save(fig,FIG/'数据结果图','39_实际面积与图块尺度',caption,sources,dict(pixel_size_m=1,native_cell_m=300,new_area_km2=9,previous_area_km2=2.25,new_SR=False))
    fig,axes=plt.subplots(1,3,figsize=(8.4,3.85))
    for ax,r in zip(axes,m['regions']):
        with Image.open(DATA/'previews'/f"{r['area']}.jpg") as im:ax.imshow(im,extent=(0,3,3,0))
        for j in range(1,10):
            ax.axvline(j*.3,color='white',alpha=.27,lw=.45);ax.axhline(j*.3,color='white',alpha=.27,lw=.45)
        ax.axvline(1.5,color='#2ABBE9',linestyle='--',lw=1.3);ax.axhline(1.5,color='#2ABBE9',linestyle='--',lw=1.3)
        ax.set_xticks([0,1.5,3]);ax.set_yticks([0,1.5,3]);ax.set_xlabel('相对东向距离（km）',fontsize=9)
        ax.set_title(r['area']+'：四原图 / 9 km²',fontsize=10,loc='left')
    axes[0].set_ylabel('相对南向距离（km）',fontsize=9)
    fig.subplots_adjust(left=.065,right=.99,top=.86,bottom=.24,wspace=.23)
    fig.text(.5,.095,'每图3000×3000原生像素；白细线为300米网格，蓝虚线为1500米源图边界。',ha='center',fontsize=9)
    fig.text(.5,.025,'训练源文件不复用、原像素不融合；影像为展示缩略图。当前不是未知地理区域或导航成功率验证。',ha='center',fontsize=9)
    caption='三张连续区域的展示缩略图；原始数据为有坐标的3000×3000无损GeoTIFF，原生1m/像素，每格300m。白细线表示10×10网格，蓝虚线表示四个1500源图边界；四图次序由投影坐标确定，不对像素做融合或补洞。十二原源来自已知train，三组不复用源文件且不覆盖原val/test内部足迹，不能据此声称地理独立。影像仅作为展示栅格嵌入，SVG文字/网格可编辑。'
    graphics.save(fig,FIG/'数据结果图','40_连续区域与跨图接缝',caption,[rel(DATA/'数据清单.json'),rel(DATA/'图块来源.json'),rel(QA/'独立复核.json')],dict(regions=3,source_rasters=12,scope='known training engineering',navigation_SR=None))
    for item in graphics.ITEMS:item['reproduction_script']=rel(Path(__file__))
    items=graphics.ITEMS
    write(FIG/'绘图数据/实际区域扩展数据_v1.json',dict(date='2026-10-02',figures=items,script_sha256=digest(Path(__file__)),
        source_sha256={p:digest(ROOT/p) for item in items for p in item['sources']},manifest=m))
    text(FIG/'实际区域扩展图注.md','\n\n'.join('# '+i['id']+'\n\n'+i['caption'] for i in items)+'\n')
    catalog=read(FIG/'图表来源清单.json');assert not {i['id'] for i in items}&{i['id'] for i in catalog['figures']}
    catalog['figures'].extend(items);(FIG/'图表来源清单.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text(FIG/'审阅/实际区域扩展设计_v1.md','图39用同物理坐标轴比较足迹、图块与预算，图40显示真实连续拼接区和原源边界；均明确无新SR。文字9pt以上、建议插入宽度≥18cm，PDF嵌入字体、PNG300dpi。图39矢量；图40嵌入展示栅格，叠加文字/网格为矢量，不能称整幅影像可编辑矢量。另保存实际像素与导出核验。\n')
    (FIG/'审阅/实际区域扩展_作图源码_v1.py').write_bytes(Path(__file__).read_bytes())


def navigation():
    summary='''本地151张Masa GeoTIFF的坐标与1米分辨率已核验。三组连续2×2训练区域已构成3000×3000原生地图，每格300米、单图投影面积9 km²（原单图2.25 km²的4倍）；300格与同源旧5×5图块精确一致。新`masa-local-grid10-area-v1`及75题已冻结，11测试/逐像素复核通过。本轮只做已知训练区域的数据/协议工程，没有导航模型调用、新SR或默认升级。下一项为冻结M0兼容试跑。'''
    header='## 最新阶段：实际搜索足迹的数据与协议工程通过\n\n'+summary+'\n\n[数据准备报告]('+rel(RUN/'实际区域扩展数据准备报告.md')+') / [数据入口]('+rel(RUN/'README.md')+') / [复核]('+rel(QA/'独立复核.json')+') / [下一项草案]('+rel(RUN/'下一项导航兼容试跑草案.md')+') / [科研图](绘图/实际区域扩展图注.md)。\n\n'
    p=ROOT/'README.md';v=p.read_text('utf-8');v=v.replace('## 最新结果：回访新格筛选独立确认',header+'## 最近导航结果：回访新格筛选独立确认',1)
    v=v.replace('下一项优先实际搜索区域扩展的数据准备。','实际搜索足迹的数据与协议工程已通过，下一项为冻结M0兼容试跑。',1)
    v=v.replace('全部61批实验索引','全部62批实验与数据索引').replace('61批实验入口','62批实验与数据入口').replace('38张实验结果图','40张实验与数据协议图')
    v=v.replace('后续优先实际区域扩展数据准备。','后续使用新实际足迹数据单独登记冻结M0兼容试跑。')
    p.write_text(v,encoding='utf-8')
    p=NAV/'当前进度.md';v=p.read_text('utf-8');v=v.replace('后续优先实际区域扩展的数据准备。','实际足迹数据工程现已通过，下一项为冻结M0兼容试跑。',1)
    v+='\n## 实际面积扩展：数据与协议\n\n'+summary+'\n\n[数据报告](../'+rel(RUN/'实际区域扩展数据准备报告.md')+') / [数据工程判定](../'+rel(QA/'验收结论.json')+')。不把数据通过当作实际面积导航或未知地理泛化通过。\n';p.write_text(v,encoding='utf-8')
    idx=read(NAV/'实验索引.json');assert len(idx['batches'])==61
    idx['batches'].append(dict(dataset='Masa',kind='数据准备',name=RUN.name,category='实际面积工程',status='数据/协议通过；导航未启动',
        scope='3已知train连续地图/12源/9km²各图；300格/75题；无新SR',path=rel(RUN),reports=[rel(RUN/'实际区域扩展数据准备报告.md')],
        evidence=[rel(QA/'验收结论.json'),rel(QA/'独立复核.json'),rel(QA/'最终状态.json')]))
    (NAV/'实验索引.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=NAV/'实验索引.md';v=p.read_text('utf-8');v+='| Masa / 数据准备 | 实际区域扩展_v1 | 数据/协议通过；导航未启动 | 3已知train连续地图/12源/9km²每图；[报告](../'+rel(RUN/'实际区域扩展数据准备报告.md')+') / [复核](../'+rel(QA/'独立复核.json')+') / [原目录](../'+rel(RUN)+') |\n';p.write_text(v,encoding='utf-8')
    for n in ('README.md','目录结构与维护.md'):
        p=NAV/n;p.write_text(p.read_text('utf-8').replace('61批实验索引','62批实验与数据索引').replace('61批索引','62批索引'),encoding='utf-8')
    p=FIG/'README.md';v=p.read_text('utf-8');first,rest=v.split('\n',1)
    p.write_text(first+'\n\n## 最新：实际面积扩展数据与协议\n\n'+summary+'\n\n[图39：物理尺度](数据结果图/39_实际面积与图块尺度.pdf) / [图40：连续区域](数据结果图/40_连续区域与跨图接缝.pdf) / [图注](实际区域扩展图注.md) / [数据与SHA](绘图数据/实际区域扩展数据_v1.json)。\n'+rest,encoding='utf-8')
    p=ROOT/'中期报告相关/README.md';v=p.read_text('utf-8');v+='\n实际面积数据工程的新报告和图39—40见[当前进度](../项目导航/当前进度.md)；三已知训练区域尚未跑导航，本轮未回写定稿Word/PDF/PPT。\n';p.write_text(v,encoding='utf-8')


def main():
    assert not (RUN/'实际区域扩展数据准备报告.md').exists(),'immutable data closeout exists'
    assert not (FIG/'数据结果图/39_实际面积与图块尺度.pdf').exists()
    v,a,m,s=required();reports(v,a,m,s);plots(m);navigation()
    print(dict(report=True,figures=2,data_engineering_passed=True,new_SR=False,navigation_model_calls=0),flush=True)


if __name__=='__main__':main()
