"""Publish CV039 frozen design only; never launch engineering or training."""
import json
from project.src.data.rgb_spatial_supervision_protocol_v1 import ROOT,BASE,OUT,DATA,entry,read


def update(p,old,text):
    assert p.read_bytes()==old,str(p)+' changed during publication'
    p.write_bytes(text.encode('utf-8'))


def publish():
    md=OUT/'24_普通RGB空间监督与不匹配拒绝协议_v1.md'
    if md.exists():raise ValueError('Already published; inspect, do not overwrite or re-freeze')
    d=read(DATA);audit_path=OUT/'24_协议复核_v1.json';a=read(audit_path)
    assert a['passed'] and read(OUT/'24_合成验证_v1.json')['passed']
    for e in d['inputs']+d['sources']+d['encoder']['asset_identity']+[d['synthetic_validation'],d['protected']['old_role_ledger']]+d['protected']['runtime']:
        assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    assert read(ROOT/'项目导航/实验索引.json')['batches']==d['protected']['index_prefix']
    assert read(ROOT/'docs/维护/版本生命周期.json')['current']==d['protected']['current']
    rows=''
    for f in d['folds']:
        rows+=f"|{f['fold']}|{'、'.join(f['fit_sources'])}|{f['unrelated_reference']}|{f['wrong_target_donor']}|\n"
    source_rows=''
    for name,s in d['source_table'].items():
        source_rows+=f"|{name}|{s['width']}×{s['height']}|{s['mpp']:.9f}|{s['metric_extent_m'][0]:.5f}|{s['metric_extent_m'][0]/5:.5f}|\n"
    text=f'''# CV-039 普通RGB空间监督与不匹配拒绝协议

**本轮冻结一项小型开发验证，不运行模型或训练。** 唯一候选是冻结现有CLIP图像骨干、训练65,537参数的空间关系/不匹配头。与同预算训练的无目标头、同特征的冻结余弦比较；先证明目标图改善局部定位和拒绝，再考虑导航。不是完整论文复现或新区域确认，不替换原M0。

## 1. 假设与边界

旧8192参数全局目标变换未提供可靠的精确拍摄点信号；旧冻结CCVPE普通照片直接迁移也已收束。本轮不继续这两条路线的阈值/裁剪/填边扫描。新假设是：保留当前图像的7×7空间token，用真实数据集相机点与不匹配监督，能比原始余弦及同预算无目标拟合更可靠。

与本轮各基线共用骨干、预处理、任务清单和输入权限，比较的是小头空间监督收益；不能拿它与旧CV036的不同骨干、窗口或队列直接相减当作单因素消融。现素材全部是已查看的车载前视照片，不能外推任意手机照片、全景、真实无人机或独立地理泛化。

## 2. 数据、尺度与标签

只使用CV038已有3对/6图，不取新数据、不读MM-GAG6确认或Mapillary上游test。原输入/角色/封存保持，本协议只冻结后续开发批次的局部折分工；没有改写旧用途账本，也未执行角色迁移或训练。

|城市|原生航片|米/像素（原标签）|覆盖宽度/米|5格划分每格宽度/米|
|---|---|---:|---:|---:|
{source_rows}
各航片按原生像素5×5半开区间划分，边界floor(iW/5)，不把Stockholm强行补成100米。主标签统一用已存在、available=True的OpenSfM修正相机点：x=W/2+dx，y=H/2−dy。统一选择来自修正位姿这一来源语义，不依据接触表道路外观或之后成绩逐样本选标签。raw仅作完整敏感性统计，不选择checkpoint、阈值、来源或门槛；两套原值均保留。

raw/修正点差约24.55/20.26/13.86米，说明标签来源差异可能影响任务。5米为**数据集修正标签的定位容差**，不称独立测绘精度。2.5米高斯是固定标签平滑核，不是测得的相机位置不确定性；独立真值/成像链仍未知。不得把可见地标当拍摄点或把点夹到格子中心/边界追分。

## 3. 三折已知来源开发分工

|本折留出查询及其配对航片|拟合来源|开发异源参考（已见）|Wrong目标供体|
|---|---|---|---|
{rows}
每折优化器只用另外两城，Mean只由这两城的目标特征算，不能用留出照片求均值。每折拟合清单100题：两查询×两参考×25格，每查询1窗内、24窗外、25异源；开发50题：留出查询×（本城参考＋固定拟合城参考）×25格，类别1/24/25。

**限制必须保留：** 三城早已工程查看，三折都属于开发；每折异源参考来自已见拟合来源，不能声称异源查询与参考均完全未见。三个种子/三折/重复格不增加独立地域；地理来源3，独立确认0。本轮不能补齐“可靠学习基线＋全新来源共同确认”的全部论文证据。

Wrong固定为另一拟合城，既不同于留出查询，也不同于开发异源参考；不让Wrong照片意外与异源参考成真实配对。Wrong条件仍按原Full任务标签评测，不改变目标真值。

## 4. 合法输入与冻结特征

匹配头严格只接`given_target_feature[512]`和`current_patch_features[49,512]`。编码器只接给定地面RGB与一个真实当前航拍格RGB；不得输入K、GPS、heading、相机点、目标格、源/任务身份、裁剪原点、未知父图/特征库或外部历史。位置/分辨率/源身份仅供数据切图、分工、标签与评测，不能旁路给头。

复用本地`models/clip_vit_b32_public_v1/`的唯一已存权重与配置，不下载模型。仅实例化图像塔，地面/航拍共享；不使用文本塔、Sat2Cap或CCVPE。图像塔预期87,849,216参数（本轮按配置另算，实际张量/完整参数计数在工程阶段核实），全部eval/冻结，0骨干训练参数。

地面：末层CLS经post-LN、既有visual_projection、L2，得512维。航拍：末层49个patch token各经同post-LN和visual_projection、L2，保留7×7顺序。把CLS投影用于patch是本候选工程选择，不能声称原CLIP已训练成像素定位器；真实输出形状/冻结身份/重复一致性未验证前不启动训练。

地面整图保留：双线性缩放长边到224，尺寸floor(x+0.5)，CLIP均值居中填边，余像素放下/右；952×532变224×125，上49/下50像素填边。航拍当前格直接双线性缩224²，保留完整格，不编码全父图或稀疏父画布。按固定CLIP均值/标准差归一化，不做朝向/FoV/K补偿、裁剪/旋转/尺度扫描。

## 5. 小头、损失与两种基线

两个矩阵Wg、Wa均512×64，另一个无匹配标量b，共2×512×64+1=**65,537可训练参数**；无可训练位置偏置或源身份通道。对目标g与当前patch a_j：

q=L2(g Wg)，k_j=L2(a_j Wa)，z_j=10·qᵀk_j，z_empty=b。

softmax共49＋1类。窗内软标签按各patch中心到修正相机点的实际米制距离，以sigma=2.5m高斯归一化；窗外/异源为空类one-hot。训练50类交叉熵。位置取49空间logit最大者（同分最小序号），映射回本格原生像素；接受分数为49空间概率之和。无接受也保留位置/误差诊断。

|方法|训练与输入|作用|
|---|---|---|
|Spatial|真实目标，65,537参数|唯一新候选|
|TrainedMean|相同参数/初始化/采样/更新，只给固定拟合目标均值|同预算强无目标拟合，检验道路/位置先验|
|RawCos|相同冻结512维特征的原始余弦、温度10、b=ln49，不训练|冻结语义参照；训练预算差异明确披露|

Spatial和TrainedMean的Wg/Wa初始取同一随机矩阵，std=1/√512，b=ln49，种子0/1/2；同折同种子的两方法所有初始张量一致。每格不传真实坐标给头。

## 6. 目标对照与校准

Spatial与RawCos均有Full/Disabled/Mean/Wrong：Disabled在目标路径前替换零向量，Mean为本折两个拟合目标的归一化均值，Wrong供体见表；当前patch完全不变。唯一目标路径是g→Wg→q，没有其它目标残差或匹配层；工程阶段须证明替换真实生效。TrainedMean是另外训练的头，与Spatial在推理时临时Mean控制分开。

每个训练模型最终状态只在同一100拟合题上校准一次；RawCos每折一次，种子间原样复用。目标为每查询三类宏平均：窗内接受且≤5m、窗外拒绝、异源拒绝。候选为各distinct score与nextafter(max,+inf)，用计数有理数比较宏目标，精确同分取较高tau；float64比较score≥tau。所有目标控制复用对应Full的tau；不得开发校准/回cast阈值或按开发成绩选epoch。使用拟合数据校准的偏差需披露，不称独立验证。

## 7. 固定训练与执行总账（尚未启动）

每折每种子两个头，合计18模型。Adam lr0.001、betas0.9/0.999、eps1e-8、weight_decay0、float32、梯度范数裁剪1；每模型固定200更新、每批6对（每个拟合查询各1窗内＋1窗外＋1异源）。固定按种子打乱循环顺序，两个方法复用同一采样清单SHA，不按成绩改采样。

|项目|固定预算|
|---|---:|
|正式训练更新|18×200＝3600|
|正式训练样本使用|3600×6＝21600|
|18头初始/最终100题拟合检查|3600对预测|
|RawCos拟合校准|300对预测|
|Spatial四目标开发|1800对预测|
|TrainedMean开发|450对预测|
|RawCos四目标开发（不重复算种子）|600对预测|
|开发总量|2850对预测|
|拟合检查/校准＋开发|6750对预测|

模型前向次数与样本对使用是不同计数，实际执行器两者都记录；这些不是导航轨迹数。无中间checkpoint选优，仅最终头；初始状态由冻结种子/实现复原，首更新前保存初始身份。

独特编码输入78个：3目标＋75当前格。下一工程最多8次编码，包含3目标/3首格的6独特输入＋2精确重复；全部工程通过才能复用6份有效特征，再执行剩72次，**编码累计最多80次**，不额外热身/重试。工程另最多8次小头更新×6对、32对控制/恢复检查，正式18模型都重新初始化，不能继承工程拟合状态。特征float32净7,532,544字节，由数据/evaluator持有，不能将缓存库传给匹配头。

CPU执行，不占GPU或停止ComfyUI；RSS≤4GiB，工程≤180秒，正式累计≤1800秒。后续新批次总存储≤48MiB，特征缓存≤12MiB，保留18个最终小头/指标/轨迹式记录与一个活动optimizer恢复状态，不复制冻结骨干、不堆里程碑/完整优化器。当前仅冻结预算，不代表这些目录或真实执行已获授权。

## 8. 预登记必要门槛与停止条件

- 工程：精确资产/参数、形状、骨干冻结、重复输出、各目标路径、梯度、恢复等价、输入隔离、资源与另算全部通过。
- 拟合：9个Spatial运行最终同100题拟合损失均低于初始，最终拟合宏联合BA各≥0.80；不能只看接受率。
- 开发定位：Full在9个来源×种子正例中接受且≤5m至少6个，每城至少1/3；每城×种子窗外误接受≤2/24、异源≤2/25。
- 目标有效性：Full相对每个Disabled/Mean/Wrong，来源宏联合BA至少＋5个百分点，正例净恢复至少2/9；恢复与损伤都报告。
- 基线：Full相对TrainedMean、RawCos Full分别满足同样＋5点/净恢复≥2要求。RawCos只计算一次的预测与种子配对复用，不伪造额外独立观察。
- 完整性：所有计划分母、来源/种子/类别/米制误差、拒绝前全部正例误差、恢复/损伤、输入/权重/特征/源码绑定、资源及另实现全过。

按三城市聚类、来源内先合并种子，配对bootstrap10000次/seed20261024区间仅描述；样本太少，不以正下界或伪重复p值声称稳定泛化。主门槛只判这次已知来源开发候选，不能升级默认。

只做一次固定方案。数值、冻结、权限或资源失败立即保留前缀停机，不自动重试；正常完成但任一效果门槛未过，收束本候选，不扫描tau/crop/seed/来源/epoch/参数继续追分。即使通过，下一确认也需真正新且隔离的空间来源另冻协议；当前3城不能充当那个确认池。

## 9. 当前交付与恢复

[完整机器协议/源用途/任务/资产及源码绑定](24_普通RGB空间监督与不匹配拒绝协议_v1.json) · [9项合成检查](24_合成验证_v1.json) · [另实现复核](24_协议复核_v1.json)。合成验证覆盖半开边界/尺度、50类标签、稳定概率、特征权限、折分/采样、校准来源与同分、定位错误不能冒充成功；另实现用有理数分格、循环序列重算和独立参数/计数核验。由同Agent完成另一实现，非独立人员审查。

权重文件身份已按SHA绑定，但没有加载模型/提取特征，实际张量参数/特征与执行器尚未绑定，不能直接开训。下一CV-040实现普通RGB空间特征与小头执行器，先做上述8编码/梯度/控制/恢复工程；目录与真实资源范围需先明确。本批0解码/模型加载forward/训练/导航/下载/API或6确认消费，无新SR/感知成绩，也未改旧角色。

平台4.12/server19/service11/pack23及原M0/S2S3S4保持，现有场景/地面人工观察与证据页照常；本轮无GUI升级。当前166索引且旧165值/current保持，旧CV038输入/报告/源码均原址原SHA。恢复只读本协议/JSON/验证/文件绑定，不能重跑本批写盘冻结/审计/发布覆盖。无新目录、Git提交推送、C盘任务输出；全程离线，未使用代理。
'''
    with md.open('x',encoding='utf-8') as f:f.write(text)
    report=md.relative_to(ROOT).as_posix();js=DATA.relative_to(ROOT).as_posix();au=audit_path.relative_to(ROOT).as_posix()
    p=ROOT/'项目导航/实验索引.json';old=p.read_bytes();idx=read(p);assert len(idx['batches'])==165
    idx['date']='2026-10-24';idx['batches'].append(dict(category='CV-039',dataset='OpenCVL三城市已消费开发来源',kind='普通RGB空间监督与拒绝协议冻结',
        name='普通RGB空间监督与不匹配拒绝协议_v1',status='设计冻结，实际执行器/特征/工程未准入，未训练',path='调研',reports=[report],evidence=[js,au]))
    update(p,old,json.dumps(idx,ensure_ascii=False,indent=2)+'\n')
    p=ROOT/'docs/维护/版本生命周期.json';old=p.read_bytes();life=read(p);life['updated']='2026-10-24'
    life['cv039_rgb_spatial_protocol']=dict(status='design_frozen_not_execution_admitted',report=report,evidence=js,audit=au,
        replaces='CV038 pending protocol only; no old model/default/data role replaced',
        dependencies=['DATA/processed_data/OpenCVL/三城市照片与空间标签预检_v1/阶段封存.json','models/clip_vit_b32_public_v1/pytorch_model.bin'],
        recovery='read frozen JSON and bindings; no automatic real run or overwrite',training_authorized=False,
        next='CV040 actual executor + 8-encoding engineering, requires concrete directory/resources scope',
        artifacts=[entry(p) for p in (md,DATA,audit_path,OUT/'24_合成验证_v1.json')])
    update(p,old,json.dumps(life,ensure_ascii=False,indent=2)+'\n')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;old=p.read_bytes();s=old.decode('utf-8-sig').replace('\r','');h,_,rest=s.partition('\n');pre='' if name=='README.md' else '../'
        link=f'[CV-039 · 普通RGB空间监督与拒绝协议]({pre}{report})：冻结65,537参数小头/同预算无目标对照与三城市开发分工，尚未运行模型或训练。'
        update(p,old,h+'\n\n'+link+'\n\n'+rest)
    p=ROOT/'docs/PROJECT_STATE.md';old=p.read_bytes();s=old.decode('utf-8-sig').replace('\r','');h,_,rest=s.partition('\n')
    note=f'''\n\n### CV-039：普通RGB空间监督与拒绝设计冻结，真实执行未准入（2026-10-24）

[协议、公式、来源分工、预算与门槛](../{report})。固定现有CLIP图像塔共享冻结（预期87,849,216参数，未实际加载核算），65,537参数空间头输出7×7位置+不匹配；主标签OpenSfM、raw敏感性、逐源真实尺度与image-only权限固定，无GPS/heading/K/未知父图给模型。三城市轮流留出只是已消费开发；异源参考来自拟合来源的限制保留，独立确认0。

计划18个头、3600更新/21600样本使用，同预算TrainedMean与RawCos基线，Full/Disabled/Mean/Wrong、fit-only一次校准、6750次非训练对预测；先8编码工程（含6unique/2repeat），通过后剩72，累计80，不含自动执行授权。9合成/{len(a['checks'])}另实现通过，实际0模型/训练/解码/下载/导航，无新SR或感知准确率。下一CV040先实现/绑定执行器并核梯度/恢复/冻结/真实控制，目录资源另明确；未过单次候选即收束不追分。166索引且旧165/current/旧角色/6确认/平台4.12与原M0保持；离线无代理/新目录/C盘输出/Git。\n'''
    update(p,old,h+note+'\n'+rest)
    files=[p for p in OUT.glob('24_*') if p.is_file()];assert sum(p.stat().st_size for p in files)<1024**2-16384
    with (OUT/'24_文件绑定_v1.json').open('x',encoding='utf-8') as f:
        json.dump(dict(files=[entry(p) for p in files],sources=d['sources'],seal_itself_excluded=True,design_only=True,training_authorized=False),f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(published=True,index=166,source_cities=3,head_parameters=65537,actual_forward=0,
                          bytes=sum(p.stat().st_size for p in OUT.glob('24_*') if p.is_file())),ensure_ascii=False))


if __name__=='__main__':publish()
