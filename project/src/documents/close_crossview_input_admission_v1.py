"""Close CV037 metadata admission; does not acquire pixels or run/train any model."""
import hashlib
import json
from pathlib import Path
from project.src.tools.crossview_supervision_admission_v1 import ROOT,BASE,OUT,DATA,JOURNAL,read,entry,SOURCES


def publish():
    d=read(DATA);a=read(OUT/'23_准入复核_v1.json');assert a['passed']
    for e in d['local_inputs']+d['sources']:assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    z=d['zod'];c=d['city'];mm=d['existing_other_pools']['mmgag'];sw=d['existing_other_pools']['swiss']
    table='|池|元数据/真实查询规模|保守地理组与用途|当前严格拍摄点训练准入|\n|---|---|---|---|\n'
    table+=f"|ZOD已有样例|{z['records_count']}标签，已取得4对像素|{z['groups_count']}元数据组，已消费4组；余5标签只多{z['additional_groups']}组|未准入；现角色是校准/开发/未分配，不自动转训练|\n"
    table+=f"|Mapillary cities样例|{c['record_count']}标签，0像素|{c['groups_count']}个500m/序列元数据组，{len(c['categories'])}城市类别|候选元数据；待图像/投影/K适用性/角色与隔离|\n"
    table+=f"|MM-GAG|现12训练准备查询/9组，另有开发与保留|原6确认组/24素材继续封存|旧像素可读不等于相机点变换已核；严格相机点未准入|\n"
    table+=f"|SwissView Monuments|{sw['metadata_records']}记录仅{sw['unique_ground_photos']}地面图|弱场景/地标代理标签|不作相机拍摄点精确空间监督|\n"
    reps='|候选城市|固定代表|地面/航拍压缩字节|当前状态|\n|---|---|---:|---|\n'
    total=0
    for r in c['representatives']:
        g,rgb=r['ground_member'],r['aerial_member'];size=(g['compressed_bytes']+rgb['compressed_bytes']) if g and rgb else None
        total+=size or 0;reps+=f"|{r['category']}|{g['name'] if g else '路径映射待核'}|{size if size is not None else '未知'}|仅拟像素/投影预检，不分配训练|\n"
    text=f'''# CV-037：普通照片输入与空间监督数据准入

**有更匹配的监督来源候选，但当前仍未获严格拍摄点训练准入。** 本轮新增的是可执行输入/准入契约和小标签元数据核查，不训练、不换大模型、不重新试CCVPE。不能把“可编写训练程序”和“有资格证明目标定位效果”混为一谈。

## 1. 精确定义主任务

给定普通地面视角照片，目标是数据集相机拍摄位置对应的地理格/局部点位；可见地标位置不能直接当相机位置。主模型输入仅目标RGB、当前已观测航拍RGB、公开位置/网格/步数/预算/动作；历史由会话内部真实观测形成。目标GPS、heading、真实x/y、源身份、未知父图/候选库/外部历史只可用于离线标签/损失/事后评测，不进入策略。

对真正窗外/异源样本，可以监督“不匹配/无可靠位置”；不能强迫RGB没有证据时输出一个真实方向。窗内定位能力也不自动提高first-arrival导航SR，完整闭环还需要在未得到目标证据时合法探索的策略。

## 2. 两种输入规范，不能暗中混用

- **image_only**：普通RGB目标，无K或朝向/GPS前提。候选表征与监督任务必须支持这种输入，不继承KITTI固定相机FoV或全景假设后声称支持任意手机照片。
- **calibrated_photo**：只在相机K能追溯到实际目标像素、尺寸吻合时才允许。3×3矩阵有效不代表校准适用；resize/crop/letterbox必须同步作K'=A K，fx/fy随缩放，主点随缩放及偏移。原始畸变/裁剪/相机成像链仍需来源证据。

新data/crossview_input_admission_v1.py提供严格字段/RGBFrame/公开状态、K几何与角色/准入检查；6项纯合成检查通过。K可选不等于没有K就不能做RGB-only监督，也不能用目标真实heading来补未知朝向。本轮没有训练新的投影/位置头。

## 3. 监督池真实规模与边界

{table}
ZOD的drives和sequences各3个邻帧，邻近相机间距约1.8—1.9m，不能拆作独立训练/开发来源。连通分组是保守的500m或同序列元数据规则，不能替代像素重叠/精确测绘证明；余5条不是5个新地域。4已消费来源的角色保持原样。

Mapillary城市新增只读标签来自官方样例的cities/SE/stockholm、cities/PL/krakow、cities/NL/amsterdam三个标签成员。字段结构完整的空间标签{c['valid_geometry_records']}/{c['record_count']}，GPS位移与dx/dy×resolution差≤1m为{c['coordinate_consistent_records']}/{c['record_count']}；这是坐标约定相容，不是独立测绘精度证明。K缺失/尺寸/投影类型仍按各行原样标记，未取图像前不假装已核普通透视照片或排除全景。

只从固定官方ZIP按精确206范围读三城市标签，标签正文网络读取{d['public_metadata']['label_network_body_bytes']}字节≤32KiB，逐成员长度/CRC/SHA核验，0图像字节/不保留ZIP；未打开Mapillary/in_the_wild_test的标签或图像。该上游test名称不能变成随意训练材料。原MM-GAG6确认像素不读，用途账本不改。

本地MM-GAG相机GPS字段存在，但旧审查确认GPS到像素变换/独立相机点均0已核；不能把中心代理的12/9组或先前失败小头训练，换个名称就当新的精确空间训练。SwissView的375条重复配对也不能按375张独立照片计量。

## 4. 本轮准入决定

不设置拍脑袋的“来源数必须20/100”通行证。准入逐项检查：拍摄点标签语义、几何相容、图像/坐标预检、明确的源角色/隔离、实际普通照片输入规范。Mapillary目前缺像素/投影与角色，现ZOD没有新训练角色授权；因此没有可直接启动的新训练协议。允许未来有界工程拟合，并不意味着其泛化或论文有效性得到承诺。

下一轻量路线优先考虑**有空间位置/不匹配监督的局部关系模块**，而非重训旧全局8192头或继续冻结CCVPE的tau/padding扫描。骨干与参数量尚未选定；先证明监督与公开输入兼容、真实/均值/禁用/错误目标能形成合理对照，再冻结结构/损失/训练资源/停止条件。不立即安装或下载另一个模型。

## 5. 唯一下一步：最多三个城市代表的像素工程预检

按各城市原ground_image字典序固定一个代表（未看像素/成绩），不擅自改为训练或确认。本轮选定的是待审获取清单，不是图像已取得或角色已授权：

{reps}
对应压缩像素载荷合计{total}字节（仅中心目录声明），以后可提案总网络≤16MiB、保存≤20MiB、3对6图，仍0模型/训练；需先明确新目录/获取范围授权。检查照片投影/质量、尺寸、相机K适用性、北向/点位和完整像素SHA，失败保留不换样例。若元数据路径不能精确映射，则先修来源解释，不去猜路径下载。

像素与投影通过后才讨论训练/开发/保留分工；每组不可跨角色，邻帧/共用父图不可人为扩样本数。原已消费4组仍只称已知工程/开发。任何训练、源用途变更或新确认必须另明确授权及冻结，不借本轮准入自动开训。

## 6. 来源与复核

官方固定提交[README](https://github.com/Open-CVL/opencvl/blob/{'4838d1627cf28b28e81d2dd830995fe1fd3a1e2c'}/README.md)及[NOTICE](https://github.com/Open-CVL/opencvl/blob/{'4838d1627cf28b28e81d2dd830995fe1fd3a1e2c'}/NOTICE.md)的SHA/字节记录在JSON；各数据源适用自己的条款，公开样例不自动证明完整数据与训练/再分发已获许可。本轮不注册或联系作者，不替用户接受条款。

[全部元数据/输入与源码绑定](23_普通照片输入与空间监督准入_v1.json) · [另实现复核](23_准入复核_v1.json) · [逐请求范围回执](23_公开元数据取得回执_v1.jsonl)。另实现使用单位球弦长距离与图遍历重计空间连通组，核角色/分母/test保护/预算，同Agent另一实现非独立人员。不会重新解码原图或调用模型。

既有平台4.12和原M0/S2S3S4可继续使用；地面自主导航缺口保留。本轮0真实图像/权重读取、编码预测/训练/导航/模型API或6确认消费，无新目录/Git提交推送/主动C盘任务文件。官方网页工具取规范失败后经现有7897代理读取官方文本与小标签，无镜像/安装/持久配置改动。
'''
    md=OUT/'23_普通照片输入与空间监督准入_v1.md'
    with md.open('x',encoding='utf-8') as f:f.write(text)
    # Metadata artifacts are exclusive-created; publication adds only mutable navigation/state.
    report=md.relative_to(ROOT).as_posix();js=DATA.relative_to(ROOT).as_posix();au=(OUT/'23_准入复核_v1.json').relative_to(ROOT).as_posix()
    p=ROOT/'项目导航/实验索引.json';old=p.read_bytes();idx=read(p);prior=idx['batches'];assert len(prior)==163
    idx['date']='2026-10-24';idx['batches']=prior+[dict(category='CV-037',dataset='ZOD/Mapillary cities元数据+旧MM-GAG/Swiss角色',kind='普通照片输入/空间监督准入',name='普通照片输入与空间监督准入_v1',status='新增城市元数据候选；当前不准入新训练',path='调研',reports=[report],evidence=[js,au])]
    assert p.read_bytes()==old;p.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');assert read(p)['batches'][:-1]==prior
    p=ROOT/'docs/维护/版本生命周期.json';old=p.read_bytes();life=read(p);current=life['current'];life['updated']='2026-10-24'
    life['cv037_input_supervision_admission']=dict(status='metadata_candidate_only_not_training',report=report,evidence=js,audit=au,
        replaces='CV036 pending input/data admission only;no old result/model/default replaced',recovery='read metadata and acquisition journal;do not repeat remote ranges blindly',
        training_admitted=False,next='3 fixed city representatives pixel/preprocessing precheck needs explicit scope',artifacts=[entry(md),entry(DATA),entry(OUT/'23_准入复核_v1.json'),entry(JOURNAL)])
    assert p.read_bytes()==old;p.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');assert read(p)['current']==current
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;old=p.read_bytes();s=p.read_text(encoding='utf-8-sig');h,_,rest=s.partition('\n');prefix='' if name=='README.md' else '../'
        link=f'[CV-037 · 普通照片输入与监督准入]({prefix}{report})：ZOD9标签仅5组；城市样例{c['record_count']}标签/{c['groups_count']}组候选，训练未准入。'
        assert p.read_bytes()==old;p.write_text(h+'\n\n'+link+'\n\n'+rest,encoding='utf-8')
    p=ROOT/'docs/PROJECT_STATE.md';old=p.read_bytes();s=p.read_text(encoding='utf-8-sig');h,_,rest=s.partition('\n')
    state=f'''\n\n### CV-037：普通照片输入/空间监督准入完成，城市候选待像素工程（2026-10-24）

[完整契约与清单](../{report})。新核心区分image-only与校准照片/K像素坐标链，不把目标GPS/heading/真实位置或外部历史给策略。ZOD9标签仅5个500m/同序列组，4已消费组之外剩5条只多1组；MM-GAG12/9组仍缺已核严格相机点变换，Swiss375条仅15照片且为弱代理。未自动改旧角色或准入训练。

从官方ZIP精确206读取3 cities标签，{c['record_count']}行/{c['groups_count']}空间元数据组（{c['coordinate_consistent_records']}行GPS/offset≤1m），未读上游in_the_wild_test/任何新图像。下一拟3固定城市代表0模型像素/投影/校准预检，先明确目录/获取授权，不直接训旧头或换大模型。6合成检查/另实现通过；164索引且旧163值保持，平台4.12/旧默认/6确认不变。网络仅规范/标签，用7897代理，无模型训练/Git/新目录/C盘输出。\n'''
    assert p.read_bytes()==old;p.write_text(h+state+'\n'+rest,encoding='utf-8')
    names=[md,DATA,OUT/'23_准入复核_v1.json',JOURNAL];size=sum(p.stat().st_size for p in names);assert size<1024**2
    print(json.dumps(dict(published=True,index=164,artifacts=4,bytes=size,training_admitted=False,raw_images=0),ensure_ascii=False))


if __name__=='__main__':publish()
