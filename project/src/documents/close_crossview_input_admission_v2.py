"""One-time CV037 report/navigation publication, without changing old artifacts."""
import json
from project.src.tools.crossview_supervision_admission_v2 import ROOT, OUT, DATA, entry, read


def checked_write(path, old, text):
    assert path.read_bytes()==old, str(path)+' changed during publication'
    path.write_text(text,encoding='utf-8')


def publish():
    md = OUT/'23_普通照片输入与空间监督准入_v2.md'
    if md.exists(): raise ValueError('Already published: inspect state, never overwrite/repeat')
    d = read(DATA); audit_path = OUT/'23_准入复核_v2.json'; a = read(audit_path)
    assert a['passed']
    for e in d['sources']+d['local_inputs']+d['official_schema_receipts']+[d['synthetic_tests'],d['original_acquisition_journal']]+d['protected']['runtime_files']:
        assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    tests = read(ROOT/d['synthetic_tests']['path']); assert tests['passed'] and tests['tests_run']==12
    assert read(ROOT/'docs/维护/版本生命周期.json')['current']==d['protected']['current']
    assert read(ROOT/'项目导航/实验索引.json')['batches']==d['protected']['prior_index']
    c = d['city']; table = ''; payload = 0
    for r in c['representatives']:
        g, s = r['ground_member'],r['aerial_member']; payload += g['compressed_bytes']+s['compressed_bytes']
        table += f"|{r['category'].split('/')[-1]}|`{g['name'].rsplit('/',1)[-1]}`|{g['compressed_bytes']+s['compressed_bytes']}|\n"
    rev = '4838d1627cf28b28e81d2dd830995fe1fd3a1e2c'
    content = f'''# CV-037 普通照片输入规范与空间监督数据准入

**本轮找到更匹配的空间监督候选，但尚不准入训练。** 已交付可执行的输入契约、明确位姿字段的适配器、元数据清单、12项合成检查与另实现复核。未取新图/权重、未调用模型或导航，不能产生新定位成绩或SR。

## 主任务与输入边界

给定普通地面照片，逐步关联当前/历史已观测的航拍画面，寻找数据集相机拍摄位置对应的格。可见地标位置与拍摄点不同；标签可靠也不等于图像中一定有足够证据推断拍摄点。窗外/不相关目标应允许无可靠匹配，不能强迫输出真值方向。

|规范|前提|模型可接收|尚不能假定|
|---|---|---|---|
|image_only|两幅合法、尺寸受限的RGB|地面目标、当前航拍、公开格位/网格/步数/剩余预算/合法动作；会话内积累真实历史|相机K、真实朝向/GPS、KITTI固定FoV、全景覆盖|
|calibrated_photo|K可追溯到实际目标像素，尺寸和成像链已核|上述信息及匹配该像素坐标的K|仅凭3×3矩阵就认为校准可用|

禁止目标GPS、heading、真实位置/距离/目标格、源身份、未知父图/未访问图块特征及外部历史进入策略。真值只用于标签、损失、采样与事后评测。resize/crop/letterbox需要同步变换K'=AK；照片投影、畸变与原始校准链另核。输入规范不是导航策略，也不自动开放完整参考图。

## 各数据池的真实容量

|来源|已核实规模|当前可用性与边界|
|---|---|---|
|ZOD本地样例|9标签，500m或同序列连通仅5组；已取得4对覆盖4组，剩5条只新增1组|原0/1校准、2/3已知开发不变，其余未分配；邻帧不能当独立地区|
|Mapillary cities官方样例|新读3标签成员共9行；raw与修正坐标均为9个500m元数据组，来自3城市|候选标签，0像素；不把元数据组直接称独立影像/泛化测试地区|
|MM-GAG|已有12训练准备查询/9组|现任务仍缺已核相机点到航拍像素变换，旧小头失败证据保留；不据此否定整个原数据集|
|SwissView Monuments|375记录仅15张不同地面照片|当前弱场景/地标代理，不能改称精确相机点监督|

地理组只为保守元数据统计，不是独立测绘或像素重叠验证。MM-GAG 6确认组/24素材保持封存；本轮未改用途账本，也不读取其像素。上游Mapillary cities归训练、in_the_wild_test归测试是官方划分；上游train名称不等于本项目自动训练授权。

## Mapillary适配与标签含义

该样例使用`mapil_image`，原GPS别名`ground_latlon`等于`ground_latlon_mapilraw`；还提供独立的`ground_latlon_mapilopensfm`、对应的dx/dy/heading及可用标志。新适配器要求显式选raw或opensfm，无自动回退；GPS、dx/dy、heading必须来自同一命名空间。两套都保存，后续协议需预先指定，不能看到模型分数后选标签。

这9行都无`aerial_latlon`，不虚构中心GPS。数据集像素约定为x=W/2+dx、y=H/2−dy。利用同一航拍参考的两位姿差，检查GPS差与(dx差,dy差)×resolution，9/9内部残差≤1m，最大{c['maximum_internal_residual_m']:.4f}m；另实现用三维单位球局部东/北投影重算通过。**这是标签字段内部一致性，不是独立测绘精度、绝对坐标验收或模型能力。** 图像边界/尺寸、投影类型和K像素链仍未读图验证。

首次v1元数据结果因沿用ZOD字段要求把城市行记为未适配，保留其JSON/源码/请求日志，不能解读为标签坏数据。本轮v2只离线重用已取得9行，未重新请求标签；v1发布器未执行，当前阅读入口是本v2。

## 准入决定与轻量路线

当前不准入新训练、模型候选或自主导航。缺项是：真实像素/投影/质量、标签落点在航片内、K与实际像素适配、重叠/隔离及明确来源角色；不存在拍脑袋的“必须20或100来源”门槛。小样本可以验证工程拟合，但不能当独立泛化证据。

优先保留**普通照片兼容的空间位置＋不匹配监督**路线：只有已观测区域产生可靠局部证据时才介入探索。先验证数据，再决定冻结骨干与小型空间关系模块的结构/损失/预算。不复活旧8192全局头，不继续当前冻结CCVPE阈值、填边或视窗扫描，不新增更大骨干。

## 唯一下一步：CV-038三城市代表像素预检（计划，未授权执行）

在看任何像素/模型分数前，按每城市ground路径字典序固定一个代表：

|城市|图像文件名|地面＋航拍压缩载荷字节|
|---|---|---:|
{table}
合计{payload}压缩字节，仅来自旧ZIP中心目录声明。完整成员路径、偏移/CRC/尺寸与raw/opensfm标签在JSON中。拟下一批最多3对/6图，网络≤16MiB、保存≤20MiB；只做像素/照片投影、尺寸/K、北向与标签落点/质量检查，0模型、训练、导航。新目录与获取范围按规则九另行确认后才能执行。失败保留，不换更有利样例；只有预检通过再规划角色分工与训练协议，原已消费来源仍不能称未见确认。

## 来源、可追溯与验证

官方固定提交：[数据规范与划分](https://github.com/Open-CVL/opencvl/blob/{rev}/README.md)、[路径解析](https://github.com/Open-CVL/opencvl/blob/{rev}/opencvl_tools/dataset.py)、[各来源许可](https://github.com/Open-CVL/opencvl/blob/{rev}/NOTICE.md)。SDK路径解析兼容ground_image/mapil_image；本项目显式位姿选择不是声称官方默认选择。各来源条款各自适用，公开样例不自动授予任意训练/再分发许可；本轮不注册、联系作者或运行修正管线。

[元数据、3代表及源码绑定](23_普通照片输入与空间监督准入_v2.json) · [另实现复核](23_准入复核_v2.json) · [12项合成检查](23_合成契约验证_v2.json) · [取得回执](23_公开元数据取得回执_v1.jsonl)。测试覆盖字段混用/缺失修正位姿拒绝、偏移符号、有限数、K像素链、禁用隐藏信息、地理连通及非来源数量门槛；复核为同一Agent另一实现，不是独立人员审查。

原CV036封存{d['protected']['seal_files_verified']}文件/源码重核，旧163索引值、当前平台4.12/server19/service11/pack23与原M0/旧S2S3S4保持；本轮没有GUI/服务升级。平台现有场景、地面人工连续观测与证据查阅仍可使用。

官方网页工具取文失败后用现有7897代理读取官方规范与3小标签（206正文共2414字节，≤32KiB）；官方树API另返回403限额后改读同官方raw，不用镜像、不保留ZIP、不取图像。0原图/权重/模型调用、训练、导航、模型API或6确认消费；无新目录、Git提交推送、安装/持久配置变更或主动C盘任务文件。
'''
    with md.open('x',encoding='utf-8') as f: f.write(content)
    report = md.relative_to(ROOT).as_posix(); js = DATA.relative_to(ROOT).as_posix(); au = audit_path.relative_to(ROOT).as_posix()
    p=ROOT/'项目导航/实验索引.json'; old=p.read_bytes(); idx=read(p)
    idx['date']='2026-10-24'; idx['batches'].append(dict(category='CV-037',dataset='ZOD / Mapillary cities元数据 / 旧MM-GAG与Swiss角色',
        kind='普通照片输入规范与空间监督准入',name='普通照片输入与空间监督准入_v2',status='元数据与契约完成；三城市像素预检待授权，训练未准入',path='调研',reports=[report],evidence=[js,au]))
    checked_write(p,old,json.dumps(idx,ensure_ascii=False,indent=2)+'\n')
    p=ROOT/'docs/维护/版本生命周期.json'; old=p.read_bytes(); life=read(p); life['updated']='2026-10-24'
    life['cv037_input_supervision_admission']=dict(status='metadata_candidate_only_not_training',report=report,evidence=js,audit=au,
        replaces='CV036 pending admission only; v1 is immutable acquisition/schema-unsupported evidence, not invalid data',
        prior_artifact_role='v1 source and JSON retained in place; unpublished v1 report generator must not run',
        recovery='Read v2 report/JSON/audit and acquisition journal; do not rerun collectors or publisher',
        current_model_or_platform_replaced=False,training_admitted=False,next='CV038 bounded 3-city pixel precheck needs directory/acquisition permission',
        artifacts=[entry(p) for p in (md,DATA,audit_path,OUT/'23_合成契约验证_v2.json')])
    checked_write(p,old,json.dumps(life,ensure_ascii=False,indent=2)+'\n')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name; old=p.read_bytes(); s=old.decode('utf-8-sig'); h,_,rest=s.partition('\n')
        prefix='' if name=='README.md' else '../'
        link=f'[CV-037 · 普通照片输入与监督准入]({prefix}{report})：ZOD9标签/5组、城市9行/9元数据组；先做3城市像素预检，训练未准入。'
        checked_write(p,old,h+'\n\n'+link+'\n\n'+rest)
    p=ROOT/'docs/PROJECT_STATE.md'; old=p.read_bytes(); s=old.decode('utf-8-sig'); h,_,rest=s.partition('\n')
    note=f'''\n\n### CV-037：普通照片输入与空间监督元数据准入完成（2026-10-24）

[契约、证据与下一范围](../{report})。12合成检查与另实现复核通过，明确image_only与校准照片K像素链、禁止真值/未知图进入策略。ZOD9标签仅5组，余5条只多1组；新取得Mapillary cities9行/9元数据组（3城市），raw/opensfm分开适配，9内部位姿增量残差≤1m，最大{c['maximum_internal_residual_m']:.4f}m，不是测绘精度。0新图像，城市投影/尺寸/K与重叠/角色尚未核，当前不准入训练/自主策略。

v1 schema未适配记录保留，未发布旧报告；v2离线复用标签不重抓。MM-GAG12/9组仍缺已核严格相机点变换，Swiss375条仅15照片为弱代理；6确认/旧角色不变。下一CV038拟3固定城市代表像素工程预检，需另授权目录/获取，0模型；不继续旧小头/CCVPE变体追分。当前164索引且旧163值保持，平台4.12/默认未变，本轮无GUI升级或新SR。官方文本/2414字节标签经7897代理，网页取文失败与树API403均已记录；无新目录/C盘任务文件/Git提交推送。\n'''
    checked_write(p,old,h+note+'\n'+rest)
    seal=OUT/'23_最终文件绑定_v2.json'
    artifacts=[p for p in OUT.glob('23_*') if p.is_file() and p!=seal]
    assert sum(p.stat().st_size for p in artifacts)<1024**2-16384
    with seal.open('x',encoding='utf-8') as f:
        json.dump(dict(files=[entry(p) for p in artifacts],sources=d['sources'],
                       seal_itself_excluded=True,metadata_only=True,training_admitted=False),f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(dict(published=True,index=164,training_admitted=False,
                          bytes=sum(p.stat().st_size for p in OUT.glob('23_*') if p.is_file())),ensure_ascii=False))


if __name__=='__main__': publish()
