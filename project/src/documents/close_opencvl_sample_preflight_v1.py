"""CV028 publication from sealed engineering outputs; never acquires data."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
REL='DATA/processed_data/OpenCVL/空间监督小样本预检_v1'
OUT=ROOT/REL


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def entry(p):
    with p.open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=s)


def put(n,v):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2));f.write('\n')


def close():
    if (OUT/'阶段封存.json').exists():raise FileExistsError('Already published; no repeated closure')
    b=read(OUT/'首图像取得绑定.json');s=read(OUT/'空间监督检查.json');a=read(OUT/'另实现核验.json');tests=read(OUT/'合成测试.json');done=read(OUT/'执行完成.json');q=read(OUT/'固定四对清单.json')['pairs']
    assert a['passed'] and tests['passed']
    for e in b['sources']+b['inputs']+b['protected']+[b['protocol'],b['queue']]:assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    rows=[]
    for p,r in zip(q,s['pairs']):
        rows.append(f"| {p['pair_id']} | {p['label']['ground_image'].split('/')[0]} | {r['width']}×{r['height']} | {r['resolution_m_per_pixel']:.2f} | {r['axis_error_m']:.3f} | {r['goal_cell_evaluator_only']} | {r['horizontal_fov_degrees']:.1f}° |")
    status='通过' if s['passed'] else '未通过，停止本路线后续执行'
    report=f'''# CV-028：OpenCVL四对空间监督工程预检

**工程检查{status}；不代表跨视角模型有效或导航SR通过。** 本轮取得4对原始地面/航拍PNG和3份小型原标签JSON，原字节SHA/ZIP CRC均验证；没有加载模型、训练、编码或执行导航。MM-GAG六确认组与原航拍M0/平台4.11不改。

## 实际取得与保留

官方[项目样例](https://open-cvl.github.io/)指向[OpenCVL_sample_data.zip](https://drive.google.com/file/d/1xKol68v_dtLaXXbwVWZxg2AxeWeg-Kvi/view)。整包标称44,220,854字节/186成员；本轮只取ZIP末22字节、25,680字节中央目录、3个ZOD标签和冻结的8个PNG成员，不取得或保留完整ZIP。不访问Mapillary像素，不需要注册或联系作者。

响应正文累计{done['network_body_bytes']}字节（{done['network_body_bytes']/1024**2:.2f}MiB），含公开HTML探测、一次失败的中央目录解析读取和所有成功范围请求；不包括HTTP头/TLS流量。所有实际成员使用206精确Content-Range、解压长度、CRC与SHA核对。最初严格Windows文件名守卫拒绝上游ISO时间戳冒号，仅在元数据阶段停止；失败及额外25,680字节读取如实保留。最终不按ZIP路径解压，采用平面安全文件名+原成员映射，原标签内名称保留不改。没有重复图像取得或解码。

当前仅输入/核验内的小型工程副本，8原图各解码一次，核验阶段不重复解码原图。采用原标签先固定选择：frames优先、再drives、再sequences，文件名字典序，按拍摄点中心至少500m贪心隔离，固定4对后不按像素或模型成绩替换。读取3份ZOD标签共9条；其余5条仅元数据已见，未取得像素，不称未消费盲确认。

## 坐标、尺度与照片输入

| 工程对 | 来源类型 | 航片像素 | 分辨率m/px | GPS—像素偏移差m | 目标格（评测侧） | K推导水平FoV |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

沿官方约定x=W/2+dx、y=H/2−dy；dx/dy为原始航拍像素偏移。经纬度使用局部切平面重新推导东/北位移，再与偏移×分辨率比较，最大差{max(r['axis_error_m'] for r in s['pairs']):.3f}m，冻结上限1m。另实现采用球面反解距离/方位，和切平面分解最大差{a['independent_spherical_vs_tangent_max_m']:.8f}m。该一致性支持本批数值/轴向契约，不是独立测绘验证，也不证明RGB关系可学。

4对均为ZOD车载普通前视照片，有原K、heading及位置标签；不将其等同任意手机、山地/冰川照片。K只用于评测侧检查照片与标签，不传真实GPS/偏移/目标格给Agent。图像亮度方差/尺寸/EXIF/内参完整检查见[像素回执](像素质量回执.json)。没有校准或更改原标签，没有新模型预测。

每张父航片约100m×100m，切5格为约20m/格、0.01km²；不是更大真实搜索面积。像素裁窗采用整数半开边界，625px图也完整覆盖且不重复。目标位置仍集中于中部，不覆盖全部25目标格；不能照抄Masa C4—8队列或与旧300m/格成绩直接比较。

## 三种空间监督情形

- 同源窗内：{s['class_counts']['inside']}条。
- 同源窗外：{s['class_counts']['outside']}条；不裁到边缘伪造正例，评测侧位移不表示模型一定能从RGB判断方位。
- 空间隔离异源：{s['class_counts']['unrelated']}条；通过公开位置与不同源航片中心距离>500m核验，方位标签为null。没有只凭文件名不同就当负例。

这些合计400条是4对素材的派生监督记录，绝不是400个独立场景、400次导航或训练样本有效性已证。4个工程空间组间最小距离{min(r['meters'] for r in s['pair_separations']):.1f}m；上游训练切分不等于本项目训练集，预训练覆盖未知。当前全部标为engineering_development，尚未分配正式训练/确认。

## 观察者QA面板

![四对工程面板：红点为原标签，不是预测](核验/四对空间标签工程面板.png)

地面图来源ZOD，©2022 Zenseact AB；航拍与样例来源OpenCVL，原来源/条款按[官方NOTICE](https://github.com/Open-CVL/opencvl/blob/4838d1627cf28b28e81d2dd830995fe1fd3a1e2c/NOTICE.md)。红点是原始相机标签，网格仅为工程诊断，不能当作定位成功可视化。本批未发布到外部。

## 验证与下一步

9项内存合成测试通过：范围错位/200响应拒绝、CRC/长度/压缩上限、路径穿越、上游时间戳的安全映射、奇数尺寸网格覆盖、北向符号及无模型依赖。另实现核对8个PNG SHA/CRC/IHDR、空间三分类、球面坐标和累计网络/解码次数，通过；同Agent复核不称独立人员审查。

当前training_admitted=false、navigation_admitted=false；通过仅证明这4对具备更明确、可追溯的拍摄位置监督接口。下一步应先冻结一项轻量空间关系感知验证：原目标/均值/错误/禁用与不匹配拒绝必须保留，先看窗外方向和拒绝证据，再决定是否闭环。4工程对只作工程开发，不拿来证明泛化；不立即训练旧8192小头、扩参或消费原6确认组。若本批任何必要门槛未过，应保留失败并停止，不能更换样例追过线。

## 恢复和绑定

[冻结协议](冻结工程协议.json)、[四对清单](固定四对清单.json)、[首图像绑定](首图像取得绑定.json)、[取得/解码回执](取得回执.jsonl)、[空间标签检查](空间监督检查.json)、[另实现核验](另实现核验.json)、[阶段封存](阶段封存.json)。全部输入映射原ZIP成员、CRC/SHA与源码，先查看落盘状态；labels/freeze/execute/写盘audit/close均已消费，不能覆盖重跑。恢复只读核验导入audit_opencvl_sample_preflight_v1.audit()，不得重新下载/解码已成功输入。

本批通过既有127.0.0.1:7897代理取得官方样例范围；网页工具样例页只给登录入口，标准文件下载URL先返回确认HTML，确认参数后206成功，未换镜像/未改系统代理。0模型加载/forward/训练/导航/模型API/原MM-GAG确认，未新增C盘任务文件，未提交/推送Git。
'''
    put('README.md',report)
    put('下一步边界.json',dict(engineering_passed=s['passed'],training_admitted=False,navigation_admitted=False,new_SR=False,
        sources=4,purpose='engineering_development',next='freeze one lightweight spatial-relation perception and rejection study before considering navigation' if s['passed'] else 'stop and retain failure',
        no_automatic_execution=True,old_reserved_groups_unchanged=True))
    p=ROOT/'项目导航/实验索引.json';idx=read(p);old=idx['batches'];assert len(old)==152
    old_sha=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    idx['date']='2026-10-23';idx['batches']=old+[dict(dataset='OpenCVL官方样例4对',kind='取得与空间监督工程',name=OUT.name,category='CV-028',
        status='空间监督工程'+status+'，无模型/导航准入',scope='4对8原图/3原标签；0模型训练导航与6确认；非新SR',path=REL,
        reports=[REL+'/README.md'],evidence=[REL+'/'+n for n in ('冻结工程协议.json','首图像取得绑定.json','空间监督检查.json','另实现核验.json','下一步边界.json','阶段封存.json')])]
    p.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');assert read(p)['batches'][:-1]==old
    put('索引原值核验.json',dict(old_batches=152,new_batches=153,old_values_preserved=True,old_canonical_sha256=old_sha))
    p=ROOT/'docs/维护/版本生命周期.json';life=read(p);life['updated']='2026-10-23';life['cv028_spatial_supervision']=dict(status='engineering_passed' if s['passed'] else 'engineering_failed',
        report=REL+'/README.md',replaces='CV027 planned data precheck only; prior evidence preserved',depends_on='CV027 original task/asset limits',
        recovery=REL+'/阶段封存.json',training_admitted=False,navigation_admitted=False,retention='one selected input copy;no completeZIP/model/features')
    p.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;t=p.read_text(encoding='utf-8-sig');header,_,rest=t.partition('\n');prefix='' if name=='README.md' else '../'
        p.write_text(header+'\n\n[CV-028 · OpenCVL空间监督工程]('+prefix+REL+'/README.md)：4对官方素材/坐标与三类监督检查'+status+'；没有训练、模型编码或导航SR。\n\n'+rest,encoding='utf-8')
    files=[entry(p) for p in sorted(OUT.rglob('*')) if p.is_file()]
    put('阶段封存.json',dict(task='CV-028',files=files,sources=b['sources'],engineering_passed=s['passed'],real_images=8,actual_decodes=8,
        encoder_calls=0,navigation=0,old_evidence_unchanged=True,retention='one original selected-input copy plus small QA/metadata'))
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert size<=68*1024**2
    print(json.dumps(dict(published=True,files=len(files),bytes=size,index=153,engineering_passed=s['passed']),ensure_ascii=False))


if __name__=='__main__':close()
