"""CV038 one-time publication after bounded acquisition + visual/alternate QA."""
import json
from project.src.tools.city_pixel_acquisition_v1 import ROOT,OUT,entry,read,save


def checked_write(p,old,text):
    assert p.read_bytes()==old,str(p)+' changed during write'
    p.write_bytes(text.encode('utf-8'))


def publish():
    assert not (OUT/'README.md').exists(),'Already published; inspect, do not overwrite'
    s=read(OUT/'像素与标签预检汇总.json');b=read(OUT/'首获取冻结绑定.json')
    audit=read(OUT/'核验/另实现复核.json');visual=read(OUT/'核验/视觉审查.json');tests=read(OUT/'核验/合成检查.json')
    assert audit['passed'] and tests['passed'] and tests['tests_run']==9
    assert visual['actual_pairs_viewed']==3 and len(visual['pairs'])==3
    for e in b['sources']+[b['authorization'],b['admission'],b['unchanged_role_ledger']]+b['runtime']:
        assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    assert read(ROOT/'项目导航/实验索引.json')['batches']==b['index_prefix']
    assert read(ROOT/'docs/维护/版本生命周期.json')['current']==b['current']
    table='';passed=0;K_matches=0;shifts=[]
    for r in s['results']:
        im=r['images'];g=r['geometry'];ground=im['ground']['png'];aerial=im['aerial']['png']
        points=[]
        for variant in ('raw','opensfm'):
            p=g['variants'][variant];points.append(f"{variant}({p['x']:.1f},{p['y']:.1f})")
        p,q=g['variants']['raw'],g['variants']['opensfm'];shift=((p['x']-q['x'])**2+(p['y']-q['y'])**2)**.5*r['label']['resolution'];shifts.append(shift)
        passed+=int(r['engineering_nonblank_and_points_inside']);K_matches+=int(g['principal_matches_actual_image_center_within1px'])
        table+=f"|{r['city']}|{ground['width']}×{ground['height']}|{aerial['width']}×{aerial['height']}|{'通过' if r['engineering_nonblank_and_points_inside'] else '未过'}|{'；'.join(points)}|{shift:.2f}|\n"
    image_only_candidate=passed==3 and all(p['ordinary_photo_candidate_from_visual_review'] for p in visual['pairs'])
    save('准入判定.json',dict(task='CV-038',complete=True,pixel_engineering_passed_pairs=passed,
        actual_pairs=3,ordinary_RGB_input_data_candidate=image_only_candidate,
        K_center_matches=K_matches,strict_calibrated_photo_input_admitted=False,
        independent_camera_point_accuracy_verified=False,training_admitted=False,autonomous_navigation_admitted=False,
        raw_to_opensfm_label_shift_m=shifts,original6_confirmation_preserved=True,
        next='CV039 freeze an ordinary-photo image-only spatial-supervision data/role and small-head protocol; no training implied',
        visual_review=entry(OUT/'核验/视觉审查.json'),alternate_audit=entry(OUT/'核验/另实现复核.json')))
    views='\n'.join('- '+p['city']+'：'+p['observation'] for p in visual['pairs'])
    network='本轮直连成功，未使用代理。' if not s['proxy'] else f"直连打开失败后，使用现有代理{s['proxy']}，错误与端口检查记录在获取事件；未使用镜像。"
    report=f'''# CV-038 三城市普通照片与空间标签像素预检

**实际取得固定3对/6图，基础像素与两套标签落点通过{passed}/3；普通RGB输入数据候选={image_only_candidate}。** 这是数据工程准入，不是模型成绩、导航SR或训练授权。严格相机校准链、独立测绘精度仍未证明。

## 固定输入与实际执行

沿用CV037各城市ground路径字典序代表，不按像素/分数选图；Amsterdam10000258569988630、Krakow1000015350919599、Stockholm1000081124702485。精确读取ZIP的6个成员，每个核本地头/名字、解压长度/CRC32、原文件SHA256、PNG各chunk CRC；不取得完整ZIP，不读上游test或MM-GAG6确认组。每图真实解码1次，共6次，复用解码结果作统计和接触表；审计不重解码。

|城市|地面尺寸|航拍尺寸|非空白与两位姿落点|原生像素坐标|raw→修正标签位移（米）|
|---|---|---|---|---|---:|
{table}
基础质量规则在首获取前冻结：平均灰度std≥2、近黑/近白比例各<95%；不是人脸/隐私或通用图像质量评分，也不据此挑有利样例。各图原模式/统计/EXIF存在性、文件/解码RGB哈希、标签和边界余量见逐城市JSON。

## 照片投影与K

人工视觉检查固定接触表，记录如下：
{views}

这是外观证据，不证明原始镜头模型或畸变已标定。K主点与实际图像中心在1像素内匹配{K_matches}/3，K矩阵/主点合法与尺寸吻合不能替代原相机、resize/crop/rectification链。当前`usable_calibration=False`；不能据此宣称任意手机照片的已校准FoV。未来优先普通`image_only`路线；若候选依赖K，需单独补成像链来源，不能把当前尺寸检查写成完整校准通过。

## 拍摄点标签、北向与隔离边界

raw和OpenSfM修正GPS/dx/dy/heading各自绑定，全部保存；x=W/2+dx，y=H/2−dy，在原生航片坐标计算，不夹到边界/中心、不按成绩换标签。两位姿位移是数据集估计差异，不是当前模型误差或测绘真值。heading/GPS/真实像素位置只留在审查/标签侧，禁止作为Agent输入。

当前坐标约定dx向东、dy向北可与标签增量相容，但PNG不提供独立测绘控制点或完整栅格地理变换；标签约定北向与“独立核实影像真实北向”分开。不得由可见地标推断相机精确站点已经肉眼证实。普通照片可辨认场景不保证精确拍摄点可辨识，未来监督损失/容差需披露标签不确定性。

三城市图像文件和解码像素均无重复，保留原GPS作离线来源分组；不同城市/远距离元数据支持彼此不同地域，但这3个工程样例不能充当未见确认池，不能作为充分泛化证据。各自标记`consumed_pixel_engineering_unassigned_to_training`，没有自动划分训练/开发/确认；旧ZOD0/1校准、2/3开发和MM-GAG6确认保持。

## 决定与下一步

不扩大骨干、不复活旧8192全局头或CCVPE阈值/填边扫描。下一项CV039只冻结**普通照片兼容的空间监督与不匹配拒绝**方案：明确来源角色与地理隔离、所用raw/修正标签及不确定性、原生像素到训练窗口的变换、RGB-only输入、局部空间输出/窗外拒绝、同预算基线与真实/禁用/均值/错误目标对照，再确定小头结构、参数/资源上限和停止条件。当前3对先用于工程开发，不自行转成训练或正式测试。

本轮没有训练器启动、模型forward/权重读取、导航或新SR；数据预检通过不自动通过后续定位/导航门槛。当前平台4.12/server19/service11/pack23/原M0与S2S3S4保持，可继续使用现有场景和地面人工连续观测；本轮未升级GUI或改服务。

## 复核与恢复

9合成测试通过；另实现{len(audit['checks'])}项通过（独立PNG chunk/CRC/尺寸与SHA、两位姿坐标、预算、角色与原件绑定）。NumPy与Pillow在线双统计一致；离线审计不重新解码全部像素。由同一Agent完成另一实现，不称独立人员审查。视觉判断不是模型预测。

[首获取绑定](首获取冻结绑定.json) · [像素/标签汇总](像素与标签预检汇总.json) · [准入决定](准入判定.json) · [另实现](核验/另实现复核.json) · [视觉记录](核验/视觉审查.json) · [固定接触表](核验/三城市固定样例接触表.png)。本目录输入PNG是本批唯一新获取副本，按研究原件保留；无完整ZIP/额外临时包。恢复只读这些原件，不重复运行获取器或publisher覆盖。

网络正文实际{s['network_body_bytes']}字节≤16MiB，目录长期总≤20MiB；逐次范围/CRC/失败与代理记录见获取事件。{network}没有安装或持久环境修改、C盘任务文件、Git提交推送、作者联系、模型API或6确认消费。
'''
    with (OUT/'README.md').open('x',encoding='utf-8') as f:f.write(report)
    rel=OUT.relative_to(ROOT).as_posix();report_path=rel+'/README.md'
    p=ROOT/'项目导航/实验索引.json';old=p.read_bytes();idx=read(p)
    idx['date']='2026-10-24';idx['batches'].append(dict(category='CV-038',dataset='OpenCVL Mapillary cities三固定代表',kind='普通照片/空间标签像素预检',
        name='三城市照片与空间标签预检_v1',status=f'基础像素{passed}/3，训练/严格校准/导航未准入',path=rel,reports=[report_path],evidence=[rel+'/像素与标签预检汇总.json',rel+'/准入判定.json',rel+'/核验/另实现复核.json']))
    checked_write(p,old,json.dumps(idx,ensure_ascii=False,indent=2)+'\n')
    p=ROOT/'docs/维护/版本生命周期.json';old=p.read_bytes();life=read(p);life['updated']='2026-10-24'
    life['cv038_city_pixel_precheck']=dict(status='pixel_engineering_complete_no_training_admission',report=report_path,
        replaces='CV037 pending pixel check only; no source/model/platform/default replaced',
        dependencies=['调研/23_普通照片输入与空间监督准入_v2.json'],recovery='read fixed inputs and receipts; no automatic reacquisition',
        input_level='ordinary RGB data candidate; K provenance and independent survey accuracy not verified',training_admitted=False,
        next='CV039 frozen data roles and image-only spatial supervision protocol',artifacts=[entry(OUT/'准入判定.json'),entry(OUT/'README.md')])
    checked_write(p,old,json.dumps(life,ensure_ascii=False,indent=2)+'\n')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;old=p.read_bytes();text=old.decode('utf-8-sig').replace('\r','');h,_,rest=text.partition('\n');pre='' if name=='README.md' else '../'
        line=f'[CV-038 · 三城市照片与空间标签预检]({pre}{report_path})：3对6图已核，基础像素{passed}/3；先冻结空间监督方案，未训练/导航。'
        checked_write(p,old,h+'\n\n'+line+'\n\n'+rest)
    p=ROOT/'docs/PROJECT_STATE.md';old=p.read_bytes();text=old.decode('utf-8-sig').replace('\r','');h,_,rest=text.partition('\n')
    note=f'''\n\n### CV-038：三城市像素与空间标签工程完成（2026-10-24）

[输入、质量、落点与限制](../{report_path})。取得固定3对6图，基础非空白/两位姿落点{passed}/3，K主点与实际尺寸相容{K_matches}/3但完整校准链未核；raw/opensfm分开，不改标签，不宣称独立测绘精度。9合成/{len(audit['checks'])}另实现过，0模型/训练/导航；这些仅新消费工程来源，不自动转训练或未见确认。下一CV039冻结普通RGB空间监督与不匹配拒绝方案，不复活旧全局头/CCVPE变体。当前165索引且旧164值/current保持，平台未升级；6确认与原默认不变。网络正文{s['network_body_bytes']}字节≤16MiB，总≤20MiB，无完整ZIP/C盘任务输出/Git提交推送。\n'''
    checked_write(p,old,h+note+'\n'+rest)
    files=[p for p in OUT.rglob('*') if p.is_file()];assert sum(p.stat().st_size for p in files)<20*1024**2-16384
    save('阶段封存.json',dict(files=[entry(p) for p in files],sources=b['sources'],seal_itself_excluded=True,training_admitted=False))
    print(json.dumps(dict(published=True,index=165,pairs=3,basic_passed=passed,storage_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())),ensure_ascii=False))


if __name__=='__main__':publish()
