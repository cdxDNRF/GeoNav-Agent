"""CV-016 report and training-branch closure after observer review/alternate audit."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/同任务拍摄点标签与可辨识性审查_v1'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    audit=load(OUT/'核验/另实现复核.json')
    if not audit['passed']:raise ValueError('Alternate audit required')
    protocol=load(OUT/'冻结审查范围.json');pixels=load(OUT/'像素回执与面板.json');reviews=load(OUT/'逐源视觉审查.json')
    obs={r['source_id']:r for r in reviews['sources']};scenes=Counter(r['scene_correspondence'] for r in reviews['sources'])
    gps=sum(r['gps_ifd_present'] for r in pixels['receipts']);heading=sum(r['heading_present'] for r in pixels['receipts'])
    mapped=sum(x['center_mapping_consistent_at_old_MAE8_tolerance'] for x in pixels['panels'])
    coordinates=sum(s['metadata']['camera_coordinate_available'] for s in protocol['sources'])
    transforms=sum(s['metadata']['pixel_to_GPS_transform_available'] for s in protocol['sources'])
    multi=sum(s['metadata']['observed_distinct_queries_in_group']>1 and s['metadata']['same_area_multiple_camera_points_verified'] for s in protocol['sources'])
    if multi!=0 or transforms!=0:raise ValueError('Unexpected evidence; review training conclusion rather than auto-admit')
    verdict={'task':'CV-016','sources':14,'fit_sources':9,'development_sources':5,'raw_reads':42,'raw_decodes':42,
        'pixel_center_mapping_consistent':mapped,'camera_coordinates_available':coordinates,'GPS_to_pixel_transform_verified':transforms,
        'GPS_ifd_present_in42_images':gps,'heading_present_in42_images':heading,'scene_correspondence_counts':dict(scenes),
        'independent_geographic_camera_labels_verified':0,'camera_points_visually_proven':0,'camera_points_identifiability_unknown':14,
        'verified_same_area_multiple_query_relations':multi,'training_branch_status':'closed_for_current_materials',
        'new_training_candidate_admitted':False,'new_model_forward_calls':0,'new_predictions':0,'new_training_steps':0,
        'new_navigation_SR':None,'confirmation_access':0,'default_upgraded':False,
        'reason':'center geometry/scene correspondence do not supply georeferenced camera visibility or within-area multi-query positives; stop current9-pair head-training branch, not the whole cross-view task'}
    write('审查汇总与收束判定.json',verdict)
    table=[];gallery=[]
    for s,p in zip(protocol['sources'],pixels['panels']):
        r=obs[s['source_id']];scene={'plausible':'场景可关联','unclear':'场景不清'}[r['scene_correspondence']]
        table.append(f"| {s['source_id']} | {s['role']} | {scene} | 未知 | {r['observation']} |")
        gallery.append(f"- [{s['source_id']} / {s['role']}](面板/{Path(p['panel']['path']).name})：{scene}；拍摄点仍未知。")
    write('拍摄点标签与可辨识性审查报告.md',f'''# 14来源拍摄点代理标签与局部可辨识性审查

CV-016 / 2026-10-18。一次有界审查已完成。**收束当前素材的8192参数小头训练支线，不追加训练或扩参追分。** 结论针对既有数据与监督准入，不是否定整个跨视角主动定位方向，也不认定原标签错误。

## 审查范围与证据层次

只使用CV-011/014已消费的9fit＋5dev代表，每来源1地面照、原中心sat_5_5和已有11×11拼图，总42幅，各读取解码一次。6确认组24素材、同组非代表与其它原始文件本轮未打开。原代码/像素中心约定、独立地理映射与照片可观测性分别判定，不按模型排序改变正格。原16对CV-004审查不替代新14来源，本轮也未使用模型分数选“好样本”。

这是同Agent事后非盲观察，不是实测拍摄点标注或独立人员审查。场景形态相近只作描述，不能把远处建筑/山体/水面自动当成相机所在格。用于图审的整图只给观察者，不输入导航Agent。

## 核实结果

| 证据项 | 本轮结果 | 能说明什么 |
|---|---:|---|
| 原中心格与拼图中心像素缩略一致 | {mapped}/14 | 中心代码约定与行列工程一致，非地理测绘认证 |
| 元数据有相机坐标 | {coordinates}/14 | 提供点坐标记录，不能单独映射到具体影像像素 |
| 已绑定资料提供独立GPS→像素变换 | {transforms}/14 | 地理footprint/相机到图块对应仍未认证 |
| 42幅已读JPEG带GPS IFD/朝向字段 | {gps}/{heading} | 本轮文件内记录；缺项不代表世界上不存在其它资料 |
| 场景可关联/不清 | {scenes.get('plausible',0)}/{scenes.get('unclear',0)} | 非盲视觉描述，不是真值或新验收子集 |
| 能独立证明精确拍摄点格 | 0/14 | 14个保持未知，不能解释成14个标签错误 |
| 同区域多查询/不同拍摄点正格关系已认证 | {multi}/14 | 当前每组仍只有一个受审代表/一条正配对关系 |

## 逐来源观察

| 来源 | 既有用途 | 场景层关联 | 精确拍摄点 | 可核查的观察与限制 |
|---|---|---|---|---|
{chr(10).join(table)}

完整观察者面板见[阅读索引](README.md)，原始字段与风险标签见[逐源记录](逐源视觉审查.json)。所有14来源保留，不按“可关联”标签筛样重算模型成绩。

## 为什么停止这条训练支线

CV-014已证明500次更新能降低9查询的拟合损失，但5来源开发精确格和目标对照未过。CV-015提出先审标签与关系覆盖，本轮没有补出可信的同区域多拍摄点监督，也不能从这些照片和JPEG拼图独立证明拍摄点像素。继续用相同9条正配对增加更新次数/秩/骨干，没有新增证据支持；停止当前训练方案比重复试参更符合既定停止条件。

此结论不是归因试验：不能锁定失败完全来自标签、数据量、视角、预训练表征或优化器。场景可关联的样本与Swiss头的部分排序信号是反证，不能写“地面照片没有信息”。也不能通过把近邻/地标改为成功格、放宽半径、打开确认集来让原试验通过。

## 后续主线和平台边界

保留用户的“地面目标→局部航拍关联→主动探索”目标，但当前小头训练分支结束。后续若继续研究，首先需要可合法使用且能核对相机位置与航拍范围、同区域多目标关系的配对素材或完整可核验公开任务；是否获取/新增模型/训练需另立来源与资源范围。当前不指定未经核验的新数据集、不请求作者私有资料，不以“以后可能改善”自动续训。

现有航拍M0/V4.5继续按已通过协议运行。平台可展示本批面板、证据与失败边界；地面模式标为研究未准入，没有跨视角导航SR。观察者全图不得进入Agent；现有Word和生产GUI/服务保持。6确认组继续封存，不因原训练失败改作训练/校准。

## 执行、复核与恢复

原图42次SHA/读取/解码，84条attempt/decoded事件；14面板及输入/源码绑定。另实现只用已保存16px中心缩略重算MAE、核42回执/事件覆盖/面板SHA和14来源观察完整性，不再解码原图。它核验几何与记录一致性，不验证主观视觉判断为地理真值；同Agent另一实现，非独立人员审查。

本批0模型forward、新预测、新训练、导航/API/下载或确认消费。总≤8MiB、面板≤6MiB，无完整原图/特征/权重副本。只读恢复核[封存](阶段封存.json)、[绑定](输入与源码绑定.json)、[像素回执](像素回执与面板.json)、[另实现](核验/另实现复核.json)；freeze/prepare/audit/close已消费，不覆盖或盲重读原图。
''')
    write('README.md',f'''# 同任务拍摄点标签与可辨识性审查（CV-016）

14已消费来源的一次审查完成：中心映射{mapped}/14，精确拍摄点仍未知14/14，同区域多查询关系未补足。当前素材小头训练支线收束，不加训/扩参；6确认保留。场景可关联{scenes.get('plausible',0)}/14不是模型通过或地理真值。

- [完整报告](拍摄点标签与可辨识性审查报告.md) / [汇总与收束判定](审查汇总与收束判定.json)
- [逐源观察](逐源视觉审查.json) / [42像素回执与中心检查](像素回执与面板.json)
- [范围](冻结审查范围.json) / [源码输入绑定](输入与源码绑定.json) / [另实现复核](核验/另实现复核.json) / [封存](阶段封存.json)

## 观察者面板

面板是事后观察者资料，不是Agent输入；不显示或使用模型排名选样。每图依次为地面照、原中心格、带中心框的完整拼图。

{chr(10).join(gallery)}

0模型调用/训练/新SR，原默认/Word/生产GUI保持。只读核SHA与已存缩略即可复核；所有执行/生成脚本已消费，不重复打开原始影像。
''')
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='README.md':
            data=p.read_bytes();files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());panelbytes=sum(p.stat().st_size for p in (OUT/'面板').glob('*.jpg'))
    if total>8*1024**2 or panelbytes>6*1024**2:raise ValueError('Storage cap exceeded')
    sources=load(OUT/'输入与源码绑定.json')['sources']
    write('阶段封存.json',{'task':'CV-016','files':files,'sources':sources,'sealed_bytes':sum(x['bytes'] for x in files),
        'mutable_entries_excluded':['README.md'],'training_branch':'closed_for_current_materials','new_training_candidate_admitted':False,
        'new_model_forward_calls':0,'new_navigation_SR':None,'confirmation_access':0})
    print(json.dumps({'sealed_files':len(files),'bytes_before_seal':total,'panel_bytes':panelbytes,'training_branch':'closed_for_current_materials'},ensure_ascii=False))


if __name__=='__main__':main()
