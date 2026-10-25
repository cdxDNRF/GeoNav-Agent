"""CV-004 saved-evidence report/gallery builder, never executes a model."""
import collections,hashlib,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/目标格定义与视觉尺度审查_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))
def main():
    data=read(OUT/'静态审查结果.json');notes=read(OUT/'逐对视觉观察.json');audit=read(OUT/'核验/另实现复核.json');assert audit['passed']
    assert len(notes['pairs'])==16 and notes['prediction_ranks_visible_during_review'] is False
    visual={r['pair_id']:r for r in notes['pairs']};fullrows=[];groupstats={};cards=[]
    for r in data['pairs']:
        pid=r['pair_id'];v=visual[pid];group=groupstats.setdefault(r['group_id'],{'pairs':0,'plausible':0,'not_clear':0,'unknown':0});group['pairs']+=1;group[v['shared_semantic_evidence']]+=1
        fullrows.append(f"| {pid} | {r['group_id']} | {v['ground_scene']} | {v['aerial_center_scene']} | {v['shared_semantic_evidence']} | {r['crop']['retained_area_fraction']:.1%} | {v['notes']} |")
        cards.append(f'''<article id="{pid}"><h2>{pid} · {html.escape(v['ground_scene'])}</h2><a href="面板/{pid}.jpg"><img loading="lazy" src="面板/{pid}.jpg" alt="{pid}地面与航拍审查面板"></a><p><strong>中心格：</strong>{html.escape(v['aerial_center_scene'])}</p><p><strong>观察：</strong>{html.escape(v['notes'])}</p><p class="muted">共同语义：{v['shared_semantic_evidence']}；拍摄点/地标是否同格：未知；图为离线评测材料，不进入Agent输入。</p></article>''')
    old=read(ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1/逐对排序结果.json')['rows'];local=read(ROOT/'DATA/processed_data/MM-GAG/局部语义匹配单因素验证_v1/逐对排序结果.json')['rows'];ranks=[]
    for r in data['pairs']:
        pid=r['pair_id'];g=next(x for x in old if x['pair_id']==pid and x['condition']=='Full');l=next(x for x in local if x['pair_id']==pid and x['condition']=='Full')
        ranks.append({'pair_id':pid,'group_id':r['group_id'],'visual_semantics':visual[pid]['shared_semantic_evidence'],'global_full_rank':g['true_rank'],'local_full_rank':l['true_rank']})
    result={'task':'CV-004','planned_pairs':16,'development_groups':10,'raw_target_center_mapping_passed':16,'EXIF_GPS_present':audit['EXIF_GPS_present'],'camera_heading_present':audit['camera_heading_present'],
      'independently_verified_geographic_footprints':0,'camera_vs_visible_landmark_unknown':16,'visual_semantics_counts':audit['visual_semantics_counts'],'visual_risks':audit['visual_risks'],'retained_area_min_max':audit['retained_area_min_max'],
      'per_group_visual_observations':groupstats,'post_review_rank_join':ranks,'review_is_subjective':True,'person_independent':False,'model_forward_calls':0,'training_steps':0,'navigation_episodes':0,'new_SR':None,
      'conclusion':'numeric geometry passed; precise capture-point/landmark and metric footprints unverified; images expose plausible viewpoint/scale risks, not proof of wrong labels or sole failure cause',
      'new_candidate_selected':False,'reason':'two fixed representation probes failed and exact visual/geo semantics remain uncertain; no justified new zero-training scorer or navigation admission',
      'next':'small team-assisted semantic label checklist on these16; no new source/model/parameter/navigation experiments from this batch'}
    put('审查汇总.json',json.dumps(result,ensure_ascii=False,indent=2))
    table='\n'.join(fullrows);gtable='\n'.join(f"| {k} | {v['pairs']} | {v['plausible']} | {v['not_clear']} | {v['unknown']} |" for k,v in groupstats.items())
    shared=audit['visual_semantics_counts'];risk=audit['visual_risks']
    put('目标格与尺度审查报告.md',f'''# 目标格定义与视觉尺度审查

CV-004 / 2026-10-17。**原16对取窗/索引映射全部通过；尚无证据把精确地理标签错误认定为匹配失败原因。** 本批没有训练、模型forward、导航或新SR。可打开[16对可视化审查页](样例审查.html)，查看已矫正方向的真实图像、实际中心裁剪、冻结5×5窗口与11×11上下文。

## 问题与证据层级

两次冻结感知探针未过后，本轮固定原16开发对/10临时组，20保留组不参与；审查字段先冻结。面板不含预测分数、排名或模型选中位置；先逐对视觉观察，再把既有全局/局部排名加入汇总。观察者仍是同一开发Agent，不是独立人员标注，也不可能消除之前已知汇总结果的全部认知影响。

1. 数字标签：原`GOMAA-Geo/models/ppo.py:412`从11×11特征库按`5-goal_row`/`5-goal_col`取窗，保证原(5,5)映到公开目标格。逐对25格共400项及16个中心位置另实现反向核验通过；已有原拼接轴证据16对均支持首索引为行。本轮没有发现转置、错位一格或中心映射错误，也未修改原标签。
2. 坐标来源：本地CSV可按文件名一一匹配16对，和原库存坐标一致；这16张JPEG均未保留可解析EXIF GPS、相机朝向亦为0。CSV不足以独立证明相机相对航拍格的像素位置，更不能给出画面里远处物体的坐标。16对“拍摄位置与可见地标是否同格”均保留未知，不能由模型排名反推正确格。
3. 文献名义尺度：本地GOMAA论文PDF第20页附录M描述来自带地理标记的智能手机照片，并报告航拍0.6m/px、256×256、10×10图块。若名义尺度适用，单图宽153.6m；但本地素材是11×11，缺采集缩放级别、影像角点与像素到GPS变换，因此153.6m只作条件算术，不能作为本批已测得格距或米制SG。不能把论文文字当成对本地工件的测绘验证。
4. 表征复现差异：同页实现注脚指向CLIP B/16，当前冻结探针使用已授权公开B/32及现有Sat2Cap；这是需要披露的差异，不能据此断定它就是失败原因，也不在本轮改模型或下载B/16。

完整提取及代码行见[定义来源与尺度证据](定义来源与尺度证据.json)。本地原始README只有许可信息，没有逐对象边界、相机视锥或地标标注。

## 逐对视觉证据

共同语义仅指可以从图片提出对应假设，不能证明地理同格。`plausible`={shared.get('plausible',0)}对，`not_clear`={shared.get('not_clear',0)}对，`unknown`={shared.get('unknown',0)}对。所有16对仍无独立验证的地理footprint。

| 对 | 临时组 | 地面画面 | 约定中心航拍格 | 共同语义观察 | 实际裁剪保留面积 | 观察与未知 |
|---|---|---|---|---|---:|---|
{table}

| 组 | 对数 | plausible | not_clear | unknown |
|---|---:|---:|---:|---:|
{gtable}

实际CLIP中心裁剪保留面积介于{audit['retained_area_min_max'][0]:.2%}～{audit['retained_area_min_max'][1]:.2%}，其余部分在编码前裁去；这是确定的像素事实，不等于裁剪造成了失败。视觉风险标记计数（可重叠、主观观察）为：{json.dumps(risk,ensure_ascii=False)}。本轮没有裁剪消融、重新预测或重新标注，不能量化这些因素对Top-k/SR的因果贡献。

## 结论与下一步

可以排除本16对在既定代码约定下的数字取窗/行列映射错误；不能排除或确认拍摄点、远处地标与航拍足迹之间的地理歧义。图像提供了视角、尺度、遮挡和裁剪的具体风险证据，但没有证明一个唯一瓶颈。CV-002/CV-003原失败及M0默认均保持。

当前不选择新的零训练打分候选，也不凭这个小审查直接训练投影或扩参。优先让团队依据本页，补充原16对“定位拍摄点还是指定可见地标”的明确目标语义及可核查依据；证据足够后再冻结一个候选。资料缺失时诚实保留“原代码约定的代理格标签”，不要伪造测绘真值或从相似度结果改答案。见[后续准入清单](后续目标语义补证据清单.md)。

## 工程与平台交付

交付可离线打开的16对审查页与面板，帮助用户查阅失败样例；不修改生产GUI，不开放未经验证的地面目标导航。5项几何/裁剪/GPS边界测试和另实现400格/16裁剪/原件哈希复核通过；同Agent复核不称独立人员审查。所有新源码/输入与旧封存绑定，总输出≤8MiB，0模型加载/forward、训练、导航、API、下载、保留组消费。

复核仅运行`D:/PYTHON/python.exe project/src/eval/audit_crossview_label_scale_v1.py`（不加`--save`）。本批freeze/build/报告生成器均已消费，不覆盖重跑；原始图、模型、默认、Word/PDF与旧实验保持。
''')
    put('样例审查.html','''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CV-004 · 目标格与视觉尺度审查</title><style>body{margin:0;background:#edf2f6;color:#19354b;font:16px/1.65 "Microsoft YaHei",sans-serif}header{background:#0e263a;color:white;padding:30px max(24px,calc((100vw - 1480px)/2))}h1{font-size:28px;margin:0 0 12px}main{max-width:1480px;margin:24px auto;padding:0 20px}nav{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}nav a{background:#fff;padding:5px 12px;border:1px solid #d7e2ea;border-radius:8px;color:#17676b;text-decoration:none}article{background:white;border-radius:14px;padding:20px;margin-bottom:28px;box-shadow:0 3px 18px #1231}img{width:100%;height:auto;display:block;border-radius:8px}h2{font-size:21px}.muted{color:#657b8a}.notice{border-left:4px solid #f4a843;background:#fff3df;padding:12px 18px}a{color:#067d86}</style><header><h1>CV-004 · 目标格定义与视觉尺度审查</h1><div>16对固定开发素材 · 0模型调用 · 全图仅供离线审查</div></header><main><p class="notice">橙框来自作者代码的中心格约定，尚未独立测绘。画面里的可见地标不一定等于照片拍摄点。这里的观察与标签未知项不能当成新的导航成绩。</p><p><a href="目标格与尺度审查报告.md">完整报告</a> · <a href="审查汇总.json">结构化汇总</a> · <a href="逐对视觉观察.json">观察原件</a></p><nav>'''+''.join(f'<a href="#{r["pair_id"]}">{r["pair_id"]}</a>' for r in data['pairs'])+'</nav>'+''.join(cards)+'</main></html>')
    put('后续目标语义补证据清单.md','''# 后续目标语义补证据清单

CV-004完成后的准入说明，不是新实验协议；没有启动训练/导航或消费保留组。

当前原代码把每个来源(5,5)作为代理目标格。建议团队只审查本16对：逐对说明希望定位“拍照设备所在格”还是“照片中指定物体所在格”；有依据才确认，不知道就保留未知。若改为物体定位，应补物体名称/像素区域、对应航拍位置及标注依据，另建数据版本和协议，旧16题答案不能回写。若仍是拍摄点定位，应尽量补CSV点到原始影像足迹的公开采集信息/可核查变换；不能仅从图片相似度判断坐标。

可以在样例审查页直接逐对查看，提出异议只写新审核记录。本次同Agent视觉判断是观察性材料，不替代团队或独立标注。没有新资料时，本轮诊断即收束，保留原“代码约定标签、地理footprint未知”的边界，不要求联系作者，也不追加模型下载。

后续如选小型关系投影训练，至少先说明可用训练/开发来源、相机/目标语义、单一因素、冻结骨干、新增参数与显存/保存上限、均值/错误目标对照和停止条件，并另行授权。本轮证据不足以指定必然有效的训练候选；不以平台演示需要放宽门槛。
''')
    put('README.md','''# 目标格定义与视觉尺度审查

CV-004 / 2026-10-17。原16对数字取窗映射通过；精确拍摄点/可见地标与航拍足迹关系仍未独立验证，不认定标签错，也不升级模型。

| 内容 | 入口 |
|---|---|
| 直接看16对实际图像与观察 | [离线样例审查页](样例审查.html) |
| 结论/定义/尺度/分组与限制 | [审查报告](目标格与尺度审查报告.md) |
| 全部静态与主观观察原件 | [静态结果](静态审查结果.json) / [逐对视觉观察](逐对视觉观察.json) / [汇总](审查汇总.json) |
| 冻结与证据 | [协议](冻结审查协议.json) / [输入源码SHA](输入与源码绑定.json) / [代码与本地论文](定义来源与尺度证据.json) |
| 核验/恢复 | [另实现复核](核验/另实现复核.json) / [阶段封存](阶段封存.json) |
| 下一步仅准入清单 | [团队目标语义补证据](后续目标语义补证据清单.md) |

0模型加载/forward、训练、重编码、导航、模型API、下载与20保留组消费，无新SR；原默认/当前GUI/Word/模型/旧实验保持。面板全图/真值是离线审查材料，不进入Agent。

恢复只运行 `D:/PYTHON/python.exe project/src/eval/audit_crossview_label_scale_v1.py`，不加`--save`。freeze/build/报告生成器均已消费，不能盲重跑或覆盖。全批≤8MiB，面板为派生缩略图而非额外原图副本。
''')
    print(json.dumps({'reports_written':True,'panels':16,'new_model_calls':0,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
if __name__=='__main__':main()
