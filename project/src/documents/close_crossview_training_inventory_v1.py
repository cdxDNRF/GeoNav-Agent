"""CV-006 eligibility report and one conditional small-head design; no training."""
import html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'DATA/processed_data/MM-GAG/对齐头训练素材准入_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(n,s):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s if isinstance(s,str) else json.dumps(s,ensure_ascii=False,indent=2))
def main():
    s=read(OUT/'素材核验汇总.json');a=read(OUT/'核验/另实现复核.json');assert a['passed']
    scenes=read(OUT/'逐场景去重与共用图块.json')['scenes'];records=read(OUT/'逐对素材核验.json')['records']
    table='\n'.join(f"| {x['scene']} | {x['records']} | {x['strict_pairs']} | {x['candidate_pairs']} | {x['unique_ground_pixels']} | {x['unique_300px_cells']} | {x['common_300px_anchor_count']} |" for x in scenes)
    aliases=[]
    for scene in scenes:
        first=next(x for x in records if x['scene']==scene['scene']);changes=[]
        for side in ['ground','aerial']:
            if not first['strict_paths'][side]['exists']:
                changes.append(f"{first['strict_paths'][side]['metadata']} → {first['candidate_paths'][side]}")
        if changes:aliases.append({'scene':scene['scene'],'records_affected':25,'examples':changes,'status':'candidate path correspondence only, not approved target label'})
    put('候选路径别名记录.json',{'entries':aliases,'raw_files_changed':0,'metadata_changed':0,'label_approval':False})
    design={'task':'CV-006','status':'conditional_design_only_not_training_admitted','only_candidate':'rank8 residual projection on frozen ground512D features',
      'formula':'normalize(g + U @ V @ g), U512x8,V8x512; U starts0 so initial prediction equals identity baseline',
      'trainable_parameters':2*512*8,'frozen_assets':['existing CLIP ViT-B/32 vision','existing Sat2Cap'],'baseline':'identity target projection, existing CV-002 center-crop/global-cosine protocol; no simultaneous fullframe change',
      'hard_prerequisites':['review candidate aliases into a new derived manifest without editing raw metadata','explicitly establish each positive300px cell and its capture-point or alternative task meaning','group all25 shifted views and identical/shared tiles before any train/dev split','training source roles cannot overlap the existing16 development pairs or their groups, and MM-GAG20reserved remain protected','declare semantic/geographic limits if a weak scene/object task is adopted instead of capture-point localization'],
      'training_manifest':None,'development_manifest':None,'confirmation_manifest':None,'sources_assigned':False,'geographic_nonoverlap_confirmed':False,
      'proposed_engineering_budget_only':{'seed':0,'maximum_updates':500,'maximum_wall_minutes':30,'CUDA_allocation_bytes':2*1024**3,'feature_cache_bytes':32*1024**2,'head_checkpoint_bytes':1024**2,'retention':'initial/final head and one active recovery only; no duplicate encoder/optimizer bundles'},
      'protocol_not_frozen':'loss, exact feature bindings/source assignments/negative construction and all stop gates must be frozen after eligibility; no model execution allowed by this file',
      'candidate_acceptance_proposal':'same admitted development queue Full/Mean/Wrong, recovered/harmed and source groups; preserve prior gates for any original16 diagnostic, no weaker post-score thresholds',
      'stop':'at most one authorized pilot budget; no rank/loss/temperature sweep or automatic navigation; insufficient data closes proposal',
      'training_started':False,'new_model_calls':0,'navigation_started':False}
    put('条件式小头方案_未准入.json',design)
    alias_md='\n'.join(f"- {x['scene']}："+'；'.join(x['examples']) for x in aliases)
    text=f'''# 小型跨视角对齐头训练素材准入核查

CV-006 / 2026-10-17。**现有文件可支撑受限的数据工程/配对拟合准备，但当前不准入原“拍摄位置格”任务的正式小头训练。** 不是训练程序无法运行，而是可用独立来源、路径与目标格语义的条件尚未同时满足；本轮0模型加载/forward、训练、编码、导航、API和下载。

## 数量与来源：375条不等于375个地点

SwissViewMonuments元数据有{s['swiss_metadata_records']}条，按名称为{s['named_scenes']}个场景，每场景25条。GeoExplorer本地PDF第11—12页S1.4明确说明：15个地面—航拍对，各提供25个平移航拍窗口，让固定目标遍历5×5位置。原论文把此子集用于未见目标评测，通常只用Masa训练；如果本项目今后用Monuments训练，必须改称新的训练/迁移协议，不能仍声称是在该子集零样本复现原论文。

实际逐图解码/尺寸与RGB像素去重得到**{s['unique_ground_pixel_images']}张不同地面照片**（文件字节SHA种类{s['unique_ground_byte_hashes']}），不是375次独立拍摄。依据同地面像素或共用航拍300px图块连接后，有{len(s['pixel_connected_scene_groups'])}个保守内容组；这只防已知内容重复，不能证明{len(s['pixel_connected_scene_groups'])}个空间隔离区域。重复25窗口不能拆入训练和开发，也不能当作25个独立地理样本。

| 场景名 | 记录数 | 元数据路径直接配对 | 候选别名后可解码 | 不同地面图 | 不同300px图块 | 所有25窗共有图块数 |
|---|---:|---:|---:|---:|---:|---:|
{table}

## 路径与几何：分别报告严格原件和候选适配

严格按原JSON路径，仅{s['strict_pairs_present']}/375对两侧文件均存在；地面路径缺失{s['strict_ground_paths_missing']}条、航拍路径缺失{s['strict_aerial_paths_missing']}条。部分问题是名称或扩展名不一致。按首动作前列明的候选别名/同stem扩展名查找后，{s['candidate_pairs_decodable']}/375对可解码；这不是原始元数据已经正确，也不表示别名对应的地理目标已验真。没有改动或重命名任何原始文件。

候选例子（仅保存派生记录，不自动作为训练标签）：

{alias_md}

有{s['common_anchor_scene_count']}个场景在25窗中恰有一个共用300px像素图块。已另实现验证全部窗口内的位置映射，见[逐场景共同图块证据](逐场景去重与共用图块.json)。共用图块支持同一场景的平移派生关系，但**不自动等于照片拍摄点或地标本体的正确格**；不能单凭文件后缀/图像交集生成正式拍摄点标签。本地元数据字段只有{s['metadata_fields']}，缺少显式目标格与逐格地理变换。地面图中可解析GPS的唯一照片为{s['unique_ground_pixel_images_with_EXIF_GPS']}张（含派生重复记录计{s['ground_records_with_EXIF_GPS']}条），仍不提供点到航拍像素的变换。

原论文描述的Monuments目标含独特建筑、景观或场景，其地面视图与航拍目标配对不能无条件当作严格“拍摄设备所在格”的标签。这种任务定义差异必须在未来协议中处理。更不能把整张1500×1500搜索窗当成一个正确300×300目标格，训练时喂整图、部署时却只给当前格。

## 原MM-GAG哪些可以训练

当前65套静态配对素材全部已有角色：10开发临时组共23来源，20保留组共42来源。原16对覆盖全部10开发组；另外7来源也在这些组中，不能把它们放训练却继续声称原16对来源隔离。**保持原16对开发角色和20保留组保护时，未分配的MM-GAG训练来源为0。** 本轮没有重新划分角色、读取保留图像、训练或编码它们。

按旧库存哈希检查，Swiss辅助图片与MM-GAG原件同字节匹配数为{len(s['cross_dataset_exact_byte_matches'])}；这仅排查字节重复，不能排除地理相邻或不同裁图重叠。由于Swiss Monuments缺少完整可比地理足迹，跨数据集地理隔离仍未知。历史SwissView100账本登记了{s['SwissView100_historical_use']}幅已导航使用影像，但该100子集没有逐条地面配对，不能把它们再计为100个跨视角训练对；也不能据两子集名称不同就证明地理互斥。既有航拍验收不被本轮回写。

## 准入判定与唯一条件式方案

| 层级 | 当前结论 |
|---|---|
| 文件可读、配对与去重工程 | 候选适配后可做小型工程准备，正式路径修复仍须另存映射并核验 |
| 原拍摄点细格任务训练标签 | 未准入；正格对应及拍摄点/可见主体语义未明确验证 |
| 保持原开发/保留角色的MM-GAG训练池 | 0个未分配来源，不自行挪用开发或保留素材 |
| 新来源泛化与论文确认 | 尚不具备；15派生场景与内容去重组不代表可靠地理独立样本规模 |

仅保存一个**条件式**小头构想：冻结现有两个编码器，在512维地面特征上加秩8残差投影`normalize(g+UVg)`，最多8192可训练参数，U初始0保证身份基线。它比直接全512×512映射的参数少；没有证据保证能提高成功率，不能因为参数小就跳过数据门槛。来源/特征未绑定、训练/开发清单为空，**本文件不是已冻结的训练执行协议**。

若后续数据条件通过，可另冻结一次seed0、最多500更新/30分钟、显存≤2GiB、特征缓存≤32MiB、头检查点≤1MiB的工程小试；只保留初始/最终头和一个活动恢复文件。具体损失、源组划分、负样本规则、同题身份基线/真实均值错误对照、恢复损伤和停止门槛必须在训练前确认。原16对如仍用于开发就不能进训练，其历史结果不是新确认。当前没有训练授权消耗或新参数加载，见[条件式方案](条件式小头方案_未准入.json)。

下一项优先是**派生路径适配与目标语义准入**：把候选名称/扩展名映射保存为新清单、核实平移共同格与任务语义、明确训练来源角色。若缺的是同任务样本，可由团队决定另立新数据/用途协议，或保留地面目标为未通过的研究支线；不能自动动用20保留组或把旧开发改名作未见数据。无需联系原作者，不追加大模型或下载数据来绕过这些条件。

## 复核、存储和平台交付

6项路径/25变体归组/连通闭包/角色保护测试通过；另实现重新解码/以PIL裁格重算{a['independently_recomputed_cell_occurrences']}个300px图块出现记录，以union-find复核内容组，并重核输入/源码/旧封存。由同一Agent完成，不称独立人员审查。所有新图像读取均限Swiss辅助子集，MM-GAG仅复用旧元数据/哈希，0新MM-GAG/保留图像读取。

输出只含轻量清单、哈希、统计和报告（≤4MiB），不复制原图/权重/特征。平台侧交付[数据可用性说明页](数据可用性.html)，用于展示来源数量与准入边界；未更改生产GUI或开放地面导航。原M0/模型/默认/Word/PDF/历史验收保持，本轮无新SR、训练或模型调用。

恢复仅仓库根运行`D:/PYTHON/python.exe project/src/eval/audit_crossview_training_inventory_v1.py`（不加--save）。freeze/build/报告生成器已消费，不盲重跑覆盖。
'''
    put('训练素材准入报告.md',text)
    cards=[('原始记录',str(s['swiss_metadata_records']),'15个场景的25个平移窗口，不是375个地点'),('不同地面照片',str(s['unique_ground_pixel_images']),'依据方向矫正后的RGB像素哈希'),('严格路径配对',str(s['strict_pairs_present'])+'/375','原JSON逐条查找，两侧都存在'),('候选适配配对',str(s['candidate_pairs_decodable'])+'/375','候选别名/扩展名映射，不等于标签已验真')]
    put('数据可用性.html','''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>跨视角训练素材准入</title><style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;background:#edf3f7;color:#17384d;margin:0}main{max-width:1100px;margin:40px auto;padding:0 24px}h1{font-size:30px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px}.card,section{background:white;border:1px solid #dce5eb;border-radius:12px;padding:22px;margin:16px 0}.number{font-size:36px;color:#087f8c}.notice{border-left:5px solid #e89a3c;background:#fff4e5}a{color:#087b87}.muted{color:#5e7587}</style><main><h1>跨视角训练素材 · 当前准入</h1><p class="muted">CV-006 · 2026-10-17 · 静态核查，0模型调用、0训练</p><section class="notice"><strong>当前未准入拍摄点细格任务的正式小头训练。</strong><br>文件可用于工程准备；目标格语义、来源角色与地理隔离仍有缺口。小头方案仅为条件式草案。</section><div class="cards">'''+''.join(f'<div class="card"><div>{html.escape(title)}</div><div class="number">{html.escape(value)}</div><p>{html.escape(note)}</p></div>' for title,value,note in cards)+f'''</div><section><h2>保留数据角色</h2><p>MM-GAG：10开发组／23来源＋20保留组／42来源。保持这些角色时，未分配训练来源为0。旧16对不能混进训练后再当独立开发，20保留组继续保护。</p></section><section><h2>下一步</h2><p>先核实候选路径映射、平移共同格及拍摄点语义，再另立训练清单与协议。未训练8192参数的小头，未更换或扩大骨干。</p><p><a href="训练素材准入报告.md">完整报告</a> · <a href="素材核验汇总.json">数据汇总</a> · <a href="条件式小头方案_未准入.json">条件式方案</a> · <a href="候选路径别名记录.json">路径问题</a></p></section></main></html>''')
    put('README.md','''# 小型跨视角对齐头训练素材准入

CV-006 / 2026-10-17。静态盘点完成：现有文件可做受限工程准备；原拍摄点细格任务的训练标签/分组条件尚未满足，当前不启动小头训练。

| 内容 | 入口 |
|---|---|
| 简明可视化说明 | [数据可用性](数据可用性.html) |
| 来源/数量/去重/角色与结论 | [准入报告](训练素材准入报告.md) |
| 原始结果 | [汇总](素材核验汇总.json) / [375条核验](逐对素材核验.json) / [场景共用图块](逐场景去重与共用图块.json) |
| 路径问题与论文来源 | [候选别名，不改原件](候选路径别名记录.json) / [本地论文证据](论文与数据来源证据.json) |
| 冻结/复核/保存 | [范围](冻结静态核查范围.json) / [输入源码SHA](输入与源码绑定.json) / [另实现](核验/另实现复核.json) / [阶段封存](阶段封存.json) |
| 唯一条件式方案，未准入 | [8192参数秩8残差头](条件式小头方案_未准入.json) |

不复制原图/权重/特征，输出≤4MiB；0模型加载/forward、训练、编码、导航、API和下载。MM-GAG只读旧元数据/哈希，20保留组不作训练/选型；原默认/GUI/Word与历史判定保持。这里的说明页没有接入生产平台。

恢复仅运行 `D:/PYTHON/python.exe project/src/eval/audit_crossview_training_inventory_v1.py`（不加--save）。freeze/build/报告生成器已消费，不覆盖重跑。
''')
    print(json.dumps({'reports_written':True,'training_admitted':False,'head_design_parameters_only':8192,'output_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
if __name__=='__main__':main()
