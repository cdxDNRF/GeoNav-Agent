"""CV-009 concise evidence report/static reading page; no production platform changes."""
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from eval.weak_transfer_diagnosis_v1 import OUT,RUN,read,put,entry,verify


def main():
    audit=read(OUT/'核验/另实现复核.json');assert audit['passed']
    ranks=read(OUT/'排序与目标对照重计.json');support=read(OUT/'缓存特征支持度.json')
    qa=read(OUT/'旧主观审查关联.json');weights=read(OUT/'参数与训练记录统计.json')
    binding=read(OUT/'输入与源码绑定.json')
    for k in ['inputs','sources','protected','previous_seals']:verify(binding[k])
    doc_binding=read(OUT/'文档解释补充绑定.json')
    verify(doc_binding['sources']+doc_binding['inputs'])
    source_check=read(OUT/'核验/来源控制另实现重计.json')
    control_flags={ds:{name:p['Rank8Final']['Full']['top1']>max(p['Rank8Final'][c]['top1'] for c in ['Mean','Wrong']) for name,p in ranks[ds]['per_source'].items()} for ds in ['weak_dev','MMGAG']}
    assert control_flags==source_check['flags']
    only_weak=sum(control_flags['weak_dev'].values());only_mm=sum(control_flags['MMGAG'].values())
    names='、'.join(name for name,flag in control_flags['weak_dev'].items() if flag)
    put('来源目标特异性补充.json',{'weak_sources_better_than_both_controls':only_weak,'weak_total_sources':5,
        'MMGAG_groups_better_than_both_controls':only_mm,'MMGAG_total_groups':10,'flags':control_flags,
        'basis':'strict Full Top1 > Mean Top1 AND > Wrong Top1, per existing source; not a new acceptance threshold',
        'interpretation':'4/5 sources improve vs Identity but only2/5 surpass both controls; mean-target successes can explain part of non-specific benefit',
        'new_predictions':0,'new_candidate':False})
    weak=ranks['weak_dev'];mm=ranks['MMGAG'];interval=weak['paired_source_bootstrap']['percentile95']
    table=[]
    for ds,label in [('weak_dev','Swiss弱场景125窗/5查询'),('MMGAG','MM-GAG拍摄点16对/10组')]:
        for arm in ['Identity','Rank8Final']:
            c=ranks[ds]['counts'][arm]
            table.append(f"| {label} | {arm} | {c['Full']['top1']} | {c['Mean']['top1']} | {c['Wrong']['top1']} | {c['Full']['top5']} | {c['Mean']['top5']} | {c['Wrong']['top5']} |")
    sources=[]
    for name,p in weak['per_source'].items():
        b=p['Identity']['Full'];f=p['Rank8Final']['Full'];m=p['Rank8Final']['Mean'];w=p['Rank8Final']['Wrong']
        sources.append(f"| {name} | {b['top1']}/25 | {f['top1']}/25 | {m['top1']}/25 | {w['top1']}/25 | {100*p['Full_top1_rate_gain']:+.0f} |")
    features=[]
    for role,label in [('fit','9fit（近邻排除自身）'),('weak_dev','5弱开发'),('MMGAG','16 MM开发')]:
        s=support['summaries'][role]
        features.append(f"| {label} | {s['nearest_fit_cosine']['mean']:.4f} | {s['distance_to_fit_centroid']['mean']:.4f} | {s['residual_to_fit_affine_span']['mean']:.4f} |")
    qa_table=[]
    for category,arms in qa['by_old_subjective_semantics'].items():
        initial=arms['Identity']['Full'];final=arms['Rank8Final']['Full']
        qa_table.append(f"| {category} | {final['planned']} | {initial['top5']} | {final['top5']} | {arms['Rank8Final']['Mean']['top5']} | {arms['Rank8Final']['Wrong']['top5']} |")
    support_diff=support['summaries']['MMGAG']['nearest_fit_cosine']['mean']-support['summaries']['weak_dev']['nearest_fit_cosine']['mean']
    h=weights['Rank8Final']['residual_operator_singular_values'][0]
    report=f'''# 弱场景收益为什么不能直接当成拍摄点定位收益

CV-009 / 2026-10-17。**既有结果证实弱同景关联改善，没有证实拍摄点目标关联改善。两类标签、来源覆盖与评测重复结构不同；这些是可核实差异，尚不能据此锁定唯一失败原因。** 本轮只是已有数据的描述统计，0训练、模型forward、新预测、导航、原图读取、API、下载及保留组消费。

## 1. 两项任务的正确格不是同一语义

Swiss辅助标签来自发布同景窗口的共同300px图块。每场景一张地面照片、81个唯一航拍格、25个平移窗；每个窗含同一个正格，图中可见地标可与它有语义关联，但相机是否位于该格未经验证。14个场景固定9fit/5弱开发；训练只有9个不同地面查询，并非225个独立训练位置。

MM-GAG保持原CSV/代码中心(5,5)经移窗映射的拍摄点代理格。用户给定照片的可见主体可能位于相机前方，室内主体可能在俯视屋顶下方；不能把主体所在格改成成功格。CV-004已经核过16格映射与裁窗，但16对地理footprint和“相机/地标是否同格”都未独立认证。本轮沿用旧语义，不宣布标签错，也不根据预测重标答案。

因此，“识别与照片同景的显著格”与“找到相机拍摄点格”不能视为已经等价；这也不证明两者必然不重合。现有训练是受限的弱监督迁移，不能称完成MM-GAG拍摄点训练或相关论文完整复现。

## 2. 保存的真实/均值/错误目标对照

| 队列 | 头 | Full Top1 | Mean Top1 | Wrong Top1 | Full Top5 | Mean Top5 | Wrong Top5 |
|---|---|---|---|---|---|---|---|
{chr(10).join(table)}

所有数字来自CV-008封存的846条排序，没有生成新预测。Swiss真实Top1 20→70/125，恢复57/损伤7；Top5 113→115，恢复10/损伤8。改善不是全来源一致：

| 弱开发来源 | 原Full Top1 | Final Full Top1 | Final Mean | Final Wrong | Full变化pp |
|---|---|---|---|---|---|
{chr(10).join(sources)}

MM-GAG Full Top1仍0/16，弱于Mean2/16和Wrong1/16；Full Top5仍4/16，恢复1/损伤1，低于两控制各6/16。真实目标排名中位数14→10.5是排序变化，不能抵消这些目标有效性反证。完整16对/10来源、控制分数变化和Top1/Top5恢复损伤见[重计结果](排序与目标对照重计.json)。

**来源控制的补充反证：** 4/5来源较训练前改善，不等于4/5来源形成目标特异收益。最终Full Top1同时高于Mean和Wrong的只有{only_weak}/5：{names}；MM-GAG为{only_mm}/10。Sion是25/25/0（Full/Mean/Wrong），均值也能全中；Bern为1/6/0；Luzern为2/0/9。说明部分收益与非目标特异的场景偏向相容，但不是已隔离因果或可以只挑两来源宣称有效。该计数是事后诊断，**不新增或回写CV-008门槛**，详见[来源特异性补充](来源目标特异性补充.json)。

## 3. 125窗不是125个独立地理样本

5个弱开发场景各25窗共享一张地面图和同一个正格；每场景所有窗口合计81种航拍格，25窗全交集恰1格。变换只改变同一目标在候选窗内的位置与干扰格，不能扩大独立场景数。等权来源平均Top1增益为+40pp；5^5=3125次完整来源重采样的描述性95%百分位区间为[{100*interval[0]:+.2f},{100*interval[1]:+.2f}]pp。

区间按5场景配对，不按125窗重采样；只有5个已知开发场景且地理隔离未认证，不能把该区间当正式新地理泛化验收。MM-GAG同样只报告10个既有临时来源组，16题/三条件不增加地理样本。原20保留组未使用。

## 4. 缓存特征的支持分布

下表仅描述原始CLIP地面缓存到9fit查询的余弦近邻、质心距离和拟合仿射子空间残差，不计算任何地面→航拍的新排名，也不把头应用到特征。

| 查询集合 | 最近fit余弦均值 | 到fit质心距离均值 | 到fit仿射空间残差均值 |
|---|---|---|---|
{chr(10).join(features)}

MM最近fit余弦均值相对弱开发为{support_diff:+.4f}。这些支持度差异只能说明当前编码空间的分布差异，不能证明模型必然失败、地理距离或具体语义类别错误，也不据此设新阈值。fit仿射空间秩{support['fit_affine_rank']}，fit残差接近0是构造结果，不是泛化证据；fit近邻排除自己，另外两集合在9个fit查询中查找。逐查询值见[支持度](缓存特征支持度.json)。

头仍只有8192参数，残差算子最多秩8，参数最大奇异值{h:.4f}；该统计只用U/V参数，不执行投影。原500步fit损失4.147856→0.080667只说明九查询拟合充分，未记录中间开发曲线，不能推断“再训一倍就会泛化”。弱开发有真实收益，也不支持简单宣布全部收益都是训练记忆。现有证据同样不足以要求扩大骨干。

## 5. 旧视觉审查与MM结果相连

| CV-004旧主观标记 | 题数 | 原Full Top5 | Final Full Top5 | Final Mean Top5 | Final Wrong Top5 |
|---|---|---|---|---|---|
{chr(10).join(qa_table)}

旧审查包含4对plausible、12对not_clear，审查者曾接触早期结果，既不严格盲审也非地理标注真值。本轮只按pair_id关联，不看新图、不增加标注。分组题数少，不能把室内/远景等主观描述当唯一失败原因。逐对原场景文字、前后排名、控制与恢复损伤见[旧审查关联](旧主观审查关联.json)。

## 6. 收束与下一步

**保持CV-008未过结论与停止条件，保留弱同景辅助结果，不升级航拍默认或开放未经验证的地面导航。** 未识别出唯一因果机制；本轮不另选秩/温度/损失/骨干候选，不继续裁剪、填边或相似度扫描。

下一项优先处理**同任务训练数据分工**：评估能否在另案授权后，用主任务同语义的拍摄点代理配对做训练，并保留未消费的独立确认来源。原有16对所在10组继续是已知开发，不能把同组额外7份挪入训练再宣称独立验证。当前20保留组42来源不自动改用途；若团队决定使用，可另拟9fit/5dev/6确认组的元数据方案，先查分组/空间隔离和监督格，再冻结完整来源与工程预算。9/5/6只是与九查询小头预算相容的一种待审方案，不是本轮冻结分配，也不保证数据足够或成绩提高；没有授权/来源证据就不启动。

拍摄点代理标签与已测绘坐标必须分别命名：若缺逐格地理变换，不声称真实米制拍摄点精度；可先沿用明确披露的原代码代理任务做受限工程，但不能把同景地标标签换名当拍摄点。目标是补齐任务一致性证据，不能靠重标答案或扩参绕过问题。具体待核条件见[下一项资料准入建议](下一项同任务来源分工与资料准入建议.md)，本轮没有开始该分工或新实验。

## 验证与平台交付

另实现以不同计数逻辑、索引重采样、最小二乘仿射残差与完整参数矩阵SVD复核，含846排序/21150已保存分数、14场景重复结构、16旧QA关联与输入/源码/原件保护；同一Agent另一实现，不称独立人员审查。无推理/图像/网络框架导入和调用边界另作AST核验，测试仅内存合成。

平台侧提供可直接打开的[离线诊断阅读页](诊断阅读页.html)及可复用小型JSON结果；它不是生产GUI新增导航模式，不接触Agent输入。M0/V4.5、旧模型/默认/Word和历史验收保持；不新增大型缓存或图像副本，总输出≤4MiB。

只读复核：`D:/PYTHON/python.exe project/src/eval/audit_weak_transfer_diagnosis_v1.py`（不加--save）。freeze/build/收尾生成器已消费，不盲重跑；无本任务活动训练/导航进程。
'''
    put('迁移差异诊断报告.md',report)
    put('下一项同任务来源分工与资料准入建议.md','''# 下一项建议：先审同任务来源分工，不启动新训练

状态：建议/未冻结/未执行；不是20保留组用途变更授权。

1. 研究目标继续是地面照片给定的拍摄点代理格。Swiss共同格辅助训练不再追加更新；旧16对及所在10组保持已知开发。
2. 若团队希望验证“同任务监督是否改善迁移”，先只读检查尚未消费来源的分组与标签资料。原20保留组42来源继续保留；任何重新分配须另案明确授权并登记不可逆用途变化，未授权不看新像素或模型分数。
3. 9fit/5开发/6确认组可作为一种元数据规划备选，以维持9个不同训练查询和轻量预算；未选定任何具体来源，地理隔离/样本适用性尚未验收，不应把20临时组数量等同20个独立地区。
4. 每条拟训练资料应能绑定照片、对应航拍来源、拍摄点格映射、坐标/像素约定与未知项。代理格与测绘真值分开命名；缺地理变换时不声称米制定位。可见地标只作为线索，不能按模型排名改答案。
5. 数据准入通过后才冻结唯一因素与精确训练/开发/确认清单、权重/特征、同题基线、Full/Mean/Wrong/必要禁用控制、Top-k或导航指标、恢复损伤/来源区间、资源和停止条件。沿用冻结骨干/小头是优先方向，不自动扩大参数；没有新实验协议不沿用任意SR60门槛。

本轮不选新模型候选、不分配来源、不训练/推理/导航/下载/API；不要求原作者提供私有资料。来源或标签证据不足时明确保留限制，不继续测试追分。
''')
    cards=''.join(f'<tr><td>{html.escape(name)}</td><td>{p["Identity"]["Full"]["top1"]}/25</td><td>{p["Rank8Final"]["Full"]["top1"]}/25</td><td>{100*p["Full_top1_rate_gain"]:+.0f}pp</td></tr>' for name,p in weak['per_source'].items())
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CV-009 · 弱场景与拍摄点</title><style>
body{margin:0;background:#edf3f7;color:#19364a;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1080px;margin:36px auto;padding:0 22px}h1{font-size:30px;line-height:1.4}h2{font-size:21px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card,section{border:1px solid #d4e0e7;background:#fff;border-radius:12px;padding:22px;margin-bottom:20px}.tag{color:#0d7180;font-weight:bold;font-size:13px;letter-spacing:1px}.big{font-size:35px;line-height:1.3;font-weight:bold;color:#107582}.warn{border-left:5px solid #d39037;background:#fff9ed}a{color:#087b8a}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #e4edf2}small{color:#607b8c}nav{display:flex;flex-wrap:wrap;gap:16px;margin:20px 0}@media(max-width:700px){.grid{grid-template-columns:1fr}h1{font-size:25px}.big{font-size:29px}}</style><main><div class="tag">GEONAV · 既有证据诊断 · CV-009</div><h1>同景关联改善，还没有成为拍摄点定位能力</h1><nav><a href="迁移差异诊断报告.md">完整报告</a><a href="排序与目标对照重计.json">排序与控制</a><a href="核验/另实现复核.json">另实现复核</a></nav><div class="grid"><article class="card"><h2>Swiss弱场景</h2><div class="big">16% → 56%</div><p>Top1：20 → 70 / 125窗口<br>5张不同地面照片 · 4/5场景改善</p><small>每场景25窗重复包含同一个弱正格；不是125个独立地点。</small></article><article class="card"><h2>MM-GAG拍摄点代理任务</h2><div class="big">0 / 16 → 0 / 16</div><p>最终Full / Mean / Wrong Top1：0 / 2 / 1<br>目标有效性门槛未过</p><small>Top5仍4/16，恢复1、损伤1。不是导航SR。</small></article></div><section class="warn"><strong>本轮结论：保留收益，停止该因素追分。</strong><p>4/5来源较前改善，但只有2/5真实目标同时胜过均值和错误目标；Sion均值也25/25。弱共同格不等于已验证的相机所在格。分布、任务与来源覆盖存在差异，但尚未分离唯一因果；不宣布标签错误或模型容量不足。</p></section><section><h2>弱开发的逐来源变化</h2><table><tr><th>来源</th><th>原Full</th><th>Final Full</th><th>变化</th></tr>'''+cards+'''</table></section><section><h2>后续只先审同任务的数据分工</h2><p>原16对保持已知开发，20保留组未消费；未经另案授权不改角色。优先核拍摄点代理监督与来源隔离，不扩大骨干或追加训练。</p><a href="下一项同任务来源分工与资料准入建议.md">资料准入建议（未执行）</a> · <a href="缓存特征支持度.json">原缓存分布支持度</a> · <a href="旧主观审查关联.json">旧主观QA关联</a></section><p><small>0训练 · 0模型forward · 0新预测 · 0原图读取 · 0导航/API/下载。静态证据阅读页，不是生产导航界面；同Agent另一实现复核，非独立人员审查。</small></p></main></html>'''
    put('诊断阅读页.html',page)
    put('README.md','''# 弱场景与拍摄点迁移差异诊断

CV-009。只用已有结果/缓存/元数据；弱关联收益成立，MM-GAG拍摄点迁移未过的结论保持，没有识别唯一失败原因或选择新模型因素。

- [完整诊断](迁移差异诊断报告.md) / [静态阅读页](诊断阅读页.html)
- [排序/目标对照/恢复损伤/来源区间](排序与目标对照重计.json) / [重复结构](来源与目标重复结构.json)
- [缓存支持度](缓存特征支持度.json) / [参数与原训练日志统计](参数与训练记录统计.json) / [旧主观QA关联](旧主观审查关联.json)
- [来源目标特异性：只有2/5同时胜控制](来源目标特异性补充.json) / [补充另实现计数](核验/来源控制另实现重计.json)
- [标签与证据边界](目标语义与证据边界.json) / [下一项资料准入建议，未执行](下一项同任务来源分工与资料准入建议.md)
- [冻结范围](冻结诊断范围.json) / [输入源码绑定](输入与源码绑定.json) / [另实现审计](核验/另实现复核.json) / [封存](阶段封存.json)

0训练/模型forward/新预测/原图读取/导航/API/下载/保留消费。旧默认、模型、Word、GUI保持；不把Top-k当导航SR，不把125窗当125个独立地点，总≤4MiB且无原图/特征/模型副本。

仓库根只读复核：`D:/PYTHON/python.exe project/src/eval/audit_weak_transfer_diagnosis_v1.py`（不加--save）。本批freeze/build/报告生成已消费，不能盲重跑覆盖。
''')
    files=[entry(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['README.md','阶段封存.json']]
    put('阶段封存.json',{'task':'CV-009','files':files,'sources':binding['sources']+doc_binding['sources'],'sealed_bytes':sum(x['bytes'] for x in files),
        'mutable_entries_excluded':['README.md'],'default_upgraded':False,'model_forward_calls':0,'new_predictions':0,
        'recovery':'read-only audit only; freeze/build/close consumed'})
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=4*1024**2
    print(json.dumps({'reports_complete':True,'files':len(files),'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'new_model_factor':False}),flush=True)


if __name__=='__main__':main()
