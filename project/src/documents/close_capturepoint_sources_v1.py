"""CV-010 report/nonexecutable source proposal; never activates roles or experiments."""
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from tools.plan_capturepoint_sources_v1 import OUT,read,put,entry,verify


def main():
    audit=read(OUT/'核验/另实现复核.json');assert audit['passed']
    summary=read(OUT/'规划汇总.json');draft=read(OUT/'待审九五六分组.json')
    refs=read(OUT/'候选代表与监督引用.json');graph=read(OUT/'坐标与旧哈希分组复核.json')
    binding=read(OUT/'输入与源码绑定.json')
    for k in ['inputs','sources','protected','previous_seals']:verify(binding[k])
    role_names={'fit':'拟训练','development':'拟开发','confirmation':'拟确认'}
    capacity=[]
    for role,c in draft['proposed_counts'].items():
        capacity.append(f"| {role_names[role]} | {c['groups']} | {c['member_sources']} | {c['representative_queries']} | 原reserved，未激活 |")
    groups=[]
    for g in draft['groups']:
        r=next(x for x in refs['representatives'] if x['proposal_group_id']==g['proposal_group_id'])
        groups.append(f"| {role_names[g['proposed_role']]} | {g['proposal_group_id']} | {', '.join(g['members'])} | {g['representative_source_id']} | {'有旧拼图引用' if r['stitched_reference_inventory_paths'] else '无旧完整拼图引用'} |")
    minimum=graph['independence']['all_cross_group']['minimum_camera_distance_m']
    distances=[]
    for name,r in graph['independence']['proposed_role_pairs'].items():
        d=r['minimum_camera_distance_m']
        distances.append(f"| {name} | {r['source_pairs']} | {d:.2f} | {r['shared_aerial_byte_hash_pairs']} |")
    report=f'''# 同拍摄点任务的数据与来源规划

CV-010 / 2026-10-17。**元数据层面已有一份具体、可审阅的9训练/5开发/6确认组草案，尚未更改角色，也不具备立即训练的工程准入。** 原20保留组42套素材保持reserved；本轮0原图读取/模型forward/训练/新预测/导航/API/下载，没有新SR或默认升级。

## 数据是否够安排一次小型同任务试验

旧库存82条来源中65套具有地面照、CSV相机坐标和完整11×11航拍图块引用，其他17条不进入本草案。65套按相机坐标≤3000米、共享相同航拍字节或地面字节的连通分组，另实现重建恰为原30组；其中10组23套属于已知开发，20组42套仍保留。原16题覆盖了全部10开发组，剩余7套在同组，不能挪去训练后继续称原16独立验证。

数量上可从20组各固定1个代表查询，形成如下**影子规划**。同组其它素材跟随拟用途但本草案不把它们增加成训练查询，也不跨用途拆组。活动角色账本/原划分文件没有改动。

| 草案用途 | 组数 | 组内全部素材数 | 拟用不同查询数 | 当前角色 |
|---|---|---|---|---|
{chr(10).join(capacity)}

分组排序只用旧地面文件SHA，前9/次5/末6；组内代表也按固定SHA排序，不根据照片内容或模型成绩挑选。本轮只有这一份候选，不搜索多个分组方案。整组18套之类的来源不会因照片多而占18个独立样本。

## 具体待审组与代表

| 拟用途 | 草案组ID | 组成员 | 固定代表 | 旧轴向核验素材 |
|---|---|---|---|---|
{chr(10).join(groups)}

完整[分组草案](待审九五六分组.json)和[监督路径/旧SHA引用](候选代表与监督引用.json)可直接检查。所有条目current_role=reserved、activated=false、training_executable=false；这是元数据规划，不是训练任务清单或用途变更授权。

## 隔离证据到哪一步

独立重算65来源共2080对的球面相机距离及旧字节哈希，重建30连通组；跨组最小相机距离{minimum:.2f}米，跨组相同地面字节/航拍字节对均为0，没有相机间距≤3000米的跨组对。更严格的地面字节重复合并没有改变原分组。

| 既有/拟用途组合 | 来源对数 | 最小相机间距m | 共享航拍字节对 |
|---|---|---|---|
{chr(10).join(distances)}

这些是**相机坐标与已存字节哈希的保护证据**，不是影像真实footprint不重叠的证明。不同压缩/裁切但同内容的图块未在本轮重解码；相机位置不等于航拍窗中心的测绘证明，像素地理变换/朝向与编码器预训练地理覆盖仍未知。不把20个临时组直接称20个独立地区，也不将上述相机距离当导航SG。

## 同任务监督格与工程缺项

目标保持MM-GAG的CSV/原代码拍摄点代理格，不再用Swiss发布共同格替代。每个拟代表都能从旧库存引用`source/sat_5_5.jpg`；9×9候选范围为文件索引1…9，中心在81格排序的40号。所有20代表×25个假想5×5目标槽位均按两种文件轴约定核对：代理中心落在给定槽位，500个静态映射通过。只创建引用，没有读取任何图块。

9×9完整候选集及中心标签对行/列交换不变，因此可先做受限关联数据准备；**带偏移的窗口和导航方向仍须逐来源验证轴向**。原16来源的轴向核验不自动覆盖这些新代表：本草案{summary['representative_sources_with_prior_axis_verification']}/20已有该核验；{summary['representatives_with_old_stitch_available']}/20在旧库存有完整拼图引用，但“有文件引用”不等于轴向通过或当前字节未变。缺完整拼图的来源不能用整组其它图随意替换代表追过线；需另冻明确工程路径，或者判该候选不准入。

旧库存的解码/图块大小和SHA只提供可追溯引用，本轮没有复验原文件。后续先核原字节/EXIF与质量、检查相机代理约定和必要拼图轴向；没有像素到GPS变换时仍标明proxy，不声称测绘米制定位。照片中可见地标只是线索，不自动改成它所在格为正确答案。

## 轻量预算可以保持，当前未授权运行

拟9个训练查询×81候选、5开发查询×81格，合计14地面+1134航拍=1148编码输入、batch8最多144次编码器调用，和CV-008开发试验的编码上限相同。可优先保留8192参数头/冻结原骨干/seed0/500更新/4500监督使用作资源参考；**训练执行器、实际特征与效果门槛均未在本轮冻结**。改变数据域与同任务监督同时发生，未来即便改善也不能单独归因于标签语义。

6确认组若以后单独正式验证，另需6地面+486航拍=492输入/62调用；它们不在前14个工程/开发来源内，未授权不看新像素或成绩。原16题继续保持已知开发，不能替代这6组的新确认。所有数字只是计划，未新增编码、参数、优化器或预测。

## 下一步只做工程预检的具体建议

需先明确批准是否采用这份具体9/5/6分组，才可将9组转训练准备、5组转开发，6组继续确认保留。CV-009资料建议明确要求“任何重新分配须另案明确授权并登记不可逆用途变化”；本轮继续审查不自动视为批准。

批准后下一项仅对拟9训练+5开发的14个代表核原始字节、EXIF/解码、候选图块质量、必要拼图轴向和输入隔离，先不执行模型forward或训练。可将最多14地面+1134候选格+14参考拼图=1162图像解码作为工程草案上限，实际清单和源码仍须新批次冻结；衍生缩略图只按必要最小量保留。任一必要条件失败则保留失败，不换代表/源/阈值追过线。工程通过后再冻结单一同任务对齐训练与同题目标对照，不能让像素工程通过冒充目标有效性通过。

如团队决定维持20组全部保留，则此草案不激活，应另找有相同拍摄点标签的训练来源，不将已知开发/Swiss共同格重命名为独立同任务训练数据。不需要原作者私有资料，也不在本轮下载新数据。

## 验证和平台交付

8项内存测试覆盖坐标/跨经线/连通闭包/共享字节去重、非激活分组、完整数字网格、双轴25槽位中心映射和禁止原始字节读取/推理框架导入。另实现用3D单位球夹角、union-find、直接文件名引用复核65来源/2080对/30组/20代表/1620图块引用/500目标映射、原角色和资源分母；同Agent另一实现，非独立人员审查。

平台侧新增[可直接打开的待审来源页](来源规划阅读页.html)，帮助检查具体组、代表、门槛与授权状态；生产GUI/运行时数据登记/实际角色账本未改，不能从该静态页直接启动训练。当前M0/V4.5、Word和历史模型/正负验收保持。总输出≤3MiB，不复制原图/特征/模型。

只读恢复：`D:/PYTHON/python.exe project/src/eval/audit_capturepoint_sources_v1.py`（不加--save）。freeze/build/收尾已消费，禁止盲重跑覆盖；原保留用途尚未调整。
'''
    put('同任务来源与监督资料规划报告.md',report)
    approvals={'task':'CV-010','status':'proposal_reviewable_not_activated','role_change_requires_explicit_approval':True,
        'current_reserved_groups':20,'current_reserved_sources':42,'actual_roles_changed':0,
        'proposal_binding':entry(OUT/'待审九五六分组.json'),'references_binding':entry(OUT/'候选代表与监督引用.json'),
        'request_scope':'approve9fit-preparation+5development groups;6confirmation groups remain withheld; next only14representative pixel/axis engineering, no forward/training',
        'engineering_draft_cap_only':{'representatives':14,'image_decodes_maximum':1162,'encoder_forward_calls':0,'training_steps':0},
        'requirements_origin':'CV-009 下一项同任务来源分工与资料准入建议.md item2; no automatic reserved-role reassignment',
        'approval_received':False,'execution_started':False}
    put('待审用途调整与工程范围.json',approvals)
    rows=''.join('<tr><td>'+html.escape(role_names[g['proposed_role']])+'</td><td>'+html.escape(g['representative_source_id'])+'</td><td>'+str(len(g['members']))+'</td><td>reserved · 未激活</td></tr>' for g in draft['groups'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CV-010 · 同拍摄点来源规划</title><style>body{margin:0;background:#edf3f7;color:#18374b;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1060px;margin:32px auto;padding:0 22px}section{background:white;border:1px solid #d3e0e8;border-radius:12px;padding:22px;margin:18px 0}h1{font-size:29px;line-height:1.4}h2{font-size:21px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metric{background:#e5f1f1;border-radius:8px;padding:15px;color:#116b77}.metric b{display:block;font-size:34px}.warn{border-left:5px solid #d18e31;background:#fff9ed}table{border-collapse:collapse;width:100%}th,td{padding:8px;text-align:left;border-bottom:1px solid #e2eaf0}a{color:#087b88}small{color:#5e7c8f}@media(max-width:650px){.grid{grid-template-columns:1fr}h1{font-size:25px}}</style><main><small>GEONAV · CV-010 · 仅元数据待审方案</small><h1>先对齐拍摄点任务，再安排轻量训练</h1><section class="warn"><strong>20组仍全部保留，草案未激活。</strong><p>当前仅检查旧库存/坐标/哈希。没有原图读取、模型forward、训练或新成绩；不能用本页启动实验。</p><a href="同任务来源与监督资料规划报告.md">完整报告</a> · <a href="待审用途调整与工程范围.json">具体待批工程范围</a></section><section><h2>一份固定的来源分工草案</h2><div class="grid"><div class="metric"><b>9组</b>拟训练准备</div><div class="metric"><b>5组</b>拟开发</div><div class="metric"><b>6组</b>拟确认 · 继续封存</div></div><p>整组划分，每组1个按SHA选出的代表。原16题所在10组继续是已知开发；不能拿其同组其它图训练后声称独立验证。</p><a href="待审九五六分组.json">完整组成员</a> · <a href="候选代表与监督引用.json">具体文件与旧SHA</a></section><section><h2>固定代表与当前状态</h2><table><tr><th>拟用途</th><th>代表</th><th>同组素材数</th><th>当前状态</th></tr>'''+rows+'''</table></section><section><h2>什么已经核对，什么还没通过</h2><p>已重建65来源/2080对/30组；跨组无已知相同字节，坐标距离保护通过。它不等于真实航拍footprint不重叠。拍摄点仍是代码代理标签，不冒充测绘真值。</p><p>9×9候选中心为40号；500个静态目标映射已核。新20代表的逐来源方向核验仍待完成，旧16来源的轴向证据不能自动继承。</p><a href="核验/另实现复核.json">另实现证据</a> · <a href="后续工程准入条件.json">工程缺项与资源参考</a></section><section class="warn"><h2>下一项范围很有限</h2><p>得到明确用途调整授权后，先检查14代表的原图/质量/轴向，不训练、不推理；6确认组继续不看。相同8192参数头/500更新只作资源参考，正式训练协议和效果门槛尚未冻结。</p></section><small>静态审查页 · 当前M0/生产GUI/Word保持 · 无新导航SR</small></main></html>'''
    put('来源规划阅读页.html',page)
    put('README.md','''# 同任务来源与监督资料规划

CV-010。元数据规划完成，具体9fit/5开发/6确认影子分组可审阅；实际20保留组角色未改，未读取新图像或启动模型/训练。

- [报告](同任务来源与监督资料规划报告.md) / [来源阅读页](来源规划阅读页.html)
- [具体分组，未激活](待审九五六分组.json) / [代表和监督引用](候选代表与监督引用.json)
- [当前角色](当前角色与候选容量.json) / [2080坐标/旧哈希对及分组](坐标与旧哈希分组复核.json)
- [待审用途调整与工程范围](待审用途调整与工程范围.json) / [后续准入条件](后续工程准入条件.json)
- [冻结范围](冻结元数据规划范围.json) / [输入源码绑定](输入与源码绑定.json) / [另实现复核](核验/另实现复核.json) / [封存](阶段封存.json)

前提：同任务拍摄点代理，不把地标当拍摄点；组距/旧字节不等于已测绘影像隔离。原16/10组开发用途保持；原20/42来源仍reserved，重新分配须另案明确授权。0原图/模型forward/训练/预测/导航/API/下载/保留消费，默认/GUI/Word不改，总≤3MiB，无数据/模型副本。

仓库根只读核验：`D:/PYTHON/python.exe project/src/eval/audit_capturepoint_sources_v1.py`（不加--save）。freeze/build/报告已消费，不盲重跑覆盖。
''')
    files=[entry(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['README.md','阶段封存.json']]
    put('阶段封存.json',{'task':'CV-010','files':files,'sources':binding['sources'],'sealed_bytes':sum(e['bytes'] for e in files),
        'mutable_entries_excluded':['README.md'],'roles_changed':False,'model_calls':0,'raw_image_reads':0,
        'recovery':'read-only audit; freeze/build/close consumed, proposal not activation'})
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert total<=3*1024**2
    print(json.dumps({'report_complete':True,'sealed_files':len(files),'bytes':total,'roles_activated':0,'awaiting_explicit_approval':True}),flush=True)


if __name__=='__main__':main()
