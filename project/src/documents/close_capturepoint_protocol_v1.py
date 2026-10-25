"""CV-012 readable protocol and release seal; no experiment runner."""
import html,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from tools.freeze_capturepoint_alignment_v1 import OUT,read,put,entry,verify


def main():
    audit=read(OUT/'核验/另实现复核.json');assert audit['passed']
    b=read(OUT/'输入与协议源码绑定.json')
    for k in ['inputs','sources','reference_math_sources_only','protected','previous_seals']:verify(b[k])
    p=read(OUT/'同任务对齐训练冻结协议.json');q=read(OUT/'固定编码与训练开发队列.json')
    pre=read(ROOT/'DATA/processed_data/MM-GAG/同任务首批图像轴向预检_v1/首像素冻结协议.json')['sources']
    sources='\n'.join(f"| {'训练' if s['role']=='fit' else '开发'} | {s['group']} | {s['source_id']} | 1 | {'81候选/1中心代理标签' if s['role']=='fit' else '25平移窗/同一中心代理格'} |" for s in pre)
    report=f'''# 同任务拍摄点代理对齐：训练与目标对照冻结协议

CV-012 / 2026-10-18。**研究设计、任务清单和开发验收门槛已冻结；实际执行器/特征尚未绑定，本轮没有模型推理或训练。** 6确认组继续封存；本协议只研究受限离线关联，Top1不是导航SR。

## 目的与唯一研究因素

检验在同一拍摄点代理任务上训练8192参数秩8目标残差头，能否在来源分开的开发队列中，优于身份映射和既有冻结Swiss训练头。头仍为 `h(g)=normalize(g+UVg)`，U512×8、V8×512，无bias；输入只含归一化512维地面特征，原CLIP视觉编码器/Sat2Cap及PIL处理冻结。

唯一比较因素定义为**训练数据方案**：9个MM-GAG拍摄点代理配对替代9个Swiss共同场景配对，模型/损失/训练预算保持。该改变同时涉及数据域和监督语义，不能把未来差异单独归因于“标签语义修正”。MM标签沿原CSV/代码中心约定，不是独立测绘相机像素，不把可见地标替换为正确格。

## 已固定来源与队列

| 用途 | 临时来源组 | 固定代表 | 不同查询数 | 监督/评测结构 |
|---|---|---|---|---|
{sources}

9训练来源各1地面查询，对应文件索引1…9的81格、中心sat_5_5为40号。5开发来源各1查询，固定25个平移窗覆盖所有5×5目标槽位，共125窗；同场景始终是同一真实中心格，不算125个独立地理样本。5来源才是区间的单位，地理footprint隔离仍未独立认证。

顺序/代表沿CV-010批准清单，14来源均已完成CV-011像素轴向工程，已有工程消费但尚无这5开发来源的模型成绩。新队列只包含14代表，不读其同组其它4素材、原16已知开发或6确认组24素材；旧16失败结论保持，不额外重跑。目标顺序在首模型调用前按固定SHA排列，pair_id仅不含答案的顺序号，源名/坐标/正确格不进头输入。以[1148编码引用和125开发任务](固定编码与训练开发队列.json)的精确分母为准。

## 三臂与四类目标条件

| 方案 | 权重/训练来源 | 当前状态 |
|---|---|---|
| Identity | CV-008保留初值U0；不做目标适配 | 已有权重SHA冻结 |
| SwissRank8Frozen | CV-008最终500更新，9个Swiss弱场景查询 | 已有权重SHA冻结，不复活为默认、不再训练 |
| MMRank8Final | 新同任务头，必须从seed0身份初值重新训练一次500更新 | 权重尚不存在，后续完成才绑定SHA |

三臂使用同一125窗和同一目标条件：Full为真实目标；Mean固定为9个MM训练目标归一化特征的均值，再归一化后传入各臂；Wrong为5开发来源间固定的下一来源循环，25窗复用该错误目标。不能用新开发/确认目标或旧16均值来做训练域Mean控制。

另有**共享Disabled目标基线**：每窗25个常数0分数，按原CV-002哈希规则破平局，不调用头，也不把零向量传给需要单位向量的头。禁用目标不重复伪造三份独立证据。故计划排序记录为125×(3臂×3条件+1共享禁用)=**1250条**，不是导航轨迹。

## 训练与资源上限

固定seed0，U全0、V为Normal(0,0.02)，仅头8192参数可训练。9查询全批，每查询81候选；温度0.1的对比交叉熵，AdamW lr0.001、weight_decay0.0001、梯度范数裁到1、无调度，**恰500更新/4500监督使用，只评最终权重**。不续接Swiss权重/优化器，不用开发集挑epoch，不继续扫秩/温度/损失/来源。邻近图块可能具有相同语义，是已披露的弱负样本风险，不在看到成绩后改标签或屏蔽负例。

编码最多14地面+1134航拍=1148输入、batch8最多144次编码器forward；两骨干顺序执行，启动空闲GPU≥2560MiB用CUDA，否则CPU，不停其它程序，CUDA分配≤2GiB。头在CPU训练≤30分钟。特征单份≤32MiB，初始/最终头和一个活动恢复≤1MiB，整执行批次≤40MiB；不复制骨干/原图，不堆多个优化器。

工程子集包含在上述1148/144内，成功特征必须复用；禁止因日志中断盲重编码。编码器张量SHA/无梯度与首批身份复现证据及时逐编码器保存，使用追加事件及覆盖记录，避免重复CV-008旧日志缺项。实际执行/提取/训练/审计源码须首forward前绑定，真实特征SHA和fit专用张量须首更新前绑定。

## 开发验收：全部必要项通过才保留候选

| 必要条件 | 固定门槛 |
|---|---|
| 完成及工程证据 | 1250计划记录完整，身份/梯度/恢复/编码器冻结/权限和另实现复核全部过 |
| 绝对关联能力 | MM最终Full Top1至少32/125（25%向上取整后25.6%） |
| 目标作用 | Full比Mean、Wrong、Disabled各至少多16/125（12.5pp向上取整为12.8pp） |
| 新训练数据方案收益 | Full比Identity和Swiss冻结头**各**至少多5/125，即+4pp |
| 来源上的目标作用 | 至少3/5来源的Full Top1严格高于本臂Mean和Wrong两者 |
| 来源上的方案收益 | 至少3/5来源的Full Top1严格高于Identity和Swiss冻结头两者 |

绝对25%及目标差12.5pp延续原4/16与2/16的探索尺度；+4pp参考既有弱开发的有限增益门槛；3/5来源条件针对CV-009只2/5胜双控制的缺口。它们是**本批开发关联约定**，不是论文验收线或任意SR60通行证，不能在看到成绩后放宽。

Top5、真实排名、Full相对三控制及新头相对两基线的Top1/Top5恢复与损伤、全部逐来源结果必须报告。5来源的3125个完整配对bootstrap重采样区间也必须报告，但不再追加未声明的CI门槛；5个开发来源和单一种子不足以宣称稳定新地理泛化。不能只报最好来源、净增或训练loss。

**任一必要项未过即关闭本数据方案因素，保留失败；不追加更新、换来源或改超参追过线。** 全过也只得到开发候选，不升级航拍M0/地面GUI或宣称导航成功。6确认组未来还需独立冻结确认协议，mmgag_011缺完整拼图的已知缺项必须先有明确工程路径，不能在看到开发成绩后随便替换它。

## 当前准入与交付

research_design_frozen=true；execution_admitted=false、training_authorized=false、actual_executor_binding=null、actual_feature_binding=null，MM最终权重也未生成。当前旧数学模块可以作为后续只读依赖，但不能直接执行已消费的CV-008 runner/generator冒充新执行器。

本轮9项内存合成边界测试验证32取整、控制净差16、基线净差5、至少3来源、完整分母/复核前提、非法计数拒绝、125窗口标签/不透明ID与零原图/推理导入。另实现按文件名独立重建125窗/3125候选引用、角色/模型/损失预算和整数门槛；同Agent另一实现，不是独立人员审查。当前0原图读取/模型加载或forward/训练/预测/导航/API/下载，6确认像素0。

平台侧提供[协议阅读页](协议阅读页.html)和可供执行器读取的固定队列/门槛函数，生产GUI不改，未挂出新的运行按钮。输出≤3MiB，原角色/默认/模型/Word与历史正负验收保持。下一项先实现实际队列适配/执行器和运行准入，不先扩大参数；模型运行需相应授权和工程完成后的源码/特征绑定。

只读复核：`D:/PYTHON/python.exe project/src/eval/audit_capturepoint_protocol_v1.py`（不加--save）。冻结/文档生成已消费，不能覆盖重跑；未来真正执行另建批次。
'''
    put('同任务训练与目标对照协议说明.md',report)
    rows=''.join(f'<tr><td>{html.escape(s["role"])}</td><td>{html.escape(s["source_id"])}</td><td>{html.escape(s["group"])}</td></tr>' for s in pre)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CV-012 · 同任务对齐协议</title><style>body{margin:0;background:#edf3f7;color:#19384b;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1060px;margin:32px auto;padding:0 22px}h1{font-size:28px}h2{font-size:21px}section{border:1px solid #d3e0e6;border-radius:12px;background:#fff;margin:18px 0;padding:22px}.warn{background:#fff9ed;border-left:5px solid #ce8c30}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metric{background:#e5f1f0;border-radius:8px;padding:14px}.metric b{display:block;color:#0d7280;font-size:30px}a{color:#087c8a}th,td{text-align:left;border-bottom:1px solid #e0e9ee;padding:8px}table{width:100%;border-collapse:collapse}small{color:#5f7a8a}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style><main><small>GEONAV · CV-012 · 研究协议冻结</small><h1>同拍摄点代理任务：先固定证据，再训练一次</h1><section class="warn"><b>协议已冻结，执行尚未准入。</b><p>没有模型推理或训练，没有新Top1/SR。6确认组继续封存；实际执行器与特征还须绑定。</p><a href="同任务训练与目标对照协议说明.md">完整说明</a> · <a href="同任务对齐训练冻结协议.json">机器协议</a></section><section><div class="grid"><div class="metric"><b>8192</b>可训练参数 · 骨干冻结</div><div class="metric"><b>500</b>固定更新 · 仅评最终权重</div><div class="metric"><b>1250</b>计划排序 · 5不同开发来源</div></div></section><section><h2>同题比较与目标作用</h2><p>Identity、Swiss冻结头、MM新头使用同一125窗和Full/Mean/Wrong；另设共享Disabled目标基线。Mean只能来自9训练目标，Wrong固定跨开发来源。</p><p>MM真实Top1≥32/125；比各目标控制多≥16题；比两基线各多≥5题；至少3/5来源同时胜双控制，至少3/5同时胜双基线。任何必要项失败即收束，不换参数追分。</p><small>这是关联开发门槛，不是导航SR、论文通行线或6来源独立确认。</small></section><section><h2>固定来源</h2><table><tr><th>用途</th><th>代表</th><th>临时来源组</th></tr>'''+rows+'''</table><a href="固定编码与训练开发队列.json">完整引用与125任务</a></section><section><h2>资源与审计</h2><p>最多1148图/144编码调用，CUDA≤2GiB；单份缓存、初始/最终头和一个恢复状态。先记源码，后记真实特征SHA，再首更新；使用追加事件，复用成功缓存。</p><a href="核验/另实现复核.json">另实现协议核验</a> · <a href="输入与协议源码绑定.json">输入/数学依赖绑定</a></section><p><small>0新原图 · 0模型调用 · 0训练 · 0预测 · 0导航/API/下载 · 默认/GUI/Word保持</small></p></main></html>'''
    put('协议阅读页.html',page)
    put('README.md','''# 同任务拍摄点代理对齐协议

CV-012 / 2026-10-18。研究设计和125开发窗口/1250计划排序已冻结；实际执行器/特征未绑定，未授权或启动训练/推理，6确认组继续封存。

- [协议说明](同任务训练与目标对照协议说明.md) / [静态阅读页](协议阅读页.html)
- [冻结协议与门槛](同任务对齐训练冻结协议.json) / [固定编码/训练/开发队列](固定编码与训练开发队列.json)
- [输入/新协议源码与旧数学依赖](输入与协议源码绑定.json) / [另实现核验](核验/另实现复核.json) / [封存](阶段封存.json)

参考资源保持8192参数/9训练查询81候选/500更新/1148编码输入、144编码调用。三臂同题，Full/Mean/Wrong与共享禁用目标；门槛全部通过才是开发候选，不是导航验收或默认升级。

本批0原图读取/模型forward/训练/新预测/导航/API/下载，旧角色/模型/默认/GUI/Word保持，总≤3MiB，无原图/特征/模型副本。只读复核：`D:/PYTHON/python.exe project/src/eval/audit_capturepoint_protocol_v1.py`（不加--save）；本批冻结/文档生成已消费，实际执行另建批次。
''')
    files=[entry(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['README.md','阶段封存.json']]
    put('阶段封存.json',{'task':'CV-012','date':'2026-10-18','files':files,'sources':b['sources'],
        'sealed_bytes':sum(f['bytes'] for f in files),'mutable_entries_excluded':['README.md'],
        'research_design_frozen':True,'execution_admitted':False,'training_started':False,'model_calls':0})
    size=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file());assert size<=3*1024**2
    print(json.dumps({'report_complete':True,'files':len(files),'bytes':size,'training_started':False}),flush=True)
if __name__=='__main__':main()
