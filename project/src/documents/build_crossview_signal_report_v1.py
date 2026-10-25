"""Render CV-021 report and PLATFORM-013 self-contained evidence page once."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'DATA/processed_data/MM-GAG/场景与精确格信号分解_v1'
PAGE=ROOT/'平台/实验工作台_v4/地面目标证据_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,text):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

def main():
    if (OUT/'场景与精确格诊断报告.md').exists() or PAGE.exists():raise FileExistsError('Reports/page already exist: inspect, never regenerate sealed artifacts')
    audit=read(OUT/'另实现复核.json');assert audit['passed']
    binding=read(OUT/'输入与源码绑定.json')
    for e in binding['sources']+binding['dependencies']+binding['inputs']+[binding['protocol']]:assert sha(ROOT/e['path'])==e['sha256']
    spatial=read(OUT/'格级已存分数重组.json');scene=read(OUT/'场景分数与查询投影.json');decision=read(OUT/'开发信号筛选.json')
    parts=['# CV-021场景与精确拍摄格信号诊断\n',
        '2026-10-18；同5个已消费开发目标、四个既有冻结变换，不训练/扩大模型，不消费6确认组24素材。125窗口仅对应5张地面照片/5来源组，不是125独立场景。\n',
        '## 总体结果\n',
        ('有方案通过预登记开发方向筛选，可另冻结一个最小闭环假设；这不是方法正式通过或导航成功。\n' if decision['direction_signal_present'] else '四种方案均未通过预登记开发方向筛选。按停止条件收束当前冻结表征/小头路线，不追加轮数、参数、配对、池化变体或种子追分，不开启确认或升级默认。\n'),
        '| 方案 | 场景Top1 Full/Mean/Wrong（各5） | 81格Top1 Full/Mean/Wrong（各5） | 81格Top5 Full/Mean/Wrong | Full格排名胜双控制来源 | 方向筛选 |',
        '|---|---|---|---|---|---|']
    for arm,x in decision['arms'].items():
        tri=lambda values:'/'.join(str(values[c]) for c in ('Full','Mean','Wrong'))
        parts.append(f"| {arm} | {tri(x['scene_top1_counts'])} | {tri({c:x['spatial_81_counts'][c]['top1'] for c in x['spatial_81_counts']})} | {tri({c:x['spatial_81_counts'][c]['top5'] for c in x['spatial_81_counts']})} | {len(x['cell_rank_better_than_both'])}/5 | {'通过' if x['signal_screen_passed'] else '未通过'} |")
    parts.extend(['\n## 各来源证据\n','场景排名为5选1，格排名为同来源81选1。最高分格距离是事后曼哈顿格数，不是SG或新导航轨迹。\n',
        '| 方案 | 来源 | 场景排名F/M/W | 81格排名F/M/W | Full最高分距代理格 |', '|---|---|---|---|---|'])
    for arm,x in scene['arms'].items():
        for s in x['per_source']:
            cells={c:next(r for r in spatial['cases'] if (r['source'],r['arm'],r['condition'])==(s['source'],arm,c)) for c in ['Full','Mean','Wrong']}
            ranks='/'.join(str(s['conditions'][c]['true_rank']) for c in cells)
            cellranks='/'.join(str(cells[c]['true_rank']) for c in cells)
            parts.append(f"| {arm} | {s['source']} | {ranks} | {cellranks} | {cells['Full']['best_cell_distance']} |")
    parts.extend(['\n## 方程、输入与停止条件\n',
        '格级4860分数由旧1625排序中四变换×三条件的窗口按物理格去重，不重新调用头给格子打分；重复窗口同格差≤1e−12才接受。每来源覆盖原9×9的81格，代理格维持原中心(4,4)，未重标可见地标。排序以降序分数及CV-002同族SHA tie规则固定，tie key不含隐藏目标。\n',
        '场景向量 b_s = unit(mean_{81}(unit(a_s,i)))；目标 Full = unit(g_9..13)，Mean = unit(mean(unit(g_0..8)))，Wrong固定循环5目标。冻结头 z = unit(g + U(Vg))，Identity的U=0；场景分数为 z·b_s。四方案×(5个Full＋1个Mean)×5场景=120个值，Wrong复用Full行。仅24个既有小头目标投影；该场景库含评测器特权，绝不能交给导航Agent。\n',
        '冻结筛选：Full场景Top1≥3/5、胜Mean和Wrong各≥2个来源，且同至少3个来源的81格Full排名严格同时胜Mean/Wrong。四方案全部报告；若多方案过则固定优先Identity、Swiss、MM9、MM12。门槛仅用于本次开发方向筛选，不能替代CV-019、正式方法验收、导航SR或地理泛化证据。\n',
        '5个相同地面目标的重复窗口不增加有效样本量。120个粗场景分数不能证明精确拍摄点可解；81格Top5和邻近一格指标不改变“到达原目标格”的成功定义。失败不能证明数据无效、标签必错或主任务不可能。\n',
        '## 验证与资源\n',
        f"另实现审计通过：4860已存槽位误差{audit['max_saved_score_error']:.3g}，120场景分数最大误差{audit['max_scene_arithmetic_error']:.3g}，目标投影另一算式最大误差{audit['max_projection_arithmetic_error']:.3g}；逐来源排名、物理坐标、分层、控制与全部筛选条件一致。审计用逐维求和，不调用头对象/骨干；同Agent另一实现，不称独立人员审查。\n",
        '无torch/transformers导入，0骨干forward/训练/原图/API/导航/确认。CPU预算≤30分钟、研究输出≤4MiB；未复制缓存或参数。实际计时见执行完成.json，非受控速度基准。旧输入/源码/缓存/头/Word与默认SHA复核保持，原M0/V4.5及CV-019失败结论不变。\n',
        '## 平台交付与后续\n',
        '[地面目标证据页](../../../../平台/实验工作台_v4/地面目标证据_v1/地面目标证据.html)可独立打开，支持本地选图预览、四方案与条件/案例筛选、81格证据、场景混淆、代理格说明及JSON导出。该页不训练、预测、上传或创建导航实验；本地预览照片不会改变固定诊断目标。生产服务与已有GUI不重启，航拍稳定队列继续使用原入口。\n',
        ('下一项可基于固定优先方案另冻单一语义证据记忆闭环设计；尚无导航准入，来源、参数、预算、基线与停止条件必须另登记。\n' if decision['direction_signal_present'] else '本轮到此停止当前小头路线。下一项建议推进平台地面目标任务管理和诊断过程接入，再为跨视角表征另立有机制/来源证据的方案；不复活覆盖因素，不重复图审/裁剪/池化扫描，不以平台演示为由放宽科研门槛。\n'),
        '\n绑定、原始诊断JSON、执行完成及另实现复核均在本目录，旧封存不重跑。'])
    write(OUT/'场景与精确格诊断报告.md','\n'.join(parts)+'\n')
    payload={'spatial':spatial,'scene':scene,'decision':decision,'binding':{
        'task':'CV-021','protocol_sha256':sha(OUT/'冻结诊断协议.json'),'input_source_binding_sha256':sha(OUT/'输入与源码绑定.json'),
        'audit_sha256':sha(OUT/'另实现复核.json'),'capture_point_semantics':'CSV/code capture-point proxy, not necessarily visible landmark',
        'goals':'five fixed known-development photos; user preview does not change results','source_units':5,
        'not_navigation':True,'confirmation_groups_reserved':6,'backbone_forwards':0,'training':0}}
    PAGE.mkdir()
    write(PAGE/'诊断数据.json',json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    template=(ROOT/'project/src/webapp/crossview_evidence_v1.html').read_text(encoding='utf-8')
    assert template.count('__EVIDENCE_JSON__')==1
    embedded=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    write(PAGE/'地面目标证据.html',template.replace('__EVIDENCE_JSON__',embedded))
    write(PAGE/'README.md','# 地面目标证据页\n\n[打开交互页](地面目标证据.html) / [可导出数据](诊断数据.json) / [真实诊断报告](../../../DATA/processed_data/MM-GAG/场景与精确格信号分解_v1/场景与精确格诊断报告.md) / [返回工作台说明](../README.md)\n\n本页是现有V4.5的辅助证据阅读交付，不替换主服务或新增并行平台。自包含HTML可离线打开；地面目标本地照片预览不上传、不保存、不调用模型，不改变本页固定目标与数值。可切换四方案、Full/Mean/Wrong、五已消费案例、失败筛选、81格分数与场景排名，导出当前案例或完整JSON。\n\n“地面照片实时主动导航”仍未准入；航拍M0继续使用原单一启动入口。全图格及代理真值是事后用户证据，禁止传策略。6确认组未消费，所有开发诊断与正式验收分账。模板位于project/src/webapp/crossview_evidence_v1.html，生成源为documents/build_crossview_signal_report_v1.py，输入为CV-021封存JSON；复核源码和SHA，不重跑旧训练/编码。\n')
    write(OUT/'README.md','# CV-021：场景与精确格信号分解\n\n[诊断报告](场景与精确格诊断报告.md) / [逐方案筛选](开发信号筛选.json) / [81格已存分数](格级已存分数重组.json) / [场景分数与投影](场景分数与查询投影.json) / [另实现复核](另实现复核.json)\n\n[冻结协议](冻结诊断协议.json) / [输入源码绑定](输入与源码绑定.json) / [执行状态](执行完成.json) / [交互证据页](../../../../平台/实验工作台_v4/地面目标证据_v1/README.md)\n\n本批已执行一次，不覆写重跑。只用5个已消费开发目标及既有缓存/头，0骨干forward/训练/原图/导航/API/确认。完整81格与5场景检索均含评测器信息，不是Agent输入或新SR。下一动作以报告与当前PROJECT_STATE为准，旧失败/默认/用途不回写。\n')
    total=sum(p.stat().st_size for p in PAGE.iterdir())
    if total>2*1024**2:raise ValueError('Platform output cap exceeded')
    print(json.dumps({'reports_built':True,'platform_page_bytes':total,'direction_signal':decision['direction_signal_present']},ensure_ascii=False))

if __name__=='__main__':main()
