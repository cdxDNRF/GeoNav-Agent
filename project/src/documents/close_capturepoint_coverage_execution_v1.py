"""CV-019 complete the frozen decision after independent-equation full audit."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.capturepoint_coverage_protocol_v1 import coverage_gate,CANDIDATE

BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'配对覆盖同预算执行_v1';DESIGN=BASE/'三新增来源工程与配对覆盖协议_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    audit=read(OUT/'核验/完整另实现复核.json');engineering=read(OUT/'核验/工程另实现复核.json')
    if not audit['passed'] or not engineering['passed']:raise ValueError('Both audit stages required')
    s=read(OUT/'开发对照汇总.json');log=read(OUT/'训练日志.json');q=read(DESIGN/'固定训练与开发队列.json');b=read(OUT/'首forward绑定.json')
    decision=coverage_gate(s['gate_counts'],complete=True,engineering_and_audit=True)
    if decision['gates']!=audit['independent_effect_gates']:raise ValueError('Independent gates differ')
    write('核验/最终门槛判定.json',decision)
    rows=[]
    for arm,conditions in s['counts'].items():
        for condition,c in conditions.items():rows.append(f"| {arm} | {condition} | {c['top1']}/125 | {c['top5']}/125 | {c['median_rank']:g} |")
    source_ids={t['group_evaluator_only']:t['source_id_evaluator_only'] for t in q['development_tasks']};sources=[]
    for group,c in s['per_source'].items():
        n=c[CANDIDATE];old=c['MMRank8Final']['Full'];sources.append(f"| {source_ids[group]} | {old['top1']}/{old['top5']} | {n['Full']['top1']}/{n['Full']['top5']} | {n['Mean']['top1']}/{n['Mean']['top5']} | {n['Wrong']['top1']}/{n['Wrong']['top5']} |")
    effects=[]
    for name,c in s['contrasts'].items():
        for metric,x in c.items():effects.append(f"| {name} | {metric} | {x['recovered']} | {x['harmed']} | {x['net']} | [{100*x['paired_source_percentile95'][0]:.1f}, {100*x['paired_source_percentile95'][1]:.1f}] |")
    gates='\n'.join(f"- {k}：{'通过' if v else '未过'}" for k,v in decision['gates'].items())
    verdict='开发候选达到本批冻结门槛；不自动进入确认、导航或默认升级' if decision['passed'] else '必要门槛未全部通过，按停止条件收束该覆盖因素，不追加参数/轮数或消费确认追分'
    write('十二配对同预算训练与目标对照报告.md',f'''# 12配对与9配对的固定预算开发对照

CV-019 / 2026-10-18。**{verdict}。** 本批是已知开发拍摄点代理格关联，不是导航SR或独立新来源确认；6确认组24素材保持，旧9配对结果/航拍M0/生产V4.5/Word不变。

## 实际执行与公平预算

冻结8192参数秩8小头、两既有视觉骨干和PIL处理，seed0固定500更新/batch9，共4500次优化器监督使用，12查询各375次；原9配对基线每查询500次，最终权重/原1250排序复用，没有重训基线或择优。唯一因素为同9训练组内正配对9→12，不增加独立地理来源。未改变损失/温度/优化器/阈值，不新增模型或API。

保持原14地面/1134航拍缓存，额外3地面+243航拍引用中35个航拍按原SHA、同编码器张量/原处理身份复用旧向量。实际新增3地面+208航拍=211编码输入/27调用，低于246/32上限；两个87849216参数编码器张量SHA前后相同并与旧缓存一致，无梯度。设备/显存见编码环境与编码后记录，不停止其它程序；CPU/GPU耗时不作受控性能基准。

实际训练按0..8及14..16取目标，开发9..13从未进入训练；12×81候选与正格40仅作训练/评测监督，头每次只收512目标向量。原9目标均值、Wrong映射、125已知开发窗及哈希破平局保持。旧1250条与新375条以引用组合成1625计划记录，未重复复制旧逐题文件。

完整12训练集的初始/最终拟合损失为{log['initial_fit12_loss']:.6f}→{log['final_fit12_loss']:.6f}；优化器训练与检查点保存耗时{log['seconds']:.2f}秒。首尾完整12查询只做无梯度损失评估，4500统计指优化器监督使用；逐步日志是变化的9查询批，不把不同批次曲线冒充同题评估。

## 固定125窗目标对照

| 方法 | 目标条件 | Top1 | Top5 | 真目标排名中位数 |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

125窗只有5个不同地面目标/开发来源；Top-k不是导航SR，开发缓存不能给在线Agent。完整原结果和新增375记录见[引用](合并结果引用.json)。

## 来源分层（各25窗，单元格Top1/Top5）

| 来源 | 旧MM9 Full | 新MM12 Full | 新MM12 Mean | 新MM12 Wrong |
|---|---:|---:|---:|---:|
{chr(10).join(sources)}

## 恢复、损伤和5来源配对区间

| 候选Full相对条件 | 指标 | 恢复 | 损伤 | 净值 | 来源配对95%描述区间pp |
|---|---|---:|---:|---:|---:|
{chr(10).join(effects)}

区间按5来源完整3125次重采样，不把多窗口或多个头算独立地域；区间必报但不新增隐形通过门槛。这里只检验一个固定seed/final500，不能据此锁定失败原因或宣称跨域/跨视角全场景能力。

## 预登记门槛

{gates}

本轮不在看到结果后调整阈值/配对/种子/更新次数，任何必要未过都保留失败；原9因素不回写。即使通过也只是已知开发候选，默认升级/新来源确认/真实导航需另案。

## 工程、另实现和保留

工程验证实际新增缓存35复用、fit12防泄漏、原初始化和真实目标头算子；合成梯度/恢复逐位一致，仅8192头参数进入优化器。完整另实现核375新排序/9375分数、全部1625记录分母/来源、6对照的Top1/Top5恢复损伤与来源区间、500步/4500使用/每查询375、初始最终损失与500步恢复状态，以及12门槛。分数最大差{audit['max_score_error']:.3g}；审计0编码器forward/训练/原图再读取。同Agent另一实现，非独立人员审查。

单份新增向量≤2MiB，头初始/最终＋一个恢复≤1MiB，全批≤8MiB，不复制骨干/完整原图。所有旧输入/源码/权重/缓存SHA绑定，记录见[首forward绑定](首forward绑定.json)、[首更新前绑定](首更新前特征绑定.json)、[复核](核验/完整另实现复核.json)、[封存](阶段封存.json)。当前只有结果JSON/报告入口更新，生产GUI/服务不改。

prepare/engineer/execute/audit/报告均已消费，不覆盖或盲重跑；若复核需仅核已存SHA/JSON/检查点，任何新模型或训练另立范围，不重编码成功输入或消耗确认追分。
''')
    write('README.md',f'''# 配对覆盖同预算执行（CV-019）

12配对/9训练组、8192参数小头、500更新/4500次使用、375新增排序已完成。{verdict}。旧1250条复用，共1625计划记录；无导航SR，6确认保留。

- [训练与完整目标对照报告](十二配对同预算训练与目标对照报告.md) / [最终判定](核验/最终门槛判定.json)
- [完整开发汇总/分来源/恢复损伤/区间](开发对照汇总.json) / [新375排序](新增375排序.json) / [旧新结果引用](合并结果引用.json)
- [执行授权](运行授权.json) / [实际源码和模型/旧缓存绑定](首forward绑定.json) / [35旧向量复用计划](特征复用计划.json)
- [工程](工程结果.json) / [首更新前特征绑定](首更新前特征绑定.json) / [训练日志](训练日志.json)
- [工程另实现](核验/工程另实现复核.json) / [完整另实现](核验/完整另实现复核.json) / [测试](核验/边界测试记录.json) / [封存](阶段封存.json)

所有真实输入/源码/预算已绑定，prepare/engineer/execute/audit/报告均已消费，不覆盖重跑。只读核SHA/已存结果可恢复；M0/V4.5/Word和历史验收保持，本批不自动导航/默认升级。
''')
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for x in iter(lambda:f.read(1048576),b''):h.update(x)
        return h.hexdigest()
    for key in ('sources','dependencies','inputs','models','protected','old_features','old_heads'):
        for e in b[key]:assert digest(ROOT/e['path'])==e['sha256']
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='README.md':files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size})
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if total>8*1024**2:raise ValueError('Output cap')
    write('阶段封存.json',{'task':'CV-019','files':files,'sources':b['sources'],'readonly_dependencies':b['dependencies'],
        'sealed_bytes':sum(x['bytes'] for x in files),'mutable_entries_excluded':['README.md'],'candidate_passed':decision['passed'],
        'training_steps':500,'optimizer_query_uses':4500,'new_encoder_inputs':211,'new_encoder_calls':27,
        'navigation_SR':None,'default_upgraded':False,'confirmation_access':0})
    print(json.dumps({'sealed_files':len(files),'bytes_before_seal':total,'candidate_passed':decision['passed'],'candidate_counts':s['counts'][CANDIDATE]},ensure_ascii=False))


if __name__=='__main__':main()
