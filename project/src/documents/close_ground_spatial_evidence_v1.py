"""CV-026 bounded report and publication; no runtime model/policy imports."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
REL='DATA/processed_data/MM-GAG/连续观测空间证据诊断_v1'
OUT=ROOT/REL


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def put(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text if isinstance(text,str) else json.dumps(text,ensure_ascii=False,indent=2));f.write('\n')


def main():
    if (OUT/'阶段封存.json').exists():raise FileExistsError('Publication already consumed')
    r=read(OUT/'诊断汇总.json');a=read(OUT/'另实现复核.json');tests=read(OUT/'合成测试.json')
    assert a['passed'] and tests['passed']
    b=read(OUT/'输入与源码绑定.json')
    for e in b['inputs']+b['sources']+[b['scope']]:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    if r['screen_passed']:raise ValueError('Positive screen requires a separately reasoned candidate; do not auto-publish a negative conclusion')
    aggregates=r['aggregate'];conditions=('Full','Disabled','Mean','Wrong');table=[]
    for c in conditions:
        x=aggregates[c];table.append(f"| {c} | {x['source_macro_edge_accuracy']:.2%} | {x['local_proxy_peaks']}/5 |")
    sources=[]
    for s in r['spatial_maps']:
        x=s['conditions'];sources.append(f"| {s['source']} | {x['Full']['half_tie_accuracy']:.2%} | {x['Mean']['half_tie_accuracy']:.2%} | {x['Wrong']['half_tie_accuracy']:.2%} | {x['Full']['proxy_minus_best_neighbor']:+.4f} |")
    comparisons=[]
    for c in ('Disabled','Mean','Wrong'):
        x=r['contrasts'][c];lo,hi=x['source_bootstrap95'];comparisons.append(f"| Full − {c} | {x['macro_difference']*100:+.2f} | {x['positive_sources']}/5 | [{lo*100:+.2f}, {hi*100:+.2f}] |")
    prefix=[]
    for s in r['source_order']:
        rows={x['condition']:x for x in r['blind_route_deduplicated_physical_edges'] if x['source']==s}
        prefix.append('| '+s+' | '+str(rows['Full']['edges'])+' | '+' | '.join(f"{rows[c]['half_tie_accuracy']:.2%}" for c in ('Full','Mean','Wrong'))+' |')
    qualities=[];rank2=[]
    for arm in (*conditions,'Frontier'):
        q=[x for x in r['old_trajectory_quality'] if x['arm']==arm];n=sum(x['decisions'] for x in q)
        ranks=[sum(x['prefix_geometry_rank_counts'][i] for x in q) for i in range(3)]
        qualities.append(f"| {arm} | {n} | {sum(x['toward'] for x in q)} | {sum(x['away'] for x in q)} | {ranks[0]}/{ranks[1]}/{ranks[2]} |")
        toward=sum(x['recorded_rank2_gradient_alignment']['toward'] for x in q);away=sum(x['recorded_rank2_gradient_alignment']['away'] for x in q);zero=sum(x['recorded_rank2_gradient_alignment']['zero'] for x in q)
        rank2.append(f"| {arm} | {toward} | {away} | {zero} |")
    old=[]
    for arm in (*conditions,'Frontier'):
        q=r['old_navigation_summary_unchanged'][arm];old.append(f"| {arm} | {q['success']}/{q['planned']} | {q['SG_cells']:.2f} |")
    manual=r['manual'];manual_lines=[]
    for e in manual['edges']:
        manual_lines.append(f"| {e['step']} | {e['a']}→{e['b']} | {e['score_delta']:+.6f} | {e['posthoc_distance_progress']:+d} | {'越界，无新空间边' if e['consistency'] is None else '一致' if e['consistency']==1 else '相反' if e['consistency']==-1 else '持平'} |")
    report='''# CV-026：连续观测是否提供可靠空间证据

**结论：当前原始相似度尚不足以支持新的自主方向/接近判据；本轮不冻结新导航候选，不重跑已关闭的CV024趋势策略。** 人工连续观测平台4.11已可用，但工程可用不等于目标方向可靠。此结论只限现有配对代理任务、冻结CLIP/Sat2Cap和原始余弦，不否定其它跨视角方法，也不能唯一归因于编码器或标签。

## 检查范围和公式

只重计既有CV021 Identity分数、CV024的20题/100轨迹及PLATFORM024的1条人工记录。统计单位仍是5个已消费开发来源组，未另行证明地理独立性；81格、20窗、条件、重访和动作不增加地理样本。没有读取新像素/模型/全量特征，没有模型forward、训练、导航、下载、API或6确认消费。

每条空间邻接边只记一次。接近一致性为 sign((s_b−s_a)·(d(a,g)−d(b,g)))：正数表示较近格得分更高；平局给半分，仅作诊断。每源9×9有144条物理边，报告五源宏平均。真值g和全图分数只供评测器事后检查，绝不进入策略或用户实时编码请求。局部峰要求代理格分数严格高于四邻格；不是更改目标标签或到达规则。

冻结的探索性准入检查为：Full边一致性宏平均>50%；至少3/5来源的代理格为局部峰；Full对Mean/Wrong各至少3/5正收益且完整来源配对bootstrap区间下界>0。它只筛查“原始标量单调接近/局部峰”机制，不是所有自主算法的必要条件，也不是正式验收。该研究是有历史结果背景的开发诊断，不包装为盲确认。

## 全图事后空间证据

| 已存目标条件 | 边一致性五源宏平均 | 代理格为严格四邻峰 |
|---|---:|---:|
'''+ '\n'.join(table)+'''

| 来源 | Full边一致性 | Mean | Wrong | Full代理格−最高邻格 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(sources)+'''

| 配对差 | 宏平均差（百分点） | 正收益来源 | 来源95% bootstrap区间（百分点） |
|---|---:|---:|---:|
'''+ '\n'.join(comparisons)+'''

Disabled为恒定零，因此所有边平局、半分基线50%；不能说它有方向辨识能力。Full高于或接近50%也不足以证明使用正确目标：还必须检查Mean/Wrong及跨来源一致性。五源区间仅量化当前已消费队列，不是新区域泛化证明。

## 只使用已观测前缀

固定使用旧无目标Frontier路线作为相同访问过程，比较各目标条件在这些已访问格的旧分数，防止用各自策略挑过的路线证明自身。只取已记录决策输入的position，不把成功后未编码的终点补成观测；同源重叠窗口和重复走过的边按物理格去重。

| 来源 | 去重已观测物理边 | Full一致性 | Mean | Wrong |
|---|---:|---:|---:|---:|
'''+ '\n'.join(prefix)+'''

以下是旧策略实际决策的重计，不产生新动作。几何秩0/1/2分别表示单位置、共线、非共线观测。二维可观测位置不等于目标信号可靠；秩不足只能解释一部分前缀，不能当作全部失败的唯一原因。

| 旧条件 | 已存决策 | 朝向目标 | 背向目标 | 前缀几何秩0/1/2 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(qualities)+'''

仅在几何秩2时，检查**旧日志已保存**的梯度与当前到目标位移的内积方向（无新拟合、无动作选择）：

| 旧条件 | 梯度指向目标半平面 | 背向 | 零梯度 |
|---|---:|---:|---:|
'''+ '\n'.join(rank2)+'''

## 人工实时记录核对

同一已消费场景，4人工动作、5观测、3唯一格、4编码、2缓存命中；只覆盖2条不同物理边，不能作为新的独立导航样本。观察者全图曾打开做隔离测试，这不是人类盲测。距离改善正数表示接近旧代理格。

| 步 | 已访转移 | 相似度变化 | 距离改善（格） | 事后一致性 |
|---|---|---:|---:|---|
'''+ '\n'.join(manual_lines)+'''

相似度已能稳定随当前图像更新，并与旧缓存一致；这证明接口稳定，不能证明单次上升或下降具有可靠方向含义。工程记录仍不计Agent SR，也不利用真值给人或策略选下一步。

## 保留的旧导航结果

| 原CV024条件 | 成功/计划（旧结果） | 全终局SG格数（旧结果） |
|---|---:|---:|
'''+ '\n'.join(old)+'''

这些数字沿用封存原件，没有重跑。真实目标能改变行动，但未证明可靠提升，因此不能用新诊断追认其通过或调阈值恢复旧因素。

## 决策与下一步

本轮准入检查未全部通过，**没有新冻结自主导航协议**，不扩大参数、不重新训练旧小头、不再扫描余弦阈值、裁剪或局部梯度。保留人工探索/单对编码为平台实验功能，显示原始证据且不输出伪位置概率。航拍M0、默认、原S2/S3/S4及6确认组均保持。

下一研究任务应以一条完整路线为单位：先明确可验证的目标空间关系（配对中心代理、实测拍摄点或标注地标），再取得与该关系匹配的成对表征/方向监督证据；优先核查现有公开资产是否满足此准入条件，限定一个方案及资源上限，不自动下载/训练/用确认集筛选。现有5开发来源只能继续作开发，不能改名为新确认。若暂时没有匹配资产，就将当前地面自主线标为“证据不足，保留人工实验模式”，平台和报告仍可交付。

## 复核与恢复

[冻结范围](冻结诊断范围.json)、[输入与源码绑定](输入与源码绑定.json)、[逐来源/逐旧题完整统计](诊断汇总.json)、[合成边界测试](合成测试.json)、[另实现复核](另实现复核.json)、[阶段封存](阶段封存.json)。另实现采用较近/较远直接比较、整数协方差秩、多项式权重bootstrap和保存向量的标量余弦，区别于主统计实现；由同一Agent完成，不称独立人员审查。

本批0新编码/导航/训练/下载/API/原图/6确认，不改运行服务和历史模型/均值/阈值/标签。只新增小型源码/统计/文档，≤3MiB，不复制模型、特征库或图像。恢复查绑定与seal，不能覆盖重跑freeze/execute/close；审计再读请导入audit()而非执行带写盘main。没有Git提交推送，没有外网/代理或C盘任务输出。
'''
    put('README.md',report)
    put('下一步决策.json',dict(task='CV-026',raw_similarity_candidate_admitted=False,new_navigation_protocol_frozen=False,
        old_CV024_factor_closed=True,platform411_human_mode_retained=True,
        next='task-matched spatial relationship and paired asset eligibility; no automatic model/training/navigation',
        screen=r['screen'],reserved_sources_unchanged=True))
    p=ROOT/'项目导航/实验索引.json';idx=read(p);old=idx['batches'];assert len(old)==150
    old_sha=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    idx['date']='2026-10-23';idx['batches']=old+[dict(dataset='MM-GAG已消费开发5来源',kind='连续观测离线空间诊断',name=OUT.name,
        category='CV-026',status='诊断完成；原始余弦自主因素证据不足，不新开导航',scope='只计旧JSON/100轨迹/人工1条；0新预测模型训练导航API确认',path=REL,
        reports=[REL+'/README.md'],evidence=[REL+'/'+n for n in ('冻结诊断范围.json','输入与源码绑定.json','诊断汇总.json','另实现复核.json','下一步决策.json','阶段封存.json')])]
    assert p.read_text(encoding='utf-8-sig')==json.dumps(read(p),ensure_ascii=False,indent=2)+'\n','Index formatting/concurrent change; inspect before publishing'
    p.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert read(p)['batches'][:-1]==old
    put('索引原值核验.json',dict(previous_batches=150,current_batches=151,previous_values_preserved=True,previous_canonical_sha256=old_sha))
    p=ROOT/'docs/维护/版本生命周期.json';life=read(p);life['updated']='2026-10-23'
    life['cv026_spatial_evidence']=dict(status='completed_diagnostic_no_autonomy_candidate',report=REL+'/README.md',
        replaces='no runtime, model or previous acceptance',depends_on=['CV-021 stored Identity scores','CV-024 sealed traces','PLATFORM-024 saved human record'],
        recovery=REL+'/输入与源码绑定.json',platform_current='4.11 human ground walk retained')
    p.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;t=p.read_text(encoding='utf-8-sig');h,_,rest=t.partition('\n');prefix='' if name=='README.md' else '../'
        p.write_text(h+'\n\n[CV-026 · 连续观测空间证据诊断]('+prefix+REL+'/README.md)：重计旧分数与轨迹；当前原始相似度不足以冻结新自主因素，平台4.11人工模式保留。\n\n'+rest,encoding='utf-8')
    files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()]
    put('阶段封存.json',dict(task='CV-026',files=files,sources=b['sources'],no_model_or_navigation=True,historical_inputs_unchanged=True))
    total=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file());assert total<=3*1024**2
    print(json.dumps(dict(published=True,files=len(files),bytes=total,index=151,screen_passed=False),ensure_ascii=False))


if __name__=='__main__':main()
