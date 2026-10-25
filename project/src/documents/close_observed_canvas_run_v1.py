"""CV034 completion or stopped-prefix publication, no inference or history rewriting."""
import hashlib
import json
import math
from project.src.eval.observed_canvas_run_v1 import ROOT,OUT,SOURCES,read,entry,put,verify


def describe_baselines(cases,rows,thresholds):
    lookup={c['case_id']:c for c in cases};baselines=[];diagnostic=[]
    for c in cases:
        seen=c['public_visited_cells'];centers={k:[(k%5)*125+62.5,(k//5)*125+62.5] for k in seen};current=centers[c['public_current_cell']]
        centroid=[sum(v[i] for v in centers.values())/len(centers) for i in (0,1)]
        selected=min(seen,key=lambda k:(math.dist(centers[k],centroid),k))
        candidates=dict(parent_center=[312.5,312.5],current_center=current,visited_centroid_projected=centers[selected])
        labels=c['labels_evaluator_only'];points={}
        for name,xy in candidates.items():
            legal=int(xy[1]//125)*5+int(xy[0]//125) in seen
            error=math.dist(xy,labels['xy_native'])*.16 if labels['primary_relation']=='inside' else None
            points[name]=dict(xy=xy,legal=legal,error_m_if_inside=error)
        baselines.append(dict(case_id=c['case_id'],source=c['query_evaluator_only'],primary_relation=labels['primary_relation'],positions=points))
    for r in rows:
        c=lookup[r['case_id']];lab=c['labels_evaluator_only'];error=math.dist(r['current_roi']['xy'],lab['xy_native'])*.16 if lab['current_relation']=='inside' else None
        accepted=r['current_roi']['score']>=thresholds[r['mode']]['tau']
        diagnostic.append(dict(case_id=r['case_id'],mode=r['mode'],arm=r['arm'],relation=lab['current_relation'],accepted=accepted,error_m=error,
            joint_correct=accepted and error<=5 if error is not None else not accepted))
    return dict(position_baselines=baselines,secondary_current_ROI_detailed=diagnostic,always_reject_BA=2/3,always_accept_BA=1/3,
        note='position baselines are not detector accuracies;secondary ROI is not used in primary gates')


def publish():
    binding=verify();resource=read(OUT/'资源回执.json')
    done=read(OUT/'完整执行完成.json') if (OUT/'完整执行完成.json').exists() else None
    audit=read(OUT/'另实现复核.json') if (OUT/'另实现复核.json').exists() else None
    complete=bool(done and done['complete'] and resource['passed'] and audit and audit['passed'])
    effect=False;summary=None
    if complete:
        summary=read(OUT/'开发对照汇总.json');effect=summary['effect_gate_passed']
        status='有界开发感知候选通过，未获导航准入' if effect else '工程与完整对照通过，效果门槛未过并收束'
    else:status='工程/执行或复核未通过，停止并保留前缀'
    events=[json.loads(l) for l in (OUT/'执行事件.jsonl').read_text(encoding='utf-8').splitlines()] if (OUT/'执行事件.jsonl').exists() else []
    started=sum(e['kind']=='forward_started' for e in events);finished=sum(e['kind']=='forward_complete' for e in events);decodes=sum(e['kind']=='image_decode_complete' for e in events)
    rows=[read(p) for p in sorted(OUT.glob('调用_*.json'))]
    body='本批未达到完整准入，不能从缺失前缀得出方法收益；不自动重试或扩大资源。\n'
    if complete:
        thresholds=read(OUT/'两表示固定校准.json')['thresholds']
        details=describe_baselines(binding['cases']['development'],rows[96:],thresholds);put('位置基线与次要ROI明细.json',details)
        table='|表示/目标|来源2联合正例|来源3联合正例|来源宏检测BA|\n|---|---:|---:|---:|\n'
        source_table='|表示/目标|来源|窗外误接受|异源误接受|正例均值误差（含拒绝）|\n|---|---|---:|---:|---:|\n'
        for key,m in summary['metrics'].items():
            a=m['by_source']['opencvl_engineering_2'];b=m['by_source']['opencvl_engineering_3']
            table+=f"|{key}|{a['joint_positive']}/{a['positive_n']}|{b['joint_positive']}/{b['positive_n']}|{m['macro_BA']:.2%}|\n"
            for source,v in m['by_source'].items():
                o=v['counts']['outside'];u=v['counts']['unrelated']
                source_table+=f"|{key}|{source.rsplit('_',1)[-1]}|{o['accepted']}/{o['n']}|{u['accepted']}/{u['n']}|{v['positive_MAE_all_predictions_m']:.3f}m|\n"
        contrasts='|累计Full相对|正例恢复/损伤|窗外恢复/损伤|异源恢复/损伤|宏BA差|逐源联合净恢复|\n|---|---:|---:|---:|---:|---|\n'
        for key,c in summary['contrasts'].items():
            groups=[f"{c['by_class'][k]['recovered']}/{c['by_class'][k]['harmed']}" for k in ('inside','outside','unrelated')]
            contrasts+=f"|{key}|{'|'.join(groups)}|{100*c['macro_BA_gain']:+.2f}pp|{[v['joint_positive_net'] for v in c['by_source'].values()]}|\n"
        gates='\n'.join('- '+k+'：'+('通过' if v else '未通过') for k,v in summary['gates'].items())
        tau='；'.join(f"{m} τ={t['tau']!r}，校准效用={t['utility']:.2%}" for m,t in thresholds.items())
        body=f'''## 主结果（共同已访ROI）

{tau}。阈值仅来自0/1 Full88输出，已在首开发调用前保存；两表示分别校准，各表示所有前缀和四目标条件共用。主检测BA与位置精度分开，全拒绝BA66.67%、全接受33.33%。Disabled的raw分数为0，其接受行为受阈值符号影响，不单凭对Disabled的BA优势证明目标有效。

{table}
{source_table}
{contrasts}
{gates}

全部44基础×2表示×4控制的逐条分数、坐标、同题恢复/损伤均可追溯。所有inside误差包括拒绝样本，不筛掉错误来算平均。次要当前格ROI及三种公开位置基线在[单独明细](位置基线与次要ROI明细.json)，不混入上述主门槛。

{'当前只通过已消费开发来源上的感知候选，仍不能升级自主导航；下一须独立设计行动机制及未消费来源确认。' if effect else '依原CV033停止条件，当前累计画布候选收束，不在该队列上改阈值/路径/来源/裁剪或扩参追分。此结果只限制本完整适配与四工程组，不等于原论文或主动探索方向无效。'}
'''
    report=f'''# CV-034：双画布同预算真实开发对照

**{status}。** 工程/模型运行成功与方法效果过线分别判断；本批为固定观测前缀感知对照，未运行主动导航，没有新增导航SR。

## 冻结内容与实际消费

复用唯一CCVPE公开KITTI/no_orientation_prior权重60416657参数与CV0300/1六层Mean。原0/1校准、2/3开发，旧4工程来源已消费，非独立确认；已知夜间/日照分工与目标偏中部。公开左中点起步的固定B10行扫掠，Single仅当前纹理、Cumulative保留已访纹理，主评价都在共同M_t；模型不见父图未知纹理、真值、GPS、heading或源身份。

计划448=工程8+校准88+开发352，无额外热身。实际started={started}、completed={finished}、原图解码完成={decodes}；完整准入={complete}。8图只在该worker中解码，哈希绑定的原始字节读取与解码分开计数。全部CPU，采样峰值RSS={resource['peak_sampled_RSS']/1024**3:.3f}GiB，总时长={resource['wall_seconds']:.2f}秒；退出码{resource['exit_code']}，资源停止={resource['stop']}。总时长含载入/解码/另峰值/哈希/日志，不作纯推理速度基准。

实际执行器先验11项合成检查通过，完整44身份/原队列SHA必须一致，不能删题后重定义分母。目标与当前RGB帧为唯一影像入口；宿主只裁当前位置125格并用NumPy切片交叉核裁图，不将父图交给Replay。首步两表示与目标控制工程8次通过后才进校准；模型818张量与CV030原strict状态吻合且前后冻结。网络在导入远端代码前禁用，不安装包或下载骨干，不关闭其它进程。异常前缀以原事件/回执为准；未完整时不能套用这些完成条件。

{body}

## 证据、复核与边界

[绑定](首真实调用绑定.json) · [资源](资源回执.json) · [事件](执行事件.jsonl)。完整时另有工程八次核验、两表示固定校准、开发对照汇总、冻结权重前后核验、完整执行完成和另实现复核JSON。四个固定锚点为来源2/3自配对、step10、Full×两表示，保存512定位图与256yaw-max图；其余输出只保留峰值/标量/哈希和在线两算法一致性回执，不按效果选图。

另实现审计状态={bool(audit and audit['passed'])}；完整情况下离线重算440主输出的阈值/各来源指标/同题恢复损伤/门槛，检查448在线双算法峰值和4锚点，核全部层控制路由与输入/输出身份、来源/预算/哈希。非锚点的完整全图未保存，其峰值全局一致性依赖在线第二实现及绑定代码；不假称离线重建全部448张全图。复核同Agent另一实现，不称独立人员审查，不追加模型前向或图像解码。

32或44个前缀不增加独立地域数；开发只有2个已知来源。报告原值/来源增益，不报伪独立显著性、跨数据集胜出或与旧CV031不同ROI队列的累计提升。平台4.12/原M0/S2S3S4及6确认保持，本轮未改GUI。0训练/GPU/导航/云API/下载/额外模型/确认组消费/Git提交推送，无网络代理或主动C盘任务文件。

恢复只读绑定、事件、资源、实际完成和本报告；bind/execute/worker/写盘audit/publish不能盲重跑，任何新尝试必须另版本/明确新资源与已消费输入，不替换旧成绩。
'''
    with (OUT/'README.md').open('x',encoding='utf-8') as f:f.write(report)
    put('最终准入判定.json',dict(engineering_and_complete_audit_passed=complete,effect_gates_passed=effect,candidate_closed=not effect,
        navigation_admitted=False,new_navigation_SR=None,actual_started=started,actual_completed=finished,confirmation_groups_consumed=0))
    rel=OUT.relative_to(ROOT).as_posix();path=ROOT/'项目导航/实验索引.json';old=path.read_bytes();idx=read(path);prior=idx['batches'];assert len(prior)==160
    assert hashlib.sha256(json.dumps(prior,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==binding['index_sha']
    idx['date']='2026-10-24';idx['batches']=prior+[dict(category='CV-034',dataset='OpenCVL原4工程组/2开发来源',kind='双画布实际感知开发对照',name='双画布同预算真实开发对照_v1',status=status,path=rel,reports=[rel+'/README.md'],evidence=[rel+'/最终准入判定.json',rel+'/首真实调用绑定.json'])]
    assert path.read_bytes()==old;path.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=ROOT/'docs/维护/版本生命周期.json';old=path.read_bytes();life=read(path);assert life['current']==binding['current'];life['updated']='2026-10-24'
    life['cv034_observed_canvas_run']=dict(status=status,report=rel+'/README.md',replaces='CV033 pending execution only;no default/model/platform replaced',
        dependencies=['CCVPE unique public checkpoint','CV030 calibration Mean','CV032 public canvas','CV033 fixed protocol'],
        recovery='read completed/stopped prefix;no blind rerun',complete_and_audit=complete,effect_passed=effect,navigation_admitted=False)
    assert path.read_bytes()==old;path.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        path=ROOT/name;old=path.read_bytes();s=path.read_text(encoding='utf-8-sig');h,_,rest=s.partition('\n');prefix='' if name=='README.md' else '../'
        link=f'[CV-034 · 双画布真实对照]({prefix}{rel}/README.md)：{status}，实际完成{finished}/448；无导航SR。'
        assert path.read_bytes()==old;path.write_text(h+'\n\n'+link+'\n\n'+rest,encoding='utf-8')
    path=ROOT/'docs/PROJECT_STATE.md';old=path.read_bytes();s=path.read_text(encoding='utf-8-sig');h,_,rest=s.partition('\n')
    state=f'''\n\n### CV-034：{status}（2026-10-24）

[完整证据](../{rel}/README.md)。固定CV033队列与448预算实际completed={finished}，原图解码={decodes}；完整执行/复核={complete}，效果门槛={effect}。CPU采样RSS{resource['peak_sampled_RSS']/1024**3:.3f}GiB/{resource['wall_seconds']:.2f}秒。完整时按来源/表示/目标条件报告共同已访ROI与次要当前ROI，详见报告，不与旧CV031不同队列直接相减。161索引、旧160值保持，current平台入口与原M0/S2S3S4/6确认不变。没有新导航SR/训练/下载/API或默认升级。\n'''
    assert path.read_bytes()==old;path.write_text(h+state+'\n'+rest,encoding='utf-8')
    verify()
    files=[entry(p) for p in sorted(OUT.iterdir()) if p.is_file()]
    put('阶段封存.json',dict(files=files,sources=binding['sources'],complete_audit=complete,effect_passed=effect,seal_itself_excluded=True))
    size=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file());assert size<16*1024**2
    print(json.dumps(dict(published=True,index=161,complete_audit=complete,effect_passed=effect,bytes=size),ensure_ascii=False))


if __name__=='__main__':publish()
