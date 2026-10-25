"""Close CV036, retaining full or stopped-prefix facts, without new inference."""
import hashlib
import json
from project.src.eval.full_reference_anchor_v1 import ROOT,OUT,SOURCES,read,put,entry,verify


def publish():
    b=verify();resource=read(OUT/'资源回执.json')
    done=read(OUT/'完整执行完成.json') if (OUT/'完整执行完成.json').exists() else None
    audit=read(OUT/'另实现复核.json') if (OUT/'另实现复核.json').exists() else None
    complete=bool(done and done['complete'] and resource['passed'] and audit and audit['passed'])
    summary=read(OUT/'能力锚定汇总.json') if complete else None
    goal=bool(summary and summary['full_reference_goal_advantage_supported'])
    status=('完整参考目标优势得到有界支持' if goal else '完整参考仍未支持可靠目标定位，停止当前直接迁移') if complete else '静态诊断执行/复核未通过，保留停止前缀'
    events=[json.loads(l) for l in (OUT/'执行事件.jsonl').read_text(encoding='utf-8').splitlines()] if (OUT/'执行事件.jsonl').exists() else []
    started=sum(e['kind']=='forward_started' for e in events);finished=sum(e['kind']=='forward_complete' for e in events);decodes=sum(e['kind']=='decode_complete' for e in events)
    content='本批未完整，不能从缺失前缀判断目标定位优势；停止并保留证据，不自动重试。\n'
    if complete:
        table='|条件|5米内位置数|平均误差|中位误差|来源0/1/2/3误差（米）|\n|---|---:|---:|---:|---|\n'
        for arm,m in summary['metrics'].items():table+=f"|{arm}|{m['within5m']}/4|{m['mean_error_m']:.3f}m|{m['median_error_m']:.3f}m|{', '.join(f'{v:.3f}' for v in m['by_pair'])}|\n"
        contrasts='|Full相对|逐对误差优势（控制−Full，米）|优势≥1米配对数|恢复/损伤|5米内净收益|\n|---|---|---:|---:|---:|\n'
        for arm,c in summary['contrasts'].items():contrasts+=f"|{arm}|{', '.join(f'{v:+.3f}' for v in c['control_minus_Full_error_m'])}|{c['Full_advantage_at_least1m']}/4|{c['recovered']}/{c['harmed']}|{c['joint_net']:+d}|\n"
        gates='\n'.join('- '+k+'：'+('通过' if v else '未通过') for k,v in summary['gates'].items())
        content=f'''## 实际位置能力

{table}
{contrasts}
{gates}

Full是原地面目标，三个控制仍对原参考图拍摄点标签计误差。Wrong照片的实际对应地点不在当前参考图，对原答案偶然接近也不代表错误照片被正确定位或拒绝。这里不设置接受/拒绝阈值，不把softmax/raw匹配峰当存在概率；没有负例检测FAR、校准或导航SR。
'''
    text=f'''# CV-036：完整参考条件能力锚定

**{status}。** 已完成判定={complete}。这是4个已消费配对上的特权静态诊断，不是Agent实验；完整父图没有接入平台导航，不改变其信息权限。

## 固定输入、门槛与实际执行

原4对OpenCVL工程素材不换，0/1曾用于Mean估计，2/3没有参与Mean估计但同样已消费，不能称独立确认。复用唯一CCVPE/KITTI/no_orientation_prior权重60416657参数/818状态；地面图沿用先前letterbox256×1024，完整625父航片沿用CV030标准RGB resize512路径。没有使用真实heading旋转、GPS或拍摄点重居中；原参考坐标只用于事后误差。

首预测前固定：Full至少3/4位置≤5米且来源2/3都≤5米；相对每个Disabled/Mean/Wrong至少3/4配对有≥1米误差优势，同时5米位置净恢复≥1对。后验不得改这些条件。四条件固定0↔1、2↔3交换错误目标；Mean只读CV0300/1六描述子均值，Disabled六匹配图精确零，不重新估计任何权重/阈值。

计划16次完整前向，实际started={started}/completed={finished}、原图decode={decodes}；无热身重试。CPU采样峰值RSS={resource['peak_sampled_RSS']/1024**3:.3f}GiB、总时长{resource['wall_seconds']:.2f}秒，退出码{resource['exit_code']}、停止原因{resource['stop']}；上限4GiB/180秒。总时间包含载入/哈希/解码，不作纯模型速度基准。

4个Full的全部9输出必须与旧CV030完整图前向SHA完全一致；这为旧工程“只存等价SHA，没存位置证据”补齐静态能力信息，旧实验和旧判定不回写。参考图处理采用原CV030完整RGB路径，与稀疏画布的浮点通道缩放存在细微量化差异，本次不是只分离可见比例的精确消融。Full若失败，只限制这一整套普通照片/输入处理/冻结权重条件，不说明原论文方法无效。

{content}

## 复核与证据

{b['tests']['count']}项纯合成检查通过；另实现复核={bool(audit and audit['passed'])}。完整情况下离线从4保存Full热图重算位置/像素中心与raw匹配峰，从16回执独立计算米数/各对误差与控制恢复损伤，核原CV030输出/模型冻结、目标路由/六层控制、预算与输入源码SHA。没有额外前向或图像解码；同Agent另一实现，非独立人员审查。模型参数前后与原CV030 strict状态一致。

[首预测协议与绑定](首预测冻结协议与绑定.json) · [资源](资源回执.json) · [事件](执行事件.jsonl)。完整时另有能力锚定汇总、另实现复核和完整执行完成JSON；4Full数值热图固定保存，不按成绩选图，不重复保存原图/模型。

## 下一路线

{'完整参考显示目标优势后，可另案研究合法传感器视场/连续窗口以满足上下文条件，但不得把本特权全图当Agent可见范围或直接升级导航。' if goal else '完整参考也未提供足够目标空间优势，因此停止当前checkpoint对普通照片的直接迁移，不再追加单格/累计/阈值/填边变体追分。下一应先核任务匹配的空间监督/成对输入条件与可用数据，再选择轻量表征或小头；不是立即增加参数或训练。'}无论结果如何，4已知配对不能给出跨地域/任意照片泛化结论；地面自主探索仍未准入。当前平台4.12/旧M0/S2S3S4、CV031/034失败原件与6确认保持，0训练/GPU/导航/云API/下载/默认升级。

本轮使用已有依赖/权重，无网络/代理、作者联系、安装或系统配置变更、Git提交推送或主动C盘任务输出。恢复先读绑定/事件/资源/完成/本报告，不重跑bind/execute/worker/写盘audit/publish或旧封存脚本。
'''
    with (OUT/'README.md').open('x',encoding='utf-8') as f:f.write(text)
    put('最终诊断与路线.json',dict(complete_audit=complete,full_reference_goal_advantage_supported=goal,
        next_route='legitimate_sensor_window_design' if goal else 'stop_direct_checkpoint_transfer_and_require_taskmatched_supervision',
        navigation_admitted=False,new_navigation_SR=None,no_rejection_threshold=True,calls_started=started,calls_completed=finished))
    rel=OUT.relative_to(ROOT).as_posix();p=ROOT/'项目导航/实验索引.json';old=p.read_bytes();idx=read(p);before=idx['batches'];assert len(before)==162
    assert hashlib.sha256(json.dumps(before,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==b['index_sha']
    idx['date']='2026-10-24';idx['batches']=before+[dict(category='CV-036',dataset='OpenCVL原4已消费工程配对/完整参考图',kind='特权静态位置能力诊断',name='完整参考条件能力锚定_v1',status=status,path=rel,reports=[rel+'/README.md'],evidence=[rel+'/首预测冻结协议与绑定.json',rel+'/最终诊断与路线.json'])]
    assert p.read_bytes()==old;p.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'docs/维护/版本生命周期.json';old=p.read_bytes();life=read(p);assert life['current']==b['current'];life['updated']='2026-10-24'
    life['cv036_full_reference_anchor']=dict(status=status,report=rel+'/README.md',replaces='CV035 full-reference evidence gap only;no old result/default replaced',
        depends_on=['uniqueCCVPE checkpoint','CV030 mean/Full hashes','original4 pairs'],recovery='read sealed static results;do not repeat16 calls',navigation_admitted=False)
    assert p.read_bytes()==old;p.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;old=p.read_bytes();s=p.read_text(encoding='utf-8-sig');h,_,rest=s.partition('\n');prefix='' if name=='README.md' else '../'
        link=f'[CV-036 · 完整参考能力锚定]({prefix}{rel}/README.md)：{status}，完成{finished}/16；特权静态，非Agent导航。'
        assert p.read_bytes()==old;p.write_text(h+'\n\n'+link+'\n\n'+rest,encoding='utf-8')
    p=ROOT/'docs/PROJECT_STATE.md';old=p.read_bytes();s=p.read_text(encoding='utf-8-sig');h,_,rest=s.partition('\n')
    result='' if not complete else '；Full≤5m='+str(summary['metrics']['Full']['within5m'])+'/4'
    state=f'''\n\n### CV-036：{status}（2026-10-24）

[完整证据](../{rel}/README.md)。原4已消费完整配对/同checkpoint/四目标条件，实际完成{finished}/16，完整与另实现核验={complete}{result}。完整Full输出与CV0309输出SHA一致，仅补旧工程定位证据不回写旧报告。CPU采样RSS{resource['peak_sampled_RSS']/1024**3:.3f}GiB/{resource['wall_seconds']:.2f}秒，未设置拒绝阈值；这是特权完整图位置诊断，不计导航SR。163索引、旧162值与current平台4.12/旧默认/6确认保持。{'下一另案合法视场/窗口适配。' if goal else '停止当前冻结权重普通照片直接迁移，先核任务匹配空间监督与数据条件，不继续变体追分。'}\n'''
    assert p.read_bytes()==old;p.write_text(h+state+'\n'+rest,encoding='utf-8')
    verify();put('阶段封存.json',dict(files=[entry(p) for p in sorted(OUT.iterdir()) if p.is_file()],sources=b['sources'],complete_audit=complete,goal_advantage_supported=goal,seal_itself_excluded=True))
    size=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file());assert size<8*1024**2
    print(json.dumps(dict(published=True,index=163,complete_audit=complete,goal_advantage=goal,bytes=size),ensure_ascii=False))


if __name__=='__main__':publish()
