"""Publish CV031 frozen single screen; no inference or historical artifact rewrites."""
import json
from project.src.eval.ccvpe_screen_v1 import ROOT,OUT,OLD,read,entry,put,verify


def publish():
    b=verify();s=read(OUT/'开发筛查汇总.json');a=read(OUT/'另实现复核.json');resource=read(OUT/'资源回执.json');done=read(OUT/'执行完成.json');cal=read(OUT/'唯一校准阈值.json')
    assert a['passed'] and resource['passed'] and done['complete']
    passed=s['perception_screen_passed'];status='局部感知开发筛查通过，未获导航准入' if passed else '局部感知开发筛查未通过，按停止条件收束'
    table='|目标条件|接受且定位≤5m|两正例平均误差（含拒绝）|来源宏平均BA|\n|---|---:|---:|---:|\n'
    source='|条件|来源|窗内接受|窗外误接受|异源误接受|BA|\n|---|---|---:|---:|---:|---:|\n'
    for arm,m in s['metrics'].items():
        table+=f"|{arm}|{m['positive_accepted_and_within5m']}/2|{m['positive_MAE_all_predictions_m']:.3f}m|{m['source_macro_balanced_accuracy']:.2%}|\n"
        for k,v in m['by_source'].items():
            cs=v['counts'];source+=f"|{arm}|{k.rsplit('_',1)[-1]}|{cs['inside']['accepted']}/1|{cs['outside']['accepted']}/4|{cs['unrelated']['accepted']}/5|{v['balanced_accuracy']:.2%}|\n"
    effects='|Full相对|正例恢复/损伤|窗外恢复/损伤|异源恢复/损伤|BA增益|两来源增益|\n|---|---:|---:|---:|---:|---|\n'
    for arm,c in s['contrasts'].items():
        parts=[f"{c['by_class'][k]['recovered']}/{c['by_class'][k]['harmed']}" for k in ('inside','outside','unrelated')]
        effects+=f"|{arm}|{'|'.join(parts)}|{100*c['source_macro_balanced_accuracy_gain']:+.2f}pp|{', '.join(f'{100*x:+.2f}pp' for x in c['source_deltas'])}|\n"
    positives='|条件|来源|接受|定位误差|\n|---|---|---|---:|\n'
    coverage='\n'.join(f"- {arm}：20条中拒绝{sum(not r['accepted'] for r in s['detailed'] if r['arm']==arm)}条，拒绝覆盖{sum(not r['accepted'] for r in s['detailed'] if r['arm']==arm)/20:.1%}。" for arm in ('Full','Disabled','Mean','Wrong'))
    for r in s['detailed']:
        if r['relation']=='inside':positives+=f"|{r['arm']}|{r['source'].rsplit('_',1)[-1]}|{'是' if r['accepted'] else '否'}|{r['error_m']:.3f}m|\n"
    gates='\n'.join('- '+k+'：'+('通过' if v else '未通过') for k,v in s['gates'].items())
    baselines='\n'.join(f"- 来源{x['source'].rsplit('_',1)[-1]}：无目标父图中心误差{x['parent_center_error_m']:.3f}m，当前格中心误差{x['current_tile_center_error_m']:.3f}m。" for x in s['position_baselines'])
    report=f'''# CV-031：CCVPE局部感知与不匹配拒绝

**{status}。** 本批100预测是固定4个已消费工程来源上的有界开发筛查；只有0/1两校准来源与2/3两开发来源，不是100个独立地域，不产生主动导航SR。

## 固定方法与输入

复用CV030官方唯一KITTI/no_orientation_prior权重及工程已通过适配器；60416657参数全部冻结。目标照片固定letterbox256×1024，当前125×125航拍格放入625均值画布后变为512输入；模型不读取父图未知纹理、GPS、heading、源ID或目标位置。父图只由评测器解码裁格，模型前向仅接目标与当前格处理后的张量。普通照片FoV与局部稀疏画布不同于原论文测试输入，不宣称完整复现。

原CV029固定每参考目标格及四邻格，每查询配同分区两参考；20校准Full、20开发基础×Full/Disabled/Mean/Wrong=80。所有目标真值仅用于事后计算误差/关系类别。Mean原样读CV030仅0/1目标的六层均值，Wrong分区0↔1、2↔3不改答案，Disabled六匹配层归零；实际100预测中的控制路由另核通过。

定位为可见区域内512热图峰值，行优先处理完全相同最大值；拒绝分数为256最细raw matching跨yaw最大值及其可见区域最大值，不使用softmax作为存在概率。各尺度按自己的像素中心映射625原图，半开边界掩码；两分辨率不混用，不把未访问区域峰值当预测。输出ROI摘要可供零模型复核。

## 单次校准与固定验收

阈值τ={cal['tau']!r}，只用0/1的20条Full输出，评估{cal['candidates']}个固定候选；校准来源宏平均BA={cal['calibration_source_macro_balanced_accuracy']:.2%}。阈值、全部候选和输入SHA在第21次前向前保存，开发成绩不回流。接受以Python float64 score≥tau比较，所有控制共用。

BA是三类检测平衡准确率，**不含定位精度**；联合正例指标另要求接受且误差≤5m。全部拒绝BA=66.67%，全部接受BA=33.33%。不能把拒绝策略的高BA解释为定位成功。

{table}
{source}
拒绝覆盖：

{coverage}

{positives}
{effects}
每类恢复/损伤用同题联合正确性：窗内为接受且≤5m，窗外/异源为拒绝。完整来源重采样只有4种支持点（2来源有放回），JSON保存各控制支持值；样本仅2工程开发源，不作稳定独立优势推断。

必要门槛：

{gates}

无目标位置基线（独立于检测阈值）：

{baselines}

## 资源与复核

8旧图各重新解码一次；100完整管线/200骨干调用，无额外热身；CPU采样峰值{resource['peak_sampled_rss']/1024**3:.3f}GiB，总时长{resource['wall_seconds']:.2f}秒，4GiB/1800秒内。该时长含载入、解码、摘要与张量检查，不当模型速度基准。818张量前后SHA一致，0训练/GPU/下载/模型API/导航或6确认消费。

合成{b['tests']['count']}项与另实现复核通过：逐条读取保存ROI独立重算分数、峰值、坐标、唯一阈值、80分层判断、恢复损伤、目标控制、完整性与来源支持点。复核无原图解码/模型加载/forward；同Agent另一实现，不称独立人员审查。源码、原8图、唯一权重、Mean、协议与旧证据已于首个预测前绑定。

完整父图/局部的定量效果差异本轮无法补齐：CV030仅保存输出等价SHA，没有保存父图定位/分数峰值；本次没有预算重算父图。这里只验证局部结果，不把这一缺项推断成某一适配环节的因果证据。

## 结论边界与后续

{'可据本批结果设计下一项合法观测下的决策机制，仍须独立冻结，当前不能直接升级自主导航。' if passed else '按预登记停止当前CCVPE局部稀疏画布候选，不在这4源上调阈值、裁剪、尺度或扩参加训追过线。这个失败只约束该完整适配方案和当前小样本分工，不能判定论文原方法或主动探索方向无效。'}本批校准前两图偏暗、开发两图较亮的分工偏置已预先披露，未换组；未消费MM-GAG6封存来源。下一轮优先把本次边界作为平台实验模式证据展示，研究新方案需要完整可见窗口/空间监督的明确机制，不再重复原余弦或旧小头试错。

平台仍提供已交付的地面目标人工连续观测与保存功能，本轮未新增GUI，也未接入不合格自主策略。旧M0/S2S3S4和平台4.11入口保持。

[冻结执行](冻结执行绑定.json) · [单次阈值](唯一校准阈值.json) · [全部分层与恢复损伤](开发筛查汇总.json) · [另实现复核](另实现复核.json) · [资源](资源回执.json) · [执行计数](执行完成.json)。恢复只读上述材料，不重跑freeze/execute/worker/写盘audit/publish。本轮无网络/代理、C盘任务文件、Git提交或推送。
'''
    with (OUT/'README.md').open('x',encoding='utf-8') as f:f.write(report)
    put('下一步边界.json',dict(screen_passed=passed,candidate_closed=not passed,navigation_admitted=False,
        research='new mechanism requires separate protocol;no tuning this consumed screen',platform='surface saved perception evidence and limits without automatic navigation',
        full_parent_quantitative_comparison_missing=True,MMGAG_reserved_consumed=0))
    path=OUT.relative_to(ROOT).as_posix();report_path=path+'/README.md'
    p=ROOT/'项目导航/实验索引.json';original=p.read_bytes();idx=read(p);old=idx['batches'];assert len(old)==155
    idx['date']='2026-10-23';idx['batches']=old+[dict(dataset='OpenCVL四已消费工程组/2校准2开发',kind='真实局部感知开发筛查',name='CCVPE局部感知开发筛查_v1',category='CV-031',status=status,path=path,
        scope='20Full校准+80目标对照；无导航SR',reports=[report_path],evidence=[path+'/开发筛查汇总.json',path+'/另实现复核.json',path+'/冻结执行绑定.json'])]
    assert p.read_bytes()==original;p.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'docs/维护/版本生命周期.json';original=p.read_bytes();life=read(p);assert life['current']==b['current'];life['updated']='2026-10-23'
    life['cv031_local_perception']=dict(status=status,report=report_path,replaces='CV030 next-screen proposal only;no default/runtime replaced',
        depends_on=['CCVPE真实工程准入_v1','CV029 protocol','single official checkpoint'],recovery='sealed outputs;no repeat inference',candidate_closed=not passed,navigation_admitted=False)
    assert p.read_bytes()==original;p.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;original=p.read_bytes();text=p.read_text(encoding='utf-8-sig');h,_,rest=text.partition('\n');prefix='' if name=='README.md' else '../'
        line=f'[CV-031 · 局部感知开发筛查]({prefix}{report_path})：{status}；20校准+80目标对照已完成，无导航SR。'
        assert p.read_bytes()==original;p.write_text(h+'\n\n'+line+'\n\n'+rest,encoding='utf-8')
    m=s['metrics']['Full'];p=ROOT/'docs/PROJECT_STATE.md';original=p.read_bytes();text=p.read_text(encoding='utf-8-sig');h,_,rest=text.partition('\n')
    state=f'''\n\n### CV-031：{status}（2026-10-23）

[全部证据](../{report_path})。固定原4工程来源0/1校准、2/3开发，20Full校准+80目标对照完成；τ={cal['tau']!r}。Full联合正例{m['positive_accepted_and_within5m']}/2，来源宏BA={m['source_macro_balanced_accuracy']:.2%}，两正例均值误差{m['positive_MAE_all_predictions_m']:.3f}m（包括拒绝）。BA只衡量接受/拒绝，不等于定位准确率。逐源误接受/恢复损伤/对照详见报告；未更换来源或调阈值追分。

{b['tests']['count']}合成测试、100ROI另实现重算和六层真实控制核验通过，同Agent非独立人员；CPU峰值{resource['peak_sampled_rss']/1024**3:.3f}GiB、{resource['wall_seconds']:.2f}秒；8图各一次解码、100管线/200骨干、818冻结张量保持。CV030完整父图只有等价SHA，缺父图定位摘要，未新增调用补算。没有训练/下载/导航或新SR，MM-GAG6确认/原M0与平台4.11保持；当前156索引、旧155值保留。{'当前候选仅开发感知通过，尚不能导航。' if passed else '按停止条件关闭当前局部稀疏CCVPE候选，不调相同因素追分；平台证据展示可继续。'}\n'''
    assert p.read_bytes()==original;p.write_text(h+state+'\n'+rest,encoding='utf-8')
    verify()
    put('阶段封存.json',dict(files=[entry(p) for p in sorted(OUT.iterdir()) if p.is_file()],sources=b['sources'],screen_passed=passed,
        audit_passed=True,current_runtime_preserved=True,index=156,seal_itself_excluded=True))
    size=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file());assert size<=16*1024**2
    print(json.dumps(dict(published=True,screen_passed=passed,index=156,bytes=size),ensure_ascii=False))


if __name__=='__main__':publish()
