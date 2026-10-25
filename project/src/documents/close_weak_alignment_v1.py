"""CV-008 final-only report and seal; refuses overwritten evidence."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'project/src'))
from tools.run_weak_alignment_v1 import OUT,ROOT,read,put,entry,verify


def main():
    audit=read(OUT/'核验/另实现复核.json');assert audit['passed']
    summary=read(OUT/'感知对照汇总.json');state=read(OUT/'执行状态.json');log=read(OUT/'训练日志.json')
    binding=read(OUT/'首forward前绑定.json');assert state['state']=='completed' and state['training_steps']==500
    for key in ['files','sources','models','protected','previous_seals','raw_inputs']:verify(binding[key])
    passed=summary['effect_gates_passed'] and audit['passed']
    decision={'task':'CV-008','engineering_passed':True,'alternate_audit_passed':True,
        'effect_gates':summary['effect_gates'],'all_required_gates_passed':passed,
        'status':'development_candidate_only' if passed else 'closed_failed_no_sweep',
        'default_upgraded':False,'ground_navigation_validated':False,'reserved_groups_consumed':0,
        'stop_rule':'No rank/source/temperature/loss/encoder search. No navigation or deployment without new protocol.',
        'training_semantics':'weak same-scene common-cell proxy, not verified camera location'}
    put('核验/门槛判定.json',decision)
    table=[]
    for ds,label in [('weak_dev','Swiss弱开发125窗/5场景'),('MMGAG','MM-GAG原16对/10开发组')]:
        for arm in ['Identity','Rank8Final']:
            for cond in ['Full','Mean','Wrong']:
                x=summary['counts'][ds][arm][cond]
                table.append(f"| {label} | {arm} | {cond} | {x['top1']}/{x['planned']} | {x['top5']}/{x['planned']} | {x['median_true_rank']:g} |")
    group_table=[]
    for ds,groups in summary['per_source'].items():
        for name,x in groups.items():
            before=x['Identity']['Full'];after=x['Rank8Final']['Full']
            group_table.append(f"| {ds} | {name} | {before['top1']}/{before['planned']} | {after['top1']}/{after['planned']} | {after['top1']-before['top1']:+d} |")
    gates='\n'.join(f"- {k}：{'通过' if v else '未过'}" for k,v in summary['effect_gates'].items())
    freeze=read(OUT/'编码器冻结与像素证据.json')
    peak=max(e['peak_cuda_allocated_bytes'] for e in freeze['encoders'])
    report=f'''# 弱场景对齐头：工程与固定训练开发结果

CV-008 / 2026-10-17。**工程与另实现复核通过；{'达到开发候选门槛，仍不是导航验收。' if passed else '效果必要门槛未全部通过，按停止条件收束该因素。'}** 本轮没有导航SR、默认升级或保留组确认。

## 唯一因素和目标边界

固定8192参数秩8头 `h(g)=normalize(g+UVg)`，U为512×8、V为8×512，无偏置。g为归一化地面512维特征；头不接收源名、位置、答案、候选图库或未访问特征。CLIP视觉编码器和Sat2Cap各87849216参数，原中心裁剪/224处理/余弦冻结，不结合局部token或填边变体。

辅助SwissMonuments350行来自14张不同地面照/场景，固定9fit、5弱开发；25个平移窗不能算25个独立查询。训练每步9个查询，各对应81个同场景图块，正格是发布同景共同格的弱代理，不是测绘拍摄点/对象占地真值。GrandeDixence25行全部隔离；MM-GAG原拍摄点代理格及16已知开发/20保留角色不变，未把开发资料挪去训练，地理隔离未知。

InfoNCE温度0.1，AdamW lr0.001/wd0.0001，梯度范数裁到1，无调度，seed0固定500更新/4500监督使用，仅评最终权重，不择最好轮次。训练CPU耗时{log['seconds']:.2f}秒（不是控制速度基准）；fit损失{log['log'][0]['loss_before_update']:.6f}→{log['final_fit_loss']:.6f}，这只说明训练拟合，不能替代开发迁移。

## 同题结果：Top-k不是导航成功率

| 队列 | 头 | 目标条件 | Top1 | Top5 | 目标排名中位数 |
|---|---|---|---|---|---|
{chr(10).join(table)}

Full使用真实目标；弱Mean固定为9fit目标特征均值；MM-GAG Mean保留原16目标均值定义。Wrong为冻结的跨场景循环/原MM-GAG映射。所有条件对两个头一视同仁。共846条排序、21150分数；训练和评分不将完整候选库交给导航Agent，本轮只是离线关联工程。

## 分来源与恢复/损伤

| 队列 | 来源组 | Identity Full Top1 | Final Full Top1 | 净变化题数 |
|---|---|---|---|---|
{chr(10).join(group_table)}

完整Full相对Mean/Wrong、Final相对Identity的Top1/Top5恢复与损伤均在[机器汇总](感知对照汇总.json)的contrasts字段，不能只报净增或最好来源。重复窗口和三条件不增加地理样本，五弱场景及十MM开发组分开报告，不外推为新来源确认。

## 预登记门槛

{gates}

另实现复核：通过。全部必要项{'通过，但只允许称开发候选' if passed else '未通过，关闭该因素'}；不能按训练损失下降或单一Top5改善升级默认。下一项须另立协议，本轮不追加更新、扫秩/温度/损失、换编码器或使用20保留组。

## 工程、冻结与恢复证据

身份初值≤1e-6、初步U梯度非零/V初步梯度为零的预期性质、解析梯度、序列化中断恢复后参数及Adam矩逐位一致、仅头参数进入优化器、权限/窗口边界和7测试通过。实际编码1148图（14地面+1134唯一300px航拍格）、144调用；每编码器首8图工程子集计入总数，成功特征复用。两编码器全张量SHA前后相同，无梯度；峰值分配{peak/1024**2:.2f}MiB≤2GiB。

首forward前绑定实际源码、原件、队列、配置和全部模型SHA；全部特征落盘SHA、初始头和fit监督绑定先于首更新。特征float32单份保存；初始/最终头与一个500步Adam恢复文件保留，未复制原图或骨干。原MM-GAG缓存只读复用，0新MM-GAG/保留像素、下载、API、导航或生产服务操作。

另实现NumPy残差投影/L2/余弦/损失与排序、原CV-006图块SHA重建125窗、独立PIL重建1148输入张量、分来源/恢复损伤/门槛计数与哈希保护通过，分数最大差{audit['maximum_score_error']:.3g}。这是同一Agent的另一实现复核，不是独立人员审查。旧封存/默认/模型/Word保持。

平台侧本轮交付为可复用目标对齐模块、固定执行器/单份缓存/可审计排序结果；生产GUI不开放尚未证实的地面导航。现有航拍M0与V4.5继续保留。

## 复核与后续

仓库根只读运行 `D:/PYTHON/python.exe project/src/eval/audit_weak_alignment_v1.py`；不加--save，不重跑freeze/run或本报告生成器覆盖原件。中断恢复先看执行状态、每批调用账本和特征覆盖，本轮已正常结束，无需续训。初始/最终头为推理和比较原件，recovery.pt是唯一500步活动恢复状态。

{'后续可论证一个独立导航接入协议；开发门槛不等于正式新来源或导航验收。' if passed else '本因素到此收束；下一项先只读归纳fit、弱开发与MM开发的迁移差异，判断现有弱标签是否足以支撑拍摄点目标，不再盲目训练追分。'}
'''
    put('工程训练与目标对照报告.md',report)
    put('README.md','''# 弱场景对齐头工程与训练

CV-008。一次固定500更新已完成，最终门槛见[报告](工程训练与目标对照报告.md)与[判定](核验/门槛判定.json)。本批只是弱场景关联开发，不是跨视角导航SR或默认升级。

- [首forward绑定](首forward前绑定.json) / [首更新特征绑定](首更新前特征与训练绑定.json)
- [编码器冻结/调用与像素证据](编码器冻结与像素证据.json) / [执行状态](执行状态.json) / [训练日志](训练日志.json)
- [逐题排序](逐题排序结果.json) / [来源与恢复损伤](感知对照汇总.json)
- [另实现复核](核验/另实现复核.json) / [阶段封存](阶段封存.json)
- [可复用目标头](../../../../project/src/agents/weak_alignment_head_v1.py) / [固定执行器](../../../../project/src/tools/run_weak_alignment_v1.py)

只读复核：`D:/PYTHON/python.exe project/src/eval/audit_weak_alignment_v1.py`。不盲目重跑freeze/run/报告。9fit5弱开发、原16MM开发与20保留角色保持；原件/默认/生产GUI/Word不改。总≤40MiB，特征单份、初始/最终小头与一个恢复状态；不复制骨干或原图。
''')
    files=[entry(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['README.md','阶段封存.json']]
    put('阶段封存.json',{'task':'CV-008','files':files,'sources':binding['sources'],
        'sealed_bytes':sum(e['bytes'] for e in files),'default_upgraded':False,'navigation_SR':None,
        'mutable_entries_excluded':['README.md'],'recovery':'read-only audit; run/freeze/close consumed'})
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=40*1024**2
    print(json.dumps({'report_and_seal_complete':True,'files':len(files),'candidate_passed':passed,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}),flush=True)


if __name__=='__main__':main()
