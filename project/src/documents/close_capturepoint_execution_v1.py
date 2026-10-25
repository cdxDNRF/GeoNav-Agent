"""Close engineering or a later independently authorized full run, without changing old seals."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from tools.run_capturepoint_execution_v1 import ENG,RUN,PROTOCOL,read,entry,put,verify


def close_engineering():
    result=read(ENG/'工程结果.json');audit=read(ENG/'核验/另实现复核.json');assert result['engineering_passed'] and audit['passed']
    b=read(ENG/'首forward输入源码绑定.json')
    for k in ['sources','inputs','models','baselines','protected']:verify(b[k])
    synthetic=read(ENG/'核验/合成梯度与恢复.json');env=read(ENG/'运行环境.json')
    rows=[]
    for kind in ['ground','aerial']:
        before=read(ENG/'核验'/f'{kind}_编码前.json');after=read(ENG/'核验'/f'{kind}_编码后.json')
        rows.append(f"| {kind} | {before['parameters']} | {after['encoder_inputs']} | {after['encoder_calls']} | {after['peak_CUDA_allocated_bytes']/1024**2:.2f} | {before['tensor_sha']==after['tensor_sha']} |")
    put(ENG,'执行器与真实模型工程报告.md',f'''# 同任务对齐执行器与真实模型工程

CV-013 / 2026-10-18。**执行器与有限真实模型工程通过，尚未训练或计算开发成绩。** 当前用户范围是工程；完整500更新需要对应后续明确运行授权，旧CV-012研究协议不回写。6确认组本轮没有新增读取/编码。

本批实现目标专用头、固定三臂三目标/共享禁用评分、提取/500更新/1250排序调度、恢复与另实现审计。头只接单512维归一化目标，候选库/来源/GPS/正确格仅在离线损失或评测器；均值只取9个MM训练目标，Wrong按冻结5来源循环，Disabled不调用头。完整队列接口用内存随机特征测试，不生成真实开发成绩。

真实工程仅使用固定队列前8个fit地面图和前8个fit航拍格，16输入/2次编码，全部在未来1148输入/144调用预算之内。成功特征以不可变分片保存，今后只剩6地面+1126航拍＝1132输入/142次调用。工程分片不会被全量写入覆盖；原图和编码器不复制。没有访问5开发来源的原图或输出其匹配排名。

| 编码器 | 参数数 | 图像输入 | forward次数 | 分配峰值MiB | 全张量SHA前后相同 |
|---|---|---|---|---|---|
{chr(10).join(rows)}

设备{env['device']}，两视觉编码器顺序加载，各87849216参数冻结，无梯度。张量SHA在首forward前落盘，编码后立即保存对照/显存/像素证据；使用追加事件，不替换共享状态文件，避免先前工程失败丢失日志。原始SHA和原PIL中心裁剪/Sat2Cap处理固定，首forward前已绑定源码/输入/模型/旧权重。

合成身份误差{synthetic['identity_max_abs']:.3g}、解析梯度误差{synthetic['analytic_gradient_max_abs']:.3g}，序列化恢复后参数与Adam状态逐位相同，优化器只含8192参数头；合成更新不使用真实MM样本或修改既有权重。真实8fit目标还验证了Identity/Swiss两头的Torch与另一NumPy算子及单位范数，最大差{audit['head_operator_max_abs_error']:.3g}，未做目标排序或开发loss。新候选头初值U0/Vseed0独立保存，完整训练会重新初始化，不续接合成或Swiss优化器。

7项工程测试覆盖单目标权限、禁用不调用头、隐藏答案不影响分数/Mean不取开发、1250合成记录及125共享禁用、梯度恢复、未授权全量拒绝及来源角色变化检查。另实现只重新解码同一16幅已用fit素材，重建像素张量SHA和残差算子，编码器调用0；不是第二套编码器推理或独立人员审查。

后续完整执行入口已实现，但必须提供明确的全量运行授权文件：绑定本CV-012协议SHA、最大500更新/1148总编码输入、不访问确认。之后另建独立执行批次，复用16特征分片、仅编码余1132输入；实际特征/fit专用张量SHA首更新前绑定，头CPU固定500更新/每步唯一恢复状态、最终才做1250排序和目标门槛判定。工程成功不保证达到开发效果，不自动升级M0/GUI。原默认/模型/Word与全部旧判定保持，无新SR、API或导航。

工程输出≤8MiB（没有完整原图/模型副本），未来执行≤40MiB/特征≤32MiB/初始最终头及一个恢复≤1MiB；不堆多个完整优化器。任何中断先看事件/成功特征覆盖，禁止盲重跑当前prepare/engineer；训练或编码不自动重试，只有文件元数据替换有有限重试。6确认组继续封存。

只读工程核验：`D:/PYTHON/python.exe project/src/eval/audit_capturepoint_execution_v1.py engineering`（不加--save，会重新解码同16图，不增加编码器调用）；像素复核次数应另登记，未出现新疑点不要重复运行。prepare/engineer/本报告已消费。
''')
    put(ENG,'README.md','''# 同任务对齐执行器工程

CV-013。有限真实工程通过，16fit输入/2编码调用已成功保存，不训练、不计算开发成绩。

- [工程报告](执行器与真实模型工程报告.md) / [工程结果](工程结果.json)
- [工程范围](工程范围冻结.json) / [首forward绑定](首forward输入源码绑定.json) / [运行环境](运行环境.json)
- [梯度/恢复合成检查](核验/合成梯度与恢复.json) / [像素/算子另实现](核验/另实现复核.json) / [封存](阶段封存.json)
- [完整执行器](../../../../project/src/tools/run_capturepoint_execution_v1.py) / [评分内核](../../../../project/src/eval/capturepoint_execution_scoring_v1.py)

6确认组不读，当前未真实训练/开发排序，无新SR。未来全量须另获对应授权并复用不可变16特征分片；仅剩1132输入/142调用、一次500更新/1250排序，单独批次，不覆盖本批。模型/原数据/默认/Word/GUI保持。

只读核验audit_capturepoint_execution_v1.py engineering（不加--save，会重新解码同16图，不增加encoder forward；仅出现新疑点时再执行并登记）。prepare/engineer/报告已消费，勿盲重跑。
''')
    seal(ENG,b['sources'],engineering_passed=True,training_steps=0)


def close_full():
    from data.capturepoint_alignment_protocol_v1 import gate
    audit=read(RUN/'核验/另实现复核.json');assert audit['passed']
    summary=read(RUN/'开发对照汇总.json');decision=gate(summary['gate_counts'],complete=True,engineering_and_audit=True)
    put(RUN,'核验/最终门槛判定.json',decision)
    rows=[]
    for arm,conditions in summary['counts'].items():
        for cond,v in conditions.items():rows.append(f"| {arm} | {cond} | {v['top1']}/125 | {v['top5']}/125 | {v['median_rank']:g} |")
    gates='\n'.join(f"- {k}：{'通过' if v else '未过'}" for k,v in decision['gates'].items())
    training=read(RUN/'训练日志.json');b=read(RUN/'首forward绑定.json');verify(b['sources']+b['models']+b['baselines']+b['protected'])
    put(RUN,'同任务训练与目标对照报告.md',f'''# 同任务拍摄点代理对齐的固定训练与开发对照

固定9fit查询/81候选、seed0/8192参数头/500更新已完成，骨干/原处理冻结。**{'达到预登记开发候选门槛' if decision['passed'] else '必要门槛未全部通过，按停止条件收束因素'}**；这不是导航SR或6来源独立确认。6确认组本轮未新增读取/编码/评测，旧默认/Word/GUI保持。

实际编码总1148输入/144调用，复用工程16输入/2调用，全量新增1132/142。首更新前实际特征与fit9监督已绑定，真实训练4500监督使用、CPU{training['seconds']:.2f}秒（非控制测速），fit loss{training['log'][0]['loss_before_update']:.6f}→{training['final_fit_loss']:.6f}；只评最终权重，没有择优或超参扫描。训练数据方案改变同时改变域/内容与监督语义，不能据结果分离标签的纯因果。

| 方案 | 目标条件 | Top1 | Top5 | 目标排名中位数 |
|---|---|---|---|---|
{chr(10).join(rows)}

125个窗口只有5个不同地面目标/来源，不是125个独立地区。Mean固定9个MM训练目标均值，Wrong固定跨来源循环，Disabled共享常数分数不调用头。完整恢复/损伤、5来源分层与3125来源配对区间见[开发汇总](开发对照汇总.json)，没有只报最好来源或净增。

{gates}

另实现复核通过，1250排序/31250分数最大差{audit['max_score_error']:.3g}，15组对照的Top1/Top5来源区间/恢复损伤一致；实际fit输入/损失、仅头更新和500步恢复状态核验过，同Agent另一实现，非独立人员审查。新执行严格按CV-012门槛，不降低标准、不重复编码成功输入、不换来源/秩/轮数追过线。当前{'只保留开发候选，确认/导航接入需新协议' if decision['passed'] else '因素收束，保留失败与适用边界，不消费确认追分'}。

单份特征、初始/最终小头和一个恢复状态保留，执行输出≤40MiB。没有新导航、API或下载，原M0/V4.5不改。只读复核：audit_capturepoint_execution_v1.py full（不加--save，无新增原图/encoder推理/训练）；execute/报告已消费，禁止盲重跑。
''')
    put(RUN,'README.md','''# 同任务拍摄点对齐执行

固定500更新/1250排序已完成，结论见[报告](同任务训练与目标对照报告.md)和[门槛](核验/最终门槛判定.json)。关联Top-k不是导航SR；6确认继续封存。

- [开发结果/来源/恢复损伤](开发对照汇总.json) / [逐题排序](逐题排序结果.json)
- [运行授权](运行授权.json) / [源码及工程分片绑定](首forward绑定.json) / [实际特征绑定](首更新前特征绑定.json)
- [训练日志](训练日志.json) / [另实现复核](核验/另实现复核.json) / [封存](阶段封存.json)

只读核验audit_capturepoint_execution_v1.py full（不加--save）；execute/报告已消费，不能覆盖重跑。旧M0/GUI/Word不改。
''')
    seal(RUN,b['sources'],development_passed=decision['passed'],training_steps=500)


def seal(out,sources,**flags):
    files=[entry(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name not in ['README.md','阶段封存.json']]
    put(out,'阶段封存.json',{'files':files,'sources':sources,'sealed_bytes':sum(e['bytes'] for e in files),
        'mutable_entries_excluded':['README.md'],'default_upgraded':False,**flags})
    print(json.dumps({'sealed':True,'files':len(files),'bytes':sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),**flags}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['engineering','full']);a=p.parse_args()
    close_engineering() if a.stage=='engineering' else close_full()
