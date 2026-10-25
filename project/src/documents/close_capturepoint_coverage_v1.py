"""CV-018 deliver extra-source engineering and frozen design; never start training."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'DATA/processed_data/MM-GAG/三新增来源工程与配对覆盖协议_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    engineering=read(OUT/'像素工程汇总.json');audit=read(OUT/'核验/工程另实现复核.json');pa=read(OUT/'核验/协议另实现复核.json')
    if not engineering['all_required_passed'] or not audit['passed'] or not pa['passed']:raise ValueError('Both engineering and protocol audits required')
    protocol=read(OUT/'配对覆盖同预算冻结协议.json');scope=read(OUT/'首像素冻结范围.json');binding=read(OUT/'输入与源码绑定.json')
    table=[]
    for r in engineering['sources']:
        table.append(f"| {r['source_id']} | {r['raw_decodes']} | {r['ground_std16_max']:.3f} | {r['center_std16_max']:.3f} | {r['axis']['best']} | {r['axis']['scores']['row']:.4f} | {r['axis']['margin']:.4f} | 通过 |")
    write('三新增来源工程与同预算协议报告.md',f'''# 新增三训练素材工程核验与配对覆盖协议

CV-018 / 2026-10-18。**031、050、065三来源工程全部通过，9对与12对同预算研究设计已冻结；模型执行和训练尚未启动。** 原9配对失败、来源用途、旧权重/特征/默认与6确认保持；本批没有新Top-k或导航SR。

## 有限工程的实际范围

用户明确开始新增三素材核验后，再冻结对照。三来源均为原训练准备组的非代表，沿用原sat_5_5拍摄点代理标签。每源仅地面1幅、内圈9×9图块81幅、完整11×11拼图1幅，共249幅，各读字节/解码一次；031的小拼图副本不读取。没有重看旧14来源、开发/确认像素或同组其它素材，没有按成绩/外观替换来源。

| 来源 | 解码数 | 地面16px最大通道std | 中心格std | 最佳轴向 | row MAE | 次优间隔 | 必要工程 |
|---|---:|---:|---:|---|---:|---:|---|
{chr(10).join(table)}

原8探针/8行列翻转假设、row平均绝对误差≤8且次优间隔≥3、地面/中心16px最大通道std>1，门槛均从CV-011继承。原字节SHA和尺寸全部核对，地面EXIF转正/RGB清元数据；均匀负格仅诊断，不删除或调整监督。质量/行列通过不等于地理测绘真值或目标有效性通过。

工程另实现从保存16px缩略独立重算24轴向分数、通道方差、249回执/498消费事件和三源覆盖，最大轴向差{audit['max_axis_score_difference']:.3g}，0原图再解码。同Agent另一实现，非独立人员审查。

## 冻结的唯一因素和同预算含义

唯一因素为训练正配对覆盖9→12，仍9个训练临时来源组；新增031/050/065追加在原9查询后，不扩大骨干或8192参数秩8头。初始化仍seed0/U0/V正态.02，全局512D表征/原PIL处理、归一化余弦/温度0.1、AdamW lr0.001/wd0.0001/clip1、无调度、最终500步单权重，均不改变。没有新增损失或合作Agent。

两组均500次更新、每步batch9、4500次查询使用。12查询按零起点step t的(9t+j)mod12取j=0..8，每查询恰375次；9查询代入同式退化为原固定0..8批，每查询500次。不能把全12批×500=6000次暴露称作同预算，也不把重复epoch或平移窗算新配对。

## 队列与输入权限

固定[队列](固定训练与开发队列.json)保字节值复用原14地面/1134航拍引用前缀，追加3地面/243航拍引用，共17/1377。训练地面索引为原0..8加14..16；原5已知开发目标和125个平移窗及Wrong映射原值保持。Mean固定原9训练目标均值，所有头共享；Disabled常数记录复用，不把零向量送进头。

索引、正格40、源名/GPS和完整候选库只属于离线训练器/评测器元数据。公开小头只接收给定地面目标的512向量，未来在线策略也不能拿未访问航拍库、真实目标格/距离或完整拼图作输入。原6确认组24素材不在新增工程/训练/开发队列；12素材不是12个独立地理组，仍9fit组、5个已知开发来源。

## 同题开发与必要门槛

原Identity/Swiss冻结头/MM9最终头各Full/Mean/Wrong及共享Disabled共1250条排序只读复用，候选12头新增Full/Mean/Wrong375条，合计1625计划记录；不重新跑旧基线择优。125窗仍只有5个不同开发目标，不是新来源确认，也没有导航SR。

全部必要门槛：1625记录完整且工程/另实现通过；候选Full Top1≥32/125；Full分别胜Mean/Wrong/Disabled各≥16题，胜Identity/Swiss各≥5题；至少3/5来源Full同时胜双目标控制、至少3/5同时胜两旧基线；另要求胜原MM9 Full≥5题且至少3/5来源有正收益。判定函数为[data/capturepoint_coverage_protocol_v1.py](../../../../project/src/data/capturepoint_coverage_protocol_v1.py)的coverage_gate，12项布尔门槛均不可缺；Top5/排名/恢复损伤/全来源及5组配对区间必报，区间不新增隐形门槛。

任一必要门槛未过即收束该覆盖因素，不换来源/阈值/秩/损失/轮数/种子追分，不消费确认，不升级默认；通过也仅是已知开发关联候选，不能宣称已实现地面主动导航。

## 执行准入和资源保留

研究设计已冻，三源像素工程通过；[协议](配对覆盖同预算冻结协议.json)的actual_executor_binding/actual_feature_binding/candidate_final_checkpoint仍为null，training_authorized/model_forward_authorized为false。当前只有取样/队列/门槛纯函数和像素预检器，没有新的实际编码/训练执行器；现有旧执行器固定14来源/全9批，不能直接运行来代替12覆盖方案。

下一项实现并绑定真实12覆盖执行器，先验证实际batch取样、梯度/恢复、编码器冻结、输入权限和缓存复用。旧原始特征/权重/1250排序只能在SHA及图像处理/模型身份匹配后复用；当前仅保存引用SHA，没有加载它们。真实新增编码器输入上限246/批8最多32调用、CUDA分配≤2GiB/不足走CPU，头训练CPU≤30分钟；按相同原图SHA复用35个旧航拍向量须另核完整处理/编码器身份，不能只凭路径。

未来独立执行输出≤8MiB、特征≤2MiB、初始/最终头＋一个恢复状态≤1MiB，当前尚未创建该目录。实际特征与12训练张量/原9均值应先于首更新绑定；本轮不启动模型/训练或自动消费下一阶段预算。旧冻结false仅作历史状态，新实际执行需对应后续运行准入，不能凭设计文件盲重跑。

## 本批可追溯交付

原图249次SHA/解码、498追加事件、三份逐源回执和单份16px缩略；没有完整原图或模型副本。协议另实现核17/1377引用、12训练查询/9组、原125开发窗不变、四阶段循环的4500次暴露及所有原门槛继承。测试记录见[核验](核验/边界测试记录.json)；原模型/默认/Word与来源角色保持。

本批0模型forward/特征或权重读取/新分数/训练/导航/API/下载/确认消费，没有新SR。平台侧交付是三源准入JSON和固定协议/任务/门槛，可供后续内核接入；生产GUI和服务未改。恢复仅核[源码输入绑定](输入与源码绑定.json)、回执/保存缩略/协议/封存SHA，freeze/preflight/protocol/audit/close均已消费，不覆盖重跑或盲重读249原始影像。
''')
    write('README.md','''# 三新增来源工程与配对覆盖协议（CV-018）

031/050/065全部工程通过，9对与12对同8192参数、500更新/batch9/4500次使用的研究协议已冻结。当前没有新模型执行或训练结果，不升级默认，6确认保留。

- [工程与协议报告](三新增来源工程与同预算协议报告.md)
- [249像素范围](首像素冻结范围.json) / [工程汇总](像素工程汇总.json) / [新增工程授权](新增三来源工程授权.json)
- [同预算冻结协议](配对覆盖同预算冻结协议.json) / [固定任务与特征引用队列](固定训练与开发队列.json)
- [首像素源码/输入绑定](输入与源码绑定.json) / [协议绑定](协议源码与输入绑定.json)
- [边界测试](核验/边界测试记录.json) / [工程另实现](核验/工程另实现复核.json) / [协议另实现](核验/协议另实现复核.json) / [封存](阶段封存.json)

实际执行器/特征绑定尚未完成，旧9批执行器不能当12批执行器运行。下一项先实现/绑定真实执行器并做工程验证；本批不创建训练目录，不自动模型forward或训练。当前0新Top-k/SR/API/下载；原M0/V4.5/Word/旧正负结论保持。

所有freeze/preflight/protocol/audit/close已消费，恢复只读SHA/回执/缩略/JSON；不重复读取249原始影像，保存缩略≤512KiB、全批≤4MiB，无大模型/完整原图副本。
''')
    for e in binding['inputs']+binding['sources']+binding['readonly_math_dependencies']+binding['protected']:
        raw=(ROOT/e['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==e['sha256']
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='README.md':
            raw=p.read_bytes();files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());thumbs=sum(p.stat().st_size for p in (OUT/'核验').glob('*_缩略.npz'))
    if total>4*1024**2 or thumbs>512*1024:raise ValueError('Storage cap exceeded')
    write('阶段封存.json',{'task':'CV-018','files':files,'sources':binding['sources'],'readonly_math_dependencies':binding['readonly_math_dependencies'],
        'sealed_bytes':sum(e['bytes'] for e in files),'mutable_entries_excluded':['README.md'],
        'engineering_passed':True,'research_design_frozen':True,'actual_executor_bound':False,'actual_features_bound':False,
        'training_steps':0,'new_model_forward_calls':0,'new_navigation_SR':None,'confirmation_access':0})
    print(json.dumps({'sealed_files':len(files),'bytes_before_seal':total,'thumbnail_bytes':thumbs,'engineering_passed':True,'training_executed':False},ensure_ascii=False))


if __name__=='__main__':main()
