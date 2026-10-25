"""Build the CV-015 evidence close from saved statistics after the alternate audit."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/MM-GAG'
OUT=BASE/'跨视角阶段证据收束_v1'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(name,text):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(text,encoding='utf-8',newline='\n')


def main():
    stats=load(OUT/'证据重计汇总.json');audit=load(OUT/'核验/另实现复核.json')
    if not audit['passed']:raise ValueError('Alternate audit required before evidence claims')
    current=stats['current']
    history=[]
    for series in stats['historical_series']:
        c=series['counts'];n=c['Full']['planned']
        history.append(f"| {series['series']} | {series['label_semantics']} | {series['source_groups']} | {c['Full']['top1']}/{n} | {c['Mean']['top1']}/{n} | {c['Wrong']['top1']}/{n} | {c['Full']['top5']}/{n} |")
    for arm,c in current['counts'].items():
        if arm=='SharedDisabled':continue
        history.append(f"| CV-014 {arm} | 拍摄点代理 | 5 | {c['Full']['top1']}/125 | {c['Mean']['top1']}/125 | {c['Wrong']['top1']}/125 | {c['Full']['top5']}/125 |")
    sensitivity=[]
    for arm,conditions in current['target_sensitivity'].items():
        for condition,c in conditions.items():
            sensitivity.append(f"| {arm} | Full vs {condition} | {c['same_top1_cell']}/125 | {c['full_target_rank_better']}/{c['full_target_rank_worse']}/{c['full_target_rank_tied']} | {100*c['mean_top5_overlap']:.1f}% |")
    source=[]
    for info in current['per_source'].values():
        for arm in ('Identity','SwissRank8Frozen','MMRank8Final'):
            c=info['arms'][arm]['Full']
            source.append(f"| {info['source_id']} | {arm} | {c['top1']}/25 | {c['top5']}/25 | {c['unique_top1_physical_tiles']} | {c['modal_top1_windows']}/25 | {c['median_target_minus_best_negative']:.5f} | {c['median_top1_to_proxy_grid_L1']} |")
    write('跨视角阶段证据与失败边界.md',f'''# 跨视角阶段证据缺项与失败边界

CV-015 / 2026-10-18。结论：跨视角数据/编码/小头训练工程已经跑通，拍摄点代理的目标有效性仍未通过；不开放稳定地面目标导航，不升级M0，不把失败等同整个课题失败。当前没有满足准入条件的新模型候选。

## 各批次可比较的范围

| 批次/方法 | 标签语义 | 来源组数 | Full Top1 | Mean Top1 | Wrong Top1 | Full Top5 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(history)}

这是一张证据清单，不是累计提升曲线。CV-002/003/005与CV-008-MM复用原16对/10临时组；CV-008-weak是Swiss共同格代理的5已知弱开发场景；CV-014是另5个已分配开发来源，每个地面目标重复25个窗口。两种标签、数据域、队列和均值来源不同，不能把它们的百分比直接相减。所有表均为离线25候选关联，尚无跨视角导航SR；旧航拍S2/S3/S4验收另算。

## 当前数据究竟提供了多少监督

固定训练只有{current['fit_queries']}个不同地面查询/9个临时来源组/9个正航拍格，每组一条查询—正格关系；81候选和500次更新不增加不同正配对数。500次更新相当于4500次重复监督使用，训练损失{stats['fit_loss_first']:.6f}→{stats['fit_loss_final']:.6f}只证明拟合改善。

开发125窗是5个地面查询/5个正格/405个不同航拍候选。每来源25窗恰遍历目标的25种窗口位置；所有窗交集只有同一正格，候选并集81格。已有分数按物理格回连，重复物理候选分数差均≤1e−12。窗口平移改变竞争格和相对位置，没有带来新地面视角、拍摄点或独立地域。

## 模型有没有响应真实目标

| 方法 | 对照 | 同一个Top1候选 | Full真目标排名变好/变差/相同 | Top5集合平均重合比例 |
|---|---|---:|---:|---:|
{chr(10).join(sensitivity)}

排名会随目标条件改变，不等于正确利用目标；同一Top1也不单独证明完全忽略目标。新头Full Top1为0，Mean/Wrong/Disabled为0/1/4；Swiss冻结头Full Top1为2，Mean/Wrong为25/4。Swiss的94/125 Top5不足以支持用扩大候选集或挑最优来源替代原必要Top1与来源对照门槛。

## 各来源错误的可见表现

| 来源 | 方法 | Full Top1 | Full Top5 | 不同物理Top1格数 | 最常选格覆盖窗数 | 目标分数−最强负格中位数 | Top1到代理格L1中位数 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(source)}

这里的L1是事后对离线候选格的误差描述，单位为该9×9库的格，不是导航终局SG，也没有可靠米制换算。它不能定义新的成功半径、改标签或接受邻格；目标/候选索引只在离线统计器内使用。

## 已成立、尚不成立与反证

- 已成立：原编码器/预处理的工程兼容、张量冻结和仅8192参数头更新有先前封存/复核；本批另实现只检查已存决定与索引，未重做模型forward或工程试验。
- 已成立：当前同任务小头拟合改善没有转化为5来源开发精确格收益，8项效果门槛失败；新头对原始特征Top5恢复9/损伤36，对Swiss恢复0/损伤71。至少一个来源mmgag_079的Top5比原始特征高（11/25对7/25），不能写成所有来源一无所得，但不足以通过总体及目标门槛。
- 支持继续核查：单来源一条正配对、跨域/监督语义差异、拍摄点可观测性不足，都是合理解释；现有实验没有隔离这些变量，不能锁定某一原因，更不能由此推断必须增加模型参数。
- 反证：单纯换全局为局部token匹配或保留完整地面画面未使原16对达到精确格门槛；不再重复扫描裁剪、池化、温度、阈值或训练轮数。
- 标签边界：CV-004原16对仅证明中心映射，独立地理footprint、相机朝向和拍摄点/可见地标同格均未确认。CV-011新14代表通过质量/行列工程，不等于这些语义已核实。不能把原16审查结论数量移植到新14，也不能按预测分数重标正格。

## 缺项和单一后续方向

见[证据与缺项表](证据与缺项表.json)、[后续方向条件与停止规则](后续方向与准入条件.md)。当前优先方向是补足同一区域中不同拍摄点的关系证据；先在已消费的14个fit/dev代表内核查标签与可辨识性资料，不直接重训。该方向针对监督关系覆盖，不扩大骨干；本轮没有准入新的模型因素，也没有冻结或启动新训练实验。

工程实现与离线关联不能代替闭环方向决策、探索预算内到达、目标干预和新来源确认。6确认组24素材继续封存。地面模式的平台入口应显示“开发关联未通过”，支持查看证据；尚不能将候选检索包装为可用主动导航，见[平台接入边界](平台接入边界.md)。

## 复核和恢复

重新计数1817条既有排序，其中最新1250条/31250个已存分数；校验固定分母、原哈希破平局、25位置/物理映射、来源/重复查询、目标对照排名和误差统计。另实现用逐候选胜出比较和独立循环计数核对，不调用生产统计函数；同Agent另一实现，非独立人员审查。结果见[复核](核验/另实现复核.json)。

本批0原图/特征/权重读取、模型forward、新预测、训练、导航、API、下载和确认消费，≤3MiB，无大数据副本。[输入/源码绑定](输入与源码绑定.json)和[封存](阶段封存.json)可只读核对；本批collect/audit/close只运行一次，不覆盖重跑，模型效果结论不回写任何旧批次。
''')
    evidence=[
        {'id':'E01','claim':'编码器、行列映射、冻结与小头训练工程可执行','status':'verified_engineering_only','evidence':['../同任务对齐执行器工程_v2/README.md','../同任务拍摄点对齐执行_v1/核验/另实现复核.json'],'missing':'不能代替目标关联和真实导航有效性'},
        {'id':'E02','claim':'地面照片可稳定区分局部拍摄点代理格','status':'not_demonstrated_effect_gates_failed','evidence':['../同任务拍摄点对齐执行_v1/核验/最终门槛判定.json','证据重计汇总.json'],'missing':'真实目标需稳定胜Mean/Wrong/Disabled及同题基线'},
        {'id':'E03','claim':'训练关系覆盖充分','status':'not_demonstrated','evidence':['证据重计汇总.json','../同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json'],'missing':'9个fit查询/正格；缺少同域同区域多拍摄点对照及更丰富且可信的正配对'},
        {'id':'E04','claim':'拍摄点标签已独立地理校核且可从照片观察','status':'unverified','evidence':['../目标格定义与视觉尺度审查_v1/审查汇总.json','../同任务首批图像轴向预检_v1/README.md'],'missing':'原16可见地标关系未知；新14质量/轴向过不等于地理footprint/拍摄方向和目标可观测性过'},
        {'id':'E05','claim':'跨视角主动探索与导航SR已验证','status':'not_run','evidence':['../同任务拍摄点对齐执行_v1/执行完成.json'],'missing':'逐步公开观察下目标判断/方向或访问后识别→真实动作→预算内到达→同题目标对照'},
        {'id':'E06','claim':'新来源泛化/统计单位充分','status':'not_confirmed','evidence':['../同任务来源用途账本_v1.json','证据重计汇总.json'],'missing':'6确认组未用；5开发查询与125重复窗不可冒充独立区域确认'},
        {'id':'E07','claim':'失败唯一由标签错误或骨干容量引起','status':'not_identified','evidence':['跨视角阶段证据与失败边界.md'],'missing':'未隔离标签/数据域/监督覆盖/表征，禁止按分数重标或直接扩参'},
        {'id':'E08','claim':'航拍平台和原协议仍可用','status':'preserved_not_retested_here','evidence':['../../../../平台/实验工作台_v4/README.md'],'missing':'现有航拍通过结论不外推到地面模式；本轮无生产GUI或服务变更'},
    ]
    write('证据与缺项表.json',json.dumps({'task':'CV-015','entries':evidence,'new_candidate_admitted':False},ensure_ascii=False,indent=2)+'\n')
    write('后续方向与准入条件.md','''# 后续方向与准入条件：监督关系覆盖

当前没有准入的新训练/打分候选。唯一保留方向为“在同任务可信配对上增加同区域不同拍摄点的关系监督”，不是再次调秩、轮数或阈值；证据只是每个fit来源仅一查询/一正格、旧弱场景收益未迁移、新同任务头开发失败，尚非该因素的因果证明。

下一项先做有界资料审查：只使用CV-011/014已消费的9fit＋5dev共14代表及其已有影像/元数据，最多14条来源审查记录，必要时每来源一地面/中心格/已有拼图，共最多42幅已消费素材；不使用原6确认或同组未消费素材。本CV-015没有打开这些像素，也没有授权新模型运行。图审需另认领任务、绑定素材、保存可核查观察，最多一次，不因不清楚就反复换尺度/改目标或换来源。

审查应区分：代码/像素中心约定、独立坐标→影像范围映射、相机朝向与拍摄点可见性、同地理窗口内是否确有两个以上可核实的不同查询/正格。没有坐标边界、无法判断或只有单正配对时写“未知/缺项”；保持原目标语义，不能把照片中的远景地标重标成拍摄点。

只有能在fit来源内部取得可核查的同域多查询/不同正格、且不泄漏dev/确认，才考虑另立一个数据关系因素：保留8192参数头和现有冻结骨干，固定相同样本/更新预算与同题基线，比较关系覆盖方案。具体配对清单、采样/损失、开发门槛、训练资源和停止条件必须在看到新模型结果前另案冻结；当前不具备训练准入，不把本页当执行协议。

如果14来源一次审查仍不能补足标签/关系证据，就收束当前可用素材的小头训练支线；总结为样本/目标可观测性与验证边界，平台保留航拍稳定功能和地面证据浏览。不要消耗6确认组来寻找可训练样例，不要求作者私有资料，不新增大模型或反复训练追分。

后续闭环即使获准，也只允许当前已访问航拍观测与地面目标特征；候选库、源ID、GPS、真实目标格和未访问影像始终由评测器掌握。先声明是方向预测还是访问后识别，再冻结导航与目标控制，不能把离线25格检索直接接成可用导航。
''')
    write('平台接入边界.md','''# 地面目标模式的平台边界

现有航拍M0和V4.5继续按各自已验证协议运行；本批没有修改GUI、服务、模型接口或原Word。本批可交付的是[证据报告](跨视角阶段证据与失败边界.md)与[机器可读汇总](证据重计汇总.json)，不是新的地面导航算法。

未来GUI需要明确分开“已有结果阅读”“离线关联诊断”“实验性逐步导航”。当前地面模式仅可展示前两类已存结果，并显示“目标有效性未通过；无跨视角导航SR”，不能显示默认可用或复用航拍通过徽标。报告中的Top-k/L1误差不能标成导航SR/SG米数。

未来执行内核的允许输入为：给定地面照片清除元数据后的RGB/既定特征、当前已访问航拍图/特征、公开动作历史/预算。离线25候选评分、真实目标索引/距离、源名/GPS、全图及未访问特征不得进入导航Agent。用户全图展示只在观察者层，不能改变Agent输入。

失败边界应随结果保存和导出，至少含task/批次、来源角色、目标语义、planned分母、不同查询/来源数、模型/特征/代码绑定、目标控制、开发或确认标识。CV-015所有汇总都是事后研究证据，不提供新的Agent策略输入。
''')
    write('README.md','''# 跨视角阶段证据收束（CV-015）

只重计已有结果，未生成新目标预测。小头工程已跑通，地面拍摄点关联效果尚未通过；当前没有准入的新模型候选，保留6确认组。

- [阶段报告/错误表现与边界](跨视角阶段证据与失败边界.md)
- [证据缺项表](证据与缺项表.json) / [离线重计JSON](证据重计汇总.json)
- [唯一后续方向及停止条件](后续方向与准入条件.md) / [平台接入边界](平台接入边界.md)
- [输入及源码绑定](输入与源码绑定.json) / [另实现复核](核验/另实现复核.json) / [封存](阶段封存.json)

0原图/特征/权重读取、模型forward、新预测、训练、导航/API/下载及确认消费。本批只保存小型文档/JSON，≤3MiB；不复制模型、特征或大数据。所有脚本已消费，不覆盖重跑；恢复只读核对封存SHA和已存JSON。旧默认/Word/生产GUI保持，旧批次原件原址保留。
''')
    binding=load(OUT/'输入与源码绑定.json')
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='README.md':
            data=p.read_bytes();files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if total>3*1024**2:raise ValueError('Output cap exceeded')
    for item in binding['inputs']+binding['sources']+binding['protected_current_files']:
        data=(ROOT/item['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError(('Changed original',item['path']))
    write('阶段封存.json',json.dumps({'task':'CV-015','files':files,'sources':binding['sources'],
        'sealed_bytes':sum(e['bytes'] for e in files),'mutable_entries_excluded':['README.md'],
        'new_model_candidate_admitted':False,'new_model_forward_calls':0,'new_training_steps':0,'new_navigation_SR':None,
        'recovery':'read-only SHA/JSON checks; collection/audit/close already consumed'},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'sealed_files':len(files),'before_seal_bytes':total,'candidate_admitted':False},ensure_ascii=False))


if __name__=='__main__':main()
