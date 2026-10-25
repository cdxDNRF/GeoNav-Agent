"""One-use CV-003 report builder. Saved results only, no model/navigation execution."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/局部语义匹配单因素验证_v1'
PREV=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,s):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s if isinstance(s,str) else json.dumps(s,ensure_ascii=False,indent=2))
def main():
    report=read(OUT/'局部匹配对照汇总.json');audit=read(OUT/'核验/另实现复核.json');resource=read(OUT/'运行资源.json')
    assert audit['passed'] and not report['candidate_passed'] and read(OUT/'执行状态.json')['state']=='completed'
    pairs=read(OUT/'冻结执行任务.json')['pairs'];rows=read(OUT/'逐对排序结果.json')['rows'];oldrows=read(PREV/'逐对排序结果.json')['rows']
    cur={(r['pair_id'],r['condition']):r for r in rows};old={(r['pair_id'],r['condition']):r for r in oldrows}
    per_pair=[];per_group={};transitions={k:[] for k in ['top5_recovered','top5_harmed','top5_retained','neither_top5']}
    for p in pairs:
        pid=p['pair_id'];before=old[pid,'Full'];after=cur[pid,'Full']
        name='top5_retained' if before['top5'] and after['top5'] else 'top5_recovered' if after['top5'] else 'top5_harmed' if before['top5'] else 'neither_top5'
        transitions[name].append(pid)
        per_pair.append({'pair_id':pid,'group_id':p['group_id'],'source_id_evaluator_only':p['source_id_evaluator_only'],
          'global_full_rank':before['true_rank'],'local_full_rank':after['true_rank'],'local_mean_rank':cur[pid,'Mean']['true_rank'],
          'local_wrong_rank':cur[pid,'Wrong']['true_rank'],'rank_change_positive_is_better':before['true_rank']-after['true_rank'],'top5_transition':name})
        g=per_group.setdefault(p['group_id'],{'planned':0,'global_full_top5':0,'local_full_top5':0,'local_mean_top5':0,'local_wrong_top5':0});g['planned']+=1
        g['global_full_top5']+=before['top5'];g['local_full_top5']+=after['top5'];g['local_mean_top5']+=cur[pid,'Mean']['top5'];g['local_wrong_top5']+=cur[pid,'Wrong']['top5']
    put('逐题与分组对照.json',{'task':'CV-003','per_pair':per_pair,'per_group_top5':per_group,'transitions':transitions,'posthoc_summary_only':True,'new_model_calls':0})
    tests=ROOT/'project/src/tests/test_crossview_local_match_v1.py'
    put('核验/源码与测试回执.json',{'task':'CV-003','command':'D:/PYTHON/python.exe -m unittest discover -s project/src/tests -p test_crossview_local_match_v1.py -v',
      'exit_code':0,'tests_passed':8,'latest_test_seconds':0.073,'test_source_sha256':sha(tests),'evidence':'actual command output in this Codex session; no invented test log',
      'bound_execution_sources':read(OUT/'输入与源码绑定.json')['sources'],
      'post_forward_document_source':{'path':Path(__file__).relative_to(ROOT).as_posix(),'sha256':sha(__file__)},
      'source_roles':'five execution/test files frozen before first forward; report builder added after scoring, no candidate changes'})
    table='\n'.join(f"| {p['pair_id']} | {p['group_id']} | {p['source_id_evaluator_only']} | {p['global_full_rank']} | {p['local_full_rank']} | {p['local_mean_rank']} | {p['local_wrong_rank']} |" for p in per_pair)
    group_table='\n'.join(f"| {k} | {g['planned']} | {g['global_full_top5']} | {g['local_full_top5']} | {g['local_mean_top5']} | {g['local_wrong_top5']} |" for k,g in per_group.items())
    gate_table='\n'.join(f"| {k} | {'通过' if v else '未过'} |" for k,v in report['gates'].items())
    peak=max(x['peak_cuda_allocated_bytes'] for x in resource['encoders'])
    put('局部语义匹配验证报告.md',f'''# 单幅图内部局部语义匹配验证

CV-003 / 2026-10-17。**工程与另实现复核通过，局部匹配候选效果未过，按停止条件收束。** 本批未训练、导航或升级默认；下表为离线精确目标格关联，不是导航SR。

## 固定条件与唯一因素

完全沿用CV-002的16对、10临时开发组、25格窗口、错误目标映射及两个冻结视觉编码器。20保留组（42套来源）没有编码或选型。来源组按重复图块/邻近坐标连通构造，不能称10个已证明地理独立区域。目标格来自原代码中心(5,5)与移窗约定，不是新增独立地理测绘。

地面CLIP ViT-B/32与航拍Sat2Cap各87849216参数，顺序加载，不扩大骨干、不新增可训练参数。地面EXIF转正及PIL/航拍归一化沿用已验证接口；去除图像元数据后只传像素。两个编码器的全局输出与CV-002共416张缓存最大差均为0，达到预先规定≤1e-5；CLS重构也达门槛。不是靠换处理流程或权重取得的不同排名。

唯一变化是全局余弦→单图内部7×7非CLS patch token对称MaxSim。最后层token使用各模型已有post_layernorm和visual_projection，不训练投影；投影原来主要服务CLS，局部跨视角对齐仅是假设。令A、G为逐token归一化后的49×512矩阵，S=AGᵀ：

`score(A,G) = 0.5 × (mean_i max_j S_ij + mean_j max_i S_ij)`。

先FP32归一化→FP16保存→从保存值读为FP64重新归一化，全部条件以相同保存精度评分。Mean用16张开发目标相同token位置求均值并归一化；Wrong沿用不同组的固定错误地面目标。排序沿用原CV-002哈希破平局，无目标格输入。

匹配内核每次只接收“单张当前航拍＋给定单张地面目标”的token矩阵，拒绝全图库张量。此处25格逐一评分属于离线评测器诊断；不允许在线Agent提前获得这25格图像/特征库、真实索引、GPS或全窗均值。

## 同题结果

| 方法/目标控制 | Top-1（计划16对） | Top-5 | 真实目标中位排名 |
|---|---:|---:|---:|
| CV-002全局余弦Full | 0/16 | 4/16 | 14 |
| CV-003局部Full | 0/16 | 4/16 | 10 |
| CV-003局部Mean | 0/16 | 3/16 | 10 |
| CV-003局部Wrong | 0/16 | 3/16 | 11 |

Full相对Mean/Wrong的Top-1恢复、损伤、净值均为0。相对全局Full，Top-1也均0；Top-5恢复4、损伤4、净0，原全局命中4题与新局部命中4题不重合。排名中位数14→10只是描述性变化：Mean中位数也是10，Wrong为11，不能据此宣称可靠利用了真实目标。小样本开发结果不支持整条跨视角路线无效，也不能证明故障唯一来自编码器或标签。

| 必要门槛 | 判定 |
|---|---|
{gate_table}

四项效果门槛分别要求Full Top-1≥4/16、比Mean与Wrong各多≥2题、至少3组同时优于两控制；均未过。0/10组在Top-1同时优于两控制。按冻结停止条件，不调池化、不扫层数/模型/阈值、不换样本、不消费保留组追分，不进入稳定导航。原CV-002失败保留，航拍M0和原S2/S3/S4等结论不变。

## 逐对与临时来源组

以下源名仅为离线复核，不进入匹配/执行接口。排名越小越好，候选总数25。

| 对 | 组 | 原来源 | 全局Full排名 | 局部Full | 局部Mean | 局部Wrong |
|---|---|---|---:|---:|---:|---:|
{table}

各组Top-1三控制全0。Top-5分组如下，不把多条件或多个样本当额外地理独立证据，也不为这个小型观察探针虚报区域置信区间。

| 组 | 计划对数 | 全局Full Top5 | 局部Full Top5 | 局部Mean Top5 | 局部Wrong Top5 |
|---|---:|---:|---:|---:|---:|
{group_table}

## 工程、资源和保存

8项边界测试涵盖手算公式、CLS排除、单图输入限制、量化保存精度、破平局和全局漂移停止。地面16＋航拍400＝416张，52次编码器forward；两个模型各87849216参数且完全冻结。CUDA峰值分配{peak}字节（{peak/1024**2:.1f}MiB），低于2GiB；峰值预留内存另存资源记录。执行器计时{resource['elapsed_seconds']:.2f}秒，不包含解释器/依赖导入与绑定检查，不作受控吞吐基准。

四份npy共{sum(x['bytes'] for x in resource['feature_files'])}字节；全批上限32MiB，原图和权重没有复制。0训练、导航、模型API、新下载和保留组编码。全局复现、局部token及原有源/输入/模型/默认/Word/PDF均有SHA绑定。

另实现以einsum/逐行列最大聚合复算1200个分数，最大差{audit['maximum_score_delta']:.3g}；逐题排名、门槛、分组、恢复/损伤与结果一致。浮点决策值先经另方程验证，再按保存值与原哈希规则核排名，避免不同运算顺序在极小差值处造成伪差异。同一开发Agent完成，不称独立人员审查。

## 交付和下一步

平台侧交付是可复用的单图匹配内核与像素输入隔离边界；**没有开放地面目标GUI或真实跨视角导航**。现有V4.5仍服务已验证航拍协议。未过候选不会包装成可用功能，也不必等所有研究问题解决才维护平台。

全局与局部两个候选均未证明精确格关联。下一步优先做一次有界的既有16对目标格标注/视觉尺度审查，核实“照片拍摄位置”与“照片中远处地标”是否被混为目标，再决定是否有训练小投影的依据。当前没有足够证据指定新的零训练打分公式，也不自动扩大骨干或启动训练。见[下一项审查方案](下一项目标格与视觉尺度审查方案.md)。

复核仅运行`D:/PYTHON/python.exe project/src/eval/audit_crossview_local_v1.py`，不加`--save`。prepare、runner和本报告生成器已消费，禁止覆盖重跑。若需新实验另建批次与首forward绑定。
''')
    put('下一项目标格与视觉尺度审查方案.md','''# CV-004拟议：先审查目标定义与视觉尺度

2026-10-17，仅下一步方案，未执行，未授权新的训练/导航/模型调用。CV-002全局与CV-003局部候选已收束，不调门槛或扫池化追分。

目的：用原16开发对和既有原图/源码/坐标说明，分开检查标签工程是否正确、照片拍摄位置与可见地标是否同一格、两视角是否有可观察共同语义。前两批仅证明取窗遵循原代码，不证明相机位置与画面中的所有目标物同格；本审查不能预设数据错。

输入固定CV-002/CV-003的16对、原映射/原拼接证据和保存排名；10开发组、20保留组不变。最多16份统一对照面板，总输出≤8MiB；代码/输入先绑定。0模型加载/forward、重编码、API、训练、导航、下载、新源或保留消费。可视化中的全图和真值只供离线诊断，不进入Agent。

逐对记录：数字行列与中心移窗是否一致；元数据说明目标是拍摄点还是可见地标（证据不足标未知）；目标格与近邻有无可辨语义、视角/遮挡/天空/远处物体是否造成尺度歧义。先列字段与判定依据，覆盖全部16对，不根据成功样本换队列。视觉判断不能伪装成精确地理测绘或独立人员标注，不能由模型分数反推正确标签。

验收为16对证据完整、所有未知显式登记、另实现几何复核及原件SHA保持，不以新SR追过线。若目标定义/尺度仍无可靠依据，明确收束并提出需要团队补充的标注，不能把训练无效归因给未经证实的数据问题。若标注可信但细格关联缺失，再另立一个小型关系投影的开发假设与同预算目标对照；保留原冻结骨干，新增参数/训练预算必须另列并得到授权。当前不硬造“必然有效”的候选，也不启动该训练。
''')
    put('README.md','''# 局部语义匹配单因素验证

CV-003 / 2026-10-17。工程复核通过，候选效果未过，已收束；离线关联Top-k不是导航SR。

| 内容 | 入口 |
|---|---|
| 结论/方法公式/逐题分组/资源 | [验证报告](局部语义匹配验证报告.md) |
| Full/Mean/Wrong及原全局恢复损伤 | [汇总](局部匹配对照汇总.json) / [逐对排序](逐对排序结果.json) / [逐题与分组](逐题与分组对照.json) |
| 冻结队列和方法 | [协议](冻结协议.json) / [16对任务](冻结执行任务.json) / [分组](来源分组副本.json) |
| 全局输出复现与另方程复核 | [全局核验](全局复现核验.json) / [1200分数复核](核验/另实现复核.json) |
| 来源/资源/精度 | [首forward绑定](首forward前绑定.json) / [输入和源码](输入与源码绑定.json) / [资源与缓存SHA](运行资源.json) / [测试回执](核验/源码与测试回执.json) |
| 下一步仅方案 | [目标定义与视觉尺度审查](下一项目标格与视觉尺度审查方案.md) |
| 保存/恢复 | [阶段封存](阶段封存.json)；只读审计命令见下 |

Full/Mean/Wrong Top-1=0/0/0、Top-5=4/3/3（各16）；原全局Full Top-5也是4/16，新局部恢复4、损伤4、净0。全部四项效果门槛未过，保留原失败，不调公式、换样本或升级默认。

复用同一CLIP/Sat2Cap、原预处理/16对；两个模型各87849216冻结参数，52次forward/416图，峰值CUDA分配395.2MiB。0训练/导航/API/下载，20保留组未动。只提供单图匹配内核，尚无地面目标实时导航或GUI；航拍M0与Word保持。

恢复：仓库根执行 `D:/PYTHON/python.exe project/src/eval/audit_crossview_local_v1.py`，不加`--save`；不加载模型。prepare、runner、报告生成器已消费，不能覆盖重跑。四份npy按既定FP16/FP32精度保留以支持无模型重算，不复制原图/权重，不生成搬迁包。原全局候选与旧协议原址保留。
''')
    print(json.dumps({'reports_written':True,'pairs':16,'candidate_passed':False,'new_model_calls':0,'output_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())},ensure_ascii=False))
if __name__=='__main__':main()
