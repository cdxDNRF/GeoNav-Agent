"""CV-005 report from saved features/results; no training or model calls."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'DATA/processed_data/MM-GAG/拍摄点语义与完整目标图验证_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(n,s):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def main():
    result=read(OUT/'完整目标图对照汇总.json');audit=read(OUT/'核验/另实现复核.json');resource=read(OUT/'运行资源.json');assert audit['passed']
    pairs=read(OUT/'冻结16对副本.json')['pairs'];rows=read(OUT/'逐对排序结果.json')['rows'];lookup={(x['arm'],x['pair_id'],x['condition']):x for x in rows}
    full=result['arms']['FullFrame'];crop=result['arms']['CropReplay'];passed=result['candidate_passed']
    verdict='开发观察门槛通过，仅为待确认候选' if passed else '完整画面候选未过，按停止条件收束'
    table='\n'.join(f"| {arm} / {ctl} | {s['top1_counts'][ctl]}/16 | {s['top5_counts'][ctl]}/16 | {s['median_true_rank'][ctl]:g} |" for arm,s in result['arms'].items() for ctl in ['Full','Mean','Wrong'])
    per='\n'.join(f"| {p['pair_id']} | {p['group_id']} | {lookup['CropReplay',p['pair_id'],'Full']['true_rank']} | {lookup['FullFrame',p['pair_id'],'Full']['true_rank']} | {lookup['FullFrame',p['pair_id'],'Mean']['true_rank']} | {lookup['FullFrame',p['pair_id'],'Wrong']['true_rank']} |" for p in pairs)
    grp='\n'.join(f"| {g} | {v['planned']} | {v['Full']} | {v['Mean']} | {v['Wrong']} |" for g,v in full['per_group'].items())
    gate='\n'.join(f"| {k} | {'通过' if v else '未过'} |" for k,v in full['gates'].items())
    next_text=('下一步另立局部闭环开发协议，验证当前/已访问观测的捕获位置线索，不能把25格候选库给在线Agent；20保留组仍不自动消费。' if passed else
      '全局中心裁剪、局部token、完整目标画面三个固定零训练候选均未达到开发门槛。停止继续微调相似度/填边/裁剪/层数；本轮不自动开展第四次零训练探针。下一阶段如研究小型关系头，应在冻结骨干和总参数不扩大的前提下单独论证训练/开发分组与拍摄点标签、资源和停止条件，并取得训练授权；现有16开发对不能改称训练或独立确认。也可以先建设平台的目标语义说明和诊断入口，保留跨视角尚未通过的边界。')
    put('拍摄点语义与完整目标图验证报告.md',f'''# 拍摄点目标语义与完整目标照片验证

CV-005 / 2026-10-17。**{verdict}。** 本轮固定了沿用原任务的目标语义，完成一个轻量单因素开发对照；不代表真实跨视角导航已通过。

## 先固定要找什么

本轮目标明确为“照片拍摄位置所对应的格”，沿用既有CSV/原代码中心(5,5)移窗得到的代理标签。照片中的建筑、桥、雕塑等是帮助推断拍摄位置的视觉线索，不自动把物体占地格当成成功答案。这是继承原队列的工程语义约定，不是新增独立测绘、用户重新标注或证明CSV点到格的精确变换。

若任务改成“找到图中指定地标本体”，需要物体指认、地面图区域/对应航拍标注和新协议，原16题答案保留。本轮不能按模型相似度、肉眼选择或可见地标位置重写旧标签。16对地理footprint未知的限制继续披露。

新公开目标接口仅包含`target_rgb`、`modality=ground`、`localization_target=capture_point`；拒绝GPS、目标索引、源名、全图库和未标注的landmark语义。它是一项可复用的接口边界，尚未改变生产平台/导航输入。契约见[目标语义约定](目标语义约定.json)。

## 一个因素：保留完整目标画面

CV-004测得原中心裁剪只保留约56.28%～75.17%的原图面积，且观察到近景、远景、室内和邻格地标等上下文风险。本轮提出假设：保留照片全部画面，可能改善拍摄位置关联；这是待验证假设，不能把像素保留等同于信息有效。

固定处理：长边等比缩到224，短边向下取整（至少1像素），居中放入224×224画布；使用PIL bicubic。内容区按原CLIP均值/标准差归一化，填边区固定为归一化后的0，等价于原RGB通道均值。奇数余量放右侧/下侧。不改变最终输入分辨率、CLIP/Sat2Cap骨干、全局余弦、目标格、队列、Mean/Wrong映射或破平局规则。

保留原全局中心裁剪为同题失败基线，未将旧候选重新判通过。唯一新因素是目标图处理方式，不叠加局部token、边缘线索或多个池化。完整画面会缩小主体，并引入预训练之外的填边分布，因此也可能变差；没有扫描填充颜色或缩放策略。

## 同队列结果

16对来自原10临时开发组；20保留组不参与。25格图库仅供离线评测器逐一打分，不提供给在线Agent。Mean在每臂16归一化目标特征上取均值再归一化；Wrong沿用原不同组目标。以下为Top-k，不是SR。

| 处理方式 / 目标 | Top-1 | Top-5 | 真实格中位排名 |
|---|---:|---:|---:|
{table}

完整画面Full相对原裁剪Full：Top-1恢复/损伤/净值为{result['fullframe_vs_crop']['top1']['recovered']}/{result['fullframe_vs_crop']['top1']['harmed']}/{result['fullframe_vs_crop']['top1']['net']}题，Top-5为{result['fullframe_vs_crop']['top5']['recovered']}/{result['fullframe_vs_crop']['top5']['harmed']}/{result['fullframe_vs_crop']['top5']['net']}题。完整画面相对Mean/Wrong的Top-1恢复损伤分别为{full['target_contrasts']['Mean']}、{full['target_contrasts']['Wrong']}。不能只看单个排名变化判断方案有效。

| 必要门槛 | 结果 |
|---|---|
{gate}

沿用Full Top1≥4/16、比Mean/Wrong各多≥2题、至少3组同时优于两控制的观察性门槛。实际同时改善组数为{full['positive_groups_against_both']}。任何必要项未过均不升级默认或接入稳定导航；不能换样本、改变语义/阈值来补过线。

## 逐对和分组

| 对 | 组 | 原裁剪Full排名 | 完整Full | 完整Mean | 完整Wrong |
|---|---|---:|---:|---:|---:|
{per}

| 临时组 | 对数 | 完整Full Top1 | 完整Mean Top1 | 完整Wrong Top1 |
|---|---:|---:|---:|---:|
{grp}

临时组防止已知重复/邻近来源拆开，不证明完全地理隔离。样本小且为开发对照，不把三条件/两臂当额外地理样本，不虚报独立区域显著性。

## 工程与资源复核

7项边界测试涵盖拍摄点语义、隐藏元数据拒绝、不可变像素、原图四边保留、填边精确为0、方图与旧处理完全一致、长宽比/整数余量和冻结224输入。真实32个地面模型输入＝16原裁剪复验＋16完整画面，共4次编码器forward；原航拍400特征直接复用，无新增航拍编码。原裁剪全局复验最大差{audit['crop_global_replay_max_delta']:.3g}，在≤1e-5门槛内后才执行完整画面。

只加载既有CLIP视觉子模块87849216冻结参数，0新增训练参数；CUDA峰值分配{resource['peak_cuda_allocated_bytes']}字节（{resource['peak_cuda_allocated_bytes']/1024**2:.1f}MiB）≤2GiB，资源计时{resource['elapsed_seconds']:.2f}秒不含导入/前置绑定，不作速度基准。两份FP32地面特征共{sum(x['bytes'] for x in resource['features'])}字节；总批次≤4MiB，不复制原图/航拍缓存/权重。

另实现逐通道重建32个像素张量SHA并从保存的FP32特征以fsum重新计算2400分数，最大差{audit['maximum_score_delta']:.3g}；排名、分组、门槛和恢复损伤一致。同Agent另实现复核，不称独立人员审查。0训练、导航、API、下载或保留组编码，无新SR；原M0、模型/默认、Word/PDF与历史验收保持。

## 下一步与平台边界

{next_text}

平台侧提供可复用的目标语义契约和完整目标照片处理函数，以及本报告；当前GUI仍只有已验证的航拍导航能力，未开放新的地面目标实时能力。语义固定解决“做什么任务”，不等于已经解决“如何正确定位”。

恢复仅运行`D:/PYTHON/python.exe project/src/eval/audit_crossview_fullframe_v1.py`（不加--save），不加载模型。prepare、runner和报告生成器已消费，禁止盲重跑/覆盖；新的研究另认领和冻结。
''')
    put('下一阶段决策.md','# 下一阶段决策\n\nCV-005 / 2026-10-17。'+verdict+'。\n\n'+next_text+'\n\n任务语义继续沿用拍摄点代理格。地理footprint未知不是已证实的错标，也不要求联系作者或无限等待私有资产；实验报告必须披露该限制。新的物体定位定义须另立标签/数据版本，不改现有16题。\n')
    put('README.md',f'''# 拍摄点语义与完整目标图验证

CV-005 / 2026-10-17。**{verdict}。** 目标沿用原CSV/代码代理拍摄位置格；可见地标作线索，不自动变成正确格。

| 内容 | 入口 |
|---|---|
| 语义/公式/结果/逐题分组 | [完整报告](拍摄点语义与完整目标图验证报告.md) |
| 原中心裁剪与完整画面结果 | [汇总](完整目标图对照汇总.json) / [排序原件](逐对排序结果.json) |
| 任务语义和冻结方法 | [语义契约](目标语义约定.json) / [协议](冻结协议.json) / [16对](冻结16对副本.json) |
| 工程/输入/资源 | [零forward工程](零forward工程核验.json) / [输入源码SHA](输入与源码绑定.json) / [首forward绑定](首forward前绑定.json) / [原裁剪复验](原裁剪复验.json) / [资源](运行资源.json) |
| 复核与保存 | [另实现核验](核验/另实现复核.json) / [阶段封存](阶段封存.json) |
| 下一阶段 | [决策及停止边界](下一阶段决策.md) |

完整画面Full/Mean/Wrong Top1={full['top1_counts']['Full']}/{full['top1_counts']['Mean']}/{full['top1_counts']['Wrong']}、Top5={full['top5_counts']['Full']}/{full['top5_counts']['Mean']}/{full['top5_counts']['Wrong']}（各16）。32地面输入4次forward；0新航拍编码/训练/参数/API/下载/导航/保留消费。Top-k不是导航SR，原默认/GUI/Word与历史判定不变。

恢复只运行 `D:/PYTHON/python.exe project/src/eval/audit_crossview_fullframe_v1.py`（不加--save）。模型执行和准备已消费，不覆盖重跑；输出≤4MiB，旧原件原址保留。
''')
    print(json.dumps({'reports_written':True,'candidate_passed':passed,'output_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())},ensure_ascii=False))
if __name__=='__main__':main()
