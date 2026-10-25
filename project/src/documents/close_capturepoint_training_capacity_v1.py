"""CV-017 traceable metadata report and one nonexecutable coverage proposal."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'训练正配对容量与划分追溯_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    audit=read(OUT/'核验/另实现复核.json');data=read(OUT/'训练容量与取样追溯汇总.json')
    if not audit['passed']:raise ValueError('Alternate audit required')
    old=read(BASE/'同任务对齐训练与目标对照协议_v1/同任务对齐训练冻结协议.json')
    q=read(BASE/'同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json')
    ordered=[r['source_id_evaluator_only'] for r in q['training_queries']]
    extras=sorted(r['additional_source'] for r in data['additional_fit_candidates'])
    proposed=ordered+extras
    usage=[sum((9*t+j)%12==i for t in range(500) for j in range(9)) for i in range(12)]
    assert len(proposed)==12 and usage==[375]*12
    inherited={k:v for k,v in old['required_development_gates'].items() if k!='all1250_rankings_and_all_audits'}
    proposal={'task':'CV-017','status':'conditional_metadata_proposal_not_executable','hypothesis':'additional distinct positive-pair coverage within the same9fit source groups may improve target-specific development association',
        'only_new_factor':'9->12 distinct MM-GAG ground/center-proxy training pairs; no source-role, head or backbone change',
        'baseline':'frozen CV-014 final9-pair head and saved1250 records; no baseline retraining or re-encoding',
        'ordered12_training_sources':proposed,'additional_sources':extras,'source_groups':9,'head':old['head'],
        'training':{k:old['training'][k] for k in ['seed','updates','temperature','optimizer','learning_rate','weight_decay','gradient_clip_norm','schedule']},
        'sampler':{'batch_size':9,'total_query_uses':4500,'rule':'zero-based step t=0..499, select ordered12[(9*t+j)%12] for j=0..8',
            'new_pair_uses_per_query':usage,'baseline9_pair_uses_per_query':[500]*9,
            'baseline_equivalence':'same formula with N=9 yields original fixed0..8 batch every update',
            'actual_training_executed_here':False},
        'evaluation':{'same_known_dev_sources':5,'same_windows':125,'existing_records_reused':1250,'new_candidate_rankings':375,'planned_total_records':1625,
            'Mean':'original9 fit target mean shared by all arms; fixed to keep controls comparable',
            'Wrong':'same frozen5-development-group cyclic mapping','Disabled':'reuse original shared zero-score records',
            'required_effect_thresholds_from_CV012':inherited,
            'additional_gate':'candidate Full minus frozen MM9 Full >=5/125, and positive Full differences in >=3/5 source groups',
            'complete_and_audit_gate':'all1625 planned records and all engineering/alternate audits pass',
            'mandatory_reports':['Top1/Top5/rank','target controls','per-source','recovery/harm','paired5-source intervals'],
            'navigation_SR':None,'new_source_confirmation':False},
        'resource_upper_bounds_proposed':{'new_raw_engineering_images':249,'new_encoder_inputs_before_optional_SHA_cache_reuse':246,
            'new_encoder_calls_batch8_upper_bound':32,'CUDA_allocated_bytes':2*1024**3,'head_CPU_seconds':1800,
            'new_output_bytes':8*1024**2,'feature_bytes':2*1024**2,'head_initial_final_one_recovery_bytes':1024**2},
        'prerequisites':['register and byte/quality/axis-check only3 new fit representatives; no repeat14-source visual audit',
            'freeze executor, actual input/features, sampler and all gates before candidate training',
            'verify all reused old tensors/checkpoints/source bindings and saved task identity'],
        'not_authorized_by_this_inventory':['new raw-image consumption','model forward or training','source-role reassignment','development/confirmation reuse as training'],
        'stopping':'one fixed final checkpoint/seed0, no rank/epoch/threshold/source sweep; required gate failure closes this factor, no confirmation consumption or default upgrade'}
    write('九对与十二对同预算开发候选草案_未执行.json',proposal)
    table=[]
    for role,c in data['by_role'].items():table.append(f"| {role} | {c['source_records']} | {c['source_groups']} | {c['distinct_ground_bytes']} | {c['distinct_positive_bytes']} | {c['distinct_byte_pairs']} |")
    extra=[]
    for e in data['additional_fit_candidates']:
        extra.append(f"| {e['additional_source']} | {e['frozen_representative']} | {e['camera_distance_m']:.1f} | {e['shared_aerial_byte_hashes81']}/{e['shared_aerial_byte_hashes121']} | {e['additional_positive_in_representative81']} / {e['representative_positive_in_additional81']} |")
    write('九对限制来源与训练配对容量报告.md',f'''# 9对限制来源与现有训练配对容量

CV-017 / 2026-10-18。**9对主要来自本项目的分组、代表选取和同预算约束，并非MM-GAG数据集只有9对。** 保持现有用途不变，训练准备区元数据可提供12张不同地面照片和12个不同中心代理格，即12个不同正配对；当前只训练了其中9个代表。新增3个候选没有增加独立地理组，仍是9个临时来源组，且尚未完成本轮字节/质量/轴向准入。

## 从本地库存到9个实际训练样本

本地原库存有{data['catalogue_entries']}条来源登记，其中{data['eligible_inventory']['source_records']}套静态配对资料齐备，分30个临时来源组。本批按保存的地面和原sat_5_5字节SHA计数，不读取原图；65配对里不同地面照片{data['eligible_inventory']['distinct_ground_bytes']}、不同中心代理图块{data['eligible_inventory']['distinct_positive_bytes']}，因此图块数、来源数、照片数、配对数和独立地理组数必须分开报告。

| 当前用途 | 来源记录 | 临时来源组 | 不同地面字节 | 不同中心格字节 | 不同正配对字节组合 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

训练准备12来源/9组 → 每组选1代表 → 9个训练查询/9正格。开发与确认既有用途不改，不能为了扩训练把其它53来源直接合并进来。23个旧开发来源和6个新开发来源共29个开发素材也不能回流训练；24个确认素材/6组继续保留。

## 限制来自哪些规则

CV-010在原20个保留组内按地面SHA排序，首9组fit、后5组开发、末6组确认，再按哈希每组挑1代表；这是一份人为冻结且经用户批准的实验划分，不是数据集容量定律。CV-012构造器进一步硬性要求9fit+5dev，并把新头的比较因素限定为“9个MM-GAG拍摄点代理配对替代9个Swiss共同场景配对”，以保持当轮训练规模可比。

可定位原件：[分组和代表规则](../../../../project/src/data/capturepoint_partition_v1.py:59)、[9fit/5dev队列限制](../../../../project/src/data/capturepoint_alignment_protocol_v1.py:9)、[逐行源码绑定](规则与源码追溯.json)。旧选择与失败判定不回写；9对上限属于那轮设计，不能推广为整个数据集不可训练。

## 当前训练准备区还能增加什么

| 额外候选 | 同组已训练代表 | 相机坐标距离m | 重叠航拍字节81库/121库 | 额外正格在代表81库 / 代表正格在额外81库 |
|---|---|---:|---:|---|
{chr(10).join(extra)}

031、050、065三者的地面照片和中心代理格字节均不同于现有9代表，因此元数据上可把正配对9→12（增加3对，约33.3%），不是靠25平移窗、裁剪或重复epoch扩大“样本量”。这不等于增加33.3%的独立地理区域，也不意味着SR必然提高。

031/063还具备更具体的空间关联证据：两9×9库共有35个完全相同的航拍格，完整11×11共有63格；031中心格在063库为(3,1)，063中心格在031库为(7,9)。这是旧图块字节元数据可复核的重叠关系，支持该训练组内存在不同照片/不同代理正格的局部关系；不等于完整地理footprint或实测相机像素已认证。另两组中心相距约1.69km/1.22km，候选库无相同字节，不能把“同临时组”直接写成“同一局部窗口”。

## 澄清此前“证据不足”的边界

CV-016只审查每组一个代表，所以没有覆盖上述3个非代表；其“14个受审来源未补出同区域多查询关系”不能外推成全部训练准备区没有额外关系。本批补充范围后发现至少一组存在字节可核查的局部重叠。旧14面板判断仍保留，不据此重标、挑选或重算旧成绩。

缺少独立GPS→像素测绘，会限制“真实精确拍摄点已核验”的声明；按公开数据/代码约定的中心代理格，仍可定义受限训练与评测任务。独立测绘认证和同区域多查询不是训练任何跨视角模型的普遍必要条件，本项目不以这些缺项直接宣布数据不可用。当前确实失败的是冻结9配对/现有编码器/8192参数头这一方案，原因没有被隔离为单一变量。

本地素材规模仍很小：保持角色最多12个训练候选、9组，无法据此宣称解决了泛化问题。预训练覆盖、JPEG同内容异编码重复以及完整地理隔离仍未知。跨用途地面/中心字节交集已单列，字节不同也不证明地理独立。

## 唯一有条件候选：9→12正配对覆盖

见[未执行草案](九对与十二对同预算开发候选草案_未执行.json)。先只对新增3个训练准备素材完成字节/质量/轴向工程，保留原9fit/5dev及确认角色；不重复旧14图审。通过后另冻结一次9对与12对开发对照，骨干及8192参数头不变。

不能简单把“每步全12查询、500更新”叫同预算，那会从4500增加到6000次查询使用。草案固定500更新、batch9、4500次使用；12查询按(9t+j)mod12循环，每查询375次，原9查询公式退化为原固定全9批、每查询500次。只改不同正配对覆盖，不调秩、损失权重或骨干。

评测仍是同5个已知开发来源/125窗，原1250排序只读复用，新增头375排序共1625记录。Mean保留原9训练目标均值供所有臂共享，Wrong/Disabled沿用，源组区间/恢复损伤必报。沿用CV-012必要效果阈值，再要求相对原MM9 Full至少+5/125且≥3/5组正收益；不将新开发结果当新来源确认或导航SR。当前只有元数据草案，未冻结实际执行器/特征，没有授权或运行新模型/训练。

## 验证与保留

65来源/30用途组、3新增候选和全部跨用途字节交集通过另一实现分组计数；距离使用三维单位球atan2与主实现haversine交叉核对，最大差{audit['max_distance_error_m']:.3g}m。原选择源码行和SHA已绑定，4边界测试通过。同Agent另一实现，非独立人员审查。

本批0原图/特征/权重读取、模型forward、新预测、训练、导航/API/下载/确认消费、角色修改。保存历史元数据SHA不代表本轮重新核过原文件。报告/JSON≤2MiB，无大数据副本；当前M0/V4.5与Word保持。恢复只读[绑定](输入与源码绑定.json)/[复核](核验/另实现复核.json)/[封存](阶段封存.json)，build/audit/close已消费，不覆盖重跑。
''')
    write('README.md','''# 训练正配对容量与划分追溯（CV-017）

9对来自既有9训练组各取1代表的实验限制，不是数据集总量。当前训练准备12来源/9组有12个不同地面/中心格配对，额外031、050、065共3候选；不挪用开发/确认。

- [完整容量与规则追溯报告](九对限制来源与训练配对容量报告.md)
- [65来源角色/字节配对与3额外候选](训练容量与取样追溯汇总.json) / [规则源码定位](规则与源码追溯.json)
- [9对与12对同预算草案，未执行](九对与十二对同预算开发候选草案_未执行.json)
- [输入/源码绑定](输入与源码绑定.json) / [测试](核验/边界测试记录.json) / [另实现](核验/另实现复核.json) / [封存](阶段封存.json)

031/063具有35个9×9候选格字节交集及互含正格证据；其余两组不能当同一局部窗口。三候选未经新工程准入，12对仍是小样本；不等于新的地理组或模型收益。按数据集中心代理约定可定义受限训练，独立地理测绘不足不单独构成不可训练判定。

0原图读取/图审/特征或权重访问/模型/新预测/训练/导航/API/下载/确认消费；旧划分、9配对失败、M0/生产GUI/Word保持。后续先冻结新增3素材工程范围，再考虑新训练协议；本页不授权自动训练。所有生成器已消费，只读核SHA/JSON恢复。
''')
    binding=read(OUT/'输入与源码绑定.json')
    for e in binding['inputs']+binding['sources']+binding['readonly_rule_sources']+binding['protected_current_files']:
        assert hashlib.sha256((ROOT/e['path']).read_bytes()).hexdigest()==e['sha256']
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='README.md':
            raw=p.read_bytes();files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if total>2*1024**2:raise ValueError('Metadata output cap exceeded')
    write('阶段封存.json',{'task':'CV-017','files':files,'sources':binding['sources'],'readonly_rule_sources':binding['readonly_rule_sources'],
        'sealed_bytes':sum(e['bytes'] for e in files),'mutable_entries_excluded':['README.md'],'new_training_candidate_executable':False,
        'new_model_forward_calls':0,'new_training_steps':0,'new_navigation_SR':None,'confirmation_access':0})
    print(json.dumps({'sealed_files':len(files),'bytes_before_seal':total,'current_fit_pair_capacity':12,'training_executed':False},ensure_ascii=False))


if __name__=='__main__':main()
