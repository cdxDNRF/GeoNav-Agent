"""CV-007 freeze a research design, then report; no executable training binding."""
import argparse,html,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'配对派生清单与弱目标准入_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))
def freeze_design():
    qa=read(OUT/'视觉语义审查.json');split=read(OUT/'来源分组与用途.json');bank=read(OUT/'唯一训练查询与图块引用.json');assert len(qa['scenes'])==15
    assert all(s['decision']=='weak_engineering_only' and s['panel_viewed'] for s in qa['scenes'] if s['scene'] in split['fit']+split['development'])
    assert next(x for x in qa['scenes'] if x['scene']=='GrandeDixence')['decision']=='quarantined'
    bindings=[OUT/n for n in ['派生素材清单.json','场景几何与异常隔离.json','来源分组与用途.json','唯一训练查询与图块引用.json','视觉语义审查.json']]
    cv2=BASE/'跨视角感知最小验证_v1';cache=[cv2/n for n in ['地面特征.npy','航拍特征.npy','冻结执行任务.json','划分与任务.json']]
    protocol={'task':'CV-007','date':'2026-10-17','protocol_id':'weak-scene-rank8-pilot-v1','research_design_frozen':True,'execution_admitted':False,'training_started':False,'execution_source_binding':None,
      'execution_requirements':['implement/bind actual extractor/head trainer/auditor in a new task','pass synthetic math/grad/resume/input-permission tests and limited real engineering with existing models','freeze all runtime sources/configs and first-forward input hashes','save actual feature hashes after extraction before optimizer steps; train only within authorized run'],
      'dataset_semantics':'weak_scene_common_cell_proxy','capture_point_labels_verified':False,'main_project_goal_unchanged':'capture_point CSV/code proxy; do not replace MM-GAG success cells with visible landmarks',
      'hypothesis':'one low-rank ground-feature residual projection may improve weak same-scene association and transfer to existing capture-point development; neither is established',
      'only_new_factor':'train ground512D residual rank8 projection; all encoders/preprocessing/scoring/data splits fixed, no simultaneous full-frame processor',
      'data_bindings':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in bindings],
      'cached_MMGAG_features':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in cache],
      'frozen_encoder_assets':[e for e in read(OUT/'输入与源码绑定.json')['protected'] if e['path'].startswith('models/')],
      'preprocessing':'exact CV-002 PIL center-crop CLIP target and Sat2Cap aerial_pixels with224 input; extract only single300px crops, never whole1500 windows',
      'head':{'formula':'normalize(g + U @ V @ g)','input':'normalized512D ground feature only','rank':8,'U_shape':[512,8],'V_shape':[8,512],'trainable_parameters':8192,'bias':False,'U_init':'zeros','V_init':'normal mean0 std0.02 using seed0','norm_epsilon':1e-12,'encoders_frozen':True},
      'training':{'seed':0,'source_groups':split['fit'],'unique_ground_queries':9,'positive_cells':9,'aerial_candidates_per_query':81,'updates':500,
        'loss':'mean over9 fit queries of -log_softmax(cosine(projected_ground, each81 same-scene aerial)/0.1)[shared_proxy_positive]',
        'temperature':0.1,'optimizer':'AdamW','learning_rate':0.001,'weight_decay':0.0001,'gradient_clip_norm':1.0,'schedule':'none','batch':'all9 unique source queries per update; each normalized query counted once, not25 times',
        'evaluation_timing':'after final500 only, no best-checkpoint selection','negative_caveat':'neighboring views may share scene content; exact-cell negatives are weak supervision and may contain semantic false negatives; do not tune away after scores'},
      'evaluation':{'arms':['Identity','Rank8Final'],'conditions':['Full','Mean','Wrong'],'weak_dev_groups':split['development'],'weak_dev_window_records':125,'weak_dev_unique_queries':5,'MMGAG_original_development_pairs':16,'planned_rankings':846,
        'score':'normalized512D cosine, all decisions reconstructed from saved features/checkpoint, unchanged CV-002 hash ties',
        'Full':'apply same arm h to genuine normalized ground feature','weak_Mean':'fixed mean of9 fit normalized ground features then normalize and apply h; no dev mean for weak scene control',
        'MMGAG_Mean':'original mean of16 normalized ground features then normalize and apply h, preserving CV-002 control definition at identity',
        'weak_Wrong':'next development group cyclically in frozen development list, no same-source wrong target','MMGAG_Wrong':'exact original different-group mapping',
        'metrics':['Top1','Top5','true rank','Full vs Mean/Wrong recovered/harmed','Rank8 vs Identity recovered/harmed','source-wise effects'],
        'sampling_unit':'five weak-development groups and ten existing MM-GAG development groups reported separately;25 variants do not add geography; geographic independence not proven','navigation_SR':None},
      'required_effect_gates':{'MMGAG_all16_complete':True,'MMGAG_Full_top1_minimum':4,'MMGAG_Full_minus_Mean_minimum':2,'MMGAG_Full_minus_Wrong_minimum':2,'MMGAG_groups_positive_against_both_minimum':3,
        'weak_dev_Full_top1_macro_gain_minimum':0.04,'weak_dev_source_groups_improving_minimum':3,'all_audits_pass':True},
      'resources':{'maximum_encoder_inputs':1148,'maximum_encoder_forward_calls':144,'batch_size':8,'CPU_fallback_if_initial_GPU_free_MiB_below':2560,'maximum_CUDA_allocated_bytes':2*1024**3,
        'feature_cache_limit_bytes':32*1024**2,'head_checkpoint_limit_bytes':1024**2,'whole_batch_outputs_limit_bytes':40*1024**2,'maximum_train_minutes':30,
        'engineering':'initial subset from same1148 bound images, saved successful features reused; engineering calls count toward total, no hidden re-extraction',
        'retention':'no raw/encoder copies; initial and final head, one active recovery, features/configs/logs; no unbounded optimizer checkpoints'},
      'engineering_gates':['initial U0 head reproduces identity to1e-6 on frozen features','encoders no gradients and tensor identities unchanged','finite nonzero head gradient and bounded head-only updates','checkpoint resume exactly reproduces uninterrupted synthetic run','candidate bank/source/label never part of head input'],
      'stopping':'one seed and one fixed500-update budget; any effect gate fails closes factor; no rank/loss/temperature/source sweeps or fourth zero-training probe, no navigation/default upgrade',
      'MMGAG_reserved_groups_consumed':0,'external_model_api_calls':0,'downloads':0,'status':'design_frozen_pending_executor_engineering_and_run_authorization'}
    assert len(bank['queries'])==14 and len(bank['cells'])==1134
    put('训练设计冻结.json',protocol)
    print(json.dumps({'design_frozen':True,'fit_groups':9,'weak_dev_groups':5,'head_parameters':8192,'execution_admitted':False,'model_calls':0}))
def report():
    audit=read(OUT/'核验/另实现复核.json');assert audit['passed'];manifest=read(OUT/'派生素材清单.json')['records'];split=read(OUT/'来源分组与用途.json');qa=read(OUT/'视觉语义审查.json');proto=read(OUT/'训练设计冻结.json');geo=read(OUT/'场景几何与异常隔离.json')['scenes'];panels=read(OUT/'像素适配与面板核验.json')['panels']
    table='\n'.join(f"| {x['scene']} | {x['decision']} | {x['role']} | {x['observation']} |" for x in qa['scenes'])
    cards=[]
    for q in qa['scenes']:
        p=next(x for x in panels if x['scene']==q['scene']);rel=Path(p['path']).name
        cards.append(f'<article id="{q["scene"]}"><h2>{html.escape(q["scene"])} · {html.escape(q["role"])}</h2><a href="审阅/{rel}"><img loading="lazy" src="审阅/{rel}" alt="{html.escape(q["scene"])}弱配对QA"></a><p>{html.escape(q["observation"])}</p><p>地理拍摄点未认证；同景代理标签仅用于辅助表征学习工程。</p></article>')
    put('派生清单与训练设计报告.md',f'''# 派生路径、弱目标格与训练设计冻结

CV-007 / 2026-10-17。**路径与弱场景代理清单已落实，训练研究设计已冻结；实际执行尚未准入，本轮没有训练或模型调用。** 这些标签用于辅助弱场景对齐，不替代主任务的拍摄点目标格。

## 已落实的数据

原375条记录全部保留，解析到原始地面/航拍文件的路径和SHA另存新[派生清单](派生素材清单.json)，没有重命名、复制或改写原图/JSON。14个正常场景共350行被标为`weak_scene_engineering_only`，GrandeDixence整组25行隔离。旧225条严格可用和150条需要名称/扩展名映射的问题在新清单中可追溯，不将路径修复冒充独立地理标注。

14个正常场景中，相同300px图块在25窗中各出现一次，其位置逐一等于文件编号；同一图块相对代理格的偏移也保持一致，形成9×9的81种唯一图块。结合发布论文S1.4的固定目标平移构造，本批批准它们为**发布同景共同格的弱代理标签**。这不证明地面相机落在该格，也不证明物体真实占地，因此不写成新的拍摄点真值。

数据适配器将训练所需图像裁成单个300×300航拍格，并返回去元数据、只读的`target_rgb/current_rgb`。原窗、坐标、源名和答案索引仅留在离线清单/损失构造，不输入对齐头或在线Agent；每次素材读取先核原始SHA。正格标签在离线监督中使用，与在推理输入中泄漏答案不同。

## GrandeDixence处理

按其他14场景支持的编号位置规则，0—19对应一个图块SHA，20—24对应另一个SHA；共有117种图块，25窗交集为空。完整[异常证据](场景几何与异常隔离.json)保留两个簇及偏移冲突，本批不猜测哪个格是正确拍摄点、不用多数结果修补末5条，也不只取前20条凑训练。整组25条保留但不进训练或弱开发。

## 逐场景QA与来源角色

全部15份[实际图像QA页](弱配对审查.html)已逐一查看。检查内容为路径/配对明显一致性及弱场景可用性，不能当独立人员标注或地理真值。划分只用预先固定场景名哈希，未使用模型成绩。

| 场景 | 审查结论 | 用途 | 可见证据与限制 |
|---|---|---|---|
{table}

9个拟训练组为：{', '.join(split['fit'])}。5个弱开发组为：{', '.join(split['development'])}。每个场景25窗整组归属一个角色；训练实际上只有9个不同地面查询，弱开发只有5个，不能说有225/125个独立地面样本。全部14个查询和1134个唯一航拍格只有路径/裁格/SHA引用，没有提前编码或复制图像。

同内容与共享格在fit/dev间交集为0；这不证明地理隔离，跨MM-GAG的地理覆盖仍未知。MM-GAG原16对仍为已知开发集，10开发组/20保留组角色不变；本轮不读新MM-GAG像素，不消费保留组。Swiss数据被用于未来训练后，本实验应称自定义弱监督工程迁移，不称该子集原论文零样本复现。

## 已冻结的研究设计

唯一新因素为秩8地面目标残差头`h(g)=normalize(g+UVg)`，U为512×8、V为8×512，共8192参数；原CLIP和Sat2Cap冻结。U初始0保证身份基线，V用seed0的正态初始化。沿用原中心裁剪/航拍224处理与余弦，不同时加入完整画面、局部token或其他模型。

9个拟训练场景每步各取一个不同地面查询，对应同场景81个航拍格，固定InfoNCE温度0.1，AdamW学习率0.001/weight_decay0.0001、梯度范数裁到1、无调度，固定500更新/seed0、最多30分钟。近邻可能同样具有场景语义，因此负样本是弱监督，风险提前披露；不在看到成绩后更换负样本/温度/源。

只在最终500步权重上做身份基线/训练头×真实/均值/错误目标对照：弱开发5场景×25窗共125条、原MM-GAG16对，共846组排序。MM-GAG沿用原Full Top1≥4/16、比Mean/Wrong各多≥2题、至少3来源组同时改善等门槛；弱开发另外固定Full Top1宏平均相对身份基线增益≥0.04、至少3/5来源正收益，所有审计通过。重复窗口不增加地理样本，不报告导航SR或升级默认。细节以[训练设计冻结](训练设计冻结.json)为准。

后续编码最多14地面＋1134唯一航拍格＝1148图像，batch8最多144次编码器调用，显存≤2GiB，特征缓存≤32MiB、头检查点≤1MiB、总输出≤40MiB。工程子集调用计入总预算并复用已保存特征；不重复下载/复制编码器或堆积优化器。当前只保存了设计，所有新增参数、编码和训练调用均为0。

## 仍须完成的执行准入

**研究设计冻结不等于训练执行器已经可用。** 下一项应实现并绑定真实特征提取/小头训练/审计源码，验证身份初值、梯度、恢复、编码器冻结和输入权限，再完成首forward输入与实际特征SHA绑定。`execution_admitted=false`、`execution_source_binding=null`明确阻止把本文件当成已经批准运行的训练作业。满足这些条件并有对应运行授权后，才在一次固定预算内训练；必要门槛未过就关闭因素，不扫参追过线。

主线目标仍是拍摄点代理格定位。弱场景训练即便改善图像关联，也不能直接宣称跨视角导航成功；导航接入/接受阈值/新来源确认需后续独立协议，20保留组本轮继续不使用。

## 验证与交付

7边界测试及5源码语法通过；另实现验证375路径、350代理标签、25隔离行、1134图块引用、14独立裁格、9/5分组无已知内容交叉、异常20/5簇和全部训练设计绑定。同一Agent复核不称独立人员审查。15份面板实看，原件/模型/默认/Word和旧验收保持；0模型加载/forward、训练、编码、导航、API或下载，无新SR。

平台侧交付只读像素素材适配器及可离线查看的QA页，生产GUI未改。恢复仅运行`D:/PYTHON/python.exe project/src/eval/audit_weak_scene_pairs_v1.py`（不加--save）；freeze/build/design/report已消费，不覆盖重跑。
''')
    put('弱配对审查.html','''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CV-007 弱配对审查</title><style>body{margin:0;background:#eef3f7;color:#19394d;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1200px;margin:32px auto;padding:0 22px}h1{font-size:29px}article,section{background:white;border:1px solid #dbe4eb;border-radius:12px;padding:20px;margin:24px 0}img{display:block;width:100%;height:auto;border-radius:8px}h2{font-size:22px}a{color:#087f8c}.notice{border-left:5px solid #e49c42;background:#fff6e8}nav{display:flex;gap:12px;flex-wrap:wrap}</style><main><h1>弱场景配对 · 路径与代理格QA</h1><section class="notice">350行可供弱场景对齐工程准备，GrandeDixence25行隔离。训练设计已冻结，执行器/实际特征未绑定，本页不表示已训练。代理格不是拍摄点地理真值，原主任务不变。</section><p><a href="派生清单与训练设计报告.md">报告</a> · <a href="训练设计冻结.json">训练设计</a> · <a href="派生素材清单.json">派生清单</a></p><nav>'''+''.join(f'<a href="#{x["scene"]}">{html.escape(x["scene"])}</a>' for x in qa['scenes'])+'</nav>'+''.join(cards)+'</main></html>')
    put('README.md','''# 配对派生清单与弱目标准入

CV-007 / 2026-10-17。派生路径和350个弱代理行已核验，GrandeDixence25行隔离；9fit/5弱开发分组和8192参数小头研究设计已冻结。未训练、未编码，实际执行仍须源码/工程/特征绑定。

| 内容 | 入口 |
|---|---|
| 路径/语义/分组/训练设计/边界 | [完整报告](派生清单与训练设计报告.md) |
| 15场景实际图像QA | [审查页](弱配对审查.html) / [逐场景视觉记录](视觉语义审查.json) |
| 派生数据与原SHA引用 | [375行清单](派生素材清单.json) / [14查询1134格](唯一训练查询与图块引用.json) / [9fit5dev](来源分组与用途.json) |
| 几何与隔离 | [共同格/GrandeDixence20和5簇](场景几何与异常隔离.json) |
| 研究设计，非执行授权 | [冻结设计](训练设计冻结.json) |
| 核验/保存 | [输入源码绑定](输入与源码绑定.json) / [另实现审计](核验/另实现复核.json) / [阶段封存](阶段封存.json) |
| 像素数据接口 | [只读成对图像适配器](../../../../project/src/data/weak_scene_pairs_v1.py) |

代理格语义是发布同景构造的弱场景标签，不是新的拍摄点真值。MM-GAG原16开发/20保留和主任务语义不改；0模型forward、训练、导航、API、下载、新MM-GAG或保留像素。原图、模型、默认、Word和生产GUI保持，总输出≤8MiB。

恢复只运行 `D:/PYTHON/python.exe project/src/eval/audit_weak_scene_pairs_v1.py`，不加--save；旧prepare/build/设计生成/报告生成已消费，禁止覆盖重跑。下一项实现真实执行器和工程验证，完成首forward/特征SHA绑定后才运行训练。
''')
    print(json.dumps({'reports_written':True,'training_design_frozen':True,'actual_execution_admitted':False,'model_calls':0,'output_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['design','report']);a=p.parse_args();freeze_design() if a.mode=='design' else report()
