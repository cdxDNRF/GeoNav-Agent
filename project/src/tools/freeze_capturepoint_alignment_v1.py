"""CV-012 freeze only; refuses raw reads, model frameworks and experimental execution."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.capturepoint_alignment_protocol_v1 import build_queues,ARMS,CONDITIONS
BASE=ROOT/'DATA/processed_data/MM-GAG';PRE=BASE/'同任务首批图像轴向预检_v1'
OLD=BASE/'弱场景对齐头工程与训练_v3';DESIGN=BASE/'配对派生清单与弱目标准入_v1'
OUT=BASE/'同任务对齐训练与目标对照协议_v1'
SOURCES=['data/capturepoint_alignment_protocol_v1.py','tools/freeze_capturepoint_alignment_v1.py',
         'eval/audit_capturepoint_protocol_v1.py','tests/test_capturepoint_protocol_v1.py','documents/close_capturepoint_protocol_v1.py']
STAGES=['数据核验与跨视角准入_v1','跨视角感知最小验证_v1','局部语义匹配单因素验证_v1',
        '目标格定义与视觉尺度审查_v1','拍摄点语义与完整目标图验证_v1','对齐头训练素材准入_v1',
        '配对派生清单与弱目标准入_v1','弱场景对齐头工程与训练_v3','弱场景与拍摄点迁移差异诊断_v1',
        '同任务来源与监督资料规划_v1','同任务首批图像轴向预检_v1']
INPUTS=[PRE/n for n in ['首像素冻结协议.json','首像素输入源码绑定.json','工程汇总.json','用途激活快照.json','用户授权记录.json','核验/另实现复核.json']]
INPUTS += [BASE/'同任务来源用途账本_v1.json',DESIGN/'训练设计冻结.json',
           BASE/'同任务来源与监督资料规划_v1/待审九五六分组.json',
           BASE/'弱场景与拍摄点迁移差异诊断_v1/来源目标特异性补充.json',
           OLD/'权重/initial.npz',OLD/'权重/final.npz',OLD/'训练日志.json']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    if Path(p).resolve().is_relative_to((ROOT/'DATA/raw_data').resolve()):raise ValueError('No new raw reads during protocol freezing')
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def entry(p):
    p=Path(p);return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def put(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))
def verify(items):
    for e in items:assert sha(ROOT/e['path'])==e['sha256'],e['path']


def freeze():
    assert not OUT.exists(),'Consumed protocol batch, never overwrite'
    seals=[entry(BASE/s/'阶段封存.json') for s in STAGES]
    for e in seals:
        s=read(ROOT/e['path']);verify(s['files']+s['sources'])
    protected=read(PRE/'首像素输入源码绑定.json')['protected'];verify(protected)
    engineering=read(PRE/'工程汇总.json');assert engineering['all_required_passed'] and len(engineering['sources'])==14
    pre=read(PRE/'首像素冻结协议.json');queue=build_queues(pre['sources'])
    ledger=read(BASE/'同任务来源用途账本_v1.json');roles={x['source_id']:x for x in ledger['records']}
    for s in pre['sources']:
        assert roles[s['source_id']]['role']==('fit_preparation' if s['role']=='fit' else 'development')
        assert roles[s['source_id']]['representative']
    assert len(queue['ground_encoder_inputs'])==14 and len(queue['aerial_encoder_inputs'])==1134 and queue['planned_rankings']==1250
    design=read(DESIGN/'训练设计冻结.json');training=dict(design['training'])
    training.update({'source_groups':[s['group'] for s in pre['sources'] if s['role']=='fit'],
        'source_ids_evaluator_only':[s['source_id'] for s in pre['sources'] if s['role']=='fit'],
        'loss':'mean over9 distinct MM-GAG fit queries: -log_softmax(cos(h(g),a81)/0.1)[40]',
        'source_semantics':'CSV/code capture-point proxy; no Swiss common-cell labels in new training',
        'initialization':'fresh seed0 U=zeros, V=Normal(0,.02), never continue Swiss optimizer or weights',
        'evaluation_timing':'final500 only, no development scores or intermediate head selection'})
    OUT.mkdir();(OUT/'核验').mkdir()
    put('固定编码与训练开发队列.json',queue)
    baselines={'Identity':{'weight':entry(OLD/'权重/initial.npz'),'role':'U0 identity, no target adaptation'},
        'SwissRank8Frozen':{'weight':entry(OLD/'权重/final.npz'),'role':'frozen CV-008 final500,9Swiss queries, same head/optimizer/update budget; never retrain'},
        'MMRank8Final':{'weight':None,'role':'one future same-task seed0 final500 checkpoint','binding_required_after_training':True}}
    protocol={'task':'CV-012','date':'2026-10-18','protocol_id':'capturepoint-rank8-same-task-v1',
        'research_design_frozen':True,'execution_admitted':False,'training_authorized':False,'training_started':False,
        'actual_executor_binding':None,'actual_feature_binding':None,
        'hypothesis':'same-task capture-point-proxy training data may improve source-disjoint development association versus identity and frozen Swiss training',
        'only_new_factor':'training data regime: nine MM-GAG camera-proxy pairs instead of nine Swiss common-scene pairs; this also changes domain/content, not a pure causal label-only test',
        'label_semantics':'CSV/code camera-center proxy, not surveyed geographic point or visible-landmark label',
        'head':design['head'],'training':training,'encoder_assets':design['frozen_encoder_assets'],
        'preprocessing':'exact original CV-002 PIL CLIP center crop and Sat2Cap224 processing, both frozen; no full-frame/padding/local-token changes',
        'baselines':baselines,'queue':entry(OUT/'固定编码与训练开发队列.json'),
        'controls':{'Full':'genuine normalized ground feature passed through each arm h',
            'Mean':'one fixed mean of9 normalized MM fit ground vectors, normalize then each arm h; never mean over dev/confirmation/old16/Swiss queries',
            'Wrong':'next of5 frozen development source groups cyclically, fixed once per source and reused for25 windows',
            'Disabled':'one shared evaluator-only constant-zero25-score control per pair with CV-002/tie/v1 ordering; head is not called, zero features not passed into normalized head'},
        'evaluation':{'arms':ARMS,'conditions_per_arm':CONDITIONS,'shared_disabled_condition':True,
            'unique_development_sources':5,'windows_per_source':25,'planned_windows':125,'planned_rankings':1250,
            'independent_source_unit':'5 provisional grouped sources;25 shifted windows share same target/center and do not add geography',
            'scoring':'saved/reloaded float32 global features, float64 normalized cosine and head evaluation; CV-002/tie/v1:pair_id:cell_index',
            'metrics':['Top1','Top5','true rank','Full minus Mean/Wrong/Disabled','recovered/harmed for Top1 and Top5','per-source and paired-source uncertainty'],
            'paired_source_interval':'all5^5=3125 source bootstrap resamples of unweighted paired source rates, percentile95; mandatory report, not a pass gate',
            'navigation_SR':None,'old16_legacy_development':'not rerun in this batch; past failed results retained, not new confirmation'},
        'required_development_gates':{'all1250_rankings_and_all_audits':True,'Full_top1_minimum':32,
            'Full_minus_Mean_minimum':16,'Full_minus_Wrong_minimum':16,'Full_minus_Disabled_minimum':16,
            'Full_minus_Identity_minimum':5,'Full_minus_SwissFrozen_minimum':5,
            'source_groups_Full_above_both_controls_minimum':3,'source_groups_Full_above_both_baselines_minimum':3},
        'threshold_rationale':{'absolute':'>=25%,ceil(125/4)=32; bounded association-development floor, follows original4/16 level, not SR or publication acceptance',
            'target_controls':'>=12.5pp,ceil(125/8)=16; original2/16 contrast scale; plus3/5 source benefit to address prior2/5 weakness',
            'training_data_effect':'>=4pp,5/125 over EACH identity and SwissFrozen, plus3/5 sources positive versus BOTH; seed0 only',
            'report_only':'Top5/ranks/recovered-harmed/intervals all mandatory; no hidden post-score criteria or generalization claim'},
        'resources':dict(design['resources'],head_training_device='cpu',one_future_run_only=True),
        'engineering_requirements':['strict encoder loading/frozen tensors/no gradients; hash and save engineering evidence immediately per encoder and first batch',
            'identity<=1e-6,head-only finite gradients, serialized synthetic resume exactly equal, real head resets fresh seed0 after engineering',
            'engineering subsets are part of1148inputs/144encoder calls; saved successful features reused, no blind retry',
            'append-only forward/update events and per-batch feature coverage; no fragile overwriting of shared status file',
            'runtime/extractor/trainer/scorer/auditor sources bound before first forward; actual feature hashes and fit-only tensors bound before first update',
            'do not call old consumed CV-008 runners/generators; immutable math/helpers may be imported only with hashes'],
        'information_permissions':{'head_input':['normalized_ground_feature_512'],
            'evaluator_or_offline_loss_only':['source/group IDs','GPS','goal/positive index','candidate feature library','crop origin'],
            'online_navigation':'not implemented/authorized by this association protocol; unseen images/maps never given to a policy'},
        'confirmation':{'groups':6,'sources':24,'pixel_or_model_access_this_protocol':False,
            'later_confirmation_requires_new_protocol':True,'known_gap':'mmgag_011 representative lacks full stitch reference; do not rescue by changing source after development scores'},
        'retention':'future caches<=32MiB,initial/final heads+oneactive recovery<=1MiB,whole execution<=40MiB;no raw/encoder copies',
        'stopping':'one seed/500updates/final only; any necessary gate fails closes this data-regime factor, no rank/loss/temperature/source/epoch sweep; passing yields development candidate only, no default/ground GUI or navigation upgrade',
        'current_task_activity':{'model_calls':0,'training_steps':0,'new_predictions':0,'raw_image_reads':0,'navigation_episodes':0,'API_calls':0,'downloads':0,'confirmation_pixels':0}}
    put('同任务对齐训练冻结协议.json',protocol)
    reusable=[ROOT/'project/src'/p for p in ['agents/weak_alignment_head_v1.py','train/weak_scene_alignment_v1.py','data/mmgag_views_v1.py','agents/crossview_local_match_v1.py']]
    put('输入与协议源码绑定.json',{'task':'CV-012','date':'2026-10-18','inputs':[entry(p) for p in INPUTS],
        'sources':[entry(ROOT/'project/src'/p) for p in SOURCES],'reference_math_sources_only':[entry(p) for p in reusable],
        'previous_seals':seals,'protected':protected,'protocol':entry(OUT/'同任务对齐训练冻结协议.json'),
        'index_before':[hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest() for x in read(ROOT/'项目导航/实验索引.json')['batches']]})
    print(json.dumps({'frozen':True,'fit_queries':9,'dev_sources':5,'dev_windows':125,'rankings':1250,'execution_admitted':False,'model_calls':0}),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser().parse_args();freeze()
