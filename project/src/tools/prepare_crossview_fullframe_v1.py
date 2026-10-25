"""CV-005 freeze a capture-point convention and one full-frame input factor."""
import hashlib,importlib.metadata,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
import numpy as np
from data.mmgag_views_v1 import clean_rgb,ground_pixels
from data.crossview_goal_semantics_v1 import public_target,fullframe_pixels,fullframe_geometry
BASE=ROOT/'DATA/processed_data/MM-GAG'
PREV=BASE/'跨视角感知最小验证_v1'
OUT=BASE/'拍摄点语义与完整目标图验证_v1'
SOURCES=['project/src/data/crossview_goal_semantics_v1.py','project/src/tools/prepare_crossview_fullframe_v1.py','project/src/eval/crossview_fullframe_probe_v1.py','project/src/eval/audit_crossview_fullframe_v1.py','project/src/tests/test_crossview_goal_semantics_v1.py','project/src/documents/close_crossview_fullframe_v1.py']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))
def main():
    assert not OUT.exists(),'Consumed prepare; never overwrite'
    seals=[]
    for directory in ['数据核验与跨视角准入_v1','跨视角感知最小验证_v1','局部语义匹配单因素验证_v1','目标格定义与视觉尺度审查_v1']:
        p=BASE/directory/'阶段封存.json';s=read(p)
        for e in s['files']+s['sources']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        seals.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
    task=read(PREV/'冻结执行任务.json');split=read(PREV/'划分与任务.json');assert len(task['pairs'])==16 and len(split['dev_groups'])==10 and len(split['reserved_groups'])==20
    old=read(BASE/'局部语义匹配单因素验证_v1/输入与源码绑定.json')
    raws=[e for e in old['raw_used'] if e['path'] in {p['ground_path'] for p in task['pairs']}];assert len(raws)==16
    for e in raws+old['models']:assert sha(ROOT/e['path'])==e['sha256']
    life=read(ROOT/'docs/维护/版本生命周期.json')
    protected=[{'path':p,'sha256':sha(ROOT/p)} for p in ['project/local_policy_default.json','project/local_policy_defaults_v1.json',life['current']['final_report'],life['current']['final_report_pdf']]]+old['models']
    dep=['project/src/data/mmgag_views_v1.py','project/src/agents/crossview_local_match_v1.py','models/clip_vit_b32_public_v1/preprocessor_config.json','models/clip_vit_b32_public_v1/config.json']
    ref=[PREV/p for p in ['地面特征.npy','航拍特征.npy','逐对排序结果.json','感知对照汇总.json','划分与任务.json','冻结执行任务.json']]
    implementation=Path('D:/PYTHON/Lib/site-packages/transformers/models/clip/modeling_clip.py')
    OUT.mkdir();(OUT/'核验').mkdir()
    binding={'task':'CV-005','date':'2026-10-17','sources':[{'path':p,'sha256':sha(ROOT/p)} for p in SOURCES],'dependencies':[{'path':p,'sha256':sha(ROOT/p)} for p in dep],
      'raw_used':raws,'protected':protected,'previous_seals':seals,'references':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in ref],
      'installed_clip':{'path':str(implementation),'sha256':sha(implementation)},'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','Pillow']}}
    put('输入与源码绑定.json',binding);put('冻结16对副本.json',task)
    put('目标语义约定.json',{'localization_target':'capture_point','evaluation_label':'existing CSV/code-implied center proxy cell','decision':'carry forward original task meaning as project convention, not a new object annotation or measured geolocation claim',
      'unchanged_pairs':16,'source_center':[5,5],'visible_landmark_role':'possible evidence about capture location; not automatically the success cell',
      'geographic_footprint':'unverified, all16 retained','public_fields':['target_rgb','modality','localization_target'],'evaluator_only_fields':['goal_index','CSV_lat_lon','source_id','candidate_bank'],
      'new_landmark_task_rule':'requires explicit object labels and a new data/protocol version; never relabel these16 based on scores'})
    cfg=read(ROOT/'models/clip_vit_b32_public_v1/preprocessor_config.json');engineering=[]
    for p in task['pairs']:
        rgb=clean_rgb(ROOT/p['ground_path'],ground=True);target=public_target({'target_rgb':rgb,'modality':'ground','localization_target':'capture_point'})
        crop=ground_pixels(target.target_rgb,cfg);full=fullframe_pixels(target.target_rgb,cfg);g=fullframe_geometry(rgb)
        assert crop.shape==full.shape==(3,224,224) and crop.dtype==full.dtype==np.float32 and np.isfinite(full).all()
        x0,y0,x1,y1=g['content_box'];mask=np.ones((224,224),bool);mask[y0:y1,x0:x1]=False;assert not full[:,mask].any()
        engineering.append({'pair_id':p['pair_id'],'geometry':g,'crop_input_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),'fullframe_input_sha256':hashlib.sha256(full.tobytes()).hexdigest(),
          'shape':[3,224,224],'normalized_padding_exactly_zero':True,'hidden_metadata_in_public_target':False})
    put('零forward工程核验.json',{'passed':True,'pairs':engineering,'model_forward_calls':0,'all_original_pixels_included_before_resampling':True,'does_not_prove_target_association':True})
    put('冻结协议.json',{'task':'CV-005','date':'2026-10-17','only_new_factor':'target-photo input center-crop replaced by full-frame aspect-preserving resize+fixed mean padding; unchanged global cosine scorer',
      'rationale':'CV-004 measured56.28%-75.17% area retention and observed context/landmark risks; causal relevance unproved',
      'task_semantics':'capture point proxy cell, unchanged from previousqueue, no landmark relabeling','pairs':16,'development_groups':10,'reserved_groups':20,
      'arms':['CropReplay','FullFrame'],'conditions':['Full','Mean','Wrong'],'scoring':'unchanged normalized global cosine, float64 from savedfloat32 vectors; CV-002 hash ties',
      'controls':'Mean of16 normalized target vectors within each arm then normalize; Wrong same frozen different-group pair mapping',
      'input_rule':'longside224, shortside floor minimum1, centered zero padding in normalized tensor, remainder right/bottom, PIL bicubic; no formula/position/model sweeps',
      'batch_size':8,'maximum_ground_input_attempts':32,'maximum_encoder_forward_calls':4,'new_aerial_forward_calls':0,'cuda_allocation_limit_bytes':2*1024**3,
      'device_rule':'CUDA if free>=2560MiB at start else CPU; no post-score choice/retry','crop_replay_tolerance':1e-5,'replay_must_pass_before_fullframe':True,
      'output_limit_bytes':4*1024**2,'training_steps':0,'new_training_parameters':0,'model_api_calls':0,'navigation_episodes':0,'downloads':0,
      'candidate_gates':{'all16_complete':True,'full_top1_minimum':4,'full_minus_mean_minimum':2,'full_minus_wrong_minimum':2,'groups_net_better_than_both_minimum':3},
      'secondary':'same-queue CropReplay vs FullFrame recovered/harmed Top1/Top5 and pergroup results, not navigation SR',
      'stop':'single fixed processor; if any required gate fails close candidate, no fill/crop/scorer/layer/model/queue sweep, no reserved consumption; no automatic fourth zero-training probe',
      'interpretation':'development observation only; preserved input area does not prove better alignment; padding changes pretrained input distribution and may hurt',
      'online_boundary':'evaluator caches25 aerial candidates for ranking only; no bank/GPS/goal information enters encoder or public target'})
    put('首forward前绑定.json',{'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(OUT.glob('*.json'))],**{k:binding[k] for k in ['sources','dependencies','raw_used','protected','references','previous_seals']}})
    print(json.dumps({'prepared':True,'public_goal':'capture_point','pairs':16,'maximum_ground_inputs':32,'maximum_forward_calls':4,'new_aerial_forwards':0,'reserved':20,'output_limit_bytes':4*1024**2}))
if __name__=='__main__':main()
