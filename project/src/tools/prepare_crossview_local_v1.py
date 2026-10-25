"""CV-003 freezes one local-token factor against CV-002; no model execution."""
import hashlib,json,sys,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
PREV=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
OUT=ROOT/'DATA/processed_data/MM-GAG/局部语义匹配单因素验证_v1'
SOURCES=['project/src/agents/crossview_local_match_v1.py','project/src/tools/prepare_crossview_local_v1.py','project/src/eval/crossview_local_probe_v1.py','project/src/eval/audit_crossview_local_v1.py','project/src/tests/test_crossview_local_match_v1.py']
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    assert not OUT.exists(),'No overwrite or blind restart of a prepared batch'
    previous=read(PREV/'阶段封存.json')
    for e in previous['files']+previous['sources']:assert digest(ROOT/e['path'])==e['sha256'],e['path']
    first=read(PREV/'首forward前绑定.json')
    for e in first['models']+first['raw_used']+first['sources']:assert digest(ROOT/e['path'])==e['sha256'],e['path']
    task=read(PREV/'冻结执行任务.json');split=read(PREV/'划分与任务.json')
    assert len(task['pairs'])==16 and len(split['dev_groups'])==10 and len(split['reserved_groups'])==20
    baseline=read(PREV/'感知对照汇总.json');assert baseline['planned_pairs']==16
    OUT.mkdir(parents=True);(OUT/'核验').mkdir();(OUT/'工程临时').mkdir()
    protection=[]
    life=read(ROOT/'docs/维护/版本生命周期.json')
    for p in ['project/local_policy_default.json','project/local_policy_defaults_v1.json',life['current']['final_report'],life['current']['final_report_pdf']]:
        protection.append({'path':p,'sha256':digest(ROOT/p)})
    deps=['project/src/data/mmgag_views_v1.py','project/src/eval/crossview_probe_v1.py']
    local_model_impl=Path('D:/PYTHON/Lib/site-packages/transformers/models/clip/modeling_clip.py')
    binding={'task':'CV-003','date':'2026-10-17','sources':[{'path':p,'sha256':digest(ROOT/p)} for p in SOURCES],
      'dependencies':[{'path':p,'sha256':digest(ROOT/p)} for p in deps],
      'previous_seal':{'path':(PREV/'阶段封存.json').relative_to(ROOT).as_posix(),'sha256':digest(PREV/'阶段封存.json')},
      'previous_bound_files':previous['files'],'models':first['models'],'raw_used':first['raw_used'],
      'protected_current_files':protection,'installed_clip_implementation':{'path':str(local_model_impl),'sha256':digest(local_model_impl)},
      'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','Pillow','safetensors']}}
    write('输入与源码绑定.json',binding)
    # Identical queue/labels/controls. Small JSON copied, no image/weight/old feature copies.
    write('冻结执行任务.json',task);write('来源分组副本.json',split)
    protocol={'task':'CV-003','date':'2026-10-17','hypothesis':'single-image local features may distinguish neighboring aerial cells better than global cosine',
      'only_new_factor':'49-token symmetric MaxSim replacing global vector cosine; no edge/global mixture, layer sweep or new parameters',
      'planned_pairs':16,'development_groups':10,'reserved_groups':20,'input_images':{'ground':16,'aerial':400},'maximum_input_attempts':416,'batch_size':8,
      'global_baseline':(PREV/'感知对照汇总.json').relative_to(ROOT).as_posix(),
      'global_reference_files':[(PREV/n).relative_to(ROOT).as_posix() for n in ['地面特征.npy','航拍特征.npy']],
      'global_replay_tolerance':1e-5,'cls_reconstruction_tolerance':1e-5,'fail_before_scoring_on_replay_error':True,
      'projection':'apply each frozen vision_model.post_layernorm and visual_projection to final 50 hidden tokens; exclude CLS from matching',
      'local_token_count':49,'dimension':512,'storage':'normalize projected tokens in float32 then cast float16; reload float16 values as float64 and renormalize before all scores',
      'score':'0.5*(mean_i max_j cos(current_i,target_j)+mean_j max_i cos(current_i,target_j))',
      'mean_control':'normalize reloaded tokens of each dev target; mean across16 at each token position; renormalize each mean token',
      'wrong_control':'exact CV-002 wrong_target_pair_id from a different development group',
      'ties':'unchanged CV-002/tie/v1:pair_id:cell_index SHA ordering after decreasing float64 scores; no truth input',
      'cuda_allocation_limit_bytes':2*1024**3,'device_rule':'CUDA if free>=2560MiB at start, else CPU; no post-score model/device selection',
      'output_limit_bytes':32*1024**2,'new_training_parameters':0,'training_steps':0,'navigation_episodes':0,'model_api_calls':0,'downloads':0,
      'gates':{'all16_complete':True,'full_top1_minimum':4,'full_minus_mean_minimum':2,'full_minus_wrong_minimum':2,'positive_groups_against_both_minimum':3},
      'secondary_reporting':'same-queue Full local vs CV-002 Full global recovered/harmed; Top-5, true rank, all per-source/group results',
      'stopping':'one fixed candidate; keep failure if any necessary gate fails; no score-driven rerun/pooling/layer/model tuning or reserved encoding',
      'scope':'offline association, not navigation SR or formal confirmation; full 25-cell bank is evaluator-only',
      'label_scope':'CV-002 row-first numeric indices and code-implied center(5,5); no newly verified GPS-to-pixel survey or metric SG',
      'local_alignment_caveat':'projection is trained for global CLS; patch-token cross-view alignment remains an experimental hypothesis'}
    write('冻结协议.json',protocol)
    files=[{'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p)} for p in sorted(OUT.glob('*.json'))]
    write('首forward前绑定.json',{'files':files,'sources':binding['sources'],'dependencies':binding['dependencies'],'models':binding['models'],'raw_used':binding['raw_used'],'previous_bound_files':binding['previous_bound_files'],'protected_current_files':protection})
    print(json.dumps({'frozen':True,'pairs':16,'sources':len(SOURCES),'input_images':416,'reserved_groups':20,'max_bytes':32*1024**2},ensure_ascii=False))
if __name__=='__main__':main()
