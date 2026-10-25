"""CV-002 fixed metadata-only split, geometry and processor admission. Zero model forward."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.mmgag_views_v1 import tile_index,window_positions,numeric_window,clean_rgb,ground_pixels
OUT=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
PREV=ROOT/'DATA/processed_data/MM-GAG/数据核验与跨视角准入_v1'
MODEL=ROOT/'models/clip_vit_b32_public_v1'
SOURCE_PATHS=['project/src/data/mmgag_views_v1.py','project/src/tools/prepare_crossview_probe_v1.py','project/src/tools/acquire_clip_b32_v1.py','project/src/eval/crossview_probe_v1.py','project/src/eval/audit_crossview_probe_v1.py','project/src/tests/test_crossview_probe_v1.py']
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def select_groups(groups,hashes):
    ordered=sorted(groups,key=lambda g:h('CV-002/group/v1\n'+'\n'.join(sorted(hashes[s] for s in g))))
    # Lexicographically first 10-group set with >=16 usable pairs at a cap of 4/group.
    selected=[];start=0;capacity=0
    for position in range(10):
        slots=9-position
        for i in range(start,len(ordered)-slots):
            cap=min(4,len(ordered[i]));future=sorted((min(4,len(g)) for g in ordered[i+1:]),reverse=True)[:slots]
            if capacity+cap+sum(future)>=16:
                selected.append(ordered[i]);capacity+=cap;start=i+1;break
        else:raise ValueError('cannot freeze 16 pairs from ten groups')
    return selected,[g for g in ordered if g not in selected]
def geometry(pairs):
    mappings={'row':lambda r,c:(r,c),'column':lambda r,c:(c,r),
      'flip_row':lambda r,c:(10-r,c),'flip_column':lambda r,c:(r,10-c),
      'rotate180':lambda r,c:(10-r,10-c),'rotate90':lambda r,c:(c,10-r),
      'rotate270':lambda r,c:(10-c,r),'antitranspose':lambda r,c:(10-c,10-r)}
    probes=[(0,1),(1,0),(2,8),(8,2),(10,0),(0,10),(5,6),(6,5)]
    results=[]
    for pair in pairs:
        directory=ROOT/pair['aerial_directory'];canvas=None
        for path in sorted(directory.glob('stitch*.jpg')):
            with Image.open(path) as im:
                if im.size==(2816,2816):canvas=im.convert('RGB').copy();break
        if canvas is None:results.append({'pair_id':pair['pair_id'],'stitched_reference':False});continue
        files=tile_index(directory);scores={}
        reference={(r,c):np.asarray(canvas.crop((c*256,r*256,(c+1)*256,(r+1)*256)).resize((16,16),Image.Resampling.BOX),np.float64) for r,c in probes}
        for name,mapper in mappings.items():
            errors=[]
            for r,c in probes:
                with Image.open(files[mapper(r,c)]) as im:a=np.asarray(im.convert('RGB').resize((16,16),Image.Resampling.BOX),np.float64)
                errors.append(float(np.abs(a-reference[r,c]).mean()))
            scores[name]=float(np.mean(errors))
        order=sorted(scores,key=lambda x:scores[x]);best=order[0];margin=scores[order[1]]-scores[best]
        results.append({'pair_id':pair['pair_id'],'stitched_reference':True,'source_path':path.relative_to(ROOT).as_posix(),'mae':scores,'best':best,'margin':margin,'confident':scores[best]<=8 and margin>=3})
    confident=[x for x in results if x.get('confident')]
    axes={x['best'] for x in confident}
    passed=len(confident)>=3 and len(axes)==1 and next(iter(axes)) in ['row','column']
    return {'passed':passed,'first_index':next(iter(axes)) if passed else None,'confident_stitches':len(confident),'minimum_required':3,'max_mae':8,'minimum_margin':3,'results':results,'meaning':'image grid axes only; no north-up, map scale or GPS-to-pixel claim'}
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'核验').mkdir(exist_ok=True);(OUT/'工程临时').mkdir(exist_ok=True)
    assert not (OUT/'准备开始绑定.json').exists(),'Preparation is already consumed'
    previous=read(PREV/'阶段封存.json')
    for e in previous['files']+previous['sources']:assert digest(ROOT/e['path'])==e['sha256'],e['path']
    sources={s['id']:s for s in read(PREV/'逐源准入清单.json')['sources'] if s['static_assets_ready']}
    raw={e['path']:e for e in read(PREV/'输入文件与源码绑定.json')['raw_inputs']}
    identities={sid:raw[s['ground_files'][0]]['sha256'] for sid,s in sources.items()}
    groups=read(PREV/'来源接近与重叠组.json')['groups'];dev,hold=select_groups(groups,identities)
    dev=sorted(dev,key=lambda g:h('CV-002/group/v1\n'+'\n'.join(sorted(identities[s] for s in g))))
    ranked=[sorted(g,key=lambda sid:h('CV-002/source/v1\n'+identities[sid])) for g in dev]
    selected=[]
    for roundno in range(4):
        for i,g in enumerate(ranked):
            if roundno<len(g) and len(selected)<16:selected.append((i,g[roundno]))
    assert len(selected)==16 and len(hold)==20
    goals=sorted(range(25),key=lambda g:h('CV-002/goal-grid/v1:'+str(g)))[:16]
    pairs=[]
    for i,(group,sid) in enumerate(selected):
        s=sources[sid];pairs.append({'pair_id':f'cvp_{i:02d}','group_id':f'dev_{group:02d}','source_id_evaluator_only':sid,
         'ground_path':s['ground_files'][0],'aerial_directory':s['aerial_directories'][0],'goal_index_evaluator_only':goals[i]})
    for pair in pairs:
        others=[p for p in pairs if p['group_id']!=pair['group_id']]
        pair['wrong_target_pair_id']=min(others,key=lambda p:h('CV-002/wrong/v1:'+pair['pair_id']+':'+p['pair_id']))['pair_id']
    used=set()
    for pair in pairs:
        used.add(pair['ground_path']);used.update(p.relative_to(ROOT).as_posix() for p in (ROOT/pair['aerial_directory']).iterdir() if p.is_file())
    for p in used:assert digest(ROOT/p)==raw[p]['sha256'],p
    assets=[MODEL/'config.json',MODEL/'preprocessor_config.json',MODEL/'pytorch_model.bin',ROOT/'models/Sat2Cap/config.json',ROOT/'models/Sat2Cap/model.safetensors']
    binding={'task':'CV-002','sources':[{'path':p,'sha256':digest(ROOT/p)} for p in SOURCE_PATHS],
     'previous_seal':{'path':(PREV/'阶段封存.json').relative_to(ROOT).as_posix(),'sha256':digest(PREV/'阶段封存.json')},
     'raw_used':[raw[p] for p in sorted(used)],'models':[{'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size} for p in assets],
     'protected_defaults':[{'path':p,'sha256':digest(ROOT/p)} for p in ['project/local_policy_default.json','project/local_policy_defaults_v1.json']],
     'no_holdout_pixels_or_features':True}
    save('准备开始绑定.json',binding)
    save('划分与任务.json',{'dev_groups':dev,'reserved_groups':hold,'grouping_scope':'provisional groups from CV-001, not proven nonoverlapping geographic regions','selection':'fixed hash order with metadata-only capacity constraint, round-robin cap4/group','pairs':pairs})
    gconfig=read(MODEL/'config.json');ac=read(ROOT/'models/Sat2Cap/config.json');gc=gconfig['vision_config'];pp=read(MODEL/'preprocessor_config.json')
    arch=(gc['patch_size']==32 and ac['patch_size']==32 and gc['hidden_size']==768 and ac['hidden_size']==768 and gconfig['projection_dim']==512 and ac['projection_dim']==512)
    assert arch,'unsupported architecture'
    from transformers import CLIPImageProcessor
    processor=CLIPImageProcessor.from_pretrained(str(MODEL),local_files_only=True)
    deltas=[];orientation=[]
    for pair in pairs:
        image=clean_rgb(ROOT/pair['ground_path'],ground=True)
        a=ground_pixels(image,pp);ref=processor(images=image,return_tensors='np')['pixel_values'][0]
        deltas.append(float(np.abs(a-ref).max()))
        orientation.append({'pair_id':pair['pair_id'],'sanitized_rgb_shape':list(image.shape),'metadata_fields':0})
    geo=geometry(pairs)
    engineering={'architecture_compatible':arch,'processor_max_delta':max(deltas),'processor_deltas':deltas,'processor_tolerance':1e-5,
       'geometry':geo,'ground_sanitization':orientation,'passed':arch and max(deltas)<=1e-5 and geo['passed'],'model_forward_calls':0,
       'label_origin':'author code array-center (5,5) plus verified stitch/file axes; not external GPS-to-pixel survey'}
    save('工程核验.json',engineering)
    if not engineering['passed']:print(json.dumps(engineering,ensure_ascii=False));raise SystemExit('Engineering gate failed; no model forward')
    for pair in pairs:
        pair['aerial_window_paths']=[p.relative_to(ROOT).as_posix() for p in numeric_window(ROOT/pair['aerial_directory'],pair['goal_index_evaluator_only'],geo['first_index'])]
    # Separate final execution tasks: preparation inputs remain immutable.
    save('冻结执行任务.json',{'pairs':pairs,'first_numeric_index':geo['first_index']})
    protocol={'task':'CV-002','date':'2026-10-16','pairs':16,'aerial_inputs':400,'ground_inputs':16,'conditions':['Full','Mean','Wrong'],
      'dev_groups':10,'reserved_groups':20,'scoring':'L2-normalized 512D cosine; float64 diagnostic arithmetic',
      'mean':'mean of 16 normalized development ground vectors, L2 normalized','wrong':'fixed different development group target; no extra encoded images',
      'ties':'sort by decreasing score then SHA256(CV-002/tie/v1:pair_id:cell_index); never by truth',
      'batch_size':8,'maximum_forward_attempts_inputs':416,'cuda_allocation_limit_bytes':2*1024**3,
      'device_rule':'CUDA if free memory >=2560MiB at run start, otherwise CPU; no post-score device/model selection',
      'training_steps':0,'navigation_episodes':0,'model_api_calls':0,'output_limit_bytes':16*1024**2,
      'candidate_gates':{'all_16_complete':True,'full_top1_minimum':4,'full_minus_mean_minimum':2,'full_minus_wrong_minimum':2,'groups_net_better_than_both_minimum':3},
      'stopping':'one candidate; no tuning or reserved encoding after results; Top-k is not SR',
      'ground_model':'models/clip_vit_b32_public_v1/pytorch_model.bin; vision_model.* + visual_projection.* only, strict load',
      'aerial_model':'models/Sat2Cap/model.safetensors; frozen',
      'runtime_input':'pixels only, metadata sanitized; candidate bank allowed only to offline evaluator, never online agent',
      'geometry':'image coordinates; first_index='+geo['first_index']+'; no metric SG claim',
      'authority':'user started CV-002 and explicitly approved the one official CLIP download'}
    save('冻结协议.json',protocol)
    save('首forward前绑定.json',{'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p)} for p in sorted(OUT.glob('*.json'))], 'sources':binding['sources'],'models':binding['models'],'raw_used':binding['raw_used']})
    print(json.dumps({'frozen':True,'pairs':16,'dev_groups':10,'reserved_groups':20,'engineering':{'axis':geo['first_index'],'confident_stitches':geo['confident_stitches'],'processor_max_delta':max(deltas)}}))
if __name__=='__main__':main()
