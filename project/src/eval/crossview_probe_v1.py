"""CV-002 one frozen offline association probe; never a navigation policy."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
import numpy as np
from data.mmgag_views_v1 import clean_rgb,ground_pixels,aerial_pixels,normalize_features
OUT=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
MODEL=ROOT/'models/clip_vit_b32_public_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def new(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def progress(x):
    p=OUT/'执行状态.json';tmp=OUT/'工程临时/执行状态.tmp';tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def rank(scores,pair):
    return sorted(range(25),key=lambda j:(-float(scores[j]),hashlib.sha256(f'CV-002/tie/v1:{pair}:{j}'.encode()).hexdigest()))
def main():
    import torch,transformers
    from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
    from safetensors.torch import load_file
    binding=read(OUT/'首forward前绑定.json');protocol=read(OUT/'冻结协议.json');task=read(OUT/'冻结执行任务.json');pairs=task['pairs']
    assert not (OUT/'执行状态.json').exists(),'Do not rerun consumed forward; inspect saved state.'
    for e in binding['files']+binding['sources']+binding['models']+binding['raw_used']:assert digest(ROOT/e['path'])==e['sha256'],e['path']
    torch.manual_seed(20261016);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    free=None
    if torch.cuda.is_available():free=int(torch.cuda.mem_get_info()[0])
    device='cuda:0' if free is not None and free>=2560*1024**2 else 'cpu'
    state={'task':'CV-002','state':'running','pid':os.getpid(),'device':device,'initial_cuda_free_bytes':free,
      'ground_done':0,'aerial_done':0,'attempted_input_images':0,'successful_forward_calls':0,'torch':torch.__version__,'transformers':transformers.__version__,
      'model_api_calls':0,'training_steps':0,'navigation_episodes':0,'recovery':'No blind rerun; completed data audit uses saved features only.'}
    progress(state);start=time.monotonic();resources=[]
    ground=np.lib.format.open_memmap(OUT/'地面特征.npy',mode='w+',dtype=np.float32,shape=(16,512))
    aerial=np.lib.format.open_memmap(OUT/'航拍特征.npy',mode='w+',dtype=np.float32,shape=(400,512))
    processor=read(MODEL/'preprocessor_config.json')
    try:
        for kind in ['ground','aerial']:
            if kind=='ground':
                cfg=read(MODEL/'config.json')['vision_config'];cfg['projection_dim']=512
                raw=torch.load(MODEL/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
                selected={k:v for k,v in raw.items() if k.startswith(('vision_model.','visual_projection.'))}
            else:
                cfg=read(ROOT/'models/Sat2Cap/config.json');raw=load_file(str(ROOT/'models/Sat2Cap/model.safetensors'),device='cpu');selected=dict(raw)
            config=CLIPVisionConfig(**cfg);config._attn_implementation='eager'
            model=CLIPVisionModelWithProjection(config)
            ignored=[]
            for k in list(selected):
                if k.endswith('.position_ids') and k not in model.state_dict():
                    assert selected[k].flatten().tolist()==list(range(50));ignored.append(k);selected.pop(k)
            loaded=model.load_state_dict(selected,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
            param_count=sum(p.numel() for p in model.parameters());model.requires_grad_(False);model.eval();model.to(device)
            del raw,selected
            if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            inputs=[p['ground_path'] for p in pairs] if kind=='ground' else [path for p in pairs for path in p['aerial_window_paths']]
            dest=ground if kind=='ground' else aerial;phase_start=time.monotonic()
            for offset in range(0,len(inputs),8):
                batch=inputs[offset:offset+8]
                images=[clean_rgb(ROOT/p,ground=kind=='ground') for p in batch]
                arrays=[ground_pixels(a,processor) if kind=='ground' else aerial_pixels(a) for a in images]
                # The model receives only sanitized pixel tensors; no task/file/GPS/goal fields.
                pixels=torch.from_numpy(np.stack(arrays)).to(device)
                state['attempted_input_images']+=len(batch);assert state['attempted_input_images']<=416;progress(state)
                with torch.inference_mode():values=model(pixel_values=pixels).image_embeds.detach().cpu().numpy()
                assert values.shape==(len(batch),512) and np.isfinite(values).all()
                dest[offset:offset+len(batch)]=values;dest.flush()
                state[kind+'_done']=offset+len(batch);state['successful_forward_calls']+=1
                if device.startswith('cuda'):
                    allocated=torch.cuda.max_memory_allocated();assert allocated<=protocol['cuda_allocation_limit_bytes'],'CUDA allocation ceiling exceeded'
                progress(state)
                del pixels,values,images,arrays
            resources.append({'encoder':kind,'parameters_used':param_count,'images':len(inputs),'ignored_nonparameter_buffers':ignored,
              'seconds':time.monotonic()-phase_start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
              'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0,'gradients_enabled':False,'training':False})
            del model
            if device.startswith('cuda'):torch.cuda.empty_cache()
        assert state['attempted_input_images']==416 and state['ground_done']==16 and state['aerial_done']==400
        g=normalize_features(ground);a=normalize_features(np.asarray(aerial).reshape(16,25,512));mean=normalize_features(g.mean(axis=0))
        pair_index={p['pair_id']:i for i,p in enumerate(pairs)};rows=[];per_group={};counts={k:0 for k in ['Full','Mean','Wrong']}
        top5=dict(counts);ranks={k:[] for k in counts}
        for i,p in enumerate(pairs):
            targets={'Full':g[i],'Mean':mean,'Wrong':g[pair_index[p['wrong_target_pair_id']]]}
            group=per_group.setdefault(p['group_id'],{'planned':0,'Full':0,'Mean':0,'Wrong':0});group['planned']+=1
            for condition,target in targets.items():
                scores=a[i]@target;order=rank(scores,p['pair_id']);truth=p['goal_index_evaluator_only'];position=order.index(truth)+1
                hit=int(position==1);counts[condition]+=hit;top5[condition]+=int(position<=5);group[condition]+=hit;ranks[condition].append(position)
                rows.append({'pair_id':p['pair_id'],'group_id':p['group_id'],'condition':condition,'goal_index_evaluator_only':truth,
                    'scores':scores.tolist(),'ranking':order,'true_rank':position,'top1':bool(hit),'top5':position<=5,
                    'true_score':float(scores[truth]),'true_minus_best_other':float(scores[truth]-max(scores[j] for j in range(25) if j!=truth))})
        improving=sum(x['Full']>x['Mean'] and x['Full']>x['Wrong'] for x in per_group.values())
        gates={'all_16_complete':True,'full_top1_at_least4':counts['Full']>=4,'full_minus_mean_at_least2':counts['Full']-counts['Mean']>=2,
          'full_minus_wrong_at_least2':counts['Full']-counts['Wrong']>=2,'positive_groups_at_least3':improving>=3}
        contrasts={}
        for control in ['Mean','Wrong']:
            rec=harm=0
            for p in pairs:
                full=next(x for x in rows if x['pair_id']==p['pair_id'] and x['condition']=='Full')['top1']
                ctl=next(x for x in rows if x['pair_id']==p['pair_id'] and x['condition']==control)['top1']
                rec+=int(full and not ctl);harm+=int(ctl and not full)
            contrasts[control]={'recovered':rec,'harmed':harm,'net':rec-harm}
        summary={'task':'CV-002','planned_pairs':16,'development_groups':len(per_group),'reserved_groups':20,'top1_counts':counts,'top5_counts':top5,
          'top1_rates':{k:v/16 for k,v in counts.items()},'median_true_rank':{k:float(np.median(v)) for k,v in ranks.items()},
          'per_group':per_group,'groups_positive_against_both_controls':improving,'contrasts':contrasts,'gates':gates,'candidate_passed':all(gates.values()),
          'navigation_SR':None,'training_steps':0,'model_api_calls':0,'formal_confirmation':False,'single_candidate':True,
          'interpretation':'Offline development association only; full candidate bank must never be supplied to a navigating agent.'}
        new('逐对排序结果.json',{'rows':rows});new('感知对照汇总.json',summary)
        new('运行资源.json',{'device':device,'elapsed_seconds':time.monotonic()-start,'encoders':resources,'attempted_input_images':416,'successful_forward_calls':state['successful_forward_calls'],
          'feature_files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in [OUT/'地面特征.npy',OUT/'航拍特征.npy']],
          'frozen_models':True,'reserved_source_images_encoded':0,'new_training_parameters':0})
        state['state']='completed';state['candidate_passed']=summary['candidate_passed'];progress(state)
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=protocol['output_limit_bytes']
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as exc:
        state['state']='failed';state['error_type']=type(exc).__name__;progress(state)
        raise
if __name__=='__main__':main()
