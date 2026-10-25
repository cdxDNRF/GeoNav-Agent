"""CV-003 frozen local-token probe; evaluator bank never enters a policy."""
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
import numpy as np
from data.mmgag_views_v1 import clean_rgb,ground_pixels,aerial_pixels
from agents.crossview_local_match_v1 import stored_unit_tokens,unit_tokens,symmetric_maxsim,project_last_layer,check_global_replay,rank_cells
OUT=ROOT/'DATA/processed_data/MM-GAG/局部语义匹配单因素验证_v1'
PREV=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
MODEL=ROOT/'models/clip_vit_b32_public_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def new(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def progress(x):
    tmp=OUT/'工程临时/执行状态.tmp';tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,OUT/'执行状态.json')

def main():
    import torch,transformers
    from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
    from safetensors.torch import load_file
    assert not (OUT/'执行状态.json').exists(),'Consumed batch: inspect state and saved caches; no blind rerun.'
    binding=read(OUT/'首forward前绑定.json');protocol=read(OUT/'冻结协议.json');pairs=read(OUT/'冻结执行任务.json')['pairs']
    for key in ['files','sources','dependencies','models','raw_used','previous_bound_files','protected_current_files']:
        for e in binding[key]:assert digest(ROOT/e['path'])==e['sha256'],e['path']
    details=read(OUT/'输入与源码绑定.json');impl=details['installed_clip_implementation'];assert digest(impl['path'])==impl['sha256']
    torch.manual_seed(20261016);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    free=int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else None
    device='cuda:0' if free is not None and free>=2560*1024**2 else 'cpu'
    state={'task':'CV-003','state':'running','pid':os.getpid(),'device':device,'initial_cuda_free_bytes':free,'ground_done':0,'aerial_done':0,
      'attempted_input_images':0,'successful_forward_calls':0,'torch':torch.__version__,'transformers':transformers.__version__,
      'training_steps':0,'navigation_episodes':0,'model_api_calls':0,'reserved_source_images_encoded':0,
      'recovery':'No blind rerun. Audit completed caches only; failed batch requires a new explicitly documented recovery version.'}
    progress(state);start=time.monotonic();resources=[];replay={}
    local={k:np.lib.format.open_memmap(OUT/n,mode='w+',dtype=np.float16,shape=(size,49,512)) for k,n,size in [('ground','地面局部特征.npy',16),('aerial','航拍局部特征.npy',400)]}
    glob={k:np.lib.format.open_memmap(OUT/n,mode='w+',dtype=np.float32,shape=(size,512)) for k,n,size in [('ground','地面全局复现.npy',16),('aerial','航拍全局复现.npy',400)]}
    processor=read(MODEL/'preprocessor_config.json')
    try:
        for kind in ['ground','aerial']:
            if kind=='ground':
                cfg=read(MODEL/'config.json')['vision_config'];cfg['projection_dim']=512
                raw=torch.load(MODEL/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
                selected={k:v for k,v in raw.items() if k.startswith(('vision_model.','visual_projection.'))}
            else:
                cfg=read(ROOT/'models/Sat2Cap/config.json');raw=load_file(str(ROOT/'models/Sat2Cap/model.safetensors'),device='cpu');selected=dict(raw)
            config=CLIPVisionConfig(**cfg);config._attn_implementation='eager';model=CLIPVisionModelWithProjection(config)
            ignored=[]
            for k in list(selected):
                if k.endswith('.position_ids') and k not in model.state_dict():
                    assert selected[k].flatten().tolist()==list(range(50));ignored.append(k);selected.pop(k)
            loaded=model.load_state_dict(selected,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
            model.requires_grad_(False);model.eval();model.to(device);params=sum(p.numel() for p in model.parameters());del raw,selected
            if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            inputs=[p['ground_path'] for p in pairs] if kind=='ground' else [s for p in pairs for s in p['aerial_window_paths']]
            reference=np.load(PREV/('地面特征.npy' if kind=='ground' else '航拍特征.npy'),allow_pickle=False)
            phase=time.monotonic();max_global=max_cls=0.0
            for offset in range(0,len(inputs),protocol['batch_size']):
                batch=inputs[offset:offset+protocol['batch_size']]
                images=[clean_rgb(ROOT/p,ground=kind=='ground') for p in batch]
                arrays=[ground_pixels(a,processor) if kind=='ground' else aerial_pixels(a) for a in images]
                pixels=torch.from_numpy(np.stack(arrays)).to(device)
                state['attempted_input_images']+=len(batch);assert state['attempted_input_images']<=protocol['maximum_input_attempts'];progress(state)
                # Only current image pixels, no filename/GPS/answer/candidate-bank fields.
                with torch.inference_mode():
                    output=model(pixel_values=pixels)
                    cls,patches=project_last_layer(model,output.last_hidden_state)
                    values=output.image_embeds.detach().cpu().numpy()
                    max_cls=max(max_cls,check_global_replay(cls.detach().cpu().numpy(),values,protocol['cls_reconstruction_tolerance']))
                    max_global=max(max_global,check_global_replay(values,reference[offset:offset+len(batch)],protocol['global_replay_tolerance']))
                    cached=stored_unit_tokens(patches.detach().cpu().numpy())
                assert cached.shape==(len(batch),49,512)
                glob[kind][offset:offset+len(batch)]=values;glob[kind].flush()
                local[kind][offset:offset+len(batch)]=cached;local[kind].flush()
                state[kind+'_done']=offset+len(batch);state['successful_forward_calls']+=1
                if device.startswith('cuda'):assert torch.cuda.max_memory_allocated()<=protocol['cuda_allocation_limit_bytes']
                progress(state)
                if offset%80==0:print(f'{kind}: {state[kind+"_done"]}/{len(inputs)}, global delta {max_global:.3g}',flush=True)
                del output,cls,patches,pixels,values,cached,images,arrays
            replay[kind]={'maximum_global_delta':max_global,'maximum_cls_delta':max_cls,'images':len(inputs),'passed':True}
            resources.append({'encoder':kind,'parameters_used':params,'images':len(inputs),'ignored_nonparameter_buffers':ignored,'seconds':time.monotonic()-phase,
              'peak_cuda_allocated_bytes':int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else 0,
              'peak_cuda_reserved_bytes':int(torch.cuda.max_memory_reserved()) if device.startswith('cuda') else 0,'gradients_enabled':False,'training':False})
            del model
            if device.startswith('cuda'):torch.cuda.empty_cache()
        assert state['ground_done']==16 and state['aerial_done']==400 and state['attempted_input_images']==416
        new('全局复现核验.json',{'passed':True,'before_local_scoring':True,'by_encoder':replay,'tolerance':protocol['global_replay_tolerance']})
        # All scores use the persisted float16 values, never transient high-precision patches.
        del local,glob
        g=np.load(OUT/'地面局部特征.npy',allow_pickle=False);a=np.load(OUT/'航拍局部特征.npy',allow_pickle=False).reshape(16,25,49,512)
        normalized=np.stack([unit_tokens(x) for x in g]);mean=unit_tokens(normalized.mean(axis=0))
        indices={p['pair_id']:i for i,p in enumerate(pairs)};rows=[];counts={k:0 for k in ['Full','Mean','Wrong']};top5=dict(counts);ranks={k:[] for k in counts};groups={}
        for i,p in enumerate(pairs):
            group=groups.setdefault(p['group_id'],{'planned':0,**dict.fromkeys(counts,0)});group['planned']+=1
            for name,target in {'Full':g[i],'Mean':mean,'Wrong':g[indices[p['wrong_target_pair_id']]]}.items():
                scores=np.array([symmetric_maxsim(current,target) for current in a[i]],np.float64)
                order=rank_cells(scores,p['pair_id']);truth=p['goal_index_evaluator_only'];pos=order.index(truth)+1
                counts[name]+=pos==1;top5[name]+=pos<=5;group[name]+=pos==1;ranks[name].append(pos)
                rows.append({'pair_id':p['pair_id'],'group_id':p['group_id'],'condition':name,'goal_index_evaluator_only':truth,'scores':scores.tolist(),'ranking':order,
                  'true_rank':pos,'top1':pos==1,'top5':pos<=5,'true_score':float(scores[truth]),'true_minus_best_other':float(scores[truth]-max(scores[j] for j in range(25) if j!=truth))})
        lookup={(r['pair_id'],r['condition']):r for r in rows};contrasts={}
        for name in ['Mean','Wrong']:
            rec=sum(lookup[p['pair_id'],'Full']['top1'] and not lookup[p['pair_id'],name]['top1'] for p in pairs)
            harm=sum(lookup[p['pair_id'],name]['top1'] and not lookup[p['pair_id'],'Full']['top1'] for p in pairs)
            contrasts[name]={'recovered':rec,'harmed':harm,'net':rec-harm}
        old={(r['pair_id'],r['condition']):r for r in read(PREV/'逐对排序结果.json')['rows']};oldcontrast={}
        for metric in ['top1','top5']:
            rec=sum(lookup[p['pair_id'],'Full'][metric] and not old[p['pair_id'],'Full'][metric] for p in pairs)
            harm=sum(old[p['pair_id'],'Full'][metric] and not lookup[p['pair_id'],'Full'][metric] for p in pairs)
            oldcontrast[metric]={'recovered':rec,'harmed':harm,'net':rec-harm}
        positive=sum(v['Full']>v['Mean'] and v['Full']>v['Wrong'] for v in groups.values())
        gates={'all_16_complete':True,'full_top1_at_least4':counts['Full']>=4,'full_minus_mean_at_least2':counts['Full']-counts['Mean']>=2,
          'full_minus_wrong_at_least2':counts['Full']-counts['Wrong']>=2,'positive_groups_at_least3':positive>=3}
        summary={'task':'CV-003','planned_pairs':16,'development_groups':len(groups),'reserved_groups':20,'top1_counts':counts,'top5_counts':top5,
          'top1_rates':{k:v/16 for k,v in counts.items()},'median_true_rank':{k:float(np.median(v)) for k,v in ranks.items()},'per_group':groups,
          'groups_positive_against_both_controls':positive,'contrasts':contrasts,'local_vs_global_full':oldcontrast,'gates':gates,'candidate_passed':all(gates.values()),
          'navigation_SR':None,'training_steps':0,'model_api_calls':0,'formal_confirmation':False,'single_candidate':True,
          'interpretation':'Offline development association only; final patch tokens were not explicitly trained for cross-view alignment.'}
        new('逐对排序结果.json',{'rows':rows});new('局部匹配对照汇总.json',summary)
        features=[OUT/n for n in ['地面局部特征.npy','航拍局部特征.npy','地面全局复现.npy','航拍全局复现.npy']]
        new('运行资源.json',{'device':device,'elapsed_seconds':time.monotonic()-start,'encoders':resources,'attempted_input_images':416,'successful_forward_calls':state['successful_forward_calls'],
          'feature_files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in features],
          'frozen_models':True,'reserved_source_images_encoded':0,'new_training_parameters':0,'training_steps':0,'navigation_episodes':0,'model_api_calls':0,'downloads':0})
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=protocol['output_limit_bytes']
        state['state']='completed';state['candidate_passed']=summary['candidate_passed'];progress(state)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as exc:
        state['state']='failed';state['error_type']=type(exc).__name__;state['error']=str(exc);progress(state);raise
if __name__=='__main__':main()
