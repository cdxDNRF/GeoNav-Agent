"""CV-005 one fixed full-frame target trial. No navigation or new aerial encoding."""
import hashlib,json,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
import numpy as np
from data.mmgag_views_v1 import clean_rgb,ground_pixels,normalize_features
from data.crossview_goal_semantics_v1 import public_target,fullframe_pixels
from agents.crossview_local_match_v1 import rank_cells,check_global_replay
BASE=ROOT/'DATA/processed_data/MM-GAG';PREV=BASE/'跨视角感知最小验证_v1';OUT=BASE/'拍摄点语义与完整目标图验证_v1';MODEL=ROOT/'models/clip_vit_b32_public_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def new(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def progress(x):
    p=OUT/'核验/执行状态.tmp';p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(p,OUT/'执行状态.json')
def main():
    import torch,transformers
    from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
    assert not (OUT/'执行状态.json').exists(),'Consumed model batch; inspect state, never blind rerun'
    first=read(OUT/'首forward前绑定.json');protocol=read(OUT/'冻结协议.json');pairs=read(OUT/'冻结16对副本.json')['pairs']
    for key in ['files','sources','dependencies','raw_used','protected','references','previous_seals']:
        for e in first[key]:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    impl=read(OUT/'输入与源码绑定.json')['installed_clip'];assert sha(impl['path'])==impl['sha256']
    torch.manual_seed(20261016);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    free=int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else None;device='cuda:0' if free is not None and free>=2560*1024**2 else 'cpu'
    state={'task':'CV-005','state':'running','pid':os.getpid(),'device':device,'initial_cuda_free_bytes':free,'ground_input_attempts':0,'successful_forward_calls':0,'CropReplay_done':0,'FullFrame_done':0,
      'new_aerial_forward_calls':0,'training_steps':0,'navigation_episodes':0,'model_api_calls':0,'reserved_encoded':0,'recovery':'Inspect saved state; no blind retry of prepare/runner'}
    progress(state);started=time.monotonic()
    try:
        cfg=read(MODEL/'config.json')['vision_config'];cfg['projection_dim']=512;config=CLIPVisionConfig(**cfg);config._attn_implementation='eager';model=CLIPVisionModelWithProjection(config)
        raw=torch.load(MODEL/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True);weights={k:v for k,v in raw.items() if k.startswith(('vision_model.','visual_projection.'))};ignored=[]
        for k in list(weights):
            if k.endswith('.position_ids') and k not in model.state_dict():assert weights[k].flatten().tolist()==list(range(50));weights.pop(k);ignored.append(k)
        model.load_state_dict(weights,strict=True);del raw,weights
        model.requires_grad_(False);model.eval();model.to(device);parameters=sum(p.numel() for p in model.parameters())
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        cfg=read(MODEL/'preprocessor_config.json');engineering={p['pair_id']:p for p in read(OUT/'零forward工程核验.json')['pairs']};reference=np.load(PREV/'地面特征.npy',allow_pickle=False)
        maxdelta=0;features={};input_hash_checks=0
        for arm in ['CropReplay','FullFrame']:
            dest=np.lib.format.open_memmap(OUT/(arm+'地面特征.npy'),mode='w+',dtype=np.float32,shape=(16,512))
            for offset in range(0,16,8):
                arrays=[]
                for p in pairs[offset:offset+8]:
                    rgb=clean_rgb(ROOT/p['ground_path'],ground=True);target=public_target({'target_rgb':rgb,'modality':'ground','localization_target':'capture_point'})
                    values=ground_pixels(target.target_rgb,cfg) if arm=='CropReplay' else fullframe_pixels(target.target_rgb,cfg)
                    expected=engineering[p['pair_id']]['crop_input_sha256' if arm=='CropReplay' else 'fullframe_input_sha256'];assert hashlib.sha256(values.tobytes()).hexdigest()==expected
                    arrays.append(values);input_hash_checks+=1
                pixels=torch.from_numpy(np.stack(arrays)).to(device);state['ground_input_attempts']+=len(arrays);assert state['ground_input_attempts']<=32;progress(state)
                with torch.inference_mode():values=model(pixel_values=pixels).image_embeds.detach().cpu().numpy()
                assert values.shape==(8,512) and np.isfinite(values).all()
                if arm=='CropReplay':maxdelta=max(maxdelta,check_global_replay(values,reference[offset:offset+8],protocol['crop_replay_tolerance']))
                dest[offset:offset+8]=values;dest.flush();state[arm+'_done']=offset+8;state['successful_forward_calls']+=1
                if device.startswith('cuda'):assert torch.cuda.max_memory_allocated()<=protocol['cuda_allocation_limit_bytes']
                progress(state);del pixels,values,arrays
            features[arm]=OUT/(arm+'地面特征.npy');del dest
            if arm=='CropReplay':new('原裁剪复验.json',{'passed':True,'maximum_delta':maxdelta,'tolerance':protocol['crop_replay_tolerance'],'fullframe_not_started_before_pass':True})
            print(arm+' complete',flush=True)
        assert state['ground_input_attempts']==32 and state['successful_forward_calls']==4
        aerial=normalize_features(np.load(PREV/'航拍特征.npy',allow_pickle=False).reshape(16,25,512));index={p['pair_id']:i for i,p in enumerate(pairs)};rows=[];summaries={}
        for arm,path in features.items():
            g=normalize_features(np.load(path,allow_pickle=False));mean=normalize_features(g.mean(axis=0));counts=dict.fromkeys(['Full','Mean','Wrong'],0);top5=dict(counts);ranks={k:[] for k in counts};groups={}
            for i,p in enumerate(pairs):
                group=groups.setdefault(p['group_id'],{'planned':0,**dict.fromkeys(counts,0)});group['planned']+=1
                for name,target in {'Full':g[i],'Mean':mean,'Wrong':g[index[p['wrong_target_pair_id']]]}.items():
                    scores=aerial[i]@target;order=rank_cells(scores,p['pair_id']);truth=p['goal_index_evaluator_only'];pos=order.index(truth)+1
                    counts[name]+=pos==1;top5[name]+=pos<=5;group[name]+=pos==1;ranks[name].append(pos)
                    rows.append({'arm':arm,'condition':name,'pair_id':p['pair_id'],'group_id':p['group_id'],'scores':scores.tolist(),'ranking':order,'true_rank':pos,'top1':pos==1,'top5':pos<=5})
            positive=sum(x['Full']>x['Mean'] and x['Full']>x['Wrong'] for x in groups.values())
            gates={'all16_complete':True,'full_top1_at_least4':counts['Full']>=4,'full_minus_mean_at_least2':counts['Full']-counts['Mean']>=2,'full_minus_wrong_at_least2':counts['Full']-counts['Wrong']>=2,'positive_groups_at_least3':positive>=3}
            summaries[arm]={'top1_counts':counts,'top5_counts':top5,'median_true_rank':{k:float(np.median(v)) for k,v in ranks.items()},'per_group':groups,'positive_groups_against_both':positive,'gates':gates,'candidate_passed':all(gates.values())}
        lookup={(x['arm'],x['pair_id'],x['condition']):x for x in rows};contrasts={}
        for arm in summaries:
            ctl={}
            for name in ['Mean','Wrong']:
                rec=sum(lookup[arm,p['pair_id'],'Full']['top1'] and not lookup[arm,p['pair_id'],name]['top1'] for p in pairs)
                harm=sum(lookup[arm,p['pair_id'],name]['top1'] and not lookup[arm,p['pair_id'],'Full']['top1'] for p in pairs)
                ctl[name]={'recovered':rec,'harmed':harm,'net':rec-harm}
            summaries[arm]['target_contrasts']=ctl
        for metric in ['top1','top5']:
            rec=sum(lookup['FullFrame',p['pair_id'],'Full'][metric] and not lookup['CropReplay',p['pair_id'],'Full'][metric] for p in pairs)
            harm=sum(lookup['CropReplay',p['pair_id'],'Full'][metric] and not lookup['FullFrame',p['pair_id'],'Full'][metric] for p in pairs)
            contrasts[metric]={'recovered':rec,'harmed':harm,'net':rec-harm}
        result={'task':'CV-005','target_semantics':'capture_point_proxy_cell','planned_pairs':16,'development_groups':10,'reserved_groups':20,'arms':summaries,'fullframe_vs_crop':contrasts,
          'candidate_passed':summaries['FullFrame']['candidate_passed'],'navigation_SR':None,'training_steps':0,'new_training_parameters':0,'single_candidate':True,'formal_confirmation':False}
        new('逐对排序结果.json',{'rows':rows});new('完整目标图对照汇总.json',result)
        new('运行资源.json',{'device':device,'torch':torch.__version__,'transformers':transformers.__version__,'ground_input_attempts':32,'successful_forward_calls':4,'new_aerial_forward_calls':0,
          'input_tensor_hash_checks':input_hash_checks,'parameters_used':parameters,'model_frozen':True,'peak_cuda_allocated_bytes':int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else 0,
          'peak_cuda_reserved_bytes':int(torch.cuda.max_memory_reserved()) if device.startswith('cuda') else 0,'elapsed_seconds':time.monotonic()-started,'ignored_nonparameter_buffers':ignored,
          'features':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in features.values()],
          'training_steps':0,'navigation_episodes':0,'model_api_calls':0,'downloads':0,'reserved_encoded':0})
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=protocol['output_limit_bytes']
        state['state']='completed';state['candidate_passed']=result['candidate_passed'];progress(state);print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:
        state['state']='failed';state['error_type']=type(exc).__name__;state['error']=str(exc);progress(state);raise
if __name__=='__main__':main()
