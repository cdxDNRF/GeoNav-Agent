"""CV-013 real engineering; later single CV-012 run requires a separate authorization file."""
from __future__ import annotations
import argparse,hashlib,io,json,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
import numpy as np
BASE=ROOT/'DATA/processed_data/MM-GAG';DESIGN=BASE/'同任务对齐训练与目标对照协议_v1'
ENG=BASE/'同任务对齐执行器工程_v1';RUN=BASE/'同任务拍摄点对齐执行_v1'
PROTOCOL=DESIGN/'同任务对齐训练冻结协议.json';QUEUE=DESIGN/'固定编码与训练开发队列.json'
SOURCES=['agents/capturepoint_target_head_v1.py','eval/capturepoint_execution_scoring_v1.py','tools/run_capturepoint_execution_v1.py',
    'eval/audit_capturepoint_execution_v1.py','tests/test_capturepoint_execution_v1.py','documents/close_capturepoint_execution_v1.py',
    'agents/weak_alignment_head_v1.py','train/weak_scene_alignment_v1.py','data/mmgag_views_v1.py',
    'agents/crossview_local_match_v1.py','data/capturepoint_alignment_protocol_v1.py']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def entry(p):
    p=Path(p);return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def put(out,name,x):
    with (out/name).open('x',encoding='utf-8') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))
def verify(items):
    for e in items:assert sha(ROOT/e['path'])==e['sha256'],e['path']
def role_signature(doc):
    return sorted((r['source_id'],r['group'],r['role'],r.get('representative')) for r in doc['records'])
def event(out,kind,**data):
    with (out/'事件.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'event':kind,**data},ensure_ascii=False)+'\n');f.flush()
def state(out,name,x):put(out,'核验/'+name+'.json',x)


def environment():
    import torch,transformers
    torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20261016)
    free=int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else None
    return {'device':'cuda:0' if free is not None and free>=2560*1024**2 else 'cpu',
            'initial_cuda_free_bytes':free,'torch':torch.__version__,'transformers':transformers.__version__,'pid':os.getpid()}


def tensor_sha(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_encoder(kind,device):
    import torch
    from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
    if kind=='ground':
        cfg=read(ROOT/'models/clip_vit_b32_public_v1/config.json')['vision_config'];cfg['projection_dim']=512
        raw=torch.load(ROOT/'models/clip_vit_b32_public_v1/pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
        selected={k:v for k,v in raw.items() if k.startswith(('vision_model.','visual_projection.'))}
    else:
        from safetensors.torch import load_file
        cfg=read(ROOT/'models/Sat2Cap/config.json');selected=load_file(str(ROOT/'models/Sat2Cap/model.safetensors'),device='cpu')
    config=CLIPVisionConfig(**cfg);config._attn_implementation='eager';model=CLIPVisionModelWithProjection(config)
    ignored=[]
    for k in list(selected):
        if k.endswith('.position_ids') and k not in model.state_dict():
            assert selected[k].flatten().tolist()==list(range(50));ignored.append(k);selected.pop(k)
    model.load_state_dict(selected,strict=True);model.requires_grad_(False);model.eval();model.to(device)
    return model,ignored


def pixels(e,kind):
    from PIL import Image
    from data.mmgag_views_v1 import clean_rgb,ground_pixels,aerial_pixels
    p=(ROOT/e['path']).resolve();assert p.is_relative_to((ROOT/'DATA/raw_data/MM-GAG').resolve())
    raw=p.read_bytes();actual=hashlib.sha256(raw).hexdigest();assert actual==e['sha256'] and len(raw)==e['bytes']
    with Image.open(io.BytesIO(raw)) as src:
        if kind=='aerial':assert src.size==(256,256)
        rgb=clean_rgb(src,ground=kind=='ground')
    processor=read(ROOT/'models/clip_vit_b32_public_v1/preprocessor_config.json')
    a=ground_pixels(rgb,processor) if kind=='ground' else aerial_pixels(rgb)
    assert a.shape==(3,224,224) and a.dtype==np.float32 and np.isfinite(a).all()
    return a,{'path':e['path'],'raw_sha256':actual,'pixel_tensor_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'sanitized_metadata_fields':0}


def prepare():
    import torch
    from train.weak_scene_alignment_v1 import engineering_checks
    assert not ENG.exists(),'Engineering preparation already consumed'
    design=read(DESIGN/'输入与协议源码绑定.json');seal=read(DESIGN/'阶段封存.json')
    verify(seal['files']+seal['sources']);verify(design['protected']+design['reference_math_sources_only'])
    protocol=read(PROTOCOL);assert protocol['research_design_frozen'] and protocol['training']['updates']==500
    ENG.mkdir();(ENG/'特征').mkdir();(ENG/'核验').mkdir()
    torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    checks=engineering_checks();put(ENG,'核验/合成梯度与恢复.json',checks)
    role_path=BASE/'同任务来源用途账本_v1.json';put(ENG,'角色快照.json',read(role_path))
    queue=read(QUEUE)
    inputs=[*queue['ground_encoder_inputs'][:8],*queue['aerial_encoder_inputs'][:8]]
    put(ENG,'工程范围冻结.json',{'task':'CV-013','date':'2026-10-18','user_request':'continue implement/bind executor and real model engineering',
        'encoder_inputs':16,'encoder_forward_calls':2,'subset':'first8fit ground and first8fit aerial; no development sources',
        'inputs':inputs,'raw_decodes_main':16,'raw_decodes_independent_audit_max':16,'CUDA_allocation_limit_bytes':2*1024**3,
        'engineering_output_limit_bytes':8*1024**2,'real_training_authorized':False,'development_scoring_authorized':False,
        'remaining_full_run_inputs':1132,'remaining_full_run_encoder_calls':142,'successful_shards_immutable_and_reused':True,
        'six_confirmation_groups':'no access','full_run_requires_separate_user_authorization':True})
    put(ENG,'首forward输入源码绑定.json',{'task':'CV-013','sources':[entry(ROOT/'project/src'/s) for s in SOURCES],
        'inputs':[entry(PROTOCOL),entry(QUEUE),entry(DESIGN/'阶段封存.json'),entry(ENG/'核验/合成梯度与恢复.json'),entry(ENG/'角色快照.json')],
        'engineering_scope':entry(ENG/'工程范围冻结.json'),'models':protocol['encoder_assets'],
        'baselines':[protocol['baselines'][a]['weight'] for a in ['Identity','SwissRank8Frozen']],
        'protected':design['protected'],'role_reference_mutable':entry(role_path),
        'index_before':[hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest() for x in read(ROOT/'项目导航/实验索引.json')['batches']]})
    print(json.dumps({'prepared':True,'synthetic_checks':checks,'actual_training_steps':0,'encoder_inputs_planned':16}),flush=True)


def encode(out,kind,items,indices,env,dest=None):
    import torch
    model,ignored=load_encoder(kind,env['device']);before=tensor_sha(model)
    assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    state(out,kind+'_编码前',{'tensor_sha':before,'parameters':sum(p.numel() for p in model.parameters()),
        'requires_grad_count':sum(p.requires_grad for p in model.parameters()),'ignored_verified_buffers':ignored})
    if out==RUN:
        original=read(ENG/'核验'/f'{kind}_编码前.json')['tensor_sha'];assert original==before
    if env['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    receipts=[];values_list=[]
    for start in range(0,len(indices),8):
        selected=indices[start:start+8];data=[pixels(items[i],kind) for i in selected]
        arrays=[a for a,_ in data];receipts.extend(dict(r,index=i) for i,(_,r) in zip(selected,data))
        x=torch.from_numpy(np.stack(arrays)).to(env['device'])
        event(out,'encoder_attempt',kind=kind,indices=selected)
        with torch.inference_mode():v=model(pixel_values=x).image_embeds.detach().cpu().numpy()
        assert v.shape==(len(selected),512) and np.isfinite(v).all()
        if dest is None:
            values_list.extend(v.copy())
            # Engineering successes are saved before post-forward checks, never re-encoded.
            np.save(out/'特征'/f'{kind}_first8.npy',np.asarray(values_list,dtype=np.float32))
        else:dest[selected]=v;dest.flush()
        event(out,'encoder_saved',kind=kind,indices=selected)
        peak=int(torch.cuda.max_memory_allocated()) if env['device'].startswith('cuda') else 0
        assert peak<=2*1024**3
        del x,v,arrays,data
    after=tensor_sha(model);assert before==after and all(p.grad is None and not p.requires_grad for p in model.parameters())
    state(out,kind+'_编码后',{'tensor_sha':after,'before_sha':before,'all_frozen':True,'no_gradients':True,
        'encoder_inputs':len(indices),'encoder_calls':(len(indices)+7)//8,'peak_CUDA_allocated_bytes':peak})
    put(out,'核验/'+kind+'_像素回执.json',{'receipts':receipts})
    del model
    if env['device'].startswith('cuda'):torch.cuda.empty_cache()
    print(kind+': saved '+str(len(indices))+' inputs; frozen tensor checks passed',flush=True)


def engineer():
    import torch
    from agents.weak_alignment_head_v1 import WeakAlignmentHead
    from agents.capturepoint_target_head_v1 import TargetHead
    from data.mmgag_views_v1 import normalize_features
    assert not any((ENG/n).exists() for n in ['事件.jsonl','工程中断.json','工程结果.json']),'Consumed/partial engineering; inspect saved shards before recovery'
    b=read(ENG/'首forward输入源码绑定.json')
    for k in ['sources','inputs','models','baselines','protected']:verify(b[k])
    verify([b['engineering_scope'],b['role_reference_mutable']])
    env=environment();put(ENG,'运行环境.json',env);q=read(QUEUE)
    try:
        encode(ENG,'ground',q['ground_encoder_inputs'],list(range(8)),env)
        encode(ENG,'aerial',q['aerial_encoder_inputs'],list(range(8)),env)
        g=normalize_features(np.load(ENG/'特征/ground_first8.npy',allow_pickle=False))
        initial=WeakAlignmentHead(seed=0);np.savez(ENG/'初始头.npz',**{k:v.detach().numpy() for k,v in initial.state_dict().items()})
        outputs={};errors={};p=read(PROTOCOL)
        for arm in ['Identity','SwissRank8Frozen']:
            weights={k:v.copy() for k,v in np.load(ROOT/p['baselines'][arm]['weight']['path'],allow_pickle=False).items()}
            head=WeakAlignmentHead().double().eval().requires_grad_(False)
            head.load_state_dict({k:torch.from_numpy(v).double() for k,v in weights.items()},strict=True)
            with torch.inference_mode():actual=head(torch.from_numpy(g)).numpy()
            reference=np.stack([TargetHead(weights)(v) for v in g]);err=float(np.abs(actual-reference).max())
            assert err<=1e-12 and np.allclose(np.linalg.norm(actual,axis=1),1,atol=1e-12)
            if arm=='Identity':assert np.abs(actual-g).max()<=1e-6
            outputs[arm]=actual;errors[arm]=err
        np.savez(ENG/'头算子工程输出.npz',**outputs)
        result={'task':'CV-013','engineering_passed':True,'encoder_inputs':16,'encoder_forward_calls':2,'real_fit_target_queries':8,
            'head_operator_forward_batches':2,'head_operator_max_abs_errors':errors,'real_training_steps':0,
            'development_rankings':0,'navigation_episodes':0,'API_calls':0,'confirmation_pixels':0,
            'remaining_encoder_inputs':1132,'remaining_encoder_calls':142,'full_execution_authorized':False,
            'next':'read-only audit, close engineering; reuse immutable shards in a separately authorized full run'}
        put(ENG,'工程结果.json',result);assert sum(p.stat().st_size for p in ENG.rglob('*') if p.is_file())<=8*1024**2
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        put(ENG,'工程中断.json',{'error':type(exc).__name__,'message':str(exc),'no_auto_retry':True});raise


def replace_recovery(source,target):
    for retry in range(20):
        try:os.replace(source,target);return
        except PermissionError:
            if retry==19:raise
            time.sleep(.1)  # Only retry file metadata, never optimizer or encoder calls.


def execute(authorization):
    import torch
    from agents.weak_alignment_head_v1 import WeakAlignmentHead
    from train.weak_scene_alignment_v1 import optimizer_for,update,checkpoint,loss_for
    from data.mmgag_views_v1 import normalize_features
    from eval.capturepoint_execution_scoring_v1 import rank_all,summarize
    assert authorization is not None,'A separate explicit full-run authorization file is required'
    auth=read(authorization)
    assert auth['explicit_user_authorized_full_run'] is True and auth['protocol_sha256']==sha(PROTOCOL)
    assert auth['maximum_updates']==500 and auth['maximum_total_encoder_inputs']==1148 and auth['confirmation_access']==False
    assert not RUN.exists(),'Consumed/partial full execution; never blind rerun'
    seal=read(ENG/'阶段封存.json');verify(seal['files']+seal['sources'])
    b=read(ENG/'首forward输入源码绑定.json')
    for k in ['sources','inputs','models','baselines','protected']:verify(b[k])
    assert role_signature(read(ROOT/b['role_reference_mutable']['path']))==role_signature(read(ENG/'角色快照.json')),'Source roles changed; new admission required'
    q=read(QUEUE);p=read(PROTOCOL)
    RUN.mkdir();(RUN/'特征').mkdir();(RUN/'权重').mkdir();(RUN/'核验').mkdir()
    put(RUN,'运行授权.json',auth)
    put(RUN,'首forward绑定.json',{'engineering_seal':entry(ENG/'阶段封存.json'),'sources':b['sources'],
        'protocol':entry(PROTOCOL),'queue':entry(QUEUE),'authorization':entry(RUN/'运行授权.json'),
        'engineering_shards':[entry(ENG/'特征'/f'{kind}_first8.npy') for kind in ['ground','aerial']],
        'models':b['models'],'baselines':b['baselines'],'protected':b['protected']})
    env=environment();put(RUN,'运行环境.json',env)
    try:
        for kind,total in [('ground',14),('aerial',1134)]:
            dest=np.lib.format.open_memmap(RUN/'特征'/f'{kind}.npy',mode='w+',dtype=np.float32,shape=(total,512))
            first=np.load(ENG/'特征'/f'{kind}_first8.npy',allow_pickle=False);assert first.shape==(8,512)
            dest[:8]=first;dest.flush()
            encode(RUN,kind,q[kind+'_encoder_inputs'],list(range(8,total)),env,dest)
            del dest
        features=[entry(RUN/'特征'/f'{k}.npy') for k in ['ground','aerial']]
        assert sum(e['bytes'] for e in features)<=32*1024**2
        g=normalize_features(np.load(RUN/'特征/ground.npy',allow_pickle=False));a=normalize_features(np.load(RUN/'特征/aerial.npy',allow_pickle=False))
        queries=q['training_queries'];qi=[x['ground_feature_index'] for x in queries];ci=[x['aerial_feature_indices'] for x in queries]
        positive=[x['positive_candidate_index_evaluator_only'] for x in queries];assert qi==list(range(9)) and positive==[40]*9
        gg=torch.from_numpy(g[qi].astype(np.float32));aa=torch.from_numpy(a[np.array(ci)].astype(np.float32));pp=torch.tensor(positive)
        head=WeakAlignmentHead(seed=0);initial={k:v.detach().numpy().copy() for k,v in head.state_dict().items()}
        eng_initial=np.load(ENG/'初始头.npz',allow_pickle=False);assert all(np.array_equal(initial[k],eng_initial[k]) for k in initial)
        np.savez(RUN/'权重/initial.npz',**initial)
        put(RUN,'首更新前特征绑定.json',{'features':features,'initial_head':entry(RUN/'权重/initial.npz'),'protocol':entry(PROTOCOL),
            'fit_queries':qi,'fit_candidate_indices':ci,'positive_indices_evaluator_only':positive,
            'ground_tensor_sha256':hashlib.sha256(gg.numpy().tobytes()).hexdigest(),'aerial_tensor_sha256':hashlib.sha256(aa.numpy().tobytes()).hexdigest(),
            'positive_tensor_sha256':hashlib.sha256(pp.numpy().tobytes()).hexdigest(),'updates':500,'device':'cpu','real_training_authorized':True})
        opt=optimizer_for(head);logs=[];start=time.monotonic()
        for step in range(1,501):
            assert time.monotonic()-start<=1800
            event(RUN,'optimizer_attempt',step=step)
            loss,norm=update(head,opt,gg,aa,pp)
            torch.save(checkpoint(head,opt,step),RUN/'权重/recovery.tmp')
            replace_recovery(RUN/'权重/recovery.tmp',RUN/'权重/recovery.pt')
            event(RUN,'optimizer_saved',step=step,loss=loss,gradient_norm_before_clip=norm)
            logs.append({'step':step,'loss_before_update':loss,'gradient_norm_before_clip':norm})
            if step%100==0:print(f'train {step}/500 loss={loss:.6f}',flush=True)
        final_loss=float(loss_for(head,gg,aa,pp).detach());np.savez(RUN/'权重/final.npz',**{k:v.detach().numpy() for k,v in head.state_dict().items()})
        put(RUN,'训练日志.json',{'updates':500,'seed':0,'unique_queries':9,'supervision_uses':4500,'seconds':time.monotonic()-start,
            'log':logs,'final_fit_loss':final_loss,'trainable_parameters':8192,'encoder_optimizer_parameters':0})
        states={arm:{k:v.copy() for k,v in np.load(ROOT/p['baselines'][arm]['weight']['path'],allow_pickle=False).items()} for arm in ['Identity','SwissRank8Frozen']}
        states['MMRank8Final']={k:v.copy() for k,v in np.load(RUN/'权重/final.npz',allow_pickle=False).items()}
        rows=rank_all(q,np.load(RUN/'特征/ground.npy',allow_pickle=False),np.load(RUN/'特征/aerial.npy',allow_pickle=False),states)
        put(RUN,'逐题排序结果.json',{'rows':rows});put(RUN,'开发对照汇总.json',summarize(rows))
        put(RUN,'执行完成.json',{'completed':True,'real_training_steps':500,'rankings':1250,'new_encoder_inputs':1132,
            'new_encoder_calls':142,'aggregate_encoder_inputs':1148,'aggregate_encoder_calls':144,'reused_engineering_inputs':16,
            'confirmation_pixels':0,'navigation_episodes':0,'API_calls':0,'next':'independent audit before gate decision'})
        assert sum(p.stat().st_size for p in (RUN/'权重').iterdir())<=1024**2
        assert sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())<=40*1024**2
        print(json.dumps({'complete':True,'updates':500,'rankings':1250,'aggregate_encoder_calls':144}),flush=True)
    except BaseException as exc:
        put(RUN,'执行中断.json',{'error':type(exc).__name__,'message':str(exc),'no_auto_retry':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','engineer','execute']);p.add_argument('--authorization')
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='engineer':engineer()
    else:execute(a.authorization)
