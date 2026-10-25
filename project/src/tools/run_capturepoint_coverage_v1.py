"""CV-019 one authorized12-pair execution; immutable old caches/results are referenced."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from tools.run_capturepoint_execution_v2 import load_encoder,tensor_sha,pixels,environment,replace_recovery
from train.capturepoint_coverage_runtime_v1 import reuse_plan,assemble,fit_tensors,synthetic_checks
from data.capturepoint_coverage_protocol_v1 import sample_indices

BASE=ROOT/'DATA/processed_data/MM-GAG';DESIGN=BASE/'三新增来源工程与配对覆盖协议_v1';OLD=BASE/'同任务拍摄点对齐执行_v1';OUT=BASE/'配对覆盖同预算执行_v1'
SOURCES=['train/capturepoint_coverage_runtime_v1.py','eval/capturepoint_coverage_scoring_v1.py','tools/run_capturepoint_coverage_v1.py',
    'eval/audit_capturepoint_coverage_execution_v1.py','tests/test_capturepoint_coverage_execution_v1.py','documents/close_capturepoint_coverage_execution_v1.py']
DEPS=['tools/run_capturepoint_execution_v2.py','agents/weak_alignment_head_v1.py','train/weak_scene_alignment_v1.py',
    'agents/capturepoint_target_head_v1.py','agents/crossview_local_match_v1.py','data/mmgag_views_v1.py','data/capturepoint_coverage_protocol_v1.py']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()


def entry(p):
    p=Path(p);return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}


def verify(items):
    for e in items:
        if sha(ROOT/e['path'])!=e['sha256']:raise ValueError(('Bound file changed',e['path']))


def put(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def event(event_type,**values):
    with (OUT/'事件.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'event':event_type,**values},ensure_ascii=False)+'\n');f.flush()


def check_bind():
    b=read(OUT/'首forward绑定.json')
    for key in ('sources','dependencies','inputs','models','protected','old_features','old_heads'):verify(b[key])
    verify([b['authorization'],b['reuse_plan']]);return b


def features():
    plan=read(OUT/'特征复用计划.json')
    return assemble(np.load(OLD/'特征/ground.npy',allow_pickle=False),np.load(OLD/'特征/aerial.npy',allow_pickle=False),
        np.load(OUT/'特征/new_ground.npy',allow_pickle=False),np.load(OUT/'特征/new_aerial.npy',allow_pickle=False),plan)


def prepare():
    import torch
    if OUT.exists():raise FileExistsError('Existing/partial run, never blindly repeat or overwrite')
    seal=read(DESIGN/'阶段封存.json');verify(seal['files']+seal['sources']+seal['readonly_math_dependencies'])
    oldseal=read(OLD/'阶段封存.json');verify(oldseal['files']+oldseal['sources'])
    p=read(DESIGN/'配对覆盖同预算冻结协议.json');q=read(DESIGN/'固定训练与开发队列.json')
    if not p['research_design_frozen'] or not p['source_engineering_passed']:raise ValueError('CV-018 must pass and freeze')
    prior=read(OLD/'首forward绑定.json');verify(prior['models']+prior['protected'])
    old_features=read(OLD/'首更新前特征绑定.json')['features'];verify(old_features)
    plan=reuse_plan(q)
    if plan['old_aerial_rows_reused']!=35 or len(plan['new_aerial_items'])!=208 or plan['new_unique_encoder_inputs']!=211 or plan['new_encoder_calls']!=27:
        raise ValueError('Frozen raw SHA reuse topology changed')
    OUT.mkdir();(OUT/'特征').mkdir();(OUT/'权重').mkdir();(OUT/'核验').mkdir()
    put('运行授权.json',{'task':'CV-019','date':'2026-10-18','explicit_user_authorized_single_comparison':True,
        'user_request':'开始：实现并验证12配对执行器，再运行固定对照',
        'protocol_sha256':sha(DESIGN/'配对覆盖同预算冻结协议.json'),'maximum_updates':500,'maximum_query_uses':4500,
        'maximum_new_encoder_inputs':246,'maximum_new_encoder_calls':32,'confirmation_access':False,'default_upgrade_authorized':False})
    put('特征复用计划.json',plan)
    paths=[DESIGN/'阶段封存.json',DESIGN/'配对覆盖同预算冻结协议.json',DESIGN/'固定训练与开发队列.json',
        BASE/'同任务来源用途账本_v1.json',OLD/'阶段封存.json',OLD/'逐题排序结果.json',OLD/'首更新前特征绑定.json',OLD/'开发对照汇总.json',
        OLD/'核验/ground_编码后.json',OLD/'核验/aerial_编码后.json']
    put('首forward绑定.json',{'task':'CV-019','sources':[entry(ROOT/'project/src'/x) for x in SOURCES],
        'dependencies':[entry(ROOT/'project/src'/x) for x in DEPS],'inputs':[entry(x) for x in paths],
        'models':prior['models'],'protected':prior['protected'][:4],'old_features':old_features,
        'old_heads':[entry(OLD/'权重/initial.npz'),entry(OLD/'权重/final.npz')]+prior['baselines'],
        'authorization':entry(OUT/'运行授权.json'),'reuse_plan':entry(OUT/'特征复用计划.json'),
        'scope':'only3 new ground and208 unique aerial inputs, raw SHA verified; old35 vectors reused with identical encoder tensors/preprocessing; no confirmation/navigation/API'})
    torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    checks=synthetic_checks();put('核验/合成梯度与恢复.json',checks)
    print(json.dumps({'prepared':True,'new_unique_encoder_inputs':211,'encoder_calls':27,'synthetic_checks_passed':checks['passed'],'real_updates':0},ensure_ascii=False),flush=True)


def encode(kind,items,env):
    import torch
    model,ignored=load_encoder(kind,env['device']);before=tensor_sha(model)
    old=read(OLD/'核验'/f'{kind}_编码后.json')['tensor_sha']
    if before!=old or model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Encoder identity/freeze mismatch; cannot reuse old35')
    put('核验/'+kind+'_编码前.json',{'tensor_sha':before,'old_tensor_sha':old,'parameters':sum(p.numel() for p in model.parameters()),'requires_grad_count':0,'ignored_buffers':ignored})
    dest=np.lib.format.open_memmap(OUT/'特征'/f'new_{kind}.npy',mode='w+',dtype=np.float32,shape=(len(items),512));dest[:]=np.nan;dest.flush()
    if env['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    receipts=[];peak=0
    for start in range(0,len(items),8):
        arrays=[]
        for i in range(start,min(start+8,len(items))):
            event('pixel_attempt',kind=kind,index=i,path=items[i]['path'])
            value,receipt=pixels(items[i],kind);arrays.append(value);receipts.append(dict(receipt,index=i))
        indices=list(range(start,start+len(arrays)));x=torch.from_numpy(np.stack(arrays)).to(env['device'])
        event('encoder_attempt',kind=kind,indices=indices)
        with torch.inference_mode():value=model(pixel_values=x).image_embeds.detach().cpu().numpy()
        if value.shape!=(len(indices),512) or not np.isfinite(value).all():raise ValueError('Invalid encoder feature')
        dest[indices]=value;dest.flush();event('encoder_saved',kind=kind,indices=indices)
        if env['device'].startswith('cuda'):peak=int(torch.cuda.max_memory_allocated())
        if peak>2*1024**3:raise ValueError('CUDA cap exceeded')
        del x,value,arrays
    after=tensor_sha(model)
    if after!=before or any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('Encoder changed')
    put('核验/'+kind+'_编码后.json',{'tensor_sha':after,'before_sha':before,'old_tensor_sha':old,'all_frozen':True,'no_gradients':True,
        'encoder_inputs':len(items),'encoder_calls':(len(items)+7)//8,'peak_CUDA_allocated_bytes':peak})
    put('核验/'+kind+'_像素回执.json',{'receipts':receipts});del model,dest
    if env['device'].startswith('cuda'):torch.cuda.empty_cache()
    print(kind+': saved'+str(len(items))+' inputs; original encoder tensors unchanged',flush=True)


def engineer():
    import torch
    from agents.weak_alignment_head_v1 import WeakAlignmentHead
    from agents.capturepoint_target_head_v1 import TargetHead
    from data.mmgag_views_v1 import normalize_features
    if (OUT/'事件.jsonl').exists() or (OUT/'工程结果.json').exists():raise FileExistsError('Engineer already attempted; inspect cache coverage, no retry')
    check_bind();env=environment();put('编码环境.json',env);plan=read(OUT/'特征复用计划.json')
    try:
        encode('ground',plan['new_ground_items'],env);encode('aerial',plan['new_aerial_items'],env)
        q=read(DESIGN/'固定训练与开发队列.json');g,a=features();gg,aa,pp,fit=fit_tensors(g,a,q)
        head=WeakAlignmentHead(seed=0);initial={k:v.detach().numpy().copy() for k,v in head.state_dict().items()}
        old_initial=np.load(OLD/'权重/initial.npz',allow_pickle=False)
        if not all(np.array_equal(initial[k],old_initial[k]) for k in initial):raise ValueError('Seed0 initialization differs from9-pair baseline')
        np.savez(OUT/'权重/initial.npz',**initial)
        targets=normalize_features(g[14:]);errors={};outputs={}
        oldp=read(BASE/'同任务对齐训练与目标对照协议_v1/同任务对齐训练冻结协议.json')
        for arm in ('Identity','SwissRank8Frozen'):
            state={k:v.copy() for k,v in np.load(ROOT/oldp['baselines'][arm]['weight']['path'],allow_pickle=False).items()}
            h=WeakAlignmentHead().double().eval().requires_grad_(False);h.load_state_dict({k:torch.from_numpy(v).double() for k,v in state.items()},strict=True)
            with torch.inference_mode():actual=h(torch.from_numpy(targets)).numpy()
            reference=np.stack([TargetHead(state)(x) for x in targets]);error=float(np.abs(actual-reference).max())
            if error>1e-12:raise ValueError('Head operator mismatch')
            errors[arm]=error;outputs[arm]=actual
        np.savez(OUT/'核验/真实头算子输出.npz',**outputs)
        feature_entries=[entry(OUT/'特征/new_ground.npy'),entry(OUT/'特征/new_aerial.npy')]
        if sum(x['bytes'] for x in feature_entries)>2*1024**2:raise ValueError('Feature cap exceeded')
        put('工程结果.json',{'task':'CV-019','engineering_passed':True,'new_encoder_inputs':211,'new_encoder_calls':27,
            'old_aerial_rows_reused':35,'old_ground_rows_reused':14,'old_aerial_prefix_rows_reused':1134,
            'head_operator_errors':errors,'initial_head_identical_to_old9':True,'fit_indices':fit['ground_indices'],
            'new_features':feature_entries,'real_training_steps':0,'development_rankings':0,'confirmation_access':0})
        print(json.dumps({'engineering_passed':True,'new_encoder_inputs':211,'real_updates':0},ensure_ascii=False),flush=True)
    except BaseException as exc:put('工程中断.json',{'error':type(exc).__name__,'message':str(exc),'no_auto_retry':True});raise


def execute():
    import torch
    from agents.weak_alignment_head_v1 import WeakAlignmentHead
    from train.weak_scene_alignment_v1 import optimizer_for,update,checkpoint,loss_for
    from eval.capturepoint_coverage_scoring_v1 import rank_candidate,summarize
    if (OUT/'首更新前特征绑定.json').exists() or (OUT/'执行完成.json').exists():raise FileExistsError('Training attempted; never restart500 steps blindly')
    b=check_bind();auth=read(OUT/'运行授权.json')
    if not auth['explicit_user_authorized_single_comparison'] or auth['protocol_sha256']!=sha(DESIGN/'配对覆盖同预算冻结协议.json'):raise ValueError('Missing runtime authorization')
    if not read(OUT/'核验/工程另实现复核.json')['passed'] or not read(OUT/'工程结果.json')['engineering_passed']:raise ValueError('Real engineering/audit required')
    verify(read(OUT/'工程结果.json')['new_features']);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    q=read(DESIGN/'固定训练与开发队列.json');g,a=features();gg,aa,pp,bound=fit_tensors(g,a,q)
    gg=torch.from_numpy(gg);aa=torch.from_numpy(aa);pp=torch.from_numpy(pp);head=WeakAlignmentHead(seed=0)
    initial=np.load(OUT/'权重/initial.npz',allow_pickle=False)
    if not all(np.array_equal(v.detach().numpy(),initial[k]) for k,v in head.state_dict().items()):raise ValueError('Initial head mismatch')
    bound.update({'task':'CV-019','old_features':b['old_features'],'new_features':read(OUT/'工程结果.json')['new_features'],
        'initial_head':entry(OUT/'权重/initial.npz'),'protocol':entry(DESIGN/'配对覆盖同预算冻结协议.json'),
        'updates':500,'query_uses':4500,'device':'cpu','real_training_authorized':True})
    put('首更新前特征绑定.json',bound)
    with torch.no_grad():initial_loss=float(loss_for(head,gg,aa,pp))
    put('首批前拟合损失.json',{'initial_fit12_loss':initial_loss,'evaluation_only_no_update':True})
    opt=optimizer_for(head);logs=[];start=time.monotonic()
    try:
        for t in range(500):
            if time.monotonic()-start>1800:raise RuntimeError('Head CPU time cap')
            indices=sample_indices(12,t);event('optimizer_attempt',step=t+1,query_indices=indices)
            loss,norm=update(head,opt,gg[indices],aa[indices],pp[indices])
            torch.save(checkpoint(head,opt,t+1),OUT/'权重/recovery.tmp');replace_recovery(OUT/'权重/recovery.tmp',OUT/'权重/recovery.pt')
            event('optimizer_saved',step=t+1,query_indices=indices,loss=loss,gradient_norm_before_clip=norm)
            logs.append({'step':t+1,'query_indices':indices,'loss_before_update':loss,'gradient_norm_before_clip':norm})
            if (t+1)%100==0:print(f'train {t+1}/500 loss={loss:.6f}',flush=True)
        with torch.no_grad():final_loss=float(loss_for(head,gg,aa,pp))
        weights={k:v.detach().numpy().copy() for k,v in head.state_dict().items()};np.savez(OUT/'权重/final.npz',**weights)
        put('训练日志.json',{'task':'CV-019','updates':500,'seed':0,'unique_queries':12,'source_groups':9,'supervision_uses':4500,'uses_per_query':[375]*12,
            'seconds':time.monotonic()-start,'initial_fit12_loss':initial_loss,'final_fit12_loss':final_loss,
            'trainable_parameters':8192,'encoder_optimizer_parameters':0,'log':logs})
        new_rows=rank_candidate(q,g,a,weights);put('新增375排序.json',{'rows':new_rows})
        old_rows=read(OLD/'逐题排序结果.json')['rows'];summary=summarize(old_rows,new_rows)
        put('合并结果引用.json',{'planned_records':1625,'old_records':1250,'new_records':375,'old':entry(OLD/'逐题排序结果.json'),'new':entry(OUT/'新增375排序.json')})
        put('开发对照汇总.json',summary)
        put('执行完成.json',{'completed':True,'updates':500,'query_uses':4500,'new_encoder_inputs':211,'new_encoder_calls':27,'cached_old_aerial_matches':35,
            'new_rankings':375,'reused_rankings':1250,'planned_records':1625,'navigation_SR':None,'confirmation_access':0,'API_calls':0})
        if sum(p.stat().st_size for p in (OUT/'权重').iterdir())>1024**2 or sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>8*1024**2:raise RuntimeError('Output cap')
        print(json.dumps({'completed':True,'candidate_counts':summary['counts']['MMRank8Coverage12'],'fit_loss12':[initial_loss,final_loss],'ranking_records':1625},ensure_ascii=False),flush=True)
    except BaseException as exc:put('训练执行中断.json',{'error':type(exc).__name__,'message':str(exc),'completed_updates':len(logs),'no_auto_retry':True});raise


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('prepare','engineer','execute'):raise SystemExit('prepare | engineer | execute; each once')
    {'prepare':prepare,'engineer':engineer,'execute':execute}[sys.argv[1]]()
