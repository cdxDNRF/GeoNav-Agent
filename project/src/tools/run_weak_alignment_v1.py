"""CV-008: one bound extraction and one final-only500-update pilot. No blind retry."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'project/src'))
import numpy as np

BASE = ROOT/'DATA/processed_data/MM-GAG'
PREV = BASE/'配对派生清单与弱目标准入_v1'
MM = BASE/'跨视角感知最小验证_v1'
OUT = BASE/'弱场景对齐头工程与训练_v1'
SOURCES = ['agents/weak_alignment_head_v1.py', 'train/weak_scene_alignment_v1.py',
           'tools/run_weak_alignment_v1.py', 'eval/weak_alignment_eval_v1.py',
           'eval/audit_weak_alignment_v1.py', 'tests/test_weak_alignment_v1.py',
           'documents/close_weak_alignment_v1.py', 'data/mmgag_views_v1.py',
           'agents/crossview_local_match_v1.py', 'data/weak_scene_pairs_v1.py']


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def entry(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


def put(name, data):
    with (OUT/name).open('x', encoding='utf-8') as f:
        f.write(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2))


def progress(state):
    tmp = OUT/'状态.tmp'
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, OUT/'执行状态.json')


def verify(entries):
    for e in entries:
        assert sha(ROOT/e['path']) == e['sha256'], e['path']
    return len(entries)


def old_seals():
    previous = read(PREV/'输入与源码绑定.json')
    return [*previous['previous_seals'], entry(PREV/'阶段封存.json')]


def old_replay():
    from data.mmgag_views_v1 import normalize_features
    from agents.crossview_local_match_v1 import rank_cells
    g = normalize_features(np.load(MM/'地面特征.npy', allow_pickle=False))
    a = normalize_features(np.load(MM/'航拍特征.npy', allow_pickle=False).reshape(16,25,512))
    pairs = read(MM/'冻结执行任务.json')['pairs']; lookup = {p['pair_id']: i for i,p in enumerate(pairs)}
    ref = read(MM/'逐对排序结果.json')['rows']; maximum = 0.
    for row in ref:
        i = lookup[row['pair_id']]; p = pairs[i]
        target = {'Full': g[i], 'Mean': normalize_features(g.mean(0)), 'Wrong': g[lookup[p['wrong_target_pair_id']]]}[row['condition']]
        scores = a[i] @ target
        maximum = max(maximum, float(np.abs(scores-np.array(row['scores'])).max()))
        assert rank_cells(scores,p['pair_id']) == row['ranking']
    assert maximum <= 1e-6
    return {'passed': True, 'rankings': len(ref), 'maximum_score_error': maximum,
            'MMGAG_new_encoder_inputs': 0}


def freeze():
    import torch
    from train.weak_scene_alignment_v1 import engineering_checks
    from eval.weak_alignment_eval_v1 import eval_tasks
    assert not (OUT/'首forward前绑定.json').exists(), 'Already frozen; inspect state, never overwrite'
    torch.set_num_threads(min(8,os.cpu_count() or 1)); torch.use_deterministic_algorithms(True)
    engineering = engineering_checks(); replay = old_replay()
    design = read(PREV/'训练设计冻结.json')
    assert design['training']['updates'] == 500 and design['head']['trainable_parameters'] == 8192
    assert design['training']['seed'] == 0 and design['training']['source_groups'] == read(PREV/'来源分组与用途.json')['fit']
    seals = old_seals(); verify(seals)
    for s in seals: verify(read(ROOT/s['path'])['files'] + read(ROOT/s['path'])['sources'])
    bindings = read(PREV/'输入与源码绑定.json')
    raw_reference = read(ROOT/bindings['raw_reference_manifest']['path'])
    verify(raw_reference['new_raw_inputs'])
    bank = read(PREV/'唯一训练查询与图块引用.json')
    tasks = eval_tasks(bank,read(PREV/'派生素材清单.json')['records'],read(MM/'冻结执行任务.json')['pairs'], design['evaluation']['weak_dev_groups'])
    assert len(bank['queries']) == 14 and len(bank['cells']) == 1134
    for name in ['特征','权重','核验']: (OUT/name).mkdir(parents=True,exist_ok=True)
    put('核验/合成工程.json',engineering); put('核验/旧缓存复现.json',replay)
    put('冻结评测任务.json',{'tasks':tasks, 'weak_pair_id_rule':'weak_ + original_id; CV-002/tie/v1 unchanged'})
    files = [entry(PREV/'训练设计冻结.json'), *design['data_bindings'], *design['cached_MMGAG_features'],
             entry(MM/'逐对排序结果.json'), entry(OUT/'冻结评测任务.json'), entry(OUT/'核验/合成工程.json'), entry(OUT/'核验/旧缓存复现.json')]
    sources = [entry(ROOT/'project/src'/p) for p in SOURCES]
    verify(files + sources + design['frozen_encoder_assets'] + bindings['protected'])
    put('首forward前绑定.json',{'task':'CV-008','date':'2026-10-17','files':files,'sources':sources,
        'models':design['frozen_encoder_assets'],'protected':bindings['protected'],'previous_seals':seals,
        'raw_reference':bindings['raw_reference_manifest'],'raw_inputs':raw_reference['new_raw_inputs'],
        'resources':design['resources'],'synthetic_engineering_passed':True, 'cached_identity_replay_passed':True,
        'user_authorized_one_fixed_run':True, 'training_admitted':False,
        'retention':'initial/final heads + one active recovery; single feature caches; <=40MiB; no raw/model copies',
        'indices_before':[hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest() for x in read(ROOT/'项目导航/实验索引.json')['batches']]})
    print(json.dumps({'frozen':True,'sources':len(sources),'synthetic':engineering,'replay':replay}),flush=True)


def model_tensor_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_encoder(kind, device):
    import torch
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    if kind == 'ground':
        cfg = read(ROOT/'models/clip_vit_b32_public_v1/config.json')['vision_config']; cfg['projection_dim'] = 512
        raw = torch.load(ROOT/'models/clip_vit_b32_public_v1/pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
        selected = {k:v for k,v in raw.items() if k.startswith(('vision_model.','visual_projection.'))}
    else:
        from safetensors.torch import load_file
        cfg = read(ROOT/'models/Sat2Cap/config.json'); selected = load_file(str(ROOT/'models/Sat2Cap/model.safetensors'),device='cpu')
    config = CLIPVisionConfig(**cfg); config._attn_implementation = 'eager'
    model = CLIPVisionModelWithProjection(config)
    ignored = []
    for k in list(selected):
        if k.endswith('.position_ids') and k not in model.state_dict():
            assert selected[k].flatten().tolist() == list(range(50)); ignored.append(k); selected.pop(k)
    model.load_state_dict(selected,strict=True)
    model.requires_grad_(False); model.eval(); model.to(device)
    return model, ignored


def pixel_array(item, kind, processor):
    from PIL import Image
    from data.mmgag_views_v1 import clean_rgb,ground_pixels,aerial_pixels
    from data.weak_scene_pairs_v1 import raw_path
    if kind == 'ground':
        p = raw_path(ROOT,item['ground_raw_path']); assert sha(p) == item['ground_raw_sha256']
        rgb = clean_rgb(p,ground=True)
        h = hashlib.sha256();h.update(f'{rgb.shape[1]}x{rgb.shape[0]}:RGB'.encode());h.update(rgb.tobytes())
        assert h.hexdigest() == item['ground_pixel_sha256']
        return ground_pixels(rgb,processor)
    p = raw_path(ROOT,item['raw_aerial_path']); assert sha(p) == item['raw_aerial_sha256']
    r,c = divmod(item['crop_index'],5)
    assert item['crop_box'] == [c*300,r*300,(c+1)*300,(r+1)*300]
    with Image.open(p) as im:
        assert im.size == (1500,1500)
        patch = im.convert('RGB').crop(tuple(item['crop_box']))
    h = hashlib.sha256();h.update(b'300x300:RGB');h.update(patch.tobytes())
    assert h.hexdigest() == item['cell_sha256']
    return aerial_pixels(np.asarray(patch))


def extract(state, bank):
    import torch
    from agents.weak_alignment_head_v1 import WeakAlignmentHead
    from data.mmgag_views_v1 import normalize_features
    processor = read(ROOT/'models/clip_vit_b32_public_v1/preprocessor_config.json')
    evidence = []
    for kind, items in [('ground',bank['queries']),('aerial',bank['cells'])]:
        model, ignored = load_encoder(kind,state['device'])
        before = model_tensor_sha(model); assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
        dest = np.lib.format.open_memmap(OUT/'特征'/f'{kind}.npy',mode='w+',dtype=np.float32,shape=(len(items),512))
        if state['device'].startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        receipts = []; first_gate = None; start = time.monotonic()
        for offset in range(0,len(items),8):
            arrays = [pixel_array(e,kind,processor) for e in items[offset:offset+8]]
            pixels = torch.from_numpy(np.stack(arrays)).to(state['device'])
            state['attempted_encoder_inputs'] += len(arrays); state['attempted_encoder_calls'] += 1
            assert state['attempted_encoder_inputs'] <= 1148 and state['attempted_encoder_calls'] <= 144
            state['pending_forward'] = {'kind':kind,'offset':offset,'count':len(arrays)}; progress(state)
            with torch.inference_mode(): values = model(pixel_values=pixels).image_embeds.detach().cpu().numpy()
            assert values.shape == (len(arrays),512) and np.isfinite(values).all()
            dest[offset:offset+len(arrays)] = values; dest.flush()
            for j, arr in enumerate(arrays):
                receipts.append({'index':offset+j,'pixel_tensor_sha256':hashlib.sha256(arr.tobytes()).hexdigest(),
                    'feature_sha256':hashlib.sha256(values[j].tobytes()).hexdigest()})
            state[kind+'_completed'] = offset+len(arrays); state['successful_encoder_calls'] += 1
            state['forward_ledger'].append({**state['pending_forward'],'status':'saved'})
            state['pending_forward'] = None
            peak = torch.cuda.max_memory_allocated() if state['device'].startswith('cuda') else 0
            assert peak <= 2*1024**3
            if offset == 0:
                after = model_tensor_sha(model)
                assert after == before and all(p.grad is None and not p.requires_grad for p in model.parameters())
                normalized = normalize_features(values).astype(np.float32)
                head = WeakAlignmentHead()
                with torch.inference_mode(): error = float((head(torch.from_numpy(normalized))-torch.from_numpy(normalized)).abs().max())
                assert error <= 1e-6
                first_gate = {'images':len(arrays),'calls':1,'identity_max_abs':error,'tensor_sha_unchanged':True,
                              'all_parameters_frozen':True,'no_encoder_gradients':True,'successful_features_reused':True}
            progress(state)
            del pixels,values,arrays
        after = model_tensor_sha(model)
        assert before == after and all(p.grad is None and not p.requires_grad for p in model.parameters())
        evidence.append({'kind':kind,'inputs':len(items),'parameters':sum(p.numel() for p in model.parameters()),
            'tensor_sha_before':before,'tensor_sha_after':after,'no_gradients':True,'all_frozen':True,
            'engineering_first_batch':first_gate,'ignored_verified_position_buffers':ignored,'seconds':time.monotonic()-start,
            'peak_cuda_allocated_bytes':peak,'receipts':receipts})
        del model,dest
        if state['device'].startswith('cuda'): torch.cuda.empty_cache()
        print(f'{kind}: {len(items)} inputs saved; frozen encoder verified',flush=True)
    assert state['attempted_encoder_inputs'] == 1148 and state['successful_encoder_calls'] == 144
    put('编码器冻结与像素证据.json',{'encoders':evidence,'engineering_passed':True,'MMGAG_new_encoder_inputs':0,'reserved_groups_consumed':0})


def features():
    from data.mmgag_views_v1 import normalize_features
    return {'weak_dev':(normalize_features(np.load(OUT/'特征/ground.npy',allow_pickle=False)),normalize_features(np.load(OUT/'特征/aerial.npy',allow_pickle=False))),
            'MMGAG':(normalize_features(np.load(MM/'地面特征.npy',allow_pickle=False)),normalize_features(np.load(MM/'航拍特征.npy',allow_pickle=False)))}


def train(state, bank, design):
    import torch
    from agents.weak_alignment_head_v1 import WeakAlignmentHead
    from train.weak_scene_alignment_v1 import optimizer_for,update,checkpoint,loss_for
    from data.mmgag_views_v1 import normalize_features
    g = normalize_features(np.load(OUT/'特征/ground.npy',allow_pickle=False))
    a = normalize_features(np.load(OUT/'特征/aerial.npy',allow_pickle=False))
    qi = {q['scene']:i for i,q in enumerate(bank['queries'])}
    fit = design['training']['source_groups']
    indices = [qi[s] for s in fit]
    candidates = [[i for i,c in enumerate(bank['cells']) if c['scene']==s and c['role']=='fit'] for s in fit]
    positive = [next(j for j,i in enumerate(ids) if bank['cells'][i]['cell_sha256']==bank['queries'][qi[s]]['positive_cell_sha256']) for s,ids in zip(fit,candidates)]
    assert len(indices) == 9 and all(bank['queries'][i]['role']=='fit' for i in indices) and all(len(ids)==81 for ids in candidates)
    head = WeakAlignmentHead(seed=0)
    gg = torch.from_numpy(g[indices].astype(np.float32)); aa = torch.from_numpy(a[np.array(candidates)].astype(np.float32)); pp = torch.tensor(positive)
    with torch.inference_mode(): identity_error = float((head(gg)-gg).abs().max())
    assert identity_error <= 1e-6
    np.savez(OUT/'权重/initial.npz',**{k:v.detach().numpy() for k,v in head.state_dict().items()})
    bound = [entry(OUT/'特征'/f'{k}.npy') for k in ['ground','aerial']]
    assert sum(e['bytes'] for e in bound) <= 32*1024**2
    put('首更新前特征与训练绑定.json',{'task':'CV-008','execution_admitted':True,'features':bound,
        'initial_head':entry(OUT/'权重/initial.npz'),'source_binding':entry(OUT/'首forward前绑定.json'),
        'engineering':[entry(OUT/n) for n in ['编码器冻结与像素证据.json','核验/合成工程.json','核验/旧缓存复现.json']],
        'fit_queries':indices,'fit_candidate_indices':candidates,'positive_indices_evaluator_only':positive,
        'ground_tensor_sha256':hashlib.sha256(gg.numpy().tobytes()).hexdigest(),'aerial_tensor_sha256':hashlib.sha256(aa.numpy().tobytes()).hexdigest(),
        'positive_tensor_sha256':hashlib.sha256(pp.numpy().tobytes()).hexdigest(),'device':'cpu','updates':500,'training_sources':fit,
        'no_dev_labels_in_training':True,'identity_max_abs':identity_error})
    state['phase'] = 'training'; state['training_admitted'] = True; progress(state)
    opt = optimizer_for(head); log = []; start = time.monotonic()
    for step in range(1,501):
        assert time.monotonic()-start <= 30*60
        loss, norm = update(head,opt,gg,aa,pp)
        log.append({'step':step,'loss_before_update':loss,'gradient_norm_before_clip':norm,'supervision_uses':step*9})
        state['training_steps'] = step
        if step % 100 == 0:
            torch.save(checkpoint(head,opt,step),OUT/'权重/recovery.tmp')
            os.replace(OUT/'权重/recovery.tmp',OUT/'权重/recovery.pt')
            progress(state); print(f'train {step}/500, loss={loss:.6f}',flush=True)
    final_loss = float(loss_for(head,gg,aa,pp).detach())
    np.savez(OUT/'权重/final.npz',**{k:v.detach().numpy() for k,v in head.state_dict().items()})
    assert sum(p.stat().st_size for p in (OUT/'权重').iterdir()) <= 1024**2
    put('训练日志.json',{'seed':0,'updates':500,'unique_fit_queries':9,'supervision_uses':4500,'device':'cpu',
        'seconds':time.monotonic()-start,'log':log,'final_fit_loss':final_loss,
        'new_trainable_parameters':8192,'encoder_optimizer_parameters':0,'checkpoint_selection':'final500_only'})
    return indices


def run():
    import torch
    import transformers
    from eval.weak_alignment_eval_v1 import rankings,summarize
    assert not (OUT/'执行状态.json').exists(), 'Consumed/partial run; inspect ledger, never retry automatically'
    binding = read(OUT/'首forward前绑定.json')
    verify(binding['files']+binding['sources']+binding['models']+binding['protected']+binding['previous_seals']+binding['raw_inputs'])
    torch.manual_seed(20261016);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    free = int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else None
    state = {'task':'CV-008','state':'running','phase':'extraction','pid':os.getpid(),
             'device':'cuda:0' if free is not None and free>=2560*1024**2 else 'cpu','initial_cuda_free_bytes':free,
             'torch':torch.__version__,'transformers':transformers.__version__,
             'attempted_encoder_inputs':0,'attempted_encoder_calls':0,'successful_encoder_calls':0,'forward_ledger':[],
             'training_steps':0,'external_api_calls':0,'navigation_episodes':0,'reserved_groups_consumed':0,
             'recovery':'Inspect this ledger and existing feature coverage; no blind rerun. Final recovery has AdamW state.'}
    progress(state); start = time.monotonic()
    try:
        bank = read(PREV/'唯一训练查询与图块引用.json');design = read(PREV/'训练设计冻结.json')
        extract(state,bank)
        indices = train(state,bank,design)
        # No development rankings are computed until all500 updates completed.
        states = {arm:{k:v.copy() for k,v in np.load(OUT/'权重'/filename,allow_pickle=False).items()}
                  for arm,filename in [('Identity','initial.npz'),('Rank8Final','final.npz')]}
        rows = rankings(read(OUT/'冻结评测任务.json')['tasks'],features(),indices,states)
        put('逐题排序结果.json',{'rows':rows});put('感知对照汇总.json',summarize(rows))
        state['phase'] = 'evaluated_pending_audit';state['state'] = 'completed';state['ranking_records'] = len(rows)
        state['elapsed_seconds'] = time.monotonic()-start;progress(state)
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) <= 40*1024**2
        print(json.dumps({'completed':True,'encoder_calls':144,'training_steps':500,'rankings':len(rows),'summary':summarize(rows)['counts']},ensure_ascii=False),flush=True)
    except BaseException as exc:
        state['state'] = 'failed';state['error_type'] = type(exc).__name__;state['error'] = str(exc);progress(state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','run']);args = parser.parse_args()
    freeze() if args.mode == 'freeze' else run()
