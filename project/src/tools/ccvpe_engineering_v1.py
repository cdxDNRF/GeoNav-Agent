"""CV030 one-shot engineering binding and CPU worker with parent RSS watchdog."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1'
VENDOR = ROOT/'project/.runtime/ccvpe_engineering_v1'
MODEL = ROOT/'models/ccvpe_kitti_public_v1/model.pt'
SOURCE_NAMES = ['tools/ccvpe_acquire_v1.py', 'tools/ccvpe_engineering_v1.py', 'agents/ccvpe_runtime_v1.py',
    'eval/audit_ccvpe_engineering_v1.py', 'tests/test_ccvpe_engineering_v1.py', 'documents/close_ccvpe_engineering_v1.py']
LIMIT = 4*1024**3


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def entry(p):
    with p.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha)


def put(name, data):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def event(kind, **kw):
    row = dict(kind=kind, elapsed=time.monotonic()-START, **kw)
    with (OUT/'运行事件.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(row, ensure_ascii=False), flush=True)


def bind():
    import unittest
    from project.src.tests.test_ccvpe_engineering_v1 import Tests
    result = unittest.TestResult(); unittest.defaultTestLoader.loadTestsFromTestCase(Tests).run(result)
    if not result.wasSuccessful():
        raise ValueError(str(result.errors+result.failures))
    acquisition = read(OUT/'工件取得完成.json')
    sources = [entry(ROOT/'project/src'/n) for n in SOURCE_NAMES]
    old = ROOT/'DATA/processed_data/OpenCVL/空间监督小样本预检_v1'
    inputs = [r['file'] for r in read(old/'像素质量回执.json')['inputs']]
    protected = []
    for folder in (old,):
        protected.extend(entry(p) for p in sorted(folder.rglob('*')) if p.is_file())
    for name in ('调研/20_轻量空间关系感知与拒绝协议_v1.md', '调研/20_轻量空间关系感知与拒绝协议_v1.json',
                 '调研/20_协议复核_v1.json'):
        protected.append(entry(ROOT/name))
    for name in ('data/spatial_perception_protocol_v1.py','eval/audit_spatial_perception_protocol_v1.py',
                 'tests/test_spatial_perception_protocol_v1.py','documents/close_spatial_perception_protocol_v1.py'):
        protected.append(entry(ROOT/'project/src'/name))
    # Protect running platform implementation without loading it.
    protected.extend(entry(p) for p in sorted((ROOT/'project/src/platform_core').rglob('*')) if p.is_file() and p.suffix in ('.py','.js','.css','.html'))
    protected.extend(entry(p) for p in sorted((ROOT/'project/src/webapp').rglob('*')) if p.is_file() and p.suffix in ('.py','.js','.css','.html'))
    for name in ('平台/启动平台.ps1','平台/启动平台.cmd'):
        protected.append(entry(ROOT/name))
    idx = read(ROOT/'项目导航/实验索引.json')['batches']
    assert len(idx) == 154
    d = dict(task='CV-030', sources=sources, checkpoint=acquisition['checkpoint'], vendor=acquisition['vendor'],
        inputs=inputs, protected=protected, runtime_current=read(ROOT/'docs/维护/版本生命周期.json')['current'],
        index_count=154, index_canonical=hashlib.sha256(json.dumps(idx, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        budgets=dict(pipeline_calls=12, reference=4, full_wrapper=4, synthetic=4, backbone_calls=24,
                     source_decodes=8, rss_bytes=LIMIT, wall_seconds=1800, gpu_calls=0, training=0, navigation=0),
        tests=dict(count=result.testsRun, passed=True, actual_model_forward=0),
        engineering_full_parent_privileged=True, calibration_sources=[0,1], screen_authorized=False,
        inference_permission='explicit user reply for engineering only', c_drive_task_outputs=0)
    for e in d['inputs']+d['vendor']+[d['checkpoint']]+sources+protected:
        if entry(ROOT/e['path'])['sha256'] != e['sha256']:
            raise ValueError('Binding mismatch: '+e['path'])
    put('首前向冻结绑定.json', d)
    print(json.dumps(dict(bound=True, tests=result.testsRun, real_forward=0)))


def verify_binding():
    d = read(OUT/'首前向冻结绑定.json')
    for e in d['inputs']+d['vendor']+[d['checkpoint']]+d['sources']+d['protected']:
        if entry(ROOT/e['path'])['sha256'] != e['sha256']:
            raise ValueError('Bound original changed: '+e['path'])
    return d


def guard_violation(rss, elapsed):
    return rss > LIMIT or elapsed > 1800


def execute():
    import psutil
    verify_binding()
    put('工程执行开始.json', dict(pid=os.getpid(), started=time.time(), executable=sys.executable))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    start = time.monotonic(); peak = 0; reason = None; samples = 0
    with (OUT/'工作进程stdout.log').open('xb') as out, (OUT/'工作进程stderr.log').open('xb') as err:
        p = subprocess.Popen([sys.executable, '-B', '-m', 'project.src.tools.ccvpe_engineering_v1', 'worker'],
            cwd=ROOT, env=env, stdout=out, stderr=err)
        process = psutil.Process(p.pid)
        put('工作进程身份.json', dict(pid=p.pid, parent=os.getpid(), command='ccvpe_engineering_v1 worker'))
        while p.poll() is None:
            try:
                rss = process.memory_info().rss
            except psutil.NoSuchProcess:
                break
            peak = max(peak, rss); samples += 1
            if guard_violation(rss, time.monotonic()-start):
                reason = 'rss' if rss > LIMIT else 'wall'
                p.terminate()  # only this freshly spawned worker
                break
            time.sleep(.05)
        code = p.wait()
    receipt = dict(exit_code=code, peak_sampled_rss=peak, wall_seconds=time.monotonic()-start, samples=samples,
                   stop_reason=reason, passed=code == 0 and reason is None, only_own_worker_controlled=True)
    put('资源护栏回执.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    if not receipt['passed']:
        raise SystemExit(1)


def worker():
    global START
    START = time.monotonic()
    # Deny network before importing torch or any vendored module; no hidden backbone download.
    import socket
    def denied(*args, **kwargs):
        raise RuntimeError('Network forbidden in engineering worker')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    d = verify_binding()
    import torch
    import numpy as np
    from PIL import Image
    from project.src.agents import ccvpe_runtime_v1 as rt
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    event('model_load_started')
    model, controlled, manifest = rt.load_runtime(VENDOR, MODEL)
    count = sum(p.numel() for p in model.parameters())
    if count > 100_000_000:
        raise ValueError('Parameter cap exceeded')
    put('严格加载与冻结前张量.json', dict(parameters=count, tensor_manifest=manifest, trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        all_modules_eval=all(not m.training for m in model.modules()), checkpoint_strict_tensor_equal=True,
        torch=str(torch.__version__), numpy=str(np.__version__), device='cpu', network_denied=True))
    event('model_loaded', parameters=count)
    del manifest
    tensors = {}
    for e in d['inputs']:
        path = ROOT/e['path']; ground = '_ground.' in path.name
        event('source_decode_started', path=e['path'])
        with Image.open(path) as im:
            im.load()
            rgb = im.convert('RGB')
            tensor = rt.prepare_rgb(rgb, ground=ground)
            dimensions = list(im.size)
            rgb.close()
        tensors[path.stem] = tensor
        event('source_decode_complete', path=e['path'], dimensions=dimensions, tensor_shape=list(tensor.shape), tensor_sha256=rt.tensor_digest(tensor))
    calls = []; backbone = dict(ground=0, aerial=0)
    for attr, method, key in [('grd_efficientnet','extract_features','ground'), ('sat_efficientnet','extract_features_multiscale','aerial')]:
        obj = getattr(model, attr); orig = getattr(obj, method)
        def wrapped(*args, _orig=orig, _key=key, **kwargs):
            backbone[_key] += 1
            return _orig(*args, **kwargs)
        setattr(obj, method, wrapped)
    def invoke(label, g, a, arm=None, means=None, wrong=None):
        if len(calls) >= 12:
            raise ValueError('Pipeline call cap exceeded')
        event('pipeline_started', number=len(calls)+1, label=label, arm=arm or 'reference')
        start = time.monotonic()
        with torch.inference_mode():
            if arm is None:
                captured = {}; hooks = []
                for level in range(1, 7):
                    def hook(module, args, output, level=level): captured[level] = output.detach().clone()
                    hooks.append(getattr(model, f'grd_feature_to_descriptor{level}').register_forward_hook(hook))
                try: outputs = model(g, a)
                finally:
                    for h in hooks: h.remove()
                descriptors = [captured[i] for i in range(1, 7)]; actual = g
            else:
                outputs, descriptors, actual = rt.run_control(model, controlled, g, a, arm, means, wrong)
        if len(outputs) != 9 or not all(torch.isfinite(t).all().item() for t in outputs):
            raise ValueError('Nonfinite or wrong output structure')
        receipt = dict(number=len(calls)+1, label=label, arm=arm or 'reference', seconds=time.monotonic()-start,
            actual_goal_sha=rt.tensor_digest(actual), aerial_sha=rt.tensor_digest(a),
            descriptors=[dict(shape=list(t.shape), sha256=rt.tensor_digest(t), values=t.tolist()) for t in descriptors],
            outputs=[dict(shape=list(t.shape), sha256=rt.tensor_digest(t), maximum_absolute=float(t.abs().max()), finite=True) for t in outputs])
        put(f'前向_{len(calls)+1:02d}.json', receipt); calls.append(receipt)
        event('pipeline_complete', number=len(calls), label=label, seconds=receipt['seconds'], backbone=dict(backbone))
        return outputs, descriptors
    equalities = []; calibration = []
    for i in range(4):
        g, a = tensors[f'pair_{i}_ground'], tensors[f'pair_{i}_aerial']
        ref, desc = invoke(f'pair{i}_reference',g,a)
        full, full_desc = invoke(f'pair{i}_Full',g,a,'Full')
        differences = [float((x-y).abs().max()) for x,y in zip(ref,full)]
        row = dict(pair=i, output_maxabs=differences, all_descriptors_equal=all(torch.equal(x,y) for x,y in zip(desc,full_desc)), passed=max(differences)<=1e-6)
        put(f'等价_{i}.json', row); equalities.append(row)
        if i < 2: calibration.append([t.clone() for t in desc])
        del ref, full, desc, full_desc
        if not row['passed'] or not row['all_descriptors_equal']:
            raise ValueError('Reference/Full mismatch')
    means = [(x+y)/2 for x,y in zip(*calibration)]
    put('校准目标多层均值.json', dict(source_pairs=[0,1], from_calls=[1,3], levels=[dict(shape=list(t.shape), values=t.tolist(),sha256=rt.tensor_digest(t)) for t in means]))
    # Deterministic synthetic tensors, not additional real images or source observations.
    gen = torch.Generator(device='cpu').manual_seed(30030)
    ga = torch.rand((1,3,256,1024),generator=gen); gb = torch.rand((1,3,256,1024),generator=gen)
    sa = torch.rand((1,3,512,512),generator=gen)
    da, dga = invoke('synthetic_A_Disabled',ga,sa,'Disabled')
    db, dgb = invoke('synthetic_B_Disabled',gb,sa,'Disabled')
    disabled_diff = [float((x-y).abs().max()) for x,y in zip(da,db)]
    disabled_zero = all(torch.count_nonzero(t).item()==0 for t in da[3:]+db[3:])
    del da, db
    mo, md = invoke('synthetic_A_Mean',ga,sa,'Mean',means=means)
    mean_exact = [torch.equal(x,y) for x,y in zip(md,means)]
    mean_replaced = [not torch.equal(x,y) for x,y in zip(md,dga)]
    del mo
    wo, wd = invoke('synthetic_A_Wrong_B',ga,sa,'Wrong',wrong=gb)
    wrong_exact = [torch.equal(x,y) for x,y in zip(wd,dgb)]
    wrong_changed = [not torch.equal(x,y) for x,y in zip(wd,dga)]
    active = [bool(torch.count_nonzero(t).item()) for t in wo[3:]]
    controls = dict(disabled_output_maxabs=disabled_diff, disabled_matching_all_zero=disabled_zero,
        mean_all_levels_exact=mean_exact, mean_all_levels_replaced=mean_replaced,
        wrong_all_levels_equal_alternate=wrong_exact, wrong_all_levels_changed=wrong_changed, wrong_matching_active=active,
        goal_A_sha=rt.tensor_digest(ga),goal_B_sha=rt.tensor_digest(gb),
        passed=disabled_zero and max(disabled_diff)==0 and all(mean_exact+mean_replaced+wrong_exact+wrong_changed+active))
    put('全部控制检查.json',controls)
    after = rt.state_manifest(model.state_dict())
    before = read(OUT/'严格加载与冻结前张量.json')['tensor_manifest']
    unchanged = after == before
    put('冻结后张量.json',dict(tensor_manifest=after,unchanged=unchanged))
    if not controls['passed'] or not unchanged:
        raise ValueError('Control or frozen-state validation failed')
    put('工程执行完成.json', dict(passed=True,pipeline_calls=len(calls),paired_calls=8,synthetic_calls=4,
        backbone_calls=backbone,source_decodes=len(tensors),parameters=count,equivalence_maxabs=max(max(r['output_maxabs']) for r in equalities),
        controls_passed=True,frozen_unchanged=unchanged,training=0,gpu=0,navigation=0,screen=0,confirm_groups_consumed=0))
    event('worker_complete',pipeline_calls=len(calls))


if __name__ == '__main__':
    if sys.argv[1:] not in (['bind'],['execute'],['worker']): raise SystemExit('bind|execute|worker')
    {'bind':bind,'execute':execute,'worker':worker}[sys.argv[1]]()
