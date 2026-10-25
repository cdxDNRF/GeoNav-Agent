"""One fixed Sample4Geo/CVUSA encoder engineering probe; not a navigation policy."""
from __future__ import annotations

import hashlib
import io
import json
import socket
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'DATA/processed_data/MM-GAG/单权重取得与工程_v1'
RUNTIME = ROOT / 'project/.runtime/sample4geo_engineering_v1'
sys.path.insert(0, str(ROOT))
from project.src.tools.sample4geo_acquire_v1 import record, sha, write_new

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SHAPES = {'ground': (140, 768), 'aerial': (384, 384)}


def preprocess(rgb, view):
    if view not in SHAPES or not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError('RGB uint8 and known view required')
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('RGB shape')
    h, w = SHAPES[view]
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
    # Same float32 Normalize convention as the public Albumentations validation transform.
    x = (resized.astype(np.float32)-np.asarray(MEAN, np.float32)*255.0) / (np.asarray(STD, np.float32)*255.0)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0), resized


def tensor_receipts(state):
    result = []
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
            raise ValueError('only named CPU tensors accepted')
        if not value.is_floating_point() or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('finite FP32 checkpoint required')
        raw = value.detach().contiguous().numpy().tobytes()
        result.append({'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                       'elements': value.numel(), 'sha256': hashlib.sha256(raw).hexdigest()})
    return result


def strict_restore(model, state):
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(f'checkpoint keys mismatch: missing={len(set(expected)-set(state))}, unexpected={len(set(state)-set(expected))}')
    for name, x in expected.items():
        if not isinstance(state[name], torch.Tensor) or x.shape != state[name].shape or x.dtype != state[name].dtype:
            raise ValueError('checkpoint shape/dtype mismatch at '+name)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return tensor_receipts(model.state_dict())


def make_model(timm):
    class SharedEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', pretrained=False, num_classes=0)
            self.logit_scale = torch.nn.Parameter(torch.ones([])*np.log(1/0.07))

        def forward(self, x):
            return self.model(x)

    return SharedEncoder()


def deny_network(*args, **kwargs):
    raise RuntimeError('network forbidden during model engineering')


def main():
    protocol = json.loads((OUT/'执行协议.json').read_text(encoding='utf-8'))
    binding = json.loads((OUT/'工程源码输入绑定.json').read_text(encoding='utf-8'))
    for item in binding['files']:
        if record(ROOT/item['path']) != item:
            raise ValueError('binding changed: '+item['path'])
    acquire = json.loads((OUT/'权重取得结果.json').read_text(encoding='utf-8'))
    if acquire['status'] != 'acquired' or record(ROOT/acquire['model']['path']) != acquire['model']:
        raise ValueError('verified weight missing/changed')
    write_new(OUT/'工程开始.json', {'protocol': record(OUT/'执行协议.json'),
                                   'binding': record(OUT/'工程源码输入绑定.json'),
                                   'weight': acquire['model'], 'recovery': 'inspect journal, never blind replay'})
    start = time.monotonic()
    events = OUT/'工程事件.jsonl'

    def event(**values):
        with events.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'elapsed_seconds': time.monotonic()-start, **values}, ensure_ascii=False)+'\n')
            f.flush()

    sys.path.insert(0, str(RUNTIME))
    import timm
    import psutil
    if timm.__version__ != '1.0.20' or not Path(timm.__file__).resolve().is_relative_to(RUNTIME.resolve()):
        raise ValueError('isolated timm identity mismatch')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    finished = threading.Event()
    peak_rss = [0]
    process = psutil.Process()

    def sample_memory():
        while not finished.wait(0.025):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    forwards = []
    raw_receipts = []
    try:
        with ExitStack() as stack:
            for target in ['socket.create_connection', 'socket.socket.connect', 'socket.socket.connect_ex',
                           'torch.hub.download_url_to_file']:
                stack.enter_context(patch(target, side_effect=deny_network))
            event(phase='before_weights_only_cpu_load')
            state = torch.load(ROOT/acquire['model']['path'], map_location='cpu', weights_only=True)
            checkpoint_tensors = tensor_receipts(state)
            event(phase='checkpoint_read', tensor_count=len(checkpoint_tensors))
            model = make_model(timm)
            params = sum(x.numel() for x in model.parameters())
            if params > protocol['max_simultaneous_model_parameters']:
                raise ValueError('parameter cap')
            before = strict_restore(model, state)
            if before != checkpoint_tensors:
                raise ValueError('restored tensor SHA differs')
            del state
            data_config = timm.data.resolve_model_data_config(model.model)
            if tuple(data_config['mean']) != MEAN or tuple(data_config['std']) != STD:
                raise ValueError('public normalization mismatch')
            write_new(OUT/'严格加载与逐张量.json', {'tensor_count':len(before), 'parameters':params,
                      'strict':True,'weights_only':True,'map_location':'cpu','all_restored_sha_equal':True,
                      'tensors':before,'timm':timm.__version__,'data_config':data_config})

            def forward(rgb, view, name):
                if len(forwards) >= protocol['max_input_forwards'] or time.monotonic()-start > 600:
                    raise ValueError('forward/time cap')
                x, resized = preprocess(rgb, view)
                event(phase='before_forward', name=name, input_shape=list(x.shape), index=len(forwards))
                then = time.monotonic()
                with torch.inference_mode():
                    y = model(x).cpu()
                if y.shape != (1,1024) or not bool(torch.isfinite(y).all()) or float(y.norm()) <= 0:
                    raise ValueError('embedding output contract')
                raw = y.numpy().copy()
                forwards.append({'name':name,'view':view,'input_shape':list(x.shape),
                    'input_tensor_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                    'output_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                    'output_shape':list(y.shape),'norm':float(y.norm()),'seconds':time.monotonic()-then})
                event(phase='forward_complete', **forwards[-1])
                return raw, resized

            synthetic = np.arange(96*64*3,dtype=np.uint8).reshape(96,64,3)
            outputs = [forward(synthetic, view, 'synthetic_'+view)[0] for view in ('ground','aerial')]
            images = []
            preview = OUT/'预览'
            preview.mkdir(exist_ok=False)
            for item, view in zip(protocol['raw_inputs'], ('ground','aerial')):
                event(phase='before_raw_read', path=item['path'], view=view)
                b = (ROOT/item['path']).read_bytes()
                if len(b) != item['bytes'] or hashlib.sha256(b).hexdigest() != item['sha256']:
                    raise ValueError('raw byte identity')
                with Image.open(io.BytesIO(b)) as img:
                    size = list(img.size)
                    orientation = img.getexif().get(274, 1)
                    rgb = np.array(ImageOps.exif_transpose(img).convert('RGB'))
                images.append(rgb)
                y, resized = forward(rgb, view, 'fit_'+view)
                outputs.append(y)
                Image.fromarray(resized).save(preview/(view+'_input.png'))
                raw_receipts.append({**item,'view':view,'original_wh':size,'exif_orientation':orientation,
                        'oriented_hw':list(rgb.shape[:2]),'network_hw':list(resized.shape[:2]),
                        'pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                        'height_scale':resized.shape[0]/rgb.shape[0],
                        'width_scale':resized.shape[1]/rgb.shape[1]})
                event(phase='raw_complete', **raw_receipts[-1])
            for rgb, view in zip(images, ('ground','aerial')):
                outputs.append(forward(rgb,view,'repeat_'+view)[0])
            if not (np.array_equal(outputs[2], outputs[4]) and np.array_equal(outputs[3], outputs[5])):
                raise ValueError('repeat encoding nondeterministic')
            after = tensor_receipts(model.state_dict())
            if before != after or any(x.grad is not None for x in model.parameters()) or model.training:
                raise ValueError('encoder not frozen')
            with (OUT/'工程特征.npy').open('xb') as f:
                np.save(f,np.concatenate(outputs),allow_pickle=False)
            write_new(OUT/'输入适配回执.json', {'raw_images':raw_receipts,'raw_read_count':2,
                    'ordinary_photo_is_not_panorama':True,'same_resize_as_public_inference':True,
                    'EXIF_adaptation':'explicit transpose before RGB, original uses cv2 BGR to RGB',
                    'aspect_preserved':False,'FOV_unknown':True,'accuracy_assessed':False})
            write_new(OUT/'编码器冻结核验.json', {'before_and_after_all_sha_equal':True,
                      'all_checkpoint_tensor_sha_equal':True,'tensor_count':len(before),
                      'trainable_parameters':0,'gradients_created':False,'eval':True,'device':'cpu'})
            result = {'status':'engineering_passed_not_perception_admission','parameters':params,
                'model_count':1,'forwards':forwards,'single_input_forward_count':len(forwards),
                'unique_raw_images':2,'unique_synthetic_inputs':2,'raw_reads':2,
                'repeat_max_absolute_difference':0.0,'network_guard_enabled':True,
                'training':0,'navigation':0,'development_rankings':0,'confirmation_access':0,
                'new_SR':None,'cuda_measured':False,'timm_path':str(Path(timm.__file__).relative_to(ROOT)),
                'CPU_threads':4,'elapsed_seconds':time.monotonic()-start}
    except Exception as e:
        event(phase='failed',error_type=type(e).__name__,error=str(e)[:300])
        result = {'status':'stopped','reason_type':type(e).__name__,'reason':str(e)[:300],
                  'completed_forwards':forwards,'raw_receipts':raw_receipts,
                  'recovery':'inspect before/after events and outputs; no blind retry'}
    finally:
        peak_rss[0] = max(peak_rss[0],process.memory_info().rss)
        finished.set()
        sampler.join(timeout=1)
    result['sampled_peak_CPU_RSS_bytes'] = peak_rss[0]
    result['RSS_measurement'] = 'whole process, sampled25ms; includes torch/runtime, not exact allocator peak'
    write_new(OUT/'工程结果.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('forwards','completed_forwards')},ensure_ascii=False),flush=True)
    return 0 if result['status'].startswith('engineering_passed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
