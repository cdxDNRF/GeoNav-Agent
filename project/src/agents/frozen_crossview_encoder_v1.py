"""Image-only CPU encoder adapter. Strict local assets; never downloads/trains.

No task/dataset/goal-position argument is accepted by encode(). The owner binds
weight/config hashes before constructing it. This does not choose an action.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame
from project.src.data.mmgag_views_v1 import ground_pixels,aerial_pixels


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pixels(frame,kind,processor):
    if type(frame) is not RGBFrame or kind not in ('ground','aerial'):raise ValueError('单幅RGB与明确模态必需')
    rgb=np.frombuffer(frame.pixels,dtype=np.uint8).reshape(frame.height,frame.width,3)
    return ground_pixels(rgb,processor) if kind=='ground' else aerial_pixels(rgb)


def tensor_digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(tuple(a.shape)).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


class FrozenCrossViewEncoder:
    mode='frozen_cpu'

    def __init__(self,root,assets,*,max_forward_calls=8,before_forward=None):
        if type(max_forward_calls) is not int or not 1<=max_forward_calls<=64:raise ValueError('编码调用上限无效')
        self.root=Path(root).resolve();self.assets={e['path']:dict(e) for e in assets}
        self.max_calls=max_forward_calls;self.calls=0;self.before_forward=before_forward
        self.models={};self.before={};self.closed=False;self.records=[]
        required={'models/clip_vit_b32_public_v1/config.json','models/clip_vit_b32_public_v1/preprocessor_config.json',
            'models/clip_vit_b32_public_v1/pytorch_model.bin','models/Sat2Cap/config.json','models/Sat2Cap/model.safetensors'}
        if set(self.assets)!=required:raise ValueError('仅允许两份已绑定编码器及其配置')
        for name,e in self.assets.items():
            p=self.root/name
            if not p.resolve().is_relative_to(self.root) or digest(p)!=e['sha256']:raise ValueError('本地编码资产绑定变化')
        self.processor=json.loads((self.root/'models/clip_vit_b32_public_v1/preprocessor_config.json').read_text(encoding='utf-8'))

    def _load(self,kind):
        if kind in self.models:return self.models[kind]
        import torch
        from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        folder=self.root/('models/clip_vit_b32_public_v1' if kind=='ground' else 'models/Sat2Cap')
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        cfg=cfg.get('vision_config',cfg);cfg=dict(cfg,projection_dim=512)
        if any(cfg.get(k)!=v for k,v in [('image_size',224),('patch_size',32),('hidden_size',768),('num_hidden_layers',12)]):raise ValueError('编码器架构不符')
        if kind=='ground':
            raw=torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
            selected={k:v for k,v in raw.items() if k.startswith(('vision_model.','visual_projection.'))}
        else:
            from safetensors.torch import load_file
            raw=load_file(str(folder/'model.safetensors'),device='cpu');selected=dict(raw)
        config=CLIPVisionConfig(**cfg);config._attn_implementation='eager'
        model=CLIPVisionModelWithProjection(config)
        ignored=[]
        for key in list(selected):
            if key.endswith('.position_ids') and key not in model.state_dict():
                if selected[key].flatten().tolist()!=list(range(50)):raise ValueError('Unexpected position buffer')
                ignored.append(key);selected.pop(key)
        model.load_state_dict(selected,strict=True);del selected,raw
        model.requires_grad_(False);model.eval();model.to('cpu')
        self.models[kind]=model;self.before[kind]=tensor_digest(model)
        self.records.append({'kind':kind,'parameters':sum(p.numel() for p in model.parameters()),'trainable':0,
            'device':'cpu','strict_load':True,'ignored_constant_buffers':ignored})
        return model

    def encode(self,frame,kind):
        if self.closed:raise ValueError('编码器已关闭')
        if type(frame) is not RGBFrame or kind not in ('ground','aerial'):raise ValueError('仅接收单幅RGB图像')
        if self.calls>=self.max_calls:raise ValueError('已达到本批编码调用上限')
        # Validate preprocessing before taking a model-call budget slot.
        array=pixels(frame,kind,self.processor)
        model=self._load(kind)
        self.calls+=1
        if self.before_forward is not None:self.before_forward({'attempt':self.calls,'kind':kind,'pixel_sha256':frame.fingerprint()})
        import torch
        tensor=torch.from_numpy(array[None]).to(device='cpu')
        with torch.inference_mode():result=model(pixel_values=tensor).image_embeds[0].detach().cpu().numpy().copy()
        if result.shape!=(512,) or not np.isfinite(result).all() or float(np.linalg.norm(result))<=1e-12:raise ValueError('编码器输出无效')
        return tuple(float(x) for x in result)

    def close(self):
        if self.closed:return self.report
        checks=[]
        for kind,model in self.models.items():
            after=tensor_digest(model)
            if after!=self.before[kind] or model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('冻结编码器被修改')
            checks.append({'kind':kind,'before_tensor_sha256':self.before[kind],'after_tensor_sha256':after,'unchanged':True})
        for name,e in self.assets.items():
            if digest(self.root/name)!=e['sha256']:raise ValueError('编码资产文件在执行期间改变')
        self.models.clear();self.closed=True
        self.report={'calls':self.calls,'encoders':list(self.records),'frozen_tensor_checks':checks,'device':'cpu','training':0}
        return self.report
