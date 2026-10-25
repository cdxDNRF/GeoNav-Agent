"""Memory-only engineering repair: meta construction, mapped weights, one slot.

Same architecture, tensors, CPU math/preprocessing and forward budget as v1.
"""
import gc
import json
from .frozen_crossview_encoder_v1 import FrozenCrossViewEncoder as Previous,tensor_digest


def construct_mapped_model(config,state,model_factory):
    import torch
    with torch.device('meta'):
        model=model_factory(config)
    # assign=True reuses the already loaded CPU tensors rather than allocating
    # and copying a second full set of model parameters.
    model.load_state_dict(state,strict=True,assign=True)
    for name,buffer in list(model.named_buffers()):
        if buffer.device.type!='meta':continue
        if not name.endswith('.position_ids') or tuple(buffer.shape)!=(1,50) or buffer.dtype!=torch.int64:
            raise ValueError('Unknown meta buffer;do not guess its content')
        parent,_,field=name.rpartition('.')
        setattr(model.get_submodule(parent),field,torch.arange(50,dtype=torch.int64,device='cpu').expand((1,-1)))
    if any(p.device.type!='cpu' for p in model.parameters()) or any(b.device.type!='cpu' for b in model.buffers()):
        raise ValueError('All model tensors must be materialized on CPU')
    model.requires_grad_(False);model.eval()
    return model


class FrozenCrossViewEncoder(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.released_checks=[]

    def _release_slot(self):
        for kind,model in self.models.items():
            after=tensor_digest(model)
            if after!=self.before[kind] or model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError('Frozen model changed before release')
            self.released_checks.append({'kind':kind,'before_tensor_sha256':self.before[kind],
                'after_tensor_sha256':after,'unchanged':True})
        self.models.clear();gc.collect()

    def _load(self,kind):
        if kind in self.models:return self.models[kind]
        if self.models:self._release_slot()
        import torch
        from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        folder=self.root/('models/clip_vit_b32_public_v1' if kind=='ground' else 'models/Sat2Cap')
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        cfg=dict(cfg.get('vision_config',cfg),projection_dim=512)
        if any(cfg.get(k)!=v for k,v in [('image_size',224),('patch_size',32),('hidden_size',768),('num_hidden_layers',12)]):
            raise ValueError('Encoder architecture differs')
        if kind=='ground':
            raw=torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
            state={k:v for k,v in raw.items() if k.startswith(('vision_model.','visual_projection.'))}
        else:
            from safetensors.torch import load_file
            raw=load_file(str(folder/'model.safetensors'),device='cpu');state=dict(raw)
        ignored=[]
        for key in list(state):
            if key.endswith('.position_ids'):
                if state[key].flatten().tolist()!=list(range(50)):raise ValueError('Unexpected position buffer')
                ignored.append(key);state.pop(key)
        config=CLIPVisionConfig(**cfg);config._attn_implementation='eager'
        model=construct_mapped_model(config,state,CLIPVisionModelWithProjection)
        del state,raw
        before=tensor_digest(model)
        if kind in self.before and before!=self.before[kind]:raise ValueError('Reloaded tensor identity differs')
        self.before[kind]=before;self.models[kind]=model
        self.records.append({'kind':kind,'parameters':sum(p.numel() for p in model.parameters()),'trainable':0,
            'device':'cpu','strict_load':True,'mapped_assign':True,'initialization_device':'meta',
            'max_live_encoders':1,'ignored_constant_buffers':ignored})
        return model

    def close(self):
        if self.closed:return self.report
        self._release_slot()
        report=super().close()
        report['frozen_tensor_checks']=list(self.released_checks)
        report['max_live_encoders']=1
        return report
