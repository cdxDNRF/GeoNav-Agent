"""CV040 shared frozen image encoder and image-only Spatial64 head.

Dataset crop/source/label lookup is outside this module. No downloads, model
search, GPS/K/heading input, navigation or unvisited parent representation.
"""
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame
from project.src.agents.frozen_crossview_encoder_v2 import construct_mapped_model
from project.src.agents.frozen_crossview_encoder_v1 import tensor_digest

MEAN=np.array([.48145466,.4578275,.40821073],np.float32)
STD=np.array([.26862954,.26130258,.27577711],np.float32)


def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes());return h.hexdigest()


def tensor_sha(t):return array_sha(t.detach().cpu().contiguous().numpy())


def prepare(frame,kind):
    if type(frame) is not RGBFrame or kind not in ('ground','aerial'):raise ValueError('only bounded RGB and explicit image modality')
    a=np.frombuffer(frame.pixels,np.uint8).reshape(frame.height,frame.width,3)
    with Image.fromarray(a) as image:
        if kind=='ground':
            scale=224/max(frame.width,frame.height);w=max(1,int(math.floor(frame.width*scale+.5)));h=max(1,int(math.floor(frame.height*scale+.5)))
            with image.resize((w,h),Image.Resampling.BILINEAR) as resized:b=np.array(resized,np.float32)/255
            left=(224-w)//2;top=(224-h)//2;canvas=np.empty((224,224,3),np.float32);canvas[:]=MEAN
            canvas[top:top+h,left:left+w]=b;padding=[left,top,224-w-left,224-h-top]
        else:
            with image.resize((224,224),Image.Resampling.BILINEAR) as resized:canvas=np.array(resized,np.float32)/255
            w=h=224;padding=[0,0,0,0]
    normalized=np.ascontiguousarray(((canvas-MEAN)/STD).transpose(2,0,1)[None],dtype=np.float32)
    return torch.from_numpy(normalized),dict(modality=kind,input_rgb_sha256=frame.fingerprint(),native_wh=[frame.width,frame.height],
        resized_wh=[w,h],padding_ltrb=padding,pixel_tensor_sha256=array_sha(normalized),shape=list(normalized.shape))


class SharedVision:
    def __init__(self,folder,before_forward,*,max_calls=8):
        if type(max_calls) is not int or max_calls<1:raise ValueError('positive call budget')
        from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        cfg=dict(cfg['vision_config'],projection_dim=512)
        if any(cfg.get(k)!=v for k,v in [('image_size',224),('patch_size',32),('hidden_size',768),('num_hidden_layers',12)]):raise ValueError('frozen architecture differs')
        raw=torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
        state={k:v for k,v in raw.items() if k.startswith(('vision_model.','visual_projection.'))}
        ignored=[]
        for key in list(state):
            if key.endswith('.position_ids'):
                if state[key].flatten().tolist()!=list(range(50)):raise ValueError('unexpected constant positions')
                ignored.append(key);state.pop(key)
        config=CLIPVisionConfig(**cfg);config._attn_implementation='eager'
        self.model=construct_mapped_model(config,state,CLIPVisionModelWithProjection)
        del state,raw
        self.parameters=sum(p.numel() for p in self.model.parameters())
        if self.parameters!=87849216:raise ValueError('actual vision parameter count differs')
        self.initial=tensor_digest(self.model);self.calls=0;self.vision_calls=0;self.maximum=max_calls;self.sink=before_forward
        self.hook=self.model.vision_model.register_forward_pre_hook(self._count_vision)
        self.report=dict(parameters=self.parameters,trainable=0,device='cpu',strict=True,mapped_assign=True,ignored_constant_buffers=ignored,initial_tensor_sha256=self.initial)

    def _count_vision(self,module,args):
        self.vision_calls+=1
        if self.vision_calls>self.maximum:raise ValueError('image tower budget exceeded')

    def encode(self,frame,kind,*,wrapper=False):
        if self.calls>=self.maximum:raise ValueError('encoder budget exhausted; no retry')
        pixels,receipt=prepare(frame,kind)
        self.calls+=1;receipt.update(attempt=self.calls,wrapper=wrapper)
        self.sink(receipt)
        with torch.inference_mode():
            if wrapper:
                out=self.model(pixel_values=pixels,return_dict=True)
                hidden=out.last_hidden_state;global_projected=out.image_embeds
                cls=self.model.vision_model.post_layernorm(hidden[:,0,:])
                reference=self.model.visual_projection(cls)
            else:
                out=self.model.vision_model(pixel_values=pixels,return_dict=True)
                hidden=out.last_hidden_state;cls=self.model.vision_model.post_layernorm(hidden[:,0,:])
                global_projected=self.model.visual_projection(cls)
                reference=self.model.visual_projection(out.pooler_output)
            delta=float((global_projected-reference).abs().max())
            if delta!=0:raise ValueError('CLS post-LN/projection differs from native pooled path')
            if tuple(hidden.shape)!=(1,50,768):raise ValueError('unexpected spatial hidden shape')
            if kind=='ground':feature=F.normalize(global_projected,dim=-1,eps=1e-12)[0]
            else:
                patch_hidden=self.model.vision_model.post_layernorm(hidden[:,1:,:])
                feature=F.normalize(self.model.visual_projection(patch_hidden),dim=-1,eps=1e-12)[0]
            feature=feature.detach().cpu().numpy().copy()
        expected=(512,) if kind=='ground' else (49,512)
        if feature.shape!=expected or not np.isfinite(feature).all() or not np.allclose(np.linalg.norm(feature,axis=-1),1,atol=1e-5,rtol=0):raise ValueError('feature shape/norm invalid')
        receipt.update(feature_shape=list(feature.shape),feature_sha256=array_sha(feature),CLS_native_max_abs_difference=delta,
                       vision_calls=self.vision_calls)
        return feature,receipt

    def close(self):
        after=tensor_digest(self.model)
        if after!=self.initial or any(p.requires_grad or p.grad is not None for p in self.model.parameters()) or any(m.training for m in self.model.modules()):raise ValueError('frozen image tower changed')
        if self.calls!=self.vision_calls:raise ValueError('pipeline/tower counter mismatch')
        self.hook.remove();self.report.update(final_tensor_sha256=after,unchanged=True,calls=self.calls,vision_calls=self.vision_calls)
        return self.report


def validate_tensor_packet(packet):
    if type(packet) is not dict or set(packet)!={'given_target_feature','current_patch_features'}:raise ValueError('target/current tensors only; metadata/bank/labels forbidden')
    g,a=packet['given_target_feature'],packet['current_patch_features']
    if not isinstance(g,torch.Tensor) or not isinstance(a,torch.Tensor) or g.ndim!=2 or a.ndim!=3 or tuple(g.shape[1:])!=(512,) or tuple(a.shape[1:])!=(49,512) or len(g)!=len(a) or not 1<=len(g)<=128:raise ValueError('B512 and B49x512 required')
    if any(t.device.type!='cpu' or t.dtype!=torch.float32 or t.requires_grad or not torch.isfinite(t).all() for t in (g,a)):raise ValueError('finite detached float32 CPU features required')
    gn=g.norm(dim=-1);an=a.norm(dim=-1)
    if not torch.all((gn-1).abs().le(1e-5)|gn.eq(0)) or not torch.all((an-1).abs().le(1e-5)):raise ValueError('unit features; zero only for disabled target')
    return g,a


class Spatial64(nn.Module):
    def __init__(self,seed=0):
        super().__init__()
        if seed not in (0,1,2):raise ValueError('frozen seeds')
        generator=torch.Generator(device='cpu').manual_seed(seed)
        w=torch.randn((512,64),generator=generator,dtype=torch.float32)/math.sqrt(512)
        self.Wg=nn.Parameter(w.clone());self.Wa=nn.Parameter(w.clone());self.b_empty=nn.Parameter(torch.tensor(math.log(49),dtype=torch.float32))
        self.forward_calls=0;self.pair_uses=0;self.last_trace=None

    def forward(self,given_target_feature,current_patch_features):
        g,a=validate_tensor_packet(dict(given_target_feature=given_target_feature,current_patch_features=current_patch_features))
        self.forward_calls+=1;self.pair_uses+=len(g)
        q=F.normalize(g@self.Wg,dim=-1,eps=1e-12);k=F.normalize(a@self.Wa,dim=-1,eps=1e-12)
        spatial=10*torch.einsum('bd,bjd->bj',q,k);empty=self.b_empty.expand(len(g),1)
        logits=torch.cat((spatial,empty),dim=1)
        self.last_trace=dict(input_g_sha256=tensor_sha(g),input_a_sha256=tensor_sha(a),q_sha256=tensor_sha(q),k_sha256=tensor_sha(k),
                             spatial_sha256=tensor_sha(spatial),q_norms=q.detach().norm(dim=-1).tolist(),spatial_abs_max=float(spatial.detach().abs().max()))
        return logits


def predict(head,packet):
    g,a=validate_tensor_packet(packet)
    with torch.no_grad():logits=head(g,a)
    return logits.detach().cpu(),dict(head.last_trace)


def raw_logits(packet):
    g,a=validate_tensor_packet(packet)
    spatial=10*torch.einsum('bd,bjd->bj',g,a)
    return torch.cat((spatial,torch.full((len(g),1),math.log(49),dtype=torch.float32)),dim=1)


def optimizer(head):return torch.optim.Adam(head.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)


def state_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):h.update(b'tensor'+tensor_sha(x).encode())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update((type(x).__name__+':'+repr(x)).encode())
    visit(value);return h.hexdigest()
