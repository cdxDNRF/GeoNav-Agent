"""In-memory S4-compatible cuts, frozen global embeddings for fit sources only."""
from io import BytesIO
from pathlib import Path
from hashlib import sha256
import json,time
import numpy as np
import torch
from PIL import Image
from data.scaled_masa import scaled_crop
from data.process_masa import MEAN,STD


def virtual_patch(source,cell):
    stream=BytesIO();scaled_crop(source,cell).save(stream,format='JPEG',quality=75)
    payload=stream.getvalue()
    with Image.open(BytesIO(payload)) as image:
        rgb=np.asarray(image.convert('RGB'),np.uint8)
        pixels=np.asarray(image.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
    tensor=torch.from_numpy(((pixels-MEAN)/STD).transpose(2,0,1))
    return tensor,sha256(payload).hexdigest(),sha256(rgb.tobytes()).hexdigest()


def build_cache(root,rows,model_dir,device):
    from transformers import CLIPVisionModelWithProjection
    from train.curiosity_controlled import write_new
    from train.dyncur_tiny import digest
    root=Path(root)
    if root.exists():raise ValueError('fit cache already exists')
    root.mkdir(parents=True);started=time.monotonic();model=CLIPVisionModelWithProjection.from_pretrained(str(model_dir),local_files_only=True).to(device).eval().requires_grad_(False)
    table={};provenance={}
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index,row in enumerate(rows):
            path=Path(row['absolute_raw_path'])
            if digest(path)!=row['raw_sha256']:raise ValueError('raw input drift')
            chunks=[]
            with Image.open(path) as source:
                source.load()
                for start in range(0,100,25):
                    batch=[]
                    for cell in range(start,start+25):
                        tensor,jpeg,rgb=virtual_patch(source,cell);batch.append(tensor)
                        provenance[row['area']+'/'+str(cell)]=dict(jpeg_sha256=jpeg,rgb_sha256=rgb)
                    chunks.append(model(torch.stack(batch).to(device)).image_embeds.float().cpu().numpy())
            table[row['area']]=np.concatenate(chunks)
            if (index+1)%10==0 or index+1==len(rows):print(dict(cache_sources=index+1,planned=len(rows)),flush=True)
    for array in table.values():
        if array.shape!=(100,512) or array.dtype!=np.float32 or not np.isfinite(array).all():raise ValueError('invalid fit embedding')
    np.savez(root/'全局特征.npz',**table);write_new(root/'图像来源.json',provenance)
    write_new(root/'缓存收据.json',dict(sources=len(rows),virtual_patches=100*len(rows),
        bytes_uncompressed=sum(a.nbytes for a in table.values()),file_sha256=digest(root/'全局特征.npz'),
        provenance_sha256=digest(root/'图像来源.json'),encoder_sha256=digest(Path(model_dir)/'model.safetensors'),
        seconds=time.monotonic()-started,peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
        local_files_only=True,encoder_trainable_parameters=0,patch_files_saved=0))
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    return table
