"""Download one user-authorized, pinned public CLIP asset; no model execution."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[3]
MODEL=ROOT/'models/clip_vit_b32_public_v1'
OUT=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
REV='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
SHA='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
BASE=f'https://huggingface.co/openai/clip-vit-base-patch32/resolve/{REV}/'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def new_json(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    MODEL.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    plan=MODEL/'获取方案.json'
    if not plan.exists():new_json(plan,{'task':'CV-002','repo':'openai/clip-vit-base-patch32','revision':REV,'files':['config.json','preprocessor_config.json','pytorch_model.bin'],'expected_bin_sha256':SHA,'source_sha256':digest(__file__),'max_retained_bytes':650*1024**2,'authorization':'user explicitly approved official ~605MB model plus configs on D drive','single_original_weight_copy':True})
    else:assert json.loads(plan.read_text(encoding='utf-8'))['source_sha256']==digest(__file__)
    sess=requests.Session();sess.headers['User-Agent']='geonav-cv002-public-asset'
    receipts=[]
    for name in ['config.json','preprocessor_config.json','pytorch_model.bin']:
        dst=MODEL/name
        if dst.exists():
            if name.endswith('.bin'):assert digest(dst)==SHA
            else:json.loads(dst.read_text(encoding='utf-8'))
            receipts.append({'file':name,'bytes':dst.stat().st_size,'sha256':digest(dst),'reused':True});continue
        partial=MODEL/(name+'.part');offset=partial.stat().st_size if partial.exists() else 0
        headers={'Range':f'bytes={offset}-'} if offset else {}
        # Public files only; no credentials or signed redirect URLs are saved or printed.
        start=time.monotonic();last=0
        response=sess.get(BASE+name+'?download=true',headers=headers,stream=True,timeout=(15,60))
        response.raise_for_status()
        if offset:assert response.status_code==206,'Server did not honor resume; partial preserved.'
        with partial.open('ab' if offset else 'xb') as f:
            for chunk in response.iter_content(1048576):
                if chunk:f.write(chunk)
                size=f.tell()
                assert size<=650*1024**2
                assert time.monotonic()-start<1200,'Download time ceiling; partial retained'
                if size-last>=32*1024**2:
                    print(json.dumps({'file':name,'downloaded_MiB':round(size/1024**2,1)}),flush=True);last=size
        if name.endswith('.bin'):assert digest(partial)==SHA,'Public SHA mismatch; partial preserved'
        else:json.loads(partial.read_text(encoding='utf-8'))
        partial.rename(dst)
        receipts.append({'file':name,'bytes':dst.stat().st_size,'sha256':digest(dst),'reused':False})
    total=sum(p.stat().st_size for p in MODEL.rglob('*') if p.is_file());assert total<=650*1024**2
    rec={'task':'CV-002','repo':'openai/clip-vit-base-patch32','revision':REV,'files':receipts,'total_bytes':total,'model_forward_calls':0,'training':0,'model_api_calls':0}
    dest=OUT/'模型获取回执.json'
    if not dest.exists():new_json(dest,rec)
    print(json.dumps({'acquired':True,'files':len(receipts),'total_bytes':total}),flush=True)
if __name__=='__main__':
    try:main()
    except requests.RequestException as exc:
        # Avoid printing tokenized CDN redirect URLs embedded in exception strings.
        print(json.dumps({'download_failed':type(exc).__name__,'partial_retained':True}),flush=True)
        raise SystemExit(1)
