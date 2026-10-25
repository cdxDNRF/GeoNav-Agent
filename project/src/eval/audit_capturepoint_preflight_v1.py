"""CV-011 alternate saved-thumbnail axis/quality audit. Never decodes raw images."""
import argparse,ast,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'同任务首批图像轴向预检_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    assert not Path(p).resolve().is_relative_to((ROOT/'DATA/raw_data').resolve())
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(save=False):
    b=read(OUT/'首像素输入源码绑定.json');checked=0
    for e in b['sources']+b['inputs']+b['protected']+[b[k] for k in ['protocol','approval','role_snapshot']]:assert sha(ROOT/e['path'])==e['sha256'];checked+=1
    for e in b['sources']:
        for n in ast.walk(ast.parse((ROOT/e['path']).read_text(encoding='utf-8-sig'))):
            if isinstance(n,ast.Import):assert not any(a.name.split('.')[0] in {'torch','transformers','safetensors','requests','httpx','subprocess'} for a in n.names)
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in {'torch','transformers','safetensors'}
    p=read(OUT/'首像素冻结协议.json');rows=read(OUT/'工程汇总.json')['sources'];r=read(OUT/'逐图解码与字节回执.json')['receipts']
    assert len(r)==len(p['raw_expected'])==1162 and len(rows)==14
    assert [e['path'] for e in r]==[e['path'] for e in p['raw_expected']]
    for actual,expected in zip(r,p['raw_expected']):
        assert actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes'] and actual['metadata_keys_after']==0 and actual['output_mode']=='RGB'
    log=[json.loads(x) for x in (OUT/'输入消费.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(log)==2324 and [e['event'] for e in log]==['attempt','decoded']*1162
    assert [e['index'] for e in log[::2]]==list(range(1162)) and [e['path'] for e in log[1::2]]==[e['path'] for e in r]
    g=np.load(OUT/'地面缩略.npy',allow_pickle=False);t=np.load(OUT/'图块缩略.npy',allow_pickle=False);c=np.load(OUT/'拼图探针.npy',allow_pickle=False)
    assert (g.shape,t.shape,c.shape)==((14,16,16,3),(14,81,16,16,3),(14,8,16,16,3))
    transforms=[('row',lambda y,x:(y,x)),('column',lambda y,x:(x,y)),('flip_row',lambda y,x:(10-y,x)),
        ('flip_column',lambda y,x:(y,10-x)),('rotate180',lambda y,x:(10-y,10-x)),('rotate90',lambda y,x:(x,10-y)),
        ('rotate270',lambda y,x:(10-x,y)),('antitranspose',lambda y,x:(10-x,10-y))]
    for i,s in enumerate(p['sources']):
        output=rows[i];assert s['source_id']==output['source_id'] and s['role']==output['role']
        scores={}
        for name,fn in transforms:
            error=0
            for j,(y,x) in enumerate(p['axis_probes']):
                yy,xx=fn(y,x);delta=t[i,(yy-1)*9+xx-1].astype(np.int16)-c[i,j].astype(np.int16)
                error+=int(np.abs(delta).sum())/(16*16*3*8)
            scores[name]=error;assert abs(error-output['axis']['scores'][name])<1e-10
        ordered=sorted(scores,key=lambda n:(scores[n],n));margin=scores[ordered[1]]-scores[ordered[0]]
        gstd=float(np.std(g[i].astype(np.float64).reshape(-1,3),axis=0).max());cstd=float(np.std(t[i,40].astype(np.float64).reshape(-1,3),axis=0).max())
        assert output['axis']['best']==ordered[0] and abs(output['axis']['margin']-margin)<1e-10
        assert abs(output['ground_std16_max']-gstd)<1e-10 and abs(output['center_std16_max']-cstd)<1e-10
        passed=ordered[0]=='row' and scores['row']<=8 and margin>=3 and gstd>1 and cstd>1
        assert output['passed']==passed and output['input_only_two_readonly_RGB']
        for j in range(81):assert r[i*83+1+j]['original_size']==[256,256]
        assert r[i*83+82]['original_size']==[2816,2816]
    summary=read(OUT/'工程汇总.json');assert summary['all_required_passed']==all(x['passed'] for x in rows)
    ledger=read(OUT/'用途激活快照.json');recs=ledger['records'];assert len(recs)==65
    assert sum(x['role']=='known_development' for x in recs)==23
    assert sum(x['role']=='fit_preparation' for x in recs)==12 and sum(x['role']=='development' for x in recs)==6 and sum(x['role']=='confirmation_reserved' for x in recs)==24
    permitted={s['source_id'] for s in p['sources']};assert len(permitted)==14
    assert not permitted&{x['source_id'] for x in recs if x['role']=='confirmation_reserved'}
    for k in ['training_steps','model_calls','new_predictions','navigation_episodes','confirmation_pixels','API_calls','downloads']:assert summary[k]==0
    result={'task':'CV-011','passed':True,'binding_checks':checked,'sources':14,'saved_pixel_axis_scores':14*8,'raw_hash_receipts':1162,
        'independent_raw_decodes':0,'method':'same Agent, alternate int16 absolute-sum axis/quality/counts over saved lossless BOX16 thumbnails; no independent raw image conversion',
        'independent_person_review':False,'model_calls':0,'confirmation_pixels':0}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
