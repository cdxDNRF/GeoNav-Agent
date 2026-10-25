"""CV-008 alternate NumPy equations/rank/count audit. No encoders or new training."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/MM-GAG'
OUT=BASE/'弱场景对齐头工程与训练_v1'
PREV=BASE/'配对派生清单与弱目标准入_v1'
MM=BASE/'跨视角感知最小验证_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    assert np.isfinite(x).all()
    norms=np.sqrt(np.sum(x*x,axis=-1,keepdims=True));assert (norms>1e-12).all()
    return x/norms


def project(x,weights):
    # Separate matrix orientation and normalization from production Torch head.
    return unit(x+weights['U'].astype(np.float64).dot(weights['V'].astype(np.float64).dot(x)))


def fit_loss(g,a,truth,weights):
    h=np.stack([project(x,weights) for x in g]);scores=np.einsum('nd,nkd->nk',h,a)/.1
    maximum=scores.max(axis=1);lse=maximum+np.log(np.exp(scores-maximum[:,None]).sum(axis=1))
    return float((lse-scores[np.arange(len(g)),truth]).mean())


def audit(save=False):
    binding=read(OUT/'首forward前绑定.json');cache={};nchecks=0
    def verify(entries):
        nonlocal nchecks
        for e in entries:
            key=e['path']
            if key not in cache:cache[key]=digest(ROOT/key)
            assert cache[key]==e['sha256'],key;nchecks+=1
    for k in ['files','sources','models','protected','previous_seals','raw_inputs']:verify(binding[k])
    prior=0
    for e in binding['previous_seals']:
        s=read(ROOT/e['path']);verify(s['files']+s['sources']);prior+=len(s['files'])+len(s['sources'])
    admitted=read(OUT/'首更新前特征与训练绑定.json')
    verify(admitted['features']+[admitted['initial_head'],admitted['source_binding']]+admitted['engineering'])
    state=read(OUT/'执行状态.json');assert state['state']=='completed' and state['training_steps']==500
    assert state['attempted_encoder_inputs']==1148 and state['attempted_encoder_calls']==state['successful_encoder_calls']==144
    assert state['reserved_groups_consumed']==state['external_api_calls']==state['navigation_episodes']==0
    assert len(state['forward_ledger'])==144 and state['pending_forward'] is None
    for kind,total in [('ground',14),('aerial',1134)]:
        ledger=[r for r in state['forward_ledger'] if r['kind']==kind]
        assert [r['offset'] for r in ledger]==list(range(0,total,8))
        assert [r['count'] for r in ledger]==[min(8,total-i) for i in range(0,total,8)]
    frozen=read(OUT/'编码器冻结与像素证据.json')
    for encoder in frozen['encoders']:
        assert encoder['tensor_sha_before']==encoder['tensor_sha_after'] and encoder['no_gradients'] and encoder['all_frozen']
        assert encoder['parameters']==87849216 and encoder['peak_cuda_allocated_bytes']<=2*1024**3
        assert encoder['engineering_first_batch']['identity_max_abs']<=1e-6
    assert read(OUT/'核验/合成工程.json')['resume_parameters_and_optimizer_bitwise_equal']
    assert read(OUT/'核验/旧缓存复现.json')['maximum_score_error']<=1e-6
    bank=read(PREV/'唯一训练查询与图块引用.json');design=read(PREV/'训练设计冻结.json')
    ground=np.load(OUT/'特征/ground.npy',allow_pickle=False);air=np.load(OUT/'特征/aerial.npy',allow_pickle=False)
    assert ground.shape==(14,512) and air.shape==(1134,512) and ground.dtype==air.dtype==np.float32
    for enc in frozen['encoders']:
        data=ground if enc['kind']=='ground' else air
        assert len(enc['receipts'])==len(data)
        for i,receipt in enumerate(enc['receipts']):
            assert receipt['index']==i and hashlib.sha256(data[i].tobytes()).hexdigest()==receipt['feature_sha256']
    # Alternate PIL preprocessing for all1148 inputs, no model invocation.
    from PIL import Image,ImageOps
    processor=read(ROOT/'models/clip_vit_b32_public_v1/preprocessor_config.json')
    for kind,items in [('ground',bank['queries']),('aerial',bank['cells'])]:
        receipts=next(e['receipts'] for e in frozen['encoders'] if e['kind']==kind)
        for i,item in enumerate(items):
            if kind=='ground':
                with Image.open(ROOT/item['ground_raw_path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
                w,h=im.size
                size=processor['size'];size=size['shortest_edge'] if isinstance(size,dict) else size
                nw,nh=(size,int(size*h/w)) if w<h else (int(size*w/h),size)
                im=im.resize((nw,nh),Image.Resampling.BICUBIC)
                im=im.crop(((nw-224)//2,(nh-224)//2,(nw-224)//2+224,(nh-224)//2+224))
                mean=processor['image_mean'];std=processor['image_std']
            else:
                with Image.open(ROOT/item['raw_aerial_path']) as src:im=src.convert('RGB').crop(tuple(item['crop_box']))
                im=im.resize((224,224),Image.Resampling.BICUBIC);mean=[.3670,.3827,.3338];std=[.2209,.1975,.1988]
            arr=(np.asarray(im,dtype=np.float32)/np.float32(255)-np.array(mean,np.float32))/np.array(std,np.float32)
            arr=np.ascontiguousarray(arr.transpose(2,0,1))
            assert hashlib.sha256(arr.tobytes()).hexdigest()==receipts[i]['pixel_tensor_sha256']
    g=unit(ground);a=unit(air);mg=unit(np.load(MM/'地面特征.npy',allow_pickle=False));ma=unit(np.load(MM/'航拍特征.npy',allow_pickle=False))
    qi={q['scene']:i for i,q in enumerate(bank['queries'])}
    assert admitted['training_sources']==design['training']['source_groups']
    expected_fit=[qi[s] for s in design['training']['source_groups']]
    assert admitted['fit_queries']==expected_fit and all(bank['queries'][i]['role']=='fit' for i in expected_fit)
    candidate=[];labels=[]
    for scene in design['training']['source_groups']:
        ids=[i for i,c in enumerate(bank['cells']) if c['scene']==scene and c['role']=='fit'];assert len(ids)==81
        candidate.append(ids);labels.append(next(j for j,i in enumerate(ids) if bank['cells'][i]['cell_sha256']==bank['queries'][qi[scene]]['positive_cell_sha256']))
    assert candidate==admitted['fit_candidate_indices'] and labels==admitted['positive_indices_evaluator_only']
    for key,arr in [('ground',g[expected_fit].astype(np.float32)),('aerial',a[np.array(candidate)].astype(np.float32)),('positive',np.array(labels,np.int64))]:
        assert hashlib.sha256(arr.tobytes()).hexdigest()==admitted[key+'_tensor_sha256']
    weights={arm:{k:v.copy() for k,v in np.load(OUT/'权重'/f,allow_pickle=False).items()} for arm,f in [('Identity','initial.npz'),('Rank8Final','final.npz')]}
    for v in weights.values():assert set(v)=={'U','V'} and v['U'].shape==(512,8) and v['V'].shape==(8,512)
    assert (weights['Identity']['U']==0).all()
    log=read(OUT/'训练日志.json');assert [r['step'] for r in log['log']]==list(range(1,501))
    assert log['supervision_uses']==4500 and log['unique_fit_queries']==9 and log['encoder_optimizer_parameters']==0
    assert log['seconds']<=1800 and all(np.isfinite(r['loss_before_update']) and np.isfinite(r['gradient_norm_before_clip']) and r['gradient_norm_before_clip']>0 for r in log['log'])
    assert abs(fit_loss(g[expected_fit],a[np.array(candidate)],labels,weights['Identity'])-log['log'][0]['loss_before_update'])<2e-5
    assert abs(fit_loss(g[expected_fit],a[np.array(candidate)],labels,weights['Rank8Final'])-log['final_fit_loss'])<2e-5
    import torch
    recovery=torch.load(OUT/'权重/recovery.pt',map_location='cpu',weights_only=True)
    assert recovery['step']==500 and set(recovery['head'])=={'U','V'}
    assert all(np.array_equal(v,recovery['head'][k].numpy()) for k,v in weights['Rank8Final'].items())
    assert len(recovery['optimizer']['state'])==2 and all(float(v['step'])==500 for v in recovery['optimizer']['state'].values())
    # Build each25-cell window independently using original CV-006 pixel hashes.
    inventory=read(BASE/'对齐头训练素材准入_v1/逐对素材核验.json')['records']
    by_original={r['id']:r for r in inventory};by_hash={(c['scene'],c['cell_sha256']):i for i,c in enumerate(bank['cells'])}
    expected={};mm_pairs=read(MM/'冻结执行任务.json')['pairs'];mi={p['pair_id']:i for i,p in enumerate(mm_pairs)}
    dev=design['evaluation']['weak_dev_groups']
    for row in read(PREV/'派生素材清单.json')['records']:
        if row['role']!='development':continue
        cells=by_original[row['original_id']]['images']['aerial']['cells']
        ids=[by_hash[row['scene'],x['sha256']] for x in cells]
        expected['weak_'+row['original_id']]=('weak_dev',row['scene'],qi[row['scene']],qi[dev[(dev.index(row['scene'])+1)%5]],ids,row['positive_index'])
    for i,p in enumerate(mm_pairs):expected[p['pair_id']]=('MMGAG',p['group_id'],i,mi[p['wrong_target_pair_id']],list(range(i*25,(i+1)*25)),p['goal_index_evaluator_only'])
    assert len(expected)==141
    for t in read(OUT/'冻结评测任务.json')['tasks']:
        assert expected[t['pair_id']]==(t['dataset'],t['group'],t['query'],t['wrong_query'],t['candidates'],t['truth'])
    rows=read(OUT/'逐题排序结果.json')['rows'];assert len(rows)==846
    actual={};maximum=0.;means={'weak_dev':unit(g[expected_fit].mean(0)),'MMGAG':unit(mg.mean(0))}
    for row in rows:
        ds,group,q,w,ids,truth=expected[row['pair_id']];gg,aa=(g,a) if ds=='weak_dev' else (mg,ma)
        assert row['dataset']==ds and row['group']==group and row['truth_evaluator_only']==truth
        target=gg[q] if row['condition']=='Full' else means[ds] if row['condition']=='Mean' else gg[w]
        target=project(target,weights[row['arm']])
        scores=np.sum(aa[ids]*target[None,:],axis=1)
        order=sorted(range(25),key=lambda j:(-float(scores[j]),hashlib.sha256(('CV-002/tie/v1:'+row['pair_id']+':'+str(j)).encode()).hexdigest()))
        maximum=max(maximum,float(np.max(np.abs(scores-row['scores']))));assert maximum<1e-12
        pos=order.index(truth)+1;assert order==row['ranking'] and pos==row['true_rank'] and row['top1']==(pos==1) and row['top5']==(pos<=5)
        key=(ds,row['arm'],row['condition'],row['pair_id']);assert key not in actual;actual[key]=(pos,group)
    # Counts/contrasts/gates independently counted from recomputed ranks.
    summary=read(OUT/'感知对照汇总.json')
    for ds in ['MMGAG','weak_dev']:
        for arm in ['Identity','Rank8Final']:
            for cond in ['Full','Mean','Wrong']:
                selected=[v for (d,b,c,p),v in actual.items() if (d,b,c)==(ds,arm,cond)]
                report=summary['counts'][ds][arm][cond]
                assert report=={'planned':len(selected),'top1':sum(x[0]==1 for x in selected),'top5':sum(x[0]<=5 for x in selected),'median_true_rank':float(np.median([x[0] for x in selected]))}
                for group in {x[1] for x in selected}:
                    subset=[x for x in selected if x[1]==group]
                    assert summary['per_source'][ds][group][arm][cond]=={'planned':len(subset),'top1':sum(x[0]==1 for x in subset),'top5':sum(x[0]<=5 for x in subset)}
        for name,report in summary['contrasts'][ds].items():
            if name.startswith('Final_vs_Identity_'):t=('Rank8Final',name.removeprefix('Final_vs_Identity_'));c=('Identity',t[1])
            else:
                arm,cond=name.split('_Full_vs_');t=(arm,'Full');c=(arm,cond)
            ids=[p for p,e in expected.items() if e[0]==ds]
            for label,k in [('top1',1),('top5',5)]:
                recovered=sum(actual[(ds,*t,p)][0]<=k<actual[(ds,*c,p)][0] for p in ids)
                harmed=sum(actual[(ds,*c,p)][0]<=k<actual[(ds,*t,p)][0] for p in ids)
                assert report[label]=={'recovered':recovered,'harmed':harmed,'net':recovered-harmed}
    m=summary['counts']['MMGAG']['Rank8Final'];positive_mm=sum(x['Rank8Final']['Full']['top1']>x['Rank8Final']['Mean']['top1'] and x['Rank8Final']['Full']['top1']>x['Rank8Final']['Wrong']['top1'] for x in summary['per_source']['MMGAG'].values())
    gains={group:(r['Rank8Final']['Full']['top1']-r['Identity']['Full']['top1'])/25 for group,r in summary['per_source']['weak_dev'].items()}
    assert summary['weak_source_top1_gains']==gains and summary['MMGAG_groups_positive_against_both']==positive_mm
    decisions=[m['Full']['planned']==16,m['Full']['top1']>=4,m['Full']['top1']-m['Mean']['top1']>=2,m['Full']['top1']-m['Wrong']['top1']>=2,positive_mm>=3,sum(gains.values())/5>=.04-1e-12,sum(v>0 for v in gains.values())>=3]
    assert list(summary['effect_gates'].values())==decisions and summary['effect_gates_passed']==all(decisions)
    assert sum(p.stat().st_size for p in (OUT/'权重').iterdir())<=1024**2
    assert sum(p.stat().st_size for p in (OUT/'特征').iterdir())<=32*1024**2
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=40*1024**2
    result={'task':'CV-008','passed':True,'binding_checks':nchecks,'prior_sealed_items':prior,'alternate_pixel_tensors':1148,
            'alternate_rankings':846,'alternate_cosine_scores':846*25,'maximum_score_error':maximum,
            'losses_and_training_scope_verified':True,'serialized_final_optimizer_step':500,'reserved_groups_consumed':0,
            'auditor_encoder_forward_calls':0,'auditor_training_updates':0,'independent_person_review':False,
            'method':'same Agent, separate NumPy projection/loss, PIL pixel reconstruction, original inventory window lookup, ranking/count equations'}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
