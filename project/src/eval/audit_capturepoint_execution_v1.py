"""Independent saved-output equations; engineering may re-decode only the same16 fit images."""
import argparse,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG'
ENG=BASE/'同任务对齐执行器工程_v1';RUN=BASE/'同任务拍摄点对齐执行_v1';DESIGN=BASE/'同任务对齐训练与目标对照协议_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(entries):
    for e in entries:assert sha(ROOT/e['path'])==e['sha256'],e['path']
def unit(v):
    x=np.asarray(v,np.float64);n=np.sqrt(np.sum(x*x,axis=-1,keepdims=True));assert np.isfinite(x).all() and (n>1e-12).all();return x/n
def projection(g,w):
    x=g+np.dot(np.dot(g,w['V'].astype(np.float64).T),w['U'].astype(np.float64).T)
    return unit(x)
def event_rows(out):return [json.loads(s) for s in (out/'事件.jsonl').read_text(encoding='utf-8').splitlines()]
def check_bind():
    b=read(ENG/'首forward输入源码绑定.json')
    for k in ['sources','inputs','models','baselines','protected']:verify(b[k])
    verify([b['engineering_scope']]);return b


def audit_engineering():
    from PIL import Image,ImageOps
    b=check_bind();p=read(DESIGN/'同任务对齐训练冻结协议.json');q=read(DESIGN/'固定编码与训练开发队列.json')
    result=read(ENG/'工程结果.json');assert result['engineering_passed'] and result['real_training_steps']==result['development_rankings']==0
    events=event_rows(ENG);assert [e['event'] for e in events]==['encoder_attempt','encoder_saved']*2
    assert [e['kind'] for e in events]==['ground','ground','aerial','aerial']
    config=read(ROOT/'models/clip_vit_b32_public_v1/preprocessor_config.json');pixels_verified=0
    for kind in ['ground','aerial']:
        before=read(ENG/'核验'/f'{kind}_编码前.json');after=read(ENG/'核验'/f'{kind}_编码后.json')
        assert before['tensor_sha']==after['tensor_sha']==after['before_sha'] and before['parameters']==87849216
        assert before['requires_grad_count']==0 and after['all_frozen'] and after['no_gradients']
        assert after['encoder_inputs']==8 and after['encoder_calls']==1 and after['peak_CUDA_allocated_bytes']<=2*1024**3
        values=np.load(ENG/'特征'/f'{kind}_first8.npy',allow_pickle=False)
        assert values.shape==(8,512) and values.dtype==np.float32 and np.isfinite(values).all() and (np.linalg.norm(values,axis=1)>1e-12).all()
        receipts=read(ENG/'核验'/f'{kind}_像素回执.json')['receipts'];assert len(receipts)==8
        for i,(e,r) in enumerate(zip(q[kind+'_encoder_inputs'][:8],receipts)):
            assert r['index']==i and r['path']==e['path'] and r['raw_sha256']==e['sha256'] and r['sanitized_metadata_fields']==0
            data=(ROOT/e['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==e['sha256']
            import io
            with Image.open(io.BytesIO(data)) as src:im=(ImageOps.exif_transpose(src) if kind=='ground' else src).convert('RGB')
            if kind=='ground':
                w,h=im.size;size=config['size']['shortest_edge'] if isinstance(config['size'],dict) else config['size']
                nw,nh=(size,int(size*h/w)) if w<h else (int(size*w/h),size)
                im=im.resize((nw,nh),Image.Resampling.BICUBIC);x,y=(nw-224)//2,(nh-224)//2
                im=im.crop((x,y,x+224,y+224));mean=config['image_mean'];std=config['image_std']
            else:im=im.resize((224,224),Image.Resampling.BICUBIC);mean=[.3670,.3827,.3338];std=[.2209,.1975,.1988]
            a=(np.asarray(im,np.float32)/np.float32(255)-np.array(mean,np.float32))/np.array(std,np.float32)
            a=np.ascontiguousarray(a.transpose(2,0,1));assert hashlib.sha256(a.tobytes()).hexdigest()==r['pixel_tensor_sha256'];pixels_verified+=1
    g=unit(np.load(ENG/'特征/ground_first8.npy',allow_pickle=False));saved=np.load(ENG/'头算子工程输出.npz',allow_pickle=False)
    maximum=0.
    for arm in ['Identity','SwissRank8Frozen']:
        w=np.load(ROOT/p['baselines'][arm]['weight']['path'],allow_pickle=False);actual=projection(g,w)
        maximum=max(maximum,float(np.abs(actual-saved[arm]).max()));assert maximum<=1e-12
    initial=np.load(ENG/'初始头.npz',allow_pickle=False);identity=np.load(ROOT/p['baselines']['Identity']['weight']['path'],allow_pickle=False)
    assert all(np.array_equal(initial[k],identity[k]) for k in ['U','V'])
    synthetic=read(ENG/'核验/合成梯度与恢复.json')
    assert synthetic['passed'] and synthetic['resume_parameters_and_optimizer_bitwise_equal'] and synthetic['optimizer_contains_only_head']
    assert synthetic['analytic_gradient_max_abs']<2e-7 and synthetic['real_training_updates']==0
    return {'stage':'engineering','passed':True,'encoder_inputs':16,'encoder_calls':2,'independent_pixel_reconstructions':pixels_verified,
        'head_operator_max_abs_error':maximum,'raw_audit_decodes':16,'audit_encoder_calls':0,'real_training_updates':0,
        'development_rankings':0,'confirmation_pixels':0,'independent_person_review':False,
        'method':'same Agent alternate PIL preprocessing hashes/NumPy residual equations and saved-state/permission checks'}


def audit_full():
    b=check_bind();binding=read(RUN/'首forward绑定.json');verify([binding['engineering_seal'],binding['protocol'],binding['queue'],binding['authorization']]+binding['engineering_shards'])
    protocol=read(DESIGN/'同任务对齐训练冻结协议.json');queue=read(DESIGN/'固定编码与训练开发队列.json')
    completed=read(RUN/'执行完成.json');assert completed['completed'] and completed['real_training_steps']==500 and completed['rankings']==1250
    assert completed['aggregate_encoder_inputs']==1148 and completed['aggregate_encoder_calls']==144 and completed['reused_engineering_inputs']==16
    events=event_rows(RUN)
    for kind,total in [('ground',14),('aerial',1134)]:
        saved=[e for e in events if e['event']=='encoder_saved' and e['kind']==kind]
        attempted=[e for e in events if e['event']=='encoder_attempt' and e['kind']==kind]
        assert [e['indices'] for e in saved]==[e['indices'] for e in attempted]
        assert [i for e in saved for i in e['indices']]==list(range(8,total))
        before=read(RUN/'核验'/f'{kind}_编码前.json');after=read(RUN/'核验'/f'{kind}_编码后.json')
        assert before['tensor_sha']==after['tensor_sha']==read(ENG/'核验'/f'{kind}_编码前.json')['tensor_sha']
        assert after['all_frozen'] and after['no_gradients'] and after['peak_CUDA_allocated_bytes']<=2*1024**3
        f=np.load(RUN/'特征'/f'{kind}.npy',allow_pickle=False)
        assert f.shape==(total,512) and np.isfinite(f).all() and np.array_equal(f[:8],np.load(ENG/'特征'/f'{kind}_first8.npy',allow_pickle=False))
        receipts=read(RUN/'核验'/f'{kind}_像素回执.json')['receipts'];assert len(receipts)==total-8
        for i,r in zip(range(8,total),receipts):
            e=queue[kind+'_encoder_inputs'][i];assert r['index']==i and r['raw_sha256']==e['sha256'] and r['path']==e['path']
    assert len([e for e in events if e['event']=='encoder_saved'])==142
    for label in ['optimizer_attempt','optimizer_saved']:assert [e['step'] for e in events if e['event']==label]==list(range(1,501))
    features=read(RUN/'首更新前特征绑定.json');verify(features['features']+[features['initial_head'],features['protocol']])
    assert features['fit_queries']==list(range(9)) and features['positive_indices_evaluator_only']==[40]*9
    assert features['fit_candidate_indices']==[list(range(i*81,(i+1)*81)) for i in range(9)]
    g=unit(np.load(RUN/'特征/ground.npy',allow_pickle=False));a=unit(np.load(RUN/'特征/aerial.npy',allow_pickle=False))
    for key,x in [('ground',g[:9].astype(np.float32)),('aerial',a[np.array(features['fit_candidate_indices'])].astype(np.float32)),('positive',np.full(9,40,np.int64))]:
        assert hashlib.sha256(x.tobytes()).hexdigest()==features[key+'_tensor_sha256']
    weights={arm:np.load(ROOT/protocol['baselines'][arm]['weight']['path'],allow_pickle=False) for arm in ['Identity','SwissRank8Frozen']}
    weights['MMRank8Final']=np.load(RUN/'权重/final.npz',allow_pickle=False)
    means=unit(g[:9].mean(0));tasks={t['pair_id']:t for t in queue['development_tasks']};rows=read(RUN/'逐题排序结果.json')['rows']
    assert len(rows)==1250;lookup={};max_error=0.
    for r in rows:
        t=tasks[r['pair_id']];arm=r['arm'];cond=r['condition'];assert r['group']==t['group_evaluator_only']
        if arm=='SharedDisabled':assert cond=='Disabled';scores=np.zeros(25,np.float64)
        else:
            target=g[t['ground_feature_index']] if cond=='Full' else means if cond=='Mean' else g[t['wrong_ground_feature_index']]
            target=projection(target,weights[arm]);scores=np.sum(a[t['aerial_feature_indices']]*target,axis=1)
        max_error=max(max_error,float(np.abs(scores-r['scores']).max()));assert max_error<=1e-12
        order=sorted(range(25),key=lambda j:(-float(scores[j]),hashlib.sha256(f"CV-002/tie/v1:{r['pair_id']}:{j}".encode()).hexdigest()))
        pos=order.index(t['goal_index_evaluator_only'])+1
        assert r['ranking']==order and r['true_rank']==pos and r['top1']==(pos==1) and r['top5']==(pos<=5)
        key=(r['pair_id'],arm,cond);assert key not in lookup;lookup[key]=r
    expected={(t['pair_id'],arm,c) for t in tasks.values() for arm in ['Identity','SwissRank8Frozen','MMRank8Final'] for c in ['Full','Mean','Wrong']}
    expected|={(t['pair_id'],'SharedDisabled','Disabled') for t in tasks.values()};assert set(lookup)==expected
    summary=read(RUN/'开发对照汇总.json');groups=sorted({r['group'] for r in rows})
    for arm,conditions in summary['counts'].items():
        for c,v in conditions.items():
            selected=[r for r in rows if r['arm']==arm and r['condition']==c]
            assert v=={'planned':125,'top1':sum(r['top1'] for r in selected),'top5':sum(r['top5'] for r in selected),'median_rank':float(np.median([r['true_rank'] for r in selected]))}
            for group in groups:
                s=[r for r in selected if r['group']==group]
                assert summary['per_source'][group][arm][c]=={'planned':25,'top1':sum(r['top1'] for r in s),'top5':sum(r['top5'] for r in s)}
    for name,values in summary['contrasts'].items():
        if name.startswith('MM_vs_'):
            rest=name[len('MM_vs_'):];arm,cond=rest.rsplit('_',1);t=('MMRank8Final',cond);c=(arm,cond)
        else:
            arm,cond=name.split('_Full_vs_');t=(arm,'Full');c=('SharedDisabled','Disabled') if cond=='Disabled' else (arm,cond)
        for metric,reported in values.items():
            recovered=harmed=0;deltas=[]
            for group in groups:
                ids=[pid for pid,task in tasks.items() if task['group_evaluator_only']==group]
                rr=sum(lookup[pid,*t][metric] and not lookup[pid,*c][metric] for pid in ids)
                hh=sum(lookup[pid,*c][metric] and not lookup[pid,*t][metric] for pid in ids)
                recovered+=rr;harmed+=hh;deltas.append((rr-hh)/25)
            assert reported['recovered']==recovered and reported['harmed']==harmed and reported['net']==recovered-harmed
            assert reported['per_source_rate_differences']==dict(zip(groups,deltas)) and reported['resamples']==3125 and reported['source_units']==5
            samples=[math.fsum(deltas[i] for i in ids)/5 for ids in itertools.product(range(5),repeat=5)]
            np.testing.assert_allclose(reported['paired_source_percentile95'],np.quantile(samples,[.025,.975]),atol=1e-12,rtol=0)
    for group,count in summary['gate_counts'].items():
        assert count['planned']==25 and count['Disabled']==summary['per_source'][group]['SharedDisabled']['Disabled']['top1']
        for arm in ['Identity','SwissRank8Frozen','MMRank8Final']:
            for cond in ['Full','Mean','Wrong']:assert count[arm][cond]==summary['per_source'][group][arm][cond]['top1']
    log=read(RUN/'训练日志.json');assert log['updates']==500 and log['seed']==0 and log['supervision_uses']==4500 and log['trainable_parameters']==8192 and log['encoder_optimizer_parameters']==0
    assert [r['step'] for r in log['log']]==list(range(1,501)) and all(np.isfinite(r['loss_before_update']) and r['gradient_norm_before_clip']>0 for r in log['log'])
    initial=np.load(RUN/'权重/initial.npz',allow_pickle=False);eng_initial=np.load(ENG/'初始头.npz',allow_pickle=False)
    assert all(np.array_equal(initial[k],eng_initial[k]) for k in ['U','V'])
    def loss(w):
        target=projection(g[:9],w);banks=a[np.array(features['fit_candidate_indices'])]
        scores=np.einsum('nd,nkd->nk',target,banks)/.1;maximum=scores.max(axis=1)
        return float((maximum+np.log(np.exp(scores-maximum[:,None]).sum(axis=1))-scores[:,40]).mean())
    assert abs(loss(initial)-log['log'][0]['loss_before_update'])<=2e-5 and abs(loss(weights['MMRank8Final'])-log['final_fit_loss'])<=2e-5
    import torch
    ck=torch.load(RUN/'权重/recovery.pt',map_location='cpu',weights_only=True);assert ck['step']==500
    assert all(np.array_equal(ck['head'][k].numpy(),weights['MMRank8Final'][k]) for k in ['U','V'])
    assert len(ck['optimizer']['state'])==2 and all(int(v['step'])==500 for v in ck['optimizer']['state'].values())
    assert sum(f.stat().st_size for f in (RUN/'权重').iterdir())<=1024**2 and sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())<=40*1024**2
    return {'stage':'full','passed':True,'planned_rankings':1250,'saved_scores':31250,'max_score_error':max_error,'contrast_pairs':len(summary['contrasts']),
        'actual_updates':500,'aggregate_encoder_inputs':1148,'aggregate_encoder_calls':144,'auditor_model_forward_calls':0,'auditor_training_steps':0,
        'confirmation_pixels':0,'independent_person_review':False,'method':'same Agent alternate residual/cosine/loss/ranking/count/source-bootstrap equations and checkpoints; no new encoder forward'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['engineering','full']);p.add_argument('--save',action='store_true');a=p.parse_args()
    result=audit_engineering() if a.stage=='engineering' else audit_full();out=ENG if a.stage=='engineering' else RUN
    if a.save:
        with (out/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)
