"""CV-019 alternate residual/loss/count/bootstrap audit, without encoder forward/raw reads."""
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from statistics import median
import numpy as np

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'配对覆盖同预算执行_v1';OLD=BASE/'同任务拍摄点对齐执行_v1';DESIGN=BASE/'三新增来源工程与配对覆盖协议_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def verify(items):
    for e in items:assert sha(ROOT/e['path'])==e['sha256'],e['path']


def unit(x):
    x=np.asarray(x,np.float64);n=np.sqrt((x*x).sum(axis=-1,keepdims=True));assert np.isfinite(x).all() and (n>1e-12).all();return x/n


def project(g,w):return unit(g+np.dot(np.dot(g,w['V'].astype(np.float64).T),w['U'].astype(np.float64).T))


def arrays():
    plan=read(OUT/'特征复用计划.json');og=np.load(OLD/'特征/ground.npy',allow_pickle=False);oa=np.load(OLD/'特征/aerial.npy',allow_pickle=False)
    ng=np.load(OUT/'特征/new_ground.npy',allow_pickle=False);na=np.load(OUT/'特征/new_aerial.npy',allow_pickle=False)
    assert og.shape==(14,512) and oa.shape==(1134,512) and ng.shape==(3,512) and na.shape==(208,512)
    rows=[]
    for e in plan['aerial_assignment']:
        assert e['global_index']==1134+len(rows);rows.append(oa[e['index']] if e['source']=='old' else na[e['index']])
    return np.vstack([og,ng]),np.vstack([oa,np.stack(rows)])


def bindings():
    b=read(OUT/'首forward绑定.json')
    for key in ('sources','dependencies','inputs','models','protected','old_features','old_heads'):verify(b[key])
    verify([b['authorization'],b['reuse_plan']]);return b


def engineering():
    bindings();q=read(DESIGN/'固定训练与开发队列.json');plan=read(OUT/'特征复用计划.json');report=read(OUT/'工程结果.json')
    assert report['engineering_passed'] and report['real_training_steps']==0 and report['development_rankings']==0
    old_by_sha={}
    for i,e in enumerate(q['aerial_encoder_inputs'][:1134]):old_by_sha.setdefault(e['sha256'],i)
    novel=[];novel_by_sha={};reused=0
    for i,(item,assignment) in enumerate(zip(q['aerial_encoder_inputs'][1134:],plan['aerial_assignment']),1134):
        if item['sha256'] in old_by_sha:
            assert assignment['source']=='old' and assignment['index']==old_by_sha[item['sha256']];reused+=1
        else:
            if item['sha256'] not in novel_by_sha:novel_by_sha[item['sha256']]=len(novel);novel.append(item)
            assert assignment['source']=='new' and assignment['index']==novel_by_sha[item['sha256']]
        assert assignment['raw_sha256']==item['sha256'] and assignment['global_index']==i
    assert reused==35 and len(novel)==208 and novel==plan['new_aerial_items'] and plan['new_ground_items']==q['ground_encoder_inputs'][14:]
    events=[json.loads(x) for x in (OUT/'事件.jsonl').read_text(encoding='utf-8').splitlines()]
    for kind,items in [('ground',plan['new_ground_items']),('aerial',novel)]:
        before=read(OUT/'核验'/f'{kind}_编码前.json');after=read(OUT/'核验'/f'{kind}_编码后.json')
        assert before['tensor_sha']==after['tensor_sha']==after['old_tensor_sha']==read(OLD/'核验'/f'{kind}_编码后.json')['tensor_sha']
        assert before['parameters']==87849216 and before['requires_grad_count']==0 and after['all_frozen'] and after['no_gradients'] and after['peak_CUDA_allocated_bytes']<=2*1024**3
        attempts=[e['indices'] for e in events if e['event']=='encoder_attempt' and e['kind']==kind]
        saved=[e['indices'] for e in events if e['event']=='encoder_saved' and e['kind']==kind]
        assert attempts==saved and [i for batch in saved for i in batch]==list(range(len(items)))
        receipts=read(OUT/'核验'/f'{kind}_像素回执.json')['receipts'];assert len(receipts)==len(items)
        assert [e['index'] for e in events if e['event']=='pixel_attempt' and e['kind']==kind]==list(range(len(items)))
        for i,(item,r) in enumerate(zip(items,receipts)):assert r['index']==i and r['path']==item['path'] and r['raw_sha256']==item['sha256'] and r['sanitized_metadata_fields']==0
    g,a=arrays();assert np.isfinite(g).all() and np.isfinite(a).all()
    verify(report['new_features']);initial=np.load(OUT/'权重/initial.npz',allow_pickle=False);old_initial=np.load(OLD/'权重/initial.npz',allow_pickle=False)
    assert all(np.array_equal(initial[k],old_initial[k]) for k in ('U','V')) and report['fit_indices']==list(range(9))+[14,15,16]
    operators=np.load(OUT/'核验/真实头算子输出.npz',allow_pickle=False);oldp=read(BASE/'同任务对齐训练与目标对照协议_v1/同任务对齐训练冻结协议.json');maximum=0
    for arm in ('Identity','SwissRank8Frozen'):
        weights=np.load(ROOT/oldp['baselines'][arm]['weight']['path'],allow_pickle=False)
        maximum=max(maximum,float(np.abs(project(unit(g[14:]),weights)-operators[arm]).max()))
    assert maximum<1e-12 and read(OUT/'核验/合成梯度与恢复.json')['sampled_resume_bitwise_equal']
    return {'stage':'engineering','passed':True,'new_encoder_inputs':211,'new_encoder_calls':27,'verified_old_cache_matches':35,
        'head_operator_max_error':maximum,'fit_global_indices':list(range(9))+[14,15,16],'raw_audit_reads':0,'auditor_encoder_forward_calls':0,
        'real_training_steps':0,'confirmation_access':0,'independent_person_review':False}


def full():
    bindings();q=read(DESIGN/'固定训练与开发队列.json');g,a=arrays();g=unit(g);a=unit(a)
    fit=read(OUT/'首更新前特征绑定.json');verify(fit['old_features']+fit['new_features']+[fit['initial_head'],fit['protocol']])
    gi=[t['ground_feature_index'] for t in q['training_queries']];ci=[t['aerial_feature_indices'] for t in q['training_queries']]
    assert gi==list(range(9))+[14,15,16] and fit['ground_indices']==gi and fit['candidate_indices']==ci and fit['positive_indices']==[40]*12
    gg=g[gi].astype(np.float32);aa=a[np.array(ci)].astype(np.float32);pp=np.full(12,40,np.int64)
    for key,value in [('ground',gg),('aerial',aa),('positive',pp)]:assert hashlib.sha256(value.tobytes()).hexdigest()==fit[key+'_tensor_sha256']
    mean=unit(g[:9].mean(0));assert hashlib.sha256(mean.tobytes()).hexdigest()==fit['mean9_sha256']
    refs=read(OUT/'合并结果引用.json');verify([refs['old'],refs['new']]);old=read(ROOT/refs['old']['path'])['rows'];new=read(ROOT/refs['new']['path'])['rows'];rows=old+new
    assert len(old)==1250 and len(new)==375 and len(rows)==1625
    weights=np.load(OUT/'权重/final.npz',allow_pickle=False);tasks={t['pair_id']:t for t in q['development_tasks']};maximum=0
    for r in new:
        t=tasks[r['pair_id']];target=g[t['ground_feature_index']] if r['condition']=='Full' else mean if r['condition']=='Mean' else g[t['wrong_ground_feature_index']]
        expected=np.sum(a[t['aerial_feature_indices']]*project(target,weights),axis=1)
        maximum=max(maximum,float(np.abs(expected-r['scores']).max()));assert maximum<1e-12
        assert r['arm']=='MMRank8Coverage12' and r['group']==t['group_evaluator_only'] and r['truth_evaluator_only']==t['goal_index_evaluator_only']
    lookup={};summary=read(OUT/'开发对照汇总.json')
    for r in rows:
        order=sorted(range(25),key=lambda j:(-r['scores'][j],hashlib.sha256(f"CV-002/tie/v1:{r['pair_id']}:{j}".encode()).hexdigest()))
        rank=order.index(r['truth_evaluator_only'])+1
        assert order==r['ranking'] and rank==r['true_rank'] and r['top1']==(rank==1) and r['top5']==(rank<=5)
        key=(r['pair_id'],r['arm'],r['condition']);assert key not in lookup;lookup[key]=r
    assert len(lookup)==1625
    for arm,conditions in summary['counts'].items():
        for condition,counts in conditions.items():
            selected=[r for r in rows if r['arm']==arm and r['condition']==condition];assert len(selected)==125
            assert counts=={'planned':125,'top1':sum(r['top1'] for r in selected),'top5':sum(r['top5'] for r in selected),'median_rank':median(r['true_rank'] for r in selected)}
            for group in summary['per_source']:
                subset=[r for r in selected if r['group']==group];assert len(subset)==25
                assert summary['per_source'][group][arm][condition]=={'planned':25,'top1':sum(r['top1'] for r in subset),'top5':sum(r['top5'] for r in subset)}
                if arm=='SharedDisabled':assert summary['gate_counts'][group]['Disabled']==sum(r['top1'] for r in subset)
                else:assert summary['gate_counts'][group][arm][condition]==sum(r['top1'] for r in subset)
    groups=sorted(summary['per_source']);assert len(groups)==5
    for name,effects in summary['contrasts'].items():
        control=name[len('Full_vs_'):];target=('MMRank8Coverage12','Full')
        other=('MMRank8Coverage12',control) if control in ('Mean','Wrong') else ('SharedDisabled','Disabled') if control=='Disabled' else (control,'Full')
        for metric,reported in effects.items():
            recovery=harm=0;deltas=[]
            for group in groups:
                ids=[p for p,t in tasks.items() if t['group_evaluator_only']==group]
                rr=sum(lookup[p,*target][metric] and not lookup[p,*other][metric] for p in ids)
                hh=sum(lookup[p,*other][metric] and not lookup[p,*target][metric] for p in ids);recovery+=rr;harm+=hh;deltas.append((rr-hh)/25)
            assert reported['recovered']==recovery and reported['harmed']==harm and reported['net']==recovery-harm
            assert reported['per_source_rate_differences']==dict(zip(groups,deltas)) and reported['source_units']==5 and reported['resamples']==3125
            values=[math.fsum(deltas[j] for j in indices)/5 for indices in itertools.product(range(5),repeat=5)]
            np.testing.assert_allclose(reported['paired_source_percentile95'],np.quantile(values,[.025,.975]),atol=1e-12,rtol=0)
    log=read(OUT/'训练日志.json');events=[json.loads(x) for x in (OUT/'事件.jsonl').read_text(encoding='utf-8').splitlines()]
    assert log['updates']==500 and log['supervision_uses']==4500 and log['trainable_parameters']==8192 and log['encoder_optimizer_parameters']==0
    usage=[0]*12
    for t,r in enumerate(log['log']):
        expected=[(t*9+j)%12 for j in range(9)];assert r['step']==t+1 and r['query_indices']==expected and r['gradient_norm_before_clip']>0
        for i in expected:usage[i]+=1
    assert usage==[375]*12 and len(log['log'])==500
    for event in ('optimizer_attempt','optimizer_saved'):
        selected=[r for r in events if r['event']==event];assert [r['step'] for r in selected]==list(range(1,501))
        assert all(r['query_indices']==[(t*9+j)%12 for j in range(9)] for t,r in enumerate(selected))
    def loss(w):
        scores=np.sum(aa.astype(np.float64)*project(gg.astype(np.float64),w)[:,None,:],axis=2)/.1
        mx=scores.max(axis=1);return float((mx+np.log(np.exp(scores-mx[:,None]).sum(axis=1))-scores[:,40]).mean())
    initial=np.load(OUT/'权重/initial.npz',allow_pickle=False);assert abs(loss(initial)-log['initial_fit12_loss'])<2e-5 and abs(loss(weights)-log['final_fit12_loss'])<2e-5
    import torch
    checkpoint=torch.load(OUT/'权重/recovery.pt',map_location='cpu',weights_only=True)
    assert checkpoint['step']==500 and all(np.array_equal(checkpoint['head'][k].numpy(),weights[k]) for k in ('U','V'))
    assert len(checkpoint['optimizer']['state'])==2 and all(int(s['step'])==500 for s in checkpoint['optimizer']['state'].values())
    sys.path.insert(0,str(ROOT/'project/src'));from data.capturepoint_coverage_protocol_v1 import coverage_gate
    gate=coverage_gate(summary['gate_counts'],complete=True,engineering_and_audit=True)
    values=list(summary['gate_counts'].values());full=sum(s['MMRank8Coverage12']['Full'] for s in values)
    independent=[True,True,full>=32]
    independent += [full-sum(s['MMRank8Coverage12'][c] for s in values)>=16 for c in ('Mean','Wrong')]
    independent += [full-sum(s['Disabled'] for s in values)>=16]
    independent += [full-sum(s[a]['Full'] for s in values)>=5 for a in ('Identity','SwissRank8Frozen')]
    independent += [sum(s['MMRank8Coverage12']['Full']>s['MMRank8Coverage12']['Mean'] and s['MMRank8Coverage12']['Full']>s['MMRank8Coverage12']['Wrong'] for s in values)>=3]
    independent += [sum(s['MMRank8Coverage12']['Full']>s['Identity']['Full'] and s['MMRank8Coverage12']['Full']>s['SwissRank8Frozen']['Full'] for s in values)>=3]
    independent += [full-sum(s['MMRank8Final']['Full'] for s in values)>=5,sum(s['MMRank8Coverage12']['Full']>s['MMRank8Final']['Full'] for s in values)>=3]
    assert list(gate['gates'].values())==independent
    return {'stage':'full','passed':True,'planned_records':1625,'new_records':375,'new_saved_scores_verified':9375,'max_score_error':maximum,
        'updates':500,'optimizer_query_uses':4500,'uses_per_query':usage,'source_units':5,'paired_contrasts':6,
        'independent_effect_gates':gate['gates'],'auditor_encoder_forward_calls':0,'auditor_training_steps':0,'raw_audit_reads':0,
        'confirmation_access':0,'independent_person_review':False}


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('engineering','full'):raise SystemExit('engineering | full; saves once')
    stage=sys.argv[1];result=engineering() if stage=='engineering' else full();p=OUT/'核验'/('工程另实现复核.json' if stage=='engineering' else '完整另实现复核.json')
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,ensure_ascii=False))
