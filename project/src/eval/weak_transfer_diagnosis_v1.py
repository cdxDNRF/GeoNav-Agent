"""CV-009 descriptive statistics only. No model, image decoding or new predictions."""
from __future__ import annotations
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'DATA/processed_data/MM-GAG'
OUT = BASE/'弱场景与拍摄点迁移差异诊断_v1'
RUN = BASE/'弱场景对齐头工程与训练_v3'
WEAK = BASE/'配对派生清单与弱目标准入_v1'
MM = BASE/'跨视角感知最小验证_v1'
QA = BASE/'目标格定义与视觉尺度审查_v1'
SOURCES = ['eval/weak_transfer_diagnosis_v1.py','eval/audit_weak_transfer_diagnosis_v1.py',
           'tests/test_weak_transfer_diagnosis_v1.py','documents/close_weak_transfer_diagnosis_v1.py']
STAGES = ['数据核验与跨视角准入_v1','跨视角感知最小验证_v1','局部语义匹配单因素验证_v1',
          '目标格定义与视觉尺度审查_v1','拍摄点语义与完整目标图验证_v1','对齐头训练素材准入_v1',
          '配对派生清单与弱目标准入_v1','弱场景对齐头工程与训练_v3']
INPUTS = [RUN/n for n in ['逐题排序结果.json','感知对照汇总.json','训练日志.json','冻结评测任务.json',
                         '首更新前特征与训练绑定.json','执行状态.json','核验/门槛判定.json',
                         '核验/另实现复核.json','特征/ground.npy','特征/aerial.npy','权重/initial.npz','权重/final.npz']]
INPUTS += [WEAK/n for n in ['训练设计冻结.json','唯一训练查询与图块引用.json','派生素材清单.json',
                           '来源分组与用途.json','视觉语义审查.json']]
INPUTS += [MM/n for n in ['地面特征.npy','航拍特征.npy','冻结执行任务.json','划分与任务.json']]
INPUTS += [QA/n for n in ['逐对视觉观察.json','审查汇总.json','静态审查结果.json']]
INPUTS += [BASE/'拍摄点语义与完整目标图验证_v1/目标语义约定.json']


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def entry(path):
    path = Path(path)
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size}


def put(name, value):
    with (OUT/name).open('x',encoding='utf-8') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))


def verify(entries):
    for e in entries: assert sha(ROOT/e['path']) == e['sha256'], e['path']
    return len(entries)


def unit(x):
    x = np.asarray(x,dtype=np.float64)
    norm = np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.isfinite(x).all() or (norm <= 1e-12).any(): raise ValueError('finite nonzero cached vectors required')
    return x/norm


def distribution(values):
    a = np.asarray(values,dtype=np.float64)
    assert a.size and np.isfinite(a).all()
    return {'n':int(a.size),'mean':float(a.mean()),'median':float(np.median(a)),
            'min':float(a.min()),'max':float(a.max())}


def exact_source_bootstrap(deltas):
    """All5^5 paired source resamples, never125 independent-window resamples."""
    if len(deltas) != 5: raise ValueError('This frozen diagnostic uses exactly five weak sources')
    means = np.array([sum(sample)/5 for sample in itertools.product(deltas,repeat=5)])
    return {'source_units':5,'resamples':len(means),'statistic':'unweighted mean of paired source rates',
            'percentile95':[float(x) for x in np.quantile(means,[.025,.975])],
            'interpretation':'descriptive development interval; five named scenes, geography not independently verified'}


def check_rank_row(row):
    scores = np.asarray(row['scores'],dtype=np.float64)
    assert scores.shape == (25,) and np.isfinite(scores).all()
    order = sorted(range(25),key=lambda j:(-float(scores[j]),hashlib.sha256(f"CV-002/tie/v1:{row['pair_id']}:{j}".encode()).hexdigest()))
    rank = order.index(row['truth_evaluator_only'])+1
    assert row['ranking'] == order and row['true_rank'] == rank
    assert row['top1'] == (rank==1) and row['top5'] == (rank<=5)


def selected_counts(rows):
    return {'planned':len(rows),'top1':sum(r['top1'] for r in rows),'top5':sum(r['top5'] for r in rows),
            'median_rank':float(np.median([r['true_rank'] for r in rows]))}


def paired(rows, treatment, control, k):
    first = {r['pair_id']:r['true_rank']<=k for r in rows if (r['arm'],r['condition'])==treatment}
    second = {r['pair_id']:r['true_rank']<=k for r in rows if (r['arm'],r['condition'])==control}
    assert set(first)==set(second)
    recovered = sum(first[p] and not second[p] for p in first)
    harmed = sum(second[p] and not first[p] for p in first)
    return {'planned':len(first),'recovered':recovered,'harmed':harmed,'net':recovered-harmed}


def rank_statistics(rows):
    assert len(rows)==846
    for r in rows: check_rank_row(r)
    unique = {(r['dataset'],r['pair_id'],r['arm'],r['condition']) for r in rows}
    assert len(unique)==846
    results = {}
    for ds,n,groups in [('weak_dev',125,5),('MMGAG',16,10)]:
        subset = [r for r in rows if r['dataset']==ds]
        source_names = sorted({r['group'] for r in subset}); assert len(source_names)==groups
        result = {'planned':n,'source_groups':groups,'counts':{},'per_source':{},'contrasts':{},'saved_score_diagnostics':{}}
        for arm in ['Identity','Rank8Final']:
            result['counts'][arm] = {}
            for condition in ['Full','Mean','Wrong']:
                chosen = [r for r in subset if r['arm']==arm and r['condition']==condition]
                assert len(chosen)==n
                result['counts'][arm][condition] = selected_counts(chosen)
                for source in source_names:
                    source_rows = [r for r in chosen if r['group']==source]
                    result['per_source'].setdefault(source,{}).setdefault(arm,{})[condition] = selected_counts(source_rows)
        for condition in ['Full','Mean','Wrong']:
            result['contrasts']['Final_vs_Identity_'+condition] = {
                f'top{k}':paired(subset,('Rank8Final',condition),('Identity',condition),k) for k in [1,5]}
        for arm in ['Identity','Rank8Final']:
            for control in ['Mean','Wrong']:
                result['contrasts'][arm+'_Full_vs_'+control] = {
                    f'top{k}':paired(subset,(arm,'Full'),(arm,control),k) for k in [1,5]}
                differences=[];rank_advantage=[];rms=[]
                full = {r['pair_id']:r for r in subset if r['arm']==arm and r['condition']=='Full'}
                controls = [r for r in subset if r['arm']==arm and r['condition']==control]
                for c in controls:
                    f = full[c['pair_id']];goal=f['truth_evaluator_only']
                    differences.append(f['scores'][goal]-c['scores'][goal])
                    rank_advantage.append(c['true_rank']-f['true_rank'])
                    rms.append(float(np.sqrt(np.mean((np.array(f['scores'])-c['scores'])**2))))
                result['saved_score_diagnostics'][arm+'_Full_vs_'+control] = {
                    'true_score_difference':distribution(differences),'true_rank_advantage':distribution(rank_advantage),
                    'candidate_score_rms_difference':distribution(rms),
                    'note':'computed only from stored predictions; a changed score is not necessarily a useful decision'}
        gains=[]
        for group in source_names:
            p=result['per_source'][group]
            d=(p['Rank8Final']['Full']['top1']-p['Identity']['Full']['top1'])/p['Identity']['Full']['planned']
            p['Full_top1_rate_gain']=d;gains.append(d)
        result['Full_source_macro_top1_gain']=float(np.mean(gains))
        result['groups_improving']=sum(d>0 for d in gains)
        result['Full_net_top1_pooled_rate']=(result['counts']['Rank8Final']['Full']['top1']-result['counts']['Identity']['Full']['top1'])/n
        if ds=='weak_dev':result['paired_source_bootstrap']=exact_source_bootstrap(gains)
        results[ds]=result
    return results


def sample_structure(bank,tasks,manifest):
    result={}
    for source in sorted({q['scene'] for q in bank['queries']}):
        queries=[q for q in bank['queries'] if q['scene']==source];assert len(queries)==1
        rows=[r for r in manifest if r['scene']==source];assert len(rows)==25
        positive={r['positive_pixel_sha256'] for r in rows};assert len(positive)==1
        windows=[set(t['candidates']) for t in tasks if t['dataset']=='weak_dev' and t['group']==source]
        value={'role':queries[0]['role'],'unique_ground_queries':1,'shifted_windows':25,
               'unique_positive_pixel_hashes':len(positive),'unique_aerial_cells':sum(c['scene']==source for c in bank['cells']),
               'weak_label':'published_scene_common_cell_proxy','capture_point_verified':False}
        if windows:
            assert len(windows)==25
            overlap=[len(x&y)/len(x|y) for x,y in itertools.combinations(windows,2)]
            common=set.intersection(*windows)
            value.update({'union_cells':len(set.union(*windows)),'all25_common_cells':len(common),
                          'pairwise_window_jaccard':distribution(overlap)})
        result[source]=value
    return {'per_source':result,'fit_unique_queries':sum(q['role']=='fit' for q in bank['queries']),
            'weak_dev_unique_queries':sum(q['role']=='development' for q in bank['queries']),
            'MMGAG_queries':16,'MMGAG_groups':10,
            'interpretation':'125 weak windows are5 repeated ground queries/positive cells; no new independent geography'}


def feature_support(weak_ground,mm_ground,bank,fit_names,mm_pairs):
    """Raw target-to-fit support statistics only; never target-to-aerial goal predictions."""
    wg=unit(weak_ground);mg=unit(mm_ground)
    qi={q['scene']:i for i,q in enumerate(bank['queries'])};fi=[qi[n] for n in fit_names]
    fit=wg[fi];center=fit.mean(axis=0);u,s,vt=np.linalg.svd(fit-center,full_matrices=False)
    tol=max((fit-center).shape)*np.finfo(np.float64).eps*s[0]
    rank=int((s>tol).sum());basis=vt[:rank]
    rows=[]
    query_sets=[('fit',name,wg[qi[name]],None) for name in fit_names]
    query_sets += [('weak_dev',q['scene'],wg[i],None) for i,q in enumerate(bank['queries']) if q['role']=='development']
    query_sets += [('MMGAG',p['pair_id'],mg[i],p['group_id']) for i,p in enumerate(mm_pairs)]
    for role,name,g,group in query_sets:
        cosine=fit@g
        if role=='fit':cosine[fit_names.index(name)]=-np.inf  # leave-self-out support
        nearest=int(np.argmax(cosine));delta=g-center
        residual=delta-(delta@basis.T)@basis
        rows.append({'role':role,'query':name,'group':group,'nearest_fit_query':fit_names[nearest],
                     'nearest_fit_cosine':float(cosine[nearest]),'distance_to_fit_centroid':float(np.linalg.norm(delta)),
                     'residual_to_fit_affine_span':float(np.linalg.norm(residual))})
    summaries={}
    for role in ['fit','weak_dev','MMGAG']:
        selected=[r for r in rows if r['role']==role]
        summaries[role]={key:distribution([r[key] for r in selected]) for key in
                         ['nearest_fit_cosine','distance_to_fit_centroid','residual_to_fit_affine_span']}
    return {'fit_affine_rank':rank,'per_query':rows,'summaries':summaries,
            'interpretation':'descriptive encoder-space support; not a confidence threshold, geography proof, new matching prediction or causal explanation',
            'head_projection_calls':0,'aerial_candidate_predictions':0}


def weight_statistics(initial,final,training):
    results={}
    for name,w in [('Identity',initial),('Rank8Final',final)]:
        assert set(w)=={'U','V'} and w['U'].shape==(512,8) and w['V'].shape==(8,512)
        assert np.isfinite(w['U']).all() and np.isfinite(w['V']).all()
        _,ru=np.linalg.qr(w['U'].astype(np.float64),mode='reduced')
        _,rv=np.linalg.qr(w['V'].T.astype(np.float64),mode='reduced')
        singular=np.linalg.svd(ru@rv.T,compute_uv=False)
        results[name]={'parameters':8192,'U_frobenius':float(np.linalg.norm(w['U'].astype(np.float64))),
                       'V_frobenius':float(np.linalg.norm(w['V'].astype(np.float64))),
                       'residual_operator_singular_values':singular.tolist(),
                       'residual_operator_frobenius':float(np.linalg.norm(singular)),
                       'note':'parameter-only spectrum; never apply U/V to cached targets'}
    results['training']={'updates':training['updates'],'distinct_fit_queries':training['unique_fit_queries'],
                         'supervision_uses':training['supervision_uses'],'first_fit_loss':training['log'][0]['loss_before_update'],
                         'final_fit_loss':training['final_fit_loss'],'criterion':'only fit losses saved during CV-008, not a dev learning curve'}
    return results


def qa_join(rows,qa_pairs):
    mm=[r for r in rows if r['dataset']=='MMGAG'];lookup={(r['pair_id'],r['arm'],r['condition']):r for r in mm}
    result=[]
    for q in qa_pairs:
        p=q['pair_id'];before=lookup[p,'Identity','Full'];after=lookup[p,'Rank8Final','Full']
        result.append({'pair_id':p,'group':q['group_id'],'old_subjective_semantics':q['shared_semantic_evidence'],
                       'old_ground_scene_description':q['ground_scene'],'camera_vs_visible_landmark_cell':q['camera_vs_landmark_cell'],
                       'rank_before':before['true_rank'],'rank_final':after['true_rank'],
                       'Mean_rank_final':lookup[p,'Rank8Final','Mean']['true_rank'],
                       'Wrong_rank_final':lookup[p,'Rank8Final','Wrong']['true_rank'],
                       'top5_recovered':after['top5'] and not before['top5'],'top5_harmed':before['top5'] and not after['top5']})
    summaries={}
    for key in sorted({r['old_subjective_semantics'] for r in result}):
        ids={r['pair_id'] for r in result if r['old_subjective_semantics']==key}
        selected=[r for r in mm if r['pair_id'] in ids]
        summaries[key]={arm:{condition:selected_counts([r for r in selected if (r['arm'],r['condition'])==(arm,condition)])
                             for condition in ['Full','Mean','Wrong']} for arm in ['Identity','Rank8Final']}
    return {'pairs':result,'by_old_subjective_semantics':summaries,
            'warning':'reuse CV-004 subjective/nonblind review only; no new annotations or verified geographic truth; post-hoc descriptive subsets'}


def freeze():
    assert not OUT.exists(),'Consumed batch; do not overwrite/rebuild'
    seals=[entry(BASE/s/'阶段封存.json') for s in STAGES]
    for e in seals:
        seal=read(ROOT/e['path']);verify(seal['files']+seal['sources'])
    protected=read(RUN/'首forward前绑定.json')['protected'];verify(protected)
    assert read(RUN/'核验/门槛判定.json')['status']=='closed_failed_no_sweep'
    OUT.mkdir();(OUT/'核验').mkdir()
    plan={'task':'CV-009','date':'2026-10-17','scope':'existing-result descriptive diagnosis, not a new evaluation',
          'planned_rank_rows':846,'weak_dev_windows':125,'weak_dev_queries':5,'MMGAG_pairs':16,'MMGAG_groups':10,
          'analyses':['saved scores/ranks counts, controls and recovered/harmed',
                      'source grouping, physical positive-cell repetition, window candidate overlap',
                      'raw ground-feature support to9 fit queries and affine span (leave-self-out fit)',
                      'parameter-only residual operator spectrum; no head applied to features',
                      'join prior16 subjective QA, never relabel or infer label wrong'],
          'weak_interval':'enumerate5^5 source bootstrap, unweighted paired source-rate mean; descriptive only',
          'no_new_head_or_encoder_forward':True,'no_new_goal_scores_or_predictions':True,'no_training':True,
          'raw_image_reads':0,'navigation_episodes':0,'model_api_calls':0,'downloads':0,'reserved_groups_consumed':0,
          'output_limit_bytes':4*1024**2,'candidate':'none selected; preserve CV-008 stop rule, no tuning',
          'decision_rule':'report supported facts, plausible contributors and unknown causes separately; no default upgrade'}
    put('冻结诊断范围.json',plan)
    put('输入与源码绑定.json',{'task':'CV-009','date':'2026-10-17','inputs':[entry(p) for p in INPUTS],
        'sources':[entry(ROOT/'project/src'/p) for p in SOURCES],'previous_seals':seals,'protected':protected,
        'plan':entry(OUT/'冻结诊断范围.json'),
        'index_values_before':[hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest() for x in read(ROOT/'项目导航/实验索引.json')['batches']]})
    print(json.dumps({'frozen':True,'inputs':len(INPUTS),'sources':len(SOURCES),'old_stages':len(seals),'model_calls':0}),flush=True)


def build():
    assert not (OUT/'迁移差异诊断汇总.json').exists(),'Build consumed'
    binding=read(OUT/'输入与源码绑定.json')
    for key in ['inputs','sources','previous_seals','protected']:verify(binding[key])
    verify([binding['plan']])
    rows=read(RUN/'逐题排序结果.json')['rows'];bank=read(WEAK/'唯一训练查询与图块引用.json')
    pairs=read(MM/'冻结执行任务.json')['pairs'];split=read(WEAK/'来源分组与用途.json')
    ranking=rank_statistics(rows)
    old=read(RUN/'感知对照汇总.json')
    for ds in ['weak_dev','MMGAG']:
        for arm in ['Identity','Rank8Final']:
            for cond in ['Full','Mean','Wrong']:
                x=ranking[ds]['counts'][arm][cond];y=old['counts'][ds][arm][cond]
                assert x['planned']==y['planned'] and x['top1']==y['top1'] and x['top5']==y['top5'] and x['median_rank']==y['median_true_rank']
    structure=sample_structure(bank,read(RUN/'冻结评测任务.json')['tasks'],read(WEAK/'派生素材清单.json')['records'])
    support=feature_support(np.load(RUN/'特征/ground.npy',allow_pickle=False),np.load(MM/'地面特征.npy',allow_pickle=False),bank,split['fit'],pairs)
    initial={k:v for k,v in np.load(RUN/'权重/initial.npz',allow_pickle=False).items()}
    final={k:v for k,v in np.load(RUN/'权重/final.npz',allow_pickle=False).items()}
    weights=weight_statistics(initial,final,read(RUN/'训练日志.json'))
    qa=qa_join(rows,read(QA/'逐对视觉观察.json')['pairs'])
    put('来源与目标重复结构.json',structure);put('缓存特征支持度.json',support);put('参数与训练记录统计.json',weights)
    put('旧主观审查关联.json',qa);put('排序与目标对照重计.json',ranking)
    semantics={'Swiss_weak_label':'published scene common cell;14 scenes/9fit5dev, no verified camera/object footprint',
               'MMGAG_label':read(BASE/'拍摄点语义与完整目标图验证_v1/目标语义约定.json'),
               'old_MMGAG_label_review':read(QA/'审查汇总.json'),
               'not_identified':['single cause of failed transfer','wrong geographic labels','camera vs landmark ground truth',
                                 'capacity shortfall','causal effect of cropping or encoder geography','navigation SR or new-source generalization']}
    put('目标语义与证据边界.json',semantics)
    put('迁移差异诊断汇总.json',{'task':'CV-009','status':'descriptive_statistics_saved; see audit/report for stage completion',
        'weak_full_top1_before_after':[ranking['weak_dev']['counts'][a]['Full']['top1'] for a in ['Identity','Rank8Final']],
        'MMGAG_full_top1_before_after':[ranking['MMGAG']['counts'][a]['Full']['top1'] for a in ['Identity','Rank8Final']],
        'weak_groups_improving':ranking['weak_dev']['groups_improving'],
        'label_equivalence_verified':False,'single_failure_cause_identified':False,
        'new_candidate_selected':False,'CV008_stop_rule_retained':True,'default_upgraded':False,
        'model_forward_calls':0,'new_predictions':0,'training_steps':0,'raw_image_reads':0,
        'navigation_episodes':0,'model_api_calls':0,'downloads':0,'reserved_groups_consumed':0,'new_SR':None,
        'interpretation':'weak-scene association development gain does not transfer to existing capture-point proxy task; task/domain/sample-support mismatch are plausible, not isolated causes'})
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=4*1024**2
    print(json.dumps({'complete':True,'model_calls':0,'new_predictions':0,'weak_source_interval':ranking['weak_dev']['paired_source_bootstrap'],
                      'feature_support':support['summaries']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','build']);args=p.parse_args()
    freeze() if args.mode=='freeze' else build()
