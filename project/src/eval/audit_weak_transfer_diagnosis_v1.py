"""CV-009 alternate saved-result counting and descriptive linear algebra only."""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/MM-GAG'
OUT=BASE/'弱场景与拍摄点迁移差异诊断_v1'
RUN=BASE/'弱场景对齐头工程与训练_v3'
WEAK=BASE/'配对派生清单与弱目标准入_v1'
MM=BASE/'跨视角感知最小验证_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10),(a,b)


def describe_check(values,reported):
    assert reported['n']==len(values)
    ordered=sorted(values);n=len(ordered)
    close(reported['mean'],math.fsum(ordered)/n)
    close(reported['min'],ordered[0]);close(reported['max'],ordered[-1])
    close(reported['median'],ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2)


def normalized(rows):
    return np.array([np.array(row,dtype=np.float64)/math.sqrt(math.fsum(float(v)*float(v) for v in row)) for row in rows])


def audit(save=False):
    binding=read(OUT/'输入与源码绑定.json');cache={};checks=0
    def verify(entries):
        nonlocal checks
        for e in entries:
            p=e['path']
            if p not in cache:cache[p]=digest(ROOT/p)
            assert cache[p]==e['sha256'],p;checks+=1
    for k in ['inputs','sources','protected','previous_seals']:verify(binding[k])
    verify([binding['plan']]);old_items=0
    for s in binding['previous_seals']:
        old=read(ROOT/s['path']);verify(old['files']+old['sources']);old_items+=len(old['files'])+len(old['sources'])
    # No inference frameworks, dynamic execution or network/CLI escape in any new code.
    forbidden={'torch','transformers','safetensors','PIL','requests','httpx','subprocess','importlib','socket'}
    for e in binding['sources']:
        tree=ast.parse((ROOT/e['path']).read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert not any(a.name.split('.')[0] in forbidden for a in node.names)
            if isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] not in forbidden
            if isinstance(node,ast.Call):
                if isinstance(node.func,ast.Name):assert node.func.id not in {'eval','exec','__import__'}
                if isinstance(node.func,ast.Attribute):assert node.func.attr not in {'forward','predict','generate','fit'}
    rows=read(RUN/'逐题排序结果.json')['rows'];assert len(rows)==846
    stats=read(OUT/'排序与目标对照重计.json');lookup={};ranks={};checked_scores=0
    for r in rows:
        order=sorted(range(25),key=lambda j:(-r['scores'][j],hashlib.sha256(('CV-002/tie/v1:'+r['pair_id']+':'+str(j)).encode()).hexdigest()))
        pos=1+order.index(r['truth_evaluator_only']);assert pos==r['true_rank'] and order==r['ranking']
        assert r['top1']==(pos==1) and r['top5']==(pos<=5)
        key=(r['dataset'],r['arm'],r['condition'],r['pair_id']);assert key not in lookup
        lookup[key]=r;ranks[key]=pos;checked_scores+=25
    for ds,n,ng in [('weak_dev',125,5),('MMGAG',16,10)]:
        sub=[r for r in rows if r['dataset']==ds];groups=sorted({r['group'] for r in sub})
        assert len(groups)==ng and stats[ds]['planned']==n and stats[ds]['source_groups']==ng
        for arm in ['Identity','Rank8Final']:
            for condition in ['Full','Mean','Wrong']:
                selected=[r for r in sub if r['arm']==arm and r['condition']==condition]
                def counts_check(selected,out):
                    assert out['planned']==len(selected) and out['top1']==sum(r['true_rank']==1 for r in selected)
                    assert out['top5']==sum(r['true_rank']<=5 for r in selected)
                    close(out['median_rank'],np.median([r['true_rank'] for r in selected]))
                assert len(selected)==n;counts_check(selected,stats[ds]['counts'][arm][condition])
                for group in groups:counts_check([r for r in selected if r['group']==group],stats[ds]['per_source'][group][arm][condition])
        for name,report in stats[ds]['contrasts'].items():
            if name.startswith('Final_vs_Identity_'):cond=name[len('Final_vs_Identity_'):];t=('Rank8Final',cond);c=('Identity',cond)
            else:arm,cond=name.split('_Full_vs_');t=(arm,'Full');c=(arm,cond)
            pairs=sorted({r['pair_id'] for r in sub})
            for label,k in [('top1',1),('top5',5)]:
                recovered=sum(ranks[(ds,*t,p)]<=k<ranks[(ds,*c,p)] for p in pairs)
                harmed=sum(ranks[(ds,*c,p)]<=k<ranks[(ds,*t,p)] for p in pairs)
                assert report[label]=={'planned':n,'recovered':recovered,'harmed':harmed,'net':recovered-harmed}
        for name,report in stats[ds]['saved_score_diagnostics'].items():
            arm,cond=name.split('_Full_vs_');difference=[];adv=[];rms=[]
            for p in sorted({r['pair_id'] for r in sub}):
                f=lookup[(ds,arm,'Full',p)];c=lookup[(ds,arm,cond,p)];goal=f['truth_evaluator_only']
                difference.append(f['scores'][goal]-c['scores'][goal]);adv.append(c['true_rank']-f['true_rank'])
                rms.append(math.sqrt(math.fsum((a-b)**2 for a,b in zip(f['scores'],c['scores']))/25))
            describe_check(difference,report['true_score_difference']);describe_check(adv,report['true_rank_advantage']);describe_check(rms,report['candidate_score_rms_difference'])
        gain=[]
        for group in groups:
            p=stats[ds]['per_source'][group]
            actual=(p['Rank8Final']['Full']['top1']-p['Identity']['Full']['top1'])/p['Identity']['Full']['planned']
            close(p['Full_top1_rate_gain'],actual);gain.append(actual)
        close(stats[ds]['Full_source_macro_top1_gain'],math.fsum(gain)/ng)
        assert stats[ds]['groups_improving']==sum(x>0 for x in gain)
        pooled=(stats[ds]['counts']['Rank8Final']['Full']['top1']-stats[ds]['counts']['Identity']['Full']['top1'])/n
        close(stats[ds]['Full_net_top1_pooled_rate'],pooled)
        if ds=='weak_dev':
            # Independent index-based resampling with explicit ordered percentile interpolation.
            values=sorted(math.fsum(gain[i] for i in ids)/5 for ids in itertools.product(range(5),repeat=5))
            def quantile(p):
                index=(len(values)-1)*p;lo=int(index);part=index-lo
                return values[lo]+part*(values[min(lo+1,len(values)-1)]-values[lo])
            interval=stats[ds]['paired_source_bootstrap'];assert interval['resamples']==3125 and interval['source_units']==5
            close(interval['percentile95'][0],quantile(.025));close(interval['percentile95'][1],quantile(.975))
    # Reconstruct window sets from virtual offsets rather than trusting frozen task candidate arrays.
    bank=read(WEAK/'唯一训练查询与图块引用.json');manifest=read(WEAK/'派生素材清单.json')['records']
    structure=read(OUT/'来源与目标重复结构.json')
    assert structure['fit_unique_queries']==9 and structure['weak_dev_unique_queries']==5
    for scene,reported in structure['per_source'].items():
        records=[r for r in manifest if r['scene']==scene];assert len(records)==25
        queries=[q for q in bank['queries'] if q['scene']==scene];assert len(queries)==1
        assert reported['role']==queries[0]['role'] and reported['unique_ground_queries']==1
        assert reported['shifted_windows']==25 and reported['unique_positive_pixel_hashes']==len({r['positive_pixel_sha256'] for r in records})==1
        assert reported['unique_aerial_cells']==81
        if reported['role']=='development':
            sets=[]
            for record in records:
                r,c=divmod(record['variant'],5)
                offsets={(y-r,x-c) for y in range(5) for x in range(5)};sets.append(offsets)
            assert reported['union_cells']==len(set.union(*sets))==81 and reported['all25_common_cells']==len(set.intersection(*sets))==1
            describe_check([len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)],reported['pairwise_window_jaccard'])
    support=read(OUT/'缓存特征支持度.json');ground=normalized(np.load(RUN/'特征/ground.npy',allow_pickle=False))
    mm=normalized(np.load(MM/'地面特征.npy',allow_pickle=False));fit_names=read(WEAK/'来源分组与用途.json')['fit']
    qi={q['scene']:i for i,q in enumerate(bank['queries'])};fit=ground[[qi[s] for s in fit_names]];center=fit.mean(axis=0);matrix=(fit-center).T
    assert support['fit_affine_rank']==np.linalg.matrix_rank(matrix) and support['head_projection_calls']==support['aerial_candidate_predictions']==0
    mm_ids={p['pair_id']:i for i,p in enumerate(read(MM/'冻结执行任务.json')['pairs'])}
    max_support_error=0.
    for r in support['per_query']:
        g=mm[mm_ids[r['query']]] if r['role']=='MMGAG' else ground[qi[r['query']]]
        available=[i for i,s in enumerate(fit_names) if r['role']!='fit' or s!=r['query']]
        cos=[math.fsum(float(x)*float(y) for x,y in zip(fit[i],g)) for i in available]
        nearest=available[max(range(len(cos)),key=cos.__getitem__)];assert fit_names[nearest]==r['nearest_fit_query']
        expected_cos=max(cos);close(expected_cos,r['nearest_fit_cosine'])
        delta=g-center;coeff=np.linalg.lstsq(matrix,delta,rcond=None)[0]
        residual=float(np.linalg.norm(matrix@coeff-delta))
        max_support_error=max(max_support_error,abs(residual-r['residual_to_fit_affine_span']))
        close(np.linalg.norm(delta),r['distance_to_fit_centroid']);close(residual,r['residual_to_fit_affine_span'])
    for role,reported in support['summaries'].items():
        selected=[r for r in support['per_query'] if r['role']==role]
        for key,value in reported.items():describe_check([r[key] for r in selected],value)
    weights=read(OUT/'参数与训练记录统计.json')
    for arm,file in [('Identity','initial.npz'),('Rank8Final','final.npz')]:
        state=np.load(RUN/'权重'/file,allow_pickle=False);u=state['U'].astype(np.float64);v=state['V'].astype(np.float64)
        # Parameter matrix only. No target inputs and no new head outputs.
        singular=np.linalg.svd(u@v,compute_uv=False)
        np.testing.assert_allclose(singular[:8],weights[arm]['residual_operator_singular_values'],atol=1e-10,rtol=1e-10)
        close(np.linalg.norm(u),weights[arm]['U_frobenius']);close(np.linalg.norm(v),weights[arm]['V_frobenius'])
        close(np.linalg.norm(u@v),weights[arm]['residual_operator_frobenius']);assert np.max(singular[8:])<1e-10
    training=read(RUN/'训练日志.json');assert weights['training']['updates']==training['updates']==500
    assert weights['training']['distinct_fit_queries']==9 and weights['training']['supervision_uses']==4500
    close(weights['training']['first_fit_loss'],training['log'][0]['loss_before_update']);close(weights['training']['final_fit_loss'],training['final_fit_loss'])
    qa=read(OUT/'旧主观审查关联.json');old_qa=read(BASE/'目标格定义与视觉尺度审查_v1/逐对视觉观察.json')['pairs']
    assert len(qa['pairs'])==len(old_qa)==16
    for r in qa['pairs']:
        q=next(q for q in old_qa if q['pair_id']==r['pair_id']);p=r['pair_id']
        assert r['old_subjective_semantics']==q['shared_semantic_evidence'] and r['group']==q['group_id']
        assert r['camera_vs_visible_landmark_cell']==q['camera_vs_landmark_cell']=='unknown'
        assert r['rank_before']==ranks['MMGAG','Identity','Full',p] and r['rank_final']==ranks['MMGAG','Rank8Final','Full',p]
        assert r['Mean_rank_final']==ranks['MMGAG','Rank8Final','Mean',p] and r['Wrong_rank_final']==ranks['MMGAG','Rank8Final','Wrong',p]
        assert r['top5_recovered']==(r['rank_final']<=5<r['rank_before']) and r['top5_harmed']==(r['rank_before']<=5<r['rank_final'])
    for category,reports in qa['by_old_subjective_semantics'].items():
        ids={r['pair_id'] for r in qa['pairs'] if r['old_subjective_semantics']==category}
        for arm,conditions in reports.items():
            for condition,counts in conditions.items():
                values=[ranks['MMGAG',arm,condition,p] for p in ids]
                assert counts['planned']==len(ids) and counts['top1']==sum(v==1 for v in values) and counts['top5']==sum(v<=5 for v in values)
                close(counts['median_rank'],np.median(values))
    result=read(OUT/'迁移差异诊断汇总.json')
    assert result['weak_full_top1_before_after']==[20,70] and result['MMGAG_full_top1_before_after']==[0,0]
    assert not result['single_failure_cause_identified'] and not result['new_candidate_selected'] and not result['default_upgraded']
    for k in ['model_forward_calls','new_predictions','training_steps','raw_image_reads','navigation_episodes','model_api_calls','downloads','reserved_groups_consumed']:assert result[k]==0
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=4*1024**2
    outcome={'task':'CV-009','passed':True,'binding_checks':checks,'previous_sealed_items':old_items,
             'rankings_recounted':846,'saved_scores_checked':checked_scores,'weak_bootstrap_source_units':5,'exact_resamples':3125,
             'support_rows':len(support['per_query']),'support_alternate_lstsq_max_abs':max_support_error,
             'weight_spectrum_alternate_full_matrix_SVD':True,'QA_pairs':16,'no_inference_imports_or_calls_AST_checked':True,
             'model_forward_calls':0,'new_predictions':0,'new_training_steps':0,'raw_image_reads':0,
             'independent_person_review':False,'method':'same Agent separate counting, index bootstrap, least-squares span, full residual-matrix spectrum, prior QA joins'}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(outcome,f,ensure_ascii=False,indent=2)
    print(json.dumps(outcome,ensure_ascii=False),flush=True)
    return outcome


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
