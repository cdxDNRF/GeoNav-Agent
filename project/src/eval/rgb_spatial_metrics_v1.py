"""Pure CV041 metrics/calibration gates. Reads saved predictions, no model calls."""
from fractions import Fraction
import math
import random


def fraction(v):return Fraction(*v['macro_joint_exact'])


def measure(cases,outputs,tau,*,raw=False):
    if len(cases)!=len(outputs) or {c['case_id'] for c in cases}!={o['case_id'] for o in outputs}:raise ValueError('full planned case denominator required')
    lookup={o['case_id']:o for o in outputs};by={};correct={};relations={}
    for query in sorted({c['query'] for c in cases}):
        counts={k:dict(n=0,accepted=0,joint_correct=0) for k in ('inside','outside','unrelated')};errors=[];accepted_errors=[]
        for c in (c for c in cases if c['query']==query):
            o=lookup[c['case_id']];relation=c['raw_sensitivity_relation'] if raw else c['relation'];xy=c['raw_camera_xy'] if raw else c['camera_xy']
            accepted=float(o['score'])>=float(tau);bucket=counts[relation];bucket['n']+=1;bucket['accepted']+=accepted
            if relation=='inside':
                error=math.hypot(o['point'][0]-xy[0],o['point'][1]-xy[1])*c['mpp'];errors.append(error)
                if accepted:accepted_errors.append(error)
                good=accepted and error<=5
            else:good=not accepted
            bucket['joint_correct']+=good;correct[c['case_id']]=bool(good);relations[c['case_id']]=relation
        if any(v['n']==0 for v in counts.values()):raise ValueError('all three classes required')
        ba=sum(Fraction(v['joint_correct'],v['n']) for v in counts.values())/3
        by[query]=dict(counts=counts,macro_joint=float(ba),macro_joint_exact=[ba.numerator,ba.denominator],positive_errors_m_all=errors,positive_errors_m_accepted=accepted_errors)
    macro=sum(fraction(v) for v in by.values())/len(by)
    return dict(by_query=by,macro_joint=float(macro),macro_joint_exact=[macro.numerator,macro.denominator]),correct,relations


def paired_interval(values):
    rng=random.Random(20261024);values=list(values)
    draws=sorted(sum(values[rng.randrange(len(values))] for _ in values)/len(values) for _ in range(10000))
    def percentile(q):
        p=(len(draws)-1)*q;a=math.floor(p);b=math.ceil(p)
        return draws[a]+(p-a)*(draws[b]-draws[a])
    return [percentile(.025),percentile(.975)]


def summarize(design,groups,models):
    G={};evaluated={}
    for g in groups:
        key=(g['method'],g['fold'],g['seed'],g['phase'],g['condition'])
        if key in G:raise ValueError('duplicate prediction group')
        G[key]=g
    folds={f['fold']:f for f in design['folds']}
    expected=set()
    for fold in folds:
        expected.add(('RawCos',fold,None,'fit_final','Full'))
        expected.update(('RawCos',fold,None,'development',c) for c in ('Full','Disabled','Mean','Wrong'))
        for seed in (0,1,2):
            for method in ('Spatial','TrainedMean'):
                condition='Full' if method=='Spatial' else 'NoTarget'
                expected.update((method,fold,seed,p,condition) for p in ('fit_initial','fit_final'))
                expected.update((method,fold,seed,'development',c) for c in (('Full','Disabled','Mean','Wrong') if method=='Spatial' else ('NoTarget',)))
    if set(G)!=expected or sum(len(g['outputs']) for g in groups)!=6750 or len(models)!=18:raise ValueError('6750 pair/96 group/18 model matrix incomplete')
    result_metrics=[]
    for key,g in G.items():
        if g['phase']=='fit_initial':continue
        f=folds[g['fold']];cases=f['fit_cases'] if g['phase']=='fit_final' else f['development_cases']
        if g['phase']=='fit_final':tau=g['calibration']['tau']
        else:
            source=G[(g['method'],g['fold'],g['seed'],'fit_final','Full' if g['method']!='TrainedMean' else 'NoTarget')]
            tau=source['calibration']['tau']
        primary,correct,relations=measure(cases,g['outputs'],tau)
        raw,_,_=measure(cases,g['outputs'],tau,raw=True)
        evaluated[key]=(primary,correct,relations)
        result_metrics.append(dict(method=g['method'],fold=g['fold'],seed=g['seed'],phase=g['phase'],condition=g['condition'],tau=tau,primary=primary,raw_sensitivity=raw))
    def result(method,fold,seed,condition):return evaluated[method,fold,seed,'development',condition]
    contrasts={}
    for name,method,condition in [('Disabled','Spatial','Disabled'),('Mean','Spatial','Mean'),('Wrong','Spatial','Wrong'),('TrainedMean','TrainedMean','NoTarget'),('RawCos','RawCos','Full')]:
        classes={k:dict(recovered=0,harmed=0,net=0,n=0) for k in ('inside','outside','unrelated')};deltas={}
        for fold in folds:
            differences=[]
            for seed in (0,1,2):
                m,full,relations=result('Spatial',fold,seed,'Full');o,other,_=result(method,fold,None if method=='RawCos' else seed,condition)
                differences.append(fraction(m)-fraction(o))
                for key,label in relations.items():
                    classes[label]['recovered']+=int(full[key] and not other[key]);classes[label]['harmed']+=int(other[key] and not full[key]);classes[label]['n']+=1
            deltas[fold]=sum(differences)/3
        for c in classes.values():c['net']=c['recovered']-c['harmed']
        gain=sum(deltas.values())/3
        contrasts[name]=dict(by_class=classes,source_macro_joint_gain=float(gain),gain_exact=[gain.numerator,gain.denominator],
            source_deltas={k:float(v) for k,v in deltas.items()},source_paired_bootstrap95_descriptive=paired_interval([float(v) for v in deltas.values()]),
            geographic_n=3,seed_independent_baseline_reused=method=='RawCos')
    positive_by_source={};negative_ok=True
    for fold in folds:
        positive_by_source[fold]=0
        for seed in (0,1,2):
            p=result('Spatial',fold,seed,'Full')[0]['by_query'][fold]['counts'];positive_by_source[fold]+=p['inside']['joint_correct']
            negative_ok &= p['outside']['accepted']<=2 and p['unrelated']['accepted']<=2
    fits=[m for m in models if m['method']=='Spatial']
    fit_ok=all(m['final_fit_loss']<m['initial_fit_loss'] and fraction(evaluated['Spatial',m['fold'],m['seed'],'fit_final','Full'][0])>=Fraction(4,5) for m in fits)
    def contrast_pass(name):
        c=contrasts[name]
        return Fraction(*c['gain_exact'])>=Fraction(5,100) and c['by_class']['inside']['net']>=2
    gates=dict(complete=True,all9_Full_fit_reduction_and_macro80=bool(fit_ok),
               Full_joint_at_least6of9_and_each_city1of3=sum(positive_by_source.values())>=6 and min(positive_by_source.values())>=1,
               each_source_seed_outside2of24_unrelated2of25=bool(negative_ok),
               target_controls_all_gain5pp_positive_net2=all(contrast_pass(n) for n in ('Disabled','Mean','Wrong')),
               two_baselines_gain5pp_positive_net2=all(contrast_pass(n) for n in ('TrainedMean','RawCos')))
    means={}
    for method,condition in [('Spatial','Full'),('Spatial','Disabled'),('Spatial','Mean'),('Spatial','Wrong'),('TrainedMean','NoTarget'),('RawCos','Full')]:
        vals=[];joint=0;positive_errors=[];rates={}
        for fold in folds:
            seeds=(None,) if method=='RawCos' else (0,1,2);source=[];local_joint=0;local_errors=[]
            for seed in seeds:
                p=result(method,fold,seed,condition)[0];source.append(fraction(p));q=p['by_query'][fold]
                local_joint+=q['counts']['inside']['joint_correct'];local_errors+=q['positive_errors_m_all']
            vals.append(sum(source)/len(source));joint+=local_joint;positive_errors+=local_errors
            rates[fold]=dict(joint=local_joint,n=len(seeds),macro_joint=float(sum(source)/len(source)),positive_errors_m_all=local_errors)
        means[method+'/'+condition]=dict(source_macro_joint=float(sum(vals)/3),positive_joint=joint,positive_n=3 if method=='RawCos' else 9,
             positive_error_mean_m=sum(positive_errors)/len(positive_errors),by_source=rates)
    return dict(task='CV-041',completed=True,candidate_passed=all(gates.values()),gates=gates,planned_nontraining_predictions=6750,
                development_predictions=2850,source_cities=3,independent_confirmation_sources=0,positive_by_source=positive_by_source,
                development_means=means,contrasts=contrasts,all_primary_raw_metrics=result_metrics,
                default_upgrade=False,navigation_admitted=False,new_navigation_SR=None,
                conclusion='limited known-source development candidate' if all(gates.values()) else 'candidate closed by preregistered gates; no tuning or retry')
