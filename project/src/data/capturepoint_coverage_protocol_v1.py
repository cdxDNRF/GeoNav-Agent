"""CV-018 model-free sampling, queue construction and fixed12-pair effect gates."""
import copy

CANDIDATE='MMRank8Coverage12'


def sample_indices(n,step):
    if type(n)!=int or n not in (9,12) or type(step)!=int or not 0<=step<500:
        raise ValueError('Frozen N9/12, zero-based step0..499 only')
    return [(9*step+j)%n for j in range(9)]


def build_queue(old,extras):
    if len(old['ground_encoder_inputs'])!=14 or len(old['aerial_encoder_inputs'])!=1134 or len(old['training_queries'])!=9 or len(old['development_tasks'])!=125:
        raise ValueError('Frozen9fit5dev/14ground1134aerial/125windows required')
    if len(extras)!=3 or len({s['source_id'] for s in extras})!=3:raise ValueError('Exactly3 extra sources')
    used={q['source_id_evaluator_only'] for q in old['training_queries']};groups={q['group_evaluator_only'] for q in old['training_queries']}
    result=copy.deepcopy(old)
    for j,s in enumerate(extras):
        if s['source_id'] in used or s['group'] not in groups or s['role']!='fit_preparation':raise ValueError('Same fit groups only, never development/confirmation')
        if len(s['tiles'])!=81 or len({t['sha256'] for t in s['tiles']})!=81 or not s['tiles'][40]['path'].endswith('/sat_5_5.jpg'):
            raise ValueError('Frozen81 distinct candidates and center proxy required')
        gi=14+j;ai=1134+j*81
        result['ground_encoder_inputs'].append(copy.deepcopy(s['ground']));result['aerial_encoder_inputs'].extend(copy.deepcopy(s['tiles']))
        result['training_queries'].append({'source_id_evaluator_only':s['source_id'],'group_evaluator_only':s['group'],
            'ground_feature_index':gi,'aerial_feature_indices':list(range(ai,ai+81)),
            'positive_candidate_index_evaluator_only':40,'label_semantics':'CSV/code_capture_point_proxy'})
    result.update({'arms':['Identity','SwissRank8Frozen','MMRank8Final',CANDIDATE],'planned_rankings':1625,
        'reuse_original_rankings':1250,'new_rankings':375,'mean_ground_feature_indices':list(range(9)),
        'sampler_rule':'batch9; ordered12[(9*t+j)%12],t0..499,j0..8;4500uses,375/query',
        'actual_new_features_bound':False,'policy_input':'normalized ground512 only; all candidate banks/labels/source IDs are trainer/evaluator metadata'})
    return result


def coverage_gate(per_source,*,complete=False,engineering_and_audit=False):
    if type(complete)!=bool or type(engineering_and_audit)!=bool or len(per_source)!=5:raise ValueError('Five source units and explicit completion flags required')
    for s in per_source.values():
        if s['planned']!=25:raise ValueError('Planned25 windows/source cannot shrink')
        values=[s[CANDIDATE][c] for c in ('Full','Mean','Wrong')]+[s[a]['Full'] for a in ('Identity','SwissRank8Frozen','MMRank8Final')]+[s['Disabled']]
        if any(type(v)!=int or not 0<=v<=25 for v in values):raise ValueError('Integer counts0..25 required')
    total=lambda arm,c='Full':sum(s[arm][c] for s in per_source.values())
    full=total(CANDIDATE);disabled=sum(s['Disabled'] for s in per_source.values())
    ctrl=sum(s[CANDIDATE]['Full']>s[CANDIDATE]['Mean'] and s[CANDIDATE]['Full']>s[CANDIDATE]['Wrong'] for s in per_source.values())
    base=sum(s[CANDIDATE]['Full']>s['Identity']['Full'] and s[CANDIDATE]['Full']>s['SwissRank8Frozen']['Full'] for s in per_source.values())
    mm9=sum(s[CANDIDATE]['Full']>s['MMRank8Final']['Full'] for s in per_source.values())
    gates={'all1625_planned_rankings_complete':complete,'all_engineering_and_audit_passed':engineering_and_audit,
        'Full_top1_at_least32':full>=32,'Full_minus_Mean_at_least16':full-total(CANDIDATE,'Mean')>=16,
        'Full_minus_Wrong_at_least16':full-total(CANDIDATE,'Wrong')>=16,'Full_minus_Disabled_at_least16':full-disabled>=16,
        'Full_minus_Identity_at_least5':full-total('Identity')>=5,'Full_minus_Swiss_at_least5':full-total('SwissRank8Frozen')>=5,
        'sources_above_both_controls_at_least3':ctrl>=3,'sources_above_both_baselines_at_least3':base>=3,
        'Full_minus_MM9_at_least5':full-total('MMRank8Final')>=5,'sources_above_MM9_at_least3':mm9>=3}
    return {'gates':gates,'passed':all(gates.values()),'planned_windows':125,'source_units':5,
        'Full_top1':full,'source_positive_controls':ctrl,'source_positive_baselines':base,'source_positive_MM9':mm9,
        'navigation_SR':None,'interpretation':'development association only, no default or confirmation admission'}
