"""CV-012 model-free task/control construction and fixed development gate."""
import hashlib
import math

ARMS=['Identity','SwissRank8Frozen','MMRank8Final']
CONDITIONS=['Full','Mean','Wrong']


def build_queues(sources):
    if len(sources)!=14 or [s['role'] for s in sources]!=['fit']*9+['development']*5:
        raise ValueError('Exactly9fit then5development sources required')
    if len({s['source_id'] for s in sources})!=14 or len({s['group'] for s in sources})!=14:
        raise ValueError('One distinct representative per distinct provisional source group')
    ground=[];aerial=[];training=[];tasks=[]
    for i,s in enumerate(sources):
        if len(s['tiles'])!=81 or len({t['sha256'] for t in s['tiles']})!=81:
            raise ValueError('81 distinct byte references required')
        for j,t in enumerate(s['tiles']):
            row,col=divmod(j,9)
            if not t['path'].endswith(f'/sat_{row+1}_{col+1}.jpg'):raise ValueError('Expected row-first1..9 inventory ordering')
        if not s['tiles'][40]['path'].endswith('/sat_5_5.jpg'):raise ValueError('Camera-center proxy drift')
        ground.append(s['ground']);aerial.extend(s['tiles'])
        if i<9:
            training.append({'source_id_evaluator_only':s['source_id'],'group_evaluator_only':s['group'],
                'ground_feature_index':i,'aerial_feature_indices':list(range(i*81,(i+1)*81)),
                'positive_candidate_index_evaluator_only':40,'label_semantics':'CSV/code_capture_point_proxy'})
        else:
            goals=sorted(range(25),key=lambda g:hashlib.sha256(f"CV-012/goal-order/v1:{s['group']}:{g}".encode()).hexdigest())
            for goal in goals:
                gr,gc=divmod(goal,5)
                positions=[(5-gr+r,5-gc+c) for r in range(5) for c in range(5)]
                indices=[i*81+(r-1)*9+c-1 for r,c in positions]
                assert indices[goal]==i*81+40
                tasks.append({'pair_id':f'cptd_{len(tasks):03d}','dataset':'capturepoint_development',
                    'source_id_evaluator_only':s['source_id'],'group_evaluator_only':s['group'],
                    'ground_feature_index':i,'wrong_ground_feature_index':9+(i-9+1)%5,
                    'aerial_feature_indices':indices,'goal_index_evaluator_only':goal,
                    'unique_ground_queries_for_this_group':1,'not_navigation':True})
    return {'ground_encoder_inputs':ground,'aerial_encoder_inputs':aerial,'training_queries':training,
            'development_tasks':tasks,'planned_rankings':len(tasks)*(len(ARMS)*len(CONDITIONS)+1),
            'arms':list(ARMS),'conditions':list(CONDITIONS),'shared_disabled_target_records':125,
            'disabled_definition':'one shared constant-zero25-score control per pair; CV-002 hash tie order; do not pass zero vectors to head',
            'policy_input':'normalized ground512 only; all indices/source IDs/candidate banks remain evaluator/trainer metadata'}


def gate(per_source,*,complete=False,engineering_and_audit=False):
    """Counts only, for a later executor; current task supplies synthetic tests only."""
    if type(complete)!=bool or type(engineering_and_audit)!=bool:raise ValueError('Explicit boolean completion/audit flags required')
    if len(per_source)!=5:raise ValueError('Five source units required, never125 independent windows')
    for p in per_source.values():
        if p['planned']!=25:raise ValueError('Fixed25 windows per source, planned denominator cannot shrink')
        counts=[p[arm][condition] for arm in ARMS for condition in CONDITIONS]+[p['Disabled']]
        if any(type(c)!=int or not 0<=c<=25 for c in counts):raise ValueError('Integer successful-window counts0..25 required')
    sums={a:{c:sum(p[a][c] for p in per_source.values()) for c in CONDITIONS} for a in ARMS}
    disabled=sum(p['Disabled'] for p in per_source.values());full=sums['MMRank8Final']['Full']
    positive_controls=sum(p['MMRank8Final']['Full']>p['MMRank8Final']['Mean'] and p['MMRank8Final']['Full']>p['MMRank8Final']['Wrong'] for p in per_source.values())
    positive_baselines=sum(p['MMRank8Final']['Full']>p['Identity']['Full'] and p['MMRank8Final']['Full']>p['SwissRank8Frozen']['Full'] for p in per_source.values())
    checks={'all1250_planned_rankings_complete':bool(complete),'all_engineering_and_audit_passed':bool(engineering_and_audit),
        'Full_Top1_at_least32_of125':full>=32,
        'Full_minus_Mean_at_least16':full-sums['MMRank8Final']['Mean']>=16,
        'Full_minus_Wrong_at_least16':full-sums['MMRank8Final']['Wrong']>=16,
        'Full_minus_Disabled_at_least16':full-disabled>=16,
        'Full_minus_Identity_at_least5':full-sums['Identity']['Full']>=5,
        'Full_minus_SwissFrozen_at_least5':full-sums['SwissRank8Frozen']['Full']>=5,
        'sources_above_both_target_controls_at_least3':positive_controls>=3,
        'sources_above_both_baselines_at_least3':positive_baselines>=3}
    return {'gates':checks,'passed':all(checks.values()),'planned_windows':125,'unique_source_units':5,
            'source_positive_against_controls':positive_controls,'source_positive_against_baselines':positive_baselines,
            'counts':sums,'shared_disabled_top1':disabled,'interpretation':'development candidate only, not navigation SR or new-source confirmation'}
