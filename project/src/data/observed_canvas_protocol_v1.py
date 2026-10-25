"""CV033 metadata/pure metric protocol; never reads pixels or constructs a model."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/OpenCVL/空间监督小样本预检_v1'
CAL=('opencvl_engineering_0','opencvl_engineering_1')
DEV=('opencvl_engineering_2','opencvl_engineering_3')
MODES=('Single','Cumulative')
ARMS=('Full','Disabled','Mean','Wrong')
SOURCES=['data/observed_canvas_protocol_v1.py','eval/audit_observed_canvas_protocol_v1.py','tests/test_observed_canvas_protocol_v1.py','documents/close_observed_canvas_protocol_v1.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h)


def target_blind_path(grid=5,budget=10):
    """Fixed west-middle entry, two row sweeps, no label/source argument."""
    if type(grid) is not int or grid!=5 or type(budget) is not int or budget!=10:raise ValueError('Frozen5/B10 only')
    cells=[r*5+c for r in range(grid//2,grid) for c in (range(grid) if (r-grid//2)%2==0 else range(grid-1,-1,-1))][:budget+1]
    rows=[]
    for t,cell in enumerate(cells):
        action=None
        if t:
            delta=cell-cells[t-1];action={1:'right',-1:'left',5:'down',-5:'up'}[delta]
        rows.append(dict(step=t,public_cell=cell,position=list(divmod(cell,5)),remaining=budget-t,action=action,visited=sorted(set(cells[:t+1]))))
    return rows


def contains(cell,xy):
    r,c=divmod(cell,5)
    return c*125<=xy[0]<(c+1)*125 and r*125<=xy[1]<(r+1)*125


def valid_public_packet(v):
    fields={'given_target_rgb','current_image','position','grid_size','step_index','remaining_budget','legal_actions'}
    if type(v) is not dict or set(v)!=fields:raise ValueError('Only given goal,current public observation and action budget')
    if type(v['grid_size']) is not int or v['grid_size']!=5:raise ValueError('Frozen grid')
    t=v['step_index']
    if type(t) is not int or not 0<=t<=10 or type(v['remaining_budget']) is not int or v['remaining_budget']!=10-t:raise ValueError('Frozen B10 steps')
    pos=v['position'];legal=v['legal_actions']
    if not isinstance(pos,(tuple,list)) or len(pos)!=2 or any(type(x) is not int for x in pos) or tuple(pos)!=tuple(target_blind_path()[t]['position']):raise ValueError('Frozen target-blind prefix changed')
    if type(legal) is not tuple or any(type(a) is not str or a not in ('up','right','down','left') for a in legal) or len(set(legal))!=len(legal) or (t<10 and not legal):raise ValueError('Invalid public action permissions')
    return True


def make_cases(pairs,spatial):
    sources={p['pair_id']:p for p in pairs};labels={p['pair_id']:p for p in spatial}
    if set(sources)!=set(CAL+DEV) or set(labels)!=set(CAL+DEV):raise ValueError('Original four sources only')
    result={}
    for fold,ids in [('calibration',CAL),('development',DEV)]:
        rows=[]
        for q in ids:
            for r in ids:
                for p in target_blind_path():
                    xy=labels[q]['camera_xy_evaluator_only'];same=q==r
                    relation='unrelated' if not same else 'inside' if any(contains(c,xy) for c in p['visited']) else 'outside'
                    current='unrelated' if not same else 'inside' if contains(p['public_cell'],xy) else 'outside'
                    rows.append(dict(case_id=f"{fold}/{q}/{r}/{p['step']}",fold=fold,query_evaluator_only=q,reference_evaluator_only=r,
                        step=p['step'],public_current_cell=p['public_cell'],public_visited_cells=p['visited'],
                        prefix=target_blind_path()[:p['step']+1],target_file_evaluator_only=sources[q]['ground_file'],parent_file_evaluator_only=sources[r]['aerial_file'],
                        wrong_query_evaluator_only=ids[1-ids.index(q)],labels_evaluator_only=dict(primary_relation=relation,current_relation=current,
                            xy_native=xy if same else None,resolution=labels[r]['resolution_m_per_pixel'])))
        result[fold]=rows
    return result


def check_rows(cases,outputs,modes=MODES,arms=ARMS):
    expected={(c['case_id'],m,a) for c in cases for m in modes for a in arms}
    actual={(r['case_id'],r['mode'],r['arm']) for r in outputs}
    if len(outputs)!=len(expected) or actual!=expected:raise ValueError('Every planned case/mode/goal condition required once')
    lookup={c['case_id']:c for c in cases}
    for r in outputs:
        c=lookup[r['case_id']]
        if type(r['score']) not in (int,float) or not math.isfinite(r['score']) or not -1.00001<=r['score']<=1.00001:raise ValueError('Raw finite cosine required')
        xy=r['xy']
        if not isinstance(xy,(list,tuple)) or len(xy)!=2 or not all(type(x) in (int,float) and math.isfinite(x) for x in xy):raise ValueError('Finite point required')
        if not any(contains(cell,xy) for cell in c['public_visited_cells']):raise ValueError('Primary peak outside shared legal visited ROI')


def strata(cases,outputs,tau,mode,arm):
    lookup={c['case_id']:c for c in cases};selected=[r for r in outputs if r['mode']==mode and r['arm']==arm];by={};details=[]
    for source in sorted({c['query_evaluator_only'] for c in cases}):
        group=[r for r in selected if lookup[r['case_id']]['query_evaluator_only']==source];counts={};rates={};joint=0;errors=[]
        for label in ('inside','outside','unrelated'):
            rows=[r for r in group if lookup[r['case_id']]['labels_evaluator_only']['primary_relation']==label]
            if not rows:raise ValueError('All three classes required per query source')
            accepted=sum(float(r['score'])>=float(tau) for r in rows)
            counts[label]=dict(n=len(rows),accepted=accepted)
            rates[label]=Fraction(accepted,len(rows)) if label=='inside' else Fraction(len(rows)-accepted,len(rows))
            for r in rows:
                c=lookup[r['case_id']];error=math.dist(r['xy'],c['labels_evaluator_only']['xy_native'])*c['labels_evaluator_only']['resolution'] if label=='inside' else None
                correct=(r['score']>=tau and error<=5) if label=='inside' else r['score']<tau
                if label=='inside':joint+=int(correct);errors.append(error)
                details.append(dict(case_id=r['case_id'],source=source,relation=label,accepted=r['score']>=tau,error_m=error,joint_correct=bool(correct)))
        ba=sum(rates.values())/3
        by[source]=dict(counts=counts,balanced_accuracy=float(ba),joint_positive=joint,positive_n=len(errors),joint_positive_rate=joint/len(errors),
            positive_MAE_all_predictions_m=sum(errors)/len(errors))
    return by,details


def calibrate(cases,outputs):
    if {c['query_evaluator_only'] for c in cases}!=set(CAL) or any(c['fold']!='calibration' for c in cases):raise ValueError('Only frozen calibration sources')
    check_rows(cases,outputs,arms=('Full',));answer={}
    for mode in MODES:
        rows=[r for r in outputs if r['mode']==mode];values=sorted({float(r['score']) for r in rows})
        candidates=[math.nextafter(values[0],-math.inf)]+[(a+b)/2 for a,b in zip(values,values[1:])]+[math.nextafter(values[-1],math.inf)]
        evaluated=[]
        for t in candidates:
            # Exact rational class rates avoid a floating utility tie changing tau.
            utility=Fraction(0)
            for source in CAL:
                rates=[]
                for label in ('inside','outside','unrelated'):
                    sub=[r for r in rows if next(c for c in cases if c['case_id']==r['case_id'])['query_evaluator_only']==source and next(c for c in cases if c['case_id']==r['case_id'])['labels_evaluator_only']['primary_relation']==label]
                    if not sub:raise ValueError('Missing calibration stratum')
                    n=sum(r['score']>=t for r in sub);rates.append(Fraction(n if label=='inside' else len(sub)-n,len(sub)))
                utility+=sum(rates)/6
            evaluated.append((utility,t))
        utility,t=max(evaluated)
        answer[mode]=dict(tau=t,utility=float(utility),candidates=len(candidates),tie='largest tau under exact rational utility',
            all_candidates=[dict(tau=x,utility=float(u),utility_fraction=str(u)) for u,x in evaluated],source_ids=list(CAL))
    return answer


def evaluate(cases,outputs,thresholds):
    if any(c['fold']!='development' for c in cases) or {c['query_evaluator_only'] for c in cases}!=set(DEV):raise ValueError('Frozen development only')
    check_rows(cases,outputs);metrics={};details={}
    for m in MODES:
        for a in ARMS:
            by,d=strata(cases,outputs,thresholds[m]['tau'],m,a);metrics[m+'/'+a]=dict(planned=44,by_source=by,macro_BA=sum(v['balanced_accuracy'] for v in by.values())/2)
            details[m+'/'+a]={x['case_id']:x for x in d}
    reference='Cumulative/Full';contrasts={}
    for other in ('Single/Full','Cumulative/Disabled','Cumulative/Mean','Cumulative/Wrong'):
        per_source={};by_class={}
        for label in ('inside','outside','unrelated'):
            ids=[c['case_id'] for c in cases if c['labels_evaluator_only']['primary_relation']==label]
            recovered=sum(details[reference][k]['joint_correct'] and not details[other][k]['joint_correct'] for k in ids)
            harmed=sum(details[other][k]['joint_correct'] and not details[reference][k]['joint_correct'] for k in ids)
            by_class[label]=dict(n=len(ids),recovered=recovered,harmed=harmed,net=recovered-harmed)
        for source in DEV:
            x,y=metrics[reference]['by_source'][source],metrics[other]['by_source'][source]
            per_source[source]=dict(joint_positive_net=x['joint_positive']-y['joint_positive'],BA_delta=x['balanced_accuracy']-y['balanced_accuracy'])
        contrasts[other]=dict(by_class=by_class,by_source=per_source,macro_BA_gain=metrics[reference]['macro_BA']-metrics[other]['macro_BA'])
    full=metrics[reference]['by_source'];main=contrasts['Single/Full'];control=[contrasts[k] for k in ('Cumulative/Disabled','Cumulative/Mean','Cumulative/Wrong')]
    gates=dict(complete=True,joint_positive_rate_at_least_half_each_source=all(v['joint_positive_rate']>=.5 for v in full.values()),
        per_source_outside_FAR_at_most10pct=all(v['counts']['outside']['accepted']/v['counts']['outside']['n']<=.1 for v in full.values()),
        per_source_unrelated_FAR_at_most10pct=all(v['counts']['unrelated']['accepted']/v['counts']['unrelated']['n']<=.1 for v in full.values()),
        cumulative_vs_single_joint_net_at_least1_each=all(v['joint_positive_net']>=1 for v in main['by_source'].values()),
        cumulative_vs_single_BA_gain_at_least5pp_and_no_source_harm=main['macro_BA_gain']>=.05-1e-12 and all(v['BA_delta']>=-1e-12 for v in main['by_source'].values()),
        each_goal_control_positive_net_and_BA_gain=all(v['by_class']['inside']['net']>=1 and v['macro_BA_gain']>=.05-1e-12 and all(z['joint_positive_net']>=0 for z in v['by_source'].values()) for v in control))
    return dict(metrics=metrics,contrasts=contrasts,gates=gates,effect_gate_passed=all(gates.values()),
        engineering_resource_permission_audit_gates_separate=True,navigation_admitted=False,new_navigation_SR=None)


def make_design():
    pairs=read(BASE/'固定四对清单.json')['pairs'];spatial=read(BASE/'空间监督检查.json')['pairs'];cases=make_cases(pairs,spatial)
    asset=ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1'
    checkpoint=read(asset/'工件取得完成.json')['checkpoint'];mean=asset/'校准目标多层均值.json'
    if read(mean)['source_pairs']!=[0,1]:raise ValueError('Mean calibration sources changed')
    local=[BASE/'固定四对清单.json',BASE/'空间监督检查.json',BASE/'像素质量回执.json',asset/'工件取得完成.json',mean,
        ROOT/'DATA/processed_data/OpenCVL/累计已访问画布合成接口_v1/阶段封存.json',ROOT/'DATA/processed_data/OpenCVL/CCVPE局部感知开发筛查_v1/阶段封存.json']
    runtime=['project/src/agents/ccvpe_runtime_v1.py','project/src/platform_core/observed_canvas_v1.py']
    return dict(task='CV-033',design_frozen=True,actual_executor_bound=False,actual_inference_authorized=False,training_authorized=False,
        candidate='Single vs Cumulative legal-observation canvas;same frozen CCVPE checkpoint',checkpoint_identity_from_CV030_receipt=checkpoint,
        checkpoint_rehash_and_strict_load_required_before_actual_calls=True,mean_identity=entry(mean),
        sources=4,roles=dict(calibration=list(CAL),development=list(DEV),confirmation=[],all_previously_consumed_engineering=True,
            design_labels_previously_seen=True,known_bias='0/1 dark vs2/3 daytime;targets near parent center;not fresh independent data'),
        path=target_blind_path(),path_rule='west-middle public start(2,0),row-boustrophedon to B10;no goal/source inputs or path/seed scan',
        cases=cases,modes=list(MODES),goal_conditions=list(ARMS),
        input='given ground RGB and sequential current125 RGB/public step/position/permissions only;history built inside CV032 core',
        forbidden=['true_goal_cell','true_xy','GPS','heading','source_id','unvisited_RGB','external_history','full_parent_image'],
        only_new_factor='retain already observed RGB across the same prefix vs retain current RGB only;same public mask/ROI for scoring',
        primary_ROI='same union of visited native cell rectangles for BOTH modes and all conditions;half-open pixel-center membership at each output scale',
        primary_score='max finest256 raw matching across yaw and common visited ROI;not softmax existence probability',
        primary_location='argmax512 localization heatmap inside same visited ROI,row-major first tie;native center mapping625/512',
        secondary='current125 ROI score/location/labels computed from the SAME outputs,descriptive only,no additional calls or primary gate',
        labels='same-source camera point in visited union=inside;not yet visited=outside;other source=unrelated,all evaluator-only',
        controls=dict(Full='correct ground target',Disabled='all six matching stacks exact zero',
            Mean='existing CV030 six descriptor means from0/1 only,no refit',Wrong='swap0<->1,2<->3;original labels and observation prefix unchanged'),
        calibration='two thresholds,one per representation,Full44 on0/1 only;maximize source macro3-class BA using exact rational utility,largest tau tie;shared across steps/goal controls;Python float64 comparison',
        future_calls=dict(engineering=8,calibration=88,development=352,total=448,maximum_backbone_calls=896,additional_warmups=0),
        engineering='4 calls:cal0 andcal1/ref-self/step0 Single vsCumulative Full exact9-output equivalence<=1e-6;2 calls:cal0/reference0 step5 Cumulative Disabled target0 vs1;2 calls:same Mean target0 vs1;all state frozen/strict;Wrong routes verified against paired Full records in main',
        gates=dict(positive_joint_rate_each_source=.5,per_source_outside_FAR=.1,per_source_unrelated_FAR=.1,
            vs_single_joint_net_each_source=1,vs_single_macro_BA_gain=.05,vs_single_source_BA_min=0,
            vs_each_goal_control_joint_net_total=1,vs_each_control_macro_BA_gain=.05,vs_each_control_source_joint_net_min=0,
            engineering_resource_input_complete_audit_required=True),
        reporting=['planned denominators and all per-prefix outcomes','source/mode/condition/primary-class results','all inside errors including rejected ones',
            'classwise recovery/harm and source paired deltas','always_reject BA2/3 and always_accept BA1/3',
            'public parent center marked off-protocol if unvisited;current center;visited-centroid projected to nearest visited tile center with lowest-cell tie',
            'secondary current-ROI diagnostics separate from main',
            'only2 known development groups;prefixes/modes/controls not independent geography;no inferential CI or paper superiority'],
        resource_proposal=dict(batch=1,CPU_RSS_bytes=4*1024**3,wall_seconds=1800,max_pipeline_calls=448,max_backbones=896,
            original_image_decodes=8,new_model_downloads=0,new_weight_copies=0,new_training_steps=0,result_bytes=16*1024**2),
        retention='scalar/two-ROI peak records+input/output/descriptor hashes+per-layer control receipts;online alternate peak extraction before discard;ONLY four fixed development Full step10 anchors(q=r=2/3 x2modes) save512heat+256yaw-max2D maps<=5MiB raw;no per-case16yaw maps',
        stop='one fixed queue/one factor/one calibration rule;any required gate fails closes this candidate;no path/tau/crop/source/model tuning after results',
        current_work=dict(real_images_read=0,model_loaded=0,forward=0,training=0,navigation=0,API=0,confirmation_groups=0),
        local_inputs=[entry(p) for p in local]+[entry(ROOT/n) for n in runtime],image_references=read(BASE/'像素质量回执.json')['inputs'],
        source_bindings=[entry(ROOT/'project/src'/n) for n in SOURCES])
