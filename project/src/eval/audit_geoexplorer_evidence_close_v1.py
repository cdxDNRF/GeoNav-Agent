"""Separately implemented arithmetic checks; no repeated trajectory/model audit."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BATCH=ROOT/'项目导航/GeoExplorer复现差异与诊断收束_v1'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def close(a,b,tolerance=1e-12):
    if not math.isclose(a,b,rel_tol=0,abs_tol=tolerance):raise ValueError(f'{a} != {b}')


def analytic_classes(grid,half_circle):
    classes={}
    for r in range(grid):
        for c in range(grid):
            key=(r,c) if half_circle else (r%(grid-1),c%(grid-1))
            classes.setdefault(key,[]).append(r*grid+c)
    return sorted(classes.values())


def independent_progress(cell,goal,k):
    r,c=divmod(cell,k);gr,gc=divmod(goal,k);old=abs(r-gr)+abs(c-gc)
    result=[]
    for a,(dr,dc) in enumerate([(-1,0),(0,1),(1,0),(0,-1)]):
        rr,cc=r+dr,c+dc
        if 0<=rr<k and 0<=cc<k and abs(rr-gr)+abs(cc-gc)<old:result.append(a)
    return result


def audit():
    protocol=load(BATCH/'元数据/冻结收束协议_v1.json')
    for p,h in protocol['bindings'].items():
        with (ROOT/p).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=h:raise ValueError('input changed: '+p)
    result=load(BATCH/'核验/证据与静态几何汇总_v1.json')
    navigation=result['navigation'];metrics=navigation['metrics'];arms={(m['method'],m['condition']):m for m in metrics}
    if len(arms)!=8 or sum(m['planned'] for m in metrics)!=6000:raise ValueError('denominator')
    for m in metrics:close(m['successes']/m['planned'],m['sr'])
    for contrast in navigation['contrasts']:
        left,right=(arms[tuple(contrast[k])] for k in ['left','right'])
        close(contrast['sr_difference'],left['sr']-right['sr'])
        close((contrast['recovered']-contrast['harmed'])/contrast['paired_observations'],contrast['sr_difference'])
        close(contrast['sg_difference'],left['sg_cells_all_planned']-right['sg_cells_all_planned'])
    goals=result['diagnostics']['goal'];curve=result['diagnostics']['curve']
    for row in goals:
        n=row['planned_direction_labels']
        for label in ['full','mean','wrong']:close(row[label+'_correct']/n,row[label+'_accuracy'])
        if row['head']=='pretrain_direction':
            match=next(r for r in curve if r['partition']==row['partition'] and r['stage']=='epoch300')
            for key in ['full_accuracy','mean_accuracy','wrong_accuracy','full_minus_mean']:close(row[key],match[key])
    context=result['diagnostics']['context'];counts=context['regime_positions']
    active=counts['unseen']+counts['seen_before']
    if active!=context['active_positions'] or sum(counts.values())!=context['supervised_positions']:raise ValueError('context denominator')
    close(counts['unseen']/active,context['unseen_share_of_active'])
    mass=context['regime_microbatch_mean_mass']
    close(mass['unseen']/(mass['unseen']+mass['seen_before']),context['unseen_share_of_active_BCE_mass'])
    if 4*context['supervised_positions']!=context['binary_labels']:raise ValueError('binary labels')
    expected_seeds={0,1,2}
    runs=result['training']['runs']
    if {r['seed'] for r in runs}!=expected_seeds:raise ValueError('training seeds')
    close(sum(r['seconds'] for r in runs),result['training']['total_seconds'])
    mathematical_checks=[]
    for row in result['coordinate_math']:
        k=row['grid'];half=row['phase_multiplier']==1
        expected=analytic_classes(k,half)
        if sorted(row['classes'])!=expected:raise ValueError('classes')
        pairs=sorted([list(p) for c in expected for p in itertools.combinations(c,2)])
        if sorted(row['near_pairs'])!=pairs:raise ValueError('near pair set')
        if row['class_count']!=len(expected) or row['near_pair_count']!=len(pairs):raise ValueError('pair/class counts')
        if row['non_singleton_classes']!=sum(len(c)>1 for c in expected):raise ValueError('groups')
        if row['cells_in_non_singletons']!=sum(len(c) for c in expected if len(c)>1):raise ValueError('aliased cells')
        conflicts=[]
        for a,b in pairs:
            for g in range(k*k):
                aa,bb=independent_progress(a,g,k),independent_progress(b,g,k)
                if aa and bb and not set(aa).intersection(bb):
                    conflicts.append({'current_a':a,'current_b':b,'offline_goal':g,'progress_a':aa,'progress_b':bb})
        if conflicts!=row['coordinate_only_disjoint_direction_labels']:raise ValueError('coordinate-only label count')
        if half:close(row['minimum_distinct_cell_l2'],2*math.sin(math.pi/(2*(k-1))),1e-6)
        elif row['maximum_near_pair_l2']>protocol['near_tolerance_l2']:raise ValueError('FP32 tolerance')
        if row['complete_visual_input_collision_asserted'] or row['cuda_byte_equivalence_asserted']:raise ValueError('unsupported inference')
        mathematical_checks.append({'phase_multiplier':row['phase_multiplier'],'analytic_classes':len(expected),
                                    'near_pairs':len(pairs),'coordinate_only_disjoint_labels':len(conflicts)})
    for anchor in result['source_contract'].values():
        lines=(ROOT/anchor['file']).read_text(encoding='utf-8-sig').splitlines()
        if not 1<=anchor['start']<=anchor['end']<=len(lines):raise ValueError('anchor bounds')
        if 'text' in anchor and anchor['text'] not in lines[anchor['start']-1]:raise ValueError('anchor content')
    for key in ['new_model_forward','new_training','new_navigation','api_calls','reserve_consumption']:
        if result[key]!=0:raise ValueError('out of scope: '+key)
    return {'status':'passed','bound_files_checked':len(protocol['bindings']),
            'metric_arms_checked':8,'contrast_arithmetic_checked':len(navigation['contrasts']),
            'goal_curve_reuse_crosschecks':2,'mathematical_checks':mathematical_checks,
            'audit_scope':'same Agent; separate analytic modular classes, neighbour-distance labels, summary fraction/transition checks; historical full trajectory/gradient/count audits reused, not rerun',
            'independent_person_review':False,'model_forward':0,'new_training':0,'new_navigation':0,'new_sr':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');args=p.parse_args()
    value=audit()
    if args.save:
        with (BATCH/'核验/另实现复核_v1.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
