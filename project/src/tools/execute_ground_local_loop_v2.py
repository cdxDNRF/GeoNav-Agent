"""CV-024 pre-action import repair. Preserve all four bound v1 source files.

The installed top-level `tests` package shadows our namespace directory.
Use an explicit repository-qualified test import. No policy/protocol changes.
"""
import argparse
import hashlib
import json
import sys
import time
import unittest
import numpy as np

from project.src.tools import run_ground_local_loop_v1 as base

OUT, ROOT = base.OUT, base.ROOT


def bind():
    if (OUT/'执行开始.json').exists() or (OUT/'轨迹.jsonl').exists():
        raise ValueError('Repair allowed only before first action')
    b=base.read(OUT/'首动作绑定.json')
    base.verify(b['sources']+b['inputs']+b['protected']+[b['protocol'],b['tasks']])
    base.put('执行入口修复绑定.json',{
        'task':'CV-024','stage':'pre-first-action', 'prior_attempt': 'v1 execute failed at tests import, no execution-start or trajectories created',
        'prior_error': "ModuleNotFoundError: No module named tests.test_ground_local_loop_v1",
        'cause':'a top-level tests package shadows project namespace; explicit project.src.tests import resolves it',
        'sources':[base.entry(ROOT/'project/src/tools/execute_ground_local_loop_v2.py')],
        'original_binding':base.entry(OUT/'首动作绑定.json'),
        'unchanged':'20 tasks/100episodes, four v1 sources, encoders, conditions, formula and stopping gates',
        'initial_freeze_error':'before directory creation, old hash-only protection entry lacked bytes; fixed before v1 source binding',
        'real_actions_so_far':0})
    print('Supplemental entry bound;0 navigation actions')


def verify_repair():
    repair=base.read(OUT/'执行入口修复绑定.json')
    base.verify(repair['sources']+[repair['original_binding']])


def execute():
    verify_repair()
    if (OUT/'执行开始.json').exists(): raise FileExistsError('Execution consumed; inspect partial/completed outputs')
    b=base.read(OUT/'首动作绑定.json')
    base.verify(b['sources']+b['inputs']+b['features']+b['models']+b['protected']+[b['protocol'],b['tasks']])
    from project.src.tests.test_ground_local_loop_v1 import Tests
    tests=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(Tests).run(tests)
    base.put('测试核验.json',{'tests':tests.testsRun,'failures':tests.failures,'errors':tests.errors,'passed':tests.wasSuccessful(),'real_inputs':0})
    if not tests.wasSuccessful():raise ValueError('Engineering tests failed; no action')
    from project.src.agents.ground_local_loop_v1 import GroundLocalLoop
    tasks=base.read(OUT/'任务清单.json')['tasks']
    ground=np.load(ROOT/b['features'][0]['path'],allow_pickle=False)
    aerial=np.load(ROOT/b['features'][1]['path'],allow_pickle=False)
    if ground.shape!=(14,512) or aerial.shape!=(1134,512) or not np.isfinite(ground).all() or not np.isfinite(aerial).all():
        raise ValueError('Frozen cache differs')
    g=ground.astype(np.float64); g/=np.linalg.norm(g,axis=1,keepdims=True)
    mean=g[:9].mean(0);mean/=np.linalg.norm(mean)
    base.put('均值与权限绑定.json',{'fit_indices':list(range(9)),'mean':mean.tolist(),
        'sha256_float64':hashlib.sha256(mean.tobytes()).hexdigest(),'goal_shape':[512],
        'public_fields':['position','grid_size','step_index','remaining_steps','current_feature'],
        'cache_bank_never_policy_argument':True})
    base.put('执行开始.json',{'task':'CV-024','entry':'execute_ground_local_loop_v2','single_run':True,'model_forward':0})
    started=time.perf_counter(); rows=[]
    with (OUT/'轨迹.jsonl').open('x',encoding='utf-8') as output:
        for t in tasks:
            for arm in base.ORDER:
                goals={'Full':g[t['ground_feature_index']],'Disabled':None,'Mean':mean,
                       'Wrong':g[t['wrong_ground_feature_index']],'Frontier':None}
                p=GroundLocalLoop();p.begin(grid_size=5,budget=10,goal_feature=goals[arm],use_evidence=arm!='Frontier')
                pos,truth=t['start'],t['goal_index_evaluator_only'];visited={pos};events=[]
                for step in range(10):
                    event=p.observe({'position':pos,'grid_size':5,'step_index':step,'remaining_steps':10-step,
                        'current_feature':aerial[t['aerial_feature_indices'][pos]].copy()})
                    r,c=divmod(pos,5);dr,dc=((-1,0),(0,1),(1,0),(0,-1))[event['action']]
                    rr,cc=r+dr,c+dc;nxt=rr*5+cc if 0<=rr<5 and 0<=cc<5 else pos
                    event.update(next_position=nxt,moved=nxt!=pos);events.append(event)
                    pos=nxt;visited.add(pos)
                    if pos==truth:break
                p.finish()
                row={'task_id':t['task_id'],'arm':arm,'source':t['source_id_evaluator_only'],
                    'group':t['group_evaluator_only'],'distance':t['distance_evaluator_only'],'start':t['start'],
                    'goal_evaluator_only':truth,'final':pos,'success':pos==truth,'SG':base.distance(pos,truth),
                    'steps':len(events),'moves':sum(e['moved'] for e in events),'coverage':len(visited)/25,
                    'events':events,'experimental_cached_closed_loop':True}
                output.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');output.flush();rows.append(row)
                if time.perf_counter()-started>600:raise TimeoutError('Execution stopped; retain partials')
    sums=base.summary(rows);full={r['task_id']:r for r in rows if r['arm']=='Full'};contrasts={}
    for arm in base.ORDER[1:]:
        control={r['task_id']:r for r in rows if r['arm']==arm}
        recover=sum(full[k]['success'] and not control[k]['success'] for k in full)
        harm=sum(control[k]['success'] and not full[k]['success'] for k in full)
        deltas=[sums['Full']['by_source'][s]['SR']-sums[arm]['by_source'][s]['SR'] for s in base.SOURCE_ORDER]
        samples=[sum(deltas[i] for i in ids)/5 for ids in base.itertools.product(range(5),repeat=5)]
        contrasts[arm]={'recovered':recover,'harmed':harm,'net':recover-harm,'source_SR_deltas':deltas,
            'source_bootstrap95':np.quantile(samples,[.025,.975]).tolist(),'positive_sources':sum(d>0 for d in deltas),
            'actions_changed_paired_tasks':sum([e['action'] for e in full[k]['events']]!=[e['action'] for e in control[k]['events']] for k in full)}
    gates={'all100_completed':len(rows)==100,'net_at_least2_each':all(c['net']>=2 for c in contrasts.values()),
        'SG_no_worse_each':all(sums['Full']['SG_cells']<=sums[a]['SG_cells'] for a in base.ORDER[1:]),
        'positive_frontier_sources_at_least3':contrasts['Frontier']['positive_sources']>=3}
    base.put('结果汇总.json',{'task':'CV-024','arms':sums,'contrasts':contrasts,'gates':gates,
        'mechanism_screen_passed_pending_audit':all(gates.values()),'formal_acceptance':False,
        'goal_semantics':'dataset paired center proxy','sources':5,'unique_tasks':20,'episodes':100,'seed_repeats':1,
        'not_paper_reproduction_or_default_upgrade':True})
    if any(x in sys.modules for x in ('torch','transformers','timm','requests','urllib.request')):
        raise ValueError('Forbidden framework/network import')
    base.verify(b['protected']+b['sources']+b['inputs']);verify_repair()
    base.put('执行完成.json',{'state':'completed','seconds':time.perf_counter()-started,'episodes':len(rows),
        'actions':sum(r['steps'] for r in rows),'encoder_forward':0,'training':0,'API':0,'raw_pixel_reads':0,
        'downloads':0,'reserved_groups_consumed':0,'protected_verified':len(b['protected']),
        'forbidden_modules_imported':False,'repair_bound':True})
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<=8*1024**2
    print(json.dumps({a:{'success':sums[a]['success'],'SR':sums[a]['SR'],'SG':sums[a]['SG_cells']} for a in base.ORDER}))


def audit():
    verify_repair()
    from project.src.eval.audit_ground_local_loop_v1 import audit as alternate
    result=alternate();result['supplemental_entry_SHA_verified']=True
    if '--save' in sys.argv:base.put('另实现复核.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['bind','execute','audit']);parser.add_argument('--save',action='store_true')
    {'bind':bind,'execute':execute,'audit':audit}[parser.parse_args().phase]()
