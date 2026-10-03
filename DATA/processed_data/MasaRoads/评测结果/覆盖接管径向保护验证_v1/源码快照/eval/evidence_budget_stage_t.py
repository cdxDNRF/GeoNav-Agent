"""Frozen static public-prefix opportunity check; zero model/API/navigation calls."""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.evidence_budget import public_trigger,paths_for,EvidenceLedger

ROOT=Path(__file__).resolve().parents[3];SRC=ROOT/'project/src'
OLD=ROOT/'DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1'
OUT=ROOT/'DATA/processed_data/Masa/评测结果/证据账本预算规划验证_v1'
DOC=ROOT/'选题报告相关/证据账本预算规划执行协议_v1.md'
TESTS=ROOT/'选题报告相关/证据账本预算规划阶段T测试_v1.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    with Path(p).open(encoding='utf-8') as f:
        for s in f:
            if s.strip():yield json.loads(s)
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2)


def select_tasks(bank):
    sources=sorted({(x['split'],x['area'],x['source_tile']) for x in bank},key=lambda x:sha256(('ledger-g-v1:'+x[2]).encode()).hexdigest())
    if len(sources)!=20:raise ValueError('20 parent source files required')
    chosen=[]
    for i,(split,area,source) in enumerate(sources):
        pool=[x for x in bank if x['split']==split and x['area']==area and x['dist']==12+i%5]
        chosen.append(min(pool,key=lambda x:sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()))
    if Counter(x['dist'] for x in chosen)!={d:4 for d in range(12,17)}:raise ValueError('distance balance')
    return chosen


def public_views(row):
    ledger=EvidenceLedger(10,20);accepted=False
    for i,d in enumerate(row['decisions']):
        ledger.observe_fields(tuple(d['public_position']),d['remaining_budget'],d['public_visited'],
            d['current_image_sha256'],d['target_image_sha256'],d['cue_action'],None if i==0 else row['decisions'][i-1]['action'])
        trigger,reason=public_trigger(d['public_visited'],d['remaining_budget'],20,d['cue_action'] is not None,accepted)
        candidates=paths_for(tuple(d['public_position']),d['public_visited'],10,d['remaining_budget'],d['explorer_action'])
        yield dict(index=i,elapsed=20-d['remaining_budget'],remaining=d['remaining_budget'],trigger=trigger,
            trigger_reason=reason,executable=bool(candidates),protected=d['cue_action'] is not None,
            old_conflict=d['cue_action'] is not None and d['cue_action']!=d['explorer_action'],
            ledger=ledger.snapshot(),candidates=candidates)
        accepted |= d['cue_action'] is not None


def state_record(row,view,kind):
    # Evaluator references/truth stay in this local manifest, never in a model request.
    return dict(episode_id=row['episode_id'],index=view['index'],kind=kind,
                source_key=row['episode_id'].split('_d')[0],baseline_success=row['success'],elapsed=view['elapsed'])


def main():
    if OUT.exists():raise ValueError('immutable ledger batch already exists')
    test=read(TESTS)
    if not test['successful']:raise ValueError('offline tests required')
    (OUT/'阶段T').mkdir(parents=True)
    bank=read(OLD/'导航任务.json');chosen=select_tasks(bank)
    write(OUT/'总体任务.json',chosen)
    references=[OLD/'导航任务.json',OLD/'验收结论.json',OLD/'独立复核.json',ROOT/'project/local_policy_default.json']
    references+=[OLD/f'Edge_s{s}/导航_CueFull_轨迹.jsonl' for s in (0,1,2)]
    for name,source in [('执行协议.md',DOC),('阶段T测试.json',TESTS),('调研建议设计.md',ROOT/'调研/14_多Agent候选冻结实验设计_v1.md')]:
        (OUT/name).write_bytes(source.read_bytes())
    code=[SRC/'agents/evidence_budget.py',SRC/'eval/evidence_budget_stage_t.py',SRC/'tests/test_evidence_budget.py']
    for f in code:
        dest=OUT/'源码快照'/f.relative_to(SRC);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    reg=dict(version='ledger-budget-T-v1',utc=datetime.now(timezone.utc).isoformat(),grid_size=10,budget=20,
        primary_checkpoint=0,descriptive_checkpoints=[1,2],maximum_chat_requests=420,
        phase_caps=dict(engineering=12,D=48,G=360),new_training_steps=0,model_calls=0,
        trigger=dict(stagnation_window=2,max_new_in_window=1,require_repeat=True,checkpoint_step=7,
                     protect_accepted_cue=True,minimum_remaining=2,max_interventions=1),
        D_selection='two different-source baseline-failure first eligibleT prefixes; two different-source elapsed2 prefixes (success/failure); sha256 order',
        D_ablation=['hidden','shuffled','hidden','shuffled'],
        G_selection='all20 sources hash order, distance12+i%5, minimum task JSON hash, no outcomes',
        T_gate=dict(different_failure_sources=2,nonfailure_protection_required=True),
        sources_sha256={f.relative_to(ROOT).as_posix():digest(f) for f in references},
        source_sha256={f.relative_to(SRC).as_posix():digest(f) for f in code},
        default_sha256=digest(ROOT/'project/local_policy_default.json'),
        frozen_files_sha256={f.name:digest(f) for f in OUT.iterdir() if f.is_file()})
    write(OUT/'阶段T/预登记.json',reg)
    results={};diagnostic=None;T_rows0=[];fixed=[];protected_success=0
    for seed in (0,1,2):
        rows=[]
        for row in lines(OLD/f'Edge_s{seed}/导航_CueFull_轨迹.jsonl'):
            views=list(public_views(row));eligible=next((v for v in views if v['trigger'] and v['executable']),None)
            old=next((v for v in views if v['old_conflict']),None)
            source=row['episode_id'].split('_d')[0]
            rec=dict(episode_id=row['episode_id'],source=source,baseline_success=row['success'],
                old_trigger=old is not None,new_trigger=eligible is not None,
                elapsed=None if eligible is None else eligible['elapsed'],remaining=None if eligible is None else eligible['remaining'],
                reason=None if eligible is None else eligible['trigger_reason'],
                accepted_cue_states=sum(v['protected'] for v in views),
                protected_due_states=sum(v['protected'] and v['elapsed']>=7 for v in views))
            rows.append(rec)
            if seed==0:
                if eligible:T_rows0.append((row,eligible))
                at2=next((v for v in views if v['elapsed']==2 and v['executable']),None)
                if at2:fixed.append((row,at2))
                protected_success+=int(row['success'] and rec['accepted_cue_states']>0)
        failures=[r for r in rows if not r['baseline_success']];successes=[r for r in rows if r['baseline_success']]
        results[str(seed)]=dict(tasks=len(rows),original_successes=len(successes),original_failures=len(failures),
            old_trigger_tasks=sum(r['old_trigger'] for r in rows),new_trigger_tasks=sum(r['new_trigger'] for r in rows),
            old_failure_trigger_tasks=sum(r['old_trigger'] for r in failures),new_failure_trigger_tasks=sum(r['new_trigger'] for r in failures),
            new_success_trigger_tasks=sum(r['new_trigger'] for r in successes),
            new_failure_source_count=len({r['source'] for r in failures if r['new_trigger']}),
            trigger_reason_counts=dict(Counter(r['reason'] for r in rows if r['new_trigger'])),
            earliest_elapsed=min((r['elapsed'] for r in rows if r['new_trigger']),default=None),
            remaining_min=min((r['remaining'] for r in rows if r['new_trigger']),default=None))
        write(OUT/f'阶段T/逐题机会_s{seed}.json',rows)
    chosen_D=[];used=set()
    for pool,kind,success in [(T_rows0,'T_failure',False),(T_rows0,'T_failure',False),(fixed,'fixed2_success',True),(fixed,'fixed2_failure',False)]:
        possible=sorted([(r,v) for r,v in pool if r['success']==success and r['episode_id'].split('_d')[0] not in used],
            key=lambda z:sha256((z[0]['episode_id']+':'+str(z[1]['index'])).encode()).hexdigest())
        if not possible:break
        row,view=possible[0];record=state_record(row,view,kind);used.add(record['source_key']);chosen_D.append(record)
    main_result=results['0'];checks=dict(two_failure_sources=main_result['new_failure_source_count']>=2,
        nonfailure_protection=protected_success>0,four_diagnostic_states=len(chosen_D)==4)
    write(OUT/'阶段T/诊断状态.json',chosen_D)
    report=dict(per_checkpoint=results,checks=checks,T_passed=all(checks.values()),protected_success_tasks=protected_success,
        zero_model_calls=True,zero_navigation_replay=True,no_new_SR=True,truth_only_posthoc=True,
        diagnostic_states=chosen_D)
    write(OUT/'阶段T/机会汇总.json',report)
    write(OUT/'阶段T/验收结论.json',dict(T_numeric_passed=report['T_passed'],independent_audit_pending=True,
        allow_D=False,summary_sha256=digest(OUT/'阶段T/机会汇总.json')))
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
