"""Independent static check of trigger arithmetic and ledger provenance; no inference."""
from collections import Counter
import math
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.evidence_budget_stage_t import ROOT,SRC,OLD,OUT,read,lines,digest,write,select_tasks


def check(value,msg):
    if not value:raise ValueError('T audit: '+msg)


def audit():
    reg=read(OUT/'阶段T/预登记.json');summary=read(OUT/'阶段T/机会汇总.json');prefixes=0;rows0={};eligible0={}
    for n,h in reg['sources_sha256'].items():check(digest(ROOT/n)==h,'old input hash')
    for n,h in reg['source_sha256'].items():check(digest(SRC/n)==h and digest(OUT/'源码快照'/n)==h,'source snapshot')
    for n,h in reg['frozen_files_sha256'].items():check(digest(OUT/n)==h,'frozen doc/task')
    check(read(OUT/'总体任务.json')==select_tasks(read(OLD/'导航任务.json')),'balanced score-free G selection')
    for seed in (0,1,2):
        saved={r['episode_id']:r for r in read(OUT/f'阶段T/逐题机会_s{seed}.json')};old_count=new_count=fail_old=fail_new=success_new=0;failure_sources=set()
        for row in lines(OLD/f'Edge_s{seed}/导航_CueFull_轨迹.jsonl'):
            accepted=False;first_new=None;first_old=None
            for i,d in enumerate(row['decisions']):
                prefix=row['trajectory'][:i+1];visited=[x['patch_id'] for x in prefix]
                check(visited==d['public_visited'] and list(divmod(visited[-1],10))==d['public_position'],'public state versus past trajectory')
                check(20-i==d['remaining_budget'] and d['step']==i+1,'budget and input index')
                repeat=any(x in visited[:j] for j,x in enumerate(visited) if j>=len(visited)-2)
                new=sum(x not in visited[:j] for j,x in enumerate(visited) if j>=len(visited)-2) if len(visited)>=3 else 2
                stagnant=len(visited)>=3 and repeat and new<=1
                checkpoint=i>=7 and not accepted
                due=d['cue_action'] is None and d['remaining_budget']>=2 and (stagnant or checkpoint)
                # Independent enumeration checks existence of a new cell reachable in <=2 legal moves.
                r,c=d['public_position'];known=set(visited);reachable=set()
                for dr,dc in [(-1,0),(0,1),(1,0),(0,-1)]:
                    a,b=r+dr,c+dc
                    if 0<=a<10 and 0<=b<10:
                        reachable.add(a*10+b)
                        for er,ec in [(-1,0),(0,1),(1,0),(0,-1)]:
                            if 0<=a+er<10 and 0<=b+ec<10:reachable.add((a+er)*10+b+ec)
                executable=bool(reachable-known)
                if due and executable and first_new is None:first_new=dict(elapsed=i,remaining=d['remaining_budget'],reason='stagnation' if stagnant else 'checkpoint',index=i)
                if d['cue_action'] is not None and d['cue_action']!=d['explorer_action'] and first_old is None:first_old=i
                accepted|=d['cue_action'] is not None;prefixes+=1
            s=saved[row['episode_id']]
            check(s['new_trigger']==(first_new is not None) and s['old_trigger']==(first_old is not None),'earliest trigger')
            if first_new:
                check(all(s[k]==first_new[k] for k in ('elapsed','remaining','reason')),'remaining and classification')
                new_count+=1
                if row['success']:success_new+=1
                else:fail_new+=1;failure_sources.add(s['source'])
            if first_old is not None:
                old_count+=1;fail_old+=not row['success']
            if seed==0:rows0[row['episode_id']]=row;eligible0[row['episode_id']]=first_new
        aggregate=summary['per_checkpoint'][str(seed)]
        check((old_count,new_count,fail_old,fail_new,success_new,len(failure_sources))==tuple(aggregate[k] for k in
              ('old_trigger_tasks','new_trigger_tasks','old_failure_trigger_tasks','new_failure_trigger_tasks','new_success_trigger_tasks','new_failure_source_count')),'independent aggregate')
    ds=read(OUT/'阶段T/诊断状态.json');sources=set()
    for d in ds:
        row=rows0[d['episode_id']];check(d['source_key'] not in sources,'four separate source keys');sources.add(d['source_key'])
        check(d['baseline_success']==row['success'],'posthoc diagnostic stratum')
        if d['kind']=='T_failure':check(not row['success'] and eligible0[d['episode_id']]['index']==d['index'],'failure T prefix')
        else:check(d['index']==2,'fixed clock diagnostic')
    check(summary['T_passed']==all(summary['checks'].values()),'gate')
    receipt=dict(passed=True,episodes=1500,public_prefixes=prefixes,zero_model_calls=True,zero_navigation_replay=True,
        no_new_SR=True,default_unchanged=digest(ROOT/'project/local_policy_default.json')==reg['default_sha256'],
        registration_sha256=digest(OUT/'阶段T/预登记.json'),summary_sha256=digest(OUT/'阶段T/机会汇总.json'),
        audit_authorship='separate implementation by same executing agent')
    write(OUT/'阶段T/独立复核.json',receipt)
    write(OUT/'阶段T/放行结论.json',dict(allow_D=summary['T_passed'],audit_passed=True,audit_sha256=digest(OUT/'阶段T/独立复核.json'),
        summary_sha256=digest(OUT/'阶段T/机会汇总.json')))
    print(receipt)


if __name__=='__main__':audit()
