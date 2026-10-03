"""Offline receipt audit with independent path arithmetic and message-gate recomputation."""
from collections import Counter
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.evidence_compact_adapter import compact_request,parse_compact_evidence,parse_compact_plan,replay_legacy,ablate_compact
from agents.collaboration_pilot import digest_payload
from agents.vlm import APIConfig
from eval.evidence_compact_d import OUT,PREVIOUS,ROOT,SRC,read,lines,write,digest
from eval.evidence_budget_stage_d import load_states


def check(v,msg):
    if not v:raise ValueError('compact audit: '+msg)


def independent_status(plan,state):
    if plan is None:return dict(executable=False,reason='API_or_parse_failure',candidate_id=None,score=None)
    if plan['proposal_type']=='abstain':return dict(executable=False,reason='abstain',candidate_id=None,score=None)
    check(plan['proposal_type']=='exploration_plan','target claim not authorized')
    c=next(c for c in state['candidates'] if c['candidate_id']==plan['candidate_id'])
    delta={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}
    k=state['grid_size'];known=set(state['ledger']['visited_order']);r,q=state['position'];path=[]
    for a in c['actions']:
        dr,dc=delta[a];r+=dr;q+=dc;check(0<=r<k and 0<=q<k,'boundary');path.append(r*k+q)
    new=len(set(path)-known);revisit=sum(x in known for x in path)
    frontier=sum(0<=r+dr<k and 0<=q+dc<k and (r+dr)*k+q+dc not in known.union(path) for dr,dc in delta.values())
    check(c['cells']==path and c['anticipated_new_cells']==new and plan['expected_new_cells']==new,'derived coverage')
    check('g:'+c['candidate_id'] in plan['evidence_ids'],'explicit selected geometry citation')
    return dict(executable=True,reason='legal_exploration',candidate_id=c['candidate_id'],score=[new,-revisit,frontier,-len(path)])


def main():
    reg=read(OUT/'预登记.json');samples=load_states();cfg=APIConfig(reg['model']['base_url'],reg['model']['model'],'unused',60,512)
    for n,h in reg['source_sha256'].items():check(digest(SRC/n)==h and digest(OUT/'源码快照'/n)==h,'frozen implementation')
    for n,h in reg['frozen_core_sha256'].items():check(digest(SRC/n)==h,'frozen core')
    for n,h in reg['protected_files_sha256'].items():check(digest(ROOT/n)==h,'protected artifact/input/weight '+n)
    check(digest(ROOT/'project/local_policy_default.json')==reg['default_sha256'],'default unchanged')
    check(digest(OUT/'阶段D/公共输入.json')==reg['public_inputs_sha256'],'frozen public inputs')
    from hashlib import sha256
    check(read(OUT/'阶段D/公共输入.json')==[dict(state=s['state'],image_labels=[x[0] for x in s['images']],
        image_sha256=[sha256(p).hexdigest() for _,p in s['images']]) for s in samples],'actual diagnostic inputs match freeze')
    check(digest(ROOT/'选题报告相关/证据账本简化接口执行协议_v2.md')==reg['protocol_sha256'],'frozen protocol')
    check(digest(PREVIOUS/'阶段T/放行结论.json')==reg['previous_T_release_sha256'],'T release unchanged')
    old=list(lines(PREVIOUS/'API响应.jsonl'));replays=read(OUT/'离线重放/逐响应判定.json');check(len(old)==len(replays)==14,'14 legacy replies')
    for r,rec in zip(old,replays):
        check(rec['call_id']==r['call_id'] and digest_payload(r)==rec['raw_response_sha256'],'legacy exact raw binding')
        try:obj,receipt,fence=replay_legacy(r);accepted=True
        except (ValueError,TypeError,KeyError):accepted=False
        check(accepted==rec['accepted'],'legacy replay decision')
        if accepted:check(rec['normalized']==obj and rec['receipt']==receipt and rec['markdown_fence']==fence,'legacy repair receipt')
    replies=list(lines(OUT/'API响应.jsonl'));intents=list(lines(OUT/'请求意图.jsonl'))
    check(len(replies)==len(intents) and [r['call_id'] for r in replies]==list(range(1,len(replies)+1)),'exact contiguous request log')
    counts=Counter(r['tag']['phase'] for r in replies)
    check(len(replies)<=406 and 14+len(replies)<=420 and all(counts[k]<=v for k,v in reg['phase_caps'].items()),'combined budgets')
    for intent,r in zip(intents,replies):
        check(all(r[k]==v for k,v in intent.items()),'intent/response binding')
        check(r['tag']['global_call_id']==14+r['call_id'],'combined call identity')
        i=r['tag']['state'];s=samples[i];arm='M4' if r['tag']['arm']=='M4_ablation' else r['tag']['arm']
        _,expected=compact_request(s['state'],s['images'],cfg,arm,r['workflow_stage'],r['provided_message'],r['previous_plan'])
        check(all(r[k]==v for k,v in expected.items()),'reconstructed request and actual image/message hashes')
        check(r['image_count']<=4 and not any(x in r['public_input'] for x in ('goal','distance','episode_id','source_tile','baseline_success')),'public input permissions')
        fn=(lambda:parse_compact_evidence(r['raw_content'],r['finish_reason'],s['state'],r['image_labels'])) if r['workflow_stage']=='evidence' else \
           (lambda:parse_compact_plan(r['raw_content'],r['finish_reason'],s['state'],r['provided_message']))
        try:obj,_=fn();valid=True
        except (ValueError,KeyError,TypeError):valid=False;obj=None
        # Transport failures carry no final content; handle them without inventing schema success.
        check(valid==(r['status']=='ok'),'raw reply replay')
        if valid:check(obj==r['parsed'],'derived parsed receipt')
    eng=read(OUT/'阶段D/工程接口诊断.json');records=read(OUT/'阶段D/逐状态决策.json');check(len(records)==24,'24 planned D records')
    used=set()
    for row in eng['records']+records:
        sample=samples[row['state']]
        if row['call_ids']:
            expected_slots=1 if row['arm']=='M4_ablation' else 2
            all_valid=len(row['call_ids'])==expected_slots and all(replies[cid-1]['status']=='ok' for cid in row['call_ids'])
            check(row['all_responses_valid']==all_valid,'all expected responses valid')
            check(row['status']==independent_status(row['plan'],sample['state']),'plan legality/score')
            for cid in row['call_ids']:
                check(cid not in used,'request counted exactly once');used.add(cid)
                actual=replies[cid-1];check(actual['tag']['state']==row['state'] and actual['tag']['arm']==row['arm'] and actual['tag']['repeat']==row['repeat'],'diagnostic slot binding')
            if row['plan'] is not None:check(row['plan']==replies[row['call_ids'][-1]-1]['parsed'],'actual final plan')
            if row['arm']!='M4_ablation':check(row['evidence']==replies[row['call_ids'][0]-1].get('parsed'),'actual E message')
            if row['arm']=='M4_ablation':
                original=replies[row['shared_E_call_id']-1]
                check(original['tag']['arm']=='M4' and original['tag']['state']==row['state'] and original['tag']['repeat']==row['repeat'],'ablation matched original E')
                check(row['evidence']==ablate_compact(original['parsed'],reg['ablations'][row['state']]),'preassigned actual message transform')
    check(used==set(range(1,len(replies)+1)),'all requests accounted')
    eng_pass=len(eng['records'])==3 and all(r['all_responses_valid'] and r['status']['executable'] for r in eng['records'])
    check(eng_pass==eng['passed'],'engineering gate')
    effects=[]
    for i in range(4):
        p=[]
        for rep in (0,1):
            full=next(r for r in records if (r['state'],r['repeat'],r['arm'])==(i,rep,'M4'))
            ab=next(r for r in records if (r['state'],r['repeat'],r['arm'])==(i,rep,'M4_ablation'))
            fs,ass=full['status'],ab['status']
            p.append(dict(repeat=rep,full_candidate=fs['candidate_id'],ablation_candidate=ass['candidate_id'],
                changed=fs['candidate_id']!=ass['candidate_id'],
                full_strictly_better=bool(fs['executable'] and (not ass['executable'] or tuple(fs['score'])>tuple(ass['score']))),
                message_reported_used=bool(full['plan'] and full['plan']['consumed_message_id']=='m0')))
        useful=all(x['changed'] and x['full_strictly_better'] and x['message_reported_used'] for x in p)
        useful &= len({x['full_candidate'] for x in p})==1 and len({x['ablation_candidate'] for x in p})==1
        effects.append(dict(state=i,pairs=p,repeated_useful_message_effect=bool(useful)))
    fulls=[r for r in records if r['arm']=='M4']
    checks=dict(engineering_passed=eng_pass,all_D_valid=all(r['all_responses_valid'] for r in records),
        grounded_geometry_message_nonempty=all(bool(r['evidence'] and r['evidence']['ranked_candidates']) for r in fulls),
        two_failure_states_executable=all(r['status']['executable'] for r in fulls if r['state'] in (0,1)),
        no_unvalidated_target_claim=all(r['status']['reason']!='unvalidated_target_claim' for r in records),
        repeated_useful_message_effect=any(e['repeated_useful_message_effect'] for e in effects))
    summary=read(OUT/'阶段D/机制汇总.json')
    check(summary['D_mechanism_test_completed']==all(len(r['call_ids'])==(1 if r['arm']=='M4_ablation' else 2) for r in records),'complete diagnostic calls')
    check(summary['checks']==checks and summary['paired_message_effects']==effects and summary['D_numeric_passed']==all(checks.values()),'independent D gate')
    check(summary['new_requests']==len(replies) and summary['combined_requests']==14+len(replies) and summary['phase_counts']==dict(counts),'request totals')
    audit=dict(passed=True,protected_files=len(reg['protected_files_sha256']),core_files=len(reg['frozen_core_sha256']),
        old_raw_responses=14,new_raw_responses=len(replies),planned_D_records=24,
        HTTP_successes=sum(r.get('http_status')==200 for r in replies),schema_valid=sum(r['status']=='ok' for r in replies),
        normalized_responses=sum(bool(r.get('parsed',{}).get('normalization_receipt')) for r in replies),
        repeated_useful_message_states=sum(e['repeated_useful_message_effect'] for e in effects),
        independent_checks=checks,new_training_steps=0,new_navigation_episodes=0,new_SR_available=False,
        default_changed=False,visual_semantic_accuracy_proven=False)
    write(OUT/'独立复核.json',audit)
    verdict=dict(T_passed=True,engineering_passed=eng_pass,D_passed=all(checks.values()),
        D_mechanism_test_completed=bool(summary['D_mechanism_test_completed']),allow_G=all(checks.values()),G_started=False,
        new_requests=len(replies),combined_requests=14+len(replies),maximum_combined_requests=420,
        new_navigation_episodes=0,new_training_steps=0,new_SR_available=False,default_changed=False,
        audit_sha256=digest(OUT/'独立复核.json'),failed_checks=[k for k,v in checks.items() if not v])
    write(OUT/'验收结论.json',verdict)
    write(OUT/'最终执行状态.json',dict(status='ready_for_G' if verdict['allow_G'] else 'closed_at_D_message_gate' if eng_pass else 'closed_at_engineering_gate',
        **verdict))
    print(audit,flush=True);print(verdict,flush=True)


if __name__=='__main__':main()
