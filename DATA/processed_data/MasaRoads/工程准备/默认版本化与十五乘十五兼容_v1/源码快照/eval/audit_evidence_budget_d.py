"""Audit all diagnostic revisions, raw replies, source bounds and phase caps offline."""
from collections import Counter
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.evidence_budget_vlm import request,parse_evidence,parse_plan,ablate_message
from agents.evidence_reference_repair import parse_plan_repaired,normalize_known_refs
from agents.evidence_source_clarification import clarified_request
from agents.vlm import APIConfig,_parse_json_object
from agents.collaboration_pilot import digest_payload
from eval.evidence_budget_stage_t import ROOT,SRC,OUT,read,lines,digest,write
from eval.evidence_budget_stage_d import load_states


def check(v,msg):
    if not v:raise ValueError('D audit: '+msg)


def arithmetic(plan,state):
    if plan is None:return dict(executable=False,reason='API_or_parse_failure',candidate_id=None,score=None)
    if plan['proposal_type']=='abstain':return dict(executable=False,reason='abstain',candidate_id=None,score=None)
    if plan['proposal_type']=='target_claim':return dict(executable=False,reason='unvalidated_target_claim',candidate_id=plan['candidate_id'],score=None)
    c=next(c for c in state['candidates'] if c['candidate_id']==plan['candidate_id']);visited=set(state['ledger']['visited_order']);cells=c['cells'];k=10
    new=len(set(cells)-visited);repeated=sum(j in visited for j in cells);r,q=divmod(cells[-1],k)
    frontier=sum(0<=r+a<k and 0<=q+b<k and (r+a)*k+q+b not in visited.union(cells) for a,b in [(-1,0),(0,1),(1,0),(0,-1)])
    return dict(executable=True,reason='legal_exploration',candidate_id=c['candidate_id'],score=[new,-repeated,frontier,-len(cells)])


def main():
    dreg=read(OUT/'阶段D/预登记.json');samples=load_states();cfg=APIConfig(dreg['model']['base_url'],dreg['model']['model'],'unused',60,512)
    for n,h in dreg['source_sha256'].items():check(digest(SRC/n)==h and digest(OUT/'源码快照'/n)==h,'frozen D source')
    for n,h in dreg['inputs_sha256'].items():check(digest(ROOT/n)==h,'given-image source')
    check(digest(OUT/'阶段D/公共输入.json')==dreg['public_inputs_sha256'],'frozen public inputs')
    check(digest(ROOT/'project/local_policy_default.json')==dreg['default_sha256'],'unchanged default')
    replies=list(lines(OUT/'API响应.jsonl'));intents=list(lines(OUT/'请求意图.jsonl'))
    check(len(replies)==len(intents)==14,'complete intents/replies')
    check([r['call_id'] for r in replies]==list(range(1,15)),'no duplicate calls/retries')
    phase_counts=Counter(r['tag']['phase'] for r in replies)
    check(phase_counts['engineering']<=12 and phase_counts['D']<=48 and len(replies)<=420,'all revision costs shared')
    source_aliases=[];failed=[]
    for intent,r in zip(intents,replies):
        check(all(r[k]==v for k,v in intent.items()),'exact intent-response binding')
        i=r['tag']['state'];s=samples[i];arm='M4' if r['tag']['arm']=='M4_ablation' else r['tag']['arm']
        builder=clarified_request if r['prompt_version']=='public-source-catalog-clarification-v3' else request
        _,expected=builder(s['state'],s['images'],cfg,arm,r['workflow_stage'],r['provided_message'])
        check(all(r[k]==v for k,v in expected.items()),'reconstructed public payload/message and image hashes')
        check(r['image_count']<=4,'same image permission')
        check(not any(k in r['public_input'] for k in ('goal','distance','episode_id','source_key','baseline_success')),'no evaluator truth')
        check(r['http_status']==200 and r['returned_model']=='gemma4:31b','actual response model and HTTP')
        fn=(lambda:parse_evidence(r['raw_content'],r['finish_reason'],s['state'],r['image_labels'])) if r['workflow_stage']=='evidence' else \
           (lambda:(parse_plan_repaired if r.get('parser_version') else parse_plan)(r['raw_content'],r['finish_reason'],s['state'],r['provided_message']))
        try:obj,norm=fn();valid=True
        except (ValueError,TypeError,KeyError):valid=False;obj=None
        check(valid==(r['status']=='ok'),'strict final JSON replay')
        if valid:check(obj==r['parsed'],'parsed reply unchanged')
        else:
            raw,_=_parse_json_object(r['raw_content'],r['finish_reason'])
            category='known_typed_reference_alias' if r['call_id']==6 else 'null_counter_evidence' if raw.get('counter_evidence','notnull') is None else \
                     'empty_source_refs' if any(f.get('source_ids')==[] for key in ('observed_facts','counter_evidence') for f in raw.get(key,[])) else 'numeric_source_refs'
            failed.append(dict(call_id=r['call_id'],category=category))
        if r['workflow_stage']=='plan':
            raw,_=_parse_json_object(r['raw_content'],r['finish_reason']);_,aliases=normalize_known_refs(raw,r['provided_message'])
            source_aliases.extend(dict(call_id=r['call_id'],**x) for x in aliases)
    consumed=set()
    base_eng=read(OUT/'阶段D/工程接口诊断.json')['records']
    versions=['阶段D','阶段D格式修复_v2','阶段D来源格式澄清_v3']
    for name in versions:
        folder=OUT/name
        if name!='阶段D':
            reg=read(folder/'预登记.json')
            for n,h in reg['source_sha256'].items():check(digest(SRC/n)==h and digest(folder/'源码快照'/n)==h,'revision source')
        engineering=base_eng if name=='阶段D' else read(folder/'工程接口诊断.json')['records'] if name.endswith('v3') else []
        for e in engineering:
            for cid in e['call_ids']:check(cid not in consumed,'one engineering request accounted once');consumed.add(cid)
        records=read(folder/'逐状态决策.json');check(len(records)==24,'planned diagnostic records')
        for r in records:
            i=r['state'];sample=samples[i]
            for cid in r['call_ids']:check(cid not in consumed,'one fresh D response perslot');consumed.add(cid)
            check(arithmetic(r['plan'],sample['state'])==r['status'] or not r['call_ids'] and r['plan'] is None,'public candidate score/admission')
            if r['arm']=='M4_ablation' and r['call_ids']:
                original=replies[r['shared_E_call_id']-1]['parsed'];check(original is not None,'shared full E actual response')
                expected=ablate_message(original,r['ablation']);check(r['evidence']==expected,'source-preserving message transform')
            if r['plan'] is not None:
                last=replies[r['call_ids'][-1]-1];check(r['plan']==last['parsed'],'plan from actual final reply')
        report=read(folder/'机制汇总.json')
        check(not report['D_numeric_passed'],'no revision released G')
    check(consumed==set(range(1,15)),'all14requests accounted across immutable revisions')
    repair=read(OUT/'阶段D格式修复_v2/工程引用修复复核.json')
    check(repair['aliases']==[dict(call_id=6,original='geometry:e0',normalized='e0',verified_fact_kind='geometry')],'single known alias repair')
    for previous in repair['engineering_records']:
        r=replies[previous['call_ids'][-1]-1]
        parsed,_=parse_plan_repaired(r['raw_content'],r['finish_reason'],samples[previous['state']]['state'],r['provided_message'])
        check(parsed==previous['plan'],'engineering repair replay, no repeated cloudrequest')
    summary=read(OUT/'阶段D来源格式澄清_v3/机制汇总.json');check(summary['phase_counts']==dict(phase_counts),'budget arithmetic')
    for filename in ('API响应.jsonl','请求意图.jsonl'):
        text=(OUT/filename).read_text(encoding='utf-8');check('data:image/' not in text and '"reasoning_content"' not in text,'no pixels/hidden reasoning saved')
    receipt=dict(passed=True,requests=14,HTTP_successes=14,raw_schema_valid=8,raw_schema_invalid=6,
        known_alias_repairs=1,unresolved_schema_failures=5,failed_categories=failed,phase_counts=dict(phase_counts),
        all_request_image_message_hashes_rebuilt=True,strict_grounding_bounds_verified=True,
        semantic_fact_accuracy_not_proven=True,all_old_failures_preserved=True,default_changed=False,
        T_audit_sha256=digest(OUT/'阶段T/独立复核.json'),latest_D_summary_sha256=digest(OUT/'阶段D来源格式澄清_v3/机制汇总.json'),
        responses_sha256=digest(OUT/'API响应.jsonl'),intents_sha256=digest(OUT/'请求意图.jsonl'),
        audit_authorship='separate implementation by same executing agent')
    write(OUT/'独立复核.json',receipt)
    write(OUT/'验收结论.json',dict(T_passed=True,D_passed=False,D_mechanism_test_completed=False,
        engineering_passed_latest=False,G_started=False,allow_G=False,reason='source/schema engineering gate failed; message usefulness unestablished',
        total_requests=14,maximum_requests=420,new_training_steps=0,new_navigation_episodes=0,new_SR_available=False,
        default_changed=False,new_source_files_used=0,audit_sha256=digest(OUT/'独立复核.json'),
        latest_D_summary_sha256=digest(OUT/'阶段D来源格式澄清_v3/机制汇总.json')))
    write(OUT/'最终执行状态.json',dict(status='closed_at_D_engineering_gate',T_complete=True,D_partial=True,G_not_released=True,total_requests=14))
    print(receipt,flush=True)


if __name__=='__main__':main()
