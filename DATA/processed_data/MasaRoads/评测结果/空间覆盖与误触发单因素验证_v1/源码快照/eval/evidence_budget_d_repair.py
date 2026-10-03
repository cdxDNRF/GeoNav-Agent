"""Resume only after audited naming repair, retaining old6 requests and420 totalcap."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.vlm import _parse_json_object
from agents.evidence_budget_vlm import request,parse_evidence,ablate_message
from agents.evidence_reference_repair import parse_plan_repaired,normalize_known_refs
from agents.curl_transport import CurlTransport
from agents.collaboration_pilot import digest_payload
from eval.evidence_budget_stage_t import ROOT,SRC,OUT,read,lines,digest,write
from eval.evidence_budget_stage_d import load_states,ScopedAPI,configuration,plan_status,skipped,diagnostic_summary

FOLDER=OUT/'阶段D格式修复_v2'
TESTS=ROOT/'选题报告相关/证据引用格式修复测试_v2.json'


def query(api,sample,arm,stage,tag,evidence=None):
    state=sample['state'];payload,audit=request(state,sample['images'],api.config,arm,stage,evidence)
    audit['parser_version']='known-typed-fact-reference-v2'
    parser=(lambda text,finish:parse_evidence(text,finish,state,audit['image_labels'])) if stage=='evidence' else (lambda text,finish:parse_plan_repaired(text,finish,state,evidence))
    return api.exchange(payload,audit,tag,parser)


def pair(api,sample,arm,index,repeat):
    tag=dict(phase='D',state=index,repeat=repeat,arm=arm,repair_version=2)
    E,first=query(api,sample,arm,'evidence',dict(tag,slot=0));P=second=None
    if E is not None and not api.circuit_reason:P,second=query(api,sample,arm,'plan',dict(tag,slot=1),E)
    return dict(arm=arm,state=index,repeat=repeat,phase='D',evidence=E,plan=P,
        call_ids=[first['call_id']]+([] if second is None else [second['call_id']]),
        all_responses_valid=first['status']=='ok' and second is not None and second['status']=='ok',
        status=plan_status(P,sample['state']),input_sha256=digest_payload(sample['state'])),E


def main():
    if FOLDER.exists():raise ValueError('immutable repair batch exists')
    tests=read(TESTS)
    if not tests['successful']:raise ValueError('repair tests required')
    samples=load_states();old=list(lines(OUT/'API响应.jsonl'))
    if len(old)!=6 or any(r['http_status']!=200 or r['tag']['phase']!='engineering' for r in old):raise ValueError('only schema failure repair permitted')
    engineering_old=read(OUT/'阶段D/工程接口诊断.json');engineering=[];aliases=[]
    for previous in engineering_old['records']:
        i=previous['state'];r0,r1=[old[c-1] for c in previous['call_ids']]
        E,_=parse_evidence(r0['raw_content'],r0['finish_reason'],samples[i]['state'],r0['image_labels'])
        P,_=parse_plan_repaired(r1['raw_content'],r1['finish_reason'],samples[i]['state'],E)
        raw,_=_parse_json_object(r1['raw_content'],r1['finish_reason']);_,normalizations=normalize_known_refs(raw,E)
        aliases.extend(dict(call_id=r1['call_id'],**n) for n in normalizations)
        engineering.append(dict(previous,state=i,plan=P,all_responses_valid_after_reference_repair=True,
            status=plan_status(P,samples[i]['state']),raw_response_status=r1['status']))
    if len(engineering)!=3 or aliases!=[dict(call_id=6,original='geometry:e0',normalized='e0',verified_fact_kind='geometry')]:
        raise ValueError('repair must target exactly the observed unambiguous naming issue')
    FOLDER.mkdir();cfg=configuration()
    if cfg.public()!=read(OUT/'阶段D/预登记.json')['model']:raise ValueError('service config unchanged')
    code=[SRC/'agents/evidence_reference_repair.py',SRC/'eval/evidence_budget_d_repair.py',SRC/'tests/test_evidence_reference_repair.py']
    for f in code:
        dest=FOLDER/'源码快照'/f.relative_to(SRC);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    write(FOLDER/'工程引用修复复核.json',dict(passed=True,aliases=aliases,reused_chat_requests=6,new_engineering_requests=0,
        old_raw_log_sha256=digest(OUT/'API响应.jsonl'),engineering_records=engineering,old_failure_preserved=True,
        rule='only typed ID whose actual evidence fact kind matches is normalized; allunknown/type mismatched refs rejected'))
    reg=dict(version='ledger-budget-D-reference-repair-v2',utc=datetime.now(timezone.utc).isoformat(),model=cfg.public(),
        previous_D_registration_sha256=digest(OUT/'阶段D/预登记.json'),reference_repair_sha256=digest(FOLDER/'工程引用修复复核.json'),
        reused_engineering_calls=6,max_all_chat_requests=420,unchanged_phase_caps=dict(engineering=12,D=48,G=360),
        source_sha256={f.relative_to(SRC).as_posix():digest(f) for f in code},tests=tests,gate_unchanged=True)
    write(FOLDER/'预登记.json',reg)
    api=ScopedAPI(cfg,CurlTransport(60));api.total_calls=len(old);api.phase_counts=dict(Counter(r['tag']['phase'] for r in old))
    records=[]
    for rep in (0,1):
        for i,sample in enumerate(samples):
            if api.circuit_reason:
                records.extend(skipped(i,rep,a,api.circuit_reason) for a in ('M2','M4','M4_ablation'));continue
            order=['M2','M4'] if (i+rep)%2==0 else ['M4','M2'];full_E=None;full=None
            for arm in order:
                if api.circuit_reason:r=skipped(i,rep,arm,api.circuit_reason);E=None
                else:r,E=pair(api,sample,arm,i,rep)
                records.append(r)
                if arm=='M4':full_E=E;full=r
            kind=['hidden','shuffled','hidden','shuffled'][i]
            if full_E is None or api.circuit_reason:ab=skipped(i,rep,'M4_ablation','no_usable_full_E_or_circuit')
            else:
                changed=ablate_message(full_E,kind)
                P,r=query(api,sample,'M4','plan',dict(phase='D',state=i,repeat=rep,arm='M4_ablation',slot=1,repair_version=2),changed)
                ab=dict(arm='M4_ablation',state=i,repeat=rep,phase='D',evidence=changed,plan=P,call_ids=[r['call_id']],
                    shared_E_call_id=full['call_ids'][0],ablation=kind,all_responses_valid=r['status']=='ok',
                    status=plan_status(P,sample['state']),input_sha256=digest_payload(sample['state']))
            records.append(ab);print(dict(D_state=i,API_repeat=rep,requests=api.total_calls,circuit=api.circuit_reason),flush=True)
    write(FOLDER/'逐状态决策.json',records)
    result=diagnostic_summary(records,dict(passed=True));result.update(chat_requests=api.total_calls,phase_counts=api.phase_counts,
        circuit=api.circuit_reason,old_engineering_invalid_before_repair=1,new_engineering_requests=0)
    write(FOLDER/'机制汇总.json',result)
    write(FOLDER/'执行状态.json',dict(status='D_completed',D_numeric_passed=result['D_numeric_passed'],
        chat_requests=api.total_calls,G_started=False,G_reason='pending_audit' if result['D_numeric_passed'] else 'message_gate_failed'))
    print(result,flush=True)


if __name__=='__main__':main()
