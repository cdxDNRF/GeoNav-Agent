"""One bounded format clarification within original phase caps; no threshold tuning."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.evidence_source_clarification import clarified_query
from agents.evidence_budget_vlm import ablate_message
from agents.curl_transport import CurlTransport
from agents.collaboration_pilot import digest_payload
from eval.evidence_budget_stage_t import ROOT,SRC,OUT,read,lines,digest,write
from eval.evidence_budget_stage_d import load_states,ScopedAPI,configuration,plan_status,skipped,diagnostic_summary

FOLDER=OUT/'阶段D来源格式澄清_v3'
TESTS=ROOT/'选题报告相关/证据来源格式澄清测试_v3.json'


def pair(api,sample,arm,phase,index,repeat):
    tag=dict(phase=phase,state=index,repeat=repeat,arm=arm,format_version=3)
    E,r0=clarified_query(api,sample,arm,'evidence',dict(tag,slot=0));P=r1=None
    if E is not None and not api.circuit_reason:P,r1=clarified_query(api,sample,arm,'plan',dict(tag,slot=1),E)
    return dict(arm=arm,state=index,repeat=repeat,phase=phase,evidence=E,plan=P,
        call_ids=[r0['call_id']]+([] if r1 is None else [r1['call_id']]),
        all_responses_valid=r0['status']=='ok' and r1 is not None and r1['status']=='ok',
        status=plan_status(P,sample['state']),input_sha256=digest_payload(sample['state'])),E


def main():
    if FOLDER.exists():raise ValueError('immutable clarification exists')
    tests=read(TESTS)
    if not tests['successful']:raise ValueError('clarification tests required')
    old=list(lines(OUT/'API响应.jsonl'));spent=Counter(r['tag']['phase'] for r in old)
    if len(old)!=13 or spent!={'engineering':6,'D':7} or any(r.get('http_status')!=200 for r in old):
        raise ValueError('format-only repair precondition; do not reopen provider errors')
    samples=load_states();cfg=configuration()
    if cfg.public()!=read(OUT/'阶段D/预登记.json')['model']:raise ValueError('same model/limits required')
    FOLDER.mkdir();code=[SRC/'agents/evidence_source_clarification.py',SRC/'eval/evidence_budget_d_clarification.py',SRC/'tests/test_evidence_source_clarification.py']
    for f in code:
        p=FOLDER/'源码快照'/f.relative_to(SRC);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(f.read_bytes())
    write(FOLDER/'预登记.json',dict(version='source-catalog-format-v3',utc=datetime.now(timezone.utc).isoformat(),
        previous_chat_requests=13,previous_phase_counts=dict(spent),remaining_engineering_cap=6,additional_D_cap=40,
        total_cap=420,phase_caps=dict(engineering=12,D=48,G=360),no_further_format_version_this_batch=True,
        same_T_candidates_images_and_mechanism_gate=True,strict_source_validation=True,model=cfg.public(),tests=tests,
        source_sha256={f.relative_to(SRC).as_posix():digest(f) for f in code},
        previous_response_log_sha256=digest(OUT/'API响应.jsonl')))
    api=ScopedAPI(cfg,CurlTransport(60));api.total_calls=len(old);api.phase_counts=dict(spent);engineering_rows=[]
    for i in (0,1,2):
        if api.circuit_reason:break
        r,_=pair(api,samples[i],'M4','engineering',i,None);engineering_rows.append(r)
        print(dict(engineering_state=i,valid=r['all_responses_valid'],requests=api.total_calls),flush=True)
        if not r['all_responses_valid']:
            api.circuit_reason=api.circuit_reason or 'clarified_engineering_failed';break
    engineering=dict(passed=len(engineering_rows)==3 and all(r['all_responses_valid'] for r in engineering_rows),records=engineering_rows)
    write(FOLDER/'工程接口诊断.json',engineering)
    records=[]
    if not engineering['passed']:api.circuit_reason=api.circuit_reason or 'clarified_engineering_failed'
    for rep in (0,1):
        for i,sample in enumerate(samples):
            if api.circuit_reason:
                records.extend(skipped(i,rep,a,api.circuit_reason) for a in ('M2','M4','M4_ablation'));continue
            full=full_E=None
            for arm in (['M2','M4'] if (rep+i)%2==0 else ['M4','M2']):
                if api.circuit_reason:r=skipped(i,rep,arm,api.circuit_reason);E=None
                else:r,E=pair(api,sample,arm,'D',i,rep)
                records.append(r)
                if arm=='M4':full=r;full_E=E
            kind=['hidden','shuffled','hidden','shuffled'][i]
            if full_E is None or api.circuit_reason:ab=skipped(i,rep,'M4_ablation','no_usable_full_E_or_circuit')
            else:
                altered=ablate_message(full_E,kind)
                P,r=clarified_query(api,sample,'M4','plan',dict(phase='D',state=i,repeat=rep,arm='M4_ablation',slot=1,format_version=3),altered)
                ab=dict(arm='M4_ablation',state=i,repeat=rep,phase='D',evidence=altered,plan=P,call_ids=[r['call_id']],
                    shared_E_call_id=full['call_ids'][0],ablation=kind,all_responses_valid=r['status']=='ok',
                    status=plan_status(P,sample['state']),input_sha256=digest_payload(sample['state']))
            records.append(ab);print(dict(D_state=i,API_repeat=rep,requests=api.total_calls,circuit=api.circuit_reason),flush=True)
    write(FOLDER/'逐状态决策.json',records);summary=diagnostic_summary(records,engineering)
    summary.update(chat_requests=api.total_calls,phase_counts=api.phase_counts,circuit=api.circuit_reason,
        previous_versions_preserved=True,engineering_budget_exhausted=api.phase_counts.get('engineering',0)>=12)
    write(FOLDER/'机制汇总.json',summary)
    write(FOLDER/'执行状态.json',dict(status='D_completed',D_numeric_passed=summary['D_numeric_passed'],chat_requests=api.total_calls,
        G_started=False,G_reason='pending_audit' if summary['D_numeric_passed'] else 'D_gate_failed'))
    print(summary,flush=True)


if __name__=='__main__':main()
