"""A separate frozen compact-interface batch; historical judgments remain immutable."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.collaboration_pilot import PilotAPI,digest_payload
from agents.curl_transport import CurlTransport
from agents.evidence_compact_adapter import (VERSION,PROMPTS,E_INSTRUCTION,P_INSTRUCTION,compact_query,
    geometry_catalog,ablate_compact,replay_legacy)
from eval.evidence_budget_stage_t import ROOT,SRC,OUT as PREVIOUS,read,lines,write,digest
from eval.evidence_budget_stage_d import load_states,configuration,plan_status,skipped

OUT=ROOT/'DATA/processed_data/Masa/评测结果/证据账本简化接口验证_v2'
DOC=ROOT/'选题报告相关/证据账本简化接口执行协议_v2.md'
TESTS=ROOT/'选题报告相关/证据账本简化接口测试_v2.json'


class CompactAPI(PilotAPI):
    def __init__(self,cfg):
        super().__init__(cfg,CurlTransport(60),OUT,max_calls=406)
        self.phase_counts=Counter();self.caps={'engineering':6,'D':40,'G':360}

    def exchange(self,payload,audit,tag,parser):
        phase=tag['phase']
        if self.phase_counts[phase]>=self.caps[phase]:raise RuntimeError('phase budget closed')
        self.phase_counts[phase]+=1
        return super().exchange(payload,audit,dict(tag,global_call_id=14+self.total_calls+1),parser)


def prepare():
    if OUT.exists():raise ValueError('immutable new batch exists')
    tests=read(TESTS)
    if not tests['successful']:raise ValueError('offline tests required')
    old=list(lines(PREVIOUS/'API响应.jsonl'))
    if len(old)!=14 or read(PREVIOUS/'最终执行状态.json')['status']!='closed_at_D_engineering_gate':raise ValueError('historical batch must remain closed')
    if not read(PREVIOUS/'阶段T/放行结论.json')['allow_D']:raise ValueError('T not passed')
    cfg=configuration();samples=load_states()
    OUT.mkdir();(OUT/'离线重放').mkdir();(OUT/'阶段D').mkdir()
    for name,path in [('执行协议.md',DOC),('测试记录.json',TESTS),('总体任务.json',PREVIOUS/'总体任务.json')]:
        (OUT/name).write_bytes(path.read_bytes())
    public=[dict(state=s['state'],image_labels=[x[0] for x in s['images']],
                 image_sha256=[__import__('hashlib').sha256(p).hexdigest() for _,p in s['images']]) for s in samples]
    write(OUT/'阶段D/公共输入.json',public)
    replay=[]
    for r in old:
        item=dict(call_id=r['call_id'],raw_schema_status=r['status'],raw_response_sha256=digest_payload(r),
                  new_navigation_SR=False)
        try:
            obj,receipt,fence=replay_legacy(r);item.update(accepted=True,normalized=obj,receipt=receipt,markdown_fence=fence)
        except (ValueError,TypeError,KeyError) as e:item.update(accepted=False,receipt=[],rejection=str(e))
        replay.append(item)
    write(OUT/'离线重放/逐响应判定.json',replay)
    write(OUT/'离线重放/汇总.json',dict(responses=14,original_valid=sum(r['status']=='ok' for r in old),
        accepted_after_equivalent_representation=sum(r['accepted'] for r in replay),
        repaired_call_ids=[r['call_id'] for r in replay if r['accepted'] and r['receipt']],
        rejected_call_ids=[r['call_id'] for r in replay if not r['accepted']],
        old_judgments_unchanged=True,new_requests=0,not_a_test_of_new_compact_prompt=True))
    files=[SRC/'agents/evidence_compact_adapter.py',SRC/'eval/evidence_compact_d.py',SRC/'eval/audit_evidence_compact.py',
           SRC/'tests/test_evidence_compact_adapter.py']
    for f in files:
        p=OUT/'源码快照'/f.relative_to(SRC);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(f.read_bytes())
    # Protect all previous experiment artifacts and the executed local dependencies.
    protected={p.relative_to(ROOT).as_posix():digest(p) for p in PREVIOUS.rglob('*') if p.is_file()}
    for rel,h in read(PREVIOUS/'阶段T/预登记.json')['sources_sha256'].items():
        if digest(ROOT/rel)!=h:raise ValueError('T referenced source changed')
        protected[rel]=h
    core={p.relative_to(SRC).as_posix():digest(p) for p in SRC.rglob('*.py')
          if p.relative_to(SRC).parts[0] in ('agents','env','data','train')}
    defaults=read(ROOT/'project/local_policy_default.json')
    for entry in defaults['checkpoints']+defaults['cue_heads']+list(defaults['means'].values()):
        p=ROOT/entry['path'];protected[p.relative_to(ROOT).as_posix()]=digest(p)
    for s in samples:
        for p in s['input_paths']:protected[p.relative_to(ROOT).as_posix()]=digest(p)
    reg=dict(version=VERSION,utc=datetime.now(timezone.utc).isoformat(),model=cfg.public(),previous_requests=14,
        maximum_new_requests=406,maximum_combined_requests=420,phase_caps={'engineering':6,'D':40,'G':360},
        D_states=4,API_repeats=2,engineering_state_indices=[0,1,2],ablations=['hidden','shuffled','hidden','shuffled'],
        no_more_interface_revision_this_batch=True,new_training_steps=0,new_source_files=0,
        prompts=PROMPTS,evidence_instruction=E_INSTRUCTION,plan_instruction=P_INSTRUCTION,
        protocol_sha256=digest(DOC),public_inputs_sha256=digest(OUT/'阶段D/公共输入.json'),
        previous_T_release_sha256=digest(PREVIOUS/'阶段T/放行结论.json'),
        source_sha256={f.relative_to(SRC).as_posix():digest(f) for f in files},
        frozen_core_sha256=core,protected_files_sha256=protected,tests=tests,
        default_sha256=digest(ROOT/'project/local_policy_default.json'),
        G_gate='all original functional message conditions + independent audit; not format validity alone')
    write(OUT/'预登记.json',reg)
    print(dict(prepared=True,offline_responses=14,offline_accepted=10,new_request_cap=406),flush=True)


def pair(api,sample,arm,phase,index,repeat):
    tag=dict(phase=phase,state=index,repeat=repeat,arm=arm)
    E,r0=compact_query(api,sample,arm,'evidence',dict(tag,slot=0));P=r1=None
    if E is not None and not api.circuit_reason:P,r1=compact_query(api,sample,arm,'plan',dict(tag,slot=1),E)
    return dict(arm=arm,state=index,repeat=repeat,phase=phase,evidence=E,plan=P,
        call_ids=[r0['call_id']]+([] if r1 is None else [r1['call_id']]),
        all_responses_valid=r0['status']=='ok' and r1 is not None and r1['status']=='ok',
        status=plan_status(P,sample['state']),input_sha256=digest_payload(sample['state'])),E


def summarize(records,engineering):
    pairs=[]
    for i in range(4):
        observations=[]
        for rep in (0,1):
            f=next(r for r in records if (r['state'],r['repeat'],r['arm'])==(i,rep,'M4'))
            a=next(r for r in records if (r['state'],r['repeat'],r['arm'])==(i,rep,'M4_ablation'))
            fs,ass=f['status'],a['status']
            observations.append(dict(repeat=rep,full_candidate=fs['candidate_id'],ablation_candidate=ass['candidate_id'],
                changed=fs['candidate_id']!=ass['candidate_id'],
                full_strictly_better=bool(fs['executable'] and (not ass['executable'] or tuple(fs['score'])>tuple(ass['score']))),
                message_reported_used=bool(f['plan'] and f['plan']['consumed_message_id']=='m0')))
        useful=all(x['changed'] and x['full_strictly_better'] and x['message_reported_used'] for x in observations)
        useful &= len({x['full_candidate'] for x in observations})==1 and len({x['ablation_candidate'] for x in observations})==1
        pairs.append(dict(state=i,pairs=observations,repeated_useful_message_effect=bool(useful)))
    full=[r for r in records if r['arm']=='M4']
    checks=dict(engineering_passed=engineering['passed'],all_D_valid=all(r['all_responses_valid'] for r in records),
        grounded_geometry_message_nonempty=all(r['evidence'] and r['evidence']['ranked_candidates'] for r in full),
        two_failure_states_executable=all(r['status']['executable'] for r in full if r['state'] in (0,1)),
        no_unvalidated_target_claim=all(r['status']['reason']!='unvalidated_target_claim' for r in records),
        repeated_useful_message_effect=any(p['repeated_useful_message_effect'] for p in pairs))
    checks={k:bool(v) for k,v in checks.items()}
    return dict(checks=checks,D_numeric_passed=all(checks.values()),paired_message_effects=pairs,
                D_mechanism_test_completed=all(len(r['call_ids'])==(1 if r['arm']=='M4_ablation' else 2) for r in records),
                legal_executable=sum(r['status']['executable'] for r in records),planned_records=len(records),
                no_new_navigation_SR=True,visual_semantic_accuracy_not_proven=True,allow_G=False,audit_pending=True)


def run():
    if (OUT/'API响应.jsonl').exists() or (OUT/'阶段D/机制汇总.json').exists():raise ValueError('no rerun')
    reg=read(OUT/'预登记.json')
    for n,h in reg['source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('source changed after registration')
    api=CompactAPI(configuration());samples=load_states();engineering=[]
    for i in reg['engineering_state_indices']:
        r,_=pair(api,samples[i],'M4','engineering',i,None);engineering.append(r)
        print(dict(engineering_state=i,valid=r['all_responses_valid'],executable=r['status']['executable'],new_requests=api.total_calls),flush=True)
        if not r['all_responses_valid'] or not r['status']['executable']:
            api.circuit_reason=api.circuit_reason or 'compact_engineering_failed';break
    eng=dict(passed=len(engineering)==3 and all(r['all_responses_valid'] and r['status']['executable'] for r in engineering),records=engineering)
    write(OUT/'阶段D/工程接口诊断.json',eng);records=[]
    if not eng['passed']:api.circuit_reason=api.circuit_reason or 'engineering_not_passed'
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
            if full_E is None or api.circuit_reason:ab=skipped(i,rep,'M4_ablation','no_usable_E_or_circuit')
            else:
                altered=ablate_compact(full_E,reg['ablations'][i])
                P,r=compact_query(api,sample,'M4','plan',dict(phase='D',state=i,repeat=rep,arm='M4_ablation',slot=1),altered)
                ab=dict(arm='M4_ablation',state=i,repeat=rep,phase='D',evidence=altered,plan=P,call_ids=[r['call_id']],
                        shared_E_call_id=full['call_ids'][0],ablation=reg['ablations'][i],all_responses_valid=r['status']=='ok',
                        status=plan_status(P,sample['state']),input_sha256=digest_payload(sample['state']))
            records.append(ab)
            print(dict(D_state=i,API_repeat=rep,new_requests=api.total_calls,circuit=api.circuit_reason),flush=True)
    write(OUT/'阶段D/逐状态决策.json',records)
    summary=summarize(records,eng);summary.update(new_requests=api.total_calls,combined_requests=14+api.total_calls,
                                               phase_counts=dict(api.phase_counts),circuit=api.circuit_reason)
    write(OUT/'阶段D/机制汇总.json',summary)
    write(OUT/'执行状态.json',dict(status='D_complete_pending_audit',new_requests=api.total_calls,combined_requests=14+api.total_calls,
                                  D_numeric_passed=summary['D_numeric_passed'],G_started=False,new_navigation_episodes=0,new_training_steps=0))
    print(dict(engineering_passed=eng['passed'],D_numeric_passed=summary['D_numeric_passed'],checks=summary['checks']),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run-frozen',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    elif args.run_frozen:run()
    else:p.error('choose prepare or run-frozen')
