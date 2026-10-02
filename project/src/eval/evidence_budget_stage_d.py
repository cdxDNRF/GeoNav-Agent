"""Bounded same-prefix message diagnostics; no navigation or newly trained models."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.collaboration_pilot import PilotAPI,digest_payload
from agents.curl_transport import CurlTransport
from agents.vlm import APIConfig
from agents.evidence_budget import geometry_score
from agents.evidence_budget_vlm import query,ablate_message,PROMPTS,E_INSTRUCTION,P_INSTRUCTION
from env.environment import image_payload
from eval.evidence_budget_stage_t import ROOT,SRC,OLD,OUT,read,lines,write,digest,public_views

DATA=ROOT/'DATA/processed_data/Masa/网格扩展_v1/正式数据'
TESTS=ROOT/'选题报告相关/证据账本预算规划阶段D测试_v1.json'


def load_states():
    refs=read(OUT/'阶段T/诊断状态.json');eps={e['episode_id']:e for e in read(OLD/'导航任务.json')}
    need={r['episode_id'] for r in refs};records={r['episode_id']:r for r in lines(OLD/'Edge_s0/导航_CueFull_轨迹.jsonl') if r['episode_id'] in need}
    result=[]
    for ref in refs:
        ep=eps[ref['episode_id']];row=records[ref['episode_id']];index=ref['index'];d=row['decisions'][index];view=list(public_views(row))[index]
        current=d['public_position'][0]*10+d['public_position'][1]
        path=lambda cell:DATA/f'patches/{ep["split"]}/{ep["area"]}/patch_{cell}.jpg'
        images=[('target',image_payload(path(ep['goal']))),('current',image_payload(path(current)))];history=[]
        for cell in reversed(d['public_visited'][:-1]):
            if cell!=current and cell not in history:history.append(cell)
            if len(history)==2:break
        for i,cell in enumerate(reversed(history)):images.append((f'history_{i}',image_payload(path(cell))))
        if sha256(images[0][1]).hexdigest()!=d['target_image_sha256'] or sha256(images[1][1]).hexdigest()!=d['current_image_sha256']:
            raise ValueError('saved public input pixel hash mismatch')
        state=dict(grid_size=10,position=d['public_position'],remaining_budget=d['remaining_budget'],
            ledger=view['ledger'],candidates=view['candidates'],
            local_proposals={k:d[k] for k in ('action','explorer_action','cue_action','probabilities')})
        result.append(dict(state=state,images=images,reference=ref,
            input_paths=[path(ep['goal']),path(current)]+[path(c) for c in reversed(history)]))
    return result


class ScopedAPI(PilotAPI):
    def __init__(self,config,transport):
        super().__init__(config,transport,OUT,max_calls=420);self.phase_counts={};self.caps={'engineering':12,'D':48,'G':360}

    def exchange(self,payload,audit,tag,parser):
        phase=tag['phase'];count=self.phase_counts.get(phase,0)
        if count>=self.caps[phase]:raise RuntimeError('frozen phase cap exceeded')
        self.phase_counts[phase]=count+1
        return super().exchange(payload,audit,tag,parser)


def configuration():
    raw=APIConfig.load()
    if raw.base_url.rstrip('/')!='https://ollama.com/v1' or raw.model!='gemma4:31b':raise ValueError('Gemma provider/model fixed')
    return APIConfig(raw.base_url,raw.model,raw.api_key,timeout=60,max_tokens=512)


def plan_status(plan,state):
    if plan is None:return dict(executable=False,reason='API_or_parse_failure',candidate_id=None,score=None)
    if plan['proposal_type']=='abstain':return dict(executable=False,reason='abstain',candidate_id=None,score=None)
    if plan['proposal_type']=='target_claim':return dict(executable=False,reason='unvalidated_target_claim',candidate_id=plan['candidate_id'],score=None)
    c=next(c for c in state['candidates'] if c['candidate_id']==plan['candidate_id'])
    return dict(executable=True,reason='legal_exploration',candidate_id=c['candidate_id'],score=list(geometry_score(c)))


def run_pair(api,sample,arm,phase,index,repeat):
    state=sample['state'];images=sample['images'];tag=dict(phase=phase,state=index,repeat=repeat,arm=arm)
    E,first=query(api,state,images,arm,'evidence',dict(tag,slot=0))
    P=second=None
    if E is not None and not api.circuit_reason:P,second=query(api,state,images,arm,'plan',dict(tag,slot=1),evidence=E)
    ids=[first['call_id']]+([] if second is None else [second['call_id']])
    return dict(arm=arm,state=index,repeat=repeat,phase=phase,evidence=E,plan=P,call_ids=ids,
        all_responses_valid=first['status']=='ok' and second is not None and second['status']=='ok',
        status=plan_status(P,state),input_sha256=digest_payload(state)),E


def skipped(index,repeat,arm,reason):
    return dict(arm=arm,state=index,repeat=repeat,phase='D',evidence=None,plan=None,call_ids=[],
        all_responses_valid=False,status=dict(executable=False,reason=reason,candidate_id=None,score=None))


def diagnostic_summary(records,engineering):
    paired=[]
    for index in range(4):
        pairs=[]
        for rep in (0,1):
            full=next(r for r in records if r['state']==index and r['repeat']==rep and r['arm']=='M4')
            ab=next(r for r in records if r['state']==index and r['repeat']==rep and r['arm']=='M4_ablation')
            f,a=full['status'],ab['status'];message_used=full['plan'] is not None and full['plan']['consumed_message_id']=='m0'
            strictly_better=f['executable'] and (not a['executable'] or tuple(f['score'])>tuple(a['score']))
            changed=f['candidate_id']!=a['candidate_id']
            pairs.append(dict(repeat=rep,full_candidate=f['candidate_id'],ablation_candidate=a['candidate_id'],
                changed=changed,full_strictly_better=strictly_better,message_reported_used=message_used))
        repeated=all(p['changed'] and p['full_strictly_better'] and p['message_reported_used'] for p in pairs)
        repeated &= len({p['full_candidate'] for p in pairs})==1 and len({p['ablation_candidate'] for p in pairs})==1
        paired.append(dict(state=index,pairs=pairs,repeated_useful_message_effect=bool(repeated)))
    fulls=[r for r in records if r['arm']=='M4'];checks=dict(engineering_passed=engineering['passed'],
        all_D_valid=all(r['all_responses_valid'] for r in records),
        grounded_evidence_nonempty=all(r['evidence'] is not None and bool(r['evidence']['observed_facts'] or r['evidence']['counter_evidence']) for r in fulls),
        two_failure_states_executable=all(r['status']['executable'] for r in fulls if r['state'] in (0,1)),
        no_unvalidated_target_claim=all(r['status']['reason']!='unvalidated_target_claim' for r in records),
        repeated_useful_message_effect=any(r['repeated_useful_message_effect'] for r in paired))
    return dict(checks=checks,D_numeric_passed=all(checks.values()),paired_message_effects=paired,
        workflow_counts={a:dict(records=sum(r['arm']==a for r in records),executable=sum(r['arm']==a and r['status']['executable'] for r in records),
            message_reported_used=sum(r['arm']==a and r['plan'] is not None and r['plan']['consumed_message_id']=='m0' for r in records))
            for a in ('M2','M4','M4_ablation')},no_new_navigation_SR=True,semantic_accuracy_not_proven=True,
        audit_pending=True,allow_G=False)


def main(authorized):
    if not authorized:raise ValueError('cloud authorization required')
    t=read(OUT/'阶段T/放行结论.json')
    if not t['allow_D'] or digest(OUT/'阶段T/独立复核.json')!=t['audit_sha256']:raise ValueError('T gate/audit not passed')
    if (OUT/'阶段D').exists():raise ValueError('immutable diagnostic exists; never rerun')
    tests=read(TESTS)
    if not tests['successful']:raise ValueError('offline tests required')
    samples=load_states();cfg=configuration();(OUT/'阶段D').mkdir()
    source=[p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','eval','env','data','train','tests')]
    for f in source:
        dest=OUT/'源码快照'/f.relative_to(SRC)
        if dest.exists() and digest(dest)!=digest(f):raise ValueError('previously frozen source changed')
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    write(OUT/'阶段D/公共输入.json',[dict(state=s['state'],image_labels=[x[0] for x in s['images']],
        image_sha256=[sha256(x[1]).hexdigest() for x in s['images']]) for s in samples])
    paths={p for s in samples for p in s['input_paths']}
    reg=dict(version='ledger-budget-D-v1',utc=datetime.now(timezone.utc).isoformat(),model=cfg.public(),
        states=4,API_repeats=2,maximum_chat_requests=420,phase_caps=dict(engineering=12,D=48,G=360),
        expected_actual_D_requests=40,engineering_pairs=3,shared_E_in_ablation=True,
        ablations=['hidden','shuffled','hidden','shuffled'],engineering_state_indices=[0,1,2],
        prompts=PROMPTS,evidence_instruction=E_INSTRUCTION,planning_instruction=P_INSTRUCTION,
        message_gate='at least1 state with repeated same pair of distinct candidate IDs, full geometryscore strictly better andm0 reported used in both repeats',
        T_audit_sha256=digest(OUT/'阶段T/独立复核.json'),public_inputs_sha256=digest(OUT/'阶段D/公共输入.json'),
        source_sha256={f.relative_to(SRC).as_posix():digest(f) for f in source},
        inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        test_record=tests,default_sha256=digest(ROOT/'project/local_policy_default.json'),new_training_steps=0)
    write(OUT/'阶段D/预登记.json',reg)
    api=ScopedAPI(cfg,CurlTransport(60));engineering_rows=[]
    for i in reg['engineering_state_indices']:
        if api.circuit_reason:break
        r,_=run_pair(api,samples[i],'M4','engineering',i,None);engineering_rows.append(r)
        print({'engineering_state':i,'valid':r['all_responses_valid'],'requests':api.total_calls},flush=True)
        if not r['all_responses_valid']:
            api.circuit_reason=api.circuit_reason or 'engineering_schema_or_message_failure';break
    engineering=dict(passed=len(engineering_rows)==3 and all(r['all_responses_valid'] for r in engineering_rows),records=engineering_rows)
    write(OUT/'阶段D/工程接口诊断.json',engineering)
    records=[]
    if not engineering['passed']:api.circuit_reason=api.circuit_reason or 'engineering_failed'
    for rep in (0,1):
        for i,sample in enumerate(samples):
            if api.circuit_reason:
                records.extend(skipped(i,rep,a,api.circuit_reason) for a in ('M2','M4','M4_ablation'));continue
            order=['M2','M4'] if (rep+i)%2==0 else ['M4','M2'];full=None
            for arm in order:
                if api.circuit_reason:r=skipped(i,rep,arm,api.circuit_reason);E=None
                else:r,E=run_pair(api,sample,arm,'D',i,rep)
                records.append(r)
                if arm=='M4':full=r;full_E=E
            kind=reg['ablations'][i]
            if full_E is None or api.circuit_reason:ab=skipped(i,rep,'M4_ablation','no_usable_full_E_or_circuit')
            else:
                changed=ablate_message(full_E,kind)
                P,rec=query(api,sample['state'],sample['images'],'M4','plan',dict(phase='D',state=i,repeat=rep,arm='M4_ablation',slot=1),evidence=changed)
                ab=dict(arm='M4_ablation',state=i,repeat=rep,phase='D',evidence=changed,plan=P,call_ids=[rec['call_id']],
                    shared_E_call_id=full['call_ids'][0],ablation=kind,all_responses_valid=rec['status']=='ok',
                    status=plan_status(P,sample['state']),input_sha256=digest_payload(sample['state']))
            records.append(ab)
            print({'D_state':i,'API_repeat':rep,'requests':api.total_calls,'circuit':api.circuit_reason},flush=True)
    write(OUT/'阶段D/逐状态决策.json',records)
    summary=diagnostic_summary(records,engineering);summary.update(chat_requests=api.total_calls,phase_counts=api.phase_counts,circuit=api.circuit_reason)
    write(OUT/'阶段D/机制汇总.json',summary)
    write(OUT/'执行状态.json',dict(status='D_completed',D_numeric_passed=summary['D_numeric_passed'],chat_requests=api.total_calls,
        G_started=False,G_reason='pending_audit' if summary['D_numeric_passed'] else 'message_gate_failed',new_training_steps=0))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--confirm-cloud-transmission',action='store_true');args=p.parse_args();main(args.confirm_cloud_transmission)
