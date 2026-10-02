"""Source-constrained evidence handoff, integrated workflow, and matched reflection."""
import json
from hashlib import sha256

from agents.collaboration_pilot import canonical,digest_payload,image_part
from agents.vlm import _parse_json_object

COMMON = """You assist one aerial-image search vehicle on a 10x10 grid with20 total moves. Rows increase down, columns right. Input contains ONLY target/current/actual past images, the public full visited ledger, and geometric path candidates. You know no target coordinates, true distance, source file identity, unseen imagery or future outcome. Candidate anticipated_new_cells are computed from the visited ledger; they do not predict visual contents. Target direction may be unknown: a legal coverage exploration_plan is still allowed. Preserve already accepted local cues. Never invent unseen features or call raw local scores calibrated probabilities. Return compact JSON only; no reasoning text or explanations outside the schema."""
E_INSTRUCTION = """Organize evidence. Output EXACTLY these keys: observed_facts, counter_evidence, unknowns, evidence_ids, ranked_candidates. observed_facts contains at most2 objects, counter_evidence at most1. Each object has exactly evidence_id (e0,e1 or x0), kind (geometry or visual), source_ids (1 or2 provided labels), candidate_id (existing ID or null), statement (at most80 characters). Prefer one concise public geometry fact; use visual facts only for visible relationships, never infer unvisited imagery. unknowns is at most3 values from target_direction, unvisited_visual_content, target_relation. evidence_ids lists exactly all fact IDs. ranked_candidates lists at most3 unique existing candidate IDs, strongest first, based on public coverage quality and visible evidence; may be empty. Avoid verbose sentences."""
P_INSTRUCTION = """Choose a bounded plan, allowed even with unknown target direction. Output EXACTLY proposal_type, candidate_id, expected_new_cells, evidence_ids, consumed_message_id, stop_condition. proposal_type is exploration_plan, target_claim or abstain. candidate_id is one provided candidate ID (null only for abstain). expected_new_cells copies that candidate's anticipated_new_cells (0 for abstain). evidence_ids cites provided evidence facts or geometry:ID, ledger, current, target; at most3. consumed_message_id is m0 if any supplied m0 content was used, otherwise null. stop_condition is after_option_or_new_cue or original_edge. Consider the full ledger and geometric candidates even if an evidence message is empty. No candidate implies abstain. A target_claim without independently validated local cue cannot be executed; prefer honest geometric exploration."""
PROMPTS = {'E':COMMON+' You are a separate evidence organizer. '+E_INSTRUCTION,
           'P':COMMON+' You are a separate budget planner receiving the organizer CURRENT message. '+P_INSTRUCTION,
           'single':COMMON+' You are one integrated evidence-and-planning agent. You may organize evidence and use the same geometry tools as a two-role workflow.'}


def parse_evidence(content,finish,state,labels):
    obj,norm=_parse_json_object(content,finish)
    if not isinstance(obj,dict) or set(obj)!={'observed_facts','counter_evidence','unknowns','evidence_ids','ranked_candidates'}:
        raise ValueError('evidence schema')
    candidates={c['candidate_id'] for c in state['candidates']};sources=set(labels)|{'ledger'}|{'geometry:'+c for c in candidates};ids=[]
    for key,limit in [('observed_facts',2),('counter_evidence',1)]:
        values=obj[key]
        if not isinstance(values,list) or len(values)>limit:raise ValueError('fact limit')
        for fact in values:
            if not isinstance(fact,dict) or set(fact)!={'evidence_id','kind','source_ids','candidate_id','statement'}:raise ValueError('fact schema')
            if fact['evidence_id'] not in ('e0','e1','x0') or fact['kind'] not in ('geometry','visual'):raise ValueError('fact kind/id')
            if fact['candidate_id'] is not None and fact['candidate_id'] not in candidates:raise ValueError('unknown candidate')
            refs=fact['source_ids']
            if not isinstance(refs,list) or not 1<=len(refs)<=2 or any(not isinstance(r,str) or r not in sources for r in refs):raise ValueError('unobserved source')
            if not isinstance(fact['statement'],str) or len(fact['statement'])>80:raise ValueError('fact length')
            ids.append(fact['evidence_id'])
    if len(ids)!=len(set(ids)) or obj['evidence_ids']!=ids:raise ValueError('fact IDs do not match')
    unknown=obj['unknowns'];ranked=obj['ranked_candidates']
    if not isinstance(unknown,list) or len(unknown)>3 or any(x not in ('target_direction','unvisited_visual_content','target_relation') for x in unknown):raise ValueError('unknown tags')
    if not isinstance(ranked,list) or len(ranked)>3 or any(not isinstance(x,str) or x not in candidates for x in ranked) or len(ranked)!=len(set(ranked)):raise ValueError('ranked candidates')
    return obj,norm


def parse_plan(content,finish,state,evidence=None):
    obj,norm=_parse_json_object(content,finish)
    if not isinstance(obj,dict) or set(obj)!={'proposal_type','candidate_id','expected_new_cells','evidence_ids','consumed_message_id','stop_condition'}:
        raise ValueError('plan schema')
    if obj['proposal_type'] not in ('exploration_plan','target_claim','abstain'):raise ValueError('proposal kind')
    if obj['consumed_message_id'] not in ('m0',None) or (evidence is None and obj['consumed_message_id'] is not None):raise ValueError('message ID')
    if obj['stop_condition'] not in ('after_option_or_new_cue','original_edge'):raise ValueError('termination schema')
    candidates={c['candidate_id']:c for c in state['candidates']}
    if obj['proposal_type']=='abstain':
        if obj['candidate_id'] is not None or type(obj['expected_new_cells']) is not int or obj['expected_new_cells']!=0:raise ValueError('abstain shape')
    else:
        if obj['candidate_id'] not in candidates or type(obj['expected_new_cells']) is not int or obj['expected_new_cells']!=candidates[obj['candidate_id']]['anticipated_new_cells']:
            raise ValueError('candidate geometry')
    allowed={'ledger','current','target'}|{'geometry:'+c for c in candidates}
    if evidence:allowed|=set(evidence['evidence_ids'])
    refs=obj['evidence_ids']
    if not isinstance(refs,list) or len(refs)>3 or any(not isinstance(r,str) or r not in allowed for r in refs):raise ValueError('plan sources')
    return obj,norm


def ablate_message(evidence,kind):
    copied=json.loads(json.dumps(evidence))
    if kind=='hidden':return dict(observed_facts=[],counter_evidence=[],unknowns=['target_direction','unvisited_visual_content'],evidence_ids=[],ranked_candidates=[])
    if kind!='shuffled':raise ValueError('unknown ablation')
    copied['observed_facts'].reverse();copied['counter_evidence'].reverse();copied['ranked_candidates'].reverse()
    copied['evidence_ids']=[f['evidence_id'] for k in ('observed_facts','counter_evidence') for f in copied[k]]
    return copied


def request(state,images,config,arm,stage,evidence=None,previous_plan=None):
    if arm not in ('M2','M3','M4') or stage not in ('evidence','plan','reflect'):raise ValueError('workflow/stage')
    if not 2<=len(images)<=4:raise ValueError('picture cap')
    shared=dict(state);content=[]
    for label,pixels in images:content.extend(image_part(label,pixels))
    labels=[x[0] for x in images]
    instruction=E_INSTRUCTION if stage=='evidence' else P_INSTRUCTION
    if arm=='M4':
        role='E' if stage=='evidence' else 'P';system=PROMPTS[role]
        if stage=='plan':shared['evidence_message']=None if evidence is None else dict(message_id='m0',content=evidence)
        content.append(dict(type='text',text=canonical(shared).decode()))
        messages=[dict(role='system',content=system),dict(role='user',content=content)]
    else:
        role='single';system=PROMPTS['single'];content.append(dict(type='text',text=canonical(shared).decode()+'\n'+instruction))
        messages=[dict(role='system',content=system),dict(role='user',content=content)]
        if stage=='plan' and evidence is not None:
            messages.append(dict(role='assistant',content=canonical(dict(message_id='m0',content=evidence)).decode()))
            messages.append(dict(role='user',content='Now perform integrated planning. '+P_INSTRUCTION))
        if stage=='reflect':
            messages.append(dict(role='assistant',content=canonical(previous_plan).decode()))
            messages.append(dict(role='user',content='Reconsider your plan using the same public evidence. '+P_INSTRUCTION))
    payload=dict(model=config.model,messages=messages,temperature=0,max_tokens=config.max_tokens,response_format={'type':'json_object'},stream=False)
    audit=dict(role=role,workflow=arm,workflow_stage=stage,public_input=shared,
        provided_message=evidence,previous_plan=previous_plan,image_labels=labels,image_count=len(images),
        image_sha256=[sha256(p).hexdigest() for _,p in images],system_prompt_sha256=sha256(system.encode()).hexdigest(),
        request_sha256=digest_payload(payload),evidence_message_sha256=None if evidence is None else digest_payload(evidence),
        prompt_version='public-ledger-budget-v1')
    return payload,audit


def query(api,state,images,arm,stage,tag,evidence=None,previous_plan=None):
    payload,audit=request(state,images,api.config,arm,stage,evidence,previous_plan)
    parser=(lambda text,finish:parse_evidence(text,finish,state,audit['image_labels'])) if stage=='evidence' else (lambda text,finish:parse_plan(text,finish,state,evidence))
    return api.exchange(payload,audit,tag,parser)
