"""Program-owned geometry and bounded cloud choices; never infer missing provenance."""
import json
from hashlib import sha256
from agents.collaboration_pilot import canonical, digest_payload, image_part
from agents.evidence_budget import geometry_score
from agents.evidence_budget_vlm import parse_evidence, parse_plan
from agents.evidence_reference_repair import normalize_known_refs
from agents.vlm import _parse_json_object
from env.episode import ACTIONS

VERSION = 'program-geometry-compact-v2'
COMMON = """You assist aerial target search on a10x10 grid with20 moves. Rows increase down, columns right. You see only target/current/actual history images and the public visited ledger. Unvisited image contents and target coordinates/direction are UNKNOWN. Program tools supply exact legal candidate paths and geometry facts. Their new-cell counts mean coverage, not probability of finding the target. Visible similarity does not authorize a target direction. Never invent sources, coordinates or unseen scenes. Output compact JSON only, without explanations."""
E_INSTRUCTION = """Organize a bounded message. Output EXACTLY ranked_candidates, visual_relation, source_ids. ranked_candidates is a list of at most3 distinct supplied candidate IDs, best coverage first; [] is allowed if none is reasonable. Favor more new cells, fewer revisits, more remaining frontier, then shorter paths. visual_relation is unknown, similar or different, ONLY comparing the supplied target with a supplied current/history image. source_ids is [] for unknown, otherwise exactly [\"target\", one supplied observation label]. Similar is not identity or adjacency and never establishes target direction. Example of FORMAT ONLY: {\"ranked_candidates\":[],\"visual_relation\":\"unknown\",\"source_ids\":[]}. Do not copy computed numbers or write free-text facts."""
P_INSTRUCTION = """Choose a legal coverage option using the same tools and any supplied m0 message. Output EXACTLY candidate_id and evidence_refs. candidate_id is one supplied candidate ID or null for abstention. For a chosen candidate cXX, evidence_refs MUST include its exact tool reference g:cXX. You MAY also include m0 if you used its current nonempty candidate ranking and selected a candidate in that ranking. Otherwise omit m0. Abstention requires []. No target claims, counts, prose or other fields. Read supplied actual candidate/reference enums; never invent them. Consider the full public geometry even if m0 is empty."""
PROMPTS = dict(E=COMMON+' You are an evidence organizer. '+E_INSTRUCTION,
               P=COMMON+' You are an independent budget planner. '+P_INSTRUCTION,
               single=COMMON+' You are one integrated evidence-and-planning agent.')


def geometry_catalog(state):
    """Independently recompute tool quantities from the public prefix and path."""
    k=state['grid_size'];visited=set(state['ledger']['visited_order']);result={}
    for c in state['candidates']:
        r,col=state['position'];cells=[]
        if not 1<=len(c['actions'])<=min(2,state['remaining_budget']):raise ValueError('path budget')
        for action in c['actions']:
            dr,dc=ACTIONS[action];r+=dr;col+=dc
            if not(0<=r<k and 0<=col<k):raise ValueError('illegal tool path')
            cells.append(r*k+col)
        new=len(set(cells)-visited);revisits=sum(x in visited for x in cells)
        frontier=sum(0<=r+dr<k and 0<=col+dc<k and (r+dr)*k+col+dc not in visited.union(cells)
                     for dr,dc in ACTIONS.values())
        if c['cells']!=cells or c['waypoint']!=[r,col] or c['anticipated_new_cells']!=new or c['known_revisit_moves']!=revisits or c['remaining_frontier']!=frontier:
            raise ValueError('tool geometry does not match ledger')
        cid=c['candidate_id']
        if cid in result:raise ValueError('duplicate tool candidate')
        result[cid]=dict(reference='g:'+cid,candidate_id=cid,actions=c['actions'],cells=cells,
                         cost=len(cells),new_cells=new,revisits=revisits,frontier=frontier,
                         score=[new,-revisits,frontier,-len(cells)])
    return result


def parse_compact_evidence(content,finish,state,labels):
    obj,fence=_parse_json_object(content,finish);receipt=[]
    if not isinstance(obj,dict) or set(obj)!={'ranked_candidates','visual_relation','source_ids'}:raise ValueError('compact evidence keys')
    if obj['visual_relation'] not in ('unknown','similar','different'):raise ValueError('visual enum')
    if obj['visual_relation']=='unknown' and obj['source_ids'] is None:
        obj['source_ids']=[];receipt.append(dict(field='source_ids',operation='null_to_empty_unknown_only'))
    rank=obj['ranked_candidates'];refs=obj['source_ids'];catalog=geometry_catalog(state)
    if not isinstance(rank,list) or len(rank)>3 or any(not isinstance(x,str) or x not in catalog for x in rank) or len(set(rank))!=len(rank):raise ValueError('candidate ranking')
    if not isinstance(refs,list) or any(not isinstance(x,str) or x not in labels for x in refs) or len(refs)!=len(set(refs)):raise ValueError('unobserved source')
    if obj['visual_relation']=='unknown':
        if refs:raise ValueError('unknown must have no claimed visual sources')
    elif len(refs)!=2 or 'target' not in refs or not any(x!='target' for x in refs):
        raise ValueError('visible relation needs target and an observation')
    return dict(**obj,normalization_receipt=receipt),fence


def message_packet(evidence,state):
    if evidence is None:return None
    catalog=geometry_catalog(state)
    content={k:evidence[k] for k in ('ranked_candidates','visual_relation','source_ids')}
    return dict(message_id='m0',content=content,
                geometry_evidence=[catalog[cid] for cid in content['ranked_candidates']])


def parse_compact_plan(content,finish,state,evidence=None):
    obj,fence=_parse_json_object(content,finish);receipt=[]
    if not isinstance(obj,dict) or set(obj)!={'candidate_id','evidence_refs'}:raise ValueError('compact plan keys')
    cid=obj['candidate_id'];refs=obj['evidence_refs'];catalog=geometry_catalog(state)
    if cid is None and refs is None:
        refs=[];receipt.append(dict(field='evidence_refs',operation='null_to_empty_abstain_only'))
    if not isinstance(refs,list) or any(not isinstance(x,str) for x in refs) or len(refs)>2 or len(refs)!=len(set(refs)):raise ValueError('plan references')
    if cid is None:
        if refs:raise ValueError('abstention cannot claim evidence')
    else:
        if not isinstance(cid,str) or cid not in catalog:raise ValueError('unknown candidate')
        if 'g:'+cid not in refs or any(x not in ('g:'+cid,'m0') for x in refs):raise ValueError('missing/wrong candidate evidence')
        if 'm0' in refs and (evidence is None or cid not in evidence['ranked_candidates']):raise ValueError('unavailable message evidence')
    # These quantities are derived by the adapter, NOT repaired model claims.
    return dict(proposal_type='abstain' if cid is None else 'exploration_plan',candidate_id=cid,
                expected_new_cells=0 if cid is None else catalog[cid]['new_cells'],evidence_ids=refs,
                consumed_message_id='m0' if 'm0' in refs else None,stop_condition='after_option_or_new_cue',
                normalization_receipt=receipt),fence


def ablate_compact(evidence,kind):
    copied=json.loads(json.dumps(evidence))
    if kind=='hidden':return dict(ranked_candidates=[],visual_relation='unknown',source_ids=[],normalization_receipt=[])
    if kind!='shuffled':raise ValueError('ablation kind')
    copied['ranked_candidates'].reverse()
    return copied


def compact_request(state,images,config,arm,stage,evidence=None,previous_plan=None):
    if arm not in ('M2','M3','M4') or stage not in ('evidence','plan','reflect'):raise ValueError('workflow/stage')
    if not 2<=len(images)<=4:raise ValueError('image permission')
    if set(state)!={'grid_size','position','remaining_budget','ledger','candidates','local_proposals'}:
        raise ValueError('public state keys; evaluator metadata is forbidden')
    catalog=geometry_catalog(state);labels=[label for label,_ in images]
    if len(set(labels))!=len(labels) or labels[:2]!=['target','current'] or any(x not in ('target','current','history_0','history_1') for x in labels):
        raise ValueError('observed image labels')
    shared=dict(state,geometry_tools=list(catalog.values()),available_candidate_ids=list(catalog),
                available_source_ids=labels,available_geometry_refs=[x['reference'] for x in catalog.values()])
    instruction=E_INSTRUCTION if stage=='evidence' else P_INSTRUCTION
    role=('E' if stage=='evidence' else 'P') if arm=='M4' else 'single'
    system=PROMPTS[role];content=[]
    for label,pixels in images:content.extend(image_part(label,pixels))
    packet=message_packet(evidence,state)
    if arm=='M4':shared['evidence_message']=packet
    content.append(dict(type='text',text=canonical(shared).decode()+'\n'+instruction))
    messages=[dict(role='system',content=system),dict(role='user',content=content)]
    if arm=='M2' and stage=='plan' and packet is not None:
        messages.extend([dict(role='assistant',content=canonical(packet).decode()),dict(role='user',content=P_INSTRUCTION)])
    if stage=='reflect':
        if previous_plan is None:raise ValueError('reflection needs its actual prior plan')
        messages.extend([dict(role='assistant',content=canonical(previous_plan).decode()),
                         dict(role='user',content='Reconsider the actual prior plan with the same tools. '+P_INSTRUCTION)])
    payload=dict(model=config.model,messages=messages,temperature=0,max_tokens=config.max_tokens,
                 response_format={'type':'json_object'},stream=False)
    audit=dict(role=role,workflow=arm,workflow_stage=stage,public_input=shared,provided_message=evidence,
               previous_plan=previous_plan,image_labels=labels,image_count=len(images),
               image_sha256=[sha256(p).hexdigest() for _,p in images],system_prompt_sha256=sha256(system.encode()).hexdigest(),
               request_sha256=digest_payload(payload),evidence_message_sha256=None if packet is None else digest_payload(packet),
               prompt_version=VERSION)
    return payload,audit


def compact_query(api,sample,arm,stage,tag,evidence=None,previous_plan=None):
    state=sample['state'];payload,audit=compact_request(state,sample['images'],api.config,arm,stage,evidence,previous_plan)
    parser=(lambda text,finish:parse_compact_evidence(text,finish,state,audit['image_labels'])) if stage=='evidence' else \
           (lambda text,finish:parse_compact_plan(text,finish,state,evidence))
    return api.exchange(payload,audit,tag,parser)


def replay_legacy(record):
    """Only equivalent representations; never guess references or translate to new schema."""
    state=record['public_input'];obj,fence=_parse_json_object(record['raw_content'],record['finish_reason']);receipt=[]
    if record['workflow_stage']=='evidence':
        if isinstance(obj,dict) and obj.get('counter_evidence','missing') is None:
            obj['counter_evidence']=[];receipt.append(dict(field='counter_evidence',operation='null_to_empty'))
        parsed,_=parse_evidence(json.dumps(obj),record['finish_reason'],state,record['image_labels'])
    else:
        obj,aliases=normalize_known_refs(obj,record['provided_message']);receipt.extend(aliases)
        parsed,_=parse_plan(json.dumps(obj),record['finish_reason'],state,record['provided_message'])
    return parsed,receipt,fence
