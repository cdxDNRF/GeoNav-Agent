"""Make existing public source names explicit; do not relax grounding validation."""
from agents.evidence_budget_vlm import request,parse_evidence
from agents.evidence_reference_repair import parse_plan_repaired
from agents.collaboration_pilot import canonical,digest_payload


def clarified_request(state,images,config,arm,stage,evidence=None):
    named=dict(state)
    named['available_source_ids']=[label for label,_ in images]+['ledger']+['geometry:'+c['candidate_id'] for c in state['candidates']]
    payload,audit=request(named,images,config,arm,stage,evidence)
    instruction=("SOURCE FORMAT: source_ids must be nonempty quoted strings from available_source_ids; never numbers or empty arrays. "
        "Use ledger for public position/history facts and geometry:cXX for facts about a provided candidate. "
        "Image facts use the supplied target/current/history labels. "
        'Example fact format only: {"evidence_id":"e0","kind":"geometry","source_ids":["ledger"],"candidate_id":null,"statement":"Visited grid cells are recorded."}. '
        "Do not copy this example as a finding unless applicable. Plan evidence_ids use e0/e1/x0 for message facts, geometry:cXX for candidates, or supplied labels.")
    payload['messages'][1]['content'].append(dict(type='text',text=instruction))
    audit['request_sha256']=digest_payload(payload);audit['prompt_version']='public-source-catalog-clarification-v3'
    audit['format_instruction']=instruction;audit['parser_version']='strict-evidence-known-typed-plan-v3'
    return payload,audit


def clarified_query(api,sample,arm,stage,tag,evidence=None):
    payload,audit=clarified_request(sample['state'],sample['images'],api.config,arm,stage,evidence)
    parser=(lambda text,finish:parse_evidence(text,finish,sample['state'],audit['image_labels'])) if stage=='evidence' else (lambda text,finish:parse_plan_repaired(text,finish,sample['state'],evidence))
    return api.exchange(payload,audit,tag,parser)
