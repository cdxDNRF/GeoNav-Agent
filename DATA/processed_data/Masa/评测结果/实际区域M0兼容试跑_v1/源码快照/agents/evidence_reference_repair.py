"""Normalize only known typed fact-ID references; preserve strict provenance checks."""
import json
from agents.vlm import _parse_json_object
from agents.evidence_budget_vlm import parse_plan


def normalize_known_refs(obj,evidence):
    copied=json.loads(json.dumps(obj));aliases=[]
    if not isinstance(copied,dict) or not isinstance(copied.get('evidence_ids'),list):return copied,aliases
    facts={f['evidence_id']:f['kind'] for key in ('observed_facts','counter_evidence') for f in (evidence or {}).get(key,[])}
    for i,ref in enumerate(copied['evidence_ids']):
        if isinstance(ref,str) and ':' in ref:
            kind,identity=ref.split(':',1)
            if identity in facts and facts[identity]==kind:
                copied['evidence_ids'][i]=identity;aliases.append(dict(original=ref,normalized=identity,verified_fact_kind=kind))
    return copied,aliases


def parse_plan_repaired(content,finish,state,evidence=None):
    obj,fence=_parse_json_object(content,finish)
    normalized,_=normalize_known_refs(obj,evidence)
    result,_=parse_plan(json.dumps(normalized),finish,state,evidence)
    return result,fence
