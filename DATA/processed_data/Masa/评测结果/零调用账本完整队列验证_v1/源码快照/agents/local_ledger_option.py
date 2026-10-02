"""A public-observation-only, zero-cloud copy of the frozen M1 controller."""
from hashlib import sha256
import json
from agents.evidence_budget import EvidenceLedger,public_trigger,paths_for,deterministic_choice


def value_hash(obj):
    return sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class LocalLedgerOption:
    def __init__(self,enabled=True):
        self.enabled=enabled;self.reset()

    def reset(self):
        self.ledger=EvidenceLedger(10,20);self.used=False;self.pending=[];self.previous=None

    def decide(self,obs,base):
        if obs.grid_size!=10:raise ValueError('frozen grid10 controller')
        self.ledger.observe_fields(obs.position,obs.remaining_budget,list(obs.visited),
            sha256(obs.current_image).hexdigest(),sha256(obs.target_image).hexdigest(),base['cue_action'],self.previous)
        due,reason=public_trigger(obs.visited,obs.remaining_budget,20,base['cue_action'] is not None,
                                  self.ledger.accepted_ever,self.used)
        event=None
        if self.enabled and due:
            self.used=True;candidates=paths_for(obs.position,obs.visited,10,obs.remaining_budget,base['explorer_action'])
            selected=deterministic_choice(candidates)
            event=dict(trigger_reason=reason,candidate_id=selected,candidates=candidates,
                       public_ledger_sha256=value_hash(self.ledger.snapshot()))
            if selected is not None:self.pending=list(next(c for c in candidates if c['candidate_id']==selected)['actions'])
        if base['cue_action'] is not None:
            action=base['action'];interrupted=bool(self.pending);self.pending=[];control='accepted_cue'
        elif self.pending:
            action=self.pending.pop(0);interrupted=False;control='bounded_option'
        else:
            action=base['action'];interrupted=False;control='original_edge'
        self.previous=action
        return dict(action=action,interaction=event,control=control,option_interrupted=interrupted,
                    pending_after=list(self.pending),ledger_sha256=value_hash(self.ledger.snapshot()))
