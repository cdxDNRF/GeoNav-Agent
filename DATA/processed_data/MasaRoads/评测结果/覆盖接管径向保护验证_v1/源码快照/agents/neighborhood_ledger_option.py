"""Two-step geometry of opportunities for the frozen adjacent-target sensor."""
from hashlib import sha256
from agents.evidence_budget import EvidenceLedger,public_trigger,paths_for,geometry_score
from agents.local_ledger_option import value_hash
from env.episode import ACTIONS


def closed_neighborhood(cells,k):
    result=set()
    for cell in cells:
        if type(cell) is not int or not 0<=cell<k*k:raise ValueError('public cell out of range')
        r,c=divmod(cell,k);result.add(cell)
        for dr,dc in ACTIONS.values():
            nr,nc=r+dr,c+dc
            if 0<=nr<k and 0<=nc<k:result.add(nr*k+nc)
    return result


def neighborhood_candidates(candidates,visited,k):
    # Coverage is an opportunity, never a truth-based exclusion or visual score.
    covered=closed_neighborhood(visited,k)
    return [dict(c,new_target_hypotheses=sorted(closed_neighborhood(c['cells'],k)-covered),
        new_target_hypothesis_count=len(closed_neighborhood(c['cells'],k)-covered)) for c in candidates]


def neighborhood_choice(candidates):
    if not candidates:return None
    return max(candidates,key=lambda c:(c['new_target_hypothesis_count'],geometry_score(c),
        c['matches_explorer'],-int(c['candidate_id'][1:])))['candidate_id']


class NeighborhoodLedgerOption:
    enabled=True
    def __init__(self):self.reset()

    def reset(self):
        self.ledger=EvidenceLedger(10,20);self.used=False;self.pending=[];self.previous=None

    def decide(self,obs,base):
        if obs.grid_size!=10:raise ValueError('frozen grid10 controller')
        self.ledger.observe_fields(obs.position,obs.remaining_budget,list(obs.visited),
            sha256(obs.current_image).hexdigest(),sha256(obs.target_image).hexdigest(),base['cue_action'],self.previous)
        due,reason=public_trigger(obs.visited,obs.remaining_budget,20,base['cue_action'] is not None,self.ledger.accepted_ever,self.used)
        event=None
        if due:
            self.used=True;candidates=neighborhood_candidates(paths_for(obs.position,obs.visited,10,obs.remaining_budget,base['explorer_action']),obs.visited,10)
            selected=neighborhood_choice(candidates)
            event=dict(trigger_reason=reason,candidate_id=selected,candidates=candidates,public_ledger_sha256=value_hash(self.ledger.snapshot()),
                score_source='public_geometry_only',covered_hypotheses_are_not_eliminated=True)
            if selected is not None:self.pending=list(next(c['actions'] for c in candidates if c['candidate_id']==selected))
        if base['cue_action'] is not None:
            action=base['action'];interrupted=bool(self.pending);self.pending=[];control='accepted_cue'
        elif self.pending:action=self.pending.pop(0);interrupted=False;control='bounded_option'
        else:action=base['action'];interrupted=False;control='original_edge'
        self.previous=action
        return dict(action=action,interaction=event,control=control,option_interrupted=interrupted,
            pending_after=list(self.pending),ledger_sha256=value_hash(self.ledger.snapshot()))
