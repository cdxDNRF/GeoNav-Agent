"""One parameter-free public-state guard, keeping the original cue and logits."""
import math

ACTIONS=('up','right','down','left')
DELTAS=((-1,0),(0,1),(1,0),(0,-1))

def fresh_alternative(obs,base):
    k=obs.grid_size
    if k!=10 or not 0<obs.remaining_budget<=20:raise ValueError('fixed grid10/B20 guard')
    r,c=obs.position
    if not(0<=r<k and 0<=c<k):raise ValueError('invalid public position')
    visited=set(obs.visited)
    if r*k+c not in visited or any(type(x) is not int or not 0<=x<k*k for x in visited):raise ValueError('invalid public visits')
    logits=base['explorer_logits']
    if len(logits)!=4 or not all(math.isfinite(x) for x in logits):raise ValueError('invalid original logits')
    if base['action'] not in ACTIONS or base['cue_action'] not in (*ACTIONS,None):raise ValueError('invalid action')
    dest={}
    for a,(dr,dc) in zip(ACTIONS,DELTAS):
        nr,nc=r+dr,c+dc
        if 0<=nr<k and 0<=nc<k:dest[a]=nr*k+nc
    if base['action'] not in dest:raise ValueError('original action must be legal')
    if base['cue_action'] is not None and base['action']!=base['cue_action']:raise ValueError('accepted cue must control original action')
    available=[a for a in ACTIONS if a in dest and dest[a] not in visited]
    triggered=base['cue_action'] is None and dest[base['action']] in visited and bool(available)
    action=max(available,key=lambda a:logits[ACTIONS.index(a)]) if triggered else base['action']
    return dict(action=action,triggered=bool(triggered),fresh_actions=available,
        original_next_cell=dest[base['action']],selected_next_cell=dest[action],
        control='fresh_alternative' if triggered else 'accepted_cue' if base['cue_action'] is not None else 'original_edge')
