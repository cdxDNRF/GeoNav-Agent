"""Public full-grid ledger, fixed trigger and two-step coverage options."""
from hashlib import sha256
import itertools
import math

from env.episode import ACTIONS


def public_trigger(visited, remaining, budget, accepted_now, accepted_before, used=False):
    if used or accepted_now or remaining < 2:
        return False, 'protected_or_used_or_short_budget'
    elapsed = budget-remaining
    known = set(visited[:-2]) if len(visited) >= 3 else set(visited)
    added = repeats = 0
    if len(visited) >= 3:
        for cell in visited[-2:]:
            repeats += cell in known
            added += cell not in known
            known.add(cell)
    stagnation = len(visited) >= 3 and added <= 1 and repeats >= 1
    checkpoint = elapsed >= math.ceil(budget/3) and not accepted_before
    return stagnation or checkpoint, 'stagnation' if stagnation else 'checkpoint' if checkpoint else 'not_due'


def paths_for(position, visited, grid_size, remaining, explorer_action=None):
    """Unknown cells have geometry only; no visual value or target prior is invented."""
    if remaining < 1:
        return []
    known=set(visited);origin=position[0]*grid_size+position[1];records=[]
    for n in range(1,min(2,remaining)+1):
        for actions in itertools.product(ACTIONS,repeat=n):
            r,c=position;cells=[];legal=True
            for action in actions:
                dr,dc=ACTIONS[action];r+=dr;c+=dc
                if not(0<=r<grid_size and 0<=c<grid_size):legal=False;break
                cells.append(r*grid_size+c)
            if not legal or cells[-1]==origin:
                continue
            new=len(set(cells)-known)
            if not new:
                continue
            end=cells[-1];er,ec=divmod(end,grid_size)
            frontier=sum(0<=er+dr<grid_size and 0<=ec+dc<grid_size and (er+dr)*grid_size+ec+dc not in known.union(cells)
                         for dr,dc in ACTIONS.values())
            records.append(dict(candidate_id=f'c{len(records):02d}',actions=list(actions),cells=cells,
                waypoint=list(divmod(end,grid_size)),anticipated_new_cells=new,
                known_revisit_moves=sum(x in known for x in cells),remaining_frontier=frontier,
                matches_explorer=actions[0]==explorer_action))
    return records


def geometry_score(candidate):
    return (candidate['anticipated_new_cells'],-candidate['known_revisit_moves'],candidate['remaining_frontier'],-len(candidate['actions']))


def deterministic_choice(candidates):
    if not candidates:
        return None
    return max(candidates,key=lambda c:(geometry_score(c),c['matches_explorer'],-int(c['candidate_id'][1:])))['candidate_id']


class EvidenceLedger:
    def __init__(self,grid_size,budget):
        self.k=grid_size;self.budget=budget;self.reset()

    def reset(self):
        self.nodes={};self.edges=[];self.cues=[];self.previous=None;self.visited=[];self.accepted_ever=False

    def observe_fields(self,position,remaining,visited,current_sha,target_sha,cue_action,previous_action=None):
        cell=position[0]*self.k+position[1];elapsed=self.budget-remaining
        if visited[-1]!=cell or len(visited)!=elapsed+1 or not 0<=cell<self.k*self.k:
            raise ValueError('inconsistent public prefix')
        if self.previous is not None:
            if visited[:-1]!=self.visited or previous_action not in ACTIONS:
                raise ValueError('ledger must advance on actual consecutive observations')
            self.edges.append(dict(from_cell=self.previous,to_cell=cell,action=previous_action,step=elapsed))
        elif elapsed!=0:
            raise ValueError('ledger starts at initial public observation')
        node=self.nodes.setdefault(cell,dict(cell=cell,position=list(position),first_seen_step=elapsed,image_sha256=current_sha,visits=0))
        if node['image_sha256']!=current_sha:
            raise ValueError('immutable observed image changed')
        node['visits']+=1;self.visited=list(visited);self.previous=cell;self.target_sha=target_sha
        self.cues.append(dict(step=elapsed,accepted=cue_action is not None,cue_action=cue_action))
        self.accepted_ever |= cue_action is not None

    def observe(self,obs,base,previous_action=None):
        self.observe_fields(obs.position,obs.remaining_budget,obs.visited,sha256(obs.current_image).hexdigest(),
                            sha256(obs.target_image).hexdigest(),base['cue_action'],previous_action)

    def snapshot(self):
        return dict(grid_size=self.k,visited_order=list(self.visited),observed_nodes=list(self.nodes.values()),
            executed_edges=list(self.edges),cue_history=list(self.cues),
            target_id=self.target_sha,unobserved_cells=[j for j in range(self.k*self.k) if j not in self.nodes],
            unknown_target_direction=True)
