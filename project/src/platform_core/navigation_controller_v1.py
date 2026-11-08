"""Portable public-state controllers and shared four-action transition."""
from collections import deque

ACTIONS={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}
CONTROLLERS={'manual':dict(label='人工控制',goal_conditioned=False),
             'frontier':dict(label='Frontier规则探索',goal_conditioned=False)}


def public_state(current,visited,remaining,grid=5):
    if type(grid) is not int or not 2<=grid<=50 or type(current) is not int or not 0<=current<grid*grid:
        raise ValueError('Invalid public grid position')
    if type(remaining) is not int or remaining<0:
        raise ValueError('Invalid remaining action budget')
    if not isinstance(visited,(list,tuple,set)) or any(type(c) is not int or not 0<=c<grid*grid for c in visited) or current not in visited:
        raise ValueError('Invalid public visited history')
    return dict(current_index=current,visited_indices=sorted(set(visited)),remaining_budget=remaining,grid_size=grid)


def transition(current,action,grid=5):
    if type(current) is not int or not 0<=current<grid*grid or not isinstance(action,str) or action not in ACTIONS:
        raise ValueError('Only valid public position and four actions')
    r,c=divmod(current,grid);dr,dc=ACTIONS[action];r+=dr;c+=dc
    outside=not(0<=r<grid and 0<=c<grid)
    return (current if outside else r*grid+c),outside


def frontier(packet):
    expected={'current_index','visited_indices','remaining_budget','grid_size'}
    if type(packet) is not dict or set(packet)!=expected:
        raise ValueError('Controller accepts public grid state only; no truth/source metadata')
    state=public_state(packet['current_index'],packet['visited_indices'],packet['remaining_budget'],packet['grid_size'])
    if state['remaining_budget']==0:raise ValueError('No action budget')
    current=state['current_index'];visited=set(state['visited_indices']);grid=state['grid_size']
    queue=deque([(current,None)]);seen={current}
    while queue:
        node,first=queue.popleft()
        for action in ACTIONS:
            neighbor,outside=transition(node,action,grid)
            if outside:continue
            chosen=first or action
            if neighbor not in visited:
                return dict(action=chosen,reason='按上/右/下/左次序前往最近未访问格',controller='frontier',goal_conditioned=False)
            if neighbor not in seen:seen.add(neighbor);queue.append((neighbor,chosen))
    for action in ACTIONS:
        if not transition(current,action,grid)[1]:
            return dict(action=action,reason='全部格已访问，执行合法邻接动作',controller='frontier',goal_conditioned=False)
    raise ValueError('No legal action')


def classify(trajectory):
    origins={t.get('origin') for t in trajectory[1:]}
    if not origins:return 'unstarted'
    if not origins<={'human','rule'}:raise ValueError('Unknown action provenance')
    return 'mixed' if len(origins)==2 else next(iter(origins))
