"""Training-side geometry/reward/schedules. Never a navigation policy input."""
from __future__ import annotations
import math

ACTIONS = ('up','right','down','left')


def distance(a, b, grid=5):
    ar, ac = divmod(a, grid);br, bc = divmod(b, grid)
    return abs(ar-br)+abs(ac-bc)


def move(cell, action, grid=5):
    if action not in range(4) or cell not in range(grid*grid):
        raise ValueError('invalid action/cell')
    row, col = divmod(cell,grid)
    dr, dc = ((-1,0),(0,1),(1,0),(0,-1))[action]
    r, c = row+dr,col+dc
    return r*grid+c if 0<=r<grid and 0<=c<grid else cell


def optimal_actions(goal, current, grid=5):
    return [int(distance(move(current,a,grid),goal,grid)<distance(current,goal,grid)) for a in range(4)]


def pair_at_distance(rng, grid=5):
    d = int(rng.randint(1,2*grid-1))
    goal,current = int(rng.randint(0,grid*grid)),int(rng.randint(0,grid*grid))
    while distance(current,goal,grid)!=d:
        goal,current = int(rng.randint(0,grid*grid)),int(rng.randint(0,grid*grid))
    return goal,current,d


def random_path(rng, grid=5):
    # The upstream dataset samples length BEFORE get_random_sequence's endpoints.
    moves=int(rng.randint(1,11));goal,current,_=pair_at_distance(rng,grid)
    cells=[current];actions=[]
    for _ in range(moves):
        action=int(rng.randint(0,4));actions.append(action);cells.append(move(cells[-1],action,grid))
    return goal,cells,actions,[optimal_actions(goal,c,grid) for c in cells]


def horizon(rng, initial_distance, budget=10):
    return int(rng.randint(initial_distance,budget+1))-1


def external_reward(previous, current, goal, prior_states, best_dist, grid=5):
    if current==previous:return -1.
    if current==goal:return 2.
    if current in prior_states:return -1.
    r,c=divmod(current,grid);gr,gc=divmod(goal,grid)
    return 1. if (r-gr)**2+(c-gc)**2<best_dist else -1.


def intrinsic_reward(mse):
    if not math.isfinite(mse) or mse<0:raise ValueError('invalid prediction MSE')
    return .25*(2*(mse-.8)/.1-1)


def returns(rewards, terminals, gamma=.93):
    if len(rewards)!=len(terminals) or not rewards:raise ValueError('nonempty aligned buffer required')
    result=[];running=0.
    for reward,done in zip(reversed(rewards),reversed(terminals)):
        if done:running=0.
        running=reward+gamma*running;result.append(running)
    return list(reversed(result))


def accumulation_boundary(index, batches, accumulation=256):
    if not 0<=index<batches or accumulation<1:raise ValueError('invalid batch schedule')
    return (index+1)%accumulation==0 or index+1==batches


def updates_for_epochs(epochs,batches,accumulation=256):
    return epochs*math.ceil(batches/accumulation)


def should_update(step,interval=640):
    return step>0 and step%interval==0
