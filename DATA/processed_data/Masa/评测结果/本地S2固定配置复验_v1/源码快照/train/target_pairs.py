"""Training-side paired goals: identical public history, disjoint correct actions.

Goal indices and desired directions stay in the trainer/evaluator. The policy
receives the existing 1052-dimensional public features, without extra fields.
"""
from itertools import combinations
import numpy as np
import torch

from env.episode import ACTIONS, manhattan
from train.dyncur_tiny import policy_features


def closer_actions(current, goal):
    row, col = divmod(current, 5)
    return np.asarray([0 <= row+dr < 5 and 0 <= col+dc < 5 and
        manhattan((row+dr)*5+col+dc, goal) < manhattan(current, goal)
        for dr, dc in ACTIONS.values()], dtype=bool)


def paired_records(areas, count, seed):
    """Three observations/two legal moves. No goal was previously visited.

    Goals are equally far from the final current cell and require disjoint
    progress actions. Short ranges are necessary to remove geometry-only
    solutions; this auxiliary bank is not the C=4..8 navigation benchmark.
    """
    rng=np.random.default_rng(seed)
    records=[]
    areas=sorted(areas)
    for i in range(count):
        for attempt in range(1000):
            history=[int(rng.integers(25))]
            for _ in range(2):
                row,col=divmod(history[-1],5)
                legal=[(row+dr)*5+col+dc for dr,dc in ACTIONS.values()
                       if 0 <= row+dr < 5 and 0 <= col+dc < 5]
                history.append(int(rng.choice(legal)))
            current=history[-1]
            goals=[g for g in range(25) if g not in history]
            progress={g:closer_actions(current,g) for g in goals}
            pairs=[(a,b) for a,b in combinations(goals,2)
                   if manhattan(current,a)==manhattan(current,b)
                   and not (progress[a]&progress[b]).any()]
            if pairs:
                a,b=pairs[int(rng.integers(len(pairs)))]
                if rng.integers(2):a,b=b,a
                records.append(dict(area=areas[i%len(areas)],history=history,goals=[a,b],
                    labels=[progress[a].astype(int).tolist(),progress[b].astype(int).tolist()]))
                break
        else:
            raise RuntimeError('bounded paired-history sampling failed')
    return records


def materialize(records, store):
    features=np.empty((len(records),2,3,1052),np.float32)
    labels=np.empty((len(records),2,4),np.float32)
    for i,record in enumerate(records):
        for j,goal in enumerate(record['goals']):
            for t,current in enumerate(record['history']):
                features[i,j,t]=policy_features(store.patch(record['area'],goal),
                    store.patch(record['area'],current),divmod(current,5),10-t,record['history'][:t+1])
            labels[i,j]=record['labels'][j]
    labels/=labels.sum(-1,keepdims=True)
    return features,labels


def paired_logits(model, features):
    """features=[pairs, two goals, time, public features]; reset each history."""
    count,goals,length,width=features.shape
    sequences=features.reshape(count*goals,length,width).transpose(0,1)
    hidden=None
    for obs in sequences:
        logits,_,_,hidden=model.step(obs,hidden)
    return logits.reshape(count,goals,4)


def supervised_goal_loss(model, features, labels):
    logits=paired_logits(model,features)
    return -(labels*torch.log_softmax(logits,-1)).sum(-1).mean()


@torch.no_grad()
def paired_diagnostic(model, features, labels):
    model.eval()
    choices=[]
    correct=[]
    for xs,ys in zip(features.split(128),labels.split(128)):
        action=paired_logits(model,xs).argmax(-1)
        choices.append(action.cpu())
        correct.append((ys.gather(-1,action[...,None]).squeeze(-1)>0).cpu())
    choices=torch.cat(choices);correct=torch.cat(correct)
    return dict(pairs=len(features),action_change_rate=float((choices[:,0]!=choices[:,1]).float().mean()),
                both_correct_rate=float(correct.all(-1).float().mean()),
                marginal_correct_rate=float(correct.float().mean()),
                note='Diagnostic only; a changed action alone is not evidence of correct goal use.')
