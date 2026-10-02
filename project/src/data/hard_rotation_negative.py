"""Fit-only, balanced greedy assignment of high-scoring rotation negatives."""
import numpy as np
from data.rotation_negative import negative_angles


def hard_angles(labels,epoch,scores):
    labels=np.asarray(labels);base=negative_angles(labels,epoch);neg=np.flatnonzero(labels==4);scores=np.asarray(scores)
    if scores.shape!=(len(neg),3) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any() or len(neg)%6:raise ValueError('finite Nneg x3 joint scores; divisible by6')
    ranks=np.flatnonzero(base[neg]>0);quota=np.full(3,len(neg)//6,np.int64);result=np.zeros(len(labels),np.int64)
    ordered=np.lexsort((ranks,-scores[ranks].max(1)))
    for rank in ranks[ordered]:
        choices=np.where(quota>0,scores[rank],-1);view=int(choices.argmax());result[neg[rank]]=view+1;quota[view]-=1
    if quota.any():raise ValueError('unfilled angle quotas')
    return result
