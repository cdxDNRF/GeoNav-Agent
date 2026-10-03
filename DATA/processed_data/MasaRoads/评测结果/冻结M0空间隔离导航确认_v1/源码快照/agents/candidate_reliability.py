"""Scalar threshold screening of frozen four-view scores, using calibration labels only."""
import numpy as np
from agents.target_cue import choose_cue
from agents.rotation_candidate import select_view
from hashlib import sha256

THRESHOLDS=(.5,.6,.7,.8,.9,.95,.975,.99,.995,.999,1.)
CAL_GATE=dict(accepted=100,sources=5,precision=.95,precision_CI_low=.90)


def winners(probabilities,payloads,bank):
    p=np.asarray(probabilities);expected=(len(bank),4,5)
    if p.shape!=expected:raise ValueError('N by four raw joint vectors required')
    ids=[];values=[]
    for i,row in enumerate(bank):
        keys=[sha256(payloads[a][row['area'],row['target']]).hexdigest() for a in (0,90,180,270)]
        j,_=select_view(p[i],keys);ids.append(j);values.append(p[i,j])
    return np.array(ids),np.array(values)


def threshold_metrics(values,bank,threshold):
    p=np.asarray(values)
    if p.shape!=(len(bank),5) or not np.isfinite(p).all():raise ValueError('raw winner vectors')
    accepted=[];correct=[];adjacent=[];sources=sorted({r['area'] for r in bank});source_index={a:i for i,a in enumerate(sources)}
    counts=np.zeros((len(sources),2),np.int64)
    for prob,row in zip(p,bank):
        cue,_=choose_cue(prob,threshold,divmod(row['current'],5),(row['current'],))
        ok=cue is not None;hit=ok and int(prob.argmax())==row['label'];accepted.append(ok);correct.append(hit);adjacent.append(row['label']<4)
        counts[source_index[row['area']],0]+=ok;counts[source_index[row['area']],1]+=hit
    idx=np.random.default_rng(4119).integers(len(sources),size=(2000,len(sources)));den=counts[idx,0].sum(1);num=counts[idx,1].sum(1)
    samples=np.divide(num,den,out=np.zeros(2000,float),where=den>0);n=sum(accepted);k=sum(correct)
    return dict(samples=len(bank),accepted=n,correct_accepted=k,false_accepts=n-k,accepted_precision=k/n if n else None,
        acceptance_coverage=n/len(bank),adjacent_recall=k/sum(adjacent),sources_with_acceptances=int((counts[:,0]>0).sum()),
        by_source={a:dict(accepted=int(counts[i,0]),correct_accepted=int(counts[i,1])) for a,i in source_index.items()},
        precision_interval=dict(interval95=np.quantile(samples,[.025,.975]).tolist(),resamples=2000,seed=4119,source_count=len(sources),zero_accept_resamples=int((den==0).sum()),scope='known calibration source clusters; screening interval, not unseen-domain guarantee'))


def calibrate(values,bank):
    rows=[];selected=None
    for t in THRESHOLDS:
        m=threshold_metrics(values,bank,t);valid=m['accepted']>=100 and m['sources_with_acceptances']>=5 and m['accepted_precision']>=.95 and m['precision_interval']['interval95'][0]>=.90
        rows.append(dict(threshold=t,passed=bool(valid),metrics=m))
        if valid and selected is None:selected=t
    return dict(threshold=selected,grid=rows,calibration_only=True,all_abstain=selected is None,
        selection='lowest predefined threshold meeting minimum support, empirical precision95% and source-screening lower90%',precision_not_probabilistically_calibrated=True)
