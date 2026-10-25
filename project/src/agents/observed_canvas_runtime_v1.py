"""RGB-only public replay and independently calculated ROI peaks; no model import."""
import hashlib
import json
import math
import numpy as np
from PIL import Image
from project.src.platform_core.observed_canvas_v1 import ObservedCanvas,MEAN,STD
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame
from project.src.data.observed_canvas_protocol_v1 import valid_public_packet


def canonical(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def require_frozen_cases(cases,expected):
    if len(cases)!=44 or len({c['case_id'] for c in cases})!=44 or canonical(cases)!=canonical(expected):
        raise ValueError('Require original complete frozen44 cases;subsets cannot redefine planned denominator')


class Replay:
    def __init__(self,given_target):
        if type(given_target) is not RGBFrame:raise ValueError('Explicit RGB target required')
        self.target=given_target;self.core=ObservedCanvas();self.core.begin(given_target,budget=10);self.current=None
    def observe(self,packet,*,last_action=None):
        valid_public_packet(packet)
        if packet['given_target_rgb']!=self.target:raise ValueError('Target changed inside replay')
        public={k:v for k,v in packet.items() if k!='given_target_rgb'};public['position']=tuple(public['position'])
        receipt=self.core.observe_public(public,last_action=last_action);self.current=public['current_image'];return receipt
    def input_array(self,mode):
        if self.current is None:raise ValueError('No accepted observation')
        if mode=='Cumulative':return self.core.snapshot().normalized_nchw
        if mode!='Single':raise ValueError('Unknown frozen representation')
        cell=self.core.receipt()['position'][0]*5+self.core.receipt()['position'][1];r,c=divmod(cell,5)
        a=np.empty((625,625,3),np.float32);a[:]=MEAN
        a[r*125:(r+1)*125,c*125:(c+1)*125]=np.frombuffer(self.current.pixels,np.uint8).reshape(125,125,3).astype(np.float32)/255
        resized=np.stack([np.asarray(Image.fromarray(a[:,:,i]).resize((512,512),Image.Resampling.BILINEAR)) for i in range(3)],axis=-1)
        return ((resized-np.asarray(MEAN,np.float32))/np.asarray(STD,np.float32)).transpose(2,0,1)[None].copy()
    def close(self):self.core.close();self.current=None;self.target=None


def mask(size,cells):
    if size not in (256,512) or not cells or len(set(cells))!=len(cells) or any(type(c) is not int or not 0<=c<25 for c in cells):raise ValueError('Invalid public ROI')
    coords=(np.arange(size,dtype=np.float64)+.5)*625/size
    rowcell=(coords//125).astype(int);cellmap=rowcell[:,None]*5+rowcell[None,:]
    return np.isin(cellmap,cells)


def vector_peak(heat,matching,cells):
    if heat.shape!=(512,512) or matching.shape!=(256,256):raise ValueError('Frozen output sizes required')
    if not np.isfinite(heat).all() or not np.isfinite(matching).all():raise ValueError('Nonfinite output')
    hm=mask(512,cells);mm=mask(256,cells)
    hindex=int(np.where(hm,heat,-np.inf).argmax());y,x=divmod(hindex,512)
    score=float(matching[mm].max())
    if not -1.00001<=score<=1.00001:raise ValueError('Raw score range')
    return dict(score=score,xy=[(x+.5)*625/512,(y+.5)*625/512],heat_value=float(heat[y,x]),heat_index=hindex,
        heat_ROI_pixels=int(hm.sum()),matching_ROI_pixels=int(mm.sum()))


def alternate_peak(heat,matching,cells):
    # Independent rectangle traversal, global flat-index tie handling and per-cell maxima.
    top=None;score=-math.inf;nh=0;nm=0
    for cell in sorted(cells):
        r,c=divmod(cell,5)
        for size,array in ((512,heat),(256,matching)):
            y0=math.ceil(r*125*size/625-.5);y1=math.ceil((r+1)*125*size/625-.5)
            x0=math.ceil(c*125*size/625-.5);x1=math.ceil((c+1)*125*size/625-.5)
            part=array[y0:y1,x0:x1]
            if not np.isfinite(part).all():raise ValueError('Alternate nonfinite ROI')
            if size==512:
                n=int(part.argmax());yy,xx=divmod(n,x1-x0);index=(y0+yy)*512+x0+xx;value=float(part[yy,xx])
                candidate=(value,-index)
                if top is None or candidate>top:top=candidate
                nh+=part.size
            else:score=max(score,float(part.max()));nm+=part.size
    if top is None:raise ValueError('Empty ROI')
    index=-top[1];y,x=divmod(index,512)
    return dict(score=score,xy=[(x+.5)*625/512,(y+.5)*625/512],heat_value=top[0],heat_index=index,heat_ROI_pixels=nh,matching_ROI_pixels=nm)


def peaks_checked(heat,matching,visited,current):
    primary=vector_peak(heat,matching,visited);secondary=vector_peak(heat,matching,[current])
    independent=alternate_peak(heat,matching,visited);local=alternate_peak(heat,matching,[current])
    if primary!=independent or secondary!=local:raise ValueError('Independent ROI/peak mismatch')
    return primary,secondary,dict(passed=True,primary=independent,current=local)


def resource_violation(rss,wall,bytes_used):return rss>4*1024**3 or wall>1800 or bytes_used>16*1024**2
