"""Independent spatial pooling/profile checks on fixed public engineering images."""
from pathlib import Path
from io import BytesIO
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_native_v1 as m
from agents.massgis_edge_bridge_v1 import native_profiles,native_edge_features
from agents.edge_cue import image_profiles
from env.massgis_area_v1 import PROTOCOL,MassGISEpisode,MassGISAreaEnv

def main():
    binding=m.read(m.QA/'模型输入适配补充绑定.json')
    for k,v in binding['files_sha256'].items():
        if m.digest(m.ROOT/k)!=v:raise ValueError('Supplement changed before engineering test')
    area=m.read(m.META/'区域像素与逐格来源.json')
    if not area['quality']['passed'] or not area['source_quality_passed']:
        m.write(m.QA/'目标特征接口独立复核.json',dict(passed=False,executed=False,reason='quality not passed',new_SR=False));return
    env=MassGISAreaEnv(m.DATA);checks=[]
    for current,target in ((0,1),(28,29),(224,223)):
        ep=MassGISEpisode('bridge_interface_'+str(current),'dev','img_6000',current,target,1,20,15,PROTOCOL,'MassGIS2005/known_geo_engineering_00')
        obs=env.reset(ep)
        profiles=[]
        for payload in (obs.current_image,obs.target_image):
            with Image.open(BytesIO(payload)) as im:a=np.asarray(im,dtype=np.uint8)
            # Pillow BOX uses two-pass rounding; reproduce row then column pooling.
            x=a.astype(np.uint16);horizontal=(x[:,0::2]+x[:,1::2]+1)//2
            pooled=((horizontal[0::2]+horizontal[1::2]+1)//2).astype(np.uint8)
            actual=native_profiles(payload);expected=image_profiles(pooled)
            if not np.array_equal(actual,expected):raise ValueError('Independent model-pooling mismatch')
            profiles.append(actual)
        f=native_edge_features(obs.current_image,obs.target_image)
        if f.shape!=(20,) or not np.isfinite(f).all():raise ValueError('Frozen20-feature shape mismatch')
        checks.append(dict(current_cell=current,target_cell=target,profile_sha256=[m.old.sha256(p.tobytes()).hexdigest() for p in profiles],
            feature_sha256=m.old.sha256(f.tobytes()).hexdigest(),pooling='explicit two-by-two BOX; stored native600 unchanged'))
    m.write(m.QA/'目标特征接口独立复核.json',dict(passed=True,executed=True,checks=checks,head_forwards=0,
        policy_calls=0,new_SR=False,head_navigation_effectiveness_verified=False))
    print('Native target-feature input bridge passed on3public pairs; no head/agent forward',flush=True)

if __name__=='__main__':main()
