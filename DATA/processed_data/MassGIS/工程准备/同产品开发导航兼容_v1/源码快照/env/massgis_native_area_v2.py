"""Explicit dev-only 0.5m/600PNG protocols; no global registry mutation."""
from dataclasses import dataclass
from pathlib import Path
import json
import re
from PIL import Image
from env.area_protocol import AreaProtocol
from env.parameterized_area import AreaEpisode,ParameterizedAreaEnv
from env.scaled_grid import ScaledGridEnv,distance

def native_spec(k):
    if type(k) is not int or k not in (5,10,15):raise ValueError('Explicit native grid5/10/15 required')
    return AreaProtocol('massgis-native-grid'+str(k)+'-dev-v2',k,10 if k==5 else 20,300,600,26986,.5)

@dataclass(frozen=True)
class NativeEpisode(AreaEpisode):
    def validate(self):
        s=native_spec(self.grid_size)
        if self.protocol!=s.name or self.split!='dev' or not isinstance(self.area,str) or re.fullmatch(r'img_\d+',self.area) is None:raise ValueError('Native dev identity required')
        if any(type(getattr(self,n)) is not int for n in ('start','goal','dist','budget','grid_size')):raise ValueError('Integer fields required')
        if self.budget!=s.budget or not 0<=self.start<s.grid_size**2 or not 0<=self.goal<s.grid_size**2 or self.start==self.goal:raise ValueError('Budget/endpoints differ')
        if self.dist!=distance(self.start,self.goal,s.grid_size) or not 1<=self.dist<=s.budget:raise ValueError('Reachable goal required')
        if any(not isinstance(v,str) or not v for v in (self.episode_id,self.source_tile)):raise ValueError('Nonempty bindings required')

class NativeAreaEnv(ParameterizedAreaEnv):
    def __init__(self,root):
        root=Path(root);manifest=json.loads((root/'数据清单.json').read_text('utf-8'));self.spec=native_spec(manifest['grid_size'])
        expected=dict(protocol=self.spec.name,grid_size=self.spec.grid_size,budget=self.spec.budget,cell_size_m=300,native_cell_pixels=600,
          epsg=26986,pixel_size_m=.5,mosaic_pixels=600*self.spec.grid_size,patch_format='png',development_only=True)
        if any(type(manifest.get(k)) is not type(v) or manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Native development manifest mismatch')
        self._bindings={(r['split'],r['area']):r['source_tile'] for r in manifest['regions']}
        if not self._bindings or len(self._bindings)!=len(manifest['regions']) or any(k[0]!='dev' for k in self._bindings):raise ValueError('Only unique dev bindings allowed')
        ScaledGridEnv.__init__(self,root)
    def reset(self,ep):
        if not isinstance(ep,NativeEpisode):raise ValueError('NativeEpisode required')
        ep.validate();key=(ep.split,ep.area)
        if ep.protocol!=self.spec.name or self._bindings.get(key)!=ep.source_tile:raise ValueError('Protocol/source binding differs')
        sig=self.signature(ep);k=self.spec.grid_size
        if key not in self._validated:
            folder=self.root/'patches'/ep.split/ep.area
            if {p.name for p in folder.glob('patch_*')}!={f'patch_{j}.png' for j in range(k*k)}:raise ValueError('Exact native patch set required')
            for p in folder.glob('patch_*.png'):
                with Image.open(p) as im:
                    im.load()
                    if im.mode!='RGB' or im.size!=(600,600):raise ValueError('Native600 RGB required')
            self._validated[key]=sig
        elif self._validated[key]!=sig:raise ValueError('Immutable area changed')
        self._episode,self._position,self._remaining=ep,ep.start,ep.budget
        self._visited=[ep.start];self._trajectory=[dict(step=0,patch_id=ep.start,action=None,out_of_bounds=False,revisited=False)]
        self._done=self._success=False;self._target=self.payload(ep.goal)
        return self.observation()
    def path(self,cell):return self.root/'patches'/self._episode.split/self._episode.area/f'patch_{cell}.png'
