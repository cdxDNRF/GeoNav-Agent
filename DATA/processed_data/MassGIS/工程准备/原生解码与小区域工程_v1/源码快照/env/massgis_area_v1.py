"""Engineering-only native 0.5m/PNG adapter; sealed 1m protocols stay intact."""
from dataclasses import dataclass
from pathlib import Path
import json
import re
from PIL import Image
from env.area_protocol import AreaProtocol
from env.parameterized_area import AreaEpisode, ParameterizedAreaEnv
from env.scaled_grid import ScaledGridEnv, distance

PROTOCOL='massgis-grid15-halfm-engineering-v1'
SPEC=AreaProtocol(PROTOCOL,15,native_cell_pixels=600,pixel_size_m=.5)

@dataclass(frozen=True)
class MassGISEpisode(AreaEpisode):
    def validate(self):
        if self.protocol!=PROTOCOL or self.split!='dev' or not isinstance(self.area,str) or re.fullmatch(r'img_\d+',self.area) is None:
            raise ValueError('Registered engineering dev identity required')
        if any(type(getattr(self,n)) is not int for n in ('start','goal','dist','budget','grid_size')):
            raise ValueError('Integer episode fields required')
        if self.grid_size!=15 or self.budget!=20 or not 0<=self.start<225 or not 0<=self.goal<225 or self.start==self.goal:
            raise ValueError('Native grid15/B20 endpoints required')
        if self.dist!=distance(self.start,self.goal,15) or not 1<=self.dist<=20:raise ValueError('Reachable evaluator endpoints required')
        if any(not isinstance(s,str) or not s for s in (self.episode_id,self.source_tile)):raise ValueError('Identifiers required')

class MassGISAreaEnv(ParameterizedAreaEnv):
    def __init__(self,root):
        manifest=json.loads((Path(root)/'数据清单.json').read_text('utf-8'))
        expected=dict(protocol=PROTOCOL,grid_size=15,budget=20,cell_size_m=300,native_cell_pixels=600,
            epsg=26986,pixel_size_m=.5,mosaic_pixels=9000,patch_format='png',engineering_only=True)
        if any(manifest.get(k)!=v or type(manifest.get(k)) is not type(v) for k,v in expected.items()):
            raise ValueError('Native half-meter engineering manifest mismatch')
        self.spec=SPEC
        self._bindings={(r['split'],r['area']):r['source_tile'] for r in manifest['regions']}
        if not self._bindings or len(self._bindings)!=len(manifest['regions']) or any(k[0]!='dev' for k in self._bindings):
            raise ValueError('Unique engineering dev bindings required')
        ScaledGridEnv.__init__(self,root)

    def reset(self,ep):
        if not isinstance(ep,MassGISEpisode):raise ValueError('MassGISEpisode required')
        ep.validate();key=(ep.split,ep.area)
        if self._bindings.get(key)!=ep.source_tile:raise ValueError('Source binding mismatch')
        sig=self.signature(ep)
        if key not in self._validated:
            folder=self.root/'patches'/ep.split/ep.area
            if {p.name for p in folder.glob('patch_*')}!={f'patch_{i}.png' for i in range(225)}:
                raise ValueError('225 native PNG patches required')
            for p in folder.glob('patch_*.png'):
                with Image.open(p) as im:
                    im.load()
                    if im.mode!='RGB' or im.size!=(600,600):raise ValueError('Native600 RGB required')
            self._validated[key]=sig
        elif self._validated[key]!=sig:raise ValueError('Immutable engineering area changed')
        self._episode,self._position,self._remaining=ep,ep.start,ep.budget
        self._visited=[ep.start];self._trajectory=[dict(step=0,patch_id=ep.start,action=None,out_of_bounds=False,revisited=False)]
        self._success=self._done=False;self._target=self.payload(ep.goal)
        return self.observation()

    def path(self,cell):
        ep=self._episode
        return self.root/'patches'/ep.split/ep.area/f'patch_{cell}.png'
