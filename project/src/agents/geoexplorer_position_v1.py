"""One parameter-free position phase factor; public observations only."""
import math
from types import MethodType
from agents.geoexplorer_goal_probe_v1 import validate


def validate_phase(phase, grid=5):
    if type(phase) is not int or phase not in (1,2):raise ValueError('fixed phase 1 or 2 required')
    if type(grid) is not int or grid<2:raise ValueError('grid >=2 required')


def coordinate_features(positions, grid=5, phase=2):
    import torch
    validate_phase(phase,grid)
    if positions.dtype!=torch.long or positions.ndim!=2:raise ValueError('observed position matrix required')
    if positions.numel()==0 or bool((positions<0).any()) or bool((positions>=grid*grid).any()):
        raise ValueError('observed position out of grid')
    y=positions//grid;x=positions%grid
    return torch.stack((torch.sin(phase*math.pi*x/(grid-1)),torch.sin(phase*math.pi*y/(grid-1)),
                        torch.cos(phase*math.pi*x/(grid-1)),torch.cos(phase*math.pi*y/(grid-1))),dim=2)


def set_phase(model,phase):
    validate_phase(phase)
    original=model.transformer
    if not hasattr(original,'_original_coordinate_method'):
        original._original_coordinate_method=original.get_coord_feats
    original._position_phase=phase
    def observed_coordinates(self,patch_sequence,patch_size=5):
        return coordinate_features(patch_sequence,patch_size,self._position_phase)
    original.get_coord_feats=MethodType(observed_coordinates,original)


def use_original(model):
    t=model.transformer
    if hasattr(t,'_original_coordinate_method'):t.get_coord_feats=t._original_coordinate_method


def validate_public_request(request):return validate(request)


def validate_progress(progress,seed,arm,planned):
    if progress['seed']!=seed or progress['arm']!=arm:raise ValueError('cross-arm/seed checkpoint')
    if not 0<=progress['cursor']<=planned:raise ValueError('invalid cursor')
    expected=progress['cursor']//109
    if progress['updates']!=expected:raise ValueError('epoch flush/update mismatch')
    if progress.get('environment_steps',0)!=0:raise ValueError('no PPO/navigation in this factor')


def require_tail(tail,progress):
    if any(tail[k]!=progress[k] for k in ['seed','arm','cursor','updates']):
        raise ValueError('checkpoint older than committed input; no blind replay')
