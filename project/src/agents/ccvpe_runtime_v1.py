"""Fixed CCVPE CPU adapter. No labels, positions, network or implicit downloads."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFINITIONS = {'permute_channels', 'normalization', 'double_conv', 'CVM_KITTI'}


def selected_tree(text):
    tree = ast.parse(text)
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in DEFINITIONS]
    if {n.name for n in nodes} != DEFINITIONS:
        raise ValueError('Required official definitions missing')
    cls = next(n for n in nodes if n.name == 'CVM_KITTI')
    constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    replaced = 0
    for n in ast.walk(constructor):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'from_pretrained':
            if not isinstance(n.func.value, ast.Name) or n.func.value.id != 'EfficientNet':
                raise ValueError('Unexpected pretrained factory')
            n.func.attr = 'from_name'
            replaced += 1
    if replaced != 2:
        raise ValueError('Expected exactly two no-download factories')
    return ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))


def controlled_tree(text):
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'CVM_KITTI')
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
    if [a.arg for a in fn.args.args] != ['self', 'grd', 'sat']:
        raise ValueError('Unexpected original forward interface')
    fn.name = 'controlled_forward'
    fn.args.args.append(ast.arg(arg='disabled'))
    fn.args.defaults.append(ast.Constant(False))
    count = 0
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'matching_score':
            # Bypass cosine division BEFORE every yaw stack and either decoder.
            n.value = ast.IfExp(test=ast.Name(id='disabled', ctx=ast.Load()),
                body=ast.Call(func=ast.Attribute(value=ast.Name(id='torch', ctx=ast.Load()), attr='zeros_like', ctx=ast.Load()),
                              args=[ast.Name(id='sat_map_norm', ctx=ast.Load())], keywords=[]), orelse=n.value)
            count += 1
    if count != 6:
        raise ValueError('All six matching levels must be controlled')
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


def load_runtime(vendor, checkpoint):
    import torch
    from torch import nn
    from torch.nn import functional as F
    vendor = Path(vendor)
    name = '_cv030_efficientnet'
    if name in sys.modules:
        raise RuntimeError('One runtime per worker')
    spec = importlib.util.spec_from_file_location(name, vendor/'efficientnet_pytorch/__init__.py',
        submodule_search_locations=[str(vendor/'efficientnet_pytorch')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    text = (vendor/'upstream_models.py').read_text(encoding='utf-8-sig')
    env = dict(torch=torch, nn=nn, F=F, math=math, EfficientNet=package.EfficientNet, device=torch.device('cpu'))
    exec(compile(selected_tree(text), 'pinned_CC VPE_definitions', 'exec'), env)
    exec(compile(controlled_tree(text), 'pinned_CC VPE_controlled_forward', 'exec'), env)
    model = env['CVM_KITTI'](torch.device('cpu'))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Expected tensor-only official state_dict')
    own = model.state_dict()
    if own.keys() != state.keys():
        raise ValueError('Checkpoint key set mismatch')
    for key in own:
        if own[key].shape != state[key].shape or own[key].dtype != state[key].dtype:
            raise ValueError('Shape/dtype mismatch: '+key)
    checkpoint_manifest = state_manifest(state)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if state_manifest(model.state_dict()) != checkpoint_manifest:
        raise ValueError('Restored tensor identity mismatch')
    del own, state
    return model, env['controlled_forward'], checkpoint_manifest


def tensor_digest(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def state_manifest(state):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_digest(v)) for k, v in state.items()}


def prepare_rgb(image, ground=False):
    import numpy as np
    import torch
    from PIL import Image
    if image.mode != 'RGB':
        raise ValueError('RGB required')
    if ground:
        scale = min(1024/image.width, 256/image.height)
        w, h = round(image.width*scale), round(image.height*scale)
        a = np.empty((256, 1024, 3), dtype=np.float32)
        a[:] = MEAN
        x, y = (1024-w)//2, (256-h)//2
        a[y:y+h, x:x+w] = np.asarray(image.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)/255
    else:
        a = np.asarray(image.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32)/255
    a = (a-np.array(MEAN, np.float32))/np.array(STD, np.float32)
    return torch.from_numpy(a.copy()).permute(2, 0, 1).unsqueeze(0).contiguous()


def prepare_local_rgb(tile, cell):
    """Only the visible tile and PUBLIC current cell, never a parent image/goal."""
    import numpy as np
    import torch
    from PIL import Image
    if tile.mode != 'RGB' or tile.size != (125, 125) or not isinstance(cell, int) or not 0 <= cell < 25:
        raise ValueError('Expected one RGB125 tile and public cell0..24')
    canvas = np.empty((625, 625, 3), np.float32)
    canvas[:] = MEAN
    r, c = divmod(cell, 5)
    canvas[r*125:(r+1)*125, c*125:(c+1)*125] = np.asarray(tile, np.float32)/255
    # Float channels preserve exact mean in unknown space (no uint8 rounding).
    a = np.stack([np.asarray(Image.fromarray(canvas[:, :, i]).resize((512, 512), Image.Resampling.BILINEAR)) for i in range(3)], axis=-1)
    a = (a-np.array(MEAN, np.float32))/np.array(STD, np.float32)
    return torch.from_numpy(a.copy()).permute(2, 0, 1).unsqueeze(0).contiguous()


def route_ground(arm, ground, wrong=None):
    if arm not in ('Full', 'Disabled', 'Mean', 'Wrong'):
        raise ValueError('Unknown condition')
    if arm == 'Wrong':
        if wrong is None:
            raise ValueError('Wrong requires explicitly supplied alternate RGB tensor')
        return wrong
    return ground


def run_control(model, controlled_forward, ground, aerial, arm='Full', means=None, wrong=None):
    if arm == 'Mean' and (means is None or len(means) != 6):
        raise ValueError('Mean requires six calibration-only descriptors')
    actual = route_ground(arm, ground, wrong)
    captured, hooks = {}, []
    for level in range(1, 7):
        def hook(module, args, output, level=level):
            value = means[level-1].to(output) if arm == 'Mean' else output
            if value.shape != output.shape:
                raise ValueError('Descriptor shape mismatch')
            captured[level] = value.detach().clone()
            return value
        hooks.append(getattr(model, f'grd_feature_to_descriptor{level}').register_forward_hook(hook))
    try:
        outputs = controlled_forward(model, actual, aerial, disabled=arm == 'Disabled')
    finally:
        for h in hooks:
            h.remove()
    if sorted(captured) != list(range(1, 7)):
        raise ValueError('All descriptor levels must execute')
    return outputs, [captured[i] for i in range(1, 7)], actual
