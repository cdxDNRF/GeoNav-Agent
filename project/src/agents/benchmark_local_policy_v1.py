"""Public-only projections for the real common benchmark; lazy model loading."""
from hashlib import sha256
from pathlib import Path
import json

from agents.benchmark_contract_v1 import FeatureBundle, PolicyRequest

SCHEMA = (('global', 512), ('local', 3072), ('profile', 2304))


class GlobalOnlyPolicy:
    """Same public images, only the registered global channel enters GeoExplorer."""
    def __init__(self, policy): self.policy = policy
    def reset(self): self.policy.reset()
    def decide(self, request):
        def project(bundle):
            if bundle is None: return None
            channels = dict(bundle.channels)
            if 'global' not in channels or len(channels['global']) != 512:
                raise ValueError('registered global channel missing')
            return FeatureBundle((('global', channels['global']),))
        return self.policy.decide(PolicyRequest(request.observation, project(request.current_features),
                                 project(request.target_features), request.target_enabled, request.last_step))


class M0PublicPolicy:
    """Existing frozen M0 equations with only current/target public features.

    Disabled supplies constant fit means and disables cue acceptance. Mean is a
    method-specific fit-mean payload from the evaluator, never a test-set mean.
    """
    def __init__(self, navigator):
        self.navigator = navigator
        self.reset()

    def reset(self):
        self.navigator.reset(); self._step = 0; self._target = None; self.last_decision = None

    def decide(self, request):
        import numpy as np
        observed_step = 0 if request.last_step is None else request.last_step.step_count
        if observed_step != self._step: raise ValueError('reset required at episode boundary')
        if self._step and self._target != request.target_features:
            raise ValueError('target changed within episode')
        self._target = request.target_features
        def unpack(bundle):
            if bundle.schema != SCHEMA: raise ValueError('exact global/local/profile schema required')
            values = dict(bundle.channels)
            return (np.asarray(values['global'], np.float32),
                    np.asarray(values['local'], np.float32).reshape(4, 768),
                    np.asarray(values['profile'], np.float32).reshape(4, 3, 64, 3))
        current, local, profile = unpack(request.current_features)
        if request.target_enabled:
            target, target_local, target_profile = unpack(request.target_features)
            self.navigator.condition = 'CueFull'
        else:
            target, target_local, target_profile = self.navigator.hm, self.navigator.lm, self.navigator.pm
            self.navigator.condition = 'Baseline'
        # Both cached profiles are derived from the two permitted images (or
        # fixed training means for the artificial Mean/Disabled interventions).
        self.navigator.profiles[sha256(request.observation.current_image).hexdigest()] = profile
        self.navigator.profiles[sha256(request.observation.target_image).hexdigest()] = target_profile
        self.last_decision = self.navigator.act(request.observation, current, local, target, target_local)
        self._step += 1
        return self.last_decision['action']


def load_m0(root, config, seed, device='cuda:0'):
    from agents.frozen_edge_navigator import load_frozen_edge_default
    return M0PublicPolicy(load_frozen_edge_default(config, root, seed, device))


def file_sha(path):
    h = sha256()
    with Path(path).open('rb') as f:
        while value := f.read(4 * 1024 * 1024): h.update(value)
    return h.hexdigest()


def load_geoexplorer(root, protocol_path, final_head_path, seed, device='cuda:0'):
    """Load SHA-bound final head plus history; no feature bank enters the policy."""
    import numpy as np
    import torch
    from agents.geoexplorer_policy_v1 import GeoExplorerPublicPolicy, real_selector
    from train.geoexplorer_training_v1 import modules
    root = Path(root).resolve(); protocol_path = Path(protocol_path).resolve()
    final = Path(final_head_path).resolve()
    if root not in final.parents or root not in protocol_path.parents or final.name != 'navigation_final.pt':
        raise ValueError('registered final local checkpoint required')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8')); ph = file_sha(protocol_path)
    seal = json.loads(Path(str(final) + '.seal.json').read_text(encoding='utf-8'))
    expected = dict(seed=seed, phase='ppo', epoch=300, microbatches=32700,
                    pretrain_updates=300, environment_steps=163840, ppo_updates=256)
    if seal['protocol_sha256'] != ph or file_sha(final) != seal['sha256'] or seal['progress'] != expected:
        raise ValueError('final checkpoint provenance or budget differs')
    if torch.device(device).type != 'cuda': raise ValueError('original history implementation requires CUDA')
    payload = torch.load(final, map_location='cpu', mmap=True, weights_only=False)
    if payload['protocol_sha256'] != ph or payload['progress'] != expected:
        raise ValueError('checkpoint payload differs from seal')
    history = (root / payload['history_checkpoint']).resolve()
    if root not in history.parents or history.name != 'pretrain_epoch_300.pt' or history.parent != final.parent:
        raise ValueError('same-seed final history checkpoint required')
    if file_sha(history) != payload['history_sha256']: raise ValueError('history byte binding differs')
    history_seal = json.loads(Path(str(history) + '.seal.json').read_text(encoding='utf-8'))
    if history_seal['sha256'] != payload['history_sha256'] or history_seal['protocol_sha256'] != ph:
        raise ValueError('history seal differs')
    model_class, config, actor_class = modules(protocol)
    model = model_class(config).to(device, dtype=torch.float32)
    history_payload = torch.load(history, map_location='cpu', mmap=True, weights_only=False)
    if history_payload['progress']['seed'] != seed or history_payload['progress']['epoch'] != 300:
        raise ValueError('history payload seed/budget differs')
    model.load_state_dict(history_payload['model']); del history_payload
    actor = actor_class(512, 4).to(device)
    actor.load_state_dict(payload['model']); del payload
    model.eval().requires_grad_(False); actor.eval().requires_grad_(False)
    mean = np.load(root / protocol['fit_mean'], allow_pickle=False)
    return GlobalOnlyPolicy(GeoExplorerPublicPolicy(real_selector(model, actor), mean))
