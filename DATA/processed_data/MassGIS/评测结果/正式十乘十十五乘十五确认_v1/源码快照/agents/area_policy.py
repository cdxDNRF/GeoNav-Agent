"""Protocol-scoped defaults and a frozen, parameterized visual navigator."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import torch
from agents.frozen_edge_navigator import FrozenEdgeNavigator, load_frozen_edge_default
from agents.scaled_edge_navigator import scaled_cue_choice
from agents.target_cue import cue_features
from agents.edge_cue import edge_features
from agents.parameterized_coverage import choose_area_coverage
from env.area_protocol import get_protocol
from env.episode import ACTIONS


def area_policy_features(target, current, position, remaining, visited, k, budget):
    if type(k) is not int or k not in (10, 15) or type(budget) is not int or budget != 20:
        raise ValueError('registered grid10/grid15 B20 feature mapping required')
    if len(position) != 2 or any(type(v) is not int or not 0 <= v < k for v in position):
        raise ValueError('valid public position required')
    if type(remaining) is not int or not 1 <= remaining <= budget:
        raise ValueError('valid remaining budget required')
    visited = tuple(visited)
    if len(visited) != budget - remaining + 1 or visited[-1] != position[0] * k + position[1]:
        raise ValueError('consecutive public observation prefix required')
    if any(type(v) is not int or not 0 <= v < k * k for v in visited):
        raise ValueError('invalid visited cell')
    target, current = np.asarray(target, np.float32), np.asarray(current, np.float32)
    if target.shape != (512,) or current.shape != (512,) or not np.isfinite(target).all() or not np.isfinite(current).all():
        raise ValueError('two finite 512-wide vectors required')
    ids = np.asarray(visited, np.int64)
    pooled = (ids // k * 5 // k) * 5 + (ids % k * 5 // k)
    counts = np.minimum(np.bincount(pooled, minlength=25).astype(np.float32), 3) / 3
    r, c = position
    return np.concatenate((target, current, np.asarray([r / (k - 1), c / (k - 1), remaining / budget], np.float32), counts)).astype(np.float32)


class AreaNavigator(FrozenEdgeNavigator):
    def __init__(self, legacy, protocol, policy, condition):
        self.spec = get_protocol(protocol)
        if policy not in ('M0', 'Coverage3Radial'):
            raise ValueError('known frozen policy required')
        self.policy = policy
        super().__init__(legacy.explorer, legacy.head, legacy.device, legacy.em, legacy.hm, legacy.lm,
                         legacy.pm, .5, 'Edge', condition)

    @torch.no_grad()
    def act(self, obs, current_global, current_local, target_global, target_local):
        if obs.grid_size != self.spec.grid_size:
            raise ValueError('navigator/observation grid mismatch')
        current, target = np.asarray(current_global, np.float32), np.asarray(target_global, np.float32)
        cl, tl = np.asarray(current_local, np.float32), np.asarray(target_local, np.float32)
        if (current.shape, target.shape, cl.shape, tl.shape) != ((512,), (512,), (4, 768), (4, 768)):
            raise ValueError('only two-image feature vectors allowed')
        if any(not np.isfinite(v).all() for v in (current, target, cl, tl)):
            raise ValueError('finite image features required')
        base = area_policy_features(self.em, current, obs.position, obs.remaining_budget, obs.visited,
                                   self.spec.grid_size, self.spec.budget)
        logits, _, _, self.hidden = self.explorer.step(torch.as_tensor(base, device=self.device)[None], self.hidden)
        proposal = tuple(ACTIONS)[int(logits.argmax(-1))]
        masked = self.condition == 'CueMean'
        semantic = cue_features(self.hm if masked else target, current, self.lm if masked else tl, cl)
        seam = edge_features(self.pm if masked else self.profile(obs.target_image), self.profile(obs.current_image))
        x = np.concatenate((semantic, seam)).astype(np.float32)
        values = self.head(torch.as_tensor(x, device=self.device)[None]).softmax(-1)[0].cpu().numpy()
        cue, reason = scaled_cue_choice(values, None if self.condition == 'Baseline' else .5,
                                        obs.position, obs.visited, self.spec.grid_size)
        self.steps += 1
        result = dict(step=self.steps, public_position=list(obs.position), public_visited=list(obs.visited),
            remaining_budget=obs.remaining_budget, explorer_action=proposal, explorer_logits=logits[0].cpu().tolist(),
            action=cue or proposal, cue_action=cue, reason=reason, probabilities=values.tolist(),
            current_image_sha256=sha256(obs.current_image).hexdigest(), target_image_sha256=sha256(obs.target_image).hexdigest(),
            explorer_features_sha256=sha256(base.tobytes()).hexdigest(), cue_features_sha256=sha256(x.tobytes()).hexdigest())
        if self.policy == 'Coverage3Radial':
            control = choose_area_coverage(obs.position, obs.visited, obs.remaining_budget, proposal, cue,
                                           grid_size=self.spec.grid_size, budget=self.spec.budget)
            result.update(base_action=result['action'], coverage=control, action=control['action'])
        return result

    def act_with_profiles(self, obs, current_global, current_local, target_global, target_local, current_profile, target_profile):
        for payload, profile in ((obs.current_image, current_profile), (obs.target_image, target_profile)):
            value = np.asarray(profile, np.float32)
            if value.shape != (4, 3, 64, 3) or not np.isfinite(value).all():
                raise ValueError('two finite observed-image profiles required')
            self.profiles[sha256(payload).hexdigest()] = value
        return self.act(obs, current_global, current_local, target_global, target_local)


def checked_record(root, record):
    path = (root / record['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('configuration/evidence provenance mismatch')
    return path


def resolve_area_config(root, protocol, *, allow_experimental=False):
    root = Path(root).resolve()
    spec = get_protocol(protocol)
    registry = json.loads((root / 'project/local_policy_defaults_v1.json').read_text('utf-8'))
    if registry['version'] != 'protocol-default-registry-v1':
        raise ValueError('unknown default registry version')
    group = registry['defaults']
    if protocol not in group:
        if not allow_experimental or protocol not in registry['experimental']:
            raise ValueError('protocol has no confirmed default; explicitly opt in to engineering candidate')
        group = registry['experimental']
    config = json.loads(checked_record(root, group[protocol]).read_text('utf-8'))
    geometry = config['protocol']
    if config['version'] != 'area-policy-v1' or config['policy'] != 'Coverage3Radial':
        raise ValueError('unsupported area configuration')
    expected = dict(name=spec.name, grid_size=spec.grid_size, budget=spec.budget, cell_size_m=300,
                    native_cell_pixels=300, epsg=26986, pixel_size_m=1, source_family='MassachusettsRoads')
    if geometry != expected:
        raise ValueError('strict physical protocol scope mismatch')
    if config['thresholds'] != {'0': .5, '1': .5, '2': .5} or config['reference_seed'] != 0:
        raise ValueError('frozen thresholds/reference seed changed')
    if config['controller'] != dict(horizon=3, minimum_gain=2, accepted_cue_priority=True,
        guard='one_step_radius_not_less_than_original', radius_reference='public_visited_first_cell',
        replan_each_observation=True, no_second_candidate_selection=True):
        raise ValueError('frozen controller changed')
    parent = json.loads(checked_record(root, config['parent_M0']).read_text('utf-8'))
    for key in ('checkpoints', 'cue_heads', 'means'):
        if config[key] != parent[key]:
            raise ValueError('paired frozen model/mean records changed')
    for record in config['evidence'].values():
        checked_record(root, record)
    verdict = json.loads(checked_record(root, config['evidence']['confirmation_verdict']).read_text('utf-8'))
    if not (verdict['completed'] and verdict['eligible_for_default_upgrade']
            and verdict['independent_main_passed'] and verdict['independent_target_evidence_passed']):
        raise ValueError('complete independent confirmation evidence required')
    expected_status = 'confirmed_default' if group is registry['defaults'] else 'engineering_candidate_only'
    if config['status'] != expected_status or (spec.grid_size == 15 and expected_status != 'engineering_candidate_only'):
        raise ValueError('experimental geometry cannot inherit confirmation status')
    return config, parent


def load_area_policy(root, protocol, seed=0, condition='CueFull', device='cpu', *, allow_experimental=False, reference_M0=False):
    config, parent = resolve_area_config(root, protocol, allow_experimental=allow_experimental)
    legacy = load_frozen_edge_default(parent, root, seed=seed, device=device)
    return AreaNavigator(legacy, protocol, 'M0' if reference_M0 else config['policy'], condition)
