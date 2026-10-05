"""DEV-001门槛导航器：唯一改动=cue接受置信阈值参数化。

复用冻结的模型权重、特征映射、探索器与覆盖控制器方程；仅把冻结的0.50
接受阈值替换为构造参数threshold∈[0.5,1]。grid5路径沿用choose_cue（本就
接受参数阈值）；grid10/15路径复刻AreaNavigator.act但调用本模块的
thresholded_cue_choice（scaled_cue_choice按冻结保护拒绝非0.50阈值）。
threshold=0.50时决策必须与冻结导航器逐字节一致（回归臂验证）。
"""
from hashlib import sha256
import numpy as np
import torch

from agents.area_policy import area_policy_features, choose_area_coverage
from agents.massgis_navigator_v1 import NativeNavigator
from agents.target_cue import CLASSES, cue_features
from agents.edge_cue import edge_features
from env.episode import ACTIONS


def thresholded_cue_choice(values, threshold, position, visited, k):
    """scaled_cue_choice的参数化阈值版本；其余判据逐条一致。"""
    values = np.asarray(values, np.float64)
    if values.shape != (5,) or not np.isfinite(values).all() or (values < 0).any() \
            or (values > 1).any() or abs(values.sum() - 1) > 1e-5:
        raise ValueError('invalid five-way probabilities')
    top = int(values.argmax())
    if threshold is None:
        return None, 'uncalibrated_abstain'
    if not .5 <= threshold <= 1:
        raise ValueError('threshold must be within [.5,1] or null')
    if top == 4:
        return None, 'not_adjacent'
    if values[top] < threshold:
        return None, 'low_confidence'
    dr, dc = tuple(ACTIONS.values())[top]
    nr, nc = position[0] + dr, position[1] + dc
    if not (0 <= nr < k and 0 <= nc < k):
        return None, 'illegal_top_direction'
    if nr * k + nc in visited:
        return None, 'visited_top_destination'
    return CLASSES[top], 'accepted'


class ThresholdNativeNavigator(NativeNavigator):
    """唯一因素=threshold；0.50时与NativeNavigator决策完全一致。"""

    def __init__(self, legacy, k, policy='M0', condition='CueFull', threshold=.5):
        if not .5 <= threshold <= 1:
            raise ValueError('threshold must be within [.5,1]')
        super().__init__(legacy, k, policy, condition)
        self.threshold = threshold

    @torch.no_grad()
    def act(self, obs, current_global, current_local, target_global, target_local):
        if obs.grid_size != self.spec.grid_size:
            raise ValueError('navigator/observation grid mismatch')
        if self.spec.grid_size == 5:
            from agents.frozen_edge_navigator import FrozenEdgeNavigator
            return FrozenEdgeNavigator.act(self, obs, current_global, current_local,
                                           target_global, target_local)
        current = np.asarray(current_global, np.float32)
        target = np.asarray(target_global, np.float32)
        cl = np.asarray(current_local, np.float32)
        tl = np.asarray(target_local, np.float32)
        if (current.shape, target.shape, cl.shape, tl.shape) != ((512,), (512,), (4, 768), (4, 768)):
            raise ValueError('only two-image feature vectors allowed')
        if any(not np.isfinite(v).all() for v in (current, target, cl, tl)):
            raise ValueError('finite image features required')
        base = area_policy_features(self.em, current, obs.position, obs.remaining_budget,
                                    obs.visited, self.spec.grid_size, self.spec.budget)
        logits, _, _, self.hidden = self.explorer.step(
            torch.as_tensor(base, device=self.device)[None], self.hidden)
        proposal = tuple(ACTIONS)[int(logits.argmax(-1))]
        masked = self.condition == 'CueMean'
        semantic = cue_features(self.hm if masked else target, current, self.lm if masked else tl, cl)
        seam = edge_features(self.pm if masked else self.profile(obs.target_image),
                             self.profile(obs.current_image))
        x = np.concatenate((semantic, seam)).astype(np.float32)
        values = self.head(torch.as_tensor(x, device=self.device)[None]).softmax(-1)[0].cpu().numpy()
        cue, reason = thresholded_cue_choice(values, None if self.condition == 'Baseline' else self.threshold,
                                             obs.position, obs.visited, self.spec.grid_size)
        self.steps += 1
        result = dict(step=self.steps, public_position=list(obs.position), public_visited=list(obs.visited),
                      remaining_budget=obs.remaining_budget, explorer_action=proposal,
                      explorer_logits=logits[0].cpu().tolist(), action=cue or proposal,
                      cue_action=cue, reason=reason, probabilities=values.tolist(),
                      current_image_sha256=sha256(obs.current_image).hexdigest(),
                      target_image_sha256=sha256(obs.target_image).hexdigest(),
                      explorer_features_sha256=sha256(base.tobytes()).hexdigest(),
                      cue_features_sha256=sha256(x.tobytes()).hexdigest())
        if self.policy == 'Coverage3Radial':
            control = choose_area_coverage(obs.position, obs.visited, obs.remaining_budget,
                                           proposal, cue, grid_size=self.spec.grid_size,
                                           budget=self.spec.budget)
            result.update(base_action=result['action'], coverage=control, action=control['action'])
        return result
