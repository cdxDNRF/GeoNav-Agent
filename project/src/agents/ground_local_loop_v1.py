"""Experimental ground-goal loop. Only public current features enter this module.

No filesystem/model/answer access. Features must come from a separately bound
encoder or cache adapter. This is not an implementation of GOMAA/GeoExplorer.
"""
from collections import deque
import math
import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
FIELDS = {'position', 'current_feature', 'grid_size', 'step_index', 'remaining_steps'}


def feature(value):
    x = np.array(value, dtype=np.float64, copy=True)
    if x.shape != (512,) or not np.isfinite(x).all():
        raise ValueError('Only one finite 512D current/goal vector is permitted')
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        raise ValueError('Zero feature is invalid; disabled goal uses None')
    return x / norm


def neighbors(cell, grid):
    r, c = divmod(cell, grid)
    for action, (dr, dc) in enumerate(ACTIONS):
        rr, cc = r + dr, c + dc
        if 0 <= rr < grid and 0 <= cc < grid:
            yield action, rr * grid + cc


class GroundLocalLoop:
    """Nearest frontier; goal evidence breaks ties between equal-cost frontiers.

    At visited locations x_i, s_i = cosine(current_i, goal). A least squares
    affine fit predicts scores only to order reachable nearest frontiers.
    Rank = (path cost, -predicted score, public cell index). No threshold,
    calibrated confidence, landmark label, hidden destination or stop action.
    """

    def begin(self, *, grid_size, budget, goal_feature, use_evidence=True):
        self.active = False
        if type(grid_size) is not int or not 2 <= grid_size <= 32:
            raise ValueError('Invalid grid')
        if type(budget) is not int or not 1 <= budget <= 1000:
            raise ValueError('Invalid budget')
        if type(use_evidence) is not bool:
            raise ValueError('Invalid evidence mode')
        self.goal = None if goal_feature is None else feature(goal_feature)
        self.grid, self.budget, self.evidence = grid_size, budget, use_evidence
        self.views, self.scores = {}, {}
        self.expected_position, self.next_step = None, 0
        self.active = True

    def finish(self):
        self.active = False

    def observe(self, payload):
        if not getattr(self, 'active', False):
            raise ValueError('Begin a fresh episode; failed/finished sessions cannot continue')
        try:
            return self._observe(payload)
        except Exception:
            self.active = False
            raise

    def _observe(self, payload):
        if set(payload) != FIELDS:
            raise ValueError('Unexpected input field (bank/answer/metadata forbidden)')
        if any(type(payload[k]) is not int for k in ('grid_size', 'position', 'step_index', 'remaining_steps')):
            raise ValueError('Public geometry must use integers')
        cell, step = payload['position'], payload['step_index']
        if payload['grid_size'] != self.grid or not 0 <= cell < self.grid ** 2:
            raise ValueError('Invalid public position/grid')
        if step != self.next_step or payload['remaining_steps'] != self.budget - step or step >= self.budget:
            raise ValueError('Step/budget discontinuity or exhausted session')
        if self.expected_position is not None and cell != self.expected_position:
            raise ValueError('Observed position is inconsistent with prior action')
        vector = feature(payload['current_feature'])
        if cell in self.views and not np.allclose(self.views[cell], vector, rtol=0, atol=1e-12):
            raise ValueError('Frozen observation changed on revisit')
        self.views[cell] = vector
        self.scores[cell] = 0.0 if self.goal is None else float(vector @ self.goal)

        # Expand known topology only. Unknown cells may be frontier endpoints,
        # never intermediate nodes with unobserved imagery.
        paths, queue = {cell: []}, deque([cell])
        while queue:
            here = queue.popleft()
            for action, nxt in neighbors(here, self.grid):
                if nxt not in paths:
                    paths[nxt] = paths[here] + [action]
                    if nxt in self.views:
                        queue.append(nxt)
        frontiers = [p for p in paths if p not in self.views]
        ids = sorted(self.views)
        points = np.array([divmod(p, self.grid) for p in ids], dtype=np.float64)
        values = np.array([self.scores[p] for p in ids])
        origin, mean = points.mean(0), float(values.mean())
        gradient = np.zeros(2)
        if self.goal is not None and self.evidence and len(ids) >= 2:
            gradient = np.linalg.lstsq(points - origin, values - mean, rcond=None)[0]
        # Round at 1e-12 so algebraically equal forecasts keep the public-index
        # tie-break rather than depend on BLAS roundoff.
        predictions = {p: round(mean + float((np.array(divmod(p, self.grid)) - origin) @ gradient), 12) for p in frontiers}
        if not all(math.isfinite(x) for x in predictions.values()):
            raise ValueError('Nonfinite frontier estimate')
        if frontiers:
            selected = min(frontiers, key=lambda p: (len(paths[p]), -predictions[p] if self.evidence else 0., p))
            action = paths[selected][0]
        else:
            # All cells observed but evaluator has not terminated: remain a
            # four-action agent. No hidden goal is inferred from this condition.
            action, selected = min(neighbors(cell, self.grid))
        destination = dict(neighbors(cell, self.grid))[action]
        event = {'action': action, 'position': cell, 'step_index': step,
                 'observed_cells': ids, 'local_score': self.scores[cell],
                 'gradient': gradient.tolist(), 'frontier': selected,
                 'frontier_estimate': predictions.get(selected),
                 'reason': 'nearest frontier, local similarity tie-break' if self.evidence and self.goal is not None else 'nearest frontier, public index tie-break',
                 'confidence': None, 'experimental': True}
        self.expected_position, self.next_step = destination, step + 1
        return event
