"""Native-scale grid10/grid15 environment with unchanged public observation."""
from dataclasses import dataclass
from pathlib import Path
import json
import re
from PIL import Image
from env.environment import Observation, StepInfo
from env.episode import ACTIONS
from env.scaled_grid import ScaledGridEnv, distance
from env.area_protocol import get_protocol, validate_manifest


@dataclass(frozen=True)
class AreaEpisode:
    episode_id: str
    split: str
    area: str
    start: int
    goal: int
    dist: int
    budget: int
    grid_size: int
    protocol: str
    source_tile: str

    def validate(self):
        spec = get_protocol(self.protocol)
        if self.split not in ('dev', 'test') or not isinstance(self.area, str) or re.fullmatch(r'img_\d+', self.area) is None:
            raise ValueError('safe area identity required')
        if any(not isinstance(v, str) or not v for v in (self.episode_id, self.source_tile)):
            raise ValueError('episode and source identity required')
        if any(type(getattr(self, n)) is not int for n in ('start', 'goal', 'dist', 'budget', 'grid_size')):
            raise ValueError('integer protocol fields required')
        if (self.grid_size, self.budget) != (spec.grid_size, spec.budget):
            raise ValueError('episode protocol mismatch')
        if not 0 <= self.start < self.grid_size ** 2 or not 0 <= self.goal < self.grid_size ** 2 or self.start == self.goal:
            raise ValueError('invalid endpoints')
        if self.dist != distance(self.start, self.goal, self.grid_size) or not 1 <= self.dist <= min(2 * (self.grid_size - 1), self.budget):
            raise ValueError('initial target must be reachable within the registered budget')


class ParameterizedAreaEnv(ScaledGridEnv):
    def __init__(self, root):
        root = Path(root)
        manifest = json.loads((root / '数据清单.json').read_text('utf-8'))
        self.spec = validate_manifest(manifest)
        regions = manifest['regions']
        self._bindings = {(r['split'], r['area']): r['source_tile'] for r in regions}
        if not regions or len(self._bindings) != len(regions):
            raise ValueError('nonempty unique region bindings required')
        super().__init__(root)

    def reset(self, ep):
        if not isinstance(ep, AreaEpisode):
            raise ValueError('parameterized AreaEpisode required')
        ep.validate()
        if ep.protocol != self.spec.name or self._bindings.get((ep.split, ep.area)) != ep.source_tile:
            raise ValueError('episode/manifest binding mismatch')
        key, k = (ep.split, ep.area), self.spec.grid_size
        signature = self.signature(ep)
        if key not in self._validated:
            folder = self.root / 'patches' / ep.split / ep.area
            if {p.name for p in folder.glob('patch_*')} != {f'patch_{i}.jpg' for i in range(k * k)}:
                raise ValueError('exact full grid patch set required')
            for i in range(k * k):
                with Image.open(folder / f'patch_{i}.jpg') as im:
                    im.load()
                    if im.mode != 'RGB' or im.size != (300, 300):
                        raise ValueError('native 300x300 RGB patch required')
            self._validated[key] = signature
        elif self._validated[key] != signature:
            raise ValueError('immutable area changed')
        self._episode, self._position, self._remaining = ep, ep.start, ep.budget
        self._visited = [ep.start]
        self._trajectory = [dict(step=0, patch_id=ep.start, action=None, out_of_bounds=False, revisited=False)]
        self._success, self._done = False, False
        self._target = self.payload(ep.goal)
        return self.observation()

    def observation(self):
        return Observation(self.payload(self._position), self._target, divmod(self._position, self.spec.grid_size),
                           self.spec.grid_size, self._remaining, tuple(self._visited))

    def step(self, action):
        if self._episode is None or self._done:
            raise RuntimeError('reset a live episode')
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError('four actions only')
        k = self.spec.grid_size
        r, c = divmod(self._position, k)
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc
        outside = not (0 <= nr < k and 0 <= nc < k)
        if not outside:
            self._position = nr * k + nc
        revisited = self._position in self._visited
        self._remaining -= 1
        self._visited.append(self._position)
        count = self._episode.budget - self._remaining
        self._trajectory.append(dict(step=count, patch_id=self._position, action=action, out_of_bounds=outside, revisited=revisited))
        self._success = self._position == self._episode.goal
        self._done = self._success or self._remaining == 0
        return self.observation(), self._done, StepInfo(outside, revisited, count)

    def evaluator_result(self):
        if self._episode is None or not self._done:
            raise RuntimeError('terminal-only evaluator truth')
        steps = len(self._trajectory) - 1
        revisits = sum(t['revisited'] for t in self._trajectory)
        sg = distance(self._position, self._episode.goal, self.spec.grid_size)
        return dict(episode_id=self._episode.episode_id, success=self._success,
            termination='goal_reached' if self._success else 'budget_exhausted', sg=sg, steps=steps,
            revisits=revisits, repeat_visit_rate=revisits / steps,
            out_of_bounds=sum(t['out_of_bounds'] for t in self._trajectory), trajectory=[dict(t) for t in self._trajectory],
            protocol=self.spec.name, cell_size_m=300, sg_m=300 * sg,
            valid_travel_m=300 * sum(t['action'] is not None and not t['out_of_bounds'] for t in self._trajectory),
            map_projected_area_km2=self.spec.area_km2)
