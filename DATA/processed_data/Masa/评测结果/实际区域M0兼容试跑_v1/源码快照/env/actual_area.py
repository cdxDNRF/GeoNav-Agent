"""Explicit ground-footprint protocol, retaining the existing public observation."""
from dataclasses import dataclass
from pathlib import Path
import json
import re
from env.scaled_grid import ScaledEpisode, ScaledGridEnv, distance

PROTOCOL = 'masa-local-grid10-area-v1'


@dataclass(frozen=True)
class AreaEpisode(ScaledEpisode):
    protocol: str = PROTOCOL

    def validate(self):
        if self.protocol != PROTOCOL or self.split != 'dev' or re.fullmatch(r'img_\d+', self.area) is None:
            raise ValueError('explicit actual-area development identity required')
        if not isinstance(self.episode_id, str) or not self.episode_id or not isinstance(self.source_tile, str) or not self.source_tile:
            raise ValueError('episode/source identity required')
        if any(type(getattr(self, n)) is not int for n in ('start', 'goal', 'dist', 'budget', 'grid_size')):
            raise ValueError('integer protocol fields required')
        if self.grid_size != 10 or self.budget != 20 or not 0 <= self.start < 100 or not 0 <= self.goal < 100 or self.start == self.goal:
            raise ValueError('fixed grid10/B20 endpoints required')
        if self.dist != distance(self.start, self.goal, 10) or not 0 < self.dist <= 18:
            raise ValueError('invalid evaluation-only distance')


class ActualAreaGridEnv(ScaledGridEnv):
    def __init__(self, root):
        root = Path(root)
        manifest = json.loads((root / '数据清单.json').read_text(encoding='utf-8'))
        if (manifest.get('protocol'), manifest.get('pixel_size_m'), manifest.get('cell_size_m'),
            manifest.get('native_cell_pixels'), manifest.get('mosaic_pixels'), manifest.get('grid_size')) != (PROTOCOL, 1, 300, 300, 3000, 10):
            raise ValueError('actual-area input manifest required; density inputs forbidden')
        self._area_sources = {r['area']: r['source_tile'] for r in manifest['regions']}
        if len(self._area_sources) != len(manifest['regions']):
            raise ValueError('duplicate area identity')
        super().__init__(root)

    def reset(self, ep):
        if not isinstance(ep, AreaEpisode) or self._area_sources.get(ep.area) != ep.source_tile:
            raise ValueError('actual-area episode and manifest source binding required')
        return super().reset(ep)

    def evaluator_result(self):
        result = super().evaluator_result()
        return dict(result, protocol=PROTOCOL, cell_size_m=300, sg_m=300*result['sg'],
                    valid_travel_m=300*sum(t['action'] is not None and not t['out_of_bounds'] for t in result['trajectory']),
                    map_projected_area_km2=9)
