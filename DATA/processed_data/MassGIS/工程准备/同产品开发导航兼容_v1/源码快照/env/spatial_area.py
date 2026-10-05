"""Spatially isolated larger-area protocol; retain the existing public inputs."""
from dataclasses import dataclass
from pathlib import Path
import json
import re
from env.scaled_grid import ScaledEpisode, ScaledGridEnv, distance

PROTOCOL = 'masa-roads-grid10-spatial-v1'


@dataclass(frozen=True)
class SpatialAreaEpisode(ScaledEpisode):
    protocol: str = PROTOCOL

    def validate(self):
        if self.protocol != PROTOCOL or self.split not in ('dev', 'test') or re.fullmatch(r'img_\d+', self.area) is None:
            raise ValueError('spatial-area identity required')
        if not self.episode_id or not isinstance(self.episode_id, str) or not self.source_tile or not isinstance(self.source_tile, str):
            raise ValueError('episode/source identity required')
        if any(type(getattr(self, n)) is not int for n in ('start', 'goal', 'dist', 'budget', 'grid_size')):
            raise ValueError('integer protocol fields required')
        if self.budget != 20 or self.grid_size != 10 or not 0 <= self.start < 100 or not 0 <= self.goal < 100 or self.start == self.goal:
            raise ValueError('grid10/B20 endpoints required')
        if self.dist != distance(self.start, self.goal, 10) or not 1 <= self.dist <= 18:
            raise ValueError('evaluation distance mismatch')


class SpatialAreaGridEnv(ScaledGridEnv):
    def __init__(self, root):
        root = Path(root)
        manifest = json.loads((root / '数据清单.json').read_text(encoding='utf-8'))
        if (manifest.get('protocol'), manifest.get('epsg'), manifest.get('pixel_size_m'), manifest.get('cell_size_m'),
                manifest.get('native_cell_pixels'), manifest.get('mosaic_pixels'), manifest.get('grid_size')) != (PROTOCOL, 26986, 1, 300, 300, 3000, 10):
            raise ValueError('spatial-area manifest required')
        self._bindings = {(r['split'], r['area']): r['source_tile'] for r in manifest['regions']}
        if len(self._bindings) != len(manifest['regions']):
            raise ValueError('duplicate split/area')
        super().__init__(root)

    def reset(self, ep):
        if not isinstance(ep, SpatialAreaEpisode) or self._bindings.get((ep.split, ep.area)) != ep.source_tile:
            raise ValueError('spatial Episode/manifest source binding required')
        return super().reset(ep)

    def evaluator_result(self):
        r = super().evaluator_result()
        return dict(r, protocol=PROTOCOL, cell_size_m=300, sg_m=300*r['sg'],
                    valid_travel_m=300*sum(t['action'] is not None and not t['out_of_bounds'] for t in r['trajectory']),
                    map_projected_area_km2=9)
