"""Explicit formal native protocol referencing immutable original PNGs."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re

from PIL import Image
from env.area_protocol import AreaProtocol
from env.environment import image_payload
from env.parameterized_area import AreaEpisode, ParameterizedAreaEnv
from env.scaled_grid import ScaledGridEnv, distance


def confirmation_spec(k):
    if type(k) is not int or k not in (10, 15):
        raise ValueError('Explicit confirmation grid10/15 required')
    return AreaProtocol(f'massgis2005-native-grid{k}-confirm-v1', k, 20, 300, 600, 26986, .5)


@dataclass(frozen=True)
class ConfirmEpisode(AreaEpisode):
    def validate(self):
        spec = confirmation_spec(self.grid_size)
        if self.protocol != spec.name or self.split != 'test' or not isinstance(self.area, str) or re.fullmatch(r'img_\d+', self.area) is None:
            raise ValueError('Formal test identity required')
        if any(type(getattr(self, n)) is not int for n in ('start', 'goal', 'dist', 'budget', 'grid_size')):
            raise ValueError('Integer fields required')
        if self.budget != 20 or not 0 <= self.start < self.grid_size**2 or not 0 <= self.goal < self.grid_size**2 or self.start == self.goal:
            raise ValueError('Budget/endpoints differ')
        if self.dist != distance(self.start, self.goal, self.grid_size) or not 1 <= self.dist <= 20:
            raise ValueError('Only initially reachable formal tasks')
        if any(not isinstance(v, str) or not v for v in (self.episode_id, self.source_tile)):
            raise ValueError('Nonempty evaluator identity required')


class ConfirmAreaEnv(ParameterizedAreaEnv):
    def __init__(self, root, repository_root):
        root, repo = Path(root), Path(repository_root).resolve()
        manifest = json.loads((root/'数据清单.json').read_text('utf-8'))
        self.spec = confirmation_spec(manifest['grid_size'])
        expected = dict(protocol=self.spec.name, grid_size=self.spec.grid_size, budget=20,
            cell_size_m=300, native_cell_pixels=600, epsg=26986, pixel_size_m=.5,
            mosaic_pixels=self.spec.grid_size*600, patch_format='png', confirmation_only=True)
        if any(type(manifest.get(k)) is not type(v) or manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('Explicit formal manifest required')
        self._bindings, self._paths, self._sha = {}, {}, {}
        for rec in manifest['regions']:
            key = (rec['split'], rec['area'])
            if key in self._bindings or key[0] != 'test' or re.fullmatch(r'img_\d+', key[1]) is None:
                raise ValueError('Unique formal region identities required')
            cells = rec['cells']
            if [c['cell'] for c in cells] != list(range(self.spec.grid_size**2)):
                raise ValueError('Exact ordered native cell mapping required')
            paths = []
            for c in cells:
                relative = Path(c['path'])
                path = (repo/relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(repo) or path.suffix != '.png':
                    raise ValueError('Repository-local immutable PNG references required')
                paths.append(path)
            if len(set(paths)) != len(paths):
                raise ValueError('Duplicate referenced cells')
            self._bindings[key] = rec['source_tile']
            self._paths[key] = tuple(paths)
            self._sha[key] = tuple(c['file_sha256'] for c in cells)
        if not self._bindings:
            raise ValueError('Nonempty formal bindings required')
        ScaledGridEnv.__init__(self, root)

    def signature(self, ep):
        return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                     for p in self._paths[ep.split, ep.area])

    def reset(self, ep):
        if not isinstance(ep, ConfirmEpisode):
            raise ValueError('ConfirmEpisode required')
        ep.validate()
        key = (ep.split, ep.area)
        if ep.protocol != self.spec.name or self._bindings.get(key) != ep.source_tile:
            raise ValueError('Protocol/region/source differs')
        sig = self.signature(ep)
        if key not in self._validated:
            for p, expected in zip(self._paths[key], self._sha[key]):
                if sha256(p.read_bytes()).hexdigest() != expected:
                    raise ValueError('Original PNG bytes changed')
                with Image.open(p) as im:
                    im.load()
                    if im.mode != 'RGB' or im.size != (600, 600):
                        raise ValueError('Native600 RGB required')
            self._validated[key] = sig
        elif self._validated[key] != sig:
            raise ValueError('Immutable region changed')
        self._episode, self._position, self._remaining = ep, ep.start, ep.budget
        self._visited = [ep.start]
        self._trajectory = [dict(step=0, patch_id=ep.start, action=None, out_of_bounds=False, revisited=False)]
        self._done = self._success = False
        self._target = self.payload(ep.goal)
        return self.observation()

    def path(self, cell):
        return self._paths[self._episode.split, self._episode.area][cell]

    def payload(self, cell):
        p = self.path(cell)
        st = p.stat()
        if (str(p), st.st_size, st.st_mtime_ns) not in self._validated[self._episode.split, self._episode.area]:
            raise ValueError('Referenced PNG changed')
        if p not in self._images:
            self._images[p] = image_payload(p)
        return self._images[p]
