"""Explicit physical search-area protocols, separate from sealed legacy code."""
from dataclasses import dataclass

GRID10 = 'masa-roads-grid10-spatial-v1'
GRID15 = 'masa-roads-grid15-area-v1'


@dataclass(frozen=True)
class AreaProtocol:
    name: str
    grid_size: int
    budget: int = 20
    cell_size_m: int = 300
    native_cell_pixels: int = 300
    epsg: int = 26986
    pixel_size_m: int = 1

    @property
    def mosaic_pixels(self):
        return self.grid_size * self.native_cell_pixels

    @property
    def area_km2(self):
        return (self.grid_size * self.cell_size_m / 1000) ** 2


PROTOCOLS = {GRID10: AreaProtocol(GRID10, 10), GRID15: AreaProtocol(GRID15, 15)}


def get_protocol(name):
    if name not in PROTOCOLS:
        raise ValueError('unregistered physical area protocol')
    return PROTOCOLS[name]


def validate_manifest(manifest):
    spec = get_protocol(manifest.get('protocol'))
    expected = dict(grid_size=spec.grid_size, budget=spec.budget, cell_size_m=spec.cell_size_m,
        native_cell_pixels=spec.native_cell_pixels, epsg=spec.epsg, pixel_size_m=spec.pixel_size_m,
        mosaic_pixels=spec.mosaic_pixels)
    if any(type(manifest.get(k)) is not int or manifest[k] != v for k, v in expected.items()):
        raise ValueError('manifest differs from registered physical protocol')
    return spec
