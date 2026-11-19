"""Bounded read-only adapter and cluster partitioner for OpenCVL in_the_wild_test.

Verifies PNG headers without decoding pixels, compares against declared
historical/protected footprints, clusters SE pairs geographically, and freezes
evaluator vs policy-safe manifests without training or confirmation consumption.
"""
from collections import Counter
import math
from pathlib import Path

from .capturepoint_corrected_v1 import corrected_pose, distance, geometry, latlon, number
from .capturepoint_intake_v1 import image_header, safe_relative

CENTER9_INDICES = frozenset({6, 7, 8, 11, 12, 13, 16, 17, 18})
SE_ROLES = frozenset({
    'blocked_known_overlap',
    'quarantined_cv045_vicinity',
    'development_fit_train',
    'development_validation',
    'isolated_confirmation_blind',
})
WILD_ROLES = SE_ROLES | frozenset({'pl_pending_visual_dedup'})


def parse_png_header_from_file(path: Path) -> dict:
    """Read only the leading PNG signature + IHDR chunk (<=64 B); zero pixel decode."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size <= 0:
        raise ValueError(f'Missing or empty image asset: {p}')
    with p.open('rb') as handle:
        head = handle.read(64)
    meta = image_header(head)
    if meta['format'] != 'PNG':
        raise ValueError(f'Expected PNG asset: {p}')
    meta['bytes'] = p.stat().st_size
    return meta


def validate_pair_paths(row: dict) -> tuple[str, str, str]:
    """Validate stable Mapillary ID and unique relative ground/aerial PNG paths."""
    sid = row.get('mapil_image_id')
    if not isinstance(sid, str) or not sid.isdigit():
        raise ValueError('Exact digit-string mapil_image_id required')
    g_fields = [row.get(k) for k in ('mapil_image', 'ground_image') if isinstance(row.get(k), str) and row.get(k)]
    if not g_fields or len(set(g_fields)) != 1:
        raise ValueError(f'Missing or ambiguous ground image for {sid}')
    ground = safe_relative(g_fields[0])
    aerial = safe_relative(row['aerial_image'])
    if ground.rsplit('/', 1)[-1] != f'{sid}.png' or aerial.rsplit('/', 1)[-1] != f'{sid}.png':
        raise ValueError(f'Pair filenames must match stable ID {sid}')
    return sid, ground, aerial


def cell_from_offsets(dx: float, dy: float, heading: float, resolution: float, width: int, height: int) -> dict:
    """Compute native 5x5 grid geometry from explicit numeric offsets (half-open bins)."""
    dx, dy, heading, resolution = number(dx), number(dy), number(heading), number(resolution)
    if resolution <= 0:
        raise ValueError('Positive metric resolution required')
    x = width / 2 + dx
    y = height / 2 - dy
    x_edges = [i * width // 5 for i in range(6)]
    y_edges = [i * height // 5 for i in range(6)]
    inside = 0 <= x < width and 0 <= y < height
    col = next((i for i in range(5) if x_edges[i] <= x < x_edges[i + 1]), None) if inside else None
    row_idx = next((i for i in range(5) if y_edges[i] <= y < y_edges[i + 1]), None) if inside else None
    radius_m = math.hypot(width * resolution / 2, height * resolution / 2)
    return dict(
        x=x, y=y, heading=heading, inside=inside, row=row_idx, col=col,
        index=(row_idx * 5 + col) if inside else None,
        footprint_radius_m=radius_m,
        footprint_m=[width * resolution, height * resolution],
    )


def dual_pose_summary(row: dict, width: int, height: int, *, require_latlon: bool) -> dict:
    """Parse both mast3rpnp and mapilopensfm offsets identically across SE and PL."""
    sid, ground, aerial = validate_pair_paths(row)
    res = number(row['resolution'])
    if require_latlon:
        # Cross-verify primary mast3rpnp fields via the frozen CV-045 contract.
        cp = corrected_pose(row)
        geo = geometry(row, width, height)
        center = cp['aerial_latlon']
        mast = dict(
            x=geo['x'], y=geo['y'], heading=cp['heading'], inside=geo['inside'],
            row=geo['row'], col=geo['col'], index=geo['index'],
            footprint_radius_m=geo['reference_radius_m'], footprint_m=geo['footprint_m'],
        )
    else:
        if row.get('aerial_latlon') is not None:
            raise ValueError(f'Unexpected aerial_latlon in unreferenced split for {sid}')
        center = None
        mast = cell_from_offsets(
            row['dx_mast3rpnp'], row['dy_mast3rpnp'], row['heading_mast3rpnp'], res, width, height
        )
    osfm = cell_from_offsets(
        row['dx_mapilopensfm'], row['dy_mapilopensfm'], row['heading_mapilopensfm'], res, width, height
    )
    return dict(
        source_key=f'mapillary:{sid}', image_id=sid, country=row.get('country'),
        ground_rel=ground, aerial_rel=aerial, resolution=res,
        max_offset_m=number(row['max_offset_m']) if row.get('max_offset_m') is not None else None,
        aerial_latlon=center, creator_id=row.get('mapillary_creator_id'),
        mast3rpnp=mast, mapilopensfm=osfm,
        opensfm_cell_agree=(mast['inside'] and osfm['inside'] and mast['index'] == osfm['index']),
        in_center9=(mast['index'] in CENTER9_INDICES if mast['inside'] else False),
    )


def load_declared_known_sources(binding: dict) -> list[dict]:
    """Combine historical ledger + CV-045 reserved holdouts into a unified guard list."""
    out = []
    for item in binding['known_sources']:
        out.append(dict(
            source_key=item['source_key'],
            center_latlon=latlon(item['center_latlon']),
            radius_m=number(item['radius_m']),
            is_cv045=False,
        ))
    for cand in binding['candidates']:
        sid = cand['id']
        res = number(cand['resolution'])
        out.append(dict(
            source_key=f'mapillary:{sid}',
            center_latlon=latlon(cand['aerial_latlon']),
            radius_m=math.hypot(625 * res / 2, 625 * res / 2),
            is_cv045=True,
        ))
    return out


def connected_components_by_gap(items: list[dict], threshold_m: float, *, subtract_radii: bool) -> list[list[dict]]:
    """Deterministic Union-Find spatial clustering ordered by cluster size desc, then ID."""
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        c_i, r_i = items[i]['aerial_latlon'], items[i]['radius_m']
        for j in range(i):
            c_j, r_j = items[j]['aerial_latlon'], items[j]['radius_m']
            d = distance(c_i, c_j) - (r_i + r_j if subtract_radii else 0.0)
            if d < threshold_m:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    buckets: dict[int, list[dict]] = {}
    for i, item in enumerate(items):
        buckets.setdefault(find(i), []).append(item)
    return sorted(
        (sorted(grp, key=lambda x: x['image_id']) for grp in buckets.values()),
        key=lambda grp: (-len(grp), grp[0]['image_id']),
    )


def partition_se_clean_macros(
    clean_items: list[dict],
    known_sources: list[dict],
    *,
    macro_link_m: float = 15_000.0,
    cv045_vicinity_m: float = 15_000.0,
    known_city_m: float = 15_000.0,
    dev_neighbor_m: float = 25_000.0,
) -> tuple[dict[str, str], list[dict]]:
    """Assign clean (>=500m) SE pairs into geographically separated roles via macro regions."""
    macros = connected_components_by_gap(clean_items, macro_link_m, subtract_radii=False)
    cv045_only = [k for k in known_sources if k['is_cv045']]
    meta = []
    for idx, items in enumerate(macros, 1):
        subclusters = connected_components_by_gap(items, 500.0, subtract_radii=True)
        min_cv045_m = min(distance(it['aerial_latlon'], k['center_latlon']) for it in items for k in cv045_only)
        min_known_gap_m = min(it['min_known_gap_m'] for it in items)
        meta.append(dict(
            macro_id=f'M{idx:02d}', items=items, count=len(items), subcluster_count=len(subclusters),
            min_cv045_center_m=min_cv045_m, min_known_gap_m=min_known_gap_m,
            centroid_latlon=[
                sum(it['aerial_latlon'][0] for it in items) / len(items),
                sum(it['aerial_latlon'][1] for it in items) / len(items),
            ],
        ))

    unseen_non_cv045 = [m for m in meta if m['min_cv045_center_m'] >= cv045_vicinity_m and m['min_known_gap_m'] >= known_city_m]
    largest_unseen = unseen_non_cv045[0] if unseen_non_cv045 else None
    second_unseen = unseen_non_cv045[1] if len(unseen_non_cv045) > 1 else None

    role_by_source: dict[str, str] = {}
    macro_summaries = []
    for m in meta:
        if m['min_cv045_center_m'] < cv045_vicinity_m:
            role = 'quarantined_cv045_vicinity'
        elif m['min_known_gap_m'] < known_city_m:
            role = 'development_fit_train'
        elif largest_unseen is not None and (
            m['macro_id'] == largest_unseen['macro_id']
            or any(distance(a['aerial_latlon'], b['aerial_latlon']) < dev_neighbor_m
                   for a in m['items'] for b in largest_unseen['items'])
        ):
            role = 'development_fit_train'
        elif second_unseen is not None and m['macro_id'] == second_unseen['macro_id']:
            role = 'development_validation'
        else:
            role = 'isolated_confirmation_blind'
        for it in m['items']:
            role_by_source[it['source_key']] = role
            it['macro_id'] = m['macro_id']
        macro_summaries.append({
            k: v for k, v in m.items() if k != 'items'
        } | {'role': role, 'image_ids': [it['image_id'] for it in m['items']]})

    _assert_cross_split_clearance(clean_items, role_by_source)
    return role_by_source, macro_summaries


def _assert_cross_split_clearance(clean_items: list[dict], role_by_source: dict[str, str]) -> None:
    by_role: dict[str, list[dict]] = {}
    for it in clean_items:
        by_role.setdefault(role_by_source[it['source_key']], []).append(it)

    blind = by_role.get('isolated_confirmation_blind', [])
    if not blind:
        raise ValueError('Isolated confirmation split must be non-empty')
    if any(it['min_known_gap_m'] < 50_000.0 for it in blind):
        raise ValueError('Blind confirmation pair violated 50km known-ledger clearance')
    for other_role in ('development_fit_train', 'development_validation', 'quarantined_cv045_vicinity'):
        for a in blind:
            for b in by_role.get(other_role, []):
                if distance(a['aerial_latlon'], b['aerial_latlon']) < 50_000.0:
                    raise ValueError(f'Blind confirmation came within 50km of {other_role}')
    for a in by_role.get('development_validation', []):
        for b in by_role.get('development_fit_train', []):
            if distance(a['aerial_latlon'], b['aerial_latlon']) < 50_000.0:
                raise ValueError('Validation split came within 50km of fit_train split')


def summarize_role_stats(rows: list[dict]) -> dict:
    """Compute non-leaking summary counts and baseline majority-cell (cell 12) prior."""
    n = len(rows)
    if n == 0:
        return dict(count=0)
    c12 = sum(1 for r in rows if r['mast3rpnp']['index'] == 12)
    in9 = sum(1 for r in rows if r['in_center9'])
    agree = sum(1 for r in rows if r['opensfm_cell_agree'])
    creators = len({r['creator_id'] for r in rows if r['creator_id'] is not None})
    hist = dict(sorted(Counter(r['mast3rpnp']['index'] for r in rows).items()))
    return dict(
        count=n, unique_creators=creators, in_center9_count=in9,
        always_cell12_count=c12, always_cell12_rate=round(c12 / n, 6),
        opensfm_agree_count=agree, opensfm_agree_rate=round(agree / n, 6),
        mast3rpnp_cell_histogram=hist,
    )
