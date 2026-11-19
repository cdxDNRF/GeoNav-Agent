"""Alternate independent verification for CV-049 OpenCVL wild freeze artifacts.

Uses independent unit-sphere chord distances, bisect_right bin mapping,
raw PNG header CRC checks, and leakage scans with zero pixel decoding or models.
"""
import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[3]
RAW_BASE = ROOT / 'DATA/raw_data/OpenCVL/OpenCVL/mapillary/in_the_wild_test'
OUT_DIR = ROOT / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1'
CV045_BINDING = ROOT / 'DATA/processed_data/OpenCVL/公开瑞典三候选校正标签预检_v1/首获取冻结绑定.json'
FORBIDDEN_PUBLIC_KEYS = frozenset({
    'dx_mast3rpnp', 'dy_mast3rpnp', 'heading_mast3rpnp',
    'dx_mapilopensfm', 'dy_mapilopensfm', 'heading_mapilopensfm',
    'mast3rpnp', 'mapilopensfm', 'opensfm_cell_agree', 'aerial_latlon',
})
EXPECTED_COUNTS = {
    'blocked_known_overlap': 351,
    'quarantined_cv045_vicinity': 12,
    'development_fit_train': 435,
    'development_validation': 116,
    'isolated_confirmation_blind': 28,
    'pl_pending_visual_dedup': 419,
}


def sha(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def unit_vec(latlon_pt) -> tuple[float, float, float]:
    la, lo = map(math.radians, latlon_pt)
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))


def chord_distance_m(a, b) -> float:
    c = math.dist(unit_vec(a), unit_vec(b))
    return 2.0 * 6371008.8 * math.asin(min(1.0, c / 2.0))


def check_png_ihdr(path: Path, exp_w: int, exp_h: int) -> None:
    with path.open('rb') as f:
        head = f.read(33)
    if len(head) < 33 or head[:8] != b'\x89PNG\r\n\x1a\n' or head[8:16] != b'\x00\x00\x00\rIHDR':
        raise ValueError(f'Invalid PNG signature/IHDR: {path}')
    if (zlib.crc32(head[12:29]) & 0xffffffff) != struct.unpack('>I', head[29:33])[0]:
        raise ValueError(f'IHDR CRC mismatch: {path}')
    w, h = struct.unpack('>II', head[16:24])
    if (w, h) != (exp_w, exp_h):
        raise ValueError(f'PNG dimension mismatch in {path}: {(w, h)} vs {(exp_w, exp_h)}')


def alt_cell(dx: float, dy: float, w: int, h: int) -> int | None:
    x, y = w / 2 + dx, h / 2 - dy
    if not (0 <= x < w and 0 <= y < h):
        return None
    xe = [i * w // 5 for i in range(6)]
    ye = [i * h // 5 for i in range(6)]
    return (bisect_right(ye, y) - 1) * 5 + (bisect_right(xe, x) - 1)


def audit() -> dict:
    if 'torch' in sys.modules:
        raise ValueError('Model framework must not be imported during audit')
    binding = read_json(OUT_DIR / '核验/源码与输入绑定_v1.json')
    for rel, expected in binding['protected_upstream_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Protected upstream file changed: {rel}')
    for entry in binding['generator_sources'] + binding['generated_artifacts']:
        p = ROOT / entry['path']
        if sha(p) != entry['sha256'] or p.stat().st_size != entry['bytes']:
            raise ValueError(f'Bound artifact digest mismatch: {entry["path"]}')

    proto = read_json(OUT_DIR / '协议冻结_v1.json')
    pub = read_json(OUT_DIR / '公开洁净划分清单_v1.json')
    ledger = read_json(OUT_DIR / '核验/评估器专用冻结台账_禁止入模.json')
    if proto['execution_state']['pixels_decoded'] != 0 or ledger['policy_input'] is not False:
        raise ValueError('Execution or leakage boundary violated')

    # Verify zero forbidden ground-truth fields in public manifest
    pub_ids_seen = set()
    for role, rows in pub['splits'].items():
        if len(rows) != EXPECTED_COUNTS[role]:
            raise ValueError(f'Unexpected count for {role}: {len(rows)}')
        for item in rows:
            if set(item) & FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f'Forbidden label key leaked in public manifest for {item["image_id"]}')
            key = (item['split'], item['image_id'])
            if key in pub_ids_seen:
                raise ValueError(f'Duplicate pair in public manifest: {key}')
            pub_ids_seen.add(key)

    cv045 = read_json(CV045_BINDING)
    known = [
        (k['source_key'], k['center_latlon'], float(k['radius_m']))
        for k in cv045['known_sources']
    ] + [
        ('mapillary:' + c['id'], c['aerial_latlon'], math.hypot(625 * c['resolution'] / 2, 625 * c['resolution'] / 2))
        for c in cv045['candidates']
    ]
    if len(known) != 62:
        raise ValueError(f'Expected 62 known/protected sources, got {len(known)}')

    se_raw = {r['mapil_image_id']: r for r in read_json(RAW_BASE / 'SE/labels.json')['labels']}
    pl_raw = {r['mapil_image_id']: r for r in read_json(RAW_BASE / 'PL/labels.json')['labels']}
    if len(se_raw) != 942 or len(pl_raw) != 419:
        raise ValueError('Raw label row counts changed')

    by_role = {r: [] for r in EXPECTED_COUNTS}
    max_gap_diff_m = 0.0
    png_headers = 0
    for rec in ledger['se_records'] + ledger['pl_records']:
        raw_map = se_raw if rec['split'] == 'SE' else pl_raw
        raw = raw_map[rec['image_id']]
        g_asset, a_asset = rec['ground_asset'], rec['aerial_asset']
        check_png_ihdr(RAW_BASE / g_asset['rel'], g_asset['width'], g_asset['height'])
        check_png_ihdr(RAW_BASE / a_asset['rel'], a_asset['width'], a_asset['height'])
        png_headers += 2

        m_idx = alt_cell(raw['dx_mast3rpnp'], raw['dy_mast3rpnp'], a_asset['width'], a_asset['height'])
        o_idx = alt_cell(raw['dx_mapilopensfm'], raw['dy_mapilopensfm'], a_asset['width'], a_asset['height'])
        if m_idx != rec['mast3rpnp']['index'] or o_idx != rec['mapilopensfm']['index']:
            raise ValueError(f'Alternate cell mismatch for {rec["image_id"]}')
        if rec['opensfm_cell_agree'] != (m_idx is not None and m_idx == o_idx):
            raise ValueError(f'Pose agreement flag mismatch for {rec["image_id"]}')

        if rec['split'] == 'SE':
            min_gap = min(chord_distance_m(rec['aerial_latlon'], kc) - rec['radius_m'] - kr for _, kc, kr in known)
            max_gap_diff_m = max(max_gap_diff_m, abs(min_gap - rec['min_known_gap_m']))
            is_exact = any(sk == rec['source_key'] for sk, _, _ in known)
            should_block = is_exact or min_gap < 500.0
            if (rec['role'] == 'blocked_known_overlap') != should_block:
                raise ValueError(f'Overlap classification mismatch for {rec["image_id"]}')
        by_role[rec['role']].append(rec)

    # Verify CV-045 3 candidates remain blocked
    blocked_ids = {r['image_id'] for r in by_role['blocked_known_overlap']}
    if not set(cv045['selected_ids']).issubset(blocked_ids):
        raise ValueError('CV-045 protected IDs must stay blocked')

    # Verify cross-role clearance independently via chord distance
    blind = by_role['isolated_confirmation_blind']
    min_blind_known_km = min(r['min_known_gap_m'] for r in blind) / 1000.0
    min_blind_fit_km = min(chord_distance_m(a['aerial_latlon'], b['aerial_latlon']) for a in blind for b in by_role['development_fit_train']) / 1000.0
    min_blind_val_km = min(chord_distance_m(a['aerial_latlon'], b['aerial_latlon']) for a in blind for b in by_role['development_validation']) / 1000.0
    min_val_fit_km = min(chord_distance_m(a['aerial_latlon'], b['aerial_latlon']) for a in by_role['development_validation'] for b in by_role['development_fit_train']) / 1000.0

    if min_blind_known_km < 60.0 or min_blind_fit_km < 80.0 or min_blind_val_km < 100.0 or min_val_fit_km < 90.0:
        raise ValueError('Alternate spatial clearance gate failed')
    if max_gap_diff_m > 1e-6:
        raise ValueError(f'Haversine vs chord gap exceeded tolerance: {max_gap_diff_m}')

    return dict(
        schema='cv049-alternate-audit/v1',
        passed=True,
        total_pairs_verified=len(se_raw) + len(pl_raw),
        png_headers_crc_checked=png_headers,
        role_counts={k: len(v) for k, v in by_role.items()},
        max_haversine_vs_chord_diff_m=max_gap_diff_m,
        alternate_clearance_km=dict(
            blind_vs_known_gap_min=round(min_blind_known_km, 3),
            blind_vs_fit_train_min=round(min_blind_fit_km, 3),
            blind_vs_dev_val_min=round(min_blind_val_km, 3),
            dev_val_vs_fit_train_min=round(min_val_fit_km, 3),
        ),
        auditor_image_decodes=0,
        model_calls=0,
        training=0,
        formal_confirmation_consumed=0,
        independent_personnel_review=False,
        notes='same Antigravity session, independent chord/bisect/IHDR-CRC verification',
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true', help='Save alternate audit JSON report once')
    args = parser.parse_args(argv)
    result = audit()
    if args.save:
        target = OUT_DIR / '核验/另实现复核_v1.json'
        with target.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
