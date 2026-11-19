"""CV-050 OpenCVL development discrepancy QA and Stage-1 single-step baseline evaluator.

Strictly enforces split isolation:
- Image decoding/QA is bounded to <= 24 deterministic pairs from development_fit_train.
- Statistical baselines are fitted solely on development_fit_train (435 pairs).
- Discrepancy geometry and Stage-1 baseline evaluations are blocked on
  isolated_confirmation_blind, blocked_known_overlap, quarantined_cv045_vicinity,
  and pl_pending_visual_dedup.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from .opencvl_wild_intake_v1 import CENTER9_INDICES, parse_png_header_from_file

QA_MAX_PAIRS = 24
ALLOWED_SE_GROUND_SIZES = frozenset({(952, 714), (952, 532)})
ALLOWED_SE_AERIAL_SIZES = frozenset({(625, 625)})
IMAGE_WIDTH_PX = 625
IMAGE_HEIGHT_PX = 625
DEFAULT_RESOLUTION_M = 0.16
GRID_EDGES_PX = (0.0, 125.0, 250.0, 375.0, 500.0, 625.0)
INTERNAL_EDGES_PX = (125.0, 250.0, 375.0, 500.0)
SOFT_TARGET_SIGMA_M = 5.0
INTERIOR_MARGIN_THRESHOLD_M = 2.0
LARGE_DISCREPANCY_THRESHOLD_M = 10.0

ALLOWED_ANALYSIS_ROLES = frozenset({'development_fit_train', 'development_validation'})
STRATUM_NAMES = (
    'all_pairs',
    'opensfm_cell_agree',
    'interior_margin_ge_2m',
    'large_discrepancy_gt_10m',
)

CELL_CENTERS_PX: tuple[tuple[float, float], ...] = tuple(
    ((k % 5) * 125.0 + 62.5, (k // 5) * 125.0 + 62.5) for k in range(25)
)
CELL_CENTER_DIST_TO_12_SQ: tuple[int, ...] = tuple(
    ((k % 5) - 2) ** 2 + ((k // 5) - 2) ** 2 for k in range(25)
)


def _entry_role(entry: dict) -> str:
    role = entry.get('role') or entry.get('split_role')
    if not isinstance(role, str) or not role:
        raise ValueError('Missing role/split_role in record')
    return role


def _assert_analysis_entry(entry: dict, expected_role: str | None = None) -> str:
    role = _entry_role(entry)
    if entry.get('blind_do_not_inspect') is True or role not in ALLOWED_ANALYSIS_ROLES:
        raise ValueError(f'Record role {role!r} is forbidden from CV-050 analysis/QA')
    if entry.get('policy_admitted') is False:
        raise ValueError(f'Record {entry.get("image_id")} has policy_admitted=False')
    if expected_role is not None and role != expected_role:
        raise ValueError(f'Expected role {expected_role!r}, got {role!r} for {entry.get("image_id")}')
    return role


def _quantiles_dict(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        'min': float(np.min(arr)),
        'p25': float(np.percentile(arr, 25.0)),
        'median': float(np.percentile(arr, 50.0)),
        'p75': float(np.percentile(arr, 75.0)),
        'p90': float(np.percentile(arr, 90.0)),
        'max': float(np.max(arr)),
    }


def _margin_stats(margins: list[float]) -> dict:
    n = len(margins)
    lt_1 = sum(1 for v in margins if v < 1.0)
    lt_2 = sum(1 for v in margins if v < 2.0)
    ge_2 = n - lt_2
    return {
        'count': n,
        'quantiles_m': _quantiles_dict(margins),
        'lt_1m_count': lt_1,
        'lt_1m_rate': lt_1 / n if n else 0.0,
        'lt_2m_count': lt_2,
        'lt_2m_rate': lt_2 / n if n else 0.0,
        'ge_2m_count': ge_2,
        'ge_2m_rate': ge_2 / n if n else 0.0,
    }


def compute_pair_discrepancy_geometry(entry: dict) -> dict:
    """Compute grid-boundary proximity and OpenSfM vs MASt3R-PnP cell/metric discrepancies."""
    role = _assert_analysis_entry(entry)
    mast = entry['mast3rpnp']
    osfm = entry['mapilopensfm']
    res = float(entry['resolution'])
    if res <= 0 or not math.isfinite(res):
        raise ValueError(f'Invalid metric resolution for {entry.get("image_id")}')

    x_m, y_m = float(mast['x']), float(mast['y'])
    x_o, y_o = float(osfm['x']), float(osfm['y'])
    row_m, col_m, idx_m = int(mast['row']), int(mast['col']), int(mast['index'])
    row_o, col_o, idx_o = int(osfm['row']), int(osfm['col']), int(osfm['index'])

    min_x_edge_px = min(abs(x_m - e) for e in INTERNAL_EDGES_PX)
    min_y_edge_px = min(abs(y_m - e) for e in INTERNAL_EDGES_PX)
    min_edge_px = min(min_x_edge_px, min_y_edge_px)
    edge_margin_m = min_edge_px * res

    dx_px = x_m - x_o
    dy_px = y_m - y_o
    planar_diff_px = math.hypot(dx_px, dy_px)
    planar_diff_m = planar_diff_px * res

    abs_drow = abs(row_m - row_o)
    abs_dcol = abs(col_m - col_o)
    cell_l1 = abs_drow + abs_dcol
    cell_chebyshev = max(abs_drow, abs_dcol)

    agree = bool(entry['opensfm_cell_agree'])
    if agree != (idx_m == idx_o):
        raise ValueError(f'Inconsistent opensfm_cell_agree for {entry.get("image_id")}')

    return {
        'image_id': entry['image_id'],
        'role': role,
        'macro_id': entry.get('macro_id'),
        'resolution': res,
        'mast3rpnp_xy': [x_m, y_m],
        'mast3rpnp_cell': idx_m,
        'opensfm_xy': [x_o, y_o],
        'opensfm_cell': idx_o,
        'min_internal_edge_dist_px': min_edge_px,
        'cell_edge_margin_m': edge_margin_m,
        'planar_diff_px': planar_diff_px,
        'planar_diff_m': planar_diff_m,
        'abs_drow': abs_drow,
        'abs_dcol': abs_dcol,
        'cell_l1': cell_l1,
        'cell_chebyshev': cell_chebyshev,
        'in_all_pairs': True,
        'in_opensfm_cell_agree': agree,
        'in_interior_margin_ge_2m': edge_margin_m >= INTERIOR_MARGIN_THRESHOLD_M,
        'in_large_discrepancy_gt_10m': planar_diff_m > LARGE_DISCREPANCY_THRESHOLD_M,
    }


def summarize_split_discrepancy(entries: list[dict], split_role: str) -> dict:
    """Summarize pose discrepancy topology and cell-boundary proximity across an allowed split."""
    if split_role not in ALLOWED_ANALYSIS_ROLES:
        raise ValueError(f'Split {split_role!r} not permitted for discrepancy analysis')
    if not entries:
        raise ValueError(f'Empty entry list for {split_role}')

    geos = []
    for entry in entries:
        _assert_analysis_entry(entry, expected_role=split_role)
        geos.append(compute_pair_discrepancy_geometry(entry))

    n = len(geos)
    ground_sizes = Counter(
        f'{e["ground_asset"]["width"]}x{e["ground_asset"]["height"]}'
        if isinstance(e.get('ground_asset'), dict) and 'width' in e['ground_asset'] and 'height' in e['ground_asset']
        else 'unknown'
        for e in entries
    )
    agree_geos = [g for g in geos if g['in_opensfm_cell_agree']]
    disagree_geos = [g for g in geos if not g['in_opensfm_cell_agree']]
    disagree_diffs = [g['planar_diff_m'] for g in disagree_geos]

    l1_hist = dict(sorted(Counter(g['cell_l1'] for g in disagree_geos).items()))
    cheb_hist = dict(sorted(Counter(g['cell_chebyshev'] for g in disagree_geos).items()))
    cheb_le1 = sum(1 for g in disagree_geos if g['cell_chebyshev'] <= 1)
    l1_eq1 = sum(1 for g in disagree_geos if g['cell_l1'] == 1)

    by_macro_buckets: dict[str, list[dict]] = {}
    for g in geos:
        by_macro_buckets.setdefault(str(g['macro_id']), []).append(g)

    by_macro = {}
    for mid, items in sorted(by_macro_buckets.items()):
        m_dis = [it for it in items if not it['in_opensfm_cell_agree']]
        by_macro[mid] = {
            'count': len(items),
            'agree_count': len(items) - len(m_dis),
            'disagree_count': len(m_dis),
            'disagree_rate': len(m_dis) / len(items),
            'chebyshev_eq_1_disagree_count': sum(1 for it in m_dis if it['cell_chebyshev'] == 1),
            'large_discrepancy_gt_10m_count': sum(1 for it in items if it['in_large_discrepancy_gt_10m']),
        }

    return {
        'split_role': split_role,
        'total_pairs': n,
        'ground_size_histogram': dict(sorted(ground_sizes.items())),
        'agree_count': len(agree_geos),
        'agree_rate': len(agree_geos) / n,
        'disagree_count': len(disagree_geos),
        'disagree_rate': len(disagree_geos) / n,
        'stratum_counts': {
            'all_pairs': n,
            'opensfm_cell_agree': len(agree_geos),
            'interior_margin_ge_2m': sum(1 for g in geos if g['in_interior_margin_ge_2m']),
            'large_discrepancy_gt_10m': sum(1 for g in geos if g['in_large_discrepancy_gt_10m']),
        },
        'disagree_topology': {
            'count': len(disagree_geos),
            'cell_l1_histogram': {str(k): v for k, v in l1_hist.items()},
            'cell_chebyshev_histogram': {str(k): v for k, v in cheb_hist.items()},
            'chebyshev_le_1_count': cheb_le1,
            'chebyshev_le_1_rate': cheb_le1 / len(disagree_geos) if disagree_geos else 0.0,
            'l1_eq_1_count': l1_eq1,
            'l1_eq_1_rate': l1_eq1 / len(disagree_geos) if disagree_geos else 0.0,
            'planar_diff_m_quantiles': _quantiles_dict(disagree_diffs),
            'planar_diff_le_5m_count': sum(1 for d in disagree_diffs if d <= 5.0),
            'planar_diff_gt_5m_count': sum(1 for d in disagree_diffs if d > 5.0),
            'planar_diff_gt_10m_count': sum(1 for d in disagree_diffs if d > 10.0),
            'planar_diff_gt_15m_count': sum(1 for d in disagree_diffs if d > 15.0),
        },
        'edge_margin_comparison_m': {
            'all_pairs': _margin_stats([g['cell_edge_margin_m'] for g in geos]),
            'agree_pairs': _margin_stats([g['cell_edge_margin_m'] for g in agree_geos]),
            'disagree_pairs': _margin_stats([g['cell_edge_margin_m'] for g in disagree_geos]),
        },
        'by_macro': by_macro,
    }


def _round_robin_pick(
    buckets: dict[str, list[tuple[dict, dict]]],
    macro_order: list[str],
    target_count: int,
    picked_ids: set[str],
    stratum_label: str,
) -> list[dict]:
    selected = []
    idx = 0
    active = [m for m in macro_order if buckets.get(m)]
    while len(selected) < target_count and active:
        macro = active[idx % len(active)]
        queue = buckets[macro]
        while queue and queue[0][0]['image_id'] in picked_ids:
            queue.pop(0)
        if not queue:
            active.remove(macro)
            continue
        entry, geo = queue.pop(0)
        picked_ids.add(entry['image_id'])
        selected.append({'entry': entry, 'geometry': geo, 'qa_stratum': stratum_label})
        idx += 1
    if len(selected) < target_count:
        raise ValueError(f'Insufficient candidates in {stratum_label}: need {target_count}, got {len(selected)}')
    return selected


def select_fit_train_qa_pairs(fit_train_entries: list[dict], max_pairs: int = QA_MAX_PAIRS) -> list[dict]:
    """Deterministically pick <= 24 stratified QA pairs strictly from development_fit_train."""
    if not 0 < max_pairs <= QA_MAX_PAIRS:
        raise ValueError(f'max_pairs must be in [1, {QA_MAX_PAIRS}], got {max_pairs}')
    if len(fit_train_entries) < max_pairs:
        raise ValueError(f'Need at least {max_pairs} fit_train entries, got {len(fit_train_entries)}')

    enriched = []
    for entry in fit_train_entries:
        _assert_analysis_entry(entry, expected_role='development_fit_train')
        enriched.append((entry, compute_pair_discrepancy_geometry(entry)))

    n_a = round(max_pairs * 10 / 24)
    n_b = round(max_pairs * 10 / 24)
    n_c = max_pairs - n_a - n_b

    picked_ids: set[str] = set()
    chosen: list[dict] = []

    # Group A: top largest metric discrepancies among opensfm != mast3rpnp
    disagree_sorted = sorted(
        [(e, g) for e, g in enriched if not g['in_opensfm_cell_agree']],
        key=lambda item: (-round(item[1]['planar_diff_m'], 6), item[0]['image_id']),
    )
    if len(disagree_sorted) < n_a:
        raise ValueError(f'Insufficient disagree pairs for Group A: need {n_a}, got {len(disagree_sorted)}')
    for entry, geo in disagree_sorted[:n_a]:
        picked_ids.add(entry['image_id'])
        chosen.append({'entry': entry, 'geometry': geo, 'qa_stratum': 'large_discrepancy_disagree'})

    # Group B: boundary-straddling disagreements (<=5m) round-robin by macro closest to 20m cell edges
    preferred_b_order = ['M01', 'M03', 'M04', 'M05', 'M12']
    extra_b = sorted({str(e.get('macro_id')) for e, _ in enriched} - set(preferred_b_order))
    b_order = preferred_b_order + extra_b
    b_buckets: dict[str, list[tuple[dict, dict]]] = {m: [] for m in b_order}
    for entry, geo in enriched:
        if (
            entry['image_id'] not in picked_ids
            and not geo['in_opensfm_cell_agree']
            and geo['planar_diff_m'] <= 5.0
        ):
            b_buckets[str(entry.get('macro_id'))].append((entry, geo))
    for m in b_buckets:
        b_buckets[m].sort(key=lambda item: (round(item[1]['cell_edge_margin_m'], 6), item[0]['image_id']))
    chosen.extend(_round_robin_pick(b_buckets, b_order, n_b, picked_ids, 'boundary_straddling_disagree'))

    # Group C: consensus interior controls across core metros
    preferred_c_order = ['M01', 'M03', 'M04', 'M05']
    extra_c = sorted({str(e.get('macro_id')) for e, _ in enriched} - set(preferred_c_order))
    c_order = preferred_c_order + extra_c
    c_buckets: dict[str, list[tuple[dict, dict]]] = {m: [] for m in c_order}
    for entry, geo in enriched:
        if (
            entry['image_id'] not in picked_ids
            and geo['in_opensfm_cell_agree']
            and geo['cell_edge_margin_m'] >= 3.0
        ):
            c_buckets[str(entry.get('macro_id'))].append((entry, geo))
    for m in c_buckets:
        c_buckets[m].sort(key=lambda item: item[0]['image_id'])
    chosen.extend(_round_robin_pick(c_buckets, c_order, n_c, picked_ids, 'consensus_control'))

    assert len(chosen) == max_pairs and len(picked_ids) == max_pairs
    out = []
    for rank, item in enumerate(chosen, 1):
        entry = item['entry']
        out.append({
            'qa_rank': rank,
            'qa_stratum': item['qa_stratum'],
            'image_id': entry['image_id'],
            'macro_id': entry.get('macro_id'),
            'subcluster_id': entry.get('subcluster_id'),
            'ground_rel': entry['ground_asset']['rel'],
            'aerial_rel': entry['aerial_asset']['rel'],
            'geometry': item['geometry'],
        })
    return out


def inspect_png_image_quality(path: Path) -> dict:
    """Parse PNG IHDR header and decode RGB pixels with dual-implementation moment cross-check."""
    p = Path(path)
    meta = parse_png_header_from_file(p)
    with p.open('rb') as handle:
        head33 = handle.read(33)
        file_sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
    bit_depth, color_type, _comp, _filt, interlace = struct.unpack('>BBBBB', head33[24:29])
    header = {
        'width': meta['width'],
        'height': meta['height'],
        'bit_depth': bit_depth,
        'color_type': color_type,
        'interlaced': bool(interlace),
        'bytes': meta['bytes'],
        'file_sha256': file_sha256,
    }

    with Image.open(p) as img:
        rgb = np.asarray(img.convert('RGB'), dtype=np.uint8)
    h, w, c = rgb.shape
    if (w, h, c) != (meta['width'], meta['height'], 3):
        raise ValueError(f'Decoded RGB shape mismatch in {p}: {(w, h, c)} vs {(meta["width"], meta["height"], 3)}')
    rgb_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()

    arr = rgb.astype(np.float64)
    ch_mean = arr.mean(axis=(0, 1)).tolist()
    ch_std = arr.std(axis=(0, 1), ddof=0).tolist()
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gray_std = float(np.std(gray, ddof=0))
    near_black = float(np.mean(np.max(rgb, axis=2) < 8))
    near_white = float(np.mean(np.min(rgb, axis=2) > 247))
    nonblank = bool(gray_std >= 2.0 and near_black < 0.95 and near_white < 0.95)

    # Independent raw integer first/second-moment verification without np.mean/np.std
    n_pix = float(w * h)
    flat = rgb.reshape(-1, 3).astype(np.int64)
    alt_mean = []
    alt_std = []
    for ch in range(3):
        s1 = int(flat[:, ch].sum())
        s2 = int((flat[:, ch] * flat[:, ch]).sum())
        mu = s1 / n_pix
        var = max(0.0, s2 / n_pix - mu * mu)
        alt_mean.append(mu)
        alt_std.append(math.sqrt(var))

    max_diff = max(
        max(abs(a - b) for a, b in zip(ch_mean, alt_mean)),
        max(abs(a - b) for a, b in zip(ch_std, alt_std)),
    )
    if max_diff > 1e-6:
        raise ValueError(f'Quality moment cross-check exceeded 1e-6 in {p}: {max_diff}')

    return {
        'header': header,
        'rgb_sha256': rgb_sha256,
        'quality': {
            'width': w,
            'height': h,
            'channel_mean': ch_mean,
            'channel_std': ch_std,
            'gray_std': gray_std,
            'near_black_fraction': near_black,
            'near_white_fraction': near_white,
            'nonblank': nonblank,
        },
        'alternate_mean': alt_mean,
        'alternate_std': alt_std,
        'max_moment_diff': max_diff,
    }


def gaussian_soft_target_distribution(
    x_px: float,
    y_px: float,
    resolution: float = DEFAULT_RESOLUTION_M,
    sigma_m: float = SOFT_TARGET_SIGMA_M,
) -> list[float]:
    """Build 25-cell Gaussian soft target distribution centered at ground-truth pixel location."""
    if resolution <= 0 or sigma_m <= 0:
        raise ValueError('Positive resolution and sigma_m required')
    inv_two_var = 0.5 / (sigma_m * sigma_m)
    weights = [
        math.exp(-(((cx - x_px) * resolution) ** 2 + ((cy - y_px) * resolution) ** 2) * inv_two_var)
        for cx, cy in CELL_CENTERS_PX
    ]
    total = sum(weights)
    if total <= 0:
        raise ValueError('Zero Gaussian kernel sum')
    return [w / total for w in weights]


def _validate_prediction(probs: list[float], ranking: list[int]) -> None:
    if len(probs) != 25 or any(p < 0 or not math.isfinite(p) for p in probs):
        raise ValueError('Predicted probabilities must be 25 finite non-negative floats')
    if abs(sum(probs) - 1.0) > 1e-5:
        raise ValueError(f'Predicted probabilities must sum to 1.0, got {sum(probs)}')
    if len(ranking) != 25 or set(ranking) != set(range(25)):
        raise ValueError('Predicted ranking must be a permutation of 0..24')


def evaluate_single_step_predictions(
    entries: list[dict],
    probs_by_id: dict[str, list[float]],
    rankings_by_id: dict[str, list[int]],
) -> dict[str, dict]:
    """Evaluate 25-cell probability + rank predictions across the four Stage-1 strata."""
    if not entries:
        raise ValueError('Cannot evaluate empty split')

    per_sample = []
    for entry in entries:
        _assert_analysis_entry(entry)
        sid = entry['image_id']
        probs = probs_by_id[sid]
        ranking = rankings_by_id[sid]
        _validate_prediction(probs, ranking)

        geo = compute_pair_discrepancy_geometry(entry)
        x_m, y_m = geo['mast3rpnp_xy']
        res = geo['resolution']
        true_cell = geo['mast3rpnp_cell']
        true_row, true_col = divmod(true_cell, 5)
        pred_top1 = ranking[0]
        pred_row, pred_col = divmod(pred_top1, 5)

        drow = abs(pred_row - true_row)
        dcol = abs(pred_col - true_col)
        dists_to_true_m = [
            math.hypot(cx - x_m, cy - y_m) * res for cx, cy in CELL_CENTERS_PX
        ]
        soft_q = gaussian_soft_target_distribution(x_m, y_m, resolution=res)
        eps_clip = 1e-12
        soft_ce = -sum(q * math.log(max(p, eps_clip)) for q, p in zip(soft_q, probs))
        soft_kl = sum(
            q * math.log(max(q, eps_clip) / max(p, eps_clip)) for q, p in zip(soft_q, probs) if q > 0.0
        )

        per_sample.append({
            'geometry': geo,
            'top1_hit': int(pred_top1 == true_cell),
            'top3_hit': int(true_cell in ranking[:3]),
            'top5_hit': int(true_cell in ranking[:5]),
            'chebyshev1_hit': int(max(drow, dcol) <= 1),
            'l1_1_hit': int((drow + dcol) <= 1),
            'true_cell_prob': float(probs[true_cell]),
            'top1_center_error_m': dists_to_true_m[pred_top1],
            'expected_planar_error_m': sum(p * d for p, d in zip(probs, dists_to_true_m)),
            'soft_target_ce_nats': soft_ce,
            'soft_target_kl_nats': soft_kl,
        })

    stratum_flag = {
        'all_pairs': 'in_all_pairs',
        'opensfm_cell_agree': 'in_opensfm_cell_agree',
        'interior_margin_ge_2m': 'in_interior_margin_ge_2m',
        'large_discrepancy_gt_10m': 'in_large_discrepancy_gt_10m',
    }
    out: dict[str, dict] = {}
    for name in STRATUM_NAMES:
        flag = stratum_flag[name]
        sub = [s for s in per_sample if s['geometry'][flag]]
        n = len(sub)
        if n == 0:
            out[name] = {'count': 0}
            continue
        t1 = sum(s['top1_hit'] for s in sub)
        t3 = sum(s['top3_hit'] for s in sub)
        t5 = sum(s['top5_hit'] for s in sub)
        c1 = sum(s['chebyshev1_hit'] for s in sub)
        l1 = sum(s['l1_1_hit'] for s in sub)
        out[name] = {
            'count': n,
            'top1_hits': t1,
            'top1_acc': t1 / n,
            'top3_hits': t3,
            'top3_acc': t3 / n,
            'top5_hits': t5,
            'top5_acc': t5 / n,
            'chebyshev1_hits': c1,
            'chebyshev1_acc': c1 / n,
            'l1_1_hits': l1,
            'l1_1_acc': l1 / n,
            'true_cell_prob_mean': sum(s['true_cell_prob'] for s in sub) / n,
            'top1_center_error_m': sum(s['top1_center_error_m'] for s in sub) / n,
            'expected_planar_error_m': sum(s['expected_planar_error_m'] for s in sub) / n,
            'soft_target_ce_nats': sum(s['soft_target_ce_nats'] for s in sub) / n,
            'soft_target_kl_nats': sum(s['soft_target_kl_nats'] for s in sub) / n,
        }
    return out


def build_statistical_baselines(
    fit_train_entries: list[dict],
    *,
    eps: float = 1e-3,
    dirichlet_alpha: float = 0.5,
) -> dict[str, dict]:
    """Fit zero-image statistical priors strictly from development_fit_train."""
    if not fit_train_entries:
        raise ValueError('Empty fit_train_entries')
    counts = [0] * 25
    for entry in fit_train_entries:
        _assert_analysis_entry(entry, expected_role='development_fit_train')
        counts[int(entry['mast3rpnp']['index'])] += 1
    total = sum(counts)

    # 1. Uniform25Prior: flat 1/25 probabilities with fixed row-major tie-break (distinguished from AlwaysCell12)
    u25_probs = [1.0 / 25.0] * 25
    u25_rank = list(range(25))

    # 2. UniformCenter9Prior: flat 1/9 inside Center9 (epsilon-smoothed to 25), tie-break toward cell 12
    c9_raw = [1.0 / 9.0 if k in CENTER9_INDICES else 0.0 for k in range(25)]
    c9_probs = [(1.0 - eps) * p + eps / 25.0 for p in c9_raw]
    c9_rank = sorted(range(25), key=lambda k: (0 if k in CENTER9_INDICES else 1, CELL_CENTER_DIST_TO_12_SQ[k], k))

    # 3. AlwaysCell12Prior: Dirac on cell 12 (epsilon-smoothed) + distance-to-center ranking
    c12_probs = [(1.0 - eps) * (1.0 if k == 12 else 0.0) + eps / 25.0 for k in range(25)]
    c12_rank = sorted(range(25), key=lambda k: (CELL_CENTER_DIST_TO_12_SQ[k], k))

    # 4. TrainEmpiricalPrior: Dirichlet-smoothed fit_train histogram + frequency/center ranking
    denom = total + 25.0 * dirichlet_alpha
    emp_probs = [(counts[k] + dirichlet_alpha) / denom for k in range(25)]
    emp_rank = sorted(range(25), key=lambda k: (-counts[k], CELL_CENTER_DIST_TO_12_SQ[k], k))

    return {
        'Uniform25Prior': {
            'probs': u25_probs,
            'ranking': u25_rank,
            'theoretical_random_top1': round(1.0 / 25.0, 6),
            'theoretical_random_top3': round(3.0 / 25.0, 6),
            'theoretical_random_top5': round(5.0 / 25.0, 6),
            'note': 'Row-major tie-break [0..24] for deterministic metrics; random-tie analytical Top-1/3/5 are 4%/12%/20% (see true_cell_prob_mean).',
        },
        'UniformCenter9Prior': {
            'probs': c9_probs,
            'ranking': c9_rank,
            'theoretical_random_top1': round(1.0 / 9.0, 6),
            'theoretical_random_top3': round(3.0 / 9.0, 6),
            'theoretical_random_top5': round(5.0 / 9.0, 6),
            'note': f'Smoothed eps={eps}; center-outward tie-break starts at cell 12 (matching AlwaysCell12 on discrete Top-1), while random-tie Top-1 is 11.11% (true_cell_prob_mean).',
        },
        'AlwaysCell12Prior': {
            'probs': c12_probs,
            'ranking': c12_rank,
            'note': f'Always predicts center cell 12 with eps={eps} uniform smoothing for finite KL/CE.',
        },
        'TrainEmpiricalPrior': {
            'probs': emp_probs,
            'ranking': emp_rank,
            'fit_train_total': total,
            'fit_train_counts': counts,
            'dirichlet_alpha': dirichlet_alpha,
            'note': f'Fitted exclusively on development_fit_train ({total} pairs) with Dirichlet alpha={dirichlet_alpha}.',
        },
    }


def evaluate_baseline_suite(entries: list[dict], baselines: dict[str, dict]) -> dict[str, dict]:
    """Evaluate all fitted statistical baselines across an allowed split."""
    results = {}
    for name, spec in baselines.items():
        probs_map = {e['image_id']: spec['probs'] for e in entries}
        rank_map = {e['image_id']: spec['ranking'] for e in entries}
        results[name] = {
            'ranking': spec['ranking'],
            'probs': list(spec['probs']),
            'note': spec['note'],
            'metrics': evaluate_single_step_predictions(entries, probs_map, rank_map),
        }
    return results
