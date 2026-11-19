"""Independent alternate audit for CV-050 fit_train QA and Stage-1 statistical baselines.

Zero image pixel decoding inside the auditor: verifies PNG signatures/IHDR CRCs
and SHA-256 digests independently, uses bisect_right + raw Gaussian kernel sums to
recompute all discrepancy geometry and Stage-1 baseline metrics within <= 1e-9,
and scans outputs to prove zero blind-set evaluation or non-fit_train decoding.
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
CV049_DIR = ROOT / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1'
OUT_DIR = ROOT / 'DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1'
CV045_BINDING = ROOT / 'DATA/processed_data/OpenCVL/公开瑞典三候选校正标签预检_v1/首获取冻结绑定.json'

ALLOWED_GROUND_SIZES = frozenset({(952, 714), (952, 532)})
ALLOWED_AERIAL_SIZES = frozenset({(625, 625)})
INTERNAL_EDGES = (125.0, 250.0, 375.0, 500.0)


def sha(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_png_ihdr_dimensions(path: Path, expected_w: int, expected_h: int) -> None:
    """Validate PNG signature, IHDR CRC-32, and width/height without pixel decoding."""
    with Path(path).open('rb') as f:
        head = f.read(33)
    if len(head) < 33 or head[:8] != b'\x89PNG\r\n\x1a\n' or head[8:16] != b'\x00\x00\x00\rIHDR':
        raise ValueError(f'Invalid PNG signature/IHDR: {path}')
    if (zlib.crc32(head[12:29]) & 0xffffffff) != struct.unpack('>I', head[29:33])[0]:
        raise ValueError(f'IHDR CRC-32 mismatch: {path}')
    w, h = struct.unpack('>II', head[16:24])
    if (w, h) != (expected_w, expected_h):
        raise ValueError(f'PNG dimension mismatch for {path}: {(w, h)} vs {(expected_w, expected_h)}')


def alt_cell_from_xy(x: float, y: float, w: int = 625, h: int = 625) -> int:
    xe = [i * w // 5 for i in range(6)]
    ye = [i * h // 5 for i in range(6)]
    col = bisect_right(xe, x) - 1
    row = bisect_right(ye, y) - 1
    if not (0 <= col < 5 and 0 <= row < 5):
        raise ValueError(f'Coordinates outside 5x5 grid: {(x, y)}')
    return row * 5 + col


def alt_median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def alt_geometry(rec: dict) -> dict:
    m = rec['mast3rpnp']
    o = rec['mapilopensfm']
    res = float(rec['resolution'])
    xm, ym = float(m['x']), float(m['y'])
    xo, yo = float(o['x']), float(o['y'])
    idx_m = alt_cell_from_xy(xm, ym)
    idx_o = alt_cell_from_xy(xo, yo)
    if idx_m != int(m['index']) or idx_o != int(o['index']):
        raise ValueError(f'Alternate cell mismatch on {rec["image_id"]}')
    rm, cm = divmod(idx_m, 5)
    ro, co = divmod(idx_o, 5)
    margin_m = min(min(abs(xm - e), abs(ym - e)) for e in INTERNAL_EDGES) * res
    diff_m = math.sqrt((xm - xo) ** 2 + (ym - yo) ** 2) * res
    l1 = abs(rm - ro) + abs(cm - co)
    cheb = max(abs(rm - ro), abs(cm - co))
    return {
        'xm': xm,
        'ym': ym,
        'res': res,
        'cell_m': idx_m,
        'cell_o': idx_o,
        'agree': idx_m == idx_o,
        'margin_m': margin_m,
        'diff_m': diff_m,
        'l1': l1,
        'cheb': cheb,
    }


def alt_eval_baseline(geos: list[dict], probs: list[float], ranking: list[int], sigma_m: float = 5.0) -> dict[str, dict]:
    centers = [((k % 5) * 125.0 + 62.5, (k // 5) * 125.0 + 62.5) for k in range(25)]
    top1_cell = ranking[0]
    top3_set = set(ranking[:3])
    top5_set = set(ranking[:5])
    r1, c1 = divmod(top1_cell, 5)

    buckets = {
        'all_pairs': [],
        'opensfm_cell_agree': [],
        'interior_margin_ge_2m': [],
        'large_discrepancy_gt_10m': [],
    }
    inv_2v = 0.5 / (sigma_m * sigma_m)
    clip = 1e-12

    for g in geos:
        rm, cm = divmod(g['cell_m'], 5)
        dr, dc = abs(r1 - rm), abs(c1 - cm)
        dists = [math.hypot(cx - g['xm'], cy - g['ym']) * g['res'] for cx, cy in centers]
        raw_q = [math.exp(-d * d * inv_2v) for d in dists]
        z = sum(raw_q)
        soft_q = [w / z for w in raw_q]
        exp_err = sum(p * d for p, d in zip(probs, dists))
        kl = sum(q * math.log(max(q, clip) / max(p, clip)) for q, p in zip(soft_q, probs) if q > 0.0)
        ce = -sum(q * math.log(max(p, clip)) for q, p in zip(soft_q, probs))
        row = {
            'top1': int(g['cell_m'] == top1_cell),
            'top3': int(g['cell_m'] in top3_set),
            'top5': int(g['cell_m'] in top5_set),
            'cheb1': int(max(dr, dc) <= 1),
            'l1_1': int((dr + dc) <= 1),
            'prob': probs[g['cell_m']],
            'center_err': dists[top1_cell],
            'exp_err': exp_err,
            'ce': ce,
            'kl': kl,
        }
        buckets['all_pairs'].append(row)
        if g['agree']:
            buckets['opensfm_cell_agree'].append(row)
        if g['margin_m'] >= 2.0:
            buckets['interior_margin_ge_2m'].append(row)
        if g['diff_m'] > 10.0:
            buckets['large_discrepancy_gt_10m'].append(row)

    summary = {}
    for k, rows in buckets.items():
        n = len(rows)
        if n == 0:
            summary[k] = {'count': 0}
            continue
        summary[k] = {
            'count': n,
            'top1_acc': sum(r['top1'] for r in rows) / n,
            'top3_acc': sum(r['top3'] for r in rows) / n,
            'top5_acc': sum(r['top5'] for r in rows) / n,
            'chebyshev1_acc': sum(r['cheb1'] for r in rows) / n,
            'l1_1_acc': sum(r['l1_1'] for r in rows) / n,
            'true_cell_prob_mean': sum(r['prob'] for r in rows) / n,
            'top1_center_error_m': sum(r['center_err'] for r in rows) / n,
            'expected_planar_error_m': sum(r['exp_err'] for r in rows) / n,
            'soft_target_ce_nats': sum(r['ce'] for r in rows) / n,
            'soft_target_kl_nats': sum(r['kl'] for r in rows) / n,
        }
    return summary


def audit_qa_bundle(out_dir: Path = OUT_DIR, root: Path = ROOT) -> dict:
    """Run all independent CV-050 verifications and return structured audit summary."""
    if 'torch' in sys.modules:
        raise ValueError('Model framework must not be imported during audit')

    binding = read_json(out_dir / '核验/源码与输入绑定_v1.json')
    for rel, expected in binding['upstream_frozen_sha256'].items():
        if sha(root / rel) != expected:
            raise ValueError(f'Upstream digest mismatch: {rel}')
    for entry in binding['generator_sources'] + binding['generated_artifacts']:
        p = root / entry['path']
        if sha(p) != entry['sha256'] or p.stat().st_size != entry['bytes']:
            raise ValueError(f'CV-050 digest mismatch: {entry["path"]}')
    if binding.get('alternate_audit_artifact') is not None:
        aa = binding['alternate_audit_artifact']
        ap = root / aa['path']
        if sha(ap) != aa['sha256'] or ap.stat().st_size != aa['bytes']:
            raise ValueError(f'Alternate audit digest mismatch: {aa["path"]}')

    ledger = read_json(root / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/核验/评估器专用冻结台账_禁止入模.json')
    pub = read_json(root / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/公开洁净划分清单_v1.json')
    cv045_ids = set(read_json(root / 'DATA/processed_data/OpenCVL/公开瑞典三候选校正标签预检_v1/首获取冻结绑定.json')['selected_ids'])

    qa_doc = read_json(out_dir / '抽检24对画质与分歧核验_v1.json')
    base_doc = read_json(out_dir / '单步定位统计基线对照_v1.json')
    proto_doc = read_json(out_dir / '单步视觉实验冻结协议_v1.json')

    # 1. Isolation, decode quota, and blind seal checks
    if qa_doc['sampled_pairs'] != 24 or qa_doc['image_decodes'] != 48:
        raise ValueError('QA decode quota mismatch')
    if qa_doc['decode_accounting']['non_fit_train_decodes'] != 0:
        raise ValueError('Non-fit_train decodes must be zero')
    if base_doc['blind_set_evaluated'] is not False or qa_doc['discrepancy_summary']['blind_set_inspected'] is not False:
        raise ValueError('Blind set must remain sealed and unevaluated')
    if set(base_doc['evaluations']) != {'development_fit_train', 'development_validation'}:
        raise ValueError('Unexpected splits evaluated in baseline report')

    blind_ids = {item['image_id'] for item in pub['splits']['isolated_confirmation_blind']}
    base_raw_text = (out_dir / '单步定位统计基线对照_v1.json').read_text(encoding='utf-8')
    proto_raw_text = (out_dir / '单步视觉实验冻结协议_v1.json').read_text(encoding='utf-8')
    qa_raw_text = (out_dir / '抽检24对画质与分歧核验_v1.json').read_text(encoding='utf-8')
    for bid in blind_ids:
        if bid in base_raw_text or bid in proto_raw_text or bid in qa_raw_text:
            raise ValueError(f'Blind image_id {bid} leaked into CV-050 artifacts')

    fit_train = [r for r in ledger['se_records'] if r['role'] == 'development_fit_train']
    dev_val = [r for r in ledger['se_records'] if r['role'] == 'development_validation']
    fit_by_id = {r['image_id']: r for r in fit_train}
    forbidden_ids = {
        r['image_id'] for r in ledger['se_records'] if r['role'] != 'development_fit_train'
    } | {r['image_id'] for r in ledger['pl_records']} | cv045_ids

    # 2. Verify all 24 pairs / 48 images without pixel decoding
    seen_ids = set()
    raw_base = root / 'DATA/raw_data/OpenCVL/OpenCVL/mapillary/in_the_wild_test'
    max_moment_diff = 0.0
    png_header_checks = 0
    for row in qa_doc['sampled_pair_reports']:
        sid = row['image_id']
        if sid in seen_ids or sid not in fit_by_id or sid in forbidden_ids or row['role'] != 'development_fit_train':
            raise ValueError(f'Illegal QA pair: {sid}')
        seen_ids.add(sid)
        rec = fit_by_id[sid]

        for kind, qa_key, asset_key, allowed_sizes in (
            ('ground', 'ground_qa', 'ground_asset', ALLOWED_GROUND_SIZES),
            ('aerial', 'aerial_qa', 'aerial_asset', ALLOWED_AERIAL_SIZES),
        ):
            qa_info = row[qa_key]
            hdr = qa_info['header']
            asset = rec[asset_key]
            img_path = raw_base / asset['rel']
            if sha(img_path) != hdr['file_sha256'] or img_path.stat().st_size != hdr['bytes']:
                raise ValueError(f'PNG digest/size mismatch on {img_path}')
            verify_png_ihdr_dimensions(img_path, asset['width'], asset['height'])
            png_header_checks += 1
            if (hdr['width'], hdr['height']) != (asset['width'], asset['height']) or (hdr['width'], hdr['height']) not in allowed_sizes:
                raise ValueError(f'Invalid {kind} PNG dimension for {sid}')
            q = qa_info['quality']
            if not q['nonblank'] or q['gray_std'] < 2.0 or max(q['near_black_fraction'], q['near_white_fraction']) >= 0.95:
                raise ValueError(f'Quality gate failed on {sid}/{kind}')
            diff = max(
                max(abs(a - b) for a, b in zip(q['channel_mean'], qa_info['alternate_mean'])),
                max(abs(a - b) for a, b in zip(q['channel_std'], qa_info['alternate_std'])),
            )
            max_moment_diff = max(max_moment_diff, diff)
            if diff > 1e-6:
                raise ValueError(f'Alternate moment diff > 1e-6 on {sid}/{kind}: {diff}')

    # 3. Independent discrepancy geometry & metric cross-check across fit_train (435) and dev_val (116)
    max_metric_diff = 0.0
    max_discrepancy_diff = 0.0
    geos_by_split = {}
    for split_name, recs in (('development_fit_train', fit_train), ('development_validation', dev_val)):
        geos = [alt_geometry(r) for r in recs]
        geos_by_split[split_name] = geos
        saved = qa_doc['discrepancy_summary'][split_name]
        dis = [g for g in geos if not g['agree']]
        agr = [g for g in geos if g['agree']]
        if len(dis) != saved['disagree_count'] or len(agr) != saved['agree_count']:
            raise ValueError(f'Disagree count mismatch in {split_name}')
        if sum(1 for g in dis if g['cheb'] <= 1) != saved['disagree_topology']['chebyshev_le_1_count']:
            raise ValueError(f'Chebyshev<=1 mismatch in {split_name}')
        if sum(1 for g in dis if g['l1'] == 1) != saved['disagree_topology']['l1_eq_1_count']:
            raise ValueError(f'L1==1 mismatch in {split_name}')
        max_discrepancy_diff = max(
            max_discrepancy_diff,
            abs(alt_median([g['margin_m'] for g in dis]) - saved['edge_margin_comparison_m']['disagree_pairs']['quantiles_m']['median']),
            abs(alt_median([g['margin_m'] for g in agr]) - saved['edge_margin_comparison_m']['agree_pairs']['quantiles_m']['median']),
            abs(alt_median([g['diff_m'] for g in dis]) - saved['disagree_topology']['planar_diff_m_quantiles']['median']),
        )

    # 4. Recompute statistical baselines from scratch and verify within <= 1e-9
    counts = [0] * 25
    for g in geos_by_split['development_fit_train']:
        counts[g['cell_m']] += 1
    emp_probs = [(counts[k] + 0.5) / (len(fit_train) + 12.5) for k in range(25)]
    dist_sq = [((k % 5) - 2) ** 2 + ((k // 5) - 2) ** 2 for k in range(25)]
    emp_rank = sorted(range(25), key=lambda k: (-counts[k], dist_sq[k], k))
    c12_probs = [(1.0 - 1e-3) * (1.0 if k == 12 else 0.0) + 1e-3 / 25.0 for k in range(25)]
    c12_rank = sorted(range(25), key=lambda k: (dist_sq[k], k))
    c9_set = {6, 7, 8, 11, 12, 13, 16, 17, 18}
    c9_probs = [(1.0 - 1e-3) * (1.0 / 9.0 if k in c9_set else 0.0) + 1e-3 / 25.0 for k in range(25)]
    c9_rank = sorted(range(25), key=lambda k: (0 if k in c9_set else 1, dist_sq[k], k))
    u25_probs = [1.0 / 25.0] * 25
    u25_rank = list(range(25))

    alt_specs = {
        'Uniform25Prior': (u25_probs, u25_rank),
        'UniformCenter9Prior': (c9_probs, c9_rank),
        'AlwaysCell12Prior': (c12_probs, c12_rank),
        'TrainEmpiricalPrior': (emp_probs, emp_rank),
    }

    for split_name in ('development_fit_train', 'development_validation'):
        saved_eval = base_doc['evaluations'][split_name]
        for bname, (bprobs, brank) in alt_specs.items():
            if saved_eval[bname]['ranking'] != brank:
                raise ValueError(f'Ranking mismatch for {split_name}/{bname}')
            alt_metrics = alt_eval_baseline(geos_by_split[split_name], bprobs, brank)
            saved_metrics = saved_eval[bname]['metrics']
            for stratum, sm in saved_metrics.items():
                am = alt_metrics[stratum]
                if am['count'] != sm['count']:
                    raise ValueError(f'Count mismatch in {split_name}/{bname}/{stratum}')
                if am['count'] == 0:
                    continue
                for mkey, aval in am.items():
                    if mkey == 'count':
                        continue
                    diff = abs(float(aval) - float(sm[mkey]))
                    max_metric_diff = max(max_metric_diff, diff)
                    if diff > 1e-9:
                        raise ValueError(f'Metric mismatch {split_name}/{bname}/{stratum}/{mkey}: {diff}')

    if max_discrepancy_diff > 1e-9:
        raise ValueError(f'Discrepancy statistic mismatch: {max_discrepancy_diff}')

    return {
        'schema': 'cv050-alternate-audit/v1',
        'task': 'CV-050',
        'passed': True,
        'checks': {
            'zero_torch_import': True,
            'upstream_cv049_hashes_match': True,
            'cv050_artifact_hashes_match': True,
            'qa_pairs_strictly_fit_train': True,
            'zero_non_fit_train_decodes': True,
            'png_ihdr_crc_and_sizes_match': True,
            'all_48_images_nonblank': True,
            'alternate_quality_moments_within_1e6': True,
            'alternate_discrepancy_geometry_within_1e9': True,
            'alternate_baseline_metrics_within_1e9': True,
            'blind_confirmation_sealed_and_unleaked': True,
            'cv051_protocol_floors_match': (
                proto_doc['dev_val_statistical_floors_to_beat']['AlwaysCell12Prior']['all_pairs_top1']
                == base_doc['evaluations']['development_validation']['AlwaysCell12Prior']['metrics']['all_pairs']['top1_acc']
            ),
        },
        'verified_counts': {
            'sampled_qa_pairs': len(seen_ids),
            'png_headers_verified': png_header_checks,
            'fit_train_pairs': len(fit_train),
            'dev_val_pairs': len(dev_val),
            'blind_pairs_sealed': len(blind_ids),
        },
        'max_moment_diff': max_moment_diff,
        'max_discrepancy_diff': max_discrepancy_diff,
        'max_baseline_metric_diff': max_metric_diff,
        'auditor_image_decodes': 0,
        'model_calls': 0,
        'training': 0,
        'formal_confirmation_consumed': 0,
        'independent_personnel_review': False,
        'notes': 'same Antigravity session, independent bisect_right / PNG IHDR CRC / direct-sum soft-KL verification',
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true', help='Write 核验/另实现复核_v1.json and refresh 核验/源码与输入绑定_v1.json')
    args = parser.parse_args(argv)
    result = audit_qa_bundle(OUT_DIR, ROOT)
    if args.save:
        from project.src.tools.run_opencvl_wild_qa_v1 import update_source_binding

        target = OUT_DIR / '核验/另实现复核_v1.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        update_source_binding(OUT_DIR, ROOT)
        result = audit_qa_bundle(OUT_DIR, ROOT)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
