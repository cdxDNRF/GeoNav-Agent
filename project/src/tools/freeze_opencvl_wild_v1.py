"""Generate the frozen CV-049 protocol and OpenCVL in_the_wild_test split bundle.

Zero image pixel decodes, zero training, zero model forward calls, zero
consumption of CV-045/MM-GAG/blind holdout sets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from project.src.data.capturepoint_corrected_v1 import distance
from project.src.data.opencvl_wild_intake_v1 import (
    connected_components_by_gap,
    dual_pose_summary,
    load_declared_known_sources,
    parse_png_header_from_file,
    partition_se_clean_macros,
    summarize_role_stats,
)

ROOT = Path(__file__).resolve().parents[3]
RAW_BASE = ROOT / 'DATA/raw_data/OpenCVL/OpenCVL/mapillary/in_the_wild_test'
SHA_TXT = ROOT / 'DATA/raw_data/OpenCVL/OpenCVL_mapillary_in_the_wild_test.tar.sha256'
CV045_BINDING = ROOT / 'DATA/processed_data/OpenCVL/公开瑞典三候选校正标签预检_v1/首获取冻结绑定.json'
OUT_DIR = ROOT / 'DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1'
CV044_PL_IDS = ('1004803046722240', '1016275108779955', '1020327673038045')


def file_sha256(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inspect_split_rows(split: str, *, require_latlon: bool) -> tuple[list[dict], str, int]:
    """Read SE or PL labels.json and validate PNG IHDR headers without pixel decode."""
    split_dir = RAW_BASE / split
    label_path = split_dir / 'labels.json'
    payload = load_json(label_path)
    seen = set()
    rows = []
    for raw in payload['labels']:
        sid = raw['mapil_image_id']
        if sid in seen:
            raise ValueError(f'Duplicate image ID in {split}: {sid}')
        seen.add(sid)
        g_meta = parse_png_header_from_file(split_dir / raw['mapil_image'])
        a_meta = parse_png_header_from_file(split_dir / raw['aerial_image'])
        parsed = dual_pose_summary(raw, a_meta['width'], a_meta['height'], require_latlon=require_latlon)
        parsed['split'] = split
        parsed['ground_asset'] = dict(rel=f'{split}/{parsed["ground_rel"]}', width=g_meta['width'], height=g_meta['height'], bytes=g_meta['bytes'])
        parsed['aerial_asset'] = dict(rel=f'{split}/{parsed["aerial_rel"]}', width=a_meta['width'], height=a_meta['height'], bytes=a_meta['bytes'])
        parsed['radius_m'] = parsed['mast3rpnp']['footprint_radius_m']
        rows.append(parsed)
    return sorted(rows, key=lambda r: r['image_id']), file_sha256(label_path), label_path.stat().st_size


def build_artifacts(root: Path = ROOT) -> dict[str, dict]:
    cv045 = load_json(CV045_BINDING)
    known = load_declared_known_sources(cv045)
    se_rows, se_sha, se_bytes = inspect_split_rows('SE', require_latlon=True)
    pl_rows, pl_sha, pl_bytes = inspect_split_rows('PL', require_latlon=False)

    clean_se, blocked_se = [], []
    for row in se_rows:
        exact_hit = [k['source_key'] for k in known if k['source_key'] == row['source_key']]
        min_gap = float('inf')
        nearest_key = None
        for k in known:
            gap = distance(row['aerial_latlon'], k['center_latlon']) - row['radius_m'] - k['radius_m']
            if gap < min_gap:
                min_gap = gap
                nearest_key = k['source_key']
        row['min_known_gap_m'] = min_gap
        row['nearest_known_source'] = nearest_key
        if exact_hit or min_gap < 500.0:
            row['role'] = 'blocked_known_overlap'
            row['macro_id'] = None
            row['subcluster_id'] = None
            blocked_se.append(row)
        else:
            clean_se.append(row)

    subclusters = connected_components_by_gap(clean_se, 500.0, subtract_radii=True)
    for idx, grp in enumerate(subclusters, 1):
        cid = f'C{idx:02d}'
        for item in grp:
            item['subcluster_id'] = cid

    role_by_source, macro_summaries = partition_se_clean_macros(clean_se, known)
    for item in clean_se:
        item['role'] = role_by_source[item['source_key']]

    pl_ids = {r['image_id'] for r in pl_rows}
    if not all(cid in pl_ids for cid in CV044_PL_IDS):
        raise ValueError('Missing expected CV-044 PL reference IDs')
    for r in pl_rows:
        r['role'] = 'pl_pending_visual_dedup'
        r['cv044_metadata_only'] = r['image_id'] in CV044_PL_IDS

    by_role: dict[str, list[dict]] = {}
    for item in se_rows + pl_rows:
        by_role.setdefault(item['role'], []).append(item)

    cross_clearance = _cross_role_clearances(by_role)
    role_stats = {role: summarize_role_stats(items) for role, items in sorted(by_role.items())}

    protocol = _protocol_document(role_stats, cross_clearance, len(known), len(subclusters), len(macro_summaries))
    public_manifest = _public_manifest(by_role, macro_summaries, role_stats, cross_clearance)
    evaluator_ledger = _evaluator_ledger(se_rows, pl_rows, role_stats, cross_clearance, se_sha, se_bytes, pl_sha, pl_bytes, known)
    return {
        '协议冻结_v1.json': protocol,
        '公开洁净划分清单_v1.json': public_manifest,
        '核验/评估器专用冻结台账_禁止入模.json': evaluator_ledger,
    }


def _cross_role_clearances(by_role: dict[str, list[dict]]) -> dict[str, float]:
    def min_center_km(a_role: str, b_role: str) -> float:
        return round(min(
            distance(a['aerial_latlon'], b['aerial_latlon'])
            for a in by_role[a_role] for b in by_role[b_role]
        ) / 1000.0, 3)

    return dict(
        blind_vs_known_min_gap_km=round(min(r['min_known_gap_m'] for r in by_role['isolated_confirmation_blind']) / 1000.0, 3),
        blind_vs_fit_train_min_center_km=min_center_km('isolated_confirmation_blind', 'development_fit_train'),
        blind_vs_dev_val_min_center_km=min_center_km('isolated_confirmation_blind', 'development_validation'),
        blind_vs_quarantine_min_center_km=min_center_km('isolated_confirmation_blind', 'quarantined_cv045_vicinity'),
        dev_val_vs_fit_train_min_center_km=min_center_km('development_validation', 'development_fit_train'),
        dev_val_vs_known_min_gap_km=round(min(r['min_known_gap_m'] for r in by_role['development_validation']) / 1000.0, 3),
    )


def _protocol_document(role_stats, cross_clearance, known_count, subcluster_count, macro_count) -> dict:
    return dict(
        schema='cv049-opencvl-wild-protocol/v1',
        task_id='CV-049',
        frozen_date='2026-11-14',
        revisions_adopted=[
            'R1: Abandon naive 100m shifted-crop on native 100m x 100m tiles (avoids zero-padding and CV-046 <=9-step B10 saturation); freeze Stage-1 as Single-Step Cross-View Target Localization against FixedCell12 & CenterCross priors.',
            'R2: Partition clean SE (591 pairs, 43 500m-clusters, 18 15km-macros) at macro-region level with blind confirmation gate and <=24 fit_train-only bounded QA quota.',
            'R3: Hold PL (419 pairs, aerial_latlon=null) in a separate non-isolated reference account until future visual-hash/feature deduplication against Krakow.',
        ],
        primary_pose_namespace='mast3rpnp',
        consensus_stratification='Report all_pairs AND opensfm_cell_agree subset separately',
        native_geometry=dict(
            se=dict(aerial_px=[625, 625], m_per_px=0.16, footprint_m=[100.0, 100.0]),
            pl=dict(aerial_px=[1000, 1000], m_per_px=0.10, footprint_m=[100.0, 100.0]),
            grid='5x5 half-open pixel bins; 20m x 20m per cell',
            formula='x=W/2+dx_mast3rpnp; y=H/2-dy_mast3rpnp',
            zero_padding_permitted=False,
        ),
        stage1_evaluation=dict(
            task='Single-Step Cross-View Target Localization (ground RGB + full 100m 5x5 reference -> 25-cell distribution)',
            primary_metrics=['top1_25cell_acc', 'top3_25cell_acc', 'top5_25cell_acc', 'center9_conditional_acc', 'cross_entropy_nats', 'expected_planar_error_m'],
            mandatory_baselines=['AlwaysCell12Prior', 'UniformCenter9Prior', 'CenterCross_9step_reference_disclosure'],
            promotion_rule='Must beat AlwaysCell12Prior strictly on development_validation AND post-freeze one-shot isolated_confirmation_blind (both all_pairs and opensfm_agree subsets).',
        ),
        data_governance=dict(
            declared_known_and_cv045_sources=known_count,
            se_clean_500m_clusters=subcluster_count,
            se_clean_15km_macros=macro_count,
            fit_train_visual_qa_max_pairs=24,
            dev_val_pixel_inspection_allowed=False,
            blind_confirmation_single_shot_only=True,
            pl_geographic_isolation_admitted=False,
        ),
        role_summary=role_stats,
        cross_role_clearance_km=cross_clearance,
        execution_state=dict(
            pixels_decoded=0, image_header_crc_only=True, training_runs=0, model_forward_calls=0, navigation_runs=0,
            confirmation_consumed=0, cv045_touched=False, mmgag_6groups_touched=False,
        ),
    )


def _public_manifest(by_role, macro_summaries, role_stats, cross_clearance) -> dict:
    """Public IDs & cluster tags without ground-truth offsets, cells, or blind GPS."""
    safe_macros = []
    for m in macro_summaries:
        entry = dict(
            macro_id=m['macro_id'], role=m['role'], count=m['count'],
            subcluster_count=m['subcluster_count'],
            min_known_gap_km=round(m['min_known_gap_m'] / 1000.0, 3),
            min_cv045_center_km=round(m['min_cv045_center_m'] / 1000.0, 3),
        )
        if m['role'] in ('development_fit_train', 'development_validation'):
            entry['image_ids'] = m['image_ids']
        else:
            entry['image_id_count'] = len(m['image_ids'])
            if m['role'] == 'isolated_confirmation_blind':
                entry['image_ids_blind_reserved'] = m['image_ids']
        safe_macros.append(entry)

    splits = {}
    for role, items in sorted(by_role.items()):
        splits[role] = [
            dict(
                image_id=it['image_id'],
                split=it['split'],
                ground_rel=it['ground_asset']['rel'],
                aerial_rel=it['aerial_asset']['rel'],
                macro_id=it.get('macro_id'),
                subcluster_id=it.get('subcluster_id'),
                blind_do_not_inspect=(role in ('isolated_confirmation_blind', 'blocked_known_overlap', 'quarantined_cv045_vicinity')),
            )
            for it in sorted(items, key=lambda x: x['image_id'])
        ]

    return dict(
        schema='cv049-public-split-manifest/v1',
        policy_safe_no_target_labels=True,
        cross_role_clearance_km=cross_clearance,
        role_baselines_and_counts=role_stats,
        se_macro_regions=safe_macros,
        splits=splits,
    )


def _evaluator_ledger(se_rows, pl_rows, role_stats, cross_clearance, se_sha, se_bytes, pl_sha, pl_bytes, known) -> dict:
    return dict(
        schema='cv049-evaluator-ledger/v1',
        policy_input=False,
        blind_confirmation_do_not_inspect_before_freeze=True,
        raw_sources=dict(
            se_labels_sha256=se_sha, se_labels_bytes=se_bytes,
            pl_labels_sha256=pl_sha, pl_labels_bytes=pl_bytes,
            declared_known_count=len(known),
        ),
        cross_role_clearance_km=cross_clearance,
        role_summary=role_stats,
        se_records=se_rows,
        pl_records=pl_rows,
    )


def write_bindings_and_outputs(out_dir: Path = OUT_DIR) -> dict:
    protected_paths = [
        'project/src/data/capturepoint_intake_v1.py',
        'project/src/data/capturepoint_corrected_v1.py',
        'project/src/tools/inspect_capturepoint_intake_v1.py',
        'DATA/processed_data/OpenCVL/公开瑞典三候选校正标签预检_v1/首获取冻结绑定.json',
    ]
    protected_sha = {rel: file_sha256(ROOT / rel) for rel in protected_paths}
    artifacts = build_artifacts(ROOT)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / '核验').mkdir(parents=True, exist_ok=True)
    written = []
    for rel, obj in artifacts.items():
        target = out_dir / rel
        target.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        written.append(dict(
            path=target.relative_to(ROOT).as_posix(),
            sha256=file_sha256(target),
            bytes=target.stat().st_size,
        ))

    code_paths = [
        'project/src/data/opencvl_wild_intake_v1.py',
        'project/src/tools/freeze_opencvl_wild_v1.py',
        'project/src/eval/audit_opencvl_wild_intake_v1.py',
        'project/src/tests/test_opencvl_wild_intake_v1.py',
    ]
    binding = dict(
        schema='cv049-source-input-binding/v1',
        task='CV-049',
        tar_sha256_file=dict(path=SHA_TXT.relative_to(ROOT).as_posix(), sha256=file_sha256(SHA_TXT), text=SHA_TXT.read_text('utf-8').strip()),
        protected_upstream_sha256=protected_sha,
        generator_sources=[dict(path=p, sha256=file_sha256(ROOT / p), bytes=(ROOT / p).stat().st_size) for p in code_paths],
        generated_artifacts=written,
        execution_accounting=dict(pixels_decoded=0, total_png_headers_checked=2722, model_calls=0, training=0, confirmation_consumed=0),
    )
    bpath = out_dir / '核验/源码与输入绑定_v1.json'
    bpath.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return binding


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true', help='Write frozen manifests under DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/')
    args = parser.parse_args(argv)
    if args.save:
        res = write_bindings_and_outputs(OUT_DIR)
        print(json.dumps(res['execution_accounting'], ensure_ascii=False))
        return 0
    arts = build_artifacts(ROOT)
    proto = arts['协议冻结_v1.json']
    print(json.dumps(dict(
        task=proto['task_id'], role_counts={k: v['count'] for k, v in proto['role_summary'].items()},
        cross_clearance=proto['cross_role_clearance_km'], pixels_decoded=0,
    ), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
