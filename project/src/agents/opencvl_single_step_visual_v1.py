"""CV-051 Stage-2 single-step cross-view visual localization, multi-scale tokens & ablations.

Strictly enforces OpenCVL split & leakage boundaries:
- Feature extraction decodes ONLY development_fit_train (435 pairs) and
  development_validation (116 pairs). isolated_confirmation_blind (28 pairs),
  quarantined_cv045_vicinity (12), blocked_known_overlap (351), and PL (419)
  are unconditionally rejected.
- All PCA/CCA subspace projections, feature standardization, hyperparameters,
  empirical priors, and linear spatial readout weights are fitted exclusively
  on development_fit_train (435 pairs).
- Uses single live encoder slot on CPU (max_live_encoders=1) with SHA-256 and
  tensor-digest invariance checks from frozen_crossview_encoder_v2.
"""
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import minimize

from project.src.data.mmgag_views_v1 import aerial_pixels
from project.src.data.opencvl_wild_intake_v1 import CENTER9_INDICES
from project.src.data.opencvl_wild_qa_v1 import (
    ALLOWED_SE_AERIAL_SIZES,
    ALLOWED_SE_GROUND_SIZES,
    CELL_CENTER_DIST_TO_12_SQ,
    STRATUM_NAMES,
    build_statistical_baselines,
    compute_pair_discrepancy_geometry,
    evaluate_baseline_suite,
    evaluate_single_step_predictions,
    gaussian_soft_target_distribution,
)

ALLOWED_STAGE2_ROLES = frozenset({'development_fit_train', 'development_validation'})
FORBIDDEN_STAGE2_ROLES = frozenset({
    'isolated_confirmation_blind',
    'quarantined_cv045_vicinity',
    'blocked_known_overlap',
    'pl_pending_visual_dedup',
})
STAGE2_STRATA = tuple(STRATUM_NAMES) + ('ground_size_952x714', 'ground_size_952x532')
CENTER9_CELLS: tuple[int, ...] = tuple(sorted(CENTER9_INDICES))
CENTER9_CELL_TO_POS: dict[int, int] = {cell: idx for idx, cell in enumerate(CENTER9_CELLS)}

CCA_RANK = 16
LAMBDA_GRID = (0.05, 0.2, 1.0, 5.0, 20.0)
BETA_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
TOP_LOCAL_INDICES_R1 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 26)
TOP_LOCAL_INDICES_R2 = TOP_LOCAL_INDICES_R1 + (27, 28, 29, 30, 31, 48)


def _file_sha256(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _assert_stage2_entry(entry: dict, expected_role: str | None = None) -> str:
    role = entry.get('role') or entry.get('split_role')
    if (
        not isinstance(role, str)
        or role in FORBIDDEN_STAGE2_ROLES
        or role not in ALLOWED_STAGE2_ROLES
    ):
        raise ValueError(f'Split role {role!r} is forbidden in Stage 2 (CV-051)')
    if entry.get('blind_do_not_inspect') is True or entry.get('policy_admitted') is False:
        raise ValueError(f'Restricted entry rejected in Stage 2: {entry.get("image_id")}')
    if expected_role is not None and role != expected_role:
        raise ValueError(f'Expected role {expected_role!r}, got {role!r} for {entry.get("image_id")}')
    if entry.get('country') != 'SE':
        raise ValueError(f'Stage 2 expects SE records only, got {entry.get("country")!r}')

    ga = entry.get('ground_asset') or {}
    aa = entry.get('aerial_asset') or {}
    g_size = (int(ga.get('width', -1)), int(ga.get('height', -1)))
    a_size = (int(aa.get('width', -1)), int(aa.get('height', -1)))
    if g_size not in ALLOWED_SE_GROUND_SIZES:
        raise ValueError(f'Unexpected ground size {g_size} for {entry.get("image_id")}')
    if a_size not in ALLOWED_SE_AERIAL_SIZES:
        raise ValueError(f'Unexpected aerial size {a_size} for {entry.get("image_id")}')
    return role


def build_grid_overlap_matrix(src_grid: int, dst_grid: int) -> np.ndarray:
    """Compute exact 2D interval overlap weights pooling src_grid^2 ViT patches to dst_grid^2 cells."""
    if src_grid <= 0 or dst_grid <= 0:
        raise ValueError('Grid dimensions must be positive')
    weights = np.zeros((dst_grid * dst_grid, src_grid * src_grid), dtype=np.float64)
    for dr in range(dst_grid):
        dy0, dy1 = dr / dst_grid, (dr + 1) / dst_grid
        for dc in range(dst_grid):
            dx0, dx1 = dc / dst_grid, (dc + 1) / dst_grid
            d_idx = dr * dst_grid + dc
            for sr in range(src_grid):
                sy0, sy1 = sr / src_grid, (sr + 1) / src_grid
                oy = max(0.0, min(dy1, sy1) - max(dy0, sy0))
                if oy <= 0.0:
                    continue
                for sc in range(src_grid):
                    sx0, sx1 = sc / src_grid, (sc + 1) / src_grid
                    ox = max(0.0, min(dx1, sx1) - max(dx0, sx0))
                    if ox > 0.0:
                        weights[d_idx, sr * src_grid + sc] = oy * ox
    row_sums = weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError('Zero overlap row detected')
    return weights / row_sums


def preprocess_ground_aspect_neutral(
    rgb: np.ndarray,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> np.ndarray:
    """Apply centered min(W,H) square crop, bicubic resize to 224x224, and normalize."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Ground image must be (H, W, 3) uint8 RGB')
    h, w, _ = rgb.shape
    if (w, h) not in ALLOWED_SE_GROUND_SIZES:
        raise ValueError(f'Unsupported SE ground dimensions {(w, h)}')
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = rgb[top : top + side, left : left + side]
    im = Image.fromarray(cropped, mode='RGB').resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray(image_mean, dtype=np.float32)
    std = np.asarray(image_std, dtype=np.float32)
    arr = (arr - mean) / std
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def preprocess_aerial_full_and_center9(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return full (625x625->224) and Center-9 (375x375 [125:500]->224) Sat2Cap inputs."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape != (625, 625, 3):
        raise ValueError(f'Aerial image must be (625, 625, 3) uint8 RGB, got {getattr(rgb, "shape", None)}')
    full_tensor = aerial_pixels(rgb)
    center9_patch = np.ascontiguousarray(rgb[125:500, 125:500])
    center9_tensor = aerial_pixels(center9_patch)
    return full_tensor, center9_tensor


def _load_rgb_and_check_size(path: Path, expected_w: int, expected_h: int, *, ground: bool) -> np.ndarray:
    with Image.open(path) as img:
        fixed = ImageOps.exif_transpose(img) if ground else img
        w, h = fixed.size
        if (w, h) != (expected_w, expected_h):
            raise ValueError(f'Decoded size mismatch on {path}: {(w, h)} vs {(expected_w, expected_h)}')
        return np.asarray(fixed.convert('RGB'), dtype=np.uint8)


def _l2_normalize_np(arr: np.ndarray) -> np.ndarray:
    arr64 = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr64).all():
        raise ValueError('Non-finite values before L2 normalization')
    norms = np.linalg.norm(arr64, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('Zero-norm vector encountered before L2 normalization')
    return arr64 / norms


def _safe_cosine_last_dim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=-1)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = np.maximum(na * nb, 1e-12)
    return num / denom


def _pool_aerial_and_bands(
    cls_emb: np.ndarray,
    patch_512: np.ndarray,
    *,
    w_pool: np.ndarray | None = None,
    bands_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cls_norm = _l2_normalize_np(cls_emb).astype(np.float32)
    if bands_mode:
        grid = patch_512.reshape(patch_512.shape[0], 7, 7, 512)
        b0 = grid[:, 0:2, :, :].mean(axis=(1, 2))
        b1 = grid[:, 2:5, :, :].mean(axis=(1, 2))
        b2 = grid[:, 5:7, :, :].mean(axis=(1, 2))
        stacked = np.stack([b0, b1, b2], axis=1)
        return cls_norm, _l2_normalize_np(stacked).astype(np.float32)
    if w_pool is None:
        raise ValueError('w_pool must be provided when bands_mode=False')
    pooled = np.einsum('kf,bfd->bkd', w_pool, patch_512.astype(np.float64))
    return cls_norm, _l2_normalize_np(pooled).astype(np.float32)


def _load_frozen_vision_slot(root: Path, kind: str):
    import torch
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    from project.src.agents.frozen_crossview_encoder_v2 import construct_mapped_model, tensor_digest

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    folder = root / ('models/clip_vit_b32_public_v1' if kind == 'ground' else 'models/Sat2Cap')
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    cfg = dict(cfg.get('vision_config', cfg), projection_dim=512)
    if any(cfg.get(k) != v for k, v in [('image_size', 224), ('patch_size', 32), ('hidden_size', 768), ('num_hidden_layers', 12)]):
        raise ValueError(f'Unexpected ViT config for {kind}')
    if kind == 'ground':
        raw = torch.load(folder / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
        state = {k: v for k, v in raw.items() if k.startswith(('vision_model.', 'visual_projection.'))}
    else:
        from safetensors.torch import load_file
        state = dict(load_file(str(folder / 'model.safetensors'), device='cpu'))
    for key in list(state):
        if key.endswith('.position_ids'):
            if state[key].flatten().tolist() != list(range(50)):
                raise ValueError('Unexpected position_ids buffer')
            state.pop(key)
    config = CLIPVisionConfig(**cfg)
    config._attn_implementation = 'eager'
    model = construct_mapped_model(config, state, CLIPVisionModelWithProjection)
    del state
    before_sha = tensor_digest(model)
    return model, before_sha


def _forward_clip_batches(model, pixel_batches: np.ndarray, *, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    import torch

    n = pixel_batches.shape[0]
    cls_list = []
    patch_list = []
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            xb = torch.from_numpy(pixel_batches[start : start + batch_size]).to(device='cpu')
            out = model(pixel_values=xb)
            cls_list.append(out.image_embeds.detach().cpu().numpy())
            patch_hidden = model.vision_model.post_layernorm(out.last_hidden_state[:, 1:, :])
            patch_proj = model.visual_projection(patch_hidden).detach().cpu().numpy()
            patch_list.append(patch_proj)
    return np.concatenate(cls_list, axis=0), np.concatenate(patch_list, axis=0)


def verify_encoder_assets(root: Path) -> list[dict]:
    assets_path = root / 'project/src/platform_core/ground_live_assets_v1.json'
    assets = json.loads(assets_path.read_text(encoding='utf-8'))['models']
    required = {
        'models/clip_vit_b32_public_v1/config.json',
        'models/clip_vit_b32_public_v1/preprocessor_config.json',
        'models/clip_vit_b32_public_v1/pytorch_model.bin',
        'models/Sat2Cap/config.json',
        'models/Sat2Cap/model.safetensors',
    }
    by_path = {a['path']: a for a in assets}
    if set(by_path) != required:
        raise ValueError('Unexpected encoder asset set in ground_live_assets_v1.json')
    for rel, meta in by_path.items():
        p = root / rel
        if not p.is_file() or p.stat().st_size != meta['bytes'] or _file_sha256(p) != meta['sha256']:
            raise ValueError(f'Encoder asset digest mismatch: {rel}')
    return assets


def extract_stage2_dual_encoder_features(
    entries: list[dict],
    raw_base: Path,
    root: Path,
    cache_path: Path | None = None,
    *,
    batch_size: int = 32,
) -> tuple[dict[str, np.ndarray], dict]:
    """Extract frozen single-slot CLIP+Sat2Cap embeddings and spatial tokens on allowed Stage-2 dev splits."""
    from project.src.agents.frozen_crossview_encoder_v2 import tensor_digest

    if not entries:
        raise ValueError('Empty Stage-2 entries')
    verified_assets = verify_encoder_assets(root)
    proc_cfg = json.loads((root / 'models/clip_vit_b32_public_v1/preprocessor_config.json').read_text(encoding='utf-8'))
    image_mean, image_std = proc_cfg['image_mean'], proc_cfg['image_std']

    canonical_entries = sorted(
        entries,
        key=lambda e: (0 if _assert_stage2_entry(e) == 'development_fit_train' else 1, str(e['image_id'])),
    )
    image_ids = np.array([str(e['image_id']) for e in canonical_entries])
    roles = np.array([_assert_stage2_entry(e) for e in canonical_entries])
    ground_sizes = np.array([
        f'{e["ground_asset"]["width"]}x{e["ground_asset"]["height"]}' for e in canonical_entries
    ])
    n = len(canonical_entries)

    # Slot 1: Ground encoder (CLIP ViT-B/32)
    g_tensors = np.empty((n, 3, 224, 224), dtype=np.float32)
    for i, rec in enumerate(canonical_entries):
        ga = rec['ground_asset']
        rgb = _load_rgb_and_check_size(raw_base / ga['rel'], int(ga['width']), int(ga['height']), ground=True)
        g_tensors[i] = preprocess_ground_aspect_neutral(rgb, image_mean, image_std)

    g_model, g_before = _load_frozen_vision_slot(root, 'ground')
    try:
        g_cls_raw, g_patch_raw = _forward_clip_batches(g_model, g_tensors, batch_size=batch_size)
        g_after = tensor_digest(g_model)
        if g_after != g_before or g_model.training or any(p.requires_grad for p in g_model.parameters()):
            raise ValueError('Frozen ground encoder state mutated')
    finally:
        del g_model, g_tensors
        gc.collect()

    ground_cls_512, ground_bands_512 = _pool_aerial_and_bands(g_cls_raw, g_patch_raw, bands_mode=True)

    # Slot 2: Aerial encoder (Sat2Cap)
    a_full_tensors = np.empty((n, 3, 224, 224), dtype=np.float32)
    a_c9_tensors = np.empty((n, 3, 224, 224), dtype=np.float32)
    for i, rec in enumerate(canonical_entries):
        aa = rec['aerial_asset']
        rgb = _load_rgb_and_check_size(raw_base / aa['rel'], int(aa['width']), int(aa['height']), ground=False)
        a_full_tensors[i], a_c9_tensors[i] = preprocess_aerial_full_and_center9(rgb)

    w25 = build_grid_overlap_matrix(7, 5)
    w9 = build_grid_overlap_matrix(7, 3)

    a_model, a_before = _load_frozen_vision_slot(root, 'aerial')
    try:
        af_cls_raw, af_patch_raw = _forward_clip_batches(a_model, a_full_tensors, batch_size=batch_size)
        ac_cls_raw, ac_patch_raw = _forward_clip_batches(a_model, a_c9_tensors, batch_size=batch_size)
        a_after = tensor_digest(a_model)
        if a_after != a_before or a_model.training or any(p.requires_grad for p in a_model.parameters()):
            raise ValueError('Frozen aerial encoder state mutated')
    finally:
        del a_model, a_full_tensors, a_c9_tensors
        gc.collect()

    aerial_cls_512, aerial_full25_512 = _pool_aerial_and_bands(af_cls_raw, af_patch_raw, w_pool=w25)
    aerial_center_cls_512, aerial_center9_512 = _pool_aerial_and_bands(ac_cls_raw, ac_patch_raw, w_pool=w9)

    arrays = {
        'image_ids': image_ids,
        'roles': roles,
        'ground_sizes': ground_sizes,
        'ground_cls_512': ground_cls_512,
        'ground_bands_512': ground_bands_512,
        'aerial_cls_512': aerial_cls_512,
        'aerial_full25_512': aerial_full25_512,
        'aerial_center_cls_512': aerial_center_cls_512,
        'aerial_center9_512': aerial_center9_512,
        'w25_from_49': w25.astype(np.float64),
        'w9_from_49': w9.astype(np.float64),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **arrays)

    audit = {
        'pair_count': n,
        'fit_train_pairs': int(np.sum(roles == 'development_fit_train')),
        'dev_val_pairs': int(np.sum(roles == 'development_validation')),
        'ground_forwards': n,
        'aerial_full_forwards': n,
        'aerial_center9_forwards': n,
        'total_forwards': 3 * n,
        'max_live_encoders': 1,
        'training_during_extraction': 0,
        'blind_decodes': 0,
        'verified_assets': verified_assets,
        'frozen_tensor_checks': [
            {'kind': 'ground', 'before_tensor_sha256': g_before, 'after_tensor_sha256': g_after, 'unchanged': True},
            {'kind': 'aerial', 'before_tensor_sha256': a_before, 'after_tensor_sha256': a_after, 'unchanged': True},
        ],
    }
    return arrays, audit


def _local_3x3_aerial_mean(aerial_full25: np.ndarray) -> np.ndarray:
    grid = aerial_full25.reshape(-1, 5, 5, 512)
    out = np.zeros_like(grid)
    for r in range(5):
        r0, r1 = max(0, r - 1), min(5, r + 2)
        for c in range(5):
            c0, c1 = max(0, c - 1), min(5, c + 2)
            out[:, r, c, :] = grid[:, r0:r1, c0:c1, :].mean(axis=(1, 2))
    return out.reshape(-1, 25, 512)


def _fit_cca_subspace(g_feats: np.ndarray, target_diffs: np.ndarray, rank: int = CCA_RANK) -> dict[str, np.ndarray]:
    if len(g_feats) < 2:
        return {
            'mu_g': np.zeros(512, dtype=np.float64),
            'mu_d': np.zeros(512, dtype=np.float64),
            'U': np.zeros((512, rank), dtype=np.float64),
            'Vt': np.zeros((rank, 512), dtype=np.float64),
        }
    mu_g = g_feats.mean(axis=0)
    mu_d = target_diffs.mean(axis=0)
    gc = g_feats - mu_g
    dc = target_diffs - mu_d
    cov = (gc.T @ dc) / float(len(g_feats))
    u, _s, vt = np.linalg.svd(cov, full_matrices=False)
    k = min(rank, u.shape[1], vt.shape[0])
    u_out = np.zeros((512, rank), dtype=np.float64)
    vt_out = np.zeros((rank, 512), dtype=np.float64)
    u_out[:, :k] = u[:, :k]
    vt_out[:k, :] = vt[:k, :]
    return {'mu_g': mu_g, 'mu_d': mu_d, 'U': u_out, 'Vt': vt_out}


def fit_visual_subspace(
    feats: dict[str, np.ndarray],
    train_idx: np.ndarray,
    train_cells: np.ndarray,
    *,
    rank: int = CCA_RANK,
) -> dict:
    """Fit full-tile and Center-9 PCA/CCA subspaces strictly from train_idx."""
    g_tr = feats['ground_cls_512'][train_idx].astype(np.float64)
    a25_tr = feats['aerial_full25_512'][train_idx].astype(np.float64)
    a25_mean = a25_tr.mean(axis=1)
    rows = np.arange(len(train_idx))
    a25_target = a25_tr[rows, train_cells] - a25_mean
    full_cca = _fit_cca_subspace(g_tr, a25_target, rank=rank)

    # Diagnostic principal axes on centered train ground and aerial embeddings
    _, _, vt_g = np.linalg.svd(g_tr - g_tr.mean(axis=0), full_matrices=False)
    a25_flat = a25_tr.reshape(-1, 512)
    _, _, vt_a = np.linalg.svd(a25_flat - a25_flat.mean(axis=0), full_matrices=False)

    # Center-9 cross-covariance subspace on train samples whose true cell falls in Center-9
    c9_mask = np.array([int(c) in CENTER9_CELL_TO_POS for c in train_cells], dtype=bool)
    if np.any(c9_mask):
        ac9_tr = feats['aerial_center9_512'][train_idx][c9_mask].astype(np.float64)
        ac9_pos = np.array([CENTER9_CELL_TO_POS[int(c)] for c in train_cells[c9_mask]], dtype=np.int64)
        ac9_diff = ac9_tr[np.arange(len(ac9_pos)), ac9_pos] - ac9_tr.mean(axis=1)
        c9_cca = _fit_cca_subspace(g_tr[c9_mask], ac9_diff, rank=rank)
    else:
        c9_cca = _fit_cca_subspace(np.empty((0, 512)), np.empty((0, 512)), rank=rank)

    return {
        'full_cca': full_cca,
        'c9_cca': c9_cca,
        'V_g_top16': vt_g[:rank].T,
        'V_a_top16': vt_a[:rank].T,
    }


def build_pair_cell_features(
    feats: dict[str, np.ndarray],
    subspace: dict,
    *,
    feature_mode: str = 'R2',
) -> np.ndarray:
    """Build (N, 25, D) visual feature tensor (D=27 for R1, D=49 for R2)."""
    g_cls = feats['ground_cls_512'].astype(np.float64)
    g_bands = feats['ground_bands_512'].astype(np.float64)
    a_cls = feats['aerial_cls_512'].astype(np.float64)
    a_25 = feats['aerial_full25_512'].astype(np.float64)
    n = g_cls.shape[0]

    a_mean = a_25.mean(axis=1, keepdims=True)
    a_cent = a_25 - a_mean
    a_loc = _local_3x3_aerial_mean(a_25)

    g_broad = np.broadcast_to(g_cls[:, None, :], (n, 25, 512))
    b0 = np.broadcast_to(g_bands[:, 0:1, :], (n, 25, 512))
    b1 = np.broadcast_to(g_bands[:, 1:2, :], (n, 25, 512))
    b2 = np.broadcast_to(g_bands[:, 2:3, :], (n, 25, 512))
    ac_broad = np.broadcast_to(a_cls[:, None, :], (n, 25, 512))

    base = np.stack(
        [
            _safe_cosine_last_dim(g_broad, a_25),
            _safe_cosine_last_dim(g_broad, a_cent),
            _safe_cosine_last_dim(g_broad, a_loc),
            _safe_cosine_last_dim(b0, a_25),
            _safe_cosine_last_dim(b1, a_25),
            _safe_cosine_last_dim(b2, a_25),
            _safe_cosine_last_dim(b1, a_cent),
            _safe_cosine_last_dim(b2, a_cent),
            np.linalg.norm(a_cent, axis=-1),
            _safe_cosine_last_dim(ac_broad, a_25),
        ],
        axis=-1,
    )

    fc = subspace['full_cca']
    gp = (g_cls - fc['mu_g']) @ fc['U']
    ap = (a_cent - fc['mu_d']) @ fc['Vt'].T
    cca_prod = gp[:, None, :] * ap
    cca_dot = cca_prod.sum(axis=-1, keepdims=True)
    x_r1 = np.concatenate([base, cca_prod, cca_dot], axis=-1)
    if feature_mode == 'R1':
        return x_r1

    a_c9 = feats['aerial_center9_512'].astype(np.float64)
    ac9_cent = a_c9 - a_c9.mean(axis=1, keepdims=True)
    g9 = np.broadcast_to(g_cls[:, None, :], (n, 9, 512))
    b9_0 = np.broadcast_to(g_bands[:, 0:1, :], (n, 9, 512))
    b9_1 = np.broadcast_to(g_bands[:, 1:2, :], (n, 9, 512))
    b9_2 = np.broadcast_to(g_bands[:, 2:3, :], (n, 9, 512))
    c9_base = np.stack(
        [
            _safe_cosine_last_dim(g9, a_c9),
            _safe_cosine_last_dim(g9, ac9_cent),
            _safe_cosine_last_dim(b9_0, ac9_cent),
            _safe_cosine_last_dim(b9_1, ac9_cent),
            _safe_cosine_last_dim(b9_2, ac9_cent),
        ],
        axis=-1,
    )
    cc = subspace['c9_cca']
    gp9 = (g_cls - cc['mu_g']) @ cc['U']
    ap9 = (ac9_cent - cc['mu_d']) @ cc['Vt'].T
    c9_prod = gp9[:, None, :] * ap9
    c9_dot = c9_prod.sum(axis=-1, keepdims=True)
    c9_block_9 = np.concatenate([c9_base, c9_prod, c9_dot], axis=-1)

    c9_block_25 = np.zeros((n, 25, c9_block_9.shape[-1]), dtype=np.float64)
    c9_block_25[:, CENTER9_CELLS, :] = c9_block_9
    return np.concatenate([x_r1, c9_block_25], axis=-1)


def _standardize_features_fit(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _softmax_25(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def _ranking_from_scores(scores: np.ndarray, *, center9_only: bool = False) -> list[int]:
    if center9_only:
        c9 = sorted(CENTER9_CELLS, key=lambda k: (-float(scores[k]), CELL_CENTER_DIST_TO_12_SQ[k], k))
        outer = sorted(set(range(25)) - set(CENTER9_CELLS), key=lambda k: (CELL_CENTER_DIST_TO_12_SQ[k], k))
        return c9 + outer
    return sorted(range(25), key=lambda k: (-float(scores[k]), CELL_CENTER_DIST_TO_12_SQ[k], k))


def _fit_spatial_readout_weights(
    x_tr: np.ndarray,
    q_tr: np.ndarray,
    base_log_prior: np.ndarray,
    top_indices: tuple[int, ...],
    *,
    lam: float,
    beta: float = 1.0,
    with_center9_heads: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    n, _, d = x_tr.shape
    d_top = len(top_indices)
    x_top = x_tr[:, CENTER9_CELLS, :][:, :, top_indices]
    dim = d + (9 * d_top if with_center9_heads else 0)

    def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w_shared = theta[:d]
        visual_logits = np.einsum('nkd,d->nk', x_tr, w_shared)
        if with_center9_heads:
            w_c9 = theta[d:].reshape(9, d_top)
            visual_logits = visual_logits.copy()
            visual_logits[:, CENTER9_CELLS] += np.einsum('mcd,cd->mc', x_top, w_c9)
        logits = base_log_prior[None, :] + beta * visual_logits
        probs = _softmax_25(logits)
        ce = -float(np.sum(q_tr * np.log(np.maximum(probs, 1e-12)))) / float(n)
        loss = ce + 0.5 * (lam / float(n)) * float(np.sum(theta * theta))
        diff = (probs - q_tr) / float(n)
        g_shared = beta * np.einsum('nk,nkd->d', diff, x_tr)
        if with_center9_heads:
            g_c9 = beta * np.einsum('mc,mcd->cd', diff[:, CENTER9_CELLS], x_top)
            grad = np.concatenate([g_shared, g_c9.reshape(-1)]) + (lam / float(n)) * theta
        else:
            grad = g_shared + (lam / float(n)) * theta
        return loss, grad

    res = minimize(loss_and_grad, np.zeros(dim, dtype=np.float64), jac=True, method='L-BFGS-B')
    theta = res.x
    w_shared = theta[:d]
    w_c9 = theta[d:].reshape(9, d_top) if with_center9_heads else None
    return w_shared, w_c9


def _predict_spatial_probs(
    x: np.ndarray,
    base_log_prior: np.ndarray,
    w_shared: np.ndarray,
    w_c9: np.ndarray | None,
    top_indices: tuple[int, ...],
    *,
    beta: float = 1.0,
) -> tuple[np.ndarray, list[list[int]]]:
    visual_logits = np.einsum('nkd,d->nk', x, w_shared)
    if w_c9 is not None:
        x_top = x[:, CENTER9_CELLS, :][:, :, top_indices]
        visual_logits = visual_logits.copy()
        visual_logits[:, CENTER9_CELLS] += np.einsum('mcd,cd->mc', x_top, w_c9)
    logits = base_log_prior[None, :] + beta * visual_logits
    probs = _softmax_25(logits)
    rankings = [_ranking_from_scores(logits[i]) for i in range(len(logits))]
    return probs, rankings


def _empirical_log_prior_from_cells(cells: np.ndarray, dirichlet_alpha: float = 0.5) -> np.ndarray:
    counts = np.bincount(cells, minlength=25).astype(np.float64)
    probs = (counts + dirichlet_alpha) / (float(len(cells)) + 25.0 * dirichlet_alpha)
    return np.log(np.maximum(probs, 1e-12))


def _stratified_5fold_splits(entries: list[dict], cells: np.ndarray, num_folds: int = 5) -> list[np.ndarray]:
    order = sorted(
        range(len(entries)),
        key=lambda i: (str(entries[i].get('macro_id')), int(cells[i]), str(entries[i]['image_id'])),
    )
    folds: list[list[int]] = [[] for _ in range(num_folds)]
    for pos, idx in enumerate(order):
        folds[pos % num_folds].append(idx)
    return [np.array(f, dtype=np.int64) for f in folds]


def _cv_loss_objective(entries: list[dict], probs: np.ndarray, rankings: list[list[int]]) -> float:
    probs_map = {e['image_id']: probs[i].tolist() for i, e in enumerate(entries)}
    rank_map = {e['image_id']: rankings[i] for i, e in enumerate(entries)}
    m = evaluate_single_step_predictions(entries, probs_map, rank_map)['all_pairs']
    return float(m['expected_planar_error_m'] + m['soft_target_kl_nats'])


def _cross_validate_spatial_hyperparams(
    fit_entries: list[dict],
    fit_feats: dict[str, np.ndarray],
    fit_cells: np.ndarray,
    q_fit: np.ndarray,
    *,
    feature_mode: str,
    top_indices: tuple[int, ...],
    with_prior: bool,
    with_center9_heads: bool,
) -> tuple[float, float, float]:
    folds = _stratified_5fold_splits(fit_entries, fit_cells, num_folds=5)
    n_fit = len(fit_entries)
    beta_candidates = BETA_GRID if with_prior else (1.0,)
    lam_grid = (0.2, 1.0, 5.0, 20.0) if with_prior else LAMBDA_GRID
    uniform_log_prior = np.log(np.full(25, 1.0 / 25.0, dtype=np.float64))

    fold_cache = []
    for val_idx in folds:
        val_mask = np.zeros(n_fit, dtype=bool)
        val_mask[val_idx] = True
        tr_idx = np.where(~val_mask)[0]
        sub = fit_visual_subspace(fit_feats, tr_idx, fit_cells[tr_idx])
        x_full = build_pair_cell_features(fit_feats, sub, feature_mode=feature_mode)
        mu, sd = _standardize_features_fit(x_full[tr_idx])
        x_std = _apply_standardize(x_full, mu, sd)
        prior = _empirical_log_prior_from_cells(fit_cells[tr_idx]) if with_prior else uniform_log_prior
        fold_cache.append((tr_idx, val_idx, x_std[tr_idx], q_fit[tr_idx], x_std[val_idx], prior))

    best_score = math.inf
    best_lam = lam_grid[-1]
    best_beta = beta_candidates[0]
    for lam in lam_grid:
        for beta in beta_candidates:
            oof_probs = np.zeros((n_fit, 25), dtype=np.float64)
            oof_ranks: list[list[int]] = [list(range(25))] * n_fit
            for _tr, val_idx, x_f_tr, q_f_tr, x_f_val, prior in fold_cache:
                w_s, w_c = _fit_spatial_readout_weights(
                    x_f_tr,
                    q_f_tr,
                    prior,
                    top_indices,
                    lam=lam,
                    beta=beta,
                    with_center9_heads=with_center9_heads,
                )
                p_val, r_val = _predict_spatial_probs(x_f_val, prior, w_s, w_c, top_indices, beta=beta)
                oof_probs[val_idx] = p_val
                for k_pos, orig_i in enumerate(val_idx):
                    oof_ranks[orig_i] = r_val[k_pos]
            score = _cv_loss_objective(fit_entries, oof_probs, oof_ranks)
            if score < best_score - 1e-12:
                best_score = score
                best_lam = lam
                best_beta = beta
    return float(best_lam), float(best_beta), float(best_score)


def evaluate_stage2_predictions_6strata(
    entries: list[dict],
    probs_by_id: dict[str, list[float]],
    rankings_by_id: dict[str, list[int]],
) -> dict[str, dict]:
    """Evaluate 25-cell predictions across the 4 CV-050 strata plus the 2 ground-size strata."""
    for e in entries:
        _assert_stage2_entry(e)
    base = evaluate_single_step_predictions(entries, probs_by_id, rankings_by_id)
    for size_key, wh in (('ground_size_952x714', (952, 714)), ('ground_size_952x532', (952, 532))):
        sub = [
            e for e in entries
            if (int(e['ground_asset']['width']), int(e['ground_asset']['height'])) == wh
        ]
        if not sub:
            base[size_key] = {'count': 0}
        else:
            base[size_key] = evaluate_single_step_predictions(sub, probs_by_id, rankings_by_id)['all_pairs']
    return base


def _predict_ablation_variant(
    feats_split: dict[str, np.ndarray],
    subspace: dict,
    mu: np.ndarray,
    sd: np.ndarray,
    model_cfg: dict,
    fit_feats: dict[str, np.ndarray],
    control_kind: str,
) -> tuple[np.ndarray, list[list[int]]]:
    n = feats_split['ground_cls_512'].shape[0]
    mod_feats = {k: np.array(v, copy=True) for k, v in feats_split.items()}
    if control_kind == 'Control_GroundZeroed':
        mod_feats['ground_cls_512'][:] = 0.0
        mod_feats['ground_bands_512'][:] = 0.0
    elif control_kind == 'Control_AerialShuffled':
        perm = np.array([(i + 37) % n for i in range(n)], dtype=np.int64)
        for akey in ('aerial_cls_512', 'aerial_full25_512', 'aerial_center_cls_512', 'aerial_center9_512'):
            mod_feats[akey] = mod_feats[akey][perm]
    elif control_kind == 'Control_MeanVisual':
        for fkey in (
            'ground_cls_512',
            'ground_bands_512',
            'aerial_cls_512',
            'aerial_full25_512',
            'aerial_center_cls_512',
            'aerial_center9_512',
        ):
            mean_val = fit_feats[fkey].mean(axis=0, keepdims=True)
            mod_feats[fkey][:] = np.broadcast_to(mean_val, mod_feats[fkey].shape)
    else:
        raise ValueError(f'Unknown control_kind: {control_kind}')

    x_mod = _apply_standardize(
        build_pair_cell_features(mod_feats, subspace, feature_mode=model_cfg['feature_mode']),
        mu,
        sd,
    )
    return _predict_spatial_probs(
        x_mod,
        model_cfg['prior'],
        model_cfg['w_shared'],
        model_cfg['w_c9'],
        model_cfg['top_indices'],
        beta=model_cfg['beta'],
    )


def evaluate_stage2_gates(
    val_metrics_by_model: dict[str, dict[str, dict]],
    candidate_name: str,
    control_names: Sequence[str] = ('Control_GroundZeroed', 'Control_AerialShuffled', 'Control_MeanVisual'),
) -> dict:
    """Evaluate Pass Gates G1-G4 on development_validation for the candidate visual model."""
    cand = val_metrics_by_model[candidate_name]
    c12 = val_metrics_by_model['AlwaysCell12Prior']
    emp = val_metrics_by_model['TrainEmpiricalPrior']

    g1 = bool(
        cand['all_pairs']['top1_acc'] > c12['all_pairs']['top1_acc'] + 1e-12
        and cand['opensfm_cell_agree']['top1_acc'] > c12['opensfm_cell_agree']['top1_acc'] + 1e-12
        and cand['interior_margin_ge_2m']['top1_acc'] > c12['interior_margin_ge_2m']['top1_acc'] + 1e-12
    )
    g2 = bool(
        cand['all_pairs']['soft_target_kl_nats'] < emp['all_pairs']['soft_target_kl_nats'] - 1e-12
        and cand['all_pairs']['expected_planar_error_m'] < c12['all_pairs']['expected_planar_error_m'] - 1e-12
        and cand['all_pairs']['top3_acc'] > c12['all_pairs']['top3_acc'] + 1e-12
    )
    s714 = cand.get('ground_size_952x714', {})
    s532 = cand.get('ground_size_952x532', {})
    g3_checks = []
    for skey, s_cand in (('ground_size_952x714', s714), ('ground_size_952x532', s532)):
        if s_cand.get('count', 0) == 0:
            continue
        s_c12 = c12[skey]
        s_emp = emp[skey]
        g3_checks.append(
            s_cand['top1_acc'] >= s_c12['top1_acc'] - 0.05
            and s_cand['soft_target_kl_nats'] <= s_emp['soft_target_kl_nats'] + 0.05
        )
    g3 = bool(g3_checks) and all(g3_checks)

    g4_details = {}
    g4_all = True
    for cname in control_names:
        ctrl = val_metrics_by_model[cname]['all_pairs']
        d_top1 = cand['all_pairs']['top1_acc'] - ctrl['top1_acc']
        d_err_drop = ctrl['expected_planar_error_m'] - cand['all_pairs']['expected_planar_error_m']
        passed_ctrl = bool(d_top1 >= 0.03 - 1e-12 or d_err_drop >= 0.5 - 1e-12)
        g4_all = g4_all and passed_ctrl
        g4_details[cname] = {
            'top1_gain': float(d_top1),
            'planar_error_reduction_m': float(d_err_drop),
            'passed': passed_ctrl,
        }

    overall = bool(g1 and g2 and g3 and g4_all)
    return {
        'candidate_model': candidate_name,
        'G1_top1_beats_AlwaysCell12': g1,
        'G2_distributional_and_error_advantage': g2,
        'G3_aspect_ratio_no_collapse': g3,
        'G4_visual_ablation_gain': g4_all,
        'g4_by_control': g4_details,
        'stage2_passed': overall,
        'decision': (
            'PROMOTE_TO_STAGE3_BLIND_CONFIRMATION'
            if overall
            else 'STOP_AT_STAGE2_NEGATIVE_AND_PROCEED_TO_STAGE4_CLOSURE'
        ),
    }


def _zero_shot_probs_and_ranks(scores: np.ndarray, *, center9_only: bool = False, eps: float = 1e-3) -> tuple[np.ndarray, list[list[int]]]:
    n = scores.shape[0]
    if not center9_only:
        probs = _softmax_25(10.0 * scores)
        ranks = [_ranking_from_scores(scores[i], center9_only=False) for i in range(n)]
        return probs, ranks
    c9_scores = scores[:, CENTER9_CELLS]
    c9_probs = _softmax_25(10.0 * c9_scores)
    probs = np.full((n, 25), eps / 25.0, dtype=np.float64)
    probs[:, CENTER9_CELLS] += (1.0 - eps) * c9_probs
    ranks = [_ranking_from_scores(scores[i], center9_only=True) for i in range(n)]
    return probs, ranks


def fit_and_evaluate_stage2_models(
    fit_train_entries: list[dict],
    dev_val_entries: list[dict],
    feats: dict[str, np.ndarray],
) -> dict:
    """Fit all Stage-2 visual models strictly on fit_train and evaluate 6 strata on fit_train & dev_val."""
    if not fit_train_entries or not dev_val_entries:
        raise ValueError('Non-empty fit_train and dev_val entries required')
    for e in fit_train_entries:
        _assert_stage2_entry(e, expected_role='development_fit_train')
    for e in dev_val_entries:
        _assert_stage2_entry(e, expected_role='development_validation')

    id_to_feat_idx = {str(sid): idx for idx, sid in enumerate(feats['image_ids'])}
    tr_idx = np.array([id_to_feat_idx[str(e['image_id'])] for e in fit_train_entries], dtype=np.int64)
    va_idx = np.array([id_to_feat_idx[str(e['image_id'])] for e in dev_val_entries], dtype=np.int64)
    if set(tr_idx.tolist()) & set(va_idx.tolist()):
        raise ValueError('Train/validation feature index overlap detected')

    fit_feats = {
        k: v[tr_idx] for k, v in feats.items()
        if k in {'ground_cls_512', 'ground_bands_512', 'aerial_cls_512', 'aerial_full25_512', 'aerial_center_cls_512', 'aerial_center9_512'}
    }
    val_feats = {
        k: v[va_idx] for k, v in feats.items()
        if k in {'ground_cls_512', 'ground_bands_512', 'aerial_cls_512', 'aerial_full25_512', 'aerial_center_cls_512', 'aerial_center9_512'}
    }

    fit_cells = np.array([int(e['mast3rpnp']['index']) for e in fit_train_entries], dtype=np.int64)
    q_fit = np.array(
        [
            gaussian_soft_target_distribution(
                float(e['mast3rpnp']['x']), float(e['mast3rpnp']['y']), resolution=float(e['resolution'])
            )
            for e in fit_train_entries
        ],
        dtype=np.float64,
    )

    subspace = fit_visual_subspace(fit_feats, np.arange(len(fit_train_entries)), fit_cells)
    x_tr_r2 = build_pair_cell_features(fit_feats, subspace, feature_mode='R2')
    x_va_r2 = build_pair_cell_features(val_feats, subspace, feature_mode='R2')
    mu_r2, sd_r2 = _standardize_features_fit(x_tr_r2)
    x_tr_r2_std = _apply_standardize(x_tr_r2, mu_r2, sd_r2)
    x_va_r2_std = _apply_standardize(x_va_r2, mu_r2, sd_r2)
    x_tr_r1_std = x_tr_r2_std[:, :, :27]
    x_va_r1_std = x_va_r2_std[:, :, :27]
    mu_r1, sd_r1 = mu_r2[:27], sd_r2[:27]

    uniform_log_prior = np.log(np.full(25, 1.0 / 25.0, dtype=np.float64))
    emp_log_prior = _empirical_log_prior_from_cells(fit_cells)

    # Zero-shot cosine models
    m1_scores_tr = np.einsum('nd,nkd->nk', fit_feats['ground_cls_512'], fit_feats['aerial_full25_512'])
    m1_scores_va = np.einsum('nd,nkd->nk', val_feats['ground_cls_512'], val_feats['aerial_full25_512'])
    m1_p_tr, m1_r_tr = _zero_shot_probs_and_ranks(m1_scores_tr, center9_only=False)
    m1_p_va, m1_r_va = _zero_shot_probs_and_ranks(m1_scores_va, center9_only=False)

    m1b_scores_tr = np.array(m1_scores_tr, copy=True)
    m1b_scores_va = np.array(m1_scores_va, copy=True)
    c9_tr = np.einsum('nd,nkd->nk', fit_feats['ground_cls_512'], fit_feats['aerial_center9_512'])
    c9_va = np.einsum('nd,nkd->nk', val_feats['ground_cls_512'], val_feats['aerial_center9_512'])
    m1b_scores_tr[:, CENTER9_CELLS] = 0.5 * (m1b_scores_tr[:, CENTER9_CELLS] + c9_tr)
    m1b_scores_va[:, CENTER9_CELLS] = 0.5 * (m1b_scores_va[:, CENTER9_CELLS] + c9_va)
    m1b_p_tr, m1b_r_tr = _zero_shot_probs_and_ranks(m1b_scores_tr, center9_only=True)
    m1b_p_va, m1b_r_va = _zero_shot_probs_and_ranks(m1b_scores_va, center9_only=True)

    # CV hyperparameter selection strictly inside development_fit_train
    lam_m2a, beta_m2a, cv_m2a = _cross_validate_spatial_hyperparams(
        fit_train_entries, fit_feats, fit_cells, q_fit,
        feature_mode='R1', top_indices=TOP_LOCAL_INDICES_R1, with_prior=False, with_center9_heads=False,
    )
    lam_m2b, beta_m2b, cv_m2b = _cross_validate_spatial_hyperparams(
        fit_train_entries, fit_feats, fit_cells, q_fit,
        feature_mode='R1', top_indices=TOP_LOCAL_INDICES_R1, with_prior=True, with_center9_heads=True,
    )
    lam_m2c, beta_m2c, cv_m2c = _cross_validate_spatial_hyperparams(
        fit_train_entries, fit_feats, fit_cells, q_fit,
        feature_mode='R2', top_indices=TOP_LOCAL_INDICES_R2, with_prior=True, with_center9_heads=True,
    )

    w2a_s, w2a_c = _fit_spatial_readout_weights(
        x_tr_r1_std, q_fit, uniform_log_prior, TOP_LOCAL_INDICES_R1, lam=lam_m2a, beta=beta_m2a, with_center9_heads=False
    )
    m2a_p_tr, m2a_r_tr = _predict_spatial_probs(x_tr_r1_std, uniform_log_prior, w2a_s, w2a_c, TOP_LOCAL_INDICES_R1, beta=beta_m2a)
    m2a_p_va, m2a_r_va = _predict_spatial_probs(x_va_r1_std, uniform_log_prior, w2a_s, w2a_c, TOP_LOCAL_INDICES_R1, beta=beta_m2a)

    w2b_s, w2b_c = _fit_spatial_readout_weights(
        x_tr_r1_std, q_fit, emp_log_prior, TOP_LOCAL_INDICES_R1, lam=lam_m2b, beta=beta_m2b, with_center9_heads=True
    )
    m2b_p_tr, m2b_r_tr = _predict_spatial_probs(x_tr_r1_std, emp_log_prior, w2b_s, w2b_c, TOP_LOCAL_INDICES_R1, beta=beta_m2b)
    m2b_p_va, m2b_r_va = _predict_spatial_probs(x_va_r1_std, emp_log_prior, w2b_s, w2b_c, TOP_LOCAL_INDICES_R1, beta=beta_m2b)

    w2c_s, w2c_c = _fit_spatial_readout_weights(
        x_tr_r2_std, q_fit, emp_log_prior, TOP_LOCAL_INDICES_R2, lam=lam_m2c, beta=beta_m2c, with_center9_heads=True
    )
    m2c_p_tr, m2c_r_tr = _predict_spatial_probs(x_tr_r2_std, emp_log_prior, w2c_s, w2c_c, TOP_LOCAL_INDICES_R2, beta=beta_m2c)
    m2c_p_va, m2c_r_va = _predict_spatial_probs(x_va_r2_std, emp_log_prior, w2c_s, w2c_c, TOP_LOCAL_INDICES_R2, beta=beta_m2c)

    trainable_specs = {
        'M2a_SpatialHead_PureVisual_NoPrior': {
            'cv_score': cv_m2a, 'lam': lam_m2a, 'beta': beta_m2a, 'feature_mode': 'R1',
            'mu': mu_r1, 'sd': sd_r1, 'prior': uniform_log_prior, 'w_shared': w2a_s, 'w_c9': w2a_c,
            'top_indices': TOP_LOCAL_INDICES_R1,
        },
        'M2b_SpatialHead_PriorCalibrated_Full': {
            'cv_score': cv_m2b, 'lam': lam_m2b, 'beta': beta_m2b, 'feature_mode': 'R1',
            'mu': mu_r1, 'sd': sd_r1, 'prior': emp_log_prior, 'w_shared': w2b_s, 'w_c9': w2b_c,
            'top_indices': TOP_LOCAL_INDICES_R1,
        },
        'M2c_MultiScale_PriorCalibrated_CV051b': {
            'cv_score': cv_m2c, 'lam': lam_m2c, 'beta': beta_m2c, 'feature_mode': 'R2',
            'mu': mu_r2, 'sd': sd_r2, 'prior': emp_log_prior, 'w_shared': w2c_s, 'w_c9': w2c_c,
            'top_indices': TOP_LOCAL_INDICES_R2,
        },
    }
    best_cv_model = min(trainable_specs, key=lambda k: trainable_specs[k]['cv_score'])
    ref_spec = trainable_specs[best_cv_model]

    predictions = {
        'M1_ZeroShot_FullTile_Cosine': (m1_p_tr, m1_r_tr, m1_p_va, m1_r_va, {'temperature': 10.0, 'feature_mode': 'R1'}),
        'M1b_ZeroShot_Center9Crop_Cosine': (m1b_p_tr, m1b_r_tr, m1b_p_va, m1b_r_va, {'temperature': 10.0, 'feature_mode': 'R2'}),
        'M2a_SpatialHead_PureVisual_NoPrior': (m2a_p_tr, m2a_r_tr, m2a_p_va, m2a_r_va, {'lam': lam_m2a, 'beta': beta_m2a, 'cv_objective': cv_m2a, 'feature_mode': 'R1'}),
        'M2b_SpatialHead_PriorCalibrated_Full': (m2b_p_tr, m2b_r_tr, m2b_p_va, m2b_r_va, {'lam': lam_m2b, 'beta': beta_m2b, 'cv_objective': cv_m2b, 'feature_mode': 'R1'}),
        'M2c_MultiScale_PriorCalibrated_CV051b': (m2c_p_tr, m2c_r_tr, m2c_p_va, m2c_r_va, {'lam': lam_m2c, 'beta': beta_m2c, 'cv_objective': cv_m2c, 'feature_mode': 'R2'}),
    }
    for cname in ('Control_GroundZeroed', 'Control_AerialShuffled', 'Control_MeanVisual'):
        cp_tr, cr_tr = _predict_ablation_variant(fit_feats, subspace, ref_spec['mu'], ref_spec['sd'], ref_spec, fit_feats, cname)
        cp_va, cr_va = _predict_ablation_variant(val_feats, subspace, ref_spec['mu'], ref_spec['sd'], ref_spec, fit_feats, cname)
        predictions[cname] = (cp_tr, cr_tr, cp_va, cr_va, {'reference_model': best_cv_model, 'control_kind': cname})

    # Include the 4 zero-image statistical priors on all 6 strata for direct comparison
    stat_baselines = build_statistical_baselines(fit_train_entries)
    stat_tr_ev = evaluate_baseline_suite(fit_train_entries, stat_baselines)
    stat_va_ev = evaluate_baseline_suite(dev_val_entries, stat_baselines)

    eval_tr: dict[str, dict] = {}
    eval_va: dict[str, dict] = {}
    for bname, spec in stat_baselines.items():
        p_tr_map = {e['image_id']: list(spec['probs']) for e in fit_train_entries}
        r_tr_map = {e['image_id']: list(spec['ranking']) for e in fit_train_entries}
        p_va_map = {e['image_id']: list(spec['probs']) for e in dev_val_entries}
        r_va_map = {e['image_id']: list(spec['ranking']) for e in dev_val_entries}
        eval_tr[bname] = evaluate_stage2_predictions_6strata(fit_train_entries, p_tr_map, r_tr_map)
        eval_va[bname] = evaluate_stage2_predictions_6strata(dev_val_entries, p_va_map, r_va_map)

    saved_predictions = {'development_fit_train': {}, 'development_validation': {}}
    model_hyperparams = {}
    for mname, (p_tr, r_tr, p_va, r_va, meta) in predictions.items():
        p_tr_map = {e['image_id']: p_tr[i].tolist() for i, e in enumerate(fit_train_entries)}
        r_tr_map = {e['image_id']: r_tr[i] for i, e in enumerate(fit_train_entries)}
        p_va_map = {e['image_id']: p_va[i].tolist() for i, e in enumerate(dev_val_entries)}
        r_va_map = {e['image_id']: r_va[i] for i, e in enumerate(dev_val_entries)}
        eval_tr[mname] = evaluate_stage2_predictions_6strata(fit_train_entries, p_tr_map, r_tr_map)
        eval_va[mname] = evaluate_stage2_predictions_6strata(dev_val_entries, p_va_map, r_va_map)
        model_hyperparams[mname] = meta
        saved_predictions['development_fit_train'][mname] = {'probs': p_tr_map, 'rankings': r_tr_map}
        saved_predictions['development_validation'][mname] = {'probs': p_va_map, 'rankings': r_va_map}

    gate_by_candidate = {
        cname: evaluate_stage2_gates(eval_va, cname)
        for cname in ('M2c_MultiScale_PriorCalibrated_CV051b', 'M2b_SpatialHead_PriorCalibrated_Full', 'M2a_SpatialHead_PureVisual_NoPrior', 'M1b_ZeroShot_Center9Crop_Cosine', 'M1_ZeroShot_FullTile_Cosine')
    }
    primary_gate = gate_by_candidate[best_cv_model]

    return {
        'cv_selected_reference_model': best_cv_model,
        'model_hyperparams': model_hyperparams,
        'statistical_baselines_cv050_check': {
            'fit_train_always_cell12': stat_tr_ev['AlwaysCell12Prior']['metrics']['all_pairs']['top1_acc'],
            'dev_val_always_cell12': stat_va_ev['AlwaysCell12Prior']['metrics']['all_pairs']['top1_acc'],
        },
        'evaluations': {
            'development_fit_train': eval_tr,
            'development_validation': eval_va,
        },
        'gate_evaluations': gate_by_candidate,
        'primary_stage2_gate': primary_gate,
        'predictions': saved_predictions,
    }
