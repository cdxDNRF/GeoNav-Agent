"""User-upload engineering sandbox: cut one image into a native grid for live testing.

Uploads are session-local evidence under 平台运行/uploads; they never touch the
frozen dev region, the sealed encoding cache or any registered research source.
Sessions created from them are labelled user_upload_engineering and produce no SR claims.
"""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_PREFIX = 'geonav-upload-'
GRIDS = (5, 10)
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MIN_CELL_SOURCE_PX = 60
AREA_BASE = 900000
RESAMPLE = 'LANCZOS'
ALLOWED_FORMATS = ('PNG', 'JPEG', 'WEBP')


def create_sandbox():
    """Ephemeral per-process storage: uploads, cut tiles, features and session logs."""
    return Path(tempfile.mkdtemp(prefix=SANDBOX_PREFIX))


def purge_stale_sandboxes(keep=None, max_age_seconds=3600):
    """Remove crash leftovers from previous runs; never touch recent or live ones."""
    now = time.time()
    removed = []
    for path in Path(tempfile.gettempdir()).glob(SANDBOX_PREFIX + '*'):
        if keep is not None and path.resolve() == Path(keep).resolve():
            continue
        try:
            if now - path.stat().st_mtime > max_age_seconds:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path.name)
        except OSError:
            pass
    return removed


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(data):
    return sha256(data).hexdigest()


def sha_file(path):
    with Path(path).open('rb') as f:
        import hashlib
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)


def configure_torch():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def ensure_src_path(root):
    import sys
    src = str(Path(root) / 'project/src')
    if src not in sys.path:
        sys.path.insert(0, src)


_ENCODER = {}
_ENCODER_LOCK = threading.Lock()


def encoder_model():
    with _ENCODER_LOCK:
        if 'model' not in _ENCODER:
            import torch
            from transformers import CLIPVisionModelWithProjection
            ensure_src_path(ROOT)
            from data.process_masa import MODEL_DIR
            configure_torch()
            model = CLIPVisionModelWithProjection.from_pretrained(str(MODEL_DIR), local_files_only=True).eval()
            model.requires_grad_(False)
            if any(p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError('冻结编码器必须为float32')
            _ENCODER['model'] = model
        return _ENCODER['model']


def encode_tiles(paths):
    """Same frozen pipeline as the sealed native225 extraction; batch 16."""
    import torch
    ensure_src_path(ROOT)
    from data.process_masa import preprocess_patch
    from agents.spatial_relation import quadrant_features
    model = encoder_model()
    device = next(model.parameters()).device
    gs, ls = [], []
    with torch.inference_mode():
        for start in range(0, len(paths), 16):
            batch = [preprocess_patch(p) for p in paths[start:start + 16]]
            z = model(torch.stack(batch).to(device))
            gs.append(z.image_embeds.float().cpu().numpy())
            ls.append(quadrant_features(model.vision_model.post_layernorm(z.last_hidden_state[:, 1:])).float().cpu().numpy())
    g, l = np.concatenate(gs), np.concatenate(ls)
    if g.shape != (len(paths), 512) or l.shape != (len(paths), 4, 768):
        raise ValueError('编码器输出形状异常')
    if not np.isfinite(g).all() or not np.isfinite(l).all():
        raise ValueError('编码器输出存在非有限值')
    return g.astype(np.float32), l.astype(np.float32)


def cut_image(data, k):
    """Center-crop to square, resample the mosaic to 600k and slice exact native tiles."""
    if k not in GRIDS:
        raise ValueError('网格只支持5或10')
    if not isinstance(data, bytes) or not data:
        raise ValueError('需要二进制图片内容')
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('上传图片超过32MB上限')
    try:
        with Image.open(BytesIO(data)) as im:
            if (im.format or '').upper() not in ALLOWED_FORMATS:
                raise ValueError('仅支持PNG/JPEG/WEBP格式')
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            width, height = rgb.size
            if min(width, height) < k * MIN_CELL_SOURCE_PX:
                raise ValueError(f'图片过小：每格至少需要{MIN_CELL_SOURCE_PX}源像素')
            side = min(width, height)
            left, top = (width - side) // 2, (height - side) // 2
            square = rgb.crop((left, top, left + side, top + side)).resize((600 * k, 600 * k), Image.Resampling.LANCZOS)
            tiles = []
            for j in range(k * k):
                r, c = divmod(j, k)
                cell = square.crop((c * 600, r * 600, (c + 1) * 600, (r + 1) * 600))
                buf = BytesIO()
                cell.save(buf, format='PNG')
                tiles.append(buf.getvalue())
            preview = square.resize((768, 768), Image.Resampling.LANCZOS)
            pbuf = BytesIO()
            preview.save(pbuf, format='PNG')
    except ValueError:
        raise
    except Exception:
        raise ValueError('图片解码失败，请使用PNG/JPEG/WEBP') from None
    return tiles, [width, height], pbuf.getvalue()


def default_tasks(k, area, rng_seed):
    """Deterministic per-image starter queue across short/middle/long strata."""
    ensure_src_path(ROOT)
    from env.scaled_grid import distance
    from env.massgis_native_area_v2 import native_spec
    from env.massgis_native_area_v2 import NativeEpisode
    spec = native_spec(k)
    strata = {5: ((2, 4, 'short', 5), (5, 6, 'middle', 4), (7, 8, 'long', 3)),
              10: ((2, 7, 'short', 8), (8, 12, 'middle', 7), (13, 18, 'long', 5))}[k]
    rng = np.random.default_rng(rng_seed)
    tasks, used = [], set()
    for lo, hi, stratum, count in strata:
        for _ in range(count):
            d = int(rng.integers(lo, hi + 1))
            pairs = [(a, b) for a in range(k * k) for b in range(k * k)
                     if distance(a, b, k) == d and (a, b) not in used]
            if not pairs:
                raise ValueError('采样不可行')
            a, b = pairs[int(rng.integers(len(pairs)))]
            used.add((a, b))
            episode = NativeEpisode(f'upload_g{k}_{len(tasks)}', 'dev', area, a, b, d,
                                    spec.budget, k, spec.name, 'user-upload')
            episode.validate()
            tasks.append(dict(episode_id=episode.episode_id, split='dev', area=area, start=a, goal=b,
                              dist=d, budget=spec.budget, grid_size=k, protocol=spec.name,
                              source_tile='user-upload', stratum=stratum))
    return tasks


def register_upload(sandbox, data, k, name, area, encoder=None):
    """Materialize one upload as a self-contained native env inside the sandbox."""
    if k not in GRIDS:
        raise ValueError('网格只支持5或10')
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise ValueError('名称需为1-80字符')
    if not re.fullmatch(r'img_\d+', area or ''):
        raise ValueError('区域标识非法')
    name = name.strip()
    ensure_src_path(ROOT)
    tiles, source_size, preview = cut_image(data, k)
    from env.massgis_native_area_v2 import native_spec
    spec = native_spec(k)
    upload_id = uuid.uuid4().hex[:12]
    directory = Path(sandbox) / 'uploads' / upload_id
    grid_dir = directory / '工程数据' / f'grid{k}'
    patch_dir = grid_dir / 'patches' / 'dev' / area
    patch_dir.mkdir(parents=True)
    meta = dict(upload_id=upload_id, name=name, grid=k, budget=spec.budget, area=area,
        source_tile='user-upload', original_sha256=sha_bytes(data), original_bytes=len(data),
        source_size=source_size, resample=RESAMPLE, utc=stamp(), mode='user_upload_engineering',
        known_geography=False, encoder='Sat2Cap frozen (local_policy_default lineage)')
    write_new(directory / '上传元数据.json', meta)
    (directory / '原始上传.bin').write_bytes(data)
    (directory / '预览.png').write_bytes(preview)
    tile_sha = {}
    for j, tile in enumerate(tiles):
        path = patch_dir / f'patch_{j}.png'
        path.write_bytes(tile)
        tile_sha[f'uploads/{upload_id}/工程数据/grid{k}/patches/dev/{area}/patch_{j}.png'] = sha_bytes(tile)
    manifest = dict(protocol=spec.name, grid_size=k, budget=spec.budget, cell_size_m=300,
        native_cell_pixels=600, epsg=26986, pixel_size_m=.5, mosaic_pixels=600 * k,
        patch_format='png', development_only=True, known_geography=False, origin='user_upload',
        regions=[dict(split='dev', area=area, source_tile='user-upload')])
    write_new(grid_dir / '数据清单.json', manifest)
    paths = [patch_dir / f'patch_{j}.png' for j in range(k * k)]
    g, l = (encoder(paths) if encoder is not None else encode_tiles(paths))
    npz = directory / '特征.npz'
    with npz.open('xb') as f:
        np.savez_compressed(f, global_features=g, local_features=l)
    write_new(directory / '特征完成.json', dict(sha256=sha_file(npz),
        shapes=[list(g.shape), list(l.shape)], utc=stamp(), resample=RESAMPLE, float32=True))
    tasks = default_tasks(k, area, int(sha_bytes(data)[:8], 16))
    write_new(directory / '任务.json', tasks)
    return dict(meta, tile_sha256=tile_sha, feature_sha256=read(directory / '特征完成.json')['sha256'],
        tasks=tasks, directory=str(directory))
