"""GeoNav Lite workbench server: clean, lightweight, single-screen interactive demo.

Serves an intuitive single-page dashboard designed for frictionless presentations.
Supports curated multi-source aerial imagery across 192 distinct satellite maps
(<=5 M0-verified start/goal tasks per image: Masa S2, Masa S3, SwissView batches,
MassGIS 5x5 & 10x10, and Masa extended pool), all 192 OpenCVL ground tasks,
custom start/goal selection, volatile in-memory user image uploads (multi-grid n*n),
observer-only full-map preview, local M0 model, human teleoperation, and
OpenAI-compatible cloud API policies with proxy & error diagnostics.
"""
from __future__ import annotations
import argparse
import base64
from collections import OrderedDict
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
from pathlib import Path
import secrets
import socket
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'project/src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'project/src'))

from platform_core.controller_hub_v1 import CAMERA
from webapp.cloud_v3 import CloudConfig, SYSTEM as CLOUD_SYSTEM, payload_for, parse as cloud_parse

STATIC_DIR = Path(__file__).resolve().parent
CURATED_INDEX_PATH = STATIC_DIR / 'lite_curated_tasks.json'

ACTION_LIST = ('up', 'right', 'down', 'left')
ACTION_NAMES = {'up': '向上 ↑', 'right': '向右 →', 'down': '向下 ↓', 'left': '向左 ←'}
ACTION_OFFSETS = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}

# M0 支持且表现良好的多尺度 n×n 切片规格（自定义上传场景仅存在内存，关闭即清除）
UPLOAD_GRID_SPECS = {
    3: {'budget': 6, 'note': '3×3 (9格 · 极速轻量演示 · 预算6步)'},
    4: {'budget': 8, 'note': '4×4 (16格 · 紧凑空间探索 · 预算8步)'},
    5: {'budget': 10, 'note': '5×5 (25格 · M0 原生最佳协议 · 预算10步)'},
    6: {'budget': 14, 'note': '6×6 (36格 · 中等粒度探索 · 预算14步)'},
    7: {'budget': 16, 'note': '7×7 (49格 · 扩展视野尺度 · 预算16步)'},
    10: {'budget': 20, 'note': '10×10 (100格 · 大区域径向覆盖 · 预算20步)'},
}


def _detect_local_proxy():
    """Probe standard local Clash/proxy ports without mutating OS settings."""
    for port in (7897, 7890, 10808, 10809):
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.15):
                return f'http://127.0.0.1:{port}'
        except OSError:
            continue
    return None


def _explain_cloud_exception(err, http_status=None, raw_body='', usage=None):
    """Translate HTTP/network/parse exceptions into crystal-clear Chinese diagnostics."""
    import httpx
    status = http_status
    if status is None and isinstance(err, httpx.HTTPStatusError):
        status = err.response.status_code
        try:
            raw_body = err.response.text[:240]
        except Exception:
            pass

    if status == 402:
        return 'HTTP 402 账户余额/免费额度已耗尽或需付费，当前服务商拒绝响应（提示词与参数无误）'
    if status in (401, 403):
        return f'HTTP {status} API Key 无效、已过期或无权调用该模型'
    if status == 404:
        return 'HTTP 404 接口地址或模型名称不存在（请检查 Base URL 是否包含 /v1 及 Model 拼写）'
    if status == 429:
        return 'HTTP 429 请求频次或并发触发服务商限流，请稍后再试'
    if status is not None and status >= 500:
        return f'HTTP {status} 云端模型服务端内部异常或暂时不可用'
    if status is not None:
        return f'HTTP {status} 请求被拒绝：{raw_body or type(err).__name__}'

    msg = str(err)
    if isinstance(err, (httpx.ConnectTimeout, httpx.ReadTimeout)):
        return '请求超时（云端推理或网络响应耗时超过设定超时时间）'
    if isinstance(err, httpx.ConnectError):
        return '网络连接失败（直连与本机代理均无法连通接口地址，请检查网络或代理状态）'
    if msg == 'truncated':
        usage_note = ''
        if isinstance(usage, dict) and usage:
            ct = usage.get('completion_tokens')
            rt = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
            if rt is not None and ct is not None:
                usage_note = f' [本次已用 thinking={rt}, completion={ct}]'
            elif ct is not None:
                usage_note = f' [本次已用 completion={ct}]'
        return f'模型输出被 max_tokens 截断（建议勾选“关闭长思考输出”或增大预算）{usage_note}'
    if msg in ('schema_or_illegal_action', 'duplicate_key', 'missing_text', 'invalid_evidence') or 'JSONDecode' in type(err).__name__:
        return f'模型返回内容未按 JSON 格式规范输出（原因: {msg or type(err).__name__}）'
    return f'{type(err).__name__}: {msg[:160]}'


def _is_deepseek_config(config: CloudConfig) -> bool:
    url = (config.base_url or '').lower()
    mod = (config.model or '').lower()
    return 'deepseek' in url or 'deepseek' in mod


def _prepare_cloud_payload(base_payload: dict, config: CloudConfig, *, disable_thinking: bool = True, max_tokens_override: int | None = None) -> dict:
    """Adapt OpenAI-compatible payload for thinking/reasoning VLMs (e.g., DeepSeek-V4.1-Flash)."""
    payload = dict(base_payload)
    if max_tokens_override is not None:
        payload['max_tokens'] = int(max_tokens_override)
    elif disable_thinking:
        payload['max_tokens'] = max(int(payload.get('max_tokens', 512)), 2048 if _is_deepseek_config(config) else 4096)
    else:
        payload['max_tokens'] = max(int(payload.get('max_tokens', 512)), 8192)

    if disable_thinking and _is_deepseek_config(config):
        payload['thinking'] = {'type': 'disabled'}
    return payload


def lite_cloud_request(config: CloudConfig, payload: dict):
    """Send OpenAI-compatible request; fall back to local Clash proxy ONLY on connect failure."""
    import httpx

    headers = {'Content-Type': 'application/json'}
    if config.api_key:
        headers['Authorization'] = 'Bearer ' + config.api_key
    url = config.base_url.rstrip('/') + '/chat/completions'

    try:
        with httpx.Client(timeout=config.timeout, follow_redirects=False, trust_env=False) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json(), None
    except (httpx.ConnectError, httpx.ConnectTimeout) as direct_err:
        proxy_url = _detect_local_proxy()
        if not proxy_url:
            raise direct_err
        with httpx.Client(timeout=config.timeout, follow_redirects=False, proxy=proxy_url, trust_env=False) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json(), proxy_url


def _execute_cloud_with_adaptation(config: CloudConfig, base_payload: dict, *, disable_thinking: bool = True, max_tokens_override: int | None = None, parser=None):
    """Execute request with automatic compatibility fallback for thinking / max_tokens parameters."""
    import httpx

    payload = _prepare_cloud_payload(
        base_payload, config, disable_thinking=disable_thinking, max_tokens_override=max_tokens_override
    )
    try:
        response, used_proxy = lite_cloud_request(config, payload)
    except httpx.HTTPStatusError as err:
        # 部分兼容网关不支持 thinking 字段（返回 400/422），剥离扩展字段并提高 token 预算后重放一次
        if err.response.status_code in (400, 422) and ('thinking' in payload or 'reasoning_effort' in payload):
            fallback_payload = dict(payload)
            fallback_payload.pop('thinking', None)
            fallback_payload.pop('reasoning_effort', None)
            fallback_payload['max_tokens'] = max(int(fallback_payload.get('max_tokens', 2048)), 4096)
            response, used_proxy = lite_cloud_request(config, fallback_payload)
        else:
            raise

    if parser is None:
        return response, used_proxy, None

    try:
        return response, used_proxy, parser(response)
    except ValueError as err:
        if str(err) != 'truncated':
            raise
        # 若仍发生 length 截断，自动提额至 8192/16384（或关闭 DeepSeek 长思考）再试一次
        retry_payload = dict(payload)
        if disable_thinking and 'thinking' not in retry_payload and _is_deepseek_config(config):
            retry_payload['thinking'] = {'type': 'disabled'}
        retry_payload['max_tokens'] = 8192 if disable_thinking else 16384
        if retry_payload != payload:
            response, used_proxy = lite_cloud_request(config, retry_payload)
            return response, used_proxy, parser(response)
        raise


def lite_cloud_step(obs, fallback: dict, config: CloudConfig, disable_thinking: bool = True):
    """Call cloud VLM strictly with public Observation; provide rich failure diagnostics on fallback."""
    payload, audit = payload_for(obs, config)
    begin = time.perf_counter()
    record = dict(
        **audit,
        fallback=False,
        model=config.model,
        usage={},
        error=None,
        http_status=None,
        used_proxy=None,
        detailed_error='',
        evidence=''
    )
    result = dict(fallback)
    last_usage = None
    try:
        def _parse_obs(resp):
            nonlocal last_usage
            if isinstance(resp, dict) and isinstance(resp.get('usage'), dict):
                last_usage = resp.get('usage')
            return cloud_parse(resp, obs.legal_actions)

        response, used_proxy, value = _execute_cloud_with_adaptation(
            config, payload, disable_thinking=disable_thinking, parser=_parse_obs
        )
        record['used_proxy'] = used_proxy
        usage = response.get('usage', {}) if isinstance(response, dict) else {}
        if isinstance(usage, dict):
            record['usage'] = {
                k: v for k, v in usage.items()
                if k in ('prompt_tokens', 'completion_tokens', 'total_tokens') and type(v) is int and v >= 0
            }
        result['action'] = value['action']
        result['reason'] = 'single_cloud_action'
        evidence = value.get('evidence', '')
        if config.api_key:
            evidence = evidence.replace(config.api_key, '[REDACTED]')
        record['evidence'] = evidence
    except Exception as error:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        record['fallback'] = True
        record['error'] = type(error).__name__
        record['http_status'] = status
        diag = _explain_cloud_exception(error, http_status=status, usage=last_usage)
        if config.api_key:
            diag = diag.replace(config.api_key, '[REDACTED]')
        record['detailed_error'] = diag
        record['evidence'] = f'{diag}；已由本地策略提供备选动作'
        result['reason'] = 'cloud_error_local_fallback'

    record['latency_ms'] = round((time.perf_counter() - begin) * 1000, 1)
    result['cloud'] = record
    return result


def cut_upload_image(data: bytes, k: int):
    """Pure in-memory image center-crop and k*k slicing at 300x300 RGB PNG per tile."""
    from PIL import Image, ImageOps

    if k not in UPLOAD_GRID_SPECS:
        raise ValueError(f'暂不支持 {k}×{k} 切片规格，请选择 3/4/5/6/7/10')
    if not isinstance(data, bytes) or not data:
        raise ValueError('需要有效的图片二进制内容')
    if len(data) > 32 * 1024 * 1024:
        raise ValueError('上传图片大小不能超过 32MB')

    try:
        with Image.open(BytesIO(data)) as im:
            if (im.format or '').upper() not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('仅支持 PNG / JPEG / WEBP 格式图片')
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            w, h = rgb.size
            min_px = max(96, k * 24)
            if min(w, h) < min_px:
                raise ValueError(f'图片分辨率过低（当前 {w}×{h}，{k}×{k} 切分要求短边至少 {min_px}px）')
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            tile_px = 300
            square = rgb.crop((left, top, left + side, top + side)).resize(
                (tile_px * k, tile_px * k), Image.Resampling.LANCZOS
            )
            tiles = []
            for idx in range(k * k):
                r, c = divmod(idx, k)
                cell = square.crop((c * tile_px, r * tile_px, (c + 1) * tile_px, (r + 1) * tile_px))
                buf = BytesIO()
                cell.save(buf, format='PNG', compress_level=1)
                tiles.append(buf.getvalue())
            preview_im = square.resize((768, 768), Image.Resampling.LANCZOS)
            pbuf = BytesIO()
            preview_im.save(pbuf, format='PNG', compress_level=1)
            return tiles, [w, h], pbuf.getvalue()
    except ValueError:
        raise
    except Exception:
        raise ValueError('图片解码失败，请确认文件为完整的 PNG / JPEG / WEBP 图像') from None


def _encode_png_clip_in_memory(png_bytes: bytes, encoder_model):
    """Run Sat2Cap frozen CLIP encoder on one 300x300 PNG in memory (zero disk writes)."""
    import numpy as np
    import torch
    from PIL import Image
    from data.process_masa import MEAN, STD
    from agents.spatial_relation import quadrant_features

    with Image.open(BytesIO(png_bytes)) as src:
        img = src.convert('RGB').resize((224, 224), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    device = next(encoder_model.parameters()).device
    with torch.inference_mode():
        z = encoder_model(tensor.to(device))
        g = z.image_embeds.float().cpu().numpy()[0]
        l = quadrant_features(encoder_model.vision_model.post_layernorm(z.last_hidden_state[:, 1:])).float().cpu().numpy()[0]
    return g.astype(np.float32), l.astype(np.float32)


def _generalized_coverage_choice(position, visited, remaining_budget, explorer_action, k):
    """Horizon<=3 unseen coverage lookahead + public radial non-decreasing guard for k>=6."""
    neighbors = {}
    for cell in range(k * k):
        r, c = divmod(cell, k)
        neighbors[cell] = {
            a: (r + dr) * k + (c + dc)
            for a, (dr, dc) in ACTION_OFFSETS.items()
            if 0 <= r + dr < k and 0 <= c + dc < k
        }
    closed = {
        cell: {cell, *neighbors[cell].values()}
        for cell in range(k * k)
    }
    current = position[0] * k + position[1]
    legal_actions = [a for a in ACTION_LIST if a in neighbors[current]]
    if explorer_action not in neighbors[current]:
        explorer_action = legal_actions[0]

    covered = set()
    for cell in visited:
        covered |= closed[cell]
    known = set(visited)
    horizon = min(3, max(1, remaining_budget))
    best = {}
    for seq in product(ACTION_LIST, repeat=horizon):
        cell = current
        opp = set(covered)
        seen = set(known)
        repeats = 0
        valid = True
        for idx, act in enumerate(seq, 1):
            dest = neighbors[cell].get(act)
            if dest is None:
                valid = False
                break
            if dest in seen:
                repeats += 1
            seen.add(dest)
            if idx < remaining_budget:
                opp |= closed[dest]
            else:
                opp.add(dest)
            cell = dest
        if valid:
            gain = len(opp - covered)
            first = seq[0]
            if first not in best or (gain, -repeats) > (best[first]['gain'], -best[first]['revisit']):
                best[first] = {'gain': gain, 'revisit': repeats}

    if not best:
        return explorer_action, False

    winner = max(
        best.keys(),
        key=lambda a: (best[a]['gain'], -best[a]['revisit'], a == explorer_action, -ACTION_LIST.index(a))
    )
    if best[winner]['gain'] < best.get(explorer_action, {'gain': 0})['gain'] + 2:
        return explorer_action, False

    start = visited[0]
    def radius(cell):
        return abs(cell // k - start // k) + abs(cell % k - start % k)

    if radius(neighbors[current][winner]) < radius(neighbors[current][explorer_action]):
        return explorer_action, False

    return winner, winner != explorer_action


def _frontier_action(position, visited, k):
    """Strictly goal-blind frontier heuristic: maximize new adjacent exposure and avoid revisiting."""
    import numpy as np

    r, c = position
    visited_set = set(visited)
    scores = []
    for act in ACTION_LIST:
        dr, dc = ACTION_OFFSETS[act]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < k and 0 <= nc < k):
            scores.append(-1e9)
            continue
        nxt = nr * k + nc
        score = 5.0 if nxt not in visited_set else -2.0
        unseen_nbrs = 0
        for d2r, d2c in ACTION_OFFSETS.values():
            nnr, nnc = nr + d2r, nc + d2c
            if 0 <= nnr < k and 0 <= nnc < k and (nnr * k + nnc) not in visited_set:
                unseen_nbrs += 1
        score += 0.6 * unseen_nbrs
        scores.append(score)

    arr = np.array(scores, dtype=np.float64)
    valid = arr > -1e8
    if valid.any():
        logits = np.where(valid, arr, -1e9)
        m = np.max(logits[valid])
        e = np.where(valid, np.exp(logits - m), 0.0)
        probs = (e / e.sum()).tolist()
        best_act = ACTION_LIST[int(np.argmax(logits))]
    else:
        probs = [0.25, 0.25, 0.25, 0.25]
        best_act = 'right'
    return best_act, [round(float(p), 4) for p in probs]


def parse_decision_enhanced(dec, act, actor='agent', policy='M0'):
    """Convert raw policy/engine decision trace into clean emoji-free dashboard insight."""
    import numpy as np

    if not isinstance(dec, dict):
        return {
            'branch': '智能网络推演',
            'action': act or 'wait',
            'action_name': ACTION_NAMES.get(act, '待开始') if act else '待开始',
            'probabilities': [0.25, 0.25, 0.25, 0.25],
            'reason_headline': '等待任务启动',
            'reason_detail': '选定起终点后点击【开始任务】，随后可使用【单步推演】或【自动运行】观察决策过程。'
        }

    cue_action = dec.get('cue_action')
    cue_probs = dec.get('probabilities', [])
    exp_logits = dec.get('explorer_logits')
    reason = dec.get('reason', '')
    act_name = ACTION_NAMES.get(act, act) if act else '待执行'

    # 1. 人工接管模式
    if actor == 'human':
        det = f'演示者通过方向按键直接控制智能体向【{act_name}】移动一步。'
        return {
            'branch': '人工遥控控制',
            'action': act,
            'action_name': act_name,
            'probabilities': [1.0 if d == act else 0.0 for d in ['up', 'right', 'down', 'left']],
            'reason': det,
            'reason_headline': '人工交互指令输入',
            'reason_detail': det
        }

    # 2. 云端 API 模式（明确区分正常返回 vs 异常回退与精确 HTTP 原因）
    if policy == 'Cloud' or dec.get('cloud'):
        cloud_rec = dec.get('cloud') or {}
        model_name = cloud_rec.get('model', '云端 VLM')
        is_fallback = bool(cloud_rec.get('fallback'))
        detailed_err = cloud_rec.get('detailed_error') or ''
        evidence = cloud_rec.get('evidence') or '大模型结合目标图像与当前局部视野特征做出的动作选择'
        if is_fallback:
            diag = detailed_err or evidence
            det = f'云端请求失败（{diag}）。为保证演示连续性，本步已自动回退由本地 M0 推荐动作【{act_name}】。'
            headline = f'云端失败回退本地建议：{act_name}'
        else:
            proxy_note = ' [经本机代理]' if cloud_rec.get('used_proxy') else ''
            lat = cloud_rec.get('latency_ms')
            lat_note = f'（耗时 {int(lat)}ms{proxy_note}）' if lat else ''
            det = f'云端多模态模型视觉推理依据{lat_note}：{evidence}'
            headline = f'云端模型直接输出：{act_name}'
        return {
            'branch': f'云端大模型 ({model_name})',
            'action': act,
            'action_name': act_name,
            'probabilities': [1.0 if d == act else 0.0 for d in ['up', 'right', 'down', 'left']],
            'reason': det,
            'reason_headline': headline,
            'reason_detail': det
        }

    # 3. 视觉线索门控锁定 (Cue Triggered)
    if cue_action:
        four_probs = [float(p) for p in cue_probs[:4]] if len(cue_probs) >= 4 else [0.25, 0.25, 0.25, 0.25]
        s = sum(four_probs) or 1.0
        norm_probs = [round(p / s, 4) for p in four_probs]
        act_idx = {'up': 0, 'right': 1, 'down': 2, 'left': 3}.get(act, 0)
        conf = round(four_probs[act_idx] * 100, 1)
        det = (
            f'边缘视觉匹配头在当前视野中检测到与目标图高度连续的纹理与几何缝隙线索，'
            f'单向邻接置信度达 {conf}%（>= 50% 门控阈值），跳过纯探索阶段，直接向目标格执行锁定步进。'
        )
        return {
            'branch': '视觉线索锁定门控 (Cue Gate)',
            'action': act,
            'action_name': act_name,
            'probabilities': norm_probs,
            'reason': det,
            'reason_headline': f'触发目标线索锁定（置信度 {conf}%）',
            'reason_detail': det
        }

    # 4. 探索器主动覆盖决策 (Explorer Branch)
    if exp_logits and len(exp_logits) == 4:
        arr = np.array(exp_logits, dtype=np.float64)
        e = np.exp(arr - np.max(arr))
        soft_probs = [round(float(p), 4) for p in (e / e.sum()).tolist()]
        act_idx = {'up': 0, 'right': 1, 'down': 2, 'left': 3}.get(act, 0)
        chosen_prob = round(soft_probs[act_idx] * 100, 1)
        top_prob = round(max(soft_probs) * 100, 1)
        non_adjacent_prob = (cue_probs[4] * 100) if len(cue_probs) > 4 else 99.9

        dir_explain = {
            'left': '向左拓展可避开已探明区域，朝左侧未知区域扩大搜索半径',
            'right': '向右推进能够获得更高的未观测邻域覆盖增益，避免原地徘徊',
            'up': '向上方行进可扩大纵向探索前沿，覆盖上方未知网格',
            'down': '向下方推进符合前瞻路径展开方向，能够开拓下方未知视野',
        }
        turn_reason = dir_explain.get(act, '朝未探索区域拓展视野')
        cov_note = '（结合 H=3 前瞻覆盖规划与径向防折返约束调优）' if dec.get('coverage_adjusted') else ''
        detail = (
            f'线索头判定当前暂未紧邻目标（非邻接概率 {non_adjacent_prob:.1f}%）。'
            f'由 GRU 拓扑探索器接管规划：{turn_reason}{cov_note}，当前方向决策倾向度 {chosen_prob}%（网络最高输出 {top_prob}%）。'
        )
        return {
            'branch': '拓扑探索器规划 (Explorer Network)',
            'action': act,
            'action_name': act_name,
            'probabilities': soft_probs,
            'reason': detail,
            'reason_headline': f'主动拓扑覆盖探索（倾向度 {chosen_prob}%）',
            'reason_detail': detail
        }

    # 5. 启发式/边缘匹配/规则兜底
    probs = dec.get('probabilities') if isinstance(dec.get('probabilities'), list) and len(dec.get('probabilities')) == 4 else [0.25, 0.25, 0.25, 0.25]
    branch_label = dec.get('custom_branch') or ('Frontier 边界探索' if policy == 'frontier' else '启发式探索规划')
    headline = dec.get('reason_headline') or f'执行探索动作：{act_name}'
    det = dec.get('reason_detail') or reason or '根据已访问网格拓扑优先向外扩展未观测边界。'
    return {
        'branch': branch_label,
        'action': act,
        'action_name': act_name,
        'probabilities': [round(float(p), 4) for p in probs],
        'reason': det,
        'reason_headline': headline,
        'reason_detail': det
    }


def _ensure_torch_single_thread():
    """Lock PyTorch to single-thread deterministic CPU execution before worker threads spawn."""
    import torch
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class _LiteSceneStub:
    scenes = {}

    @staticmethod
    def _disk_runs():
        return []

    @staticmethod
    def recovery_report():
        return {}

    @staticmethod
    def heartbeat(_owner):
        return {'lease_seconds': 180}

    @staticmethod
    def release(_owner):
        return None


class _LiteWorkbench:
    """Minimal Lite runtime hosting CloudConfig & 192 OpenCVL ground tasks without Mid-Term/V5 bloat."""

    def __init__(self, root: Path):
        from platform_core.capturepoint_catalog_v1 import CapturepointCatalog
        from platform_core.capturepoint_sessions_v1 import CapturepointSessions
        from platform_core.controller_hub_v1 import ControllerHub

        self.root = Path(root).resolve()
        self.store = self.root / 'DATA/processed_data/平台实验/工作台_v1'
        self.datasets = {}
        self.scenes = _LiteSceneStub()

        env_path = self.root / 'project/.env'
        if env_path.is_file():
            try:
                self.config = CloudConfig.existing(self.root)
            except Exception:
                self.config = None
        else:
            self.config = None

        cat = CapturepointCatalog(self.root)
        self.capturepoint_sessions = CapturepointSessions(self.root, self.store, cat, self._owner)
        self.controllers = ControllerHub(self)

    @staticmethod
    def _owner(owner):
        if not isinstance(owner, str) or len(owner) != 32:
            raise ValueError('无效会话身份')
        return owner

    def shutdown(self):
        try:
            self.capturepoint_sessions.release('0' * 32)
        except Exception:
            pass


class LiteServerManager:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.wb = _LiteWorkbench(self.root)
        # 确保默认 max_tokens=2048，并默认对支持的思考模型关闭冗余思维链防止 JSON 输出被截断
        if self.wb.config is not None and self.wb.config.max_tokens < 2048:
            self.wb.config = CloudConfig(
                self.wb.config.base_url,
                self.wb.config.model,
                self.wb.config.api_key,
                self.wb.config.timeout,
                2048
            )
        self.api_disable_thinking = True
        self.owner = '0' * 32
        self.lock = threading.RLock()
        self.uploads = {}  # volatile in-memory store: vanishes when backend closes
        self.active_session = None
        self._m0_legacy = None
        self._clip_encoder = None
        self._global_banks = {}      # cache loaded .npy/.npz dicts
        self._local_npz_handles = {} # open .npz handles
        self._area_cache = OrderedDict()  # (bank_id, area, grid) -> {g, l, p, patch_dir, pngs}
        self._stop_event = threading.Event()
        self._init_catalog()
        # 在主线程初始化 PyTorch 单线程并预热 Sat2Cap（规避 Windows 下子线程 from_pretrained 访问冲突）
        _ensure_torch_single_thread()
        if self._has_clip_encoder() and threading.current_thread() is threading.main_thread():
            self._get_clip_encoder()
        self._start_heartbeat_daemon()

    def _start_heartbeat_daemon(self):
        def _loop():
            while not self._stop_event.wait(20):
                try:
                    self.heartbeat()
                except Exception:
                    pass
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def shutdown(self):
        self._stop_event.set()
        for h in list(self._local_npz_handles.values()):
            try:
                h.close()
            except Exception:
                pass
        self._local_npz_handles.clear()
        try:
            self.wb.shutdown()
        except Exception:
            pass

    def heartbeat(self):
        with self.lock:
            try:
                self.wb.scenes.heartbeat(self.owner)
            except Exception:
                pass
            try:
                self.wb.capturepoint_sessions.heartbeat(self.owner)
            except Exception:
                pass
            return {'status': 'ok'}

    def _get_m0_legacy(self):
        if self._m0_legacy is None:
            _ensure_torch_single_thread()
            from agents.frozen_edge_navigator import load_frozen_edge_default
            cfg = json.loads((self.root / 'project/local_policy_default.json').read_text(encoding='utf-8-sig'))
            self._m0_legacy = load_frozen_edge_default(cfg, self.root, seed=0, device='cpu')
        return self._m0_legacy

    def _has_clip_encoder(self):
        return (self.root / 'models' / 'Sat2Cap').is_dir()

    def _get_clip_encoder(self):
        if self._clip_encoder is None:
            from webapp import upload_v4
            self._clip_encoder = upload_v4.encoder_model()
        return self._clip_encoder

    def _bank_spec(self, bank_id):
        masa = self.root / 'DATA/processed_data/Masa'
        sv = self.root / 'DATA/processed_data/SwissView'
        mg = self.root / 'DATA/processed_data/MassGIS/工程准备/同产品开发导航兼容_v1'
        specs = {
            'masa-s2': {
                'label': '航拍 · Masa S2 验证集',
                'short': 'Masa-S2',
                'patches_root': masa / 'patches/val',
                'global_npy': masa / 'papr_val_sat_embeds_grid_5.npy',
                'local_npz': masa / '评测结果/边缘线索S2正式复验_v1/验证局部区域特征.npz',
                'profile_npz': masa / '评测结果/边缘线索S2正式复验_v1/验证边缘profile.npz',
            },
            'masa-s3': {
                'label': '航拍 · Masa S3 独立测试源',
                'short': 'Masa-S3',
                'patches_root': masa / 'patches/test',
                'global_npy': masa / 'papr_test_sat_embeds_grid_5.npy',
                'local_npz': masa / '评测结果/边缘线索S3独立源图确认_v1/测试局部区域特征.npz',
                'profile_npz': masa / '评测结果/边缘线索S3独立源图确认_v1/测试边缘profile.npz',
            },
            'swissview-1': {
                'label': '航拍 · SwissView 瑞士-批1',
                'short': '瑞士SV1',
                'patches_root': sv / '五乘五冻结迁移_v1/正式数据/patches/test',
                'global_npz': sv / '五乘五冻结迁移_v1/正式数据/全局特征.npz',
                'local_npz': sv / '五乘五冻结迁移_v1/正式数据/局部特征.npz',
                'profile_npz': sv / '五乘五冻结迁移_v1/正式数据/边缘profile.npz',
            },
            'swissview-2': {
                'label': '航拍 · SwissView 瑞士-批2',
                'short': '瑞士SV2',
                'patches_root': sv / '继续训练独立确认_v1/五乘五数据/patches/test',
                'global_npz': sv / '继续训练独立确认_v1/五乘五数据/全局特征.npz',
                'local_npz': sv / '继续训练独立确认_v1/五乘五数据/局部特征.npz',
                'profile_npz': sv / '继续训练独立确认_v1/五乘五数据/边缘profile.npz',
            },
            'masa-train': {
                'label': '航拍 · Masa 大规模图库',
                'short': 'Masa库',
                'patches_root': masa / 'patches/train',
                'global_npy': masa / 'papr_train_sat_embeds_grid_5.npy',
                'local_npz': masa / '训练结果/边缘连续性可信线索对照_v1/局部区域特征.npz',
                'profile_npz': None,
            },
            'massgis': {
                'label': '航拍 · MassGIS 区域扩展 (含10×10)',
                'short': 'MassGIS',
                'native_npz': mg / '特征/native225.npz',
                'patches_base': mg / '工程数据',
            },
        }
        return specs.get(bank_id)

    def _area_available(self, bank_id, area, grid=5):
        spec = self._bank_spec(bank_id)
        if not spec:
            return False
        if bank_id == 'massgis':
            return (spec['native_npz'].is_file() and (spec['patches_base'] / f'grid{grid}/patches/dev/{area}').is_dir())
        if not (spec['patches_root'] / area).is_dir():
            return False
        if 'global_npy' in spec and not spec['global_npy'].is_file():
            return False
        if 'global_npz' in spec and not spec['global_npz'].is_file():
            return False
        if not spec['local_npz'].is_file():
            return False
        return True

    def _get_npz(self, path):
        import numpy as np
        k = str(path)
        if k not in self._local_npz_handles:
            self._local_npz_handles[k] = np.load(path, allow_pickle=False)
        return self._local_npz_handles[k]

    def _load_area_assets(self, bank_id, area, grid=5):
        """Lazily load & cache one aerial image's tiles and precomputed features."""
        import numpy as np

        cache_key = (bank_id, area, grid)
        if cache_key in self._area_cache:
            self._area_cache.move_to_end(cache_key)
            return self._area_cache[cache_key]

        spec = self._bank_spec(bank_id)
        if not spec:
            raise ValueError(f'未知航拍数据集: {bank_id}')

        if bank_id == 'massgis':
            npz = self._get_npz(spec['native_npz'])
            indices = [r * 15 + c + (15 - grid) for r in range(grid) for c in range(grid)]
            g = npz['global_features'][indices]
            l = npz['local_features'][indices]
            p = npz['profiles'][indices]
            patch_dir = spec['patches_base'] / f'grid{grid}/patches/dev/{area}'
        else:
            if 'global_npy' in spec:
                gk = str(spec['global_npy'])
                if gk not in self._global_banks:
                    self._global_banks[gk] = np.load(spec['global_npy'], allow_pickle=True).item()
                g = self._global_banks[gk][area]
            else:
                g = self._get_npz(spec['global_npz'])[area]

            l = self._get_npz(spec['local_npz'])[area]
            p = self._get_npz(spec['profile_npz'])[area] if spec.get('profile_npz') and spec['profile_npz'].is_file() else None
            patch_dir = spec['patches_root'] / area

        entry = {
            'g': np.asarray(g, dtype=np.float32),
            'l': np.asarray(l, dtype=np.float32),
            'p': np.asarray(p, dtype=np.float32) if p is not None else None,
            'p_lazy': {},
            'patch_dir': patch_dir,
            'pngs': {}
        }
        self._area_cache[cache_key] = entry
        while len(self._area_cache) > 24:
            self._area_cache.popitem(last=False)
        return entry

    def _get_area_tile_png(self, asset_entry, cell_idx):
        from PIL import Image
        pngs = asset_entry['pngs']
        if cell_idx in pngs:
            return pngs[cell_idx]
        patch_dir = asset_entry['patch_dir']
        cand = patch_dir / f'patch_{cell_idx}.png'
        if cand.is_file():
            raw = cand.read_bytes()
        else:
            jpg = patch_dir / f'patch_{cell_idx}.jpg'
            if not jpg.is_file():
                matches = list(patch_dir.glob(f'patch_{cell_idx}.*'))
                if not matches:
                    return self._placeholder_png()
                jpg = matches[0]
            with Image.open(jpg) as im:
                buf = BytesIO()
                im.convert('RGB').save(buf, format='PNG')
                raw = buf.getvalue()
        pngs[cell_idx] = raw
        return raw

    def _init_catalog(self):
        with self.lock:
            self.tasks_catalog = []
            aerial_image_set = set()
            aerial_count = 0

            # 1. 加载精选多来源航拍样本库（每张图严格 <= 5 个实测通过的起终点组合）
            curated = {}
            if CURATED_INDEX_PATH.is_file():
                curated = json.loads(CURATED_INDEX_PATH.read_text('utf-8')).get('datasets', {})

            order = ['masa-s2', 'masa-s3', 'swissview-1', 'swissview-2', 'massgis', 'masa-train']
            seen_per_img = {}
            for bank_id in order:
                rows = curated.get(bank_id, [])
                spec = self._bank_spec(bank_id)
                if not spec or not rows:
                    continue
                avail_rows = [
                    r for r in rows
                    if self._area_available(bank_id, r['area'], r.get('grid_size', 5))
                ]
                imgs_in_bank = len({r['area'] for r in avail_rows})
                if not imgs_in_bank:
                    continue
                group_label = f"{spec['label']} ({imgs_in_bank}图 · {len(avail_rows)}题)"
                for r in avail_rows:
                    area = r['area']
                    grid = int(r.get('grid_size', 5))
                    img_uid = (bank_id, area, grid)
                    seen_per_img[img_uid] = seen_per_img.get(img_uid, 0) + 1
                    if seen_per_img[img_uid] > 5:
                        continue
                    idx_num = area.split('_')[1] if '_' in area else area
                    tkey = f"{bank_id}::{r['episode_id']}"
                    dim_tag = f' {grid}×{grid}' if grid != 5 else ''
                    self.tasks_catalog.append({
                        'id': tkey,
                        'group': group_label,
                        'name': f"[{spec['short']}{dim_tag} · 图{idx_num}] 起{r['start']}→终{r['goal']} (d0={r['dist']})",
                        'dataset_id': 'aerial_bank',
                        'bank_id': bank_id,
                        'area': area,
                        'city': area,
                        'start': int(r['start']),
                        'default_goal': int(r['goal']),
                        'fixed_endpoints': False,
                        'grid': grid,
                        'budget': int(r.get('budget', 10 if grid == 5 else 20)),
                        'task_key': tkey
                    })
                    aerial_image_set.add((bank_id, area))
                    aerial_count += 1

            # 2. 地面 OpenCVL 任务 (192 题，保持完整不动)
            cat = self.wb.controllers.catalog(self.owner)
            ground_count = 0
            for t in cat['tasks']:
                if t.get('dataset_id') == CAMERA:
                    key = t['task_key']
                    try:
                        true_goal = self.wb.capturepoint_sessions.catalogue.task(key)['truth']['goal_index']
                    except Exception:
                        true_goal = 12
                    self.tasks_catalog.append({
                        'id': key,
                        'group': '地面 · OpenCVL 拍摄点 (192题)',
                        'name': f"[地面] {t['city']} · {t['task_id'][:20]}",
                        'dataset_id': CAMERA,
                        'city': t.get('city', ''),
                        'start': t.get('start', 0),
                        'default_goal': true_goal,
                        'fixed_endpoints': True,
                        'grid': t.get('grid', 5),
                        'budget': t.get('budget', 10),
                        'task_key': key
                    })
                    ground_count += 1

            self.catalog_stats = {
                'aerial_images': len(aerial_image_set),
                'aerial_tasks': aerial_count,
                'ground_tasks': ground_count
            }

    def meta(self):
        with self.lock:
            api_info = dict(self.wb.config.public(), disable_thinking=self.api_disable_thinking) if self.wb.config else None
            active_state = None
            if self.active_session:
                try:
                    active_state = self.get_session_state()
                except Exception:
                    self.active_session = None
                    active_state = None
            return {
                'title': 'GeoNav Lite 极简演示平台',
                'version': 'v1.3-lite',
                'tasks': self.tasks_catalog,
                'stats': self.catalog_stats,
                'api_configured': api_info is not None,
                'api_info': api_info,
                'clip_encoder_available': self._has_clip_encoder(),
                'upload_grids': [{'k': k, **v} for k, v in UPLOAD_GRID_SPECS.items()],
                'uploads': [{'id': k, 'name': v['name'], 'grid': v['grid']} for k, v in self.uploads.items()],
                'active_session': active_state
            }

    def save_api_config(self, base_url, model, api_key, timeout=30, disable_thinking=True):
        with self.lock:
            old = self.wb.config
            url = base_url.strip()
            mod = model.strip()
            key = api_key.strip()
            if not key and old and url == old.base_url:
                key = old.api_key
            self.api_disable_thinking = bool(disable_thinking)
            self.wb.config = CloudConfig(url, mod, key, float(timeout), 2048)
            return dict(self.wb.config.public(), disable_thinking=self.api_disable_thinking)

    def test_api_connection(self, base_url='', model='', api_key='', timeout=20, disable_thinking=None):
        """Run lightweight text-only /chat/completions connectivity check without persisting invalid edits."""
        with self.lock:
            old = self.wb.config
            url = base_url.strip() or (old.base_url if old else '')
            mod = model.strip() or (old.model if old else '')
            key = api_key.strip()
            if not key and old and url == old.base_url:
                key = old.api_key
            if not url or not mod:
                raise ValueError('请提供完整的 Base URL 与 Model 名称')
            use_disable_thinking = self.api_disable_thinking if disable_thinking is None else bool(disable_thinking)
            cfg = CloudConfig(url, mod, key, float(timeout or 20), 2048)

        payload = {
            'model': cfg.model,
            'messages': [{'role': 'user', 'content': 'Reply strictly with JSON: {"ok":true}'}],
            'temperature': 0,
            'max_tokens': 256 if use_disable_thinking else 2048,
            'stream': False
        }
        t0 = time.perf_counter()
        try:
            resp, used_proxy, _ = _execute_cloud_with_adaptation(
                cfg,
                payload,
                disable_thinking=use_disable_thinking,
                max_tokens_override=256 if use_disable_thinking else 2048
            )
            lat = round((time.perf_counter() - t0) * 1000, 1)
            msg = resp['choices'][0]['message'].get('content', '') if isinstance(resp, dict) else ''
            return {
                'ok': True,
                'latency_ms': lat,
                'used_proxy': used_proxy,
                'message': f'接口连通正常（耗时 {int(lat)}ms' + (f'，经过代理 {used_proxy}' if used_proxy else '') + f'，返回 {len(str(msg))} 字符）'
            }
        except Exception as err:
            status = getattr(getattr(err, 'response', None), 'status_code', None)
            diag = _explain_cloud_exception(err, http_status=status)
            if cfg.api_key:
                diag = diag.replace(cfg.api_key, '[REDACTED]')
            return {
                'ok': False,
                'http_status': status,
                'message': diag
            }

    def handle_upload(self, data, name="临时场景", grid=5):
        with self.lock:
            k = int(grid)
            tiles, sizes, preview_bytes = cut_upload_image(data, k)
            upload_id = secrets.token_hex(6)
            budget = UPLOAD_GRID_SPECS[k]['budget']
            clean_name = (name or '临时场景').strip()[:48] or '临时场景'
            self.uploads[upload_id] = {
                'id': upload_id,
                'name': clean_name,
                'tiles': tiles,
                'preview': preview_bytes,
                'grid': k,
                'budget': budget,
                'size': sizes,
                'clip_cache': {},
                'profile_cache': {}
            }
            default_goal = k * k - 1
            custom_task = {
                'id': f'upload_{upload_id}',
                'group': '临时上传场景 (关闭服务后清除)',
                'name': f"[临时 {k}×{k}] {clean_name}",
                'dataset_id': 'upload',
                'upload_id': upload_id,
                'city': '临时内存',
                'start': 0,
                'default_goal': default_goal,
                'fixed_endpoints': False,
                'grid': k,
                'budget': budget,
                'task_key': f'upload_{upload_id}'
            }
            self.tasks_catalog.insert(0, custom_task)
            return custom_task

    def delete_upload(self, task_key):
        with self.lock:
            if not task_key or not str(task_key).startswith('upload_'):
                raise ValueError('只能删除临时上传的场景')
            upload_id = str(task_key).removeprefix('upload_')
            self.uploads.pop(upload_id, None)
            self.tasks_catalog = [t for t in self.tasks_catalog if t.get('task_key') != task_key]
            if self.active_session and self.active_session.get('upload_id') == upload_id:
                self.active_session = None
            return {'deleted': task_key}

    def create_session(self, task_key, policy='M0', custom_start=None, custom_goal=None, seed=0):
        with self.lock:
            task = next((t for t in self.tasks_catalog if t['task_key'] == task_key), None)
            if not task:
                task = self.tasks_catalog[0]

            dataset_id = task['dataset_id']
            grid = task['grid']
            budget = task['budget']

            if dataset_id == 'upload':
                upload_id = task['upload_id']
                if upload_id not in self.uploads:
                    raise ValueError('临时上传场景已过期或不存在，请重新上传图片')
                start = max(0, min(grid * grid - 1, int(custom_start if custom_start is not None else 0)))
                goal = max(0, min(grid * grid - 1, int(custom_goal if custom_goal is not None else (grid * grid - 1))))
                if start == goal:
                    goal = (start + (grid * grid) // 2) % (grid * grid)
                if policy == 'Cloud' and not self.wb.config:
                    raise ValueError('尚未配置云端 API，请先点击右上角【云端 API】设置')
                self.active_session = {
                    'type': 'upload',
                    'upload_id': upload_id,
                    'task_key': task['task_key'],
                    'policy': policy,
                    'start': start,
                    'goal': goal,
                    'position': start,
                    'grid': grid,
                    'budget': budget,
                    'step': 0,
                    'status': 'ready',
                    'trajectory': [{'step': 0, 'cell': start, 'action': None}],
                    'visited': [start],
                    'result': None,
                    'hidden': None,
                    'last_decision': {
                        'action': 'wait',
                        'action_name': '已就绪',
                        'probabilities': [0.25, 0.25, 0.25, 0.25],
                        'reason_headline': f'临时 {grid}×{grid} 场景已就绪',
                        'reason_detail': '点击【单步推演】或【自动运行】开始探索定位。'
                    }
                }
                return self.get_session_state()

            elif dataset_id == 'aerial_bank':
                bank_id = task['bank_id']
                area = task['area']
                start = max(0, min(grid * grid - 1, int(custom_start if custom_start is not None else task['start'])))
                goal = max(0, min(grid * grid - 1, int(custom_goal if custom_goal is not None else task['default_goal'])))
                if start == goal:
                    goal = (start + (grid * grid) // 2) % (grid * grid)
                if policy == 'Cloud' and not self.wb.config:
                    raise ValueError('尚未配置云端 API，请先点击右上角【云端 API】设置')
                # 验证资产就绪并预热当前场景数组
                self._load_area_assets(bank_id, area, grid)
                self.active_session = {
                    'type': 'aerial_bank',
                    'bank_id': bank_id,
                    'area': area,
                    'task_key': task['task_key'],
                    'policy': policy,
                    'start': start,
                    'goal': goal,
                    'position': start,
                    'grid': grid,
                    'budget': budget,
                    'step': 0,
                    'status': 'running',
                    'trajectory': [{'step': 0, 'cell': start, 'action': None}],
                    'visited': [start],
                    'result': None,
                    'hidden': None,
                    'last_decision': {
                        'action': 'wait',
                        'action_name': '已就绪',
                        'probabilities': [0.25, 0.25, 0.25, 0.25],
                        'reason_headline': f'{task["name"]} 已就绪',
                        'reason_detail': '点击【单步推演】或【自动运行】开始探索定位。'
                    }
                }
                return self.get_session_state()

            elif dataset_id == CAMERA:
                spec = {
                    'name': f"lite_cam_{task['city']}",
                    'dataset_id': CAMERA,
                    'task_key': task['task_key'],
                    'controller': 'manual' if policy == 'manual' else 'frontier',
                    'seed': 0
                }
                res = self.wb.controllers.create(spec, self.owner)
                goal_idx = task.get('default_goal', 12)
                self.active_session = {
                    'type': 'camera',
                    'rid': res['id'],
                    'policy': 'manual' if policy == 'manual' else 'frontier',
                    'requested_policy': policy,
                    'task_key': task['task_key'],
                    'city': task['city'],
                    'start': task['start'],
                    'goal': goal_idx,
                    'grid': grid,
                    'budget': budget
                }
                self.heartbeat()
                return self.get_session_state()

            raise ValueError('未知场景类型')

    def _aerial_bank_agent_decide(self, s):
        """Goal-blind M0 / Frontier / Cloud decision over precomputed feature banks (4-stage Masa & SwissView & MassGIS)."""
        import numpy as np
        import torch
        from agents.target_cue import cue_features
        from agents.edge_cue import image_profiles, edge_features
        from agents.scaled_edge_navigator import scaled_cue_choice
        from train.dyncur_tiny import policy_features
        from env.environment import Observation

        assets = self._load_area_assets(s['bank_id'], s['area'], s['grid'])
        k = s['grid']
        budget = s['budget']
        curr = s['position']
        goal = s['goal']  # evaluator-only index lookup; NEVER exposed as coordinate/distance to policy
        cr, cc = divmod(curr, k)
        remaining = budget - s['step']
        visited = tuple(s['visited'])
        policy = s['policy']

        curr_png = self._get_area_tile_png(assets, curr)
        tgt_png = self._get_area_tile_png(assets, goal)

        if policy == 'frontier':
            act, probs = _frontier_action((cr, cc), visited, k)
            return act, {
                'action': act,
                'probabilities': probs,
                'custom_branch': 'Frontier 边界探索',
                'reason_headline': f'未知边界前沿探索：{ACTION_NAMES[act]}',
                'reason_detail': '无目标先验约束，优先朝未探明区域边界与最多未知邻域推进。'
            }

        legacy = self._get_m0_legacy()
        curr_g, curr_l = assets['g'][curr], assets['l'][curr]
        tgt_g, tgt_l = assets['g'][goal], assets['l'][goal]

        if assets['p'] is not None:
            curr_prof = assets['p'][curr]
            tgt_prof = assets['p'][goal]
        else:
            p_lazy = assets['p_lazy']
            if curr not in p_lazy:
                p_lazy[curr] = image_profiles(curr_png)
            if goal not in p_lazy:
                p_lazy[goal] = image_profiles(tgt_png)
            curr_prof, tgt_prof = p_lazy[curr], p_lazy[goal]

        seam = edge_features(tgt_prof, curr_prof)

        if k == 5 and budget == 10:
            base = policy_features(legacy.em, curr_g, (cr, cc), remaining, visited)
        else:
            ids = np.asarray(visited, dtype=np.int64)
            pooled = (ids // k * 5 // k) * 5 + (ids % k * 5 // k)
            counts = np.minimum(np.bincount(pooled, minlength=25).astype(np.float32), 3.0) / 3.0
            base = np.concatenate((
                legacy.em,
                curr_g,
                np.asarray([cr / max(1, k - 1), cc / max(1, k - 1), remaining / max(1, budget)], dtype=np.float32),
                counts
            )).astype(np.float32)

        with torch.no_grad():
            logits, _, _, s['hidden'] = legacy.explorer.step(
                torch.as_tensor(base, device=legacy.device)[None], s.get('hidden')
            )
            exp_logits = logits[0].cpu().tolist()
            proposal = ACTION_LIST[int(np.argmax(exp_logits))]
            semantic = cue_features(tgt_g, curr_g, tgt_l, curr_l)
            x = np.concatenate((semantic, seam)).astype(np.float32)
            cue_values = legacy.head(torch.as_tensor(x, device=legacy.device)[None]).softmax(-1)[0].cpu().numpy()

        cue_action, cue_reason = scaled_cue_choice(cue_values, 0.50, (cr, cc), visited, k)
        chosen = cue_action or proposal
        cov_adjusted = False
        if cue_action is None and k >= 6:
            # 对合法邻域做掩码确保 Coverage3Radial 首选合法动作
            masked_logits = list(exp_logits)
            for idx, a_name in enumerate(ACTION_LIST):
                dr, dc = ACTION_OFFSETS[a_name]
                if not (0 <= cr + dr < k and 0 <= cc + dc < k):
                    masked_logits[idx] = -1e9
            legal_prop = ACTION_LIST[int(np.argmax(masked_logits))]
            chosen, cov_adjusted = _generalized_coverage_choice((cr, cc), visited, remaining, legal_prop, k)

        dec = {
            'action': chosen,
            'cue_action': cue_action,
            'probabilities': cue_values.tolist(),
            'explorer_logits': exp_logits,
            'reason': cue_reason,
            'coverage_adjusted': cov_adjusted
        }

        if policy == 'Cloud':
            if not self.wb.config:
                raise ValueError('尚未配置云端 API')
            obs = Observation(curr_png, tgt_png, (cr, cc), k, remaining, visited)
            cloud_dec = lite_cloud_step(obs, dec, self.wb.config, disable_thinking=self.api_disable_thinking)
            chosen = cloud_dec['action']
            return chosen, parse_decision_enhanced(cloud_dec, chosen, actor='agent', policy='Cloud')

        return chosen, parse_decision_enhanced(dec, chosen, actor='agent', policy='M0')

    def _upload_agent_decide(self, s):
        """Goal-blind decision logic for uploaded n×n images: real M0 (+ Sat2Cap/Coverage), Cloud API, or Frontier."""
        import numpy as np
        import torch
        from agents.target_cue import cue_features
        from agents.edge_cue import image_profiles, edge_features
        from agents.scaled_edge_navigator import scaled_cue_choice
        from train.dyncur_tiny import policy_features
        from env.environment import Observation

        u_data = self.uploads[s['upload_id']]
        k = s['grid']
        budget = s['budget']
        curr = s['position']
        goal = s['goal']
        cr, cc = divmod(curr, k)
        remaining = budget - s['step']
        visited = tuple(s['visited'])
        policy = s['policy']

        curr_png = u_data['tiles'][curr]
        tgt_png = u_data['tiles'][goal]

        if policy == 'frontier':
            act, probs = _frontier_action((cr, cc), visited, k)
            return act, {
                'action': act,
                'probabilities': probs,
                'custom_branch': 'Frontier 边界探索',
                'reason_headline': f'未知边界前沿探索：{ACTION_NAMES[act]}',
                'reason_detail': '无目标先验约束，优先朝未探明区域边界与最多未知邻域推进。'
            }

        p_cache = u_data['profile_cache']
        if curr not in p_cache:
            p_cache[curr] = image_profiles(curr_png)
        if goal not in p_cache:
            p_cache[goal] = image_profiles(tgt_png)
        seam = edge_features(p_cache[goal], p_cache[curr])

        dec = None
        if self._has_clip_encoder():
            legacy = self._get_m0_legacy()
            encoder = self._get_clip_encoder()
            c_cache = u_data['clip_cache']
            if curr not in c_cache:
                c_cache[curr] = _encode_png_clip_in_memory(curr_png, encoder)
            if goal not in c_cache:
                c_cache[goal] = _encode_png_clip_in_memory(tgt_png, encoder)
            curr_g, curr_l = c_cache[curr]
            tgt_g, tgt_l = c_cache[goal]

            if k == 5 and budget == 10:
                base = policy_features(legacy.em, curr_g, (cr, cc), remaining, visited)
            else:
                ids = np.asarray(visited, dtype=np.int64)
                pooled = (ids // k * 5 // k) * 5 + (ids % k * 5 // k)
                counts = np.minimum(np.bincount(pooled, minlength=25).astype(np.float32), 3.0) / 3.0
                base = np.concatenate((
                    legacy.em,
                    curr_g,
                    np.asarray([cr / max(1, k - 1), cc / max(1, k - 1), remaining / max(1, budget)], dtype=np.float32),
                    counts
                )).astype(np.float32)

            with torch.no_grad():
                logits, _, _, s['hidden'] = legacy.explorer.step(
                    torch.as_tensor(base, device=legacy.device)[None], s.get('hidden')
                )
                exp_logits = logits[0].cpu().tolist()
                masked_logits = list(exp_logits)
                for idx, a_name in enumerate(ACTION_LIST):
                    dr, dc = ACTION_OFFSETS[a_name]
                    if not (0 <= cr + dr < k and 0 <= cc + dc < k):
                        masked_logits[idx] = -1e9
                proposal = ACTION_LIST[int(np.argmax(masked_logits))]

                semantic = cue_features(tgt_g, curr_g, tgt_l, curr_l)
                x = np.concatenate((semantic, seam)).astype(np.float32)
                cue_values = legacy.head(torch.as_tensor(x, device=legacy.device)[None]).softmax(-1)[0].cpu().numpy()

            cue_action, cue_reason = scaled_cue_choice(cue_values, 0.50, (cr, cc), visited, k)
            chosen = cue_action or proposal
            cov_adjusted = False
            if cue_action is None and k >= 6:
                chosen, cov_adjusted = _generalized_coverage_choice((cr, cc), visited, remaining, proposal, k)

            dec = {
                'action': chosen,
                'cue_action': cue_action,
                'probabilities': cue_values.tolist(),
                'explorer_logits': masked_logits,
                'reason': cue_reason,
                'coverage_adjusted': cov_adjusted
            }
        else:
            seam_4 = seam.reshape(4, 5)
            seam_scores = []
            for d_idx, a_name in enumerate(ACTION_LIST):
                dr, dc = ACTION_OFFSETS[a_name]
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < k and 0 <= nc < k) or (nr * k + nc) in visited:
                    seam_scores.append(-1e9)
                else:
                    err_l1 = float(seam_4[d_idx, 0] + seam_4[d_idx, 1] + seam_4[d_idx, 3]) / 3.0
                    corr = float(seam_4[d_idx, 4])
                    seam_scores.append(corr * 1.4 - err_l1 * 4.0)

            best_seam_idx = int(np.argmax(seam_scores))
            if seam_scores[best_seam_idx] > 0.35:
                arr = np.array([max(-10.0, v) if v > -1e8 else -10.0 for v in seam_scores], dtype=np.float64)
                e = np.exp(arr - np.max(arr))
                probs = [round(float(p), 4) for p in (e / e.sum()).tolist()]
                chosen = ACTION_LIST[best_seam_idx]
                conf = round(probs[best_seam_idx] * 100, 1)
                dec = {
                    'action': chosen,
                    'custom_branch': '边缘缝隙轮廓匹配',
                    'probabilities': probs,
                    'reason_headline': f'视觉边界像素高契合锁定（契合度 {conf}%）',
                    'reason_detail': '未加载重型编码器，采用直接像素边缘缝隙梯度与相关度匹配，检测到目标拼块相邻连续边界。'
                }
            else:
                f_act, f_probs = _frontier_action((cr, cc), visited, k)
                if k >= 6:
                    f_act, _ = _generalized_coverage_choice((cr, cc), visited, remaining, f_act, k)
                chosen = f_act
                dec = {
                    'action': chosen,
                    'custom_branch': '边缘检索 + 前瞻覆盖',
                    'probabilities': f_probs,
                    'reason_headline': f'拓扑未知区域覆盖推进：{ACTION_NAMES[chosen]}',
                    'reason_detail': '当前四向边缘缝隙未检出邻接高契合特征，按前瞻覆盖策略拓展未探明图块。'
                }

        if policy == 'Cloud':
            if not self.wb.config:
                raise ValueError('尚未配置云端 API')
            obs = Observation(curr_png, tgt_png, (cr, cc), k, remaining, visited)
            cloud_dec = lite_cloud_step(obs, dec, self.wb.config, disable_thinking=self.api_disable_thinking)
            chosen = cloud_dec['action']
            return chosen, parse_decision_enhanced(cloud_dec, chosen, actor='agent', policy='Cloud')

        return dec['action'], parse_decision_enhanced(dec, dec['action'], actor='agent', policy=policy)

    def step(self, action=None):
        with self.lock:
            if not self.active_session:
                raise ValueError('没有正在执行的推演任务，请先点击【开始任务】')

            s = self.active_session
            stype = s['type']

            if stype in ('upload', 'aerial_bank'):
                if s['status'] in ('finished', 'success', 'exhausted'):
                    return self.get_session_state()

                grid = s['grid']
                curr = s['position']
                goal = s['goal']
                r, c = divmod(curr, grid)
                gr, gc = divmod(goal, grid)

                if action:
                    if action not in ACTION_OFFSETS:
                        raise ValueError('无效移动方向')
                    chosen_act = action
                    decision_info = parse_decision_enhanced({}, chosen_act, actor='human', policy='manual')
                else:
                    if s['policy'] == 'manual':
                        raise ValueError('当前策略为人工遥控模式，请使用方向按键移动')
                    if stype == 'upload':
                        chosen_act, decision_info = self._upload_agent_decide(s)
                    else:
                        chosen_act, decision_info = self._aerial_bank_agent_decide(s)

                dr, dc = ACTION_OFFSETS[chosen_act]
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid and 0 <= nc < grid:
                    new_pos = nr * grid + nc
                else:
                    new_pos = curr
                    nr, nc = r, c

                s['step'] += 1
                s['position'] = new_pos
                if new_pos not in s['visited']:
                    s['visited'].append(new_pos)
                s['trajectory'].append({'step': s['step'], 'cell': new_pos, 'action': chosen_act})

                if new_pos == goal:
                    s['status'] = 'success'
                    s['result'] = {'success': True, 'SG_cells': 0, 'steps': s['step']}
                    decision_info['reason_headline'] = '成功到达目标位置'
                elif s['step'] >= s['budget']:
                    s['status'] = 'exhausted'
                    sg = abs(nr - gr) + abs(nc - gc)
                    s['result'] = {'success': False, 'SG_cells': sg, 'steps': s['step']}
                else:
                    s['status'] = 'running'

                s['last_decision'] = decision_info
                return self.get_session_state()

            elif stype == 'camera':
                rid = s['rid']
                try:
                    self.heartbeat()
                    self.wb.controllers.step(rid, self.owner, action=action, kind='session')
                    return self.get_session_state()
                except Exception as e:
                    if any(msg in str(e) for msg in ('不存在', '已释放', '不唯一')):
                        self.active_session = None
                        raise ValueError('地面会话已失效，请重新点击【开始任务】')
                    raise

    def get_session_state(self):
        with self.lock:
            if not self.active_session:
                return None

            s = self.active_session
            stype = s['type']

            if stype in ('upload', 'aerial_bank'):
                curr = s['position']
                goal = s['goal']
                grid = s['grid']
                cr, cc = divmod(curr, grid)
                gr, gc = divmod(goal, grid)
                manhattan = abs(cr - gr) + abs(cc - gc)
                euclidean = ((cr - gr) ** 2 + (cc - gc) ** 2) ** 0.5
                return {
                    'type': stype,
                    'task_key': s.get('task_key'),
                    'policy': s['policy'],
                    'start': s['start'],
                    'goal': goal,
                    'position': curr,
                    'coords': [cr, cc],
                    'goal_coords': [gr, gc],
                    'grid': grid,
                    'step': s['step'],
                    'budget': s['budget'],
                    'status': s['status'],
                    'visited': s['visited'],
                    'trajectory': s['trajectory'],
                    'distance_cells': manhattan,
                    'distance_euclidean': round(euclidean, 2),
                    'decision': s['last_decision'],
                    'success': s['status'] == 'success',
                    'done': s['status'] in ('success', 'exhausted')
                }

            elif stype == 'camera':
                try:
                    raw = self.wb.controllers.state(s['rid'], self.owner, 'session')
                except Exception:
                    self.active_session = None
                    return None
                pos = raw['position']
                goal = s.get('goal', 12)
                if raw.get('result') and 'goal_index' in raw['result']:
                    goal = raw['result']['goal_index']
                grid = s['grid']
                pr, pc = divmod(pos, grid)
                gr, gc = divmod(goal, grid)
                dist_cells = abs(pr - gr) + abs(pc - gc)
                is_done = raw['status'] in ('completed', 'finished', 'failed', 'cancelled') or raw['step'] >= raw['budget'] or (raw.get('result') is not None)
                success = bool((raw.get('result') or {}).get('success', False) or (pos == goal and raw['step'] > 0))
                last_event = raw['trajectory'][-1] if raw.get('trajectory') else {}
                act = last_event.get('action')
                actor = 'human' if last_event.get('origin') == 'human' else 'agent'
                extra_note = '地面拍摄点任务已自动采用 Frontier 规则边界探索。' if s.get('requested_policy') in ('M0', 'Cloud') else '结合地面全景目标线索与俯视航拍网格进行人工/边界探索。'
                return {
                    'type': 'camera',
                    'task_key': s.get('task_key'),
                    'policy': s['policy'],
                    'start': s['start'],
                    'goal': goal,
                    'position': pos,
                    'coords': [pr, pc],
                    'goal_coords': [gr, gc],
                    'grid': grid,
                    'step': raw['step'],
                    'budget': raw['budget'],
                    'status': 'success' if success else 'exhausted' if is_done else 'running',
                    'visited': raw['visited'],
                    'trajectory': raw['trajectory'],
                    'distance_cells': dist_cells,
                    'distance_euclidean': round(((pr - gr) ** 2 + (pc - gc) ** 2) ** 0.5, 2),
                    'decision': {
                        'branch': '地面跨视角探索' if actor == 'agent' else '人工遥控控制',
                        'action': act,
                        'action_name': ACTION_NAMES.get(act, act) if act else '待开始',
                        'reason_headline': f'当前推进方向：{ACTION_NAMES.get(act, "待开始")}' if act else '地面跨视角任务就绪',
                        'reason_detail': extra_note,
                        'probabilities': [1.0 if d == act else 0.0 for d in ACTION_LIST] if act else [0.25, 0.25, 0.25, 0.25]
                    },
                    'success': success,
                    'done': is_done
                }

    def get_image(self, cell=None, is_target=False, task_key=None, observer=False, full=False):
        """Return tile PNG bytes strictly isolating human observer previews from policy memory."""
        with self.lock:
            # 1. 人类观察者预览路径（完整隔绝底层智能体）
            if observer or (task_key and (not self.active_session or self.active_session.get('task_key') != task_key)):
                tkey = task_key or (self.active_session.get('task_key') if self.active_session else None)
                if not tkey:
                    return self._placeholder_png()
                task = next((t for t in self.tasks_catalog if t['task_key'] == tkey), None)
                if not task:
                    return self._placeholder_png()
                did = task['dataset_id']
                if did == 'upload':
                    u_data = self.uploads.get(task['upload_id'])
                    if not u_data:
                        return self._placeholder_png()
                    if full:
                        return u_data['preview']
                    idx = task.get('default_goal', len(u_data['tiles']) - 1) if is_target else int(cell if cell is not None else 0)
                    idx = max(0, min(idx, len(u_data['tiles']) - 1))
                    return u_data['tiles'][idx]
                elif did == 'aerial_bank':
                    try:
                        assets = self._load_area_assets(task['bank_id'], task['area'], task['grid'])
                        idx = task['default_goal'] if is_target else int(cell if cell is not None else task['start'])
                        idx = max(0, min(idx, task['grid'] * task['grid'] - 1))
                        return self._get_area_tile_png(assets, idx)
                    except Exception:
                        return self._placeholder_png()
                elif did == CAMERA:
                    try:
                        c = 'target' if is_target else int(cell if cell is not None else task['start'])
                        return self.wb.capturepoint_sessions.catalogue.image(tkey, c)
                    except Exception:
                        return self._placeholder_png()

            # 2. 活动推演会话读取路径
            if not self.active_session:
                return self._placeholder_png()

            s = self.active_session
            stype = s['type']

            if stype == 'upload':
                u_data = self.uploads.get(s['upload_id'])
                if not u_data:
                    return self._placeholder_png()
                if full:
                    return u_data['preview']
                idx = s['goal'] if is_target else (int(cell) if cell is not None else s['position'])
                idx = max(0, min(idx, len(u_data['tiles']) - 1))
                return u_data['tiles'][idx]

            elif stype == 'aerial_bank':
                try:
                    assets = self._load_area_assets(s['bank_id'], s['area'], s['grid'])
                    idx = s['goal'] if is_target else (int(cell) if cell is not None else s['position'])
                    idx = max(0, min(idx, s['grid'] * s['grid'] - 1))
                    return self._get_area_tile_png(assets, idx)
                except Exception:
                    return self._placeholder_png()

            elif stype == 'camera':
                rid = s['rid']
                try:
                    if is_target:
                        return self.wb.controllers.image(rid, self.owner, 'target', reveal=True, kind='session')
                    target_cell = int(cell) if cell is not None else s.get('start', 0)
                    return self.wb.controllers.image(rid, self.owner, target_cell, reveal=True, kind='session')
                except Exception:
                    return self._placeholder_png()

            return self._placeholder_png()

    def _placeholder_png(self):
        from PIL import Image
        im = Image.new('RGB', (64, 64), color=(230, 235, 242))
        buf = BytesIO()
        im.save(buf, format='PNG')
        return buf.getvalue()


def make_lite_server(manager, port=8768):
    class LiteHandler(BaseHTTPRequestHandler):
        def allowed(self):
            return True

        def send_data(self, data, mime, status=200):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, value, status=200):
            self.send_data(json.dumps(value, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8', status)

        def do_GET(self):
            u = urlsplit(self.path)
            q = parse_qs(u.query)
            get_arg = lambda k, d='': q.get(k, [d])[0]

            try:
                if u.path in ('/', '/index.html'):
                    p = STATIC_DIR / 'geonav_lite.html'
                    self.send_data(p.read_bytes(), 'text/html; charset=utf-8')
                elif u.path == '/geonav_lite.css':
                    p = STATIC_DIR / 'geonav_lite.css'
                    self.send_data(p.read_bytes(), 'text/css; charset=utf-8')
                elif u.path == '/geonav_lite.js':
                    p = STATIC_DIR / 'geonav_lite.js'
                    self.send_data(p.read_bytes(), 'text/javascript; charset=utf-8')
                elif u.path == '/api/meta':
                    self.send_json(manager.meta())
                elif u.path == '/api/heartbeat':
                    self.send_json(manager.heartbeat())
                elif u.path == '/api/state':
                    self.send_json(manager.get_session_state() or {})
                elif u.path == '/api/image':
                    is_target = get_arg('target') in ('1', 'true')
                    observer = get_arg('observer') in ('1', 'true')
                    full = get_arg('full') in ('1', 'true')
                    task_key = get_arg('task_key') or None
                    cell = get_arg('cell')
                    cell_val = int(cell) if cell and cell.isdigit() else None
                    data = manager.get_image(
                        cell=cell_val,
                        is_target=is_target,
                        task_key=task_key,
                        observer=observer,
                        full=full
                    )
                    self.send_data(data, 'image/png')
                else:
                    self.send_json({'error': '未找到资源'}, 404)
            except Exception as e:
                self.send_json({'error': f"处理请求异常: {str(e)}"}, 500)

        def do_POST(self):
            u = urlsplit(self.path)
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length) if length > 0 else b'{}'

                if u.path == '/api/upload':
                    try:
                        d = json.loads(body)
                        b64_str = d.get('image', '').split(',')[-1]
                        raw_bytes = base64.b64decode(b64_str)
                        name = d.get('name', '临时场景')
                        grid = int(d.get('grid', 5))
                    except Exception:
                        raw_bytes = body
                        name = '临时图片'
                        grid = 5
                    task = manager.handle_upload(raw_bytes, name, grid=grid)
                    self.send_json(task)
                    return

                d = json.loads(body) if body else {}

                if u.path == '/api/upload/delete':
                    res = manager.delete_upload(d.get('task_key'))
                    self.send_json(res)
                elif u.path == '/api/create':
                    task_key = d.get('task_key')
                    policy = d.get('policy', 'M0')
                    c_start = d.get('start')
                    c_goal = d.get('goal')
                    seed = d.get('seed', 0)
                    state = manager.create_session(task_key, policy, c_start, c_goal, seed)
                    self.send_json(state)
                elif u.path == '/api/step':
                    action = d.get('action')
                    state = manager.step(action)
                    self.send_json(state)
                elif u.path == '/api/config':
                    cfg = manager.save_api_config(
                        d.get('base_url', ''),
                        d.get('model', ''),
                        d.get('api_key', ''),
                        d.get('timeout', 30),
                        d.get('disable_thinking', True)
                    )
                    self.send_json(cfg)
                elif u.path == '/api/config/test':
                    res = manager.test_api_connection(
                        d.get('base_url', ''),
                        d.get('model', ''),
                        d.get('api_key', ''),
                        d.get('timeout', 20),
                        d.get('disable_thinking')
                    )
                    self.send_json(res)
                elif u.path == '/api/heartbeat':
                    self.send_json(manager.heartbeat())
                elif u.path == '/api/reset':
                    manager.active_session = None
                    self.send_json({'reset': True})
                else:
                    self.send_json({'error': '未识别的接口'}, 404)
            except Exception as e:
                self.send_json({'error': f"操作失败: {str(e)}"}, 400)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), LiteHandler)
    server.manager = manager
    return server


def main():
    parser = argparse.ArgumentParser(description="GeoNav Lite 极简演示工作台")
    parser.add_argument('--port', type=int, default=8768, help="服务端口，默认 8768")
    args = parser.parse_args()
    print(f"正在启动 GeoNav Lite 演示服务在 http://127.0.0.1:{args.port}/ ...")
    mgr = LiteServerManager()
    server = make_lite_server(mgr, args.port)
    print(f"GeoNav Lite 演示服务已就绪！访问地址：http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已优雅退出。")
    finally:
        mgr.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
