"""Independent audit for the two-stage spatial-area frozen-M0 evaluation.

This script rechecks pixels and feature equations, replays every saved action
from frozen checkpoints using only the public observation, and reconstructs
all navigation/probe summaries from per-record evidence. It never writes a
verdict; the evaluator owns finalization after each stage audit passes.
"""
from collections import Counter, deque
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from env.episode import ACTIONS
from env.spatial_area import SpatialAreaEpisode, SpatialAreaGridEnv, PROTOCOL
from eval import spatial_area_confirmation as protocol

ROOT, SRC, OUT, DATA, PREVIOUS, OLD = (protocol.ROOT, protocol.SRC, protocol.OUT,
                                       protocol.DATA, protocol.PREVIOUS, protocol.OLD)
SEEDS, CONDITIONS, RULES = protocol.SEEDS, protocol.CONDITIONS, protocol.RULES
PROBE_CONDITIONS, STRATA, CACHE_NAMES = protocol.PROBE_CONDITIONS, protocol.STRATA, protocol.CACHE_NAMES
STAGES, BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = protocol.STAGES, protocol.BOOTSTRAP_SEED, protocol.BOOTSTRAP_RESAMPLES
PROBE_WRONG_SEED = protocol.PROBE_WRONG_SEED
CLASS_NAMES = ('up', 'right', 'down', 'left', 'not_adjacent')
MOVES = dict(ACTIONS)


def need(value, message):
    if not value:
        raise ValueError('spatial-area confirmation audit: ' + message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)


def same_values(left, right, atol=2e-10, rtol=2e-10):
    """Recursive JSON comparison with tight tolerance for recomputed floats."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_values(left[k], right[k], atol, rtol) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same_values(a, b, atol, rtol) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)
    return left == right


def independent_distance(a, b):
    return abs(a // 10 - b // 10) + abs(a % 10 - b % 10)


def independent_seed(seed, *labels):
    payload = json.dumps([seed, *labels], ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(sha256(payload.encode('utf-8')).digest(), 'big')


def manual_profile(rgb):
    """Reconstruct the 4x3x64x3 profile from decoded patch pixels."""
    pixels = np.asarray(rgb, dtype=np.float32) / np.float32(255)
    need(pixels.shape == (300, 300, 3), 'profile input must be a decoded 300x300 RGB patch')
    cuts = np.linspace(0, 300, 65, dtype=np.int64)
    result = np.empty((4, 3, 64, 3), dtype=np.float32)
    for side in range(4):
        for scale, width in enumerate((1, 4, 8)):
            if side == 0:
                along = pixels[:width, :, :].mean(axis=0)
            elif side == 1:
                along = pixels[:, 300-width:, :].mean(axis=1)
            elif side == 2:
                along = pixels[300-width:, :, :].mean(axis=0)
            else:
                along = pixels[:, :width, :].mean(axis=1)
            for index, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
                result[side, scale, index] = along[start:end].mean(axis=0)
    return result


def manual_edge_features(target, current):
    """Independent directional RGB, gradient and correlation equations."""
    target = np.asarray(target, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    need(target.shape == current.shape == (4, 3, 64, 3), 'single-pair profile shape')
    result = np.empty((4, 5), dtype=np.float32)
    opposite = (2, 3, 0, 1)
    for side in range(4):
        a = current[side]
        b = target[opposite[side]]
        for scale in range(3):
            result[side, scale] = np.abs(a[scale]-b[scale]).mean()
        af, bf = a[0], b[0]
        result[side, 3] = np.abs(np.diff(af, axis=0)-np.diff(bf, axis=0)).mean()
        ad = af.astype(np.float64)
        bd = bf.astype(np.float64)
        ac = ad-ad.mean(axis=0, keepdims=True)
        bc = bd-bd.mean(axis=0, keepdims=True)
        numerator = float((ac*bc).sum())
        denominator = math.sqrt(float((ac*ac).sum())*float((bc*bc).sum()))
        result[side, 4] = np.clip(numerator/denominator if denominator > 1e-8 else 0., -1., 1.)
    return result.reshape(20)


def manual_cue_features(target, current, target_local, current_local):
    """Recompute the old global/local target-cue vector without agent helpers."""
    target = np.asarray(target, np.float32)
    current = np.asarray(current, np.float32)
    tl = np.asarray(target_local, np.float32)
    cl = np.asarray(current_local, np.float32)
    need(target.shape == current.shape == (512,), 'cue global vectors must have width 512')
    need(tl.shape == cl.shape == (4, 768), 'cue local vectors must be 4x768')
    a = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    b = current / np.maximum(np.linalg.norm(current, axis=-1, keepdims=True), 1e-8)
    an = tl / np.maximum(np.linalg.norm(tl, axis=-1, keepdims=True), 1e-8)
    bn = cl / np.maximum(np.linalg.norm(cl, axis=-1, keepdims=True), 1e-8)
    relations = np.einsum('ik,jk->ij', an, bn).reshape(16).astype(np.float32)
    return np.concatenate((a, b, np.asarray([(a*b).sum()], np.float32), relations)).astype(np.float32)


def manual_head_raw(agent, features):
    """Evaluate the linear/LayerNorm/tanh head equations, returning true raw logits."""
    x = torch.as_tensor(np.asarray(features, np.float32), dtype=torch.float32, device=agent.device).reshape(1, -1)
    state = agent.head.state_dict()
    need(x.shape[1] == 1061, 'manual EdgeTargetCue input width')
    hidden = F.linear(x[:, :1041].contiguous(), state['network.0.weight'], state['network.0.bias'])
    hidden = hidden+F.linear(x[:, 1041:], state['edge_projection.weight'])
    hidden = F.layer_norm(hidden, (128,), state['network.1.weight'], state['network.1.bias'], agent.head.network[1].eps)
    hidden = torch.tanh(hidden)
    raw = F.linear(hidden, state['network.3.weight'], state['network.3.bias'])
    return raw[0].detach().cpu().numpy()


def joint_probabilities(raw):
    raw = torch.as_tensor(np.asarray(raw, np.float32)).reshape(1, 5)
    joint_log = torch.cat((F.logsigmoid(raw[:, 4:5]) + F.log_softmax(raw[:, :4], dim=-1),
                           F.logsigmoid(-raw[:, 4:5])), dim=-1)
    return torch.softmax(joint_log, dim=-1)[0].numpy()


def independent_profile_preprocess(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
        x = np.asarray(rgb, dtype=np.float32)/np.float32(255)
    mean = np.asarray([.3670, .3827, .3338], np.float32)
    std = np.asarray([.2209, .1975, .1988], np.float32)
    return torch.from_numpy(((x-mean)/std).transpose(2, 0, 1))


def region_source(region, cell):
    row, col = divmod(cell, 10)
    return region['sources'][2*(row//5)+(col//5)]['id']


def verify_region_geometry(region):
    x, y = float(region['x']), float(region['y'])
    expected = [x, y-3000, x+3000, y]
    need(np.allclose(region['bounds_m'], expected, rtol=0, atol=1e-8), 'continuous mosaic 3000m footprint')
    need(len(region['sources']) == 4 and len({s['id'] for s in region['sources']}) == 4, 'four distinct source rasters')
    for index, source in enumerate(region['sources']):
        sx, sy = x+1500*(index % 2), y-1500*(index // 2)
        bounds = [sx, sy-1500, sx+1500, sy]
        need(np.allclose([source['x'], source['y']], [sx, sy], rtol=0, atol=.001), '2x2 source lattice continuity')
        need(np.allclose(source['bounds_m'], bounds, rtol=0, atol=.001), 'source raster footprint continuity')


def expected_task_bank(regions):
    tasks, strata = [], {}
    for region_index, region in enumerate(regions):
        area, split = region['area'], region['split']
        used = set()
        for stratum in STRATA:
            distances = range(12, 17) if stratum == 'long_distance' else range(8, 13)
            for distance in distances:
                rng = random.Random(independent_seed(6203, PROTOCOL, area, stratum, str(distance)))
                for draw in range(5):
                    orientation = side = None
                    if stratum == 'seam_target':
                        orientation = 'vertical' if (region_index+distance+draw) % 2 == 0 else 'horizontal'
                        side = 4+((region_index+distance+draw//2) % 2)
                    pairs = []
                    for start in range(100):
                        for goal in range(100):
                            row, col = divmod(goal, 10)
                            if stratum == 'seam_target':
                                allowed = (col == side and row not in (4, 5)) if orientation == 'vertical' else (row == side and col not in (4, 5))
                            elif stratum == 'interior_target':
                                allowed = row not in (4, 5) and col not in (4, 5)
                            else:
                                allowed = True
                            if allowed and (start, goal) not in used and independent_distance(start, goal) == distance:
                                pairs.append((start, goal))
                    need(bool(pairs), 'nonempty independently reconstructed task pool')
                    start, goal = rng.choice(pairs)
                    used.add((start, goal))
                    episode_id = f'spatial_{area}_{stratum}_d{distance}_{draw:03d}'
                    tasks.append(dict(episode_id=episode_id, split=split, area=area, start=start, goal=goal,
                                      dist=distance, budget=20, grid_size=10, protocol=PROTOCOL,
                                      source_tile=region['source_tile']))
                    strata[episode_id] = dict(stratum=stratum, initial_distance=distance,
                                              seam_orientation=orientation, seam_goal_side=side,
                                              goal_on_source_border=((goal//10 in (4, 5)) != (goal%10 in (4, 5))),
                                              evaluation_only=True)
        need(len(used) == 75, 'all 75 routes per region are unique')
    return tasks, strata


def expected_wrong_targets(tasks):
    result = {}
    for task in tasks:
        pool = [cell for cell in range(100) if cell not in (task['start'], task['goal'])
                and independent_distance(task['start'], cell) == task['dist']]
        fallback = [cell for cell in range(100) if cell not in (task['start'], task['goal'])]
        rng = random.Random(independent_seed(6211, PROTOCOL, task['episode_id'], 'wrong'))
        result[task['episode_id']] = dict(cue_cell=rng.choice(pool or fallback), matched_distance=bool(pool))
    return result


def expected_probe_bank(regions):
    result = []
    for region in regions:
        pairs = []
        for row in range(10):
            pairs.extend(((row*10+4, row*10+5, 'right'), (row*10+5, row*10+4, 'left')))
        for col in range(10):
            pairs.extend(((40+col, 50+col, 'down'), (50+col, 40+col, 'up')))
        for index, (current, target, direction) in enumerate(pairs):
            row, col = divmod(current, 10)
            tr, tc = divmod(target, 10)
            same = (2*tr-row)*10+2*tc-col
            nonadjacent = (2*row-tr)*10+2*col-tc
            reverse = {'up': 'down', 'right': 'left', 'down': 'up', 'left': 'right'}[direction]
            for kind, cell, label in (('cross_source_adjacent', current, direction),
                                      ('same_source_adjacent', same, reverse),
                                      ('nonadjacent_matched_target', nonadjacent, 'not_adjacent')):
                result.append(dict(probe_id=f"probe_{region['area']}_{index:03d}_{kind}", area=region['area'],
                                   split=region['split'], source_tile=region['source_tile'], kind=kind, pair_id=index,
                                   current_cell=cell, target_cell=target, expected_class=label,
                                   distance=independent_distance(cell, target), evaluation_only=True, counts_as_navigation=False))
    return result


def expected_probe_wrong(probes):
    groups = {}
    for probe in probes:
        groups.setdefault((probe['area'], probe['pair_id']), []).append(probe)
    rng = np.random.default_rng(PROBE_WRONG_SEED)
    plan = {}
    for group in groups.values():
        need(len(group) == 3 and len({p['target_cell'] for p in group}) == 1, 'matched probe triple')
        excluded = {group[0]['target_cell'], *(p['current_cell'] for p in group)}
        cue = int(rng.choice([cell for cell in range(100) if cell not in excluded]))
        for probe in group:
            plan[probe['probe_id']] = dict(cue_cell=cue, shared_matched_triple=True,
                                          distance_matched=independent_distance(probe['current_cell'], cue) == probe['distance'])
    return plan


def verify_design_and_data(stage):
    reg = read(OUT/'预登记.json')
    for field in ('protected_sha256', 'source_sha256', 'input_sha256', 'encoder_sha256', 'frozen_output_sha256'):
        for name, expected in reg[field].items():
            need(digest(ROOT/name) == expected, f'frozen {field} binding {name}')
    expected_stage = {'engineering': dict(split='dev', maps=4, tasks_each=300, records=4200, probe_records=4320),
                      'confirmation': dict(split='test', maps=10, tasks_each=750, records=10500, probe_records=10800)}
    need(reg['protocol'] == PROTOCOL and reg['weights'] == list(SEEDS), 'protocol and three frozen M0 weights')
    need(reg['conditions'] == list(CONDITIONS) and reg['rules'] == list(RULES)
         and reg['probe_conditions'] == list(PROBE_CONDITIONS), 'all frozen navigation/probe arms')
    need(reg['stages'] == {k: v for k, v in expected_stage.items()}, 'registered stage denominators')
    need((reg['planned_neural'], reg['planned_rules'], reg['planned_total'], reg['planned_probe_records'])
         == (12600, 2100, 14700, 15120), 'all planned cohort records')
    need(reg['navigation_gate'] == dict(primary_SR=.60, SR_gain=.05, positive_weights=2,
         primary_SG_no_worse=True, pooled_SG_no_worse=True, source95_SR_lower_positive=True),
         'fixed navigation gates')
    need(reg['seam_gate'] == dict(raw_correct_acceptance=.90, raw_nonadjacent_false_acceptance=.05,
         positive_weights=2, confidence_intervals='descriptive whole-region bootstrap'), 'fixed probe gates')
    need(reg['bootstrap']['seed'] == BOOTSTRAP_SEED and reg['bootstrap']['resamples'] == BOOTSTRAP_RESAMPLES
         and 'whole9km2region' in reg['bootstrap']['unit'], 'paired whole-map bootstrap registration')
    need(reg['probe_wrong_seed'] == PROBE_WRONG_SEED and not reg['probe_wrong_distance_matching_required'], 'probe false-cue rule')
    need(reg['new_training_steps'] == 0 and reg['cloud_calls'] == 0 and not reg['default_changed'], 'frozen M0/no-training scope')
    registration_seal = read(OUT/'预登记封存.json')
    expected_seal = dict(registration_sha256=digest(OUT/'预登记.json'),
                         protocol_sha256=digest(OUT/'执行协议.md'),
                         default_sha256=digest(ROOT/'project/local_policy_default.json'),
                         navigation_records=14700, probe_records=15120)
    need(registration_seal == expected_seal, 'complete immutable preregistration receipt')
    default_path = ROOT/'project/local_policy_default.json'
    default = read(default_path)
    need(digest(default_path) == reg['default_sha256'] and default['architecture'] == 'Small256'
         and default['cue_architecture'] == 'EdgeTargetCueHead' and default['selected_arm'] == 'EdgeTargetCue'
         and default['inference_condition'] == 'CueFull' and default['version'] == 'local-policy-default-v2'
         and default['thresholds'] == {str(seed): .5 for seed in SEEDS}, 'unchanged frozen M0 default and threshold')
    explorer_records = {record['seed']: record for record in default['checkpoints']}
    head_records = {record['seed']: record for record in default['cue_heads']}
    need(set(explorer_records) == set(head_records) == set(SEEDS), 'three paired explorer and cue-head weights')
    for seed in SEEDS:
        config_path = OLD/f'Edge_s{seed}'/'配置.json'
        config = read(config_path)
        explorer_path = OLD/f'Edge_s{seed}'/'explorer.pt'
        head_path = OLD/f'Edge_s{seed}'/'head.pt'
        need(config['seed'] == seed and config['arm'] == 'Edge' and config['threshold'] == .5,
             'frozen paired Edge configuration')
        need(digest(explorer_path) == explorer_records[seed]['sha256'] == config['explorer_sha256'],
             'unchanged NoTarget Small256 checkpoint '+str(seed))
        need(digest(head_path) == head_records[seed]['sha256'] == config['head_sha256'],
             'unchanged EdgeTargetCue checkpoint '+str(seed))
    for name in protocol.MEANS:
        record = default['means'][name]
        need(digest(ROOT/record['path']) == record['sha256'] == digest(OLD/name), 'unchanged fitted mean '+name)
    if stage == 'confirmation':
        protocol.check_engineering()

    data_manifest = read(DATA/'数据清单.json')
    regions = data_manifest['regions']
    stage_meta = expected_stage[stage]
    chosen = [r for r in regions if r['split'] == stage_meta['split']]
    need(len(chosen) == stage_meta['maps'], 'stage map role selection')
    for region in regions:
        verify_region_geometry(region)
    for i, a in enumerate(regions):
        for b in regions[i+1:]:
            dx = max(0., a['bounds_m'][0]-b['bounds_m'][2], b['bounds_m'][0]-a['bounds_m'][2])
            dy = max(0., a['bounds_m'][1]-b['bounds_m'][3], b['bounds_m'][1]-a['bounds_m'][3])
            need(math.hypot(dx, dy) >= 3000-.001, 'new full-region footprints remain spatially separated')

    tasks, strata = expected_task_bank(regions)
    wrong = expected_wrong_targets(tasks)
    probes = expected_probe_bank(regions)
    probe_wrong = expected_probe_wrong(probes)
    for filename, expected in (('导航任务.json', tasks), ('任务分层.json', strata), ('错误目标计划.json', wrong),
                               ('邻接诊断探针.json', probes), ('探针错误目标计划.json', probe_wrong)):
        actual = read(OUT/filename)
        need(actual == expected, 'independent deterministic reconstruction of '+filename)
    for filename in ('导航任务.json', '任务分层.json', '错误目标计划.json', '邻接诊断探针.json'):
        need((OUT/filename).read_bytes() == (DATA/filename).read_bytes(), 'frozen data copy bytes '+filename)
    need(Counter(t['split'] for t in tasks) == {'dev': 300, 'test': 750}, 'fixed 4/10 task split')
    need(sum(not x['matched_distance'] for x in wrong.values()) == 69, 'preserved wrong-distance exception count')
    need(len(probes) == 1680 and len(probe_wrong) == 1680, 'all matched probe records and false-target bindings')
    for region in chosen:
        entries = [p for p in probes if p['area'] == region['area']]
        need(Counter(p['kind'] for p in entries) == {'cross_source_adjacent': 40,
             'same_source_adjacent': 40, 'nonadjacent_matched_target': 40}, '40 per probe type/map')
    return reg, data_manifest, chosen, tasks, strata, wrong, probes, probe_wrong


def reextract_stage(stage, frozen):
    """Rebuild native cells, profiles and encoder banks for this stage only."""
    from transformers import CLIPVisionModelWithProjection
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder = CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),
              local_files_only=True).to('cuda').eval()
    encoder.requires_grad_(False)
    g, l, p = protocol.load_banks(stage)
    regions = [r for r in read(DATA/'数据清单.json')['regions'] if r['split'] == STAGES[stage][1]]
    keys = {r['split']+'__'+r['area'] for r in regions}
    need(set(g) == set(l) == set(p) == keys, 'feature cache keys restricted to current stage maps')
    provenance = read(DATA/'图块来源.json')
    expected_prov_keys = {r['area']+'/'+str(i) for r in regions for i in range(100)}
    need(expected_prov_keys <= set(provenance), 'complete cell source provenance')
    total = 0
    with torch.inference_mode():
        for region in regions:
            area, split = region['area'], region['split']
            key = split+'__'+area
            native_mosaic = np.empty((3000, 3000, 3), dtype=np.uint8)
            for index, source in enumerate(region['sources']):
                path = ROOT/source['raw_path']
                need(digest(path) == source['raw_sha256'], 'native source file SHA')
                with Image.open(path) as im:
                    im.load()
                    rgb = np.asarray(im.convert('RGB'), dtype=np.uint8)
                need(rgb.shape == (1500, 1500, 3), 'four native 1500x1500 RGB source tiles')
                need(sha256(rgb.tobytes()).hexdigest() == source['rgb_sha256'], 'native source decoded RGB SHA')
                rr, cc = divmod(index, 2)
                native_mosaic[rr*1500:(rr+1)*1500, cc*1500:(cc+1)*1500] = rgb
            with Image.open(DATA/'mosaics'/(area+'.tiff')) as mosaic_file:
                mosaic_file.load()
                stored_mosaic = np.asarray(mosaic_file.convert('RGB'), dtype=np.uint8)
            need(np.array_equal(stored_mosaic, native_mosaic), 'source-tile assembly equals saved continuous mosaic')
            need(sha256(native_mosaic.tobytes()).hexdigest() == region['published_mosaic_rgb_sha256'], 'mosaic decoded RGB SHA')
            vg, vl, vp, inputs = [], [], [], []
            for cell in range(100):
                row, col = divmod(cell, 10)
                crop = native_mosaic[row*300:(row+1)*300, col*300:(col+1)*300]
                path = DATA/'patches'/split/area/f'patch_{cell}.jpg'
                buffer = BytesIO()
                Image.fromarray(native_mosaic).crop((col*300, row*300, (col+1)*300, (row+1)*300)).save(buffer, format='JPEG', quality=75)
                need(path.read_bytes() == buffer.getvalue(), 'JPEG75 patch is exact native-pixel crop')
                need(sha256(crop.tobytes()).hexdigest() == provenance[area+'/'+str(cell)]['native_rgb_sha256'], 'native cell pixel SHA')
                with Image.open(path) as image:
                    image.load()
                    decoded = np.asarray(image.convert('RGB'), dtype=np.uint8)
                need(sha256(decoded.tobytes()).hexdigest() == provenance[area+'/'+str(cell)]['jpeg_rgb_sha256'], 'decoded JPEG RGB SHA')
                np.testing.assert_array_equal(manual_profile(decoded), p[key][cell])
                source_index = 2*(row//5)+(col//5)
                source = region['sources'][source_index]
                expected_window = [(col%5)*300, (row%5)*300, (col%5+1)*300, (row%5+1)*300]
                expected_mosaic_window = [col*300, row*300, (col+1)*300, (row+1)*300]
                expected_bounds = [region['x']+col*300, region['y']-(row+1)*300,
                                   region['x']+(col+1)*300, region['y']-row*300]
                prov = provenance[area+'/'+str(cell)]
                need(prov['source_id'] == source['id'] and prov['source_raw_sha256'] == source['raw_sha256']
                     and prov['source_pixel_window'] == expected_window
                     and prov['mosaic_pixel_window'] == expected_mosaic_window
                     and np.allclose(prov['projected_bounds_m'], expected_bounds, rtol=0, atol=1e-8),
                     'independent cell-to-source and projected-boundary geometry')
                # Reproduce the frozen encoder input transform without importing the evaluator preprocessor.
                inputs.append(independent_profile_preprocess(path))
            for offset in range(0, 100, 25):
                encoded = encoder(torch.stack(inputs[offset:offset+25]).to('cuda'))
                vg.append(encoded.image_embeds.float().cpu().numpy())
                tokens = encoder.vision_model.post_layernorm(encoded.last_hidden_state[:, 1:]).reshape(25, 7, 7, 768)
                vl.append(torch.stack((tokens[:, :3, :3].mean((1, 2)), tokens[:, :3, 3:].mean((1, 2)),
                                       tokens[:, 3:, :3].mean((1, 2)), tokens[:, 3:, 3:].mean((1, 2))), dim=1).float().cpu().numpy())
            np.testing.assert_array_equal(np.concatenate(vg), g[key])
            np.testing.assert_array_equal(np.concatenate(vl), l[key])
            need(g[key].shape == (100, 512) and l[key].shape == (100, 4, 768)
                 and p[key].shape == (100, 4, 3, 64, 3), 'global/local/profile cache shapes')
            total += 100
    for name, expected in frozen['files_sha256'].items():
        need(digest(ROOT/name) == expected, 'stage feature cache SHA '+name)
    del encoder
    torch.cuda.empty_cache()
    return total, (g, l, p), provenance


def independent_diagnostic(cell, goal, cue, action, accepted, remaining, region):
    destination, oob = advance(cell, action)
    source = lambda value: region_source(region, value)
    distance = independent_distance(cell, goal)
    adjacency = distance == 1 and remaining > 0
    return dict(current_source=source(cell), true_target_source=source(goal), given_target_source=source(cue),
                destination_source=source(destination), crossed_source_boundary=source(cell) != source(destination),
                true_target_distance=distance, actionable_true_adjacency=adjacency,
                true_adjacency_cross_source=adjacency and source(cell) != source(goal), cue_accepted=accepted,
                accepted_immediate_true_hit=accepted and not oob and destination == goal,
                accepted_immediate_given_hit=accepted and not oob and destination == cue,
                effective_move_m=0 if oob else 300)


def advance(cell, action):
    row, col = divmod(cell, 10)
    dr, dc = MOVES[action]
    nr, nc = row+dr, col+dc
    if not (0 <= nr < 10 and 0 <= nc < 10):
        return cell, True
    return nr*10+nc, False


def independent_rule(name, obs):
    current = obs.position[0]*10+obs.position[1]
    visited = set(obs.visited)
    if name == 'Frontier':
        queue = deque([(current, None)])
        seen = {current}
        while queue:
            cell, first = queue.popleft()
            for action in MOVES:
                neighbor, oob = advance(cell, action)
                if oob:
                    continue
                chosen = first or action
                if neighbor not in visited:
                    return chosen
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, chosen))
        return next(action for action in MOVES if not advance(current, action)[1])
    need(name == 'FixedRegion', 'known frozen public-state rule')
    visits = Counter(obs.visited)
    region_index = lambda cell: 3*(cell//10//4)+(cell%10//4)
    target_region = next((r for r in range(9) if any(region_index(cell) == r and visits[cell] == 0 for cell in range(100))), 0)
    destinations = {action: advance(current, action)[0] for action in MOVES}
    novel = [action for action in MOVES if destinations[action] not in visited]
    scores = [(region_index(destinations[a]) != target_region, region_index(destinations[a]),
               visits[destinations[a]], i, a) for i, a in enumerate(MOVES) if not novel or a in novel]
    return min(scores)[-1]


def _manual_gru(agent, feature, hidden):
    """Small256 GRUCell equations, independent of the policy's step method."""
    model = agent.explorer
    x = torch.as_tensor(feature, dtype=torch.float32, device=agent.device).reshape(1, -1)
    pre = F.linear(x, model.input[0].weight, model.input[0].bias)
    pre = F.layer_norm(pre, (256,), model.input[1].weight, model.input[1].bias, model.input[1].eps)
    pre = torch.tanh(pre)
    if hidden is None:
        hidden = torch.zeros((1, 256), dtype=pre.dtype, device=pre.device)
    ih = F.linear(pre, model.gru.weight_ih, model.gru.bias_ih)
    hh = F.linear(hidden, model.gru.weight_hh, model.gru.bias_hh)
    ir, iz, inn = ih.chunk(3, dim=1)
    hr, hz, hn = hh.chunk(3, dim=1)
    reset = torch.sigmoid(ir+hr)
    update = torch.sigmoid(iz+hz)
    candidate = torch.tanh(inn+reset*hn)
    new_hidden = candidate+update*(hidden-candidate)
    logits = F.linear(new_hidden, model.actor.weight, model.actor.bias)
    row, col = float(feature[1024]), float(feature[1025])
    mask = torch.tensor([[row > 0, col < 1, row < 1, col > 0]], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return logits[0].detach().cpu().numpy(), new_hidden


def independent_policy_step(agent, obs, cell, cue, key, banks, hidden, step_number):
    g, l, p = banks
    counts = np.zeros(25, np.float32)
    for visited in obs.visited:
        counts[(visited//10//2)*5+(visited%10//2)] += 1
    counts = np.minimum(counts, 3)/3
    feature = np.concatenate((agent.em, g[key][cell], np.asarray([obs.position[0]/9, obs.position[1]/9,
                              obs.remaining_budget/20], np.float32), counts)).astype(np.float32)
    explorer_logits, hidden = _manual_gru(agent, feature, hidden)
    explorer_action = tuple(MOVES)[int(np.argmax(explorer_logits))]
    use_mean = agent.condition == 'CueMean'
    target_g = agent.hm if use_mean else g[key][cue]
    target_l = agent.lm if use_mean else l[key][cue]
    target_p = agent.pm if use_mean else p[key][cue]
    cue = np.concatenate((manual_cue_features(target_g, g[key][cell], target_l, l[key][cell]),
                          manual_edge_features(target_p, p[key][cell]))).astype(np.float32)
    raw = manual_head_raw(agent, cue)
    probabilities = joint_probabilities(raw)
    top = int(np.argmax(probabilities))
    choice, reason = None, ''
    if agent.condition == 'Baseline':
        reason = 'uncalibrated_abstain'
    elif top == 4:
        reason = 'not_adjacent'
    elif probabilities[top] < .5:
        reason = 'low_confidence'
    else:
        proposed = CLASS_NAMES[top]
        destination, oob = advance(cell, proposed)
        if oob:
            reason = 'illegal_top_direction'
        elif destination in obs.visited:
            reason = 'visited_top_destination'
        else:
            reason, choice = 'accepted', proposed
    return dict(step=step_number, public_position=list(obs.position), public_visited=list(obs.visited),
                remaining_budget=obs.remaining_budget, explorer_action=explorer_action,
                explorer_logits=explorer_logits.tolist(), action=choice or explorer_action, cue_action=choice,
                reason=reason, probabilities=probabilities.tolist(),
                current_image_sha256=sha256(obs.current_image).hexdigest(),
                target_image_sha256=sha256(obs.target_image).hexdigest(),
                explorer_features_sha256=sha256(feature.tobytes()).hexdigest(),
                cue_features_sha256=sha256(cue.tobytes()).hexdigest()), hidden


def _check_decision(saved, replayed, prefix):
    need(saved.keys() == replayed.keys(), prefix+' full decision fields')
    for key in saved:
        if key in ('explorer_logits', 'probabilities'):
            np.testing.assert_allclose(saved[key], replayed[key], rtol=3e-6, atol=3e-7, err_msg=prefix+' '+key)
        else:
            need(saved[key] == replayed[key], prefix+' '+key)


def independent_navigation_metrics(rows):
    n = len(rows)
    need(n > 0, 'nonempty navigation cohort')
    steps = sum(r['steps'] for r in rows)
    failed = [r for r in rows if not r['success']]
    terminations = Counter(r['termination'] for r in rows)
    return dict(episodes=n, successes=sum(r['success'] for r in rows), sr=sum(r['success'] for r in rows)/n,
                mean_sg_all_episodes=statistics.mean(r['sg'] for r in rows), mean_steps=statistics.mean(r['steps'] for r in rows),
                repeat_visit_rate_micro=sum(r['revisits'] for r in rows)/steps,
                repeat_visit_rate_macro=statistics.mean(r['repeat_visit_rate'] for r in rows),
                out_of_bounds_rate=sum(r['out_of_bounds'] for r in rows)/steps,
                mean_sg_m=float(np.mean([r['sg_m'] for r in rows])),
                mean_valid_travel_m=float(np.mean([r['valid_travel_m'] for r in rows])),
                total_valid_travel_m=sum(r['valid_travel_m'] for r in rows), total_steps=steps,
                total_revisits=sum(r['revisits'] for r in rows), total_oob=sum(r['out_of_bounds'] for r in rows),
                terminations=dict(terminations), failed_episodes=len(failed),
                mean_failed_sg=None if not failed else statistics.mean(r['sg'] for r in failed),
                mean_failed_sg_m=None if not failed else statistics.mean(r['sg_m'] for r in failed),
                last_step_successes=sum(r['success'] and r['steps'] == 20 for r in rows))


def independent_diagnostics(rows):
    records = [d for row in rows for d in row['evaluation_diagnostics']]
    groups = {}
    for name, cross in (('same_source', False), ('cross_source', True)):
        selected = [d for d in records if d['actionable_true_adjacency']
                    and d['true_adjacency_cross_source'] == cross]
        denominator = len(selected)
        accepted = sum(d['cue_accepted'] for d in selected)
        correct = sum(d['accepted_immediate_true_hit'] for d in selected)
        groups[name] = dict(opportunities=denominator, accepted=accepted, accepted_correct=correct,
                            acceptance_rate=accepted/denominator if denominator else None,
                            correct_acceptance_rate=correct/denominator if denominator else None)
    failures = Counter()
    for row in rows:
        if row['success']:
            continue
        distances = [d['true_target_distance'] for d in row['evaluation_diagnostics']]
        if 1 in distances:
            category = 'missed_actionable_adjacency'
        elif row['sg'] == 1:
            category = 'adjacent_only_terminal'
        elif min(distances+[row['sg']]) <= 2:
            category = 'within2_never_adjacent'
        else:
            category = 'never_within2'
        failures[category] += 1
    accepted = sum(d['cue_accepted'] for d in records)
    true_hits = sum(d['accepted_immediate_true_hit'] for d in records)
    return dict(action_count=len(records), crossings=sum(d['crossed_source_boundary'] for d in records),
                crossing_records=sum(any(d['crossed_source_boundary'] for d in r['evaluation_diagnostics']) for r in rows),
                cue_accepts=accepted, accepted_true_hits=true_hits,
                accepted_given_hits=sum(d['accepted_immediate_given_hit'] for d in records),
                accepted_not_true_hit=accepted-true_hits, accepted_true_hit_rate=true_hits/accepted if accepted else None,
                actionable_adjacency_records=sum(any(d['actionable_true_adjacency'] for d in r['evaluation_diagnostics']) for r in rows),
                adjacency=groups,
                failure_categories={k: failures[k] for k in ('missed_actionable_adjacency','adjacent_only_terminal',
                                                             'within2_never_adjacent','never_within2')},
                reasons=dict(Counter(d['reason'] for r in rows for d in r['decisions'])))


def independent_stats(rows):
    sources = sorted({row['source'] for row in rows})
    distances = sorted({row['distance'] for row in rows})
    strata = [name for name in STRATA if any(row['stratum'] == name for row in rows)]
    return dict(metrics=independent_navigation_metrics(rows),
                by_source={source: independent_navigation_metrics([r for r in rows if r['source'] == source]) for source in sources},
                by_distance={str(d): independent_navigation_metrics([r for r in rows if r['distance'] == d]) for d in distances},
                by_stratum={name: independent_navigation_metrics([r for r in rows if r['stratum'] == name]) for name in strata},
                diagnostics=independent_diagnostics(rows))


def independent_effect(left, right):
    sources = sorted(left[0]['by_source'])
    gains = [a['metrics']['sr']-b['metrics']['sr'] for a, b in zip(left, right)]
    per_source = {source: dict(sr=float(np.mean([a['by_source'][source]['sr']-b['by_source'][source]['sr']
                                                 for a, b in zip(left, right)])),
                               sg=float(np.mean([a['by_source'][source]['mean_sg_all_episodes']-
                                                 b['by_source'][source]['mean_sg_all_episodes'] for a, b in zip(left, right)])))
                  for source in sources}
    matrix = np.asarray([[per_source[source]['sr'], per_source[source]['sg']] for source in sources])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(sources), size=(BOOTSTRAP_RESAMPLES, len(sources)))
    interval = np.quantile(matrix[draws].mean(1), [.025, .975], axis=0)
    return dict(sr_gain=float(np.mean(gains)), sr_gain_by_seed=gains, positive_seeds=sum(x > 1e-12 for x in gains),
                sg_change=float(np.mean([a['metrics']['mean_sg_all_episodes']-b['metrics']['mean_sg_all_episodes']
                                         for a, b in zip(left, right)])),
                source_effects=per_source, source95_SR=interval[:, 0].tolist(), source95_SG=interval[:, 1].tolist(),
                source_count=len(sources), bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                interpretation='paired whole-region uncertainty; weights are not independent source maps')


def independent_target_checks(primary, pooled):
    return dict(SR_gain5pp=primary['sr_gain'] >= .05-1e-12,
                two_positive_weights=primary['positive_seeds'] >= 2,
                primary_SG_no_worse=primary['sg_change'] <= 1e-12,
                pooled_SG_no_worse=pooled['sg_change'] <= 1e-12,
                source95_SR_lower_positive=primary['source95_SR'][0] > 0)


def expected_navigation_summary(rows_by, rule_rows):
    arms = {}
    for condition in CONDITIONS:
        pooled = [row for seed in SEEDS for row in rows_by[condition, seed]]
        per_seed = [independent_navigation_metrics(rows_by[condition, seed]) for seed in SEEDS]
        arms[condition] = dict(**independent_stats(pooled), sr_by_seed=[x['sr'] for x in per_seed],
                               sg_by_seed=[x['mean_sg_all_episodes'] for x in per_seed],
                               successes_by_seed=[x['successes'] for x in per_seed], tasks_each=len(rows_by[condition, SEEDS[0]]),
                               weight_count=3)
    for name, rows in rule_rows.items():
        arms[name] = dict(**independent_stats(rows), tasks_each=len(rows), weight_count=0)
    effects, paired = {}, []
    for cohort in ('all', *STRATA):
        select = lambda rows: rows if cohort == 'all' else [r for r in rows if r['stratum'] == cohort]
        full = [independent_stats(select(rows_by['CueFull', seed])) for seed in SEEDS]
        effects[cohort] = {}
        for comparison in (*CONDITIONS[1:], *RULES):
            reference = ([independent_stats(select(rows_by[comparison, seed])) for seed in SEEDS]
                         if comparison in CONDITIONS else [independent_stats(select(rule_rows[comparison]))]*3)
            effects[cohort][comparison] = independent_effect(full, reference)
    for seed in SEEDS:
        for comparison in (*CONDITIONS[1:], *RULES):
            reference = rows_by[comparison, seed] if comparison in CONDITIONS else rule_rows[comparison]
            for full, other in zip(rows_by['CueFull', seed], reference):
                need(full['episode_id'] == other['episode_id'], 'paired complete task ordering')
                paired.append(dict(seed=seed, comparison=comparison, episode_id=full['episode_id'], source=full['source'],
                                   distance=full['distance'], stratum=full['stratum'], full_success=full['success'],
                                   reference_success=other['success'], full_sg=full['sg'], reference_sg=other['sg'],
                                   recovered=full['success'] and not other['success'], harmed=other['success'] and not full['success']))
    checks = {condition: independent_target_checks(effects['long_distance'][condition], effects['all'][condition])
              for condition in CONDITIONS[1:]}
    primary = arms['CueFull']['by_stratum']['long_distance']['sr']
    return dict(arms=arms, effects=effects, target_checks=checks, primary_SR=primary,
                primary_SR60=primary >= .60-1e-12,
                target_numeric_passed=all(all(v.values()) for v in checks.values()),
                paired_recovery={c: {key: sum(row[key] for row in paired if row['comparison'] == c)
                                     for key in ('recovered', 'harmed')} for c in (*CONDITIONS[1:], *RULES)}), paired


def replay_episode(stage, seed, condition, ep, saved, agent, env, banks, means, wrong, provenance, strata, region):
    need(saved['episode_id'] == ep.episode_id, 'ordered episode ID')
    metadata = (saved['area'], saved['source'], saved['split'], saved['distance'], saved['stratum'],
                saved['condition'], saved['status'])
    expected_metadata = (ep.area, ep.source_tile, ep.split, ep.dist, strata[ep.episode_id]['stratum'], condition, 'completed')
    need(metadata == expected_metadata, 'saved navigation task metadata')
    if seed is not None:
        need(saved.get('local_checkpoint_seed') == seed, 'saved checkpoint identity')
    else:
        need('local_checkpoint_seed' not in saved, 'rule rows contain no neural checkpoint identity')
    agent.reset()
    obs = env.reset(ep)
    forbidden = {'goal', 'distance', 'dist', 'area', 'source', 'source_tile', 'stratum', 'split', 'epsg',
                 'evaluation_diagnostics', 'expected_class', 'probe_id'}
    need(not ({field.name for field in __import__('dataclasses').fields(obs)} & forbidden), 'public observation excludes evaluation truth')
    g, l, p = banks
    key = ep.split+'__'+ep.area
    cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
    given_target = env.payload(cue)
    position, visited = ep.start, [ep.start]
    trajectory = [dict(step=0, patch_id=position, action=None, out_of_bounds=False, revisited=False)]
    diagnostics = []
    hidden = None
    decisions = saved['decisions']
    need(len(decisions) == len(saved['evaluation_diagnostics']) == saved['steps'], 'action/diagnostic/step denominators')
    for index, decision in enumerate(decisions):
        need(not env.done and index < 20, 'no action after terminal or budget')
        need(obs.position == divmod(position, 10) and obs.visited == tuple(visited)
             and obs.remaining_budget == 20-index, 'observation state reset and transition')
        view = replace(obs, target_image=given_target)
        cell = position
        if seed is None:
            action = independent_rule(condition, obs)
            expected = dict(action=action, reason='rule', public_position=list(obs.position),
                            public_visited=list(obs.visited), remaining_budget=obs.remaining_budget)
            need(decision == expected, 'independent public-only rule action')
            accepted = False
        else:
            # Replay through the unchanged checkpoint implementation, then check its inputs/actions
            # a second time with explicit hand-computed policy and cue-head equations.
            replayed = agent.act_with_profiles(view, g[key][cell], l[key][cell], g[key][cue], l[key][cue], p[key][cell], p[key][cue])
            _check_decision(decision, replayed, f'{condition}/s{seed}/step{index}')
            independent, hidden = independent_policy_step(agent, view, cell, cue, key, banks, hidden, index+1)
            _check_decision(decision, independent, f'independent equations/{condition}/s{seed}/step{index}')
            expected = decision
            accepted = decision['reason'] == 'accepted'
            need(decision['current_image_sha256'] == sha256(view.current_image).hexdigest()
                 and decision['target_image_sha256'] == sha256(given_target).hexdigest(), 'given image payload bindings')
        action = expected['action']
        need(action in MOVES, 'four-action output')
        diagnostic = independent_diagnostic(position, ep.goal, cue, action, accepted, 20-index, region)
        need(diagnostic == saved['evaluation_diagnostics'][index], 'independent continuous-source/seam/goal diagnostic')
        destination, oob = advance(position, action)
        if seed is not None:
            need(not oob, 'neural boundary gate prevents out-of-bounds action')
        revisited = destination in visited
        position = destination
        visited.append(position)
        trajectory.append(dict(step=index+1, patch_id=position, action=action, out_of_bounds=oob, revisited=revisited))
        obs, done, info = env.step(action)
        need((obs.position, obs.remaining_budget, info.out_of_bounds, info.revisited)
             == (divmod(position, 10), 19-index, oob, revisited), 'independent environment step')
        need(done == (position == ep.goal or index == 19), 'first-arrival/last-budget termination')
        if position == ep.goal:
            need(index+1 == len(decisions), 'stop on first target arrival, including final step')
        diagnostics.append(diagnostic)
    need(env.done, 'all rows have terminal state')
    sg = independent_distance(position, ep.goal)
    success = position == ep.goal
    need(saved['trajectory'] == trajectory and saved['steps'] == len(trajectory)-1, 'complete trajectory replay')
    need(saved['success'] == success and saved['sg'] == sg, 'manual terminal truth and SG')
    need(saved['termination'] == ('goal_reached' if success else 'budget_exhausted'), 'terminal reason')
    need(success or saved['steps'] == 20, 'failures consume the complete fixed action budget')
    need(saved['sg_m'] == 300*sg and saved['cell_size_m'] == 300 and saved['map_projected_area_km2'] == 9,
         'physical SG and map area')
    valid_m = 300*sum(t['action'] is not None and not t['out_of_bounds'] for t in trajectory)
    need(saved['valid_travel_m'] == valid_m, 'effective legal travel distance excludes OOB stalls')
    need(saved['revisits'] == sum(t['revisited'] for t in trajectory)
         and saved['out_of_bounds'] == sum(t['out_of_bounds'] for t in trajectory), 'revisit and boundary totals')
    expected_row = dict(episode_id=ep.episode_id, success=success,
                        termination='goal_reached' if success else 'budget_exhausted', sg=sg,
                        steps=len(trajectory)-1, revisits=sum(t['revisited'] for t in trajectory),
                        repeat_visit_rate=sum(t['revisited'] for t in trajectory)/(len(trajectory)-1),
                        out_of_bounds=sum(t['out_of_bounds'] for t in trajectory), trajectory=trajectory,
                        protocol=PROTOCOL, cell_size_m=300, sg_m=300*sg, valid_travel_m=valid_m,
                        map_projected_area_km2=9, area=ep.area, source=ep.source_tile, split=ep.split,
                        distance=ep.dist, stratum=strata[ep.episode_id]['stratum'], condition=condition,
                        status='completed', decisions=decisions, evaluation_diagnostics=diagnostics)
    if seed is not None:
        expected_row['local_checkpoint_seed'] = seed
    need(same_values(saved, expected_row), 'complete terminal row identity')


def replay_navigation(stage, episodes, banks, means, wrong, provenance, strata, regions):
    folder = protocol.stage_root(stage)
    rows_by, rule_rows = {}, {}
    total_actions = neural_actions = rule_actions = 0
    env = SpatialAreaGridEnv(DATA)
    regions_by_area = {region['area']: region for region in regions}
    for seed in SEEDS:
        for condition in CONDITIONS:
            prefix = f'M0_s{seed}_{condition}'
            trajectory_path = folder/'神经对照'/(prefix+'_轨迹.jsonl')
            rows = list(lines(trajectory_path))
            need([r['episode_id'] for r in rows] == [ep.episode_id for ep in episodes], prefix+' all planned rows in order')
            agent = protocol.make_agent(seed, means, condition)
            for row, ep in zip(rows, episodes):
                replay_episode(stage, seed, condition, ep, row, agent, env, banks, means, wrong, provenance, strata,
                               regions_by_area[ep.area])
            expected = dict(**independent_stats(rows), trajectory_sha256=digest(trajectory_path))
            stored = read(trajectory_path.with_name(prefix+'_结果.json'))
            need(same_values(stored, expected), prefix+' independently reconstructed per-run statistics')
            rows_by[condition, seed] = rows
            n = sum(row['steps'] for row in rows)
            total_actions += n
            neural_actions += n
            print(dict(stage=stage, replayed=prefix, tasks=len(rows), actions=n), flush=True)
    for name in RULES:
        trajectory_path = folder/'规则基线'/(name+'_轨迹.jsonl')
        rows = list(lines(trajectory_path))
        need([r['episode_id'] for r in rows] == [ep.episode_id for ep in episodes], name+' all planned rows in order')
        for row, ep in zip(rows, episodes):
            replay_episode(stage, None, name, ep, row, None, env, banks, means, wrong, provenance, strata,
                           regions_by_area[ep.area])
        expected = dict(**independent_stats(rows), trajectory_sha256=digest(trajectory_path))
        need(same_values(read(trajectory_path.with_name(name+'_结果.json')), expected), name+' independent rule statistics')
        rule_rows[name] = rows
        n = sum(row['steps'] for row in rows)
        total_actions += n
        rule_actions += n
        print(dict(stage=stage, replayed=name, tasks=len(rows), actions=n), flush=True)
    expected_summary, paired = expected_navigation_summary(rows_by, rule_rows)
    summary_path = folder/'对照汇总.json'
    stored_summary = read(summary_path)
    expected_summary.update(stage=stage, planned_neural=len(episodes)*12, planned_rules=len(episodes)*2,
                            planned_total=len(episodes)*14, maps=STAGES[stage][2], cell_size_m=300,
                            projected_area_km2_each=9, primary_SR_and_target_gates_used_for_engineering=False)
    need(same_values(stored_summary, expected_summary), 'all pooled summaries, strata, paired effects and whole-map intervals')
    need(paired == read(folder/'逐题配对.json'), 'all per-task paired recoveries/harms')
    return total_actions, neural_actions, rule_actions, rows_by, rule_rows, digest(summary_path)


def given_class(current, cue):
    delta = (cue//10-current//10, cue%10-current%10)
    return {(-1, 0): 'up', (0, 1): 'right', (1, 0): 'down', (0, -1): 'left'}.get(delta, 'not_adjacent')


def independent_usable(probabilities, cell):
    top = int(np.argmax(probabilities))
    if top == 4:
        return None, 'not_adjacent'
    if probabilities[top] < .5:
        return None, 'low_confidence'
    action = CLASS_NAMES[top]
    destination, oob = advance(cell, action)
    if oob:
        return None, 'illegal_top_direction'
    if destination == cell:
        return None, 'visited_top_destination'
    return action, 'accepted'


def independent_probe_metrics(rows):
    adjacent = [r for r in rows if r['expected_class'] != 'not_adjacent']
    negative = [r for r in rows if r['expected_class'] == 'not_adjacent']
    accepted = sum(r['raw_accepted'] for r in rows)
    correct = sum(r['raw_correct'] for r in adjacent)
    false = sum(r['raw_accepted'] for r in negative)
    return dict(records=len(rows), raw_accepted=accepted, raw_correct_accepts=correct,
                raw_correct_acceptance=None if not adjacent else correct/len(adjacent), adjacent_denominator=len(adjacent),
                raw_false_accepts=false, raw_nonadjacent_false_acceptance=None if not negative else false/len(negative),
                nonadjacent_denominator=len(negative),
                raw_wrong_direction_accepts=sum(r['raw_accepted'] and r['top_class'] != r['expected_class'] for r in adjacent),
                usable_accepted=sum(r['usable_reason'] == 'accepted' for r in rows),
                given_target_correct_accepts=sum(r['raw_accepted'] and r['top_class'] == r['given_expected_class'] for r in rows),
                reasons=dict(Counter(r['usable_reason'] for r in rows)))


def independent_probe_stats(rows):
    return dict(metrics=independent_probe_metrics(rows),
                by_kind={kind: independent_probe_metrics([r for r in rows if r['kind'] == kind]) for kind in sorted({r['kind'] for r in rows})},
                by_source={source: independent_probe_metrics([r for r in rows if r['source_tile'] == source])
                           for source in sorted({r['source_tile'] for r in rows})},
                by_direction={direction: independent_probe_metrics([r for r in rows if r['expected_class'] == direction])
                              for direction in sorted({r['expected_class'] for r in rows})})


def independent_probe_summary(rows_by):
    arms = {condition: independent_probe_stats([r for seed in SEEDS for r in rows_by[condition, seed]])
            for condition in PROBE_CONDITIONS}
    by_seed = {str(seed): {condition: independent_probe_stats(rows_by[condition, seed]) for condition in PROBE_CONDITIONS}
               for seed in SEEDS}
    cross = arms['Full']['by_kind']['cross_source_adjacent']['raw_correct_acceptance']
    false = arms['Full']['by_kind']['nonadjacent_matched_target']['raw_nonadjacent_false_acceptance']
    positives = sum(by_seed[str(seed)]['Full']['by_kind']['cross_source_adjacent']['raw_correct_acceptance'] >= .90-1e-12
                    and by_seed[str(seed)]['Full']['by_kind']['nonadjacent_matched_target']['raw_nonadjacent_false_acceptance'] <= .05+1e-12
                    for seed in SEEDS)
    maps = sorted(arms['Full']['by_source'])
    matrix = np.asarray([[np.mean([independent_probe_stats([r for r in rows_by['Full', seed] if r['source_tile'] == source])
                                     ['by_kind']['cross_source_adjacent']['raw_correct_acceptance'] for seed in SEEDS]),
                          np.mean([independent_probe_stats([r for r in rows_by['Full', seed] if r['source_tile'] == source])
                                     ['by_kind']['nonadjacent_matched_target']['raw_nonadjacent_false_acceptance'] for seed in SEEDS])]
                         for source in maps])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(maps), size=(BOOTSTRAP_RESAMPLES, len(maps)))
    ci = np.quantile(matrix[draws].mean(1), [.025, .975], axis=0)
    checks = dict(cross_raw_correct90=cross >= .90-1e-12, nonadjacent_raw_false5=false <= .05+1e-12,
                  two_weights=positives >= 2)
    return dict(arms=arms, by_seed=by_seed, seam_checks=checks, seam_numeric_passed=all(checks.values()),
                positive_weights=positives, source95_cross_correct=ci[:, 0].tolist(),
                source95_nonadjacent_false=ci[:, 1].tolist(), source_count=len(maps), intervals_descriptive_only=True,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                scope='raw head reliability under frozen target/cell probes; not closed-loop navigation')


def replay_probes(stage, banks, means, probe_bank, wrong_plan, provenance):
    g, l, p = banks
    probes = [probe for probe in probe_bank if probe['split'] == STAGES[stage][1]]
    folder = protocol.stage_root(stage)/'探针对照'
    by = {}
    for seed in SEEDS:
        agent = protocol.make_agent(seed, means, 'CueFull')
        for condition in PROBE_CONDITIONS:
            path = folder/f'M0_s{seed}_{condition}_判断.jsonl'
            saved = list(lines(path))
            need(len(saved) == len(probes), f'{condition}/s{seed} fixed probe denominator')
            expected_rows = []
            for probe, row in zip(probes, saved):
                need(all(row.get(key) == value for key, value in probe.items()), 'immutable source probe labels/geometry')
                key = probe['split']+'__'+probe['area']
                current, target = probe['current_cell'], probe['target_cell']
                cue = wrong_plan[probe['probe_id']]['cue_cell'] if condition == 'Wrong' else target
                use_mean = condition == 'Mean'
                vector = np.concatenate((manual_cue_features(agent.hm if use_mean else g[key][cue], g[key][current],
                                                             agent.lm if use_mean else l[key][cue], l[key][current]),
                                         manual_edge_features(agent.pm if use_mean else p[key][cue], p[key][current]))).astype(np.float32)
                raw = manual_head_raw(agent, vector)
                probabilities = joint_probabilities(raw)
                top = int(np.argmax(probabilities))
                raw_accepted = bool(top < 4 and probabilities[top] >= .5)
                action, reason = independent_usable(probabilities, current)
                expected = dict(probe, seed=seed, condition=condition, cue_cell=cue, raw_logits=raw.tolist(),
                                joint_log_probabilities=None, probabilities=probabilities.tolist(), top_class=CLASS_NAMES[top],
                                raw_accepted=raw_accepted,
                                raw_correct=bool(raw_accepted and CLASS_NAMES[top] == probe['expected_class']),
                                usable_action=action, usable_reason=reason,
                                given_expected_class=given_class(current, cue),
                                cue_features_sha256=sha256(vector.tobytes()).hexdigest())
                # Recent evaluator records retain forward joint log probabilities as a separate audit trace.
                joint = torch.as_tensor(raw, dtype=torch.float32).reshape(1, 5)
                joint_log = torch.cat((F.logsigmoid(joint[:, 4:5])+F.log_softmax(joint[:, :4], dim=-1),
                                       F.logsigmoid(-joint[:, 4:5])), dim=-1)[0].numpy()
                if 'joint_log_probabilities' in row:
                    expected['joint_log_probabilities'] = joint_log.tolist()
                else:
                    expected.pop('joint_log_probabilities')
                need(same_values(row, expected, atol=5e-7, rtol=5e-7),
                     f'raw head, probability, acceptance, usability and independent label for {probe["probe_id"]}/{condition}/s{seed}')
                expected_rows.append(expected)
            result_file = read(path.with_name(f'M0_s{seed}_{condition}_结果.json'))
            expected_result = dict(**independent_probe_stats(expected_rows), judgments_sha256=digest(path))
            need(same_values(result_file, expected_result), 'per-seed per-condition probe statistics')
            by[condition, seed] = expected_rows
    summary_path = protocol.stage_root(stage)/'探针对照/探针汇总.json'
    expected_summary = independent_probe_summary(by)
    need(same_values(read(summary_path), expected_summary), 'all probe strata, map/seed tables and region bootstrap intervals')
    return sum(map(len, by.values())), digest(summary_path), expected_summary


def run_audit(stage):
    folder = protocol.stage_root(stage)
    audit_path = folder/'独立复核.json'
    need(not audit_path.exists(), 'immutable independent audit output')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reg, manifest, regions, all_tasks, strata, wrong, probes, probe_wrong = verify_design_and_data(stage)
    stage_tasks = [task for task in all_tasks if task['split'] == STAGES[stage][1]]
    episodes = [SpatialAreaEpisode(**task) for task in stage_tasks]
    for ep in episodes:
        ep.validate()
    frozen = read(folder/'特征冻结结束.json')
    need(frozen['stage'] == stage and not frozen['model_evaluation_started']
         and frozen['patches'] == STAGES[stage][2]*100, 'stage feature freeze precedes evaluation')
    patches, banks, provenance = reextract_stage(stage, frozen)
    means = {name: np.load(OLD/name, allow_pickle=False) for name in protocol.MEANS}
    # Independent full replay never gives source IDs, strata or episode truth to the M0 agent.
    actions, neural_actions, rule_actions, rows_by, rule_rows, summary_sha = replay_navigation(
        stage, episodes, banks, means, wrong, provenance, strata, regions)
    probe_count, probe_summary_sha, probe_summary = replay_probes(stage, banks, means, probes, probe_wrong, provenance)
    state = read(folder/'执行状态.json')
    maps, tasks_each = STAGES[stage][2], STAGES[stage][3]
    records, probe_records = tasks_each*14, maps*120*9
    need(state['completed'] and state['episodes'] == records and state['neural'] == tasks_each*12
         and state['rules'] == tasks_each*2 and state['probe_records'] == probe_records,
         'all predeclared stage navigation and probe actions completed')
    need(state['new_training_steps'] == 0 and state['cloud_calls'] == 0 and not state['default_changed'],
         'unchanged default, no training or cloud use')
    protocol.check_bindings(reg)
    audit = dict(passed=True, stage=stage, records=records, neural_records=tasks_each*12, rule_records=tasks_each*2,
                 actions=actions, neural_actions=neural_actions, rule_actions=rule_actions,
                 probe_records=probe_count, reconstructed_native_patches=patches,
                 global_local_reextraction_exact=True, independent_profiles_and_cue_equations_exact=True,
                 all_checkpoint_GRU_and_environment_steps_replayed=True, independent_rules_checked=True,
                 public_input_boundary_checked=True, continuous_geometry_and_source_diagnostics_checked=True,
                 all_target_mask_channels_and_condition_gates_checked=True,
                 all_planned_denominators_and_paired_whole_map_intervals_checked=True,
                 raw_probe_logits_probabilities_labels_and_acceptance_checked=True,
                 probe_summary=probe_summary, summary_sha256=summary_sha,
                 probe_summary_sha256=probe_summary_sha, protected_bindings_verified=sum(len(reg[k]) if isinstance(reg[k], dict) else 1
                 for k in ('protected_sha256', 'source_sha256', 'input_sha256', 'encoder_sha256', 'frozen_output_sha256')),
                 stage_feature_files_verified=len(frozen['files_sha256']), default_changed=False,
                 new_training_steps=0, cloud_calls=0)
    write_exclusive(audit_path, audit)
    print(dict(stage=stage, passed=True, records=audit['records'], probe_records=probe_count,
               reconstructed_native_patches=patches, summary_sha256=summary_sha,
               probe_summary_sha256=probe_summary_sha), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=tuple(STAGES), required=True)
    run_audit(parser.parse_args().stage)


if __name__ == '__main__':
    main()
