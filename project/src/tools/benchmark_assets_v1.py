"""Evaluator-owned, SHA-bound retrospective assets; no model imports/forwards.

Only read the original S3 queue and caches. The trusted pickle is opened after
verification against the immutable BENCH-006 binding, never from a caller URL.
"""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from agents.benchmark_contract_v1 import FeatureBundle
from env.environment import image_payload
from eval.benchmark_runner_v1 import Payload, Task, FitMean
from eval.benchmark_runner_v2 import RealPlan

STATIC = '项目导航/共同真实评测准备_v1/核验/静态准备协议_v1.json'
STATIC_SHA = '9fd7b1b97145793856544c292bb7dd270b8201b848b9e458f941f336837ca394'
S3 = 'DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1'
GLOBAL = 'DATA/processed_data/Masa/papr_test_sat_embeds_grid_5.npy'
TRAIN = 'DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v2/元数据/冻结执行协议.json'
HEAD = 'DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1'
HEAD_REG_SHA = 'b534d8a27cf7c795fc7fe5d2d947952c3e93b26a7e3950845b64bf5a4d7540ca'
SCHEMA = (('global', 512), ('local', 3072), ('profile', 2304))


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024): h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def record_hash(value): return sha256(canonical(value)).hexdigest()


def inside(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if root not in path.parents: raise ValueError('bound path outside workspace')
    return path


def verify_bindings(root, bindings):
    if not bindings: raise ValueError('nonempty input bindings required')
    for relative, expected in bindings.items():
        if digest(inside(root, relative)) != expected: raise ValueError('input binding drift: ' + relative)


def array(value, shape):
    v = np.asarray(value)
    if v.shape != shape or v.dtype != np.float32 or not np.isfinite(v).all():
        raise ValueError('frozen cache shape/dtype/finiteness differs')
    return v


def bundle(global_feature, local, profile):
    values = (array(global_feature, (512,)), array(local, (4, 768)), array(profile, (4, 3, 64, 3)))
    return FeatureBundle(tuple((name, tuple(float(x) for x in v.ravel()))
                               for (name, _), v in zip(SCHEMA, values)))


def queue(raw, wrong):
    if type(raw) is not list or len(raw) != 250: raise ValueError('original 250 tasks required')
    if set(wrong) != {t['episode_id'] for t in raw}: raise ValueError('Wrong queue identity differs')
    tasks = []
    source_names = {}
    exceptions = 0
    for t in raw:
        if t['split'] != 'test' or t['grid_size'] != 5 or t['budget'] != 10:
            raise ValueError('task protocol differs')
        area = t['area']; name = t['source_tile']
        if source_names.setdefault(area, name) != name: raise ValueError('source identity changed')
        start, goal, other = t['start'], t['goal'], wrong[t['episode_id']]['cue_cell']
        def dist(a, b): return abs(a // 5 - b // 5) + abs(a % 5 - b % 5)
        matched = dist(start, goal) == dist(start, other)
        if type(wrong[t['episode_id']]['matched_distance']) is not bool or wrong[t['episode_id']]['matched_distance'] != matched:
            raise ValueError('Wrong distance declaration differs')
        if t['dist'] != dist(start, goal): raise ValueError('true distance declaration differs')
        exceptions += int(not matched)
        tasks.append(Task(t['episode_id'], area, start, goal, other, f'C{t["dist"]}'))
    if len(set(source_names.values())) != 10: raise ValueError('ten distinct source-file identities required')
    plan = RealPlan(tuple(tasks), 'known_retrospective_Masa_test250'); plan.validate()
    routes = len({(t.area, t.start, t.goal) for t in tasks})
    if routes != 221 or exceptions != 90: raise ValueError('original route/exception cohort differs')
    return plan, dict(source_files=source_names, unique_routes=routes, repeated_routes=250-routes,
                     wrong_distance_exceptions=exceptions)


def payload_identity(p):
    return dict(image_sha256=sha256(p.image).hexdigest(),
                feature_schema=[list(x) for x in p.features.schema],
                features_sha256=record_hash([[n, list(v)] for n, v in p.features.channels]))


def constant_png():
    stream = BytesIO()
    Image.new('RGB', (300, 300), (127, 127, 127)).save(stream, format='PNG')
    return stream.getvalue()


@dataclass(frozen=True)
class EvaluatorAssets:
    plan: RealPlan
    cells: dict
    means: dict
    evidence: dict


def build(root):
    root = Path(root).resolve()
    static_path = inside(root, STATIC)
    if digest(static_path) != STATIC_SHA: raise ValueError('BENCH-006 immutable protocol differs')
    static = json.loads(static_path.read_text('utf-8'))
    bindings = dict(static['bindings']); verify_bindings(root, bindings)
    def load_json(name): return json.loads(inside(root, name).read_text('utf-8'))
    raw = load_json(S3 + '/导航任务.json'); wrong = load_json(S3 + '/错误目标计划.json')
    plan, cohort = queue(raw, wrong)
    # Only these known repository-produced bytes can enter np.load(pickle=True).
    globals_ = np.load(inside(root, GLOBAL), allow_pickle=True).item()
    paths = load_json(S3 + '/测试图块来源.json')
    areas = {t.area for t in plan.tasks}
    if set(globals_) != areas or set(paths) != {f'{a}/{i}' for a in areas for i in range(25)}:
        raise ValueError('image/global source cohort differs')
    cells = {}; raw_rgb_count = 0
    with np.load(inside(root, S3 + '/测试局部区域特征.npz'), allow_pickle=False) as locals_, \
         np.load(inside(root, S3 + '/测试边缘profile.npz'), allow_pickle=False) as profiles:
        if set(locals_.files) != areas or set(profiles.files) != areas:
            raise ValueError('auxiliary cache source cohort differs')
        for area in sorted(areas):
            g = array(globals_[area], (25, 512)); l = array(locals_[area], (25, 4, 768))
            p = array(profiles[area], (25, 4, 3, 64, 3)); cells[area] = {}
            for i in range(25):
                entry = paths[f'{area}/{i}']; path = inside(root, entry['path'])
                if bindings.get(entry['path']) != entry['file_sha256'] or digest(path) != entry['file_sha256']:
                    raise ValueError('patch byte binding differs')
                with Image.open(path) as image: rgb = np.asarray(image.convert('RGB'))
                if list(rgb.shape) != entry['shape'] or sha256(rgb.tobytes()).hexdigest() != entry['rgb_sha256']:
                    raise ValueError('decoded original RGB pixels differ')
                cells[area][i] = Payload(image_payload(path), bundle(g[i], l[i], p[i])); raw_rgb_count += 1
    config = load_json('project/local_policy_default.json')
    if digest(inside(root, HEAD + '/预登记.json')) != HEAD_REG_SHA:
        raise ValueError('original head training registration differs')
    registration = load_json(HEAD + '/预登记.json')
    split = inside(root, HEAD + '/源图划分.json')
    expected_split_sha = registration['frozen_inputs_sha256']['源图划分.json']
    if digest(split) != expected_split_sha: raise ValueError('head fit-source binding differs')
    bindings[HEAD + '/预登记.json'] = HEAD_REG_SHA
    bindings[HEAD + '/源图划分.json'] = expected_split_sha
    head_split = json.loads(split.read_text('utf-8'))
    if len(head_split['head_fit']) != 87 or len(head_split['head_calibration']) != 22 or len(head_split['held']) != 28:
        raise ValueError('frozen head source counts differ')
    def mean(name, shape):
        record = config['means'][name]; path = inside(root, record['path'])
        if digest(path) != record['sha256']: raise ValueError('M0 mean differs')
        return array(np.load(path, allow_pickle=False), shape)
    hg = mean('头部拟合全局均值.npy', (512,))
    hl = mean('头部拟合局部均值.npy', (4, 768)); hp = mean('头部拟合边缘均值.npy', (4, 3, 64, 3))
    training = load_json(TRAIN); geo_mean_path = training['fit_mean']
    if len(training['fit_areas']) != 109 or set(training['fit_areas']) != set(head_split['head_fit']) | set(head_split['head_calibration']):
        raise ValueError('common fit-source cohort differs')
    if set(training['fit_areas']) & set(head_split['held']): raise ValueError('held source enters training mean cohort')
    if digest(inside(root, geo_mean_path)) != training['bindings'][geo_mean_path]:
        raise ValueError('GeoExplorer fit mean differs')
    bindings[geo_mean_path] = training['bindings'][geo_mean_path]
    gm = array(np.load(inside(root, geo_mean_path), allow_pickle=False), (512,))
    placeholder = constant_png()
    # Auxiliary constants never enter GeoExplorer: GlobalOnlyPolicy projects them
    # out. They satisfy the common schema; no test mean is calculated.
    means = {
        'M0': FitMean(Payload(placeholder, bundle(hg, hl, hp)), expected_split_sha),
        'GeoExplorer': FitMean(Payload(placeholder, bundle(gm, hl, hp)), digest(inside(root, TRAIN)))
    }
    evidence = dict(status='assets_verified_no_navigation', execution_allowed=False,
        queue_sha256=record_hash(plan.record()), plan=plan.record(), cohort=cohort,
        static_protocol=dict(path=STATIC, sha256=STATIC_SHA), bindings=bindings,
        assets={a: {str(i): payload_identity(p) for i, p in v.items()} for a, v in cells.items()},
        means={m: dict(**payload_identity(v.payload), fit_manifest_sha256=v.fit_manifest_sha256,
                       source_role=v.source_role) for m, v in means.items()},
        raw_rgb_patches_checked=raw_rgb_count, common_schema=[list(x) for x in SCHEMA],
        distance_unit='grid_cells', geographic_nonoverlap_verified=False,
        source_count=10, scientific_sr=False, model_forward=0, navigation=0,
        new_training=0, cloud_calls=0, reserve_regions_consumed=0,
        mean_note='M0 head fit87; GeoExplorer global fit109; Geo ignores local/profile constants',
        source_sha256=digest(Path(__file__)))
    return EvaluatorAssets(plan, cells, means, evidence)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--save', type=Path); args = parser.parse_args()
    root = SRC.parents[1]; result = build(root)
    if args.save:
        out = args.save.resolve()
        allowed = (root / '项目导航/共同真实评测执行准备_v2/核验').resolve()
        if out.parent != allowed or not out.name.startswith('资产复核_'):
            raise ValueError('new registered asset evidence filename required')
        with out.open('x', encoding='utf-8') as f: json.dump(result.evidence, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result.evidence['status'], patches=250, tasks=250,
        queue_sha256=result.evidence['queue_sha256'], **result.evidence['cohort']), ensure_ascii=False))
