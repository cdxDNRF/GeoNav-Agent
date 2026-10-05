"""DEV-001误触发治理开发验证：cue接受门槛校准与同题开发导航。

唯一因素=cue接受置信门槛（0.50→τ*）；模型权重、特征、探索器、覆盖控制器、
预算与网格协议全部不变。校准只用EVAL-002封存探针概率；开发导航只在
DATA-007已消费开发区img_6100（grid5/10/15嵌套窗）上进行。不训练、不调用
外部模型、不消费正式10区或3备用区、不覆盖任何历史原件。
"""
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'project/src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIRM = ROOT / 'DATA/processed_data/MassGIS/评测结果/正式十乘十十五乘十五确认_v1'
DEV7 = ROOT / 'DATA/processed_data/MassGIS/工程准备/同产品开发导航兼容_v1'
OUT = ROOT / 'DATA/processed_data/MassGIS/评测结果/误触发治理开发验证_v1'
AREAS = tuple(f'img_{n}' for n in (7005, 7008, 7009, 7010, 7011, 7012, 7013, 7014, 7015, 7016))
GRIDS = (5, 10, 15)
DEV_NAV_GRIDS = (5, 10, 15)
SEEDS = (0, 1, 2)
TAU_GRID = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))  # 0.50..0.95 冻结
PRECISION_TARGET = 0.90
RECALL_FLOOR = 0.90

# 开发门槛（全部必要；来自EVAL-003冻结方案第三节）
DEV_GATES = dict(wrong_cue_action_rate_at_most=0.05,
                 SR_paired_not_lower=True,
                 stratum_SG_no_worse=True,
                 probe_adjacent_recall_at_least=0.90)
# 资源上限
LIMITS = dict(neural_records=4200, derived_bytes=2 * 1024 ** 3, minimum_free_bytes=8 * 1024 ** 3,
              wall_seconds=24 * 3600, network_requests=0, training_steps=0, spare_regions_used=0,
              formal_regions_used=0)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def sha_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'输出已存在，不覆盖: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')
    return path


def manhattan(a, b, k):
    return abs(a // k - b // k) + abs(a % k - b % k)


# ---------------- 校准（只读EVAL-002封存探针概率） ----------------

def calibration_spec():
    """冻结校准规范：先落盘规范与输入绑定，再计算曲线。"""
    inputs = {}
    for k in (10, 15):
        for area in AREAS:
            pairs = CONFIRM.parent.parent / '工程准备/正式导航协议冻结_v1' / '元数据' / f'{area}_g{k}_探针.json'
            inputs[rel(pairs)] = sha_file(pairs)
        for seed in SEEDS:
            for condition in ('CueFull', 'CueMean'):
                for area in AREAS:
                    for suffix in ('.npy', '.seal.json'):
                        p = CONFIRM / '特征' / f'probe_g{k}_s{seed}_{condition}_{area}{suffix}'
                        inputs[rel(p)] = sha_file(p)
    spec = dict(protocol='dev001-threshold-calibration-v1',
                utc_spec_frozen=stamp(),
                factor='cue_acceptance_threshold_only',
                tau_grid=list(TAU_GRID),
                accept_rule='argmax in classes 0..3 and p[argmax] >= tau; otherwise abstain to explorer',
                selection_rule=('tau* = smallest grid tau with per-grid(10,15) probe precision>=0.90 AND '
                                'adjacent recall>=0.90; if none, smallest tau with recall>=0.90 on both grids '
                                'maximizing total precision (tie -> smaller tau); CueMean curve reference only'),
                precision_target=PRECISION_TARGET,
                recall_floor=RECALL_FLOOR,
                frozen_before_curve=True,
                dev_gates=DEV_GATES,
                limits=LIMITS,
                calibration_source_sha256=inputs,
                note='规范在查看任何tau曲线之前落盘；CueMean仅作参考不参与选择')
    path = write_json(OUT / '元数据/校准规范.json', spec)
    return spec, path


def probe_curve(k, condition):
    """按tau网格计算单网格探针曲线（跨区域与种子合并）。"""
    import numpy as np
    positives = 0
    per_tau = {tau: Counter() for tau in TAU_GRID}
    per_tau_area = {tau: {} for tau in TAU_GRID}
    for area in AREAS:
        pairs = read_json(ROOT / 'DATA/processed_data/MassGIS/工程准备/正式导航协议冻结_v1'
                          / '元数据' / f'{area}_g{k}_探针.json')
        labels = [p['label'] for p in pairs]
        kinds = [p['kind'] for p in pairs]
        area_tau = {tau: Counter() for tau in TAU_GRID}
        for seed in SEEDS:
            path = CONFIRM / '特征' / f'probe_g{k}_s{seed}_{condition}_{area}.npy'
            seal = read_json(path.with_suffix('.seal.json'))
            assert sha_file(path) == seal['sha256'], path.name
            probs = np.load(path, allow_pickle=False)
            assert len(probs) == len(pairs)
            positives += sum(1 for l in labels if l != 4)
            for row, label, kind in zip(probs, labels, kinds):
                top = int(row.argmax())
                confidence = float(row[top])
                for tau in TAU_GRID:
                    counter = per_tau[tau]
                    area_c = area_tau[tau]
                    if top != 4 and confidence >= tau:
                        counter['accepted'] += 1
                        area_c['accepted'] += 1
                        if label == 4:
                            counter[f'{kind}_false_accept'] += 1
                            area_c[f'{kind}_false_accept'] += 1
                        elif top != label:
                            counter['adjacent_wrong_direction'] += 1
                            area_c['adjacent_wrong_direction'] += 1
                        else:
                            counter['adjacent_correct'] += 1
                            area_c['adjacent_correct'] += 1
        for tau in TAU_GRID:
            per_tau_area[tau][area] = dict(area_tau[tau])
    curve = {}
    for tau in TAU_GRID:
        c = per_tau[tau]
        accepted = c['accepted']
        correct = c['adjacent_correct']
        curve[tau] = dict(counts=dict(c),
                          precision=correct / accepted if accepted else None,
                          recall=correct / positives if positives else None)
    return dict(curve=curve, positives=positives, per_tau_area=per_tau_area)


def run_calibration():
    spec, _ = calibration_spec()
    full = {}
    for k in (10, 15):
        for condition in ('CueFull', 'CueMean'):
            full[f'g{k}_{condition}'] = probe_curve(k, condition)
    # 选择规则（按规范冻结）
    def ok(tau, cond):
        c = full[f'g{10}_{cond}']['curve'][tau]
        f = full[f'g{15}_{cond}']['curve'][tau]
        return (c['precision'] is not None and f['precision'] is not None
                and c['precision'] >= PRECISION_TARGET and f['precision'] >= PRECISION_TARGET
                and c['recall'] >= RECALL_FLOOR and f['recall'] >= RECALL_FLOOR)

    def recall_ok(tau):
        c = full['g10_CueFull']['curve'][tau]
        f = full['g15_CueFull']['curve'][tau]
        return c['recall'] >= RECALL_FLOOR and f['recall'] >= RECALL_FLOOR

    primary = next((tau for tau in TAU_GRID if ok(tau, 'CueFull')), None)
    if primary is not None:
        tau_star, rule_used = primary, 'precision>=0.90_and_recall>=0.90_both_grids'
    else:
        feasible = [t for t in TAU_GRID if recall_ok(t)]
        if not feasible:
            raise ValueError('校准失败：无任何tau满足召回下限；按停止条件收束')
        tau_star = max(feasible, key=lambda t: (full['g10_CueFull']['curve'][t]['precision']
                                                + full['g15_CueFull']['curve'][t]['precision'], -t))
        rule_used = 'fallback_max_precision_subject_to_recall'
    result = dict(protocol='dev001-threshold-calibration-v1', utc=stamp(),
                  spec=spec, curves=full, tau_star=tau_star, selection_rule_used=rule_used,
                  probe_adjacent_recall_at_tau_star=dict(
                      g10=full['g10_CueFull']['curve'][tau_star]['recall'],
                      g15=full['g15_CueFull']['curve'][tau_star]['recall']),
                  gate4_probe_adjacent_recall_satisfied=all(
                      full[f'g{k}_CueFull']['curve'][tau_star]['recall'] >= DEV_GATES['probe_adjacent_recall_at_least']
                      for k in (10, 15)))
    write_json(OUT / '元数据/校准结果.json', result)
    print(json.dumps({'tau_star': tau_star, 'rule': rule_used,
                      'recall_g10': result['probe_adjacent_recall_at_tau_star']['g10'],
                      'recall_g15': result['probe_adjacent_recall_at_tau_star']['g15']}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    run_calibration()
