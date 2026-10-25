"""Independent standard-library log counts and probability identities; no model calls."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = '项目导航/GeoExplorer策略分布一致性核查_v1'
TRAIN = 'DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v2'


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def recount(path, seed):
    totals = {'interior': 0, 'edge': 0, 'corner': 0}
    sources = {}; layers = {}; episodes = 0; invalid = 0; number = 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            if event.get('event') != 'environment_step':
                continue
            number += 1
            assert event['seed'] == seed and event['environment_steps'] == number
            old = event['observed_cells'][-2]; new = event['observed_cells'][-1]
            row, col = old // 5, old % 5
            neighbours = []
            for dx, dy in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                neighbours.append((row + dx, col + dy))
            degrees = sum(0 <= x < 5 and 0 <= y < 5 for x, y in neighbours)
            category = {2: 'corner', 3: 'edge', 4: 'interior'}[degrees]
            totals[category] += 1
            current_source = sources.setdefault(event['source'], {})
            current_source[category] = current_source.get(category, 0) + 1
            current_source['out_of_bounds_samples'] = current_source.get('out_of_bounds_samples', 0) + int(old == new)
            invalid += int(old == new)
            episodes += int(len(event['actions']) == 1)
            start = event['observed_cells'][0]; goal = event['goal']
            distance = abs(start // 5 - goal // 5) + abs(start % 5 - goal % 5)
            layer = layers.setdefault(str(distance), {})
            layer[category] = layer.get(category, 0) + 1
    assert number == 163840 and len(sources) == 109
    return dict(steps=number, state_classes=totals, sources=sources,
        initial_distance_steps=layers, episodes_with_actions=episodes, out_of_bounds_samples=invalid)


def audit(root=ROOT):
    root = Path(root); folder = root / OUT
    protocol = json.loads((folder / '冻结核查协议.json').read_text('utf-8'))
    summary = json.loads((folder / '离线核查汇总.json').read_text('utf-8'))
    for name, expected in protocol['input_sha256'].items():
        assert file_sha(root / name) == expected, name
    assert protocol['seeds'] == [0, 1, 2] and protocol['expected_steps_per_seed'] == 163840
    rows = []
    for seed, record in enumerate(summary['seeds']):
        own = recount(root / TRAIN / f'seed_{seed}/events.jsonl', seed)
        for key, value in own.items():
            assert record[key] == value, (seed, key)
        assert record['boundary_steps'] == own['state_classes']['corner'] + own['state_classes']['edge']
        rows.append({'seed': seed, 'steps': own['steps'], 'boundary_steps': record['boundary_steps'],
                     'sources': len(own['sources']), 'out_of_bounds_samples': own['out_of_bounds_samples']})
    epsilon = protocol['epsilon']
    for record in summary['synthetic_uniform_examples']:
        row, col = record['cell'] // 5, record['cell'] % 5
        legal = [row != 0, col != 4, row != 4, col != 0]
        assert record['legal'] == legal
        norm = sum(legal) / 4 + 4 * epsilon
        for action, allowed in enumerate(legal):
            probability = ((1 / 4 if allowed else 0) + epsilon) / norm
            ratio = norm / (1 + 4 * epsilon) if allowed else norm / (4 * epsilon)
            assert math.isclose(record['acting_probabilities'][action], probability, rel_tol=1e-12)
            assert math.isclose(record['original_same_parameter_ratios'][action], ratio, rel_tol=1e-12)
            assert record['consistent_same_parameter_ratios'][action] == 1
    assert summary['total_steps'] == 491520
    assert summary['boundary_steps'] == sum(r['boundary_steps'] for r in rows)
    assert summary['learned_ratios_measured'] is False and summary['causal_sr_claim'] is False
    assert all(summary[key] == 0 for key in ('project_forward_calls', 'training_updates', 'navigation_actions', 'model_api_calls'))
    return dict(passed=True, independently_recounted_steps=491520, per_seed=rows,
        probability_examples_checked=100, bound_files=len(protocol['input_sha256']),
        independent_implementation=True, independent_person_review=False,
        learned_probability_ratios_measured=False, new_sr=False,
        summary_sha256=file_sha(folder / '离线核查汇总.json'))


if __name__ == '__main__':
    result = audit()
    with (ROOT / OUT / '另实现复核.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
