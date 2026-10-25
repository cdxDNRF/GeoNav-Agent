"""Read-only, streamed training-log diagnosis; no torch/model/evaluator imports."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = '项目导航/GeoExplorer策略分布一致性核查_v1'
TRAIN = 'DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v2'
PUBLIC_PPO = '项目导航/GeoExplorer资源验证_v1/公开来源/GeoExplorer__models__ppo.py'
LOCAL_TRAIN = 'project/src/train/geoexplorer_training_v1.py'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(root=ROOT):
    root = Path(root)
    folder = root / OUTPUT
    folder.mkdir(exist_ok=False)
    names = [f'{TRAIN}/seed_{s}/events.jsonl' for s in range(3)] + [
        f'{TRAIN}/元数据/冻结执行协议.json', f'{TRAIN}/核验/训练复核_完整_v1.json',
        PUBLIC_PPO, LOCAL_TRAIN, 'project/src/train/geoexplorer_training_math_v1.py',
        'project/local_policy_default.json', 'project/local_policy_defaults_v1.json',
        'project/defaults/masa_roads_grid10_radial_v1.json',
        'project/src/agents/action_distribution_v1.py',
        'project/src/eval/geoexplorer_distribution_diagnosis_v1.py',
        'project/src/eval/audit_geoexplorer_distribution_v1.py',
        'project/src/tests/test_action_distribution_v1.py']
    protocol = dict(task='BENCH-023', stage='offline_mechanism_diagnosis',
        input_sha256={name: digest(root / name) for name in names},
        seeds=[0, 1, 2], expected_steps_per_seed=163840, grid=5, fit_sources=109,
        primary_question='Do acting and PPO evaluate use the same normalized action distribution?',
        scope='Actual boundary exposure only; learned probability ratios were not logged and cannot be reconstructed without forward.',
        synthetic_probability_vector=[.25] * 4, epsilon=1e-6,
        candidate_limit=1, output_limit_bytes=2 * 1024**2,
        no_training=True, no_project_forward=True, no_navigation=True, no_model_api=True,
        no_new_sources=True, defaults_unchanged=True)
    write_new(folder / '冻结核查协议.json', protocol)
    return protocol


def kind(cell, grid=5):
    row, col = divmod(cell, grid)
    axes = int(row in (0, grid - 1)) + int(col in (0, grid - 1))
    return ('interior', 'edge', 'corner')[axes]


def move(cell, action, grid=5):
    row, col = divmod(cell, grid)
    dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[action]
    r, c = row + dr, col + dc
    return r * grid + c if 0 <= r < grid and 0 <= c < grid else cell


def summarize_log(path, seed, expected=163840):
    counts = Counter(); sources = {}; initial_distances = {}; previous = None; episodes = 0
    for line in Path(path).open(encoding='utf-8'):
        event = json.loads(line)
        if event.get('event') != 'environment_step':
            continue
        cells, actions, goal = event['observed_cells'], event['actions'], event['goal']
        if event['seed'] != seed or event['environment_steps'] != sum(counts.values()) + 1:
            raise ValueError('training step identity or sequence mismatch')
        if len(cells) != len(actions) + 1 or not actions or any(type(c) is not int or not 0 <= c < 25 for c in cells):
            raise ValueError('invalid path')
        if type(goal) is not int or not 0 <= goal < 25 or actions[-1] not in range(4):
            raise ValueError('invalid training-side goal/action')
        if move(cells[-2], actions[-1]) != cells[-1] or bool(event['terminal_goal']) != (cells[-1] == goal):
            raise ValueError('training transition or terminal mismatch')
        if len(actions) == 1:
            episodes += 1
        elif previous is None or previous['observed_cells'] != cells[:-1] or previous['actions'] != actions[:-1] or previous['goal'] != goal or previous['source'] != event['source'] or previous['terminal_goal']:
            raise ValueError('episode prefix identity mismatch')
        category = kind(cells[-2]); counts[category] += 1
        src = sources.setdefault(event['source'], Counter())
        src[category] += 1
        src['out_of_bounds_samples'] += int(cells[-2] == cells[-1])
        r0, c0 = divmod(cells[0], 5); gr, gc = divmod(goal, 5)
        initial = abs(r0 - gr) + abs(c0 - gc)
        initial_distances.setdefault(str(initial), Counter())[category] += 1
        previous = event
    steps = sum(counts.values())
    if steps != expected:
        raise ValueError('planned training-step denominator mismatch')
    return dict(seed=seed, steps=steps, episodes_with_actions=episodes,
        state_classes={key: counts[key] for key in ('interior', 'edge', 'corner')},
        boundary_steps=counts['edge'] + counts['corner'],
        boundary_fraction=(counts['edge'] + counts['corner']) / steps,
        out_of_bounds_samples=sum(x['out_of_bounds_samples'] for x in sources.values()),
        sources={name: dict(x) for name, x in sorted(sources.items())},
        initial_distance_steps={d: dict(x) for d, x in sorted(initial_distances.items())})


def uniform_ratios(grid=5, epsilon=1e-6):
    rows = []
    for cell in range(grid * grid):
        legal = [move(cell, a, grid) != cell for a in range(4)]
        weights = [.25 * int(x) + epsilon for x in legal]
        q = [x / sum(weights) for x in weights]
        rows.append(dict(cell=cell, kind=kind(cell, grid), legal=legal,
            acting_probabilities=q, original_same_parameter_ratios=[.25 / x for x in q],
            consistent_same_parameter_ratios=[1.] * 4))
    return rows


def source_contract(root):
    parsed = ast.parse((root / PUBLIC_PPO).read_text('utf-8-sig'))
    actor = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'ActorCritic')
    methods = {n.name: ast.unparse(n) for n in actor.body if isinstance(n, ast.FunctionDef)}
    local = ast.parse((root / LOCAL_TRAIN).read_text('utf-8-sig'))
    functions = {n.name: ast.unparse(n) for n in local.body if isinstance(n, ast.FunctionDef)}
    checks = {'public_act_masks': 'torch.multiply' in methods['act'],
        'public_act_adds_epsilon': 'Categorical(action_probs + 1e-06)' in methods['act'],
        'public_evaluate_unmasked': 'Categorical(action_probs)' in methods['evaluate'] and 'patch_sequence' not in methods['evaluate'],
        'local_act_masks': 'probabilities * mask + 1e-06' in functions['masked_distribution'],
        'local_update_uses_evaluate': 'policy.evaluate(states, actions)' in functions['update_ppo']}
    if not all(checks.values()):
        raise ValueError('bound source differs from the diagnosed contract')
    return checks


def run(root=ROOT):
    root = Path(root); folder = root / OUTPUT
    protocol = json.loads((folder / '冻结核查协议.json').read_text('utf-8'))
    for name, expected in protocol['input_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError('input drift: ' + name)
    result = dict(task='BENCH-023', source_contract=source_contract(root),
        seeds=[summarize_log(root / TRAIN / f'seed_{s}/events.jsonl', s) for s in protocol['seeds']],
        synthetic_uniform_examples=uniform_ratios(),
        learned_ratios_measured=False, causal_sr_claim=False, new_sr=False,
        project_forward_calls=0, training_updates=0, navigation_actions=0, model_api_calls=0)
    result['total_steps'] = sum(x['steps'] for x in result['seeds'])
    result['boundary_steps'] = sum(x['boundary_steps'] for x in result['seeds'])
    result['boundary_fraction'] = result['boundary_steps'] / result['total_steps']
    write_new(folder / '离线核查汇总.json', result)
    return {'total_steps': result['total_steps'], 'boundary_steps': result['boundary_steps'],
            'boundary_fraction': result['boundary_fraction'], 'source_contract': result['source_contract']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run'])
    args = parser.parse_args()
    value = prepare() if args.command == 'prepare' else run()
    print(json.dumps({'task': 'BENCH-023', **({k: v for k, v in value.items() if k != 'input_sha256'})}, ensure_ascii=False))
