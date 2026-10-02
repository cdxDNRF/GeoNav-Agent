"""Independent post-run arithmetic and all-request input audit; no API calls."""
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.governor import COARSE_REGIONS, HierarchicalSearchGovernor
from agents.vlm import APIConfig, hierarchical_observation_request, parse_hierarchical_ranked
from data.make_episodes import MASA, json_bytes
from env.environment import GridWorldEnv
from env.episode import Episode, write_immutable_files
from eval.model_comparison_repair import ORIGINAL, OUTPUT, verify_options
from eval.s2_suite import read_json, read_lines, digest


def main():
    old_reg, new_reg = read_json(ORIGINAL / '预登记.json'), read_json(OUTPUT / '预登记.json')
    summary = read_json(OUTPUT / '对照汇总.json')
    assert old_reg['selected_episodes'] == new_reg['selected_episodes']
    assert digest(ORIGINAL / '预登记.json') == new_reg['original_registration_sha256']
    episodes = [Episode.from_dict(e) for e in old_reg['selected_episodes']]
    episode_map = {e.episode_id: e for e in episodes}
    assert len(episodes) == len(episode_map) == 20
    assert len({e.area for e in episodes}) == 4
    assert Counter(e.dist for e in episodes) == {d: 4 for d in range(4, 9)}
    assert all(e.split == 'val' and e.budget == 10 and e.grid_size == 5 for e in episodes)
    config_a, config_b = (old_reg['providers'][n] for n in ('Gemma', 'DeepSeek'))
    assert {k: v for k, v in config_a.items() if k not in ('model', 'base_url')} == {
        k: v for k, v in config_b.items() if k not in ('model', 'base_url')}
    assert new_reg['provider']['max_tokens'] == config_a['max_tokens'] == 1024
    assert new_reg['provider']['thinking'] == {'type': 'disabled'}
    totals, observed = {}, {}
    for name in ('Gemma', 'DeepSeek', 'Frontier'):
        folder = OUTPUT / name
        records, calls, intents = [read_lines(folder / f) for f in ('任务结果.jsonl', 'API调用.jsonl', '请求意图.jsonl')]
        assert len(records) == 20 and {r['episode_id'] for r in records} == set(episode_map)
        assert len(calls) == len(intents) <= 400
        assert read_json(folder / '离线审计.json')['passed']
        completed = [r for r in records if r['completion'] == 'completed']
        successes = sum(r['evaluation']['success'] for r in completed)
        m = summary['jobs'][name]['metrics']
        assert successes == m['successes']
        assert len(completed) == m['completed']
        assert abs(m['sr_gate'] - successes / 20) < 1e-12
        assert abs(m['sg_gate'] - sum(r['evaluation']['sg'] if r['completion'] == 'completed' else 8 for r in records) / 20) < 1e-12
        actions = [a for r in records for a in r['actions']]
        assert len(actions) == m['executed_actions']
        assert not actions or abs(m['repeat_rate'] - sum(a['revisited'] for a in actions) / len(actions)) < 1e-12
        assert summary['jobs'][name]['api']['errors'] == sum(c['status'] != 'ok' for c in calls)
        states, executed = {}, {}
        env = GridWorldEnv(MASA)
        for record in records:
            ep = episode_map[record['episode_id']]
            obs = env.reset(ep)
            states[ep.episode_id, 1] = obs
            for event in record['actions']:
                executed[ep.episode_id, event['step']] = event['action']
                obs, done, info = env.step(event['action'])
                states[ep.episode_id, event['step'] + 1] = obs
                assert event['patch_id'] == obs.position[0] * 5 + obs.position[1]
                assert event['revisited'] == info.revisited and event['out_of_bounds'] == info.out_of_bounds
            if record['completion'] == 'completed':
                assert env.done and record['evaluation'] == env.evaluator_result()
            else:
                assert not env.done and record['evaluation'] is None
        if name != 'Frontier':
            public = old_reg['providers'][name]
            api = APIConfig(public['base_url'], public['model'], 'audit-placeholder', max_tokens=1024, timeout=60)
            decision_keys = set()
            for call in calls:
                obs = states[call['episode_id'], call['step']]
                _, audit = hierarchical_observation_request(obs, api)
                assert all(call[k] == v for k, v in audit.items())
                if call['status'] == 'ok':
                    regions, ranked, _ = parse_hierarchical_ranked(call['raw_content'], obs.legal_actions, COARSE_REGIONS, call['finish_reason'])
                    key = call['episode_id'], call['step']
                    assert key not in decision_keys
                    assert HierarchicalSearchGovernor().choose(obs, ranked, regions).selected_action == executed[key]
                    decision_keys.add(key)
            assert decision_keys == set(executed)
            if name == 'DeepSeek':
                assert verify_options(OUTPUT, episodes, api)['passed']
        totals[name] = dict(episodes=len(records), completed=len(completed), successes=successes, requests=len(calls),
                           valid_decisions=sum(c['status'] == 'ok' for c in calls), invalid_decisions=sum(c['status'] != 'ok' for c in calls),
                           actions=len(actions), missing_outcome_SR_bounds=[successes / 20, (successes + 20 - len(completed)) / 20])
        observed[name] = {r['episode_id']: r for r in records}
    outcomes = Counter()
    for ep in episodes:
        a, b = observed['Gemma'][ep.episode_id], observed['DeepSeek'][ep.episode_id]
        if a['completion'] != 'completed' or b['completion'] != 'completed':
            outcomes['incomplete_pair'] += 1
        else:
            flags = bool(a['evaluation']['success']), bool(b['evaluation']['success'])
            outcomes[{(True, True): 'both_success', (True, False): 'gemma_only',
                      (False, True): 'deepseek_only', (False, False): 'both_failure'}[flags]] += 1
    assert dict(outcomes) == {k: v for k, v in summary['paired']['counts'].items() if v}
    origins = read_json(OUTPUT / '复用来源.json')
    for name, origin in origins.items():
        for filename, expected in origin['file_sha256'].items():
            assert digest(ORIGINAL / name / filename) == digest(OUTPUT / name / filename) == expected
    old_calls = read_lines(ORIGINAL / 'DeepSeek/API调用.jsonl')
    assert len(old_calls) == 3 and all(c['finish_reason'] == 'length' and c['raw_content'] == '' for c in old_calls)
    report = dict(status='passed', totals=totals, pairs=dict(outcomes), task_and_budget_matched=True,
                  all_request_inputs_checked_including_failures=True, parser_governor_environment_replayed=True,
                  metrics_recomputed=True, source_of_reused_runs_verified=True, old_failed_batch_preserved=True,
                  formal_S2_passed=False, api_calls_by_audit=0,
                  artifacts_sha256={p.relative_to(OUTPUT).as_posix(): digest(p) for p in OUTPUT.rglob('*') if p.is_file()})
    write_immutable_files({OUTPUT / '独立复核.json': json_bytes(report)})
    print(json.dumps({k: v for k, v in report.items() if k != 'artifacts_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
