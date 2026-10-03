"""One registered non-thinking DeepSeek batch; reuse the unchanged Gemma batch."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from threading import Event
from urllib.parse import urlparse

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.curl_transport import CurlTransport
from agents.vlm import hierarchical_observation_request
from data.make_episodes import MASA, json_bytes
from env.environment import GridWorldEnv
from env.episode import Episode, write_immutable_files
from eval import s2_suite
from eval.model_comparison import SRC, ROOT, MODELS, load_candidate, paired_results, verify_frozen
from eval.s2_suite import now, digest, read_json, read_lines

ORIGINAL = MASA / '评测结果/Gemma与DeepSeek同策略对照_v1'
OUTPUT = MASA / '评测结果/Gemma与DeepSeek同策略对照_v2'
DOC = ROOT / '选题报告相关/Gemma与DeepSeek同策略对照修复方案_v2.md'


class NonThinkingTransport(CurlTransport):
    receipts = []

    def request(self, method, url, headers, json=None):
        if urlparse(url).hostname != 'api.deepseek.com' or not url.endswith('/chat/completions'):
            raise ValueError('Repair transport is restricted to DeepSeek chat completions')
        if json is None or json.get('model') != 'deepseek-flash':
            raise ValueError('Unexpected model for non-thinking repair')
        payload = dict(json, thinking={'type': 'disabled'})
        self.receipts.append(dict(request_index=len(self.receipts) + 1, thinking=payload['thinking'],
            model=payload['model'], max_tokens=payload['max_tokens'],
            payload_sha256=sha256(json_bytes(payload)).hexdigest()))
        return super().request(method, url, headers, json=payload)


def verify_options(out, episodes, api):
    calls = read_lines(out / 'DeepSeek/API调用.jsonl')
    receipts = read_json(out / '供应商请求选项.json')
    records = read_lines(out / 'DeepSeek/任务结果.jsonl')
    expected = {e.episode_id: e for e in episodes}
    states = {}
    env = GridWorldEnv(MASA)
    for record in records:
        obs = env.reset(expected[record['episode_id']])
        states[record['episode_id'], 1] = obs
        for event in record['actions']:
            obs, _, _ = env.step(event['action'])
            states[record['episode_id'], event['step'] + 1] = obs
    assert len(receipts) == len(calls) <= 400
    for call, receipt in zip(calls, receipts):
        payload, audit = hierarchical_observation_request(states[call['episode_id'], call['step']], api)
        payload['thinking'] = {'type': 'disabled'}
        assert receipt['request_index'] == call['request_index']
        assert receipt['thinking'] == {'type': 'disabled'}
        assert sha256(json_bytes(payload)).hexdigest() == receipt['payload_sha256']
        assert all(call[k] == v for k, v in audit.items())
    return dict(passed=True, requests=len(calls), actual_payload_hashes_verified=True,
                provider_option='thinking.type=disabled', errors_also_checked=True)


def execute(out):
    prior = read_json(ORIGINAL / '预登记.json')
    assert not out.exists()
    episodes = [Episode.from_dict(e) for e in prior['selected_episodes']]
    deep_prior = read_json(ORIGINAL / 'DeepSeek/汇总.json')
    assert deep_prior['metrics']['completed'] == 0
    calls = read_lines(ORIGINAL / 'DeepSeek/API调用.jsonl')
    assert len(calls) == 3 and all(c.get('finish_reason') == 'length' for c in calls)
    url, model, path = MODELS['DeepSeek']
    api = load_candidate(path, url, model)
    out.mkdir(parents=True, exist_ok=False)
    sources = {p.relative_to(SRC).as_posix(): p for p in SRC.rglob('*.py')
               if p.relative_to(SRC).parts[0] in ('agents', 'env', 'data', 'eval', 'tests')}
    registration = dict(version='gemma-deepseek-same-G-v2-nonthinking', registered_utc=now(),
        original_registration_sha256=digest(ORIGINAL / '预登记.json'), original_directory=str(ORIGINAL),
        selected_episodes=prior['selected_episodes'], provider={**api.public(), 'thinking': {'type': 'disabled'}},
        reused_providers=['Gemma', 'Frontier'], max_new_requests=400,
        document_sha256=digest(DOC), source_sha256={n: digest(p) for n, p in sources.items()},
        reason='Three empty outputs truncated at 1024 reasoning tokens; no navigation result used to select repair',
        official_sources=['https://api-docs.deepseek.com/guides/thinking_mode', 'https://api-docs.deepseek.com/api/create-chat-completion'])
    payloads = {out / '源码快照' / n: p.read_bytes() for n, p in sources.items()}
    payloads.update({out / '预登记.json': json_bytes(registration), out / '方案冻结.md': DOC.read_bytes(),
                    out / '官方参数说明.md': ('# 已核对的官方参数\n\n核对日期：2026-09-30。\n\n'
                    'https://api-docs.deepseek.com/guides/thinking_mode\n\n'
                    'Thinking mode is enabled by default, with the default effort being high.\n\n'
                    'Thinking Mode Toggle: {"thinking": {"type": "enabled/disabled"}}\n\n'
                    'Thinking mode does not support the temperature, presence_penalty, or frequency_penalty parameters.\n\n'
                    'https://api-docs.deepseek.com/api/create-chat-completion\n\n'
                    'thinking.type: enabled / disabled; default enabled.\n').encode('utf-8')})
    write_immutable_files(payloads)
    # This module runs in a separate process. Bind this runner's transport dependency;
    # the concurrently running original Gemma process and shared source are unchanged.
    original_factory = s2_suite.CurlTransport
    NonThinkingTransport.receipts = []
    s2_suite.CurlTransport = NonThinkingTransport
    try:
        s2_suite.run_job(MASA, out, episodes, registration, dict(job_id='DeepSeek', arm='G', round=0), api, Event())
    finally:
        s2_suite.CurlTransport = original_factory
        write_immutable_files({out / '供应商请求选项.json': json_bytes(NonThinkingTransport.receipts)})
    write_immutable_files({out / '供应商选项复核.json': json_bytes(verify_options(out, episodes, api))})
    print(json.dumps(dict(repair_batch_finished=True, output=str(out)), ensure_ascii=False), flush=True)


def report(out):
    reg = read_json(out / '预登记.json')
    prior = read_json(ORIGINAL / '预登记.json')
    verify_frozen(ORIGINAL, prior)
    assert digest(ORIGINAL / '预登记.json') == reg['original_registration_sha256']
    assert digest(DOC) == digest(out / '方案冻结.md') == reg['document_sha256']
    for name, expected in reg['source_sha256'].items():
        assert digest(SRC / name) == digest(out / '源码快照' / name) == expected
    assert read_json(ORIGINAL / '执行状态.json')['status'] == 'completed'
    origins = {}
    for name in ('Gemma', 'Frontier'):
        assert read_json(ORIGINAL / name / '离线审计.json')['passed']
        src, dest = ORIGINAL / name, out / name
        hashes = {p.name: digest(p) for p in src.iterdir() if p.is_file()}
        if not dest.exists():
            shutil.copytree(src, dest)
        assert all(digest(dest / n) == h for n, h in hashes.items())
        origins[name] = dict(source=str(src), registration_sha256=reg['original_registration_sha256'], file_sha256=hashes)
    episodes = [Episode.from_dict(e) for e in reg['selected_episodes']]
    api = load_candidate(MODELS['DeepSeek'][2], *MODELS['DeepSeek'][:2])
    options = verify_options(out, episodes, api)
    jobs = {n: read_json(out / n / '汇总.json') for n in ('Gemma', 'DeepSeek', 'Frontier')}
    records = {n: read_lines(out / n / '任务结果.jsonl') for n in jobs}
    comparison = paired_results(episodes, records)
    extra = {}
    for name in ('Gemma', 'DeepSeek'):
        calls = read_lines(out / name / 'API调用.jsonl')
        returned = sorted({c['returned_model'] for c in calls if c.get('returned_model')})
        assert all(m == MODELS[name][1] for m in returned)
        extra[name] = dict(returned_models=returned, truncated=sum(c.get('finish_reason') == 'length' for c in calls),
            reported_reasoning_tokens=sum((c.get('usage') or {}).get('completion_tokens_details', {}).get('reasoning_tokens', 0) for c in calls))
    complete = all(jobs[n]['metrics']['completed'] == 20 for n in ('Gemma', 'DeepSeek'))
    summary = dict(jobs=jobs, paired=comparison, extra=extra, origins=origins, provider_options_audit=options,
                   both_models_complete=complete, formal_S2_passed=False,
                   deepseek_minus_gemma_SR_gate=jobs['DeepSeek']['metrics']['sr_macro'] - jobs['Gemma']['metrics']['sr_macro'])
    write_immutable_files({out / '复用来源.json': json_bytes(origins), out / '对照汇总.json': json_bytes(summary)})
    lines = ['# Gemma与DeepSeek同策略对照 v2', '',
        '模型服务配置：Gemma gemma4:31b（原服务模式）与DeepSeek deepseek-flash（显式非思考模式）。',
        '同一G策略、同一20道val题、4张源图、5×5、B=10、相同双图及公开信息权限、1024输出token上限、60秒超时；每模型一次。',
        'v1 DeepSeek默认高强度思考前三题均因1024个推理token耗尽而没有输出动作，本批只修复该接口模式。v1失败保留；Gemma与Frontier逐文件复用v1的唯一批次，没有重跑或挑选。', '',
        '| 方法 | 完成/计划 | 成功数 | SR_gate | SG_gate | 重访率 | API请求/错误 |', '|---|---:|---:|---:|---:|---:|---:|']
    for n, j in jobs.items():
        m, a = j['metrics'], j['api']
        repeat = f"{m['repeat_rate']:.2%}" if m['repeat_rate'] is not None else '缺失'
        lines.append(f"| {n} | {m['completed']}/20 | {m['successes']} | {m['sr_macro']:.1%} | {m['sg_macro']:.3f} | {repeat} | {a['attempts']}/{a['errors']} |")
    lines += ['', 'SR_gate分母为计划20题。中断/未运行SG_gate按8补入，仅为保守交付评估；正常完成题的SR_nav/SG_nav另见JSON。', '',
        f"两模型共同完成{comparison['common_completed']}/20题。配对计数：`{json.dumps(comparison['counts'], ensure_ascii=False)}`。",
        f"DeepSeek减Gemma的计划题SR差为{summary['deepseek_minus_gemma_SR_gate']*100:+.1f}个百分点。",
        ('两模型均完成全部题，本轮可作完整小规模描述性对照。' if complete else '至少一方未完成全部题，不能把接口可靠性差异直接当作导航能力差异。'), '',
        '| 模型 | 请求P50/P95秒 | 题耗时P50/P95秒 | 输入token | 输出token | 截断 |', '|---|---:|---:|---:|---:|---:|']
    def fmt(x): return '缺失' if x is None else f'{x:.2f}'
    for n in ('Gemma', 'DeepSeek'):
        j = jobs[n]; a = j['api']; u = a['usage'] or {}
        lines.append(f"| {n} | {fmt(a['latency_p50'])}/{fmt(a['latency_p95'])} | {fmt(j['episode_latency_p50'])}/{fmt(j['episode_latency_p95'])} | {u.get('prompt_tokens','缺失')} | {u.get('completion_tokens','缺失')} | {extra[n]['truncated']} |")
    lines += ['', '调用并非严格同步，耗时受供应商服务负载影响；价格未核实，不估算费用。失败/未运行会影响题耗时，不把这些短耗时解释为导航更快。', '',
        '| C | Gemma成功/计划 | DeepSeek成功/计划 | Frontier成功/计划 |', '|---|---:|---:|---:|']
    for d in range(4, 9):
        cells = [f"{jobs[n]['metrics']['by_distance'][str(d)]['successes']}/4" for n in jobs]
        lines.append(f"| {d} | {' | '.join(cells)} |")
    lines += ['', '| 源图 | Gemma成功/计划 | DeepSeek成功/计划 | Frontier成功/计划 |', '|---|---:|---:|---:|']
    for area in sorted({e.area for e in episodes}):
        cells = [f"{jobs[n]['metrics']['by_area'][area]['successes']}/5" for n in jobs]
        lines.append(f"| {area} | {' | '.join(cells)} |")
    lines += ['', '全部轨迹/回答解析/治理器/公开输入与图像哈希经离线重放；DeepSeek实际请求新增参数及最终负载哈希逐条复核，包括失败请求。',
        '本轮仅4张源图1轮，不宣称稳定优越；没有目标遮蔽消融，不能证明目标贡献；非思考模式结果不代表DeepSeek默认高强度思考的能力。',
        '模型服务方案差异不能单独归因于参数量；未用test，未训练本地GRU，未通过完整S2。批次结束，不自动扩展。']
    write_immutable_files({out / '对照报告.md': '\n'.join(lines).encode('utf-8'),
                          out / '执行状态.json': json_bytes(dict(status='completed', both_models_complete=complete, audits_passed=True))})
    print(json.dumps(dict(both_models_complete=complete, paired=comparison['counts']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--confirm-cloud-transmission', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        report(OUTPUT)
    elif args.confirm_cloud_transmission:
        execute(OUTPUT)
    else:
        raise ValueError('Cloud transmission must be explicitly enabled')
