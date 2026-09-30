"""Bounded Gemma/DeepSeek comparison using the frozen existing G strategy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from threading import Event

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.vlm import APIConfig, HIERARCHICAL_SYSTEM_PROMPT, HIERARCHICAL_PROMPT_VERSION
from data.make_episodes import MASA, json_bytes
from env.episode import Episode, write_immutable_files
from eval.evaluate import verify_task_file
from eval.s2_suite import run_job, now, read_lines, digest
from eval.validation_suite import select_validation

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parents[1]
DOC = ROOT / '选题报告相关/Gemma与DeepSeek同策略对照方案_v1.md'
OUTPUT = MASA / '评测结果/Gemma与DeepSeek同策略对照_v1'
MODELS = {'Gemma': ('https://ollama.com/v1', 'gemma4:31b', ROOT / 'project/.env'),
          'DeepSeek': ('https://api.deepseek.com/v1', 'deepseek-flash', ROOT / 'project/.env.deepseek')}


def load_candidate(path, expected_url, expected_model):
    values = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            values[k.strip()] = v.strip().strip('\"\'')
    if values.get('VLM_BASE_URL') != expected_url or values.get('VLM_MODEL') != expected_model:
        raise ValueError('Candidate endpoint/model does not match the registered comparison')
    return APIConfig(expected_url, expected_model, values['VLM_API_KEY'], timeout=60, max_tokens=1024)


def paired_results(episodes, results):
    indexed = {name: {r['episode_id']: r for r in records} for name, records in results.items()}
    pairs = []
    counts = dict(both_success=0, gemma_only=0, deepseek_only=0, both_failure=0, incomplete_pair=0)
    for ep in episodes:
        a, b = (indexed[name][ep.episode_id] for name in ('Gemma', 'DeepSeek'))
        if a['completion'] == b['completion'] == 'completed':
            sa, sb = a['evaluation']['success'], b['evaluation']['success']
            outcome = 'both_success' if sa and sb else 'gemma_only' if sa else 'deepseek_only' if sb else 'both_failure'
        else:
            outcome = 'incomplete_pair'
        counts[outcome] += 1
        pairs.append(dict(episode_id=ep.episode_id, area=ep.area, distance=ep.dist, outcome=outcome,
                          **{name: dict(completion=r['completion'], evaluation=r['evaluation'])
                             for name, r in [('Gemma', a), ('DeepSeek', b)]}))
    complete = len(episodes) - counts['incomplete_pair']
    return dict(counts=counts, common_completed=complete, rows=pairs,
                common_SR_difference=(counts['deepseek_only'] - counts['gemma_only']) / complete if complete else None,
                note='Common-completed subset may be selected by provider failures; planned-task metrics remain primary.')


def register(out, candidates):
    episodes, manifest = verify_task_file(MASA, MASA / '任务清单_v2/episodes_val.jsonl', 'val')
    selected = select_validation(episodes)
    out.mkdir(parents=True, exist_ok=False)
    sources = {p.relative_to(SRC).as_posix(): p for p in SRC.rglob('*.py')
               if p.relative_to(SRC).parts[0] in ('agents', 'env', 'data', 'eval', 'tests')}
    registration = dict(version='gemma-deepseek-same-G-v1', registered_utc=now(),
        providers={name: api.public() for name, api in candidates.items()},
        strategy='G / HierarchicalSearchGovernor, unchanged', prompt=HIERARCHICAL_SYSTEM_PROMPT,
        prompt_version=HIERARCHICAL_PROMPT_VERSION, selected_episodes=[asdict(e) for e in selected],
        unique_routes=len({(e.area, e.start, e.goal) for e in selected}), rounds=1,
        max_requests_per_provider=400, total_max_requests=800, max_tokens=1024, timeout=60,
        provider_concurrency=1, providers_parallel=True,
        failure_circuit='401/403/429 after one allowed transient retry, or three consecutive API-error episodes; per provider',
        task_sha256=manifest['episodes_sha256'], manifest_sha256=digest(MASA / '任务清单_v2/manifest_val.json'),
        area_sha256=manifest['area_sha256'], source_sha256={n: digest(p) for n, p in sources.items()},
        document_sha256=digest(DOC), test_used=False, formal_S2=False,
        scope='20 shared validation tasks, existing G, model-service comparison, no parameter-size causal claim')
    payloads = {out / '源码快照' / name: p.read_bytes() for name, p in sources.items()}
    payloads.update({out / '方案冻结.md': DOC.read_bytes(), out / '预登记.json': json_bytes(registration),
                     out / '原val任务清单.jsonl': (MASA / '任务清单_v2/episodes_val.jsonl').read_bytes(),
                     out / '原val清单哈希.json': (MASA / '任务清单_v2/manifest_val.json').read_bytes()})
    write_immutable_files(payloads)
    return selected, registration


def verify_frozen(out, registration):
    _, manifest = verify_task_file(MASA, MASA / '任务清单_v2/episodes_val.jsonl', 'val')
    assert manifest['episodes_sha256'] == registration['task_sha256']
    assert manifest['area_sha256'] == registration['area_sha256']
    assert digest(MASA / '任务清单_v2/manifest_val.json') == registration['manifest_sha256']
    assert digest(DOC) == digest(out / '方案冻结.md') == registration['document_sha256']
    for name, expected in registration['source_sha256'].items():
        assert digest(SRC / name) == digest(out / '源码快照' / name) == expected


def finish_report(out, episodes, registration, candidates, jobs):
    verify_frozen(out, registration)
    all_records = {name: read_lines(out / name / '任务结果.jsonl') for name in jobs}
    comparison = paired_results(episodes, all_records)
    details = {}
    for name in ('Gemma', 'DeepSeek'):
        calls = read_lines(out / name / 'API调用.jsonl')
        attempts = read_lines(out / name / '请求意图.jsonl')
        assert len(calls) == len(attempts) <= 400
        assert sum(c['status'] == 'ok' for c in calls) == sum(len(r['actions']) for r in all_records[name])
        api = candidates[name]
        details[name] = dict(returned_models=sorted({c['returned_model'] for c in calls if c.get('returned_model')}),
            truncated=sum(c.get('finish_reason') == 'length' for c in calls),
            http_errors={str(code): sum(c.get('http_status') == code for c in calls)
                         for code in sorted({c['http_status'] for c in calls if c.get('http_status', 0) and c['http_status'] >= 300})},
            observed_reasoning_tokens=sum(c.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', 0)
                                          for c in calls if isinstance(c.get('usage'), dict)),
            reasoning_tokens_reported_calls=sum(isinstance(c.get('usage'), dict)
                and 'reasoning_tokens' in c['usage'].get('completion_tokens_details', {}) for c in calls))
        for call in calls:
            if call.get('returned_model'):
                assert call['returned_model'] == api.model, 'Unexpected returned model'
    for path in out.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.md'):
            content = path.read_text(encoding='utf-8')
            assert all(api.api_key not in content for api in candidates.values())
            assert 'data:image/' not in content
    delta = jobs['DeepSeek']['metrics']['sr_macro'] - jobs['Gemma']['metrics']['sr_macro']
    full = all(jobs[n]['metrics']['completed'] == len(episodes) for n in candidates)
    summary = dict(jobs=jobs, paired=comparison, api_details=details,
        deepseek_minus_gemma_SR_gate=delta, both_models_complete=full,
        task_count_per_model=len(episodes), source_count=len({e.area for e in episodes}),
        unique_routes=registration['unique_routes'], audits_passed=True,
        formal_S2_passed=False, test_used=False, scope=registration['scope'])
    write_immutable_files({out / '对照汇总.json': json_bytes(summary)})
    lines = ['# Gemma与DeepSeek同策略对照 v1', '',
        '共同策略为原G层级搜索；相同validation20、双图/公开状态、5×5、B=10、temperature=0、max_tokens=1024、超时60秒。',
        '每模型一轮，仅4张源图；后续观察随各自轨迹变化。此批比较服务模型方案，不能分离参数量、预训练或推理实现的作用。', '',
        '| 模型/参考 | 正常完成/计划 | 成功数 | SR_gate | SG_gate | SR_nav | SG_nav | 重访率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value, percent=False):
        return '缺失' if value is None else f'{value:.2%}' if percent else f'{value:.3f}'
    for name, job in jobs.items():
        m = job['metrics']
        lines.append(f"| {name} | {m['completed']}/{m['planned']} | {m['successes']} | {m['sr_macro']:.1%} | {m['sg_macro']:.3f} | {fmt(m['sr_nav'], True)} | {fmt(m['sg_nav'])} | {fmt(m['repeat_rate'], True)} |")
    lines += ['', 'SR_gate使用全部计划题分母；未完成/未运行题的SG_gate按8补入，仅用于保守交付评估，不是实际终点距离。SR_nav/SG_nav只在正常完成题上统计。', '',
        '| 模型 | API尝试 | API错误 | 截断 | 请求P50/P95秒 | 题耗时P50/P95秒 | 报告输入/输出token |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in candidates:
        j, d = jobs[name], details[name]
        a, u = j['api'], j['api']['usage'] or {}
        lines.append(f"| {name} | {a['attempts']} | {a['errors']} | {d['truncated']} | {fmt(a['latency_p50'])}/{fmt(a['latency_p95'])} | {fmt(j['episode_latency_p50'])}/{fmt(j['episode_latency_p95'])} | {u.get('prompt_tokens', '缺失')}/{u.get('completion_tokens', '缺失')} |")
    lines += ['', '题耗时包含错误/未运行题，若有not_run其近零耗时不能解释为更快完成导航；费用未核实，不估算。输出token可能包含供应商推理token，详细已报告字段见JSON。', '',
        '## 配对结果与结论', '',
        f"双方共同完成{comparison['common_completed']}/20题；分类计数：`{json.dumps(comparison['counts'], ensure_ascii=False)}`。",
        f"DeepSeek减Gemma的计划题SR差：{delta * 100:+.1f}个百分点。",
        ('两模型均完成全部计划题。本批只提供描述性模型比较，不宣称稳定优越；需要新一轮预登记重复和更多独立源图验证。' if full else
         '至少一个模型未完成全部任务，模型导航能力对比受接口/输出失败影响。共同完成子集可能有选择偏差，不能据此宣布模型导航能力优劣。'), '',
        '## 各距离档', '', '| C | Gemma成功/计划 | DeepSeek成功/计划 | Frontier成功/计划 |', '|---|---:|---:|---:|']
    for distance in range(4, 9):
        cells = [f"{jobs[n]['metrics']['by_distance'][str(distance)]['successes']}/{jobs[n]['metrics']['by_distance'][str(distance)]['planned']}" for n in ('Gemma', 'DeepSeek', 'Frontier')]
        lines.append(f"| {distance} | {' | '.join(cells)} |")
    lines += ['', '## 各源图', '', '| 源图 | Gemma成功/计划 | DeepSeek成功/计划 | Frontier成功/计划 |', '|---|---:|---:|---:|']
    for area in sorted({e.area for e in episodes}):
        cells = [f"{jobs[n]['metrics']['by_area'][area]['successes']}/{jobs[n]['metrics']['by_area'][area]['planned']}" for n in ('Gemma', 'DeepSeek', 'Frontier')]
        lines.append(f"| {area} | {' | '.join(cells)} |")
    lines += ['', '## 核验与边界', '',
        '逐题环境重放、原始回答解析、G治理器重放、公开输入和图像哈希核对通过；源码/任务/图像/冻结方案未变。',
        '本轮未做目标遮蔽或错图消融，不证明目标视觉贡献；未评测test，不修改本地GRU，不通过完整S2。原模型历史成绩不与本批混算。']
    write_immutable_files({out / '对照报告.md': '\n'.join(lines).encode('utf-8')})
    write_immutable_files({out / '执行状态.json': json_bytes(dict(status='completed', providers_complete=full, audits_passed=True))})
    print(json.dumps(dict(final=True, both_models_complete=full,
                         models={n: jobs[n]['metrics'] for n in candidates}), ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--confirm-cloud-transmission', action='store_true')
    args = parser.parse_args()
    if not args.confirm_cloud_transmission:
        raise ValueError('Cloud image transmission must be explicitly enabled')
    candidates = {n: load_candidate(p, url, model) for n, (url, model, p) in MODELS.items()}
    selected, registration = register(args.output_dir, candidates)
    print(json.dumps(dict(registered=str(args.output_dir), models=list(candidates), tasks_each=len(selected), max_requests=800)), flush=True)
    try:
        jobs = {'Frontier': run_job(MASA, args.output_dir, selected, registration,
                     dict(job_id='Frontier', arm='Frontier', round=0), candidates['Gemma'], Event())}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(run_job, MASA, args.output_dir, selected, registration,
                       dict(job_id=name, arm='G', round=0), api, Event()) for name, api in candidates.items()}
            for name, future in futures.items():
                jobs[name] = future.result()
        finish_report(args.output_dir, selected, registration, candidates, jobs)
    except Exception as exc:
        write_immutable_files({args.output_dir / '执行异常.json': json_bytes(dict(type=type(exc).__name__))})
        raise


if __name__ == '__main__':
    main()
