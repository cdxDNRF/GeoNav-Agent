"""固定20道val任务，运行最小VLM/规则探索/Random及仅评测端诊断。"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.exploration import FrontierPolicy
from agents.vlm import APIConfig, APITrialError, PROMPT_VERSION, SYSTEM_PROMPT, VLMPolicy
from data.make_episodes import MASA, json_bytes, seed_for
from env.environment import GridWorldEnv
from env.episode import PROTOCOL, write_immutable_files
from eval.diagnostics import analyze_positions
from eval.evaluate import RandomPolicy, metrics, verify_task_file
from eval.pilot import call_statistics

POLICIES = ("random", "frontier", "vlm")


def select_validation(episodes):
    if any(ep.split != "val" for ep in episodes):
        raise ValueError("只接受验证集")
    areas = sorted({ep.area for ep in episodes}, key=lambda a: int(a.split("_")[1]))
    if len(areas) != 4:
        raise ValueError("本轮设计要求4个验证区域")
    chosen = []
    for distance in range(4, 9):
        for area in areas:
            candidates = [e for e in episodes if e.dist == distance and e.area == area]
            if not candidates:
                raise ValueError("距离/区域组合缺失")
            chosen.append(min(candidates, key=lambda e: sha256(("validation20-v1|" + e.episode_id).encode()).hexdigest()))
    return chosen


def transient(error):
    return error.reason == "transport_error" or error.record.get("http_status") in (408, 429, 500, 502, 503, 504)


def aggregate_records(records):
    finished = [r for r in records if r["completion"] == "completed"]
    diagnostics = [r["diagnostics"] for r in finished]
    return {"scheduled": len(records), "completed": len(finished),
            "errors": len(records) - len(finished),
            "metrics_completed_only": metrics([r["evaluation"] for r in finished]) if finished else None,
            "completed_diagnostics": {
                "episodes_with_two_cell_cycles": sum(d["two_cell_cycle_present"] for d in diagnostics),
                "cycle_episode_rate": statistics.mean(d["two_cell_cycle_present"] for d in diagnostics) if diagnostics else None,
                "mean_unique_cells": statistics.mean(d["unique_cells_including_start"] for d in diagnostics) if diagnostics else None,
                "mean_coverage_ratio": statistics.mean(d["coverage_ratio"] for d in diagnostics) if diagnostics else None,
                "mean_minimum_distance": statistics.mean(d["minimum_distance"] for d in diagnostics) if diagnostics else None,
                "adjacent_but_not_reached_episodes": sum(d["adjacent_but_not_reached"] for d in diagnostics),
                "departures_from_adjacent": sum(d["departures_from_adjacent"] for d in diagnostics)},
            "failure_note": "API错误题单列，不计入正常完成分母；错误题轨迹诊断仍留逐题日志"}


def summarize(records, calls, configuration):
    by_policy = {}
    for policy in POLICIES:
        selected = [r for r in records if r["policy"] == policy]
        by_policy[policy] = {"all": aggregate_records(selected),
            "by_distance": {str(d): aggregate_records([r for r in selected if r["distance"] == d]) for d in range(4, 9)},
            "by_area": {area: aggregate_records([r for r in selected if r["area"] == area]) for area in sorted({r["area"] for r in selected})}}
    paired = []
    for ep in configuration["selected_episodes"]:
        row = {"episode_id": ep["episode_id"], "area": ep["area"], "distance": ep["dist"]}
        for policy in POLICIES:
            record = next(r for r in records if r["policy"] == policy and r["episode_id"] == ep["episode_id"])
            row[policy] = {"completion": record["completion"], "success": record["evaluation"]["success"] if record["evaluation"] else None,
                           "sg": record["evaluation"]["sg"] if record["evaluation"] else None,
                           "unique_cells": record["diagnostics"]["unique_cells_including_start"],
                           "cycle": record["diagnostics"]["two_cell_cycle_present"]}
        paired.append(row)
    return {"protocol": PROTOCOL, "scope": "20 fixed validation tasks, development diagnosis, not held-out test",
            "policies": by_policy, "paired_results": paired, "api": call_statistics(calls),
            "api_error_attempts": sum(c["status"] != "ok" for c in calls),
            "format_normalized_responses": sum(c.get("format_normalization") == "markdown_fence" for c in calls),
            "limitations": ["20任务仅4个区域，不是20个独立地理样本", "VLM单次运行；无模型随机重复，不给显著性结论", "规则策略不看图；对照检验覆盖管理而非视觉上界", "无图/错图消融未做，不能证明视觉贡献", "数据与原论文split/episode未对齐，不比较论文数值"],
            "configuration_sha256": sha256(json_bytes(configuration)).hexdigest()}


def execute(root, output, consent=False, transport="httpx"):
    root, output = Path(root), Path(output)
    if not consent:
        raise ValueError("需要 --confirm-cloud-transmission 允许云端发送图像")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("目录非空，禁止静默覆盖或重复付费运行")
    episodes, manifest = verify_task_file(root, root / "任务清单_v2/episodes_val.jsonl", "val")
    selected = select_validation(episodes)
    config = APIConfig.load()
    src = Path(__file__).resolve().parents[1]
    configuration = {"suite": "validation20-v1", "protocol": PROTOCOL,
        "api": config.public(), "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT,
        "selection": "one minimum sha256(validation20-v1|episode_id) per val area per distance4..8",
        "selected_episodes": [asdict(ep) for ep in selected], "task_sha256": manifest["episodes_sha256"],
        "max_action_requests": 400, "max_successful_decisions": 200,
        "retry_policy": "one retry only for transport/408/429/5xx; same input; wait2s; log every attempt",
        "concurrency": 1, "random_policy_seed": 0, "http_transport": transport,
        "frontier": "BFS to nearest unvisited cell, tie order up/right/down/left, backtracking allowed",
        "vlm_unchanged": "same prompt/state/images/model as repaired initial pilot, no conversation history",
        "diagnostics_evaluator_only": {"cycle": "ABAB windows with A!=B; ABA counts reversals but not sustained cycle",
                                      "coverage": "unique cells including start / 25", "near_goal": "true distance1, offline only"},
        "source_sha256": {str(p.relative_to(src)).replace('\\','/'): sha256(p.read_bytes()).hexdigest()
                           for p in [src / x for x in ("agents/vlm.py", "agents/exploration.py", "agents/curl_transport.py", "env/environment.py", "env/episode.py", "eval/validation_suite.py", "eval/diagnostics.py", "eval/evaluate.py", "eval/pilot.py")]} }
    write_immutable_files({output / "运行配置.json": json_bytes(configuration)})
    def append(name, record):
        text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if config.api_key in text or "data:image/" in text:
            raise RuntimeError("敏感字段日志保护触发")
        with (output / name).open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
            stream.flush()
    calls, records = [], []
    if transport == "curl":
        from agents.curl_transport import CurlTransport
        client = VLMPolicy(config, CurlTransport(config.timeout))
    elif transport == "httpx":
        client = VLMPolicy(config)
    else:
        raise ValueError("未知HTTP传输")
    fatal = None
    env = GridWorldEnv(root)
    started = time.perf_counter()
    try:
        for policy_name in POLICIES:
            for ep in selected:
                observation = env.reset(ep)
                positions = [ep.start]
                decisions = []
                error = None
                begin = time.perf_counter()
                policy = (RandomPolicy(seed_for(0, PROTOCOL, ep.episode_id)) if policy_name == "random" else FrontierPolicy())
                if policy_name == "vlm" and fatal:
                    error = fatal
                while not env.done and error is None:
                    action = None
                    if policy_name != "vlm":
                        action = policy.act(observation)
                    else:
                        for attempt in (1, 2):
                            if len(calls) >= 400:
                                error = "global_call_limit"
                                break
                            try:
                                action, call = client.act(observation)
                                failure = None
                            except APITrialError as exc:
                                call = exc.record
                                failure = exc
                            call = dict(call, call_index=len(calls)+1, episode_id=ep.episode_id,
                                        action_step=len(decisions)+1, attempt=attempt,
                                        retry_policy=configuration["retry_policy"])
                            calls.append(call)
                            append("API调用.jsonl", call)
                            print(f"call={len(calls)} ep={ep.episode_id} step={len(decisions)+1} attempt={attempt} status={call['status']} latency={call.get('latency_seconds',0):.2f}s", flush=True)
                            if failure is None:
                                break
                            if call.get("http_status") in (401, 403) or call.get("http_status") == 404:
                                fatal = failure.reason
                            if attempt == 1 and transient(failure):
                                time.sleep(2)
                            else:
                                error = failure.reason
                                break
                    if error is not None:
                        break
                    observation, _, info = env.step(action)
                    positions.append(observation.position[0]*5 + observation.position[1])
                    event = {"episode_id": ep.episode_id, "policy": policy_name,
                             "step": info.step_count, "action": action, "patch_id": positions[-1],
                             "out_of_bounds": info.out_of_bounds, "revisited": info.revisited}
                    decisions.append(event)
                    append("动作事件.jsonl", event)
                record = {"policy": policy_name, "episode_id": ep.episode_id, "split": ep.split,
                          "area": ep.area, "distance": ep.dist, "budget": ep.budget,
                          "completion": "completed" if env.done else "api_error",
                          "error": error, "wall_seconds": time.perf_counter()-begin,
                          "positions": positions, "actions": decisions,
                          "evaluation": env.evaluator_result() if env.done else None,
                          "diagnostics": analyze_positions(positions, ep.goal)}
                records.append(record)
                append("任务结果.jsonl", record)
                print(f"{policy_name} {ep.episode_id} completion={record['completion']} steps={len(decisions)}", flush=True)
    finally:
        client.close()
    summary = summarize(records, calls, configuration)
    summary["total_wall_seconds"] = time.perf_counter()-started
    summary["log_sha256"] = {name:sha256((output/name).read_bytes()).hexdigest()
                             for name in ("API调用.jsonl", "任务结果.jsonl", "动作事件.jsonl") if (output/name).exists()}
    write_immutable_files({output / "验证汇总.json": json_bytes(summary)})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=MASA)
    parser.add_argument("--output-dir", type=Path, default=MASA/"评测结果/三项验证_v1")
    parser.add_argument("--confirm-cloud-transmission", action="store_true")
    parser.add_argument("--transport", choices=["httpx", "curl"], default="httpx")
    args = parser.parse_args()
    result = execute(args.dataset_root, args.output_dir, args.confirm_cloud_transmission, args.transport)
    print(json.dumps({"api":result["api"], "policies":{p:v["all"] for p,v in result["policies"].items()}}, ensure_ascii=False, indent=2))
