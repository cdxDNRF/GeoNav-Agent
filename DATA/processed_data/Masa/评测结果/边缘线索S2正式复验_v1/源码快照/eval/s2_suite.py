"""Frozen val100 S2 suite. No training, test access, or changes to legacy G/E."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.curl_transport import CurlTransport
from agents.exploration import FrontierPolicy
from agents.governor import COARSE_REGIONS, HierarchicalSearchGovernor, SpatialMemoryGovernor
from agents.vlm import (APIConfig, APITrialError, VLMPolicy, HIERARCHICAL_SYSTEM_PROMPT,
                        HIERARCHICAL_PROMPT_VERSION, RANKED_SYSTEM_PROMPT, RANKED_PROMPT_VERSION,
                        hierarchical_observation_request, ranked_observation_request,
                        parse_hierarchical_ranked, parse_ranked_actions)
from data.make_episodes import MASA, json_bytes, seed_for
from env.environment import GridWorldEnv
from env.episode import PROTOCOL, write_immutable_files
from eval.evaluate import RandomPolicy, verify_task_file
from eval.four_arm_suite import _append, _percentile, _usage
from eval.s2_metrics import VLM_ARMS, RULE_ARMS, summarize, acceptance
from eval.validation_suite import transient

SRC = Path(__file__).resolve().parents[1]
WORKSPACE = SRC.parents[1]
DEFAULT_OUTPUT = MASA / "评测结果/S2稳定验证_v1"
ACTIONS = ("up", "right", "down", "left")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()] if Path(path).exists() else []


def save_live(path, data):
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(json_bytes(data))
    os.replace(temp, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gray_image():
    buf = BytesIO()
    Image.new("RGB", (300, 300), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def agent_observation(arm, obs):
    # Never send original target pixels in the target-occlusion arm.
    return replace(obs, target_image=gray_image()) if arm == "G_gray_target" else obs


def decision_for(arm, obs, ranked, regions=None):
    if arm in ("G", "G_gray_target", "FixedRegion"):
        return HierarchicalSearchGovernor().choose(obs, ranked, regions)
    if arm in ("E", "G_no_region"):
        return SpatialMemoryGovernor().choose(obs, ranked)
    raise ValueError("no governor for this arm")


def planned_jobs():
    offline = [{"arm": a, "round": 0} for a in RULE_ARMS]
    offline += [{"arm": "Random", "round": r} for r in range(3)]
    # Rotate arm submission order before observing results, with two serial clients.
    online = []
    for r in range(3):
        order = VLM_ARMS[r:] + VLM_ARMS[:r]
        online += [{"arm": a, "round": r} for a in order]
    return [dict(j, job_id=f"{j['arm']}_r{j['round']}") for j in offline + online]


def register(root, output, api, workers=2):
    episodes, manifest = verify_task_file(root, root / "任务清单_v2/episodes_val.jsonl", "val")
    episodes = sorted(episodes, key=lambda e: e.episode_id)
    require(len(episodes) == 100 and len({e.area for e in episodes}) == 4, "S2 requires the fixed val100 / four areas")
    require(all(e.grid_size == 5 and e.budget == 10 for e in episodes), "S2 grid/budget mismatch")
    files = {p.relative_to(SRC).as_posix(): p for p in SRC.rglob("*.py")
             if p.relative_to(SRC).parts[0] in ("agents", "env", "data", "eval", "tests")}
    hashes = {name: digest(p) for name, p in sorted(files.items())}
    task_path = root / "任务清单_v2/episodes_val.jsonl"
    standard = WORKSPACE / "选题报告相关/分阶段验收标准_v1.md"
    frozen = {"version": "s2-val100-v1", "protocol": PROTOCOL,
              "model": api.public(), "workers": workers, "task_sha256": digest(task_path),
              "task_manifest_sha256": digest(task_path.with_name("manifest_val.json")),
              "area_sha256": manifest["area_sha256"],
              "source_sha256": hashes, "standard_sha256": digest(standard),
              "episodes": [asdict(e) for e in episodes], "jobs": planned_jobs(),
              "unique_routes": len({(e.area, e.start, e.goal) for e in episodes}),
              "grid": 5, "budget": 10, "repeats": 3, "random_seeds": [0, 1, 2],
              "gray_target": {"rgb": [128, 128, 128], "size": [300, 300], "sha256": sha256(gray_image()).hexdigest()},
              "prompts": {"E": RANKED_SYSTEM_PROMPT, "G_family": HIERARCHICAL_SYSTEM_PROMPT},
              "prompt_versions": {"E": RANKED_PROMPT_VERSION, "G_family": HIERARCHICAL_PROMPT_VERSION},
              "fixed_rule": {"ranked_regions": list(COARSE_REGIONS), "ranked_actions": list(ACTIONS)},
              "max_requests_per_online_job": 2000, "max_total_cloud_requests": 24000,
              "retry": "only one retry for transport/408/429/5xx, same observation, wait 2s",
              "failure_circuit": "401/403/429 after allowed retry, or three consecutive failed episodes: stop cloud batch; preserve all pending tasks as not_run",
              "recovery": "never repeat a started episode: replay recorded events and mark interrupted; resume only untouched tasks",
              "rounds_note": "temperature=0; three fresh online calls per task, not independent geographic samples or guaranteed independent model seeds",
              "rules_note": "deterministic rules once, fully replayed; Random seeds 0,1,2; test never accessed",
              "logging_schema": "s2-v1: actual cross-step target-region switches and G-vs-E counterfactual one-step action disagreement"}
    path = output / "预登记.json"
    if path.exists():
        existing = read_json(path)
        require(existing["frozen"] == frozen, "Frozen settings/source changed; refuse to resume this batch")
        for name, value in hashes.items():
            require(digest(output / "源码快照" / name) == value, "source snapshot mismatch")
        return episodes, existing
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "new registration needs an empty output directory")
    payloads = {output / "源码快照" / name: p.read_bytes() for name, p in files.items()}
    payloads.update({output / "固定任务.jsonl": task_path.read_bytes(),
                     output / "任务清单.json": task_path.with_name("manifest_val.json").read_bytes(),
                     output / "验收标准冻结.md": standard.read_bytes(),
                     output / "中性目标.png": gray_image()})
    registration = {"registered_utc": now(), "frozen": frozen}
    payloads[path] = json_bytes(registration)
    write_immutable_files(payloads)
    return episodes, registration


def invoke(client, arm, obs, ep, step, directory, calls, attempts, secret, stop):
    visible = agent_observation(arm, obs)
    for attempt in (1, 2):
        require(len(attempts) < 2000, "per-job request cap")
        intent = {"episode_id": ep.episode_id, "step": step, "attempt": attempt,
                  "request_index": len(attempts) + 1, "utc": now()}
        _append(directory / "请求意图.jsonl", intent, secret)
        attempts.append(intent)
        failure = None
        try:
            if arm == "E":
                ranked, record = client.rank(visible)
                regions = None
            else:
                regions, ranked, record = client.rank_hierarchical(visible)
        except APITrialError as exc:
            failure, record = exc, exc.record
        call = dict(record, **intent, arm=arm)
        calls.append(call)
        _append(directory / "API调用.jsonl", call, secret)
        if failure is None:
            return ranked, regions
        if attempt == 1 and transient(failure):
            time.sleep(2)
            continue
        if record.get("http_status") in (401, 403, 429):
            stop.set()
        raise failure


def replay_record(env, ep, events):
    obs = env.reset(ep)
    positions = [ep.start]
    for event in events:
        obs, _, info = env.step(event["action"])
        positions.append(obs.position[0] * ep.grid_size + obs.position[1])
        require(event["patch_id"] == positions[-1], "recovery trajectory mismatch")
    return obs, positions


def run_job(root, output, episodes, registration, job, api, stop):
    arm, round_id = job["arm"], job["round"]
    directory = output / job["job_id"]
    directory.mkdir(exist_ok=True)
    write_immutable_files({directory / "运行配置.json": json_bytes(dict(job, registration_sha256=digest(output / "预登记.json")))})
    records = read_lines(directory / "任务结果.jsonl")
    calls = read_lines(directory / "API调用.jsonl")
    attempts = read_lines(directory / "请求意图.jsonl")
    old_events = read_lines(directory / "动作事件.jsonl")
    require(len({r["episode_id"] for r in records}) == len(records), "duplicate terminal record")
    known = {e.episode_id for e in episodes}
    require(all(r["episode_id"] in known for r in records), "unexpected terminal record")
    finished = {r["episode_id"] for r in records}
    if (directory / "汇总.json").exists():
        audit_job(root, output, episodes, job, api)
        return read_json(directory / "汇总.json")
    client = VLMPolicy(api, CurlTransport(api.timeout)) if arm in VLM_ARMS else None
    secret = api.api_key if client else ""
    env = GridWorldEnv(root)
    failures = 0
    for ep in episodes:
        if ep.episode_id in finished:
            continue
        begin = time.monotonic()
        events = [a for a in old_events if a["episode_id"] == ep.episode_id]
        interrupted = any(a["episode_id"] == ep.episode_id for a in attempts) or bool(events)
        obs, positions = replay_record(env, ep, events)
        completion, error = None, None
        if interrupted:
            completion = "completed" if env.done else "process_interrupted"
            error = None if env.done else "started episode preserved without fresh requests"
        elif arm in VLM_ARMS and (stop.is_set() or (output / "STOP").exists()):
            completion, error = "not_run", "batch_stopped"
        policy = RandomPolicy(seed_for(round_id, PROTOCOL, ep.episode_id)) if arm == "Random" else FrontierPolicy()
        last_region = None
        while completion is None and not env.done:
            if client and (stop.is_set() or (output / "STOP").exists()):
                completion, error = "batch_stopped", "batch_stopped_during_episode"
                break
            decision, ranked, regions = None, None, None
            try:
                if client:
                    ranked, regions = invoke(client, arm, obs, ep, len(events) + 1, directory, calls, attempts, secret, stop)
                    decision = decision_for(arm, obs, ranked, regions)
                    action = decision.selected_action
                elif arm == "FixedRegion":
                    ranked, regions = ACTIONS, COARSE_REGIONS
                    decision = decision_for(arm, obs, ranked, regions)
                    action = decision.selected_action
                else:
                    action = policy.act(obs)
            except APITrialError as exc:
                completion, error = "api_error", exc.reason
                break
            comparison = SpatialMemoryGovernor().choose(obs, ranked).selected_action if ranked else None
            target_region = decision.target_region if decision else None
            switch = target_region is not None and last_region is not None and target_region != last_region
            last_region = target_region
            obs, _, info = env.step(action)
            positions.append(obs.position[0] * ep.grid_size + obs.position[1])
            event = {"episode_id": ep.episode_id, "arm": arm, "step": info.step_count,
                     "action": action, "patch_id": positions[-1],
                     "revisited": info.revisited, "out_of_bounds": info.out_of_bounds,
                     "ranked_actions": list(ranked) if ranked else None,
                     "ranked_regions": list(regions) if regions else None,
                     "governor": asdict(decision) if decision else None,
                     "target_region_switch": switch,
                     "region_vs_spatial_action_changed": bool(decision and decision.hierarchical_mode and comparison != action)}
            _append(directory / "动作事件.jsonl", event, secret)
            events.append(event)
        completion = completion or "completed"
        record = {"episode_id": ep.episode_id, "arm": arm, "round": round_id,
                  "area": ep.area, "distance": ep.dist, "completion": completion, "error": error,
                  "positions": positions, "actions": events,
                  "evaluation": env.evaluator_result() if completion == "completed" else None,
                  "elapsed_seconds": time.monotonic() - begin, "finished_utc": now()}
        _append(directory / "任务结果.jsonl", record, secret)
        records.append(record)
        failures = failures + 1 if completion == "api_error" else 0
        if failures >= 3:
            stop.set()
        progress = {"job_id": job["job_id"], "updated_utc": now(),
                    "metrics": summarize(episodes, records), "api_attempts": len(attempts)}
        save_live(directory / "进度.json", progress)
        print(json.dumps({"job": job["job_id"], "recorded": len(records),
                          "completed": progress["metrics"]["completed"],
                          "successes": progress["metrics"]["successes"], "last_status": completion}, ensure_ascii=False), flush=True)
    if client:
        client.client.close()
    audit_job(root, output, episodes, job, api)
    latencies = [c["latency_seconds"] for c in calls if "latency_seconds" in c]
    result = {**job, "audit_passed": True, "metrics": summarize(episodes, records),
              "api": {"attempts": len(attempts), "responses": len(calls),
                        "errors": sum(c.get("status") != "ok" for c in calls),
                        "usage": _usage(calls), "usage_missing": sum(not isinstance(c.get("usage"), dict) for c in calls),
                        "latency_p50": _percentile(latencies, .5), "latency_p95": _percentile(latencies, .95)},
              "episode_latency_p50": _percentile([r["elapsed_seconds"] for r in records], .5),
              "episode_latency_p95": _percentile([r["elapsed_seconds"] for r in records], .95)}
    write_immutable_files({directory / "汇总.json": json_bytes(result)})
    return result


def audit_job(root, output, episodes, job, api):
    directory, arm = output / job["job_id"], job["arm"]
    records, events, calls, intents = [read_lines(directory / name) for name in
        ("任务结果.jsonl", "动作事件.jsonl", "API调用.jsonl", "请求意图.jsonl")]
    expected = {e.episode_id: e for e in episodes}
    require(len(records) == len(expected) and {r["episode_id"] for r in records} == set(expected), "terminal record coverage")
    require(events == [a for r in records for a in r["actions"]], "action log mismatch")
    require(len(intents) <= 2000 and len(calls) <= len(intents), "request count mismatch")
    require([i["request_index"] for i in intents] == list(range(1, len(intents)+1)), "request ledger sequence")
    request_map = {i["request_index"]: i for i in intents}
    require(len({c["request_index"] for c in calls}) == len(calls), "duplicate responses")
    ledger_keys = [(i["episode_id"], i["step"], i["attempt"]) for i in intents]
    require(len(ledger_keys) == len(set(ledger_keys)), "duplicate request attempt")
    require(all(i["episode_id"] in expected and 1 <= i["step"] <= expected[i["episode_id"]].budget and i["attempt"] in (1, 2) for i in intents), "request outside registered task budget")
    response_by_attempt = {(c["episode_id"], c["step"], c["attempt"]): c for c in calls}
    for intent in intents:
        if intent["attempt"] == 2:
            previous = response_by_attempt.get((intent["episode_id"], intent["step"], 1))
            require(previous is not None and previous["status"] != "ok" and transient(APITrialError(previous["status"], previous)), "retry without a transient previous failure")
    successes = {}
    for c in calls:
        require(all(c[k] == request_map[c["request_index"]][k] for k in ("episode_id", "step", "attempt")), "request response mismatch")
        require(c["attempt"] in (1, 2), "retry count")
        if c["status"] == "ok":
            key = (c["episode_id"], c["step"])
            require(key not in successes, "duplicate successful decision")
            successes[key] = c
    env = GridWorldEnv(root)
    for record in records:
        ep = expected[record["episode_id"]]
        require(record["arm"] == arm and record["round"] == job["round"], "arm/round mismatch")
        obs = env.reset(ep)
        positions, last_region = [ep.start], None
        rule = RandomPolicy(seed_for(job["round"], PROTOCOL, ep.episode_id)) if arm == "Random" else FrontierPolicy()
        for event in record["actions"]:
            ranked, regions, decision = None, None, None
            if arm in VLM_ARMS:
                call = successes[(ep.episode_id, event["step"])]
                visible = agent_observation(arm, obs)
                _, audit = ranked_observation_request(visible, api) if arm == "E" else hierarchical_observation_request(visible, api)
                require(all(call[k] == v for k, v in audit.items()), "public input or image audit mismatch")
                if arm == "E":
                    ranked, _ = parse_ranked_actions(call["raw_content"], obs.legal_actions, call["finish_reason"])
                else:
                    regions, ranked, _ = parse_hierarchical_ranked(call["raw_content"], obs.legal_actions, COARSE_REGIONS, call["finish_reason"])
                require(list(ranked) == call["ranked_actions"], "parser mismatch")
                if regions:
                    require(list(regions) == call["ranked_regions"], "region parser mismatch")
                decision = decision_for(arm, obs, ranked, regions)
                action = decision.selected_action
            elif arm == "FixedRegion":
                ranked, regions = ACTIONS, COARSE_REGIONS
                decision = decision_for(arm, obs, ranked, regions)
                action = decision.selected_action
            else:
                action = rule.act(obs)
            require(action == event["action"], "policy replay mismatch")
            if decision:
                require(json.loads(json.dumps(asdict(decision))) == event["governor"], "governor detail mismatch")
                require(event["ranked_actions"] == list(ranked) and event["ranked_regions"] == (list(regions) if regions else None), "ranking mismatch")
                switched = decision.target_region is not None and last_region is not None and decision.target_region != last_region
                require(event["target_region_switch"] == switched, "region switch mismatch")
                last_region = decision.target_region
                comparison = SpatialMemoryGovernor().choose(obs, ranked).selected_action
                require(event["region_vs_spatial_action_changed"] == bool(decision.hierarchical_mode and action != comparison), "region action contribution mismatch")
            obs, _, info = env.step(action)
            positions.append(obs.position[0] * ep.grid_size + obs.position[1])
            require((event["step"], event["patch_id"], event["revisited"], event["out_of_bounds"]) == (info.step_count, positions[-1], info.revisited, info.out_of_bounds), "environment replay mismatch")
        require(record["positions"] == positions, "positions mismatch")
        if record["completion"] == "completed":
            require(env.done and record["evaluation"] == env.evaluator_result(), "terminal result mismatch")
            if arm in VLM_ARMS:
                require({s for eid, s in successes if eid == ep.episode_id} == {a["step"] for a in record["actions"]}, "unused model decisions in completed episode")
        else:
            require(not env.done and record["evaluation"] is None, "failure must not be a fabricated navigation result")
    for name in ("任务结果.jsonl", "动作事件.jsonl", "API调用.jsonl", "请求意图.jsonl"):
        path = directory / name
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            require(api.api_key not in raw and "data:image/" not in raw, "secret/image log leak")
    result = {"passed": True, "episodes": len(records), "trajectory_replay": True,
              "input_hash_and_parser_replay": arm in VLM_ARMS,
              "deterministic_rule_reproduction": arm in (*RULE_ARMS, "Random"),
              "record_sha256": digest(directory / "任务结果.jsonl"),
              "api_requests_made_by_audit": 0}
    write_immutable_files({directory / "离线审计.json": json_bytes(result)})
    return result


def report(output):
    registration = read_json(output / "预登记.json")
    jobs, progress = {}, {}
    for job in registration["frozen"]["jobs"]:
        directory = output / job["job_id"]
        if (directory / "汇总.json").exists():
            jobs[job["job_id"]] = read_json(directory / "汇总.json")
        elif (directory / "进度.json").exists():
            progress[job["job_id"]] = read_json(directory / "进度.json")
    gate = acceptance(jobs)
    result = {"updated_utc": now(), "complete_jobs": len(jobs), "planned_jobs": 17,
              "gate": gate, "jobs": jobs, "in_progress": progress}
    save_live(output / "S2汇总.json", result)
    lines = ["# S2 固定验证报告", "", f"完成批次：{len(jobs)}/17。验收：{gate['status']}。",
             "", "计划：val100，四张源图，86条不同路线；三轮不等于300张独立地图。",
             "SR/SG 使用计划题分母并按 C 档宏平均；未运行/中断的 SG 按8补入。",
             "未全部完成及审计前，不作 S2 通过结论。四张源图只支持开发诊断。", "",
             "| 臂/轮次 | 正常完成/计划 | 成功数 | SR_gate | SG_gate | 重访率 | API尝试 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, job in jobs.items():
        m = job["metrics"]
        repeat = f"{m['repeat_rate']:.1%}" if m["repeat_rate"] is not None else "缺失"
        lines.append(f"| {key} | {m['completed']}/{m['planned']} | {m['successes']} | {m['sr_macro']:.1%} | {m['sg_macro']:.2f} | {repeat} | {job['api']['attempts']} |")
    lines += ["", "## 分距离和分源图", "", "| 臂/轮次 | 分组 | 完成/计划 | SR_gate | SG_gate |", "|---|---|---:|---:|---:|"]
    for key, job in jobs.items():
        for field in ("by_distance", "by_area"):
            for name, m in job["metrics"][field].items():
                lines.append(f"| {key} | {field}:{name} | {m['completed']}/{m['planned']} | {m['sr_gate']:.1%} | {m['sg_gate']:.2f} |")
    if gate.get("checks"):
        lines += ["", "## 验收门槛", ""] + [f"- {'通过' if v else '未通过'}：{k}" for k, v in gate["checks"].items()]
    path = output / "S2验证报告.md"
    temp = path.with_suffix(".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return result


def verify_frozen(root, output, registration):
    frozen = registration["frozen"]
    for name, expected in frozen["source_sha256"].items():
        require(digest(SRC / name) == expected and digest(output / "源码快照" / name) == expected, "source changed during S2: " + name)
    require(digest(root / "任务清单_v2/episodes_val.jsonl") == frozen["task_sha256"], "task file changed during S2")
    verify_task_file(root, root / "任务清单_v2/episodes_val.jsonl", "val")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=MASA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--confirm-cloud-transmission", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        print(json.dumps(report(args.output_root)["gate"], ensure_ascii=False))
        return
    if not args.prepare_only and not args.offline_only:
        require(args.confirm_cloud_transmission, "cloud transmission confirmation flag required")
    api = APIConfig.load()
    episodes, registration = register(args.dataset_root, args.output_root, api, args.workers)
    if args.prepare_only:
        print(json.dumps({"registered": str(args.output_root), "jobs": 17, "cloud_task_runs": 1200, "maximum_requests": 24000}, ensure_ascii=False))
        return
    lock = args.output_root / "运行锁.json"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_utc": now()}, stream)
    stop = threading.Event()
    try:
        for job in planned_jobs():
            if job["arm"] not in VLM_ARMS:
                run_job(args.dataset_root, args.output_root, episodes, registration, job, api, stop)
        report(args.output_root)
        if not args.offline_only:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_job, args.dataset_root, args.output_root, episodes, registration, j, api, stop)
                           for j in planned_jobs() if j["arm"] in VLM_ARMS]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        stop.set()
                        save_live(args.output_root / "执行异常.json", {"type": type(exc).__name__, "utc": now(), "details": str(exc).replace(api.api_key, "[REDACTED]")})
                        raise
                    report(args.output_root)
        verify_frozen(args.dataset_root, args.output_root, registration)
        result = report(args.output_root)
        save_live(args.output_root / "执行状态.json", {"status": "finished", "pid": os.getpid(), "gate": result["gate"], "utc": now()})
        print(json.dumps(result["gate"], ensure_ascii=False), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
