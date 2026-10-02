"""有界云端试跑；Random与VLM共享任务、终止规则和结果字段。"""
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
from agents.vlm import APIConfig, APITrialError, PROMPT_VERSION, SYSTEM_PROMPT, VLMPolicy
from data.make_episodes import MASA, json_bytes, seed_for
from env.environment import GridWorldEnv
from env.episode import PROTOCOL, write_immutable_files
from eval.evaluate import RandomPolicy, metrics, verify_task_file


def select_pilot(episodes):
    # 固定两个距离档、确定选题，不基于模型表现或目标图内容挑题。
    return [min((ep for ep in episodes if ep.dist == d),
                key=lambda ep: sha256(ep.episode_id.encode()).hexdigest()) for d in (4, 6)]


def run_policy(env, ep, policy, next_action, on_call=None):
    observation = env.reset(ep)
    start = time.perf_counter()
    actions = []
    error = None
    try:
        while not env.done:
            action, call = next_action(observation)
            if on_call is not None:
                on_call(call, ep.episode_id)
            observation, done, info = env.step(action)
            actions.append({"step": info.step_count, "action": action,
                            "position": observation.position,
                            "out_of_bounds": info.out_of_bounds, "revisited": info.revisited})
    except APITrialError as exc:
        error = exc.reason
        if on_call is not None:
            on_call(exc.record, ep.episode_id)
    record = {"policy": policy, "episode_id": ep.episode_id, "split": ep.split,
              "distance": ep.dist, "budget": ep.budget,
              "wall_seconds": time.perf_counter() - start, "actions": actions,
              "completion": "completed" if env.done else "api_error",
              "error": error, "evaluation": env.evaluator_result() if env.done else None}
    return record


def aggregate(records):
    completed = [r["evaluation"] for r in records if r["completion"] == "completed"]
    return {"scheduled": len(records), "completed": len(completed),
            "api_errors": sum(r["completion"] == "api_error" for r in records),
            "not_run": sum(r["completion"] == "not_run" for r in records),
            "metrics_completed_only": metrics(completed) if completed else None,
            "note": "错误/未运行不伪装为正常任务；completed-only指标不得掩盖覆盖率"}


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def call_statistics(calls):
    latencies = [r["latency_seconds"] for r in calls if "latency_seconds" in r]
    sums = {}
    usage_reported = 0
    for call in calls:
        usage = call.get("usage")
        if isinstance(usage, dict):
            usage_reported += 1
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(name), int):
                    sums[name] = sums.get(name, 0) + usage[name]
    return {"requests": len(calls), "valid_actions": sum(c.get("status") == "ok" for c in calls),
            "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
            "latency_p50_seconds": percentile(latencies, .5),
            "latency_p95_seconds": percentile(latencies, .95),
            "usage_reported_requests": usage_reported, "reported_token_totals": sums or None,
            "usage_note": "仅供应商实际返回值；不把缺失用量当0，不推断图像/推理token明细",
            "time_to_first_token": None, "stream": False,
            "cost": None, "cost_note": "未配置或核验价格，不估算费用；以供应商账单为准"}


def execute(root: Path, output: Path, consent: bool, max_calls: int = 20):
    if not consent:
        raise ValueError("云端试跑需明确允许发送图像；使用 --confirm-cloud-transmission")
    if max_calls < 1 or max_calls > 20:
        raise ValueError("首次试跑限制1至20次动作请求")
    episodes, manifest = verify_task_file(root, root / "任务清单_v2/episodes_val.jsonl", "val")
    chosen = select_pilot(episodes)
    config = APIConfig.load()
    config_public = config.public()
    files = [output / n for n in ("运行配置.json", "API调用.jsonl", "任务结果.jsonl", "试跑汇总.json", "模型探测.json")]
    if any(p.exists() for p in files):
        raise FileExistsError("试跑目录已有结果；禁止覆盖或自动重复付费调用")
    output.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parents[1]
    run_config = {"protocol": PROTOCOL, "api": config_public,
                  "task_sha256": manifest["episodes_sha256"],
                  "selected_episodes": [asdict(ep) for ep in chosen],
                  "selection": "minimum_sha256_episode_id_at_val_dist_4_and_6",
                  "max_action_requests": max_calls, "automatic_retries": 0,
                  "split": "val", "context": "two_images_and_current_state_with_visited_ids_no_conversation_history",
                  "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT,
                  "pricing": None,
                  "source_sha256": {name: sha256((src / name).read_bytes()).hexdigest() for name in
                    ("agents/vlm.py", "eval/pilot.py", "eval/evaluate.py", "env/environment.py", "env/episode.py")}}
    write_immutable_files({output / "运行配置.json": json_bytes(run_config)})
    def log(path, record):
        text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if config.api_key in text or "data:image/" in text:
            raise RuntimeError("敏感字段日志保护触发")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
            stream.flush()
    records, calls = [], []
    for ep in chosen:
        policy = RandomPolicy(seed_for(0, PROTOCOL, ep.episode_id))
        record = run_policy(GridWorldEnv(root), ep, "random",
                            lambda obs: (policy.act(obs), None))
        records.append(record)
        log(output / "任务结果.jsonl", record)
    policy = VLMPolicy(config)
    halted = False
    probe_error = None
    try:
        try:
            probe = policy.probe()
            write_immutable_files({output / "模型探测.json": json_bytes(probe)})
        except APITrialError as exc:
            probe_error = exc.reason
            halted = True
            write_immutable_files({output / "模型探测.json": json_bytes(exc.record)})
        def on_call(call, episode_id):
            if call is None:
                return
            item = dict(call, episode_id=episode_id, call_index=len(calls) + 1)
            calls.append(item)
            log(output / "API调用.jsonl", item)
            print(f"call={len(calls)} status={call['status']} latency={call.get('latency_seconds', 0):.2f}s", flush=True)
        def next_action(obs):
            if len(calls) >= max_calls:
                raise APITrialError("call_budget_exhausted", {"status": "not_sent", "attempt": 0})
            return policy.act(obs)
        for ep in chosen:
            if halted:
                record = {"policy": "vlm", "episode_id": ep.episode_id, "split": ep.split,
                          "distance": ep.dist, "budget": ep.budget,
                          "completion": "not_run", "error": "halted_after_error",
                          "evaluation": None, "actions": []}
            else:
                record = run_policy(GridWorldEnv(root), ep, "vlm", next_action, on_call)
                halted = record["completion"] != "completed"
            records.append(record)
            log(output / "任务结果.jsonl", record)
    finally:
        policy.close()
    summary = {"scope": "2 validation episodes compatibility/latency pilot, not model-performance evidence",
               "probe_error": probe_error,
               "random": aggregate([r for r in records if r["policy"] == "random"]),
               "vlm": aggregate([r for r in records if r["policy"] == "vlm"]),
               "api": call_statistics([c for c in calls if c.get("attempt", 1) > 0]),
               "test_split_used": False, "automatic_retries": 0}
    write_immutable_files({output / "试跑汇总.json": json_bytes(summary)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=MASA)
    parser.add_argument("--output-dir", type=Path, default=MASA / "评测结果/Ollama试跑_v1")
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--confirm-cloud-transmission", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.dataset_root, args.output_dir,
                             args.confirm_cloud_transmission, args.max_calls), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
