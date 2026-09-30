"""A/B/C/D + Random 配对验证套件；不覆盖旧三项验证结果。"""
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
from agents.governor import (ActionGovernor, HierarchicalSearchGovernor,
                              NeighborhoodConfirmationGovernor, SpatialMemoryGovernor)
from agents.vlm import (APIConfig, APITrialError, CONFIRMATION_RANKED_PROMPT_VERSION,
                         CONFIRMATION_RANKED_SYSTEM_PROMPT, HIERARCHICAL_PROMPT_VERSION,
                         HIERARCHICAL_SYSTEM_PROMPT, PROMPT_VERSION,
                         RANKED_PROMPT_VERSION, SYSTEM_PROMPT, RANKED_SYSTEM_PROMPT,
                         VLMPolicy)
from data.make_episodes import MASA, json_bytes, seed_for
from env.environment import GridWorldEnv
from env.episode import PROTOCOL, write_immutable_files
from eval.diagnostics import analyze_positions
from eval.evaluate import RandomPolicy, metrics, verify_task_file
from eval.validation_suite import select_validation, transient

ARM_INFO = {
    "A_legacy-action-vlm": {"kind": "vlm_action", "label": "A legacy单动作VLM", "factor": "legacy action schema"},
    "B_ranked-vlm-top1": {"kind": "vlm_rank", "label": "B ranked top1", "factor": "ranked_actions schema, no governor"},
    "C_ranked-vlm-governed": {"kind": "vlm_governed", "label": "C ranked + Action Governor", "factor": "ranked_actions plus ABAB governor"},
    "E_ranked-vlm-spatial-memory": {"kind": "vlm_spatial_memory", "label": "E ranked + Spatial Memory", "factor": "ranked_actions plus visited-cell memory"},
    "F_ranked-vlm-neighborhood-confirmation": {"kind": "vlm_confirmation", "label": "F ranked + Neighborhood Confirmation", "factor": "E plus high-evidence local confirmation"},
    "G_ranked-vlm-hierarchical-search": {"kind": "vlm_hierarchical", "label": "G ranked + Hierarchical Search", "factor": "E plus ranked coarse-region planning"},
    "D_frontier-rule": {"kind": "frontier", "label": "D Frontier规则", "factor": "no image, frontier coverage rule"},
    "Random_sanity": {"kind": "random", "label": "Random sanity", "factor": "no image, seeded random"},
}
VLM_ARMS = {name for name, info in ARM_INFO.items() if info["kind"].startswith("vlm")}


def _source_hashes(src: Path):
    names = ("agents/vlm.py", "agents/governor.py", "agents/exploration.py", "env/environment.py", "env/episode.py", "eval/four_arm_suite.py", "eval/diagnostics.py", "eval/evaluate.py")
    return {name: sha256((src / name).read_bytes()).hexdigest() for name in names}


def _append(path: Path, record: dict, secret: str):
    text = json.dumps(record, ensure_ascii=False, sort_keys=True)
    if secret and secret in text or "data:image/" in text:
        raise RuntimeError("敏感字段日志保护触发")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()


def _make_config(root, episodes, arm_name, config, src):
    info = ARM_INFO[arm_name]
    return {"suite": "four-arm-validation-v1", "variant": arm_name, "arm_label": info["label"],
            "unique_manipulated_factor": info["factor"], "protocol": PROTOCOL,
            "selected_episodes": [asdict(ep) for ep in episodes],
            "task_sha256": sha256((root / "任务清单_v2/episodes_val.jsonl").read_bytes()).hexdigest(),
            "split": "val", "grid_size": 5, "budget": 10, "actions": ["up", "right", "down", "left"],
            "boundary": "stay_and_consume_one_step", "termination": "first_arrival_success_including_last_budget_step",
            "model": config.public() if config else None,
            "prompt_version": (PROMPT_VERSION if info["kind"] == "vlm_action" else
                                CONFIRMATION_RANKED_PROMPT_VERSION if info["kind"] == "vlm_confirmation" else
                                HIERARCHICAL_PROMPT_VERSION if info["kind"] == "vlm_hierarchical" else
                                RANKED_PROMPT_VERSION if info["kind"] in ("vlm_rank", "vlm_governed", "vlm_spatial_memory") else None),
            "system_prompt": (SYSTEM_PROMPT if info["kind"] == "vlm_action" else
                               CONFIRMATION_RANKED_SYSTEM_PROMPT if info["kind"] == "vlm_confirmation" else
                               HIERARCHICAL_SYSTEM_PROMPT if info["kind"] == "vlm_hierarchical" else
                               RANKED_SYSTEM_PROMPT if info["kind"] in ("vlm_rank", "vlm_governed", "vlm_spatial_memory") else None),
            "observation": "target_image,current_image,position,grid_size,remaining_budget,visited,legal_actions; no goal/distance/unvisited images/history",
            "transport": config.public().get("base_url") if config else None,
            "retry_policy": "one retry only for transport/408/429/5xx; same input; wait2s; per-arm",
            "concurrency": 1, "max_action_requests": 400 if info["kind"].startswith("vlm") else 0,
            "max_successful_decisions": 200 if info["kind"].startswith("vlm") else 0,
            "random_seed": 0,
            "governor": (ActionGovernor.name if info["kind"] == "vlm_governed"
                          else SpatialMemoryGovernor.name if info["kind"] == "vlm_spatial_memory"
                          else NeighborhoodConfirmationGovernor.name if info["kind"] == "vlm_confirmation"
                          else HierarchicalSearchGovernor.name if info["kind"] == "vlm_hierarchical" else None),
            "governor_avoid_boundary": False,
            "spatial_memory": info["kind"] == "vlm_spatial_memory",
            "neighborhood_confirmation": info["kind"] == "vlm_confirmation",
            "hierarchical_search": info["kind"] == "vlm_hierarchical",
            "coarse_region_scheme": "region_id=r(row//2)c(col//2), 3x3 over 5x5" if info["kind"] == "vlm_hierarchical" else None,
            "source_sha256": _source_hashes(src),
            "scope_note": "20 fixed validation tasks; not independent geographic samples and not paper reproduction"}


def _summary(records, calls, config):
    finished = [r for r in records if r["completion"] == "completed"]
    evaluation = [r["evaluation"] for r in finished]
    diagnostics = [r["diagnostics"] for r in finished]
    result = {"scheduled": len(records), "completed": len(finished), "errors": len(records)-len(finished),
              "metrics_completed_only": metrics(evaluation) if evaluation else None,
              "diagnostics": {
                  "mean_unique_cells": statistics.mean(d["unique_cells_including_start"] for d in diagnostics) if diagnostics else None,
                  "mean_coverage_ratio": statistics.mean(d["coverage_ratio"] for d in diagnostics) if diagnostics else None,
                  "repeat_cycle_episodes": sum(d["two_cell_cycle_present"] for d in diagnostics),
                  "cycle_episode_rate": statistics.mean(d["two_cell_cycle_present"] for d in diagnostics) if diagnostics else None,
                  "mean_minimum_distance": statistics.mean(d["minimum_distance"] for d in diagnostics) if diagnostics else None,
                  "adjacent_but_not_reached": sum(d["adjacent_but_not_reached"] for d in diagnostics),
                  "governor_overrides": sum(r.get("governor_overrides", 0) for r in finished),
                  "governor_cycle_detections": sum(r.get("governor_cycle_detections", 0) for r in finished),
                  "spatial_memory_overrides": sum(r.get("spatial_memory_overrides", 0) for r in finished),
                  "spatial_memory_novelty_choices": sum(r.get("spatial_memory_novelty_choices", 0) for r in finished),
                  "confirmation_overrides": sum(r.get("confirmation_overrides", 0) for r in finished),
                  "confirmation_high_decisions": sum(r.get("confirmation_high_decisions", 0) for r in finished),
                  "hierarchical_overrides": sum(r.get("hierarchical_overrides", 0) for r in finished),
                  "hierarchical_target_region_changes": sum(r.get("hierarchical_target_region_changes", 0) for r in finished)},
              "api": {"requests": len(calls), "valid_actions": sum(c.get("status") == "ok" for c in calls),
                      "errors": sum(c.get("status") != "ok" for c in calls),
                      "latency_mean_seconds": statistics.mean([c["latency_seconds"] for c in calls]) if calls else None,
                      "latency_p50_seconds": _percentile([c["latency_seconds"] for c in calls], .5),
                      "latency_p95_seconds": _percentile([c["latency_seconds"] for c in calls], .95),
                      "usage": _usage(calls), "cost": None},
              "policy": config["variant"], "failure_note": "API错误不计作正常导航失败，不自动随机回退"}
    return result


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values)-1)*q
    lo, hi = int(pos), min(int(pos)+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def _usage(calls):
    totals = {}
    for call in calls:
        usage = call.get("usage")
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(key), int):
                    totals[key] = totals.get(key, 0) + usage[key]
    return totals or None


def _step_decision(arm, policy, client, governor, obs, calls, api_path, ep, step, config, secret):
    info = {"ranked_actions": None, "selected_action": None, "governor_override": False,
            "governor_reason": None, "cycle_window_detected": False, "fallback_used": False,
            "memory_override": False, "selected_visit_count": None,
            "selected_destination": None, "spatial_memory_novelty": False,
            "target_evidence": None, "confirmation_mode": False,
            "confirmation_override": False,
            "ranked_regions": None, "target_region": None,
            "destination_region": None, "region_rank": None,
            "hierarchical_mode": False, "hierarchical_override": False,
            "decision_source": "rule" if arm in ("D_frontier-rule", "Random_sanity") else "unknown"}
    if arm == "D_frontier-rule":
        action = policy.act(obs)
        info.update(selected_action=action, decision_source="frontier_rule")
        return action, info
    if arm == "Random_sanity":
        action = policy.act(obs)
        info.update(selected_action=action, decision_source="random")
        return action, info
    for attempt in (1, 2):
        if len(calls) >= 400:
            raise APITrialError("arm_call_budget_exhausted", {"status": "not_sent", "attempt": 0})
        try:
            if arm == "A_legacy-action-vlm":
                action, api_record = client.act(obs)
                ranked = None
                target_evidence = None
            elif arm == "F_ranked-vlm-neighborhood-confirmation":
                ranked, target_evidence, api_record = client.rank_with_confirmation(obs)
                action = ranked[0]
                ranked_regions = None
            elif arm == "G_ranked-vlm-hierarchical-search":
                ranked_regions, ranked, api_record = client.rank_hierarchical(obs)
                action = ranked[0]
                target_evidence = None
            else:
                ranked, api_record = client.rank(obs)
                action = ranked[0]
                target_evidence = None
                ranked_regions = None
            failure = None
        except APITrialError as exc:
            api_record = exc.record
            failure = exc
        call = dict(api_record, arm=arm, call_index=len(calls)+1, episode_id=ep.episode_id,
                    action_step=step, attempt=attempt,
                    retry_policy=config["retry_policy"])
        calls.append(call)
        _append(api_path, call, secret)
        if failure is None:
            info["ranked_actions"] = list(ranked) if ranked is not None else None
            if arm in ("C_ranked-vlm-governed", "E_ranked-vlm-spatial-memory",
                       "F_ranked-vlm-neighborhood-confirmation", "G_ranked-vlm-hierarchical-search"):
                decision = (governor.choose(obs, ranked, target_evidence)
                            if arm == "F_ranked-vlm-neighborhood-confirmation"
                            else governor.choose(obs, ranked, ranked_regions)
                            if arm == "G_ranked-vlm-hierarchical-search"
                            else governor.choose(obs, ranked))
                action = decision.selected_action
                info.update(selected_action=action, decision_source="governor",
                            governor_override=decision.override,
                            governor_reason=decision.reason,
                            cycle_window_detected=decision.cycle_window_detected,
                            fallback_used=decision.fallback_used,
                            filtered_candidates=list(decision.filtered_actions),
                            memory_override=decision.memory_override,
                            selected_visit_count=decision.selected_visit_count,
                            selected_destination=decision.selected_destination,
                            spatial_memory_novelty=decision.reason in
                            ("prefer_unvisited_destination", "top1_unvisited"),
                            target_evidence=decision.target_evidence,
                            confirmation_mode=decision.confirmation_mode,
                            confirmation_override=decision.confirmation_override,
                            ranked_regions=list(decision.ranked_regions) if decision.ranked_regions else None,
                            target_region=decision.target_region,
                            destination_region=decision.destination_region,
                            region_rank=decision.region_rank,
                            hierarchical_mode=decision.hierarchical_mode,
                            hierarchical_override=decision.hierarchical_override)
            else:
                info.update(selected_action=action, decision_source="vlm_top1")
            return action, info
        if attempt == 1 and transient(failure):
            time.sleep(2)
            continue
        raise failure
    raise APITrialError("request_failed", {"status": "failed"})


def run_arm(root: Path, run_root: Path, arm_name: str, transport="curl"):
    arm_dir = run_root / arm_name
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise FileExistsError(f"实验臂目录非空，禁止覆盖: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=False)
    episodes, manifest = verify_task_file(root, root / "任务清单_v2/episodes_val.jsonl", "val")
    episodes = select_validation(episodes)
    info = ARM_INFO[arm_name]
    config_obj = APIConfig.load() if info["kind"].startswith("vlm") else None
    config = _make_config(root, episodes, arm_name, config_obj, Path(__file__).resolve().parents[1])
    write_immutable_files({arm_dir / "运行配置.json": json_bytes(config)})
    for name in ("API调用.jsonl", "动作事件.jsonl", "任务结果.jsonl"):
        (arm_dir / name).touch()
    calls, records = [], []
    client = None
    if info["kind"].startswith("vlm"):
        if transport == "curl":
            from agents.curl_transport import CurlTransport
            client = VLMPolicy(config_obj, CurlTransport(config_obj.timeout))
        else:
            client = VLMPolicy(config_obj)
    env = GridWorldEnv(root)
    try:
        for ep in episodes:
            obs = env.reset(ep)
            policy = (RandomPolicy(seed_for(0, PROTOCOL, ep.episode_id)) if info["kind"] == "random" else
                      FrontierPolicy() if info["kind"] == "frontier" else None)
            governor = (ActionGovernor(avoid_boundary=False) if arm_name == "C_ranked-vlm-governed"
                        else SpatialMemoryGovernor() if arm_name == "E_ranked-vlm-spatial-memory"
                        else NeighborhoodConfirmationGovernor() if arm_name == "F_ranked-vlm-neighborhood-confirmation"
                        else HierarchicalSearchGovernor() if arm_name == "G_ranked-vlm-hierarchical-search"
                        else None)
            if governor:
                governor.reset()
            positions = [ep.start]
            events = []
            error = None
            while not env.done:
                try:
                    action, decision = _step_decision(arm_name, policy, client, governor, obs, calls,
                                                      arm_dir / "API调用.jsonl", ep, len(events)+1, config,
                                                      config_obj.api_key if config_obj else "")
                except APITrialError as exc:
                    error = exc.reason
                    break
                obs, _, step_info = env.step(action)
                patch_id = obs.position[0]*5 + obs.position[1]
                positions.append(patch_id)
                event = {"arm": arm_name, "policy": arm_name, "episode_id": ep.episode_id,
                         "step": step_info.step_count, "action": action, "patch_id": patch_id,
                         "out_of_bounds": step_info.out_of_bounds, "revisited": step_info.revisited,
                         **decision}
                events.append(event)
                _append(arm_dir / "动作事件.jsonl", event, config_obj.api_key if config_obj else "")
            record = {"arm": arm_name, "policy": arm_name, "episode_id": ep.episode_id,
                      "split": ep.split, "area": ep.area, "distance": ep.dist, "budget": ep.budget,
                      "completion": "completed" if env.done else "api_error", "error": error,
                      "positions": positions, "actions": events,
                      "evaluation": env.evaluator_result() if env.done else None,
                      "diagnostics": analyze_positions(positions, ep.goal),
                      "governor_overrides": sum(e.get("governor_override", False) for e in events),
                      "governor_cycle_detections": sum(e.get("cycle_window_detected", False) for e in events),
                      "spatial_memory_overrides": sum(e.get("memory_override", False) for e in events),
                      "spatial_memory_novelty_choices": sum(e.get("spatial_memory_novelty", False) for e in events),
                      "confirmation_overrides": sum(e.get("confirmation_override", False) for e in events),
                       "confirmation_high_decisions": sum(e.get("target_evidence") == "high" for e in events),
                       "hierarchical_overrides": sum(e.get("hierarchical_override", False) for e in events),
                       "hierarchical_target_region_changes": sum(
                           e.get("target_region") != e.get("destination_region") for e in events
                           if e.get("hierarchical_mode"))}
            records.append(record)
            _append(arm_dir / "任务结果.jsonl", record, config_obj.api_key if config_obj else "")
    finally:
        if client:
            client.close()
    summary = {"variant": arm_name, "summary": _summary(records, calls, config),
               "records": records, "task_sha256": config["task_sha256"],
               "source_sha256": config["source_sha256"]}
    write_immutable_files({arm_dir / "验证汇总.json": json_bytes(summary)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=MASA)
    parser.add_argument("--output-root", type=Path, default=MASA / "评测结果/四组实验_v1")
    parser.add_argument("--arms", nargs="+", choices=list(ARM_INFO), default=list(ARM_INFO))
    parser.add_argument("--transport", choices=("curl", "httpx"), default="curl")
    parser.add_argument("--confirm-cloud-transmission", action="store_true")
    args = parser.parse_args()
    if any(ARM_INFO[a]["kind"].startswith("vlm") for a in args.arms) and not args.confirm_cloud_transmission:
        raise ValueError("运行VLM臂需 --confirm-cloud-transmission")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in args.arms:
        results[arm] = run_arm(args.dataset_root, args.output_root, arm, args.transport)
        print(json.dumps({"arm": arm, "summary": results[arm]["summary"]}, ensure_ascii=False), flush=True)
    write_immutable_files({args.output_root / "套件汇总.json": json_bytes({"suite": "four-arm-validation-v1", "arms": list(results), "task_sha256": next(iter(results.values()))["task_sha256"], "note": "详细审计与报告由离线脚本完成"})})


if __name__ == "__main__":
    main()
