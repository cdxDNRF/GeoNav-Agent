"""离线审计单个四臂结果目录；不调用API。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.governor import (ActionGovernor, HierarchicalSearchGovernor,
                              NeighborhoodConfirmationGovernor, SpatialMemoryGovernor)
from agents.vlm import APIConfig
from data.make_episodes import MASA
from env.environment import GridWorldEnv
from env.episode import Episode, write_immutable_files
from eval.diagnostics import analyze_positions
from eval.evaluate import verify_task_file
from eval.validation_suite import select_validation


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def audit(root: Path, arm_dir: Path):
    config=json.loads((arm_dir/"运行配置.json").read_text(encoding="utf-8"))
    summary=json.loads((arm_dir/"验证汇总.json").read_text(encoding="utf-8"))
    episodes,_=verify_task_file(root,root/"任务清单_v2/episodes_val.jsonl","val")
    expected={e.episode_id:e for e in select_validation(episodes)}
    selected={e["episode_id"]:Episode.from_dict(e) for e in config["selected_episodes"]}
    assert set(expected)==set(selected) and config["task_sha256"]==sha256((root/"任务清单_v2/episodes_val.jsonl").read_bytes()).hexdigest()
    records=lines(arm_dir/"任务结果.jsonl")
    events=lines(arm_dir/"动作事件.jsonl")
    calls=lines(arm_dir/"API调用.jsonl") if (arm_dir/"API调用.jsonl").exists() else []
    assert len(records)==20 and len({r["episode_id"] for r in records})==20
    assert all(r["arm"]==config["variant"] for r in records)
    src=Path(__file__).resolve().parents[1]
    for name,digest in config["source_sha256"].items():
        assert sha256((src/name).read_bytes()).hexdigest()==digest,name
    if config["variant"] in ("A_legacy-action-vlm", "B_ranked-vlm-top1",
                               "C_ranked-vlm-governed", "E_ranked-vlm-spatial-memory",
                               "F_ranked-vlm-neighborhood-confirmation",
                               "G_ranked-vlm-hierarchical-search"):
        assert len(calls)>0
        assert all("data:image/" not in json.dumps(c) for c in calls)
        assert all("action" in c or "ranked_actions" in c for c in calls if c["status"]=="ok")
    env=GridWorldEnv(root)
    calls_by_step={(c["episode_id"],c["action_step"]):c for c in calls if c.get("status")=="ok"}
    for record in records:
        ep=selected[record["episode_id"]]
        obs=env.reset(ep); positions=[ep.start]
        governor = (ActionGovernor() if config["variant"] == "C_ranked-vlm-governed"
                    else SpatialMemoryGovernor() if config["variant"] == "E_ranked-vlm-spatial-memory"
                    else NeighborhoodConfirmationGovernor() if config["variant"] == "F_ranked-vlm-neighborhood-confirmation"
                    else HierarchicalSearchGovernor() if config["variant"] == "G_ranked-vlm-hierarchical-search"
                    else None)
        if governor: governor.reset()
        for event in record["actions"]:
            step=event["step"]
            if config["variant"] in ("A_legacy-action-vlm", "B_ranked-vlm-top1",
                                       "C_ranked-vlm-governed", "E_ranked-vlm-spatial-memory",
                                       "F_ranked-vlm-neighborhood-confirmation",
                                       "G_ranked-vlm-hierarchical-search"):
                call=calls_by_step[(ep.episode_id,step)]
                assert call["public_state"]["position_row_col"]==list(obs.position)
                assert call["current_image_sha256"]==sha256(obs.current_image).hexdigest()
                assert call["target_image_sha256"]==sha256(obs.target_image).hexdigest()
                if config["variant"]=="A_legacy-action-vlm":
                    assert call["action"]==event["action"]
                else:
                    ranked=tuple(call["ranked_actions"])
                    selected_action=ranked[0]
                    if governor:
                        decision = (governor.choose(obs, ranked, call["target_evidence"])
                                    if config["variant"] == "F_ranked-vlm-neighborhood-confirmation"
                                    else governor.choose(obs, ranked, call["ranked_regions"])
                                    if config["variant"] == "G_ranked-vlm-hierarchical-search"
                                    else governor.choose(obs, ranked))
                        selected_action=decision.selected_action
                        assert event["governor_override"]==decision.override
                        assert event["governor_reason"]==decision.reason
                        if config["variant"] == "E_ranked-vlm-spatial-memory":
                            assert event["memory_override"] == decision.memory_override
                            assert event["selected_visit_count"] == decision.selected_visit_count
                            assert event["selected_destination"] == decision.selected_destination
                            assert event["spatial_memory_novelty"] == (decision.reason in
                                ("prefer_unvisited_destination", "top1_unvisited"))
                        if config["variant"] == "F_ranked-vlm-neighborhood-confirmation":
                            assert event["target_evidence"] == call["target_evidence"]
                            assert event["confirmation_mode"] is True
                            assert event["confirmation_override"] == decision.confirmation_override
                        if config["variant"] == "G_ranked-vlm-hierarchical-search":
                            assert event["ranked_regions"] == call["ranked_regions"]
                            assert event["target_region"] == decision.target_region
                            assert event["destination_region"] == decision.destination_region
                            assert event["hierarchical_override"] == decision.hierarchical_override
                    assert selected_action==event["action"]
            obs,_,info=env.step(event["action"])
            positions.append(obs.position[0]*5+obs.position[1])
            assert event["patch_id"]==positions[-1]
        assert positions==record["positions"]
        assert record["diagnostics"]==analyze_positions(positions,ep.goal)
        if record["completion"]=="completed":
            assert record["evaluation"]==env.evaluator_result()
    audit={"status":"passed","variant":config["variant"],"task_count":20,"api_attempts":len(calls),
           "trajectory_count":len(records),"source_hashes_checked":len(config["source_sha256"]),
           "task_hash_checked":True,"trajectory_replay":True,"public_input_and_image_hashes_checked":bool(calls),
           "governor_replay":config["variant"] in ("C_ranked-vlm-governed", "E_ranked-vlm-spatial-memory", "F_ranked-vlm-neighborhood-confirmation", "G_ranked-vlm-hierarchical-search"),
           "spatial_memory_replay":config["variant"] in ("E_ranked-vlm-spatial-memory", "F_ranked-vlm-neighborhood-confirmation", "G_ranked-vlm-hierarchical-search"),
           "confirmation_replay":config["variant"] == "F_ranked-vlm-neighborhood-confirmation",
           "hierarchical_replay":config["variant"] == "G_ranked-vlm-hierarchical-search",
           "offline_only":True,"api_calls_made":0,
           "no_key_or_image_base64":True}
    write_immutable_files({arm_dir/"离线审计_v3.json":json.dumps(audit,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8")})
    return audit

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",type=Path,default=MASA)
    parser.add_argument("--arm-dir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.dataset_root,args.arm_dir),ensure_ascii=False,indent=2))
