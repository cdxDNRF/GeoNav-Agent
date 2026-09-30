"""离线重验已保存的最终回答，只校验解析，不执行环境、不重算原试跑SR。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.vlm import parse_action, parse_ranked_actions
from data.make_episodes import json_bytes
from env.episode import ACTIONS, write_immutable_files


def replay(source: Path, output: Path):
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    result = []
    for row in rows:
        item = {"call_index": row["call_index"], "original_status": row["status"]}
        try:
            if row.get("output_schema") == "ranked_actions" or "ranked_actions" in row:
                ranked, normalized = parse_ranked_actions(row.get("raw_content"), tuple(ACTIONS), row.get("finish_reason"))
                item.update(replay_status="valid", ranked_actions=list(ranked), normalized_fence=normalized, output_schema="ranked_actions")
            else:
                action, normalized = parse_action(row.get("raw_content"), tuple(ACTIONS), row.get("finish_reason"))
                item.update(replay_status="valid", action=action, normalized_fence=normalized, output_schema="action")
        except (ValueError, TypeError):
            item["replay_status"] = "invalid"
        result.append(item)
    summary = {"scope": "offline_response_parse_only_not_a_completed_episode_or_new_api_run",
               "source_sha256": sha256(source.read_bytes()).hexdigest(),
               "parser_sha256": sha256((Path(__file__).resolve().parents[1] / "agents/vlm.py").read_bytes()).hexdigest(),
               "api_calls": 0, "responses": len(rows),
               "valid_responses": sum(r["replay_status"] == "valid" for r in result),
               "normalized_fences": sum(r.get("normalized_fence", False) for r in result),
               "results": result}
    write_immutable_files({output: json_bytes(summary)})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = replay(args.source, args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
