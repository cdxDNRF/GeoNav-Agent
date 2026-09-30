"""汇总空间记忆与目标邻域确认的配对对照；不调用API。"""
import argparse
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.make_episodes import MASA
from env.episode import write_immutable_files
from eval.evaluate import metrics


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records(path):
    return [json.loads(line) for line in
            (Path(path) / "任务结果.jsonl").read_text(encoding="utf-8").splitlines()]


def _summary_row(name, summary):
    s, d, a = summary["summary"], summary["summary"]["diagnostics"], summary["summary"]["api"]
    m = s["metrics_completed_only"]
    return {
        "name": name, "completed": f'{s["completed"]}/{s["scheduled"]}',
        "successes": m["successes"] if m else None, "sr": m["sr"] if m else None,
        "sg": m["mean_sg_all_episodes"] if m else None,
        "unique": d["mean_unique_cells"], "repeat": m["repeat_visit_rate_micro"] if m else None,
        "adjacent": d["adjacent_but_not_reached"], "confirmation": d.get("confirmation_overrides", 0),
        "high": d.get("confirmation_high_decisions", 0), "api": f'{a["requests"]}/{a["errors"]}',
    }


def _paired(arms):
    complete = {}
    for name, path in arms.items():
        complete[name] = {r["episode_id"]: r for r in _records(path) if r["completion"] == "completed"}
    shared = set.intersection(*(set(rows) for rows in complete.values()))
    rows = []
    for name, records in complete.items():
        selected = [records[e] for e in sorted(shared)]
        ev = [r["evaluation"] for r in selected]
        ds = [r["diagnostics"] for r in selected]
        rows.append({"name": name, "count": len(selected), "metrics": metrics(ev),
                     "unique": statistics.mean(d["unique_cells_including_start"] for d in ds),
                     "adjacent": sum(d["adjacent_but_not_reached"] for d in ds)})
    return sorted(shared), rows


def _evidence_counts(path):
    counts = {"low": 0, "medium": 0, "high": 0}
    for record in _records(path):
        for event in record["actions"]:
            evidence = event.get("target_evidence")
            if evidence in counts:
                counts[evidence] += 1
    return counts


def report(root: Path, output: Path):
    root = Path(root)
    arms = {
        "B ranked top1": root.parent / "四组实验_v1/B_ranked-vlm-top1",
        "C ranked + ABAB Governor": root.parent / "四组实验_v1/C_ranked-vlm-governed",
        "E ranked + Spatial Memory": root.parent / "空间记忆实验_v1/E_ranked-vlm-spatial-memory",
        "F ranked + Neighborhood Confirmation": root / "F_ranked-vlm-neighborhood-confirmation",
    }
    summaries = {name: _json(path / "验证汇总.json") for name, path in arms.items()}
    hashes = {s["task_sha256"] for s in summaries.values()}
    if len(hashes) != 1:
        raise ValueError("对照实验臂任务清单哈希不一致")
    shared, paired = _paired(arms)
    rows = [_summary_row(name, summaries[name]) for name in arms]
    evidence = _evidence_counts(arms["F ranked + Neighborhood Confirmation"])

    lines = [
        "# 目标邻域确认单变量对照报告", "", "## 实验定义", "",
        "- E 是 ranked VLM 加空间记忆：优先未访问格，否则访问次数最少。",
        "- F 在 E 之上要求 VLM 输出 `target_evidence`（low/medium/high）；只有 high 且 top1 指向已访问的非当前位置时，才允许一次局部回退。",
        "- F 不增加视野，不加入 stop，不读取目标坐标、距离、未访问图像或评测日志；环境和四方向动作协议不变。",
        f"- 四个实验臂共享任务清单 SHA256：`{next(iter(hashes))}`。结果是一次小规模开发诊断。", "",
        "## 在线结果", "",
        "| 实验臂 | 完整题数 | 成功 | SR（完整题） | 平均 SG | 平均不同格 | 重复访问率 | 相邻未到达 | 确认回退 | high决策 | API请求/错误 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f'| {row["name"]} | {row["completed"]} | {row["successes"] if row["successes"] is not None else "—"} | '
                     f'{row["sr"]:.1%} | {row["sg"]:.2f} | {row["unique"]:.2f} | {row["repeat"]:.1%} | '
                     f'{row["adjacent"]} | {row["confirmation"]} | {row["high"]} | {row["api"]} |')
    lines += [
        "", f"F 的 target_evidence 分布：low={evidence['low']}，medium={evidence['medium']}，high={evidence['high']}。",
        "F 有1题API错误，完整题分母为19；API错误没有用随机动作补齐。", "",
        "## 四臂共同完成题的配对结果", "",
        f"四个实验臂共同完成 `{len(shared)}` 题，以下仅比较这部分题目。", "",
        "| 实验臂 | 共同完成题数 | 成功 | SR | 平均 SG | 平均不同格 | 重复访问率 | 相邻未到达 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        m = row["metrics"]
        lines.append(f'| {row["name"]} | {row["count"]} | {m["successes"]} | {m["sr"]:.1%} | '
                     f'{m["mean_sg_all_episodes"]:.2f} | {row["unique"]:.2f} | '
                     f'{m["repeat_visit_rate_micro"]:.1%} | {row["adjacent"]} |')
    lines += [
        "", "## 结论与限制", "",
        "- F 的确认字段在本轮只产生1次 high，确认回退只触发1次；它没有形成足够的确认信号来改善结果。",
        "- F 相比 E 重新引入了少量重复访问，且 SR/SG 没有改善；当前证据不支持继续扩大该机制。",
        "- 该负结果仍然保留，因为它验证了统一日志、严格 schema、确认决策重放和单变量消融链路。下一项创新应回到层级搜索或改进证据表示，不能把本轮结果写成有效性证明。",
        "- 详细请求、轨迹、配置和离线审计保存在本目录；本报告不调用API。", "",
    ]
    write_immutable_files({Path(output): ("\n".join(lines)).encode("utf-8")})
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=MASA / "评测结果/邻域确认实验_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(report(args.root, args.output or args.root / "邻域确认对照报告.md"))
