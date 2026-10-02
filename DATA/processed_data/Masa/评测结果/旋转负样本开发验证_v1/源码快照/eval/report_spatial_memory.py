"""汇总 ranked VLM 与空间记忆治理器的配对对照；不调用API。"""
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


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path):
    return [json.loads(line) for line in (path / "任务结果.jsonl").read_text(encoding="utf-8").splitlines()]


def _row(name, summary):
    s = summary["summary"]
    m = s["metrics_completed_only"]
    d = s["diagnostics"]
    a = s["api"]
    return {
        "name": name,
        "completed": f'{s["completed"]}/{s["scheduled"]}',
        "successes": m["successes"] if m else None,
        "sr": m["sr"] if m else None,
        "sg": m["mean_sg_all_episodes"] if m else None,
        "unique": d["mean_unique_cells"],
        "repeat": m["repeat_visit_rate_micro"] if m else None,
        "cycles": d["repeat_cycle_episodes"],
        "overrides": d.get("spatial_memory_overrides", 0) or d.get("governor_overrides", 0),
        "requests": a["requests"],
        "api_errors": a["errors"],
    }


def _paired_rows(arms):
    records = {name: _records(path) for name, path in arms.items()}
    complete = {
        name: {r["episode_id"]: r for r in rows if r["completion"] == "completed"}
        for name, rows in records.items()
    }
    shared = set.intersection(*(set(items) for items in complete.values()))
    result = []
    for name, items in complete.items():
        selected = [items[e] for e in sorted(shared)]
        evals = [r["evaluation"] for r in selected]
        diagnostics = [r["diagnostics"] for r in selected]
        result.append({
            "name": name,
            "count": len(selected),
            "metrics": metrics(evals),
            "unique": statistics.mean(d["unique_cells_including_start"] for d in diagnostics),
        })
    return sorted(shared), result


def report(root: Path, output: Path):
    root = Path(root)
    arms = {
        "B ranked top1": root.parent / "四组实验_v1/B_ranked-vlm-top1",
        "C ranked + ABAB Governor": root.parent / "四组实验_v1/C_ranked-vlm-governed",
        "E ranked + Spatial Memory": root / "E_ranked-vlm-spatial-memory",
    }
    summaries = {name: _read(path / "验证汇总.json") for name, path in arms.items()}
    task_hashes = {item["task_sha256"] for item in summaries.values()}
    if len(task_hashes) != 1:
        raise ValueError("对照臂任务清单哈希不一致")
    shared, paired = _paired_rows(arms)
    rows = [_row(name, summaries[name]) for name in arms]

    lines = [
        "# 空间记忆单变量对照报告",
        "",
        "## 实验定义",
        "",
        "- B 为 ranked_actions 直接取 top-1；C 为 ranked_actions 加 ABAB Governor；E 为 ranked_actions 加一步空间记忆治理。",
        "- 三个实验臂使用同一 `validation20-v1` 任务清单、5×5 网格、预算10、四方向动作和同一双图 VLM 输入。",
        "- E 只读取公开 `visited`、当前位置和网格边界：优先选择未访问邻格；若没有未访问邻格，选择访问次数最少的候选格。它不读取目标坐标、真实距离、未访问图像或评测日志。",
        f"- 任务清单 SHA256：`{next(iter(task_hashes))}`。本报告是开发诊断，不是论文复现或显著性检验。",
        "",
        "## 全部在线结果",
        "",
        "| 实验臂 | 完整题数 | 成功 | SR（完整题） | 平均 SG（完整题） | 平均不同格 | 重复访问率 | 循环题数 | 治理改写 | API请求/错误 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["name"]} | {row["completed"]} | {row["successes"] if row["successes"] is not None else "—"} | '
            f'{row["sr"]:.1%} | {row["sg"]:.2f} | {row["unique"]:.2f} | {row["repeat"]:.1%} | '
            f'{row["cycles"]} | {row["overrides"]} | {row["requests"]}/{row["api_errors"]} |'
        )
    lines += [
        "",
        "E 臂有1题因API错误未完成，因此其 SR 分母是19，不与20题结果直接作最终优劣结论。API错误保留在日志中，没有用随机动作补齐。",
        "",
        "## 共同完成题目的配对结果",
        "",
        f"三臂共同完成 `{len(shared)}` 题；以下仅在共同完成题上比较，排除了 E 的API错误题。",
        "",
        "| 实验臂 | 共同完成题数 | 成功 | SR | 平均 SG | 平均不同格 | 重复访问率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        m = row["metrics"]
        lines.append(
            f'| {row["name"]} | {row["count"]} | {m["successes"]} | {m["sr"]:.1%} | '
            f'{m["mean_sg_all_episodes"]:.2f} | {row["unique"]:.2f} | {m["repeat_visit_rate_micro"]:.1%} |'
        )
    lines += [
        "",
        "## 观察与限制",
        "",
        "- E 在已完成题上将重复访问率降为0%，平均覆盖提高；这证明了该治理器确实改变了动作选择。",
        "- 这次 E 只有一次在线运行，且有1题API错误；不能把 SR 的变化单独归因于空间记忆，也不能据此宣称算法提升。",
        "- E 的治理器是一步候选重排，不是完整空间地图、全局搜索或多智能体记忆。下一项创新应继续保持同一评测器，并单独建立新实验臂。",
        "- 详细请求、动作、任务结果和离线审计均保存在本目录；本报告不调用API。",
        "",
    ]
    write_immutable_files({Path(output): ("\n".join(lines)).encode("utf-8")})
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=MASA / "评测结果/空间记忆实验_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "空间记忆对照报告.md"
    print(report(args.root, output))
