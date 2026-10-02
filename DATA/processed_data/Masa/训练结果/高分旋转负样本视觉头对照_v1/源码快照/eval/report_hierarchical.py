"""汇总层级搜索与已有基线的配对对照；不调用API。"""
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


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records(path):
    return [json.loads(line) for line in (Path(path) / "任务结果.jsonl").read_text(encoding="utf-8").splitlines()]


def _row(name, summary):
    s = summary["summary"]
    m = s["metrics_completed_only"]
    d = s["diagnostics"]
    a = s["api"]
    return {"name": name, "completed": f'{s["completed"]}/{s["scheduled"]}',
            "successes": m["successes"] if m else None, "sr": m["sr"] if m else None,
            "sg": m["mean_sg_all_episodes"] if m else None,
            "unique": d["mean_unique_cells"], "repeat": m["repeat_visit_rate_micro"] if m else None,
            "adjacent": d["adjacent_but_not_reached"],
            "hierarchical": d.get("hierarchical_overrides", 0),
            "region_changes": d.get("hierarchical_target_region_changes", 0),
            "api": f'{a["requests"]}/{a["errors"]}'}


def _paired(arms):
    complete = {}
    for name, path in arms.items():
        complete[name] = {r["episode_id"]: r for r in _records(path) if r["completion"] == "completed"}
    shared = set.intersection(*(set(rows) for rows in complete.values()))
    paired = []
    for name, records in complete.items():
        selected = [records[e] for e in sorted(shared)]
        ev = [r["evaluation"] for r in selected]
        ds = [r["diagnostics"] for r in selected]
        paired.append({"name": name, "count": len(selected), "metrics": metrics(ev),
                       "unique": statistics.mean(d["unique_cells_including_start"] for d in ds),
                       "adjacent": sum(d["adjacent_but_not_reached"] for d in ds)})
    return sorted(shared), paired


def report(root: Path, output: Path):
    root = Path(root)
    arms = {
        "B ranked top1": root.parent / "四组实验_v1/B_ranked-vlm-top1",
        "C ranked + ABAB Governor": root.parent / "四组实验_v1/C_ranked-vlm-governed",
        "E ranked + Spatial Memory": root.parent / "空间记忆实验_v1/E_ranked-vlm-spatial-memory",
        "F ranked + Neighborhood Confirmation": root.parent / "邻域确认实验_v1/F_ranked-vlm-neighborhood-confirmation",
        "G ranked + Hierarchical Search": root / "G_ranked-vlm-hierarchical-search",
    }
    summaries = {name: _load(path / "验证汇总.json") for name, path in arms.items()}
    task_hashes = {item["task_sha256"] for item in summaries.values()}
    if len(task_hashes) != 1:
        raise ValueError("对照臂任务清单哈希不一致")
    shared, paired = _paired(arms)
    rows = [_row(name, summaries[name]) for name in arms]
    lines = [
        "# 层级搜索单变量对照报告", "", "## 实验定义", "",
        "- E 是 ranked VLM 加空间记忆；G 在同一空间记忆上增加 3×3 粗区域排序，再执行四方向逐格移动。",
        "- 粗区域按 `region_id=r(row//2)c(col//2)` 映射，VLM 输出九区域全排列和四动作全排列。治理器只使用公开 visited、当前位置、边界和两组排序。",
        "- G 不增加视野、不允许跳跃或 stop、不读取目标坐标、真实距离、未访问图像或评测日志。",
        f"- 所有臂共享任务清单 SHA256：`{next(iter(task_hashes))}`。本报告是一次20题开发诊断。", "",
        "## 在线结果", "",
        "| 实验臂 | 完整题数 | 成功 | SR（完整题） | 平均 SG | 平均不同格 | 重复访问率 | 相邻未到达 | 层级改写 | 区域切换 | API请求/错误 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f'| {row["name"]} | {row["completed"]} | {row["successes"] if row["successes"] is not None else "—"} | '
                     f'{row["sr"]:.1%} | {row["sg"]:.2f} | {row["unique"]:.2f} | {row["repeat"]:.1%} | '
                     f'{row["adjacent"]} | {row["hierarchical"]} | {row["region_changes"]} | {row["api"]} |')
    lines += [
        "", "## 五臂共同完成题的配对结果", "",
        f"五个实验臂共同完成 `{len(shared)}` 题，以下仅比较共同完成题。", "",
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
        "- G 在本轮20题上完成20/20，SR 50%，平均 SG 2.00，重复访问率约1.1%，相邻未到达1题；区域排序实际触发110次层级改写和62次目标区域与落点区域不一致。",
        "- G 的 SR 与无图 Frontier 基线本轮同为50%，不能据此证明 VLM 或层级规划优于 Frontier；当前样本只有4个区域、每臂一次运行。",
        "- G 与 E 使用不同输出协议和提示词，结果只能说明“加入粗区域规划后的完整系统”在本轮表现更好，不能把提升单独归因于某个内部子步骤。",
        "- 详细区域排序、动作轨迹、API记录和离线审计保存在本目录；本报告不调用API。", "",
    ]
    write_immutable_files({Path(output): ("\n".join(lines)).encode("utf-8")})
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=MASA / "评测结果/层级搜索实验_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(report(args.root, args.output or args.root / "层级搜索对照报告.md"))
