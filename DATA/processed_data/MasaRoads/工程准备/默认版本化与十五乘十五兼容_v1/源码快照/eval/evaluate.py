"""本地基础评测：校验固定试卷，记录逐题轨迹与按距离/种子统计。"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.make_episodes import MASA, json_bytes, seed_for
from env.environment import GridWorldEnv, Observation
from env.episode import (ACTIONS, PROTOCOL, SPLITS, inspect_area, load_episodes,
                         write_immutable_files)


class RandomPolicy:
    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def act(self, observation: Observation) -> str:
        return self._rng.choice(observation.legal_actions)


def verify_task_file(dataset_root: Path, episode_path: Path, expected_split: str | None = None):
    if expected_split is None:
        expected_split = episode_path.stem.removeprefix("episodes_")
    if expected_split not in SPLITS or episode_path.name != f"episodes_{expected_split}.jsonl":
        raise ValueError("文件名与请求的 split 不一致")
    episodes = load_episodes(episode_path)
    split = expected_split
    if any(ep.split != split for ep in episodes):
        raise ValueError("episode 内容与请求的 split 不一致")
    manifest_path = episode_path.parent / f"manifest_{split}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or manifest.get("split") != split:
        raise ValueError("清单协议或 split 不一致")
    src = Path(__file__).resolve().parents[1]
    for field, relative in (("generator_sha256", "data/make_episodes.py"),
                            ("validator_sha256", "env/episode.py")):
        if manifest.get(field) != sha256((src / relative).read_bytes()).hexdigest():
            raise ValueError(f"{field} 与当前代码不一致，需明确迁移协议版本")
    if (manifest.get("distances") != [4, 5, 6, 7, 8]
            or manifest.get("boundary") != "stay_and_consume_one_step"
            or manifest.get("termination") != "first_arrival_success_including_last_budget_step"
            or manifest.get("sampling") != "ordered_pairs_with_replacement"
            or manifest.get("target_modality") != "aerial_image"):
        raise ValueError("manifest 任务规则不一致")
    if sha256(episode_path.read_bytes()).hexdigest() != manifest["episodes_sha256"]:
        raise ValueError("episode 哈希不匹配")
    if sha256((dataset_root / "metadata.csv").read_bytes()).hexdigest() != manifest["metadata_sha256"]:
        raise ValueError("metadata 哈希不匹配")
    if len(episodes) != manifest["episode_count"]:
        raise ValueError("episode 数量不一致")
    if manifest["actions"] != list(ACTIONS) or manifest["grid_size"] != 5:
        raise ValueError("动作集或网格协议不一致")
    if manifest["memory"] != "episode_isolated":
        raise ValueError("本评测不允许跨题记忆")
    areas = {ep.area for ep in episodes}
    if areas != set(manifest["area_sha256"]) or len(areas) != manifest["area_count"]:
        raise ValueError("区域清单不一致")
    counts = Counter()
    for ep in episodes:
        if ep.split != split or ep.budget != manifest["budget"]:
            raise ValueError("episode split 或预算与 manifest 不一致")
        counts[(ep.area, ep.dist)] += 1
    expected = {(area, d): manifest["n_per_area_per_distance"]
                for area in areas for d in manifest["distances"]}
    if dict(counts) != expected:
        raise ValueError("距离分层数量不一致")
    for area in sorted(areas):
        if inspect_area(dataset_root, split, area) != manifest["area_sha256"][area]:
            raise ValueError(f"图块哈希变化: {area}")
    return episodes, manifest


def metrics(results: list[dict]) -> dict:
    if not results:
        raise ValueError("不能统计空结果")
    count = len(results)
    steps = sum(r["steps"] for r in results)
    return {"episodes": count,
            "successes": sum(r["success"] for r in results),
            "sr": sum(r["success"] for r in results) / count,
            "mean_sg_all_episodes": statistics.mean(r["sg"] for r in results),
            "mean_steps": statistics.mean(r["steps"] for r in results),
            "repeat_visit_rate_micro": sum(r["revisits"] for r in results) / steps,
            "repeat_visit_rate_macro": statistics.mean(r["repeat_visit_rate"] for r in results),
            "out_of_bounds_rate": sum(r["out_of_bounds"] for r in results) / steps}


def summarize(results: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in results:
        grouped[(record["split"], record["policy_seed"])].append(record)
    runs = {}
    for (split, seed), records in sorted(grouped.items()):
        runs[f"{split}/seed_{seed}"] = {
            "all": metrics(records),
            "by_distance": {str(d): metrics([r for r in records if r["distance"] == d])
                            for d in sorted({r["distance"] for r in records})}}
    stability = {}
    for split in sorted({r["split"] for r in results}):
        values = [run["all"]["sr"] for key, run in runs.items() if key.startswith(split + "/")]
        stability[split] = {"seeds": len(values), "sr_mean": statistics.mean(values),
                            "sr_population_std": statistics.pstdev(values),
                            "note": "跨随机种子的描述统计，不是置信区间或独立地图样本数"}
    return {"per_seed": runs, "seed_stability": stability}


def evaluate(dataset_root: Path, task_dir: Path, splits, seeds, output_dir: Path):
    root = Path(dataset_root)
    if not splits or len(set(splits)) != len(splits) or any(s not in SPLITS for s in splits):
        raise ValueError("非法 splits")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("策略种子须为不重复的非负整数")
    tasks = {split: verify_task_file(root, Path(task_dir) / f"episodes_{split}.jsonl", split)
             for split in sorted(splits)}
    results = []
    env = GridWorldEnv(root)
    for split, (episodes, _) in tasks.items():
        for seed in sorted(seeds):
            for ep in episodes:
                observation = env.reset(ep)
                policy = RandomPolicy(seed_for(seed, PROTOCOL, ep.episode_id))
                while not env.done:
                    action = policy.act(observation)
                    observation, _, _ = env.step(action)
                result = env.evaluator_result()
                result.update(split=split, area=ep.area, distance=ep.dist, budget=ep.budget,
                              goal=ep.goal, start=ep.start, policy_seed=seed,
                              policy="uniform_random_four_actions")
                results.append(result)
    src = Path(__file__).resolve().parents[1]
    summary = {
        "protocol": PROTOCOL, "policy": "uniform_random_four_actions",
        "scope": "基础回归验证；本地配置，非论文复现和模型效果结论",
        "seeds": sorted(seeds),
        "runtime": {"python": sys.version.split()[0], "pillow": __import__("PIL").__version__},
        "tests_sha256": sha256((src / "tests/test_foundation.py").read_bytes()).hexdigest(),
        "task_sha256": {s: m["episodes_sha256"] for s, (_, m) in tasks.items()},
        "task_manifest_sha256": {s: sha256((Path(task_dir) / f"manifest_{s}.json").read_bytes()).hexdigest() for s in tasks},
        "code_sha256": {p: sha256((src / p).read_bytes()).hexdigest()
                        for p in ("env/environment.py", "env/episode.py", "eval/evaluate.py", "data/make_episodes.py")},
        "metric_definitions": {
            "sr": "首次到达目标的episode数/总episode数；最后一步到达算成功",
            "sg": "最终到目标曼哈顿距离，对全部episode平均，成功episode为0",
            "repeat": "动作后落点曾被访问则记1，含越界原地；起始格不计分母",
            "micro": "重复访问动作总数/已执行动作总数",
            "macro": "各episode重复访问率的算术平均",
            "steps": "全部执行动作，含越界；不包含reset",
        },
        "log_access": "仅评测端，包含目标坐标，不得传给策略",
        **summarize(results),
    }
    logs = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in results).encode("utf-8")
    summary["trajectory_sha256"] = sha256(logs).hexdigest()
    write_immutable_files({Path(output_dir) / "随机基线_轨迹.jsonl": logs,
                           Path(output_dir) / "随机基线_汇总.json": json_bytes(summary)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=MASA)
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["val", "test"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    summary = evaluate(args.dataset_root,
                       args.task_dir or args.dataset_root / "任务清单_v2",
                       args.splits, args.seeds,
                       args.output_dir or args.dataset_root / "评测结果/基础校验_v2")
    print(json.dumps(summary["seed_stability"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
