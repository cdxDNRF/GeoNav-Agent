"""生成本地 Masa v2 固定试卷；不覆盖旧清单，不宣称复现论文原始配置。"""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.episode import (ACTIONS, Episode, PROTOCOL, SPLITS, inspect_area,
                         manhattan, safe_area, write_immutable_files)

ROOT = Path(__file__).resolve().parents[3]
MASA = ROOT / "DATA" / "processed_data" / "Masa"
DISTANCES = (4, 5, 6, 7, 8)


def seed_for(seed: int, *labels: str) -> int:
    value = json.dumps([seed, *labels], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(sha256(value.encode("utf-8")).digest(), "big")


def pairs_at_distance(grid: int, dist: int):
    return [(s, t) for s in range(grid * grid) for t in range(grid * grid)
            if manhattan(s, t, grid) == dist]


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build_split(dataset_root: Path, split: str, seed: int = 42,
                budget: int = 10, n_per_dist: int = 5):
    if split not in SPLITS or type(seed) is not int or seed < 0:
        raise ValueError("非法 split 或 seed")
    if type(budget) is not int or type(n_per_dist) is not int or min(budget, n_per_dist) <= 0:
        raise ValueError("预算和每距离采样数必须为正整数")
    root = Path(dataset_root)
    metadata_path = root / "metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == split]
    if not rows:
        raise ValueError(f"metadata.csv 缺少 {split}")
    areas = [safe_area(row["img_id"]) for row in rows]
    if len(areas) != len(set(areas)):
        raise ValueError("metadata 存在重复区域")
    rows.sort(key=lambda row: int(row["img_id"].split("_")[1]))
    hashes = {row["img_id"]: inspect_area(root, split, row["img_id"]) for row in rows}
    pools = {d: pairs_at_distance(5, d) for d in DISTANCES}
    episodes = []
    duplicates = Counter()
    for row in rows:
        area = row["img_id"]
        for distance in DISTANCES:
            # 对每个 split / area / distance 独立派生随机流，不依赖调用或遍历顺序。
            rng = random.Random(seed_for(seed, PROTOCOL, split, area, str(distance)))
            seen = set()
            for i in range(n_per_dist):
                start, goal = rng.choice(pools[distance])
                duplicates[distance] += (start, goal) in seen
                seen.add((start, goal))
                ep = Episode(f"{split}_{area}_d{distance}_{i:03d}", split, area,
                             start, goal, distance, budget, source_tile=row["source_tile"])
                ep.validate()
                episodes.append(asdict(ep))
    payload = "".join(json.dumps(ep, ensure_ascii=False, sort_keys=True) + "\n"
                      for ep in episodes).encode("utf-8")
    manifest = {
        "protocol": PROTOCOL,
        "scope": "local_151_image_Masa_not_paper_split",
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "validator_sha256": sha256((Path(__file__).resolve().parents[1] / "env/episode.py").read_bytes()).hexdigest(),
        "split": split, "seed": seed, "budget": budget, "grid_size": 5,
        "n_per_area_per_distance": n_per_dist, "distances": DISTANCES,
        "sampling": "ordered_pairs_with_replacement",
        "rng": "python_random_sha256_per_split_area_distance",
        "actions": list(ACTIONS), "boundary": "stay_and_consume_one_step",
        "termination": "first_arrival_success_including_last_budget_step",
        "memory": "episode_isolated", "target_modality": "aerial_image",
        "area_count": len(rows), "episode_count": len(episodes),
        "available_pairs_per_distance": {str(d): len(pools[d]) for d in DISTANCES},
        "duplicate_draws_per_distance": dict(duplicates),
        "metadata_sha256": sha256(metadata_path.read_bytes()).hexdigest(),
        "area_sha256": hashes, "episodes_sha256": sha256(payload).hexdigest(),
    }
    return payload, manifest


def generate(dataset_root: Path = MASA, output_dir: Path | None = None,
             splits=SPLITS, seed=42, budget=10, n_per_dist=5):
    root = Path(dataset_root)
    output_dir = Path(output_dir) if output_dir is not None else root / "任务清单_v2"
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits 不可为空或重复")
    files = {}
    manifests = []
    for split in splits:
        payload, manifest = build_split(root, split, seed, budget, n_per_dist)
        files[output_dir / f"episodes_{split}.jsonl"] = payload
        files[output_dir / f"manifest_{split}.json"] = json_bytes(manifest)
        manifests.append(manifest)
    write_immutable_files(files)
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=MASA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--n-per-dist", type=int, default=5)
    parser.add_argument("--grid-size", type=int, choices=[5], default=5)
    args = parser.parse_args()
    manifests = generate(args.dataset_root, args.output_dir, args.splits, args.seed,
                         args.budget, args.n_per_dist)
    for manifest in manifests:
        print(f"{manifest['split']}: {manifest['area_count']} areas, "
              f"{manifest['episode_count']} episodes, sha256={manifest['episodes_sha256']}")


if __name__ == "__main__":
    main()
