"""Read sealed navigation logs; never import or execute models or policies."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

RADIAL = "DATA/processed_data/MasaRoads/评测结果/径向保护独立新区域导航确认_v1"
RADIAL_DATA = "DATA/processed_data/MasaRoads/径向保护新区域确认_v1/工程数据"
MASS = "DATA/processed_data/MassGIS/评测结果/正式十乘十十五乘十五确认_v1"
MASS_DATA = "DATA/processed_data/MassGIS/工程准备/正式导航协议冻结_v1"
ACTIONS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """Explicitly reject absolute paths, traversal and symlinks leaving root."""
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("路径不在白名单范围")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("路径越过仓库根")
    return path


def distance(a: int, b: int, grid: int) -> int:
    return abs(a // grid - b // grid) + abs(a % grid - b % grid)


def validate_record(record: dict, task: dict) -> None:
    """Independent arithmetic checks of saved actions, counts and public context."""
    grid, budget = task["grid_size"], task["budget"]
    route, decisions = record["trajectory"], record["decisions"]
    if record["status"] != "completed" or not 0 <= record["steps"] <= budget:
        raise ValueError("非完成轨迹或预算错误")
    if len(route) != record["steps"] + 1 or len(decisions) != record["steps"]:
        raise ValueError("动作/决策分母错误")
    if route[0]["patch_id"] != task["start"] or route[0]["step"] != 0:
        raise ValueError("起点与冻结任务不一致")
    visited, travel, revisits, oob = {task["start"]}, 0, 0, 0
    for i, (entry, decision) in enumerate(zip(route[1:], decisions), 1):
        previous = route[i - 1]["patch_id"]
        if previous == task["goal"]:
            raise ValueError("已到达后仍行动")
        if decision["public_position"] != [previous // grid, previous % grid]:
            raise ValueError("决策与行动前位置错位")
        if set(decision["public_visited"]) != visited or decision["remaining_budget"] != budget - i + 1:
            raise ValueError("公开访问历史或预算错位")
        action = entry["action"]
        if action != decision["action"] or action not in ACTIONS or entry["step"] != i or decision["step"] != i:
            raise ValueError("保存动作不一致")
        dr, dc = ACTIONS[action]
        r, c = previous // grid + dr, previous % grid + dc
        outside = not (0 <= r < grid and 0 <= c < grid)
        expected = previous if outside else r * grid + c
        if entry["patch_id"] != expected or bool(entry["out_of_bounds"]) != outside:
            raise ValueError("保存路径不符合四向动作")
        travel += int(expected != previous) * record["cell_size_m"]
        revisits += int(expected in visited)
        oob += int(outside)
        visited.add(expected)
    sg = distance(route[-1]["patch_id"], task["goal"], grid)
    if record["success"] != (sg == 0) or record["sg"] != sg or record["sg_m"] != sg * record["cell_size_m"]:
        raise ValueError("SR/SG与终局不一致")
    if record["valid_travel_m"] != travel or record["revisits"] != revisits or record["out_of_bounds"] != oob:
        raise ValueError("移动距离或重访统计不一致")
    if record["termination"] != ("goal_reached" if sg == 0 else "budget_exhausted"):
        raise ValueError("终止原因不一致")
    if sg and record["steps"] != budget:
        raise ValueError("失败任务提前终止")
    for field in ("explorer_logits", "probabilities"):
        width = 4 if field == "explorer_logits" else 5
        if any(len(d[field]) != width or not all(math.isfinite(x) for x in d[field]) for d in decisions):
            raise ValueError("决策向量格式错误")


class ReplayStore:
    """Only small metadata stays resident; episode payloads are loaded on demand."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.bindings: dict[str, str] = {}
        self.catalogs: dict[str, dict] = {}
        self.entries: dict[str, dict] = {}
        self.images: dict[tuple, tuple[str, str]] = {}
        self._load_radial()
        self._load_massgis()
        for catalog in self.catalogs.values():
            rows = [r for r in self.entries.values() if r["catalog"] == catalog["id"]]
            expected = catalog["planned_tasks"] * 6
            if len(rows) != expected:
                raise ValueError(f"{catalog['id']}计划分母缺失: {len(rows)}/{expected}")
            catalog["records"] = len(rows)
            catalog["areas"] = sorted({r["area"] for r in rows})
            catalog["metrics"] = {}
            for policy in ("M0", "Coverage3Radial"):
                arm = [r for r in rows if r["policy"] == policy]
                if any(sum(r["seed"] == s for r in arm) != catalog["planned_tasks"] for s in range(3)):
                    raise ValueError("权重臂计划分母不完整")
                catalog["metrics"][policy] = {
                    "planned": len(arm), "successes": sum(r["success"] for r in arm),
                    "sr": sum(r["success"] for r in arm) / len(arm),
                    "sg_m": sum(r["sg_m"] for r in arm) / len(arm),
                    "movement_m": sum(r["movement_m"] for r in arm) / len(arm),
                }

    def bound_bytes(self, relative: str, sha: str | None = None) -> bytes:
        data = safe_path(self.root, relative).read_bytes()
        actual = digest(data)
        if sha is not None and actual != sha:
            raise ValueError(f"封存SHA不一致: {relative}")
        self.bindings[relative] = actual
        return data

    def bound_json(self, relative: str, sha: str | None = None):
        return json.loads(self.bound_bytes(relative, sha))

    def _load_radial(self):
        stage = self.bound_json(f"{RADIAL}/阶段交付核验.json")
        sealed = stage["files_sha256"]
        registration_seal = self.bound_json(f"{RADIAL}/预登记封存.json")
        registration = self.bound_json(f"{RADIAL}/预登记.json", registration_seal["registration_sha256"])
        tasks_path = f"{RADIAL}/导航任务.json"
        tasks = self.bound_json(tasks_path, registration["frozen_sha256"][tasks_path])
        manifest_path = f"{RADIAL_DATA}/数据清单.json"
        manifest = self.bound_json(manifest_path, registration["input_sha256"][manifest_path])
        cid = "roads10"
        self.catalogs[cid] = {"id": cid, "label": "MassachusettsRoads · 9 km²", "grid": 10,
            "planned_tasks": 750, "regions": 10, "area_km2": 9, "cell_size_m": 300,
            "budget": 20, "acceptance": "通过 · 已验证协议", "accepted": True,
            "report": f"{RADIAL}/径向保护独立新区域导航确认报告.md", "protocol": manifest["protocol"]}
        for region in manifest["regions"]:
            for cell in range(100):
                rel = f"{RADIAL_DATA}/patches/test/{region['area']}/patch_{cell}.jpg"
                self.images[(cid, region["area"], cell)] = (rel, registration["input_sha256"][rel])
        taskmap = {t["episode_id"]: t for t in tasks}
        for policy in ("M0", "Coverage3Radial"):
            for seed in range(3):
                rel = f"{RADIAL}/主对照/{policy}_s{seed}_CueFull_轨迹.jsonl"
                self._index_log(cid, rel, sealed[rel], taskmap, policy, seed)

    def _load_massgis(self):
        stage = self.bound_json(f"{MASS}/核验/阶段封存.json")
        sealed = stage["files_sha256"]
        protocol_stage_path = f"{MASS_DATA}/核验/阶段封存.json"
        protocol_stage = self.bound_json(protocol_stage_path)
        psha = protocol_stage["files_sha256"]
        for grid, count in ((10, 750), (15, 1000)):
            cid = f"mass{grid}"
            task_path = f"{MASS_DATA}/元数据/grid{grid}任务.json"
            tasks = self.bound_json(task_path, psha[task_path])
            manifest_path = f"{MASS_DATA}/工程数据/grid{grid}/数据清单.json"
            manifest = self.bound_json(manifest_path, psha[manifest_path])
            self.catalogs[cid] = {"id": cid, "label": f"MassGIS · {9 if grid == 10 else 20.25} km²",
                "grid": grid, "planned_tasks": count, "regions": 10,
                "area_km2": 9 if grid == 10 else 20.25, "cell_size_m": 300, "budget": 20,
                "acceptance": "未通过 · 扩展验证", "accepted": False,
                "report": f"{MASS}/正式双网格确认报告.md", "protocol": manifest["protocol"]}
            for region in manifest["regions"]:
                for cell in region["cells"]:
                    self.images[(cid, region["area"], cell["cell"])] = (cell["path"], cell["file_sha256"])
            taskmap = {t["episode_id"]: t for t in tasks}
            for policy in ("M0", "Coverage3Radial"):
                for seed in range(3):
                    for region in manifest["regions"]:
                        rel = f"{MASS}/主对照/g{grid}_{policy}_s{seed}_CueFull_{region['area']}.jsonl"
                        seal_rel = str(Path(rel).with_suffix('.seal.json')).replace('\\', '/')
                        seal = self.bound_json(seal_rel, sealed[seal_rel])
                        if seal["sha256"] != sealed[rel] or seal["binding"]["tasks_sha256"] != psha[task_path]:
                            raise ValueError("轨迹seal或任务绑定错误")
                        self._index_log(cid, rel, seal["sha256"], taskmap, policy, seed,
                                        expected_records=seal["records"], expected_actions=seal["actions"])

    def _index_log(self, cid, relative, sha, tasks, policy, seed, expected_records=None, expected_actions=None):
        data = self.bound_bytes(relative, sha)
        seen, offset, actions = set(), 0, 0
        for line in data.splitlines(keepends=True):
            record = json.loads(line)
            eid = record["episode_id"]
            if eid in seen or eid not in tasks:
                raise ValueError("重复或未知任务")
            seen.add(eid)
            task = tasks[eid]
            if record["area"] != task["area"] or record["policy"] != policy or record.get("seed", record.get("local_checkpoint_seed")) != seed or record["condition"] != "CueFull":
                raise ValueError("策略/权重/目标条件或区域不一致")
            if record["protocol"] != task["protocol"]:
                raise ValueError("轨迹协议错误")
            validate_record(record, task)
            key = f"{cid}:{policy}:{seed}:{eid}"
            self.entries[key] = {"id": key, "episode_id": eid, "catalog": cid, "area": record["area"],
                "policy": policy, "seed": seed, "success": record["success"], "steps": record["steps"],
                "sg_m": record["sg_m"], "movement_m": record["valid_travel_m"],
                "stratum": record["stratum"], "distance": record["distance"],
                "_file": relative, "_offset": offset, "_length": len(line), "_line_sha": digest(line), "_task": task}
            offset += len(line)
            actions += record["steps"]
        if expected_records is not None and (len(seen) != expected_records or actions != expected_actions):
            raise ValueError("seal分母不符")

    def listing(self, cid: str, policy: str, seed: str, area: str = "", outcome: str = "", query: str = "") -> list:
        if cid not in self.catalogs or policy not in ("M0", "Coverage3Radial") or seed not in ("0", "1", "2"):
            raise ValueError("请选择已登记批次/策略/权重")
        rows = [r for r in self.entries.values() if r["catalog"] == cid and r["policy"] == policy and r["seed"] == int(seed)
                and (not area or r["area"] == area) and (not query or query.lower() in r["episode_id"].lower())
                and (not outcome or r["success"] == (outcome == "success"))]
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    def record(self, eid: str) -> tuple[dict, dict]:
        entry = self.entries[eid]
        with safe_path(self.root, entry["_file"]).open("rb") as stream:
            stream.seek(entry["_offset"])
            line = stream.read(entry["_length"])
        if digest(line) != entry["_line_sha"]:
            raise ValueError("轨迹在启动后变化，请重启核验")
        return entry, json.loads(line)

    def frame(self, eid: str, step: int, diagnostic: bool = False) -> dict:
        entry, record = self.record(eid)
        if not 0 <= step <= record["steps"]:
            raise ValueError("回放步数越界")
        task = entry["_task"]
        route = record["trajectory"][:step + 1]
        decision = record["decisions"][step] if step < record["steps"] else None
        result = {"id": eid, "episode_id": entry["episode_id"], "step": step,
            "total_steps": record["steps"], "grid": task["grid_size"], "budget": task["budget"],
            "cell_size_m": record["cell_size_m"], "current": route[-1]["patch_id"],
            "visited": sorted({r["patch_id"] for r in route}), "route": route,
            "decision": decision, "remaining": task["budget"] - step,
            "movement_m": sum(r["patch_id"] != route[i - 1]["patch_id"] for i, r in enumerate(route) if i) * record["cell_size_m"],
            "terminal": step == record["steps"], "mode": "saved_replay", "truth_is_strategy_input": False,
            "source": {"trajectory": entry["_file"], "sha256": self.bindings[entry["_file"]],
                       "line_sha256": entry["_line_sha"], "report": self.catalogs[entry["catalog"]]["report"]}}
        if result["terminal"]:
            result["result"] = {"success": record["success"], "sg_m": record["sg_m"], "termination": record["termination"]}
        if diagnostic:
            result["diagnostic"] = {"goal": task["goal"], "distance_cells": distance(result["current"], task["goal"], task["grid_size"]), "scope": "事后真值，仅回放展示"}
        return result

    def image(self, eid: str, step: int, cell: str) -> tuple[bytes, str]:
        entry, record = self.record(eid)
        if not 0 <= step <= record["steps"]:
            raise ValueError("回放步数越界")
        if cell == "goal":
            number = entry["_task"]["goal"]
        else:
            number = int(cell)
            if number not in {r["patch_id"] for r in record["trajectory"][:step + 1]}:
                raise ValueError("当前回放步尚未观察该图块")
        relative, sha = self.images[(entry["catalog"], entry["area"], number)]
        data = self.bound_bytes(relative, sha)
        return data, "image/png" if relative.endswith(".png") else "image/jpeg"

    def export(self, eid: str) -> dict:
        entry, record = self.record(eid)
        return {"mode": "saved_replay", "new_navigation": False,
                "provenance": self.frame(eid, 0)["source"], "frozen_task_for_diagnosis_only": entry["_task"],
                "saved_record": record}

    def catalog(self):
        return {"mode": "saved_replay", "catalogs": list(self.catalogs.values()),
                "records": len(self.entries), "bound_files": len(self.bindings),
                "scope": "同模态航拍目标；已封存日志回放；单导航Agent；没有实时导航或新SR"}
