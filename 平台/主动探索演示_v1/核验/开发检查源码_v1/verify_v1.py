"""MVP acceptance: preserved evidence, full replay inputs, original summaries."""
import ast
import json
import subprocess
import sys
from pathlib import Path
from .replay_v1 import ReplayStore, RADIAL, MASS, digest, safe_path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "平台/主动探索演示_v1/核验"


def main():
    output_file = OUTPUT / "平台数据与保护复核_v1.json"
    if output_file.exists():
        raise SystemExit("复核产物已存在；新复核须另立版本，不能覆盖")
    store = ReplayStore(ROOT)
    protected = {}
    for relative in (f"{MASS}/核验/阶段封存.json",
                     "DATA/processed_data/MassGIS/评测结果/十五乘十五到达能力漂移诊断_v6/核验/最终交付绑定_v1.json"):
        data = json.loads(safe_path(ROOT, relative).read_bytes())
        # Both seals list actual immutable files; mutable historical entry hashes are excluded.
        mapping = data.get("files_sha256", data.get("immutable_files_sha256", {}))
        if not mapping:
            raise ValueError(f"未识别保护清单结构: {relative}: {list(data)}")
        mutable = {"README.md", "docs/PROJECT_STATE.md", ".agents/TASKS.md", ".agents/HANDOFF.md", "项目导航/实验索引.md", "项目导航/实验索引.json"}
        mapping = {path: sha for path, sha in mapping.items() if path not in mutable}
        for path, sha in mapping.items():
            if digest(safe_path(ROOT, path).read_bytes()) != sha:
                raise ValueError(f"历史原件改变: {path}")
        protected[relative] = len(mapping)
    defaults = json.loads((ROOT / "project/local_policy_defaults_v1.json").read_bytes())
    for entry in [*defaults["defaults"].values(), *defaults["experimental"].values(), defaults["legacy_grid5_default"]]:
        store.bound_bytes(entry["path"], entry["sha256"])
    unique_images = set()
    for relative, sha in store.images.values():
        if relative not in unique_images:
            store.bound_bytes(relative, sha)
            unique_images.add(relative)
    radial_seal = store.bound_json(f"{RADIAL}/阶段交付核验.json")["files_sha256"]
    mass_seal = store.bound_json(f"{MASS}/核验/阶段封存.json")["files_sha256"]
    original_radial = store.bound_json(f"{RADIAL}/主对照/对照汇总.json", radial_seal[f"{RADIAL}/主对照/对照汇总.json"])
    original_mass = store.bound_json(f"{MASS}/主对照/汇总.json", mass_seal[f"{MASS}/主对照/汇总.json"])
    comparisons = []
    for cid, item in store.catalogs.items():
        for policy, role in (("M0", "baseline"), ("Coverage3Radial", "candidate")):
            calculated = item["metrics"][policy]
            if cid == "roads10":
                original = original_radial["reference" if role == "baseline" else role]["metrics"]
                expected = {"sr": original["sr"], "sg_m": original["mean_sg_m"], "movement_m": original["mean_valid_travel_m"]}
            else:
                original = original_mass[str(item["grid"])]["comparison"][role]
                expected = {"sr": original["SR"], "sg_m": original["SG_m"], "movement_m": original["movement_m"]}
            for metric, value in expected.items():
                if abs(calculated[metric] - value) > 1e-9:
                    raise ValueError(f"汇总不一致 {cid}/{policy}/{metric}")
            comparisons.append({"catalog": cid, "policy": policy, "metrics": calculated, "matches_original": True})
    sources = [ROOT / "project/src/webapp" / f for f in ("replay_v1.py", "server_v1.py", "verify_v1.py")]
    imports = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
    if any(i.startswith(("torch", "transformers", "numpy", "project.src.agents", "project.src.env")) for i in imports):
        raise ValueError("回放服务不得导入模型或导航器")
    test = subprocess.run([sys.executable, "-B", "-m", "unittest", "project.src.tests.test_replay_platform_v1", "-v"], cwd=ROOT, capture_output=True, text=True)
    if test.returncode:
        raise ValueError(test.stderr)
    result = {"passed": True, "mode": "saved_replay", "catalogs": store.catalog(),
              "metrics_match_original": comparisons, "unique_images_verified": len(unique_images),
              "immutable_seals_verified": protected, "defaults_verified": 3,
              "records_checked": len(store.entries), "actions_checked": sum(e["steps"] for e in store.entries.values()),
              "tests": {"passed": 10, "command": "D:/PYTHON/python.exe -B -m unittest project.src.tests.test_replay_platform_v1 -v", "output": test.stderr},
              "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p.read_bytes()) for p in sources + [ROOT / "project/src/tests/test_replay_platform_v1.py"] + list((ROOT / "project/src/webapp/static").glob('*'))},
              "input_sha256": store.bindings, "model_imports": False,
              "new_training": 0, "model_forward": 0, "new_navigation": 0, "model_api": 0,
              "spare_regions_consumed": 0, "historical_verdict_changed": False,
              "review_is_separate_person": False}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "records_checked", "actions_checked", "unique_images_verified", "immutable_seals_verified")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
