"""Read-only alternate BENCH-021 audit. No model imports or tensor values."""
import ast
import collections
import hashlib
import json
import pickletools
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BATCH = ROOT / "项目导航/原训练资产核验与共同验证准入_v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf8"))


def training_primitives_from_opcodes(data):
    """Independent narrow recognizer, not the collector's stack interpreter.

    Three top-level scalars are followed by scalar opcode then memo of NEXT key.
    Do not extrapolate metadata tokens to weights' actual training provenance.
    """
    ops = list(pickletools.genops(data))
    answer = {}
    for i, (op, value, _) in enumerate(ops):
        if op.name in {"BINUNICODE", "SHORT_BINUNICODE"} and value in {"epoch", "global_step", "pytorch-lightning_version"}:
            j = i+1
            while ops[j][0].name in {"BINPUT", "LONG_BINPUT", "MEMOIZE"}: j += 1
            if ops[j][0].name not in {"BININT", "BININT1", "BININT2", "BINUNICODE", "SHORT_BINUNICODE"}:
                raise AssertionError("not an immediate scalar")
            answer[value] = ops[j][1]
    return answer, dict(collections.Counter(op.name for op, _, _ in ops))


def validate_gate(gate):
    """Evidence readiness only; even all assets never imply scientific success.

    Verified relation claims require an explicitly recorded review, in addition
    to bytes. This checker does NOT establish author attribution on its own.
    """
    required = {"source_split", "encoder_recipe", "feature_bank", "training_run",
                "checkpoint_run_map", "paper_evaluation_map"}
    items = gate.get("requirements", [])
    names = [r.get("id") for r in items]
    if set(names) != required or len(names) != len(required):
        raise ValueError("requirement identity or duplicate")
    missing = []
    for req in items:
        if req.get("status") != "verified":
            missing.append(req["id"]); continue
        if not req.get("artifact_bindings") or not req.get("relation_evidence"):
            raise ValueError("verified claim lacks artifacts and relation evidence")
        for binding in req["artifact_bindings"]:
            p = ROOT / binding["path"]
            if not p.is_file() or digest(p) != binding["sha256"]:
                raise ValueError("asset SHA mismatch: " + binding["path"])
        for ev in req["relation_evidence"]:
            if not ev.get("authority_url") or not ev.get("quotation") or not ev.get("reviewer"):
                raise ValueError("unreviewed provenance relation")
            p = ROOT/ev["path"]
            if not p.is_file() or digest(p) != ev["sha256"]:
                raise ValueError("relation evidence SHA mismatch")
    qualification = gate.get("development_target_qualified") is True
    return {"asset_readiness": not missing, "missing": missing,
            "development_target_qualified": qualification,
            "formal_execution_allowed": not missing and qualification and gate.get("new_source_isolation_verified") is True,
            "does_not_prove_author_attribution_or_target_effect": True}


def audit():
    lock = json_read(BATCH/"元数据/启动登记_v1.json")
    for p, expected in lock["bindings"].items(): assert digest(ROOT/p) == expected, p
    checkpoints = json_read(BATCH/"元数据/checkpoint静态训练记录_v1.json")
    recognized = {}
    for record in checkpoints:
        p = ROOT/record["path"]
        assert digest(p) == record["sha256"] and p.stat().st_size == record["size"]
        with zipfile.ZipFile(p) as z:
            entry = z.getinfo(record["member"])
            assert entry.file_size <= 4*1024**2
            data = z.read(entry)
        assert hashlib.sha256(data).hexdigest() == record["metadata_sha256"]
        scalars, opcodes = training_primitives_from_opcodes(data)
        assert opcodes == record["syntax"]["opcodes"]
        for key, value in scalars.items(): assert record["metadata"][key] == value
        recognized[p.name] = scalars
        assert record["tensor_values_read"] is False and record["deserialization_executed"] is False
    acquisitions = json_read(BATCH/"元数据/采集汇总_v1.json")
    total = 0
    for request in acquisitions["requests"]:
        if "path" in request:
            p = BATCH/request["path"]
            assert digest(p) == request["sha256"] and p.stat().st_size == request["bytes"]
        total += request["bytes"]
    assert total == acquisitions["bytes"] <= lock["caps"]["total_bytes"]
    assert len(acquisitions["requests"]) <= lock["caps"]["requests"]
    source_checks = []
    for source in lock["sources"]:
        tree = ast.parse((ROOT/source).read_text(encoding="utf8"))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else ",".join(a.name for a in node.names)
                if any(x in module for x in ["torch", "transformers"]): forbidden.append(module)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"loads", "load"}:
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "pickle": forbidden.append("pickle execution")
        assert not forbidden, forbidden
        assert digest(BATCH/"源码快照"/Path(source).name) == digest(ROOT/source)
        source_checks.append(source)
    gate = validate_gate(json_read(BATCH/"元数据/资产准入清单_v1.json"))
    assert gate["formal_execution_allowed"] is False
    return {"passed": True, "bindings_checked": len(lock["bindings"]),
            "checkpoint_scalars_alternate": recognized, "network_requests": len(acquisitions["requests"]),
            "network_bytes": total, "source_checks": source_checks, "gate": gate,
            "independent_implementation_same_developer": True, "model_calls": 0,
            "tensor_values_loaded": False, "defaults_changed": False}


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == "check":
        print(json.dumps(validate_gate(json_read(Path(sys.argv[2]))), ensure_ascii=False, indent=2)); sys.exit(0)
    result = audit()
    if sys.argv[1:] == ["--save"]:
        with (BATCH/"核验/另实现复核_v1.json").open("x", encoding="utf8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2); f.write("\n")
    elif sys.argv[1:]: raise SystemExit("usage: audit_original_training_assets_v1.py [--save] or check manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
