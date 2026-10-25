"""BENCH-021: bounded public metadata and non-executing checkpoint inspection.

No torch, pickle.loads, tensor storage reads, model requests, or git mutation.
Outputs are exclusive-create. collect may not be restarted; verify is read-only.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pickletools
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BATCH = ROOT / "项目导航/原训练资产核验与共同验证准入_v1"
MAX_PICKLE = 4 * 1024**2
CAPS = {"requests": 24, "response_bytes": 4 * 1024**2,
        "total_bytes": 16 * 1024**2, "seconds": 1200}
GOMAA = "1faa1eee1608529eae95cc0af1d76330c9e46e60"
GEO = "2d9fc71a7151ec4060950cc6ae639880a74f6201"
HF = {"ppo": ("Gomaa-Geo/gomma-geo-falcon", "ccbd2b734b6e64369dba6c21c89aa4364d32b716"),
      "pretrain": ("Gomaa-Geo/falcon-pretrain", "7c24d04050cfe57c41f268b2356a0b3850cb2175"),
      "dataset": ("Gomaa-Geo/MM-GAG", "a0d6b12df62dda5c6dac29a00fa27f0fb8672b57")}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def put(path: Path, value) -> None:
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


class Symbol:
    """Inert marker. Never resolve/import/call the pickle's global functions."""
    def __init__(self, name, args=None):
        self.name, self.args = name, args


def static_pickle(data: bytes):
    """Interpret only container/primitive opcodes and THREE inert torch symbols.

    Unknown opcodes/globals are rejected; REDUCE never executes a callable.
    BUILD discards only OrderedDict tensor metadata, not training metadata.
    """
    if len(data) > MAX_PICKLE:
        raise ValueError("pickle metadata cap")
    stack, memo, mark = [], {}, object()
    allowed = {"collections OrderedDict", "torch FloatStorage",
               "torch._utils _rebuild_tensor_v2"}
    globals_seen, counts = set(), collections.Counter()

    def marked():
        idx = len(stack) - 1
        while idx >= 0 and stack[idx] is not mark:
            idx -= 1
        if idx < 0:
            raise ValueError("missing MARK")
        vals = stack[idx + 1:]
        del stack[idx:]
        return vals

    for op, arg, pos in pickletools.genops(data):
        n = op.name
        counts[n] += 1
        if n in {"PROTO", "FRAME"}:
            continue
        if n == "MARK": stack.append(mark)
        elif n in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BINUNICODE8",
                   "BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT", "BINFLOAT"}:
            stack.append(arg)
        elif n == "NONE": stack.append(None)
        elif n in {"NEWTRUE", "NEWFALSE"}: stack.append(n == "NEWTRUE")
        elif n == "EMPTY_DICT": stack.append({})
        elif n == "EMPTY_LIST": stack.append([])
        elif n == "EMPTY_TUPLE": stack.append(())
        elif n in {"BINPUT", "LONG_BINPUT", "PUT"}: memo[int(arg)] = stack[-1]
        elif n == "MEMOIZE": memo[len(memo)] = stack[-1]
        elif n in {"BINGET", "LONG_BINGET", "GET"}: stack.append(memo[int(arg)])
        elif n == "GLOBAL":
            if arg not in allowed: raise ValueError("unsupported GLOBAL: " + arg)
            globals_seen.add(arg); stack.append(Symbol(arg))
        elif n == "TUPLE": stack.append(tuple(marked()))
        elif n in {"TUPLE1", "TUPLE2", "TUPLE3"}:
            size = int(n[-1]); vals = tuple(stack[-size:]); del stack[-size:]; stack.append(vals)
        elif n in {"SETITEM", "SETITEMS"}:
            if n == "SETITEM":
                v, k = stack.pop(), stack.pop(); stack[-1][k] = v
            else:
                vals = marked()
                if len(vals) % 2: raise ValueError("odd SETITEMS")
                for i in range(0, len(vals), 2): stack[-1][vals[i]] = vals[i+1]
        elif n == "APPEND":
            v = stack.pop(); stack[-1].append(v)
        elif n == "APPENDS":
            vals = marked(); stack[-1].extend(vals)
        elif n == "BINPERSID": stack.append(Symbol("storage", stack.pop()))
        elif n == "REDUCE":
            args, function = stack.pop(), stack.pop()
            if not isinstance(function, Symbol): raise ValueError("non-symbol REDUCE")
            if function.name == "collections OrderedDict" and args == ():
                stack.append({})
            elif function.name == "torch._utils _rebuild_tensor_v2":
                stack.append(Symbol("tensor", args))
            else: raise ValueError("unsupported inert REDUCE: " + function.name)
        elif n == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], dict) or set(state) != {"_metadata"}:
                raise ValueError("unsupported BUILD")
        elif n == "STOP":
            if len(stack) != 1 or pos != len(data)-1: raise ValueError("invalid STOP")
            return stack[0], {"globals": sorted(globals_seen), "opcodes": dict(counts)}
        else: raise ValueError("unsupported opcode: " + n)
    raise ValueError("missing STOP")


def primitive(value, depth=0):
    """Retain primitive container values. Tensors are explicitly omitted."""
    if depth > 32: raise ValueError("metadata nesting cap")
    if isinstance(value, Symbol): return {"omitted": value.name}
    if isinstance(value, dict):
        return {str(k): primitive(v, depth+1) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(v, depth+1) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise ValueError("unsupported primitive")


def inspect_checkpoint(path: Path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        candidates = [i for i in entries if i.filename.endswith("/data.pkl")]
        if len(candidates) != 1 or candidates[0].file_size > MAX_PICKLE:
            raise ValueError("metadata member identity/cap")
        info = candidates[0]
        data = archive.read(info)
    obj, syntax = static_pickle(data)
    if not isinstance(obj, dict): raise ValueError("checkpoint root type")
    metadata = {}
    for k in ("epoch", "global_step", "pytorch-lightning_version", "loops", "callbacks", "lr_schedulers"):
        if k in obj: metadata[k] = primitive(obj[k])
    if "optimizer_states" in obj:
        metadata["optimizer_param_groups"] = [primitive(o.get("param_groups", [])) for o in obj["optimizer_states"]]
    return {"path": path.relative_to(ROOT).as_posix(), "size": path.stat().st_size,
            "sha256": sha(path), "member": info.filename, "metadata_bytes": len(data),
            "metadata_sha256": hashlib.sha256(data).hexdigest(),
            "archive_entries": len(entries), "top_keys_count": len(obj),
            "training_top_keys": [k for k in obj if k in metadata or k in {"state_dict", "optimizer_states", "hyper_parameters"}],
            "hyperparameters_present": "hyper_parameters" in obj,
            "metadata": metadata, "syntax": syntax, "tensor_values_read": False,
            "deserialization_executed": False}


def git_read(repo: Path, args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            encoding="utf8", errors="replace", timeout=30)
    return {"repo": repo.relative_to(ROOT).as_posix(), "args": args,
            "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr}


def collect():
    lock = BATCH / "元数据/启动登记_v1.json"
    start = time.monotonic()
    protected = {}
    for parent in ["项目导航/GOMAA权重静态资格_v1", "DATA/processed_data/Masa/评测结果/GOMAA开发闭环目标诊断_v1", "项目导航/学习基线来源与编码器补证据_v1"]:
        for path in (ROOT / parent).rglob("*"):
            if path.is_file(): protected[path.relative_to(ROOT).as_posix()] = sha(path)
    for p in ["project/local_policy_default.json", "project/local_policy_defaults_v1.json",
              "project/local_policy_default_coverage_v1.json"]:
        if (ROOT/p).is_file(): protected[p] = sha(ROOT/p)
    sources = ["project/src/tools/original_training_assets_v1.py",
               "project/src/eval/audit_original_training_assets_v1.py",
               "project/src/tests/test_original_training_assets_v1.py"]
    for p in sources:
        protected[p] = sha(ROOT/p)
        (BATCH/"源码快照"/Path(p).name).write_bytes((ROOT/p).read_bytes())
    put(lock, {"task": "BENCH-021", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "caps": CAPS, "bindings": protected, "sources": sources,
               "model_forward": False, "training": False, "navigation": False,
               "author_contact": False, "new_sources": 0, "reserve_consumption": 0})
    records = [inspect_checkpoint(p) for p in sorted((ROOT/"models/gomaa_public_v1").glob("*")) if p.is_file()]
    put(BATCH/"元数据/checkpoint静态训练记录_v1.json", records)
    for r in records:
        print(r["path"], "metadata", r["metadata_bytes"], "top", r["training_top_keys"])
    git_records = []
    for repo, revision in [(ROOT/"GOMAA-Geo", GOMAA), (ROOT/"project/.runtime/geoexplorer_probe_v1/source", GEO)]:
        for args in [["rev-parse", "--is-shallow-repository"], ["show-ref"],
                     ["ls-tree", "-r", revision], ["log", "--all", "--format=%H %aI %s", "--name-only"]]:
            git_records.append(git_read(repo, args))
    put(BATCH/"元数据/本地官方Git只读记录_v1.json", git_records)
    requests = []
    for name, (repo, revision) in HF.items():
        kind = "datasets" if name == "dataset" else "models"
        requests.append((f"hf_{name}_history", f"https://huggingface.co/api/{kind}/{repo}/commits/{revision}"))
        requests.append((f"hf_{name}_tree", f"https://huggingface.co/api/{kind}/{repo}/tree/{revision}?recursive=true&expand=false&limit=1000"))
    for name, repo, revision in [("gomaa", "mvrl/GOMAA-Geo", GOMAA), ("geo", "limirs/GeoExplorer", GEO)]:
        for label, suffix in [("tree", f"git/trees/{revision}?recursive=1"), ("branches", "branches?per_page=100"), ("tags", "tags?per_page=100"), ("releases", "releases?per_page=100")]:
            requests.append((f"gh_{name}_{label}", f"https://api.github.com/repos/{repo}/{suffix}"))
    receipts, total = [], 0
    # No automatic retries or speculative large downloads. Every failure is preserved.
    for index, (label, url) in enumerate(requests):
        if index >= CAPS["requests"] or time.monotonic()-start >= CAPS["seconds"]:
            raise RuntimeError("request/time cap")
        t = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": "GeoNav-BENCH021-metadata/1.0"})
        limit = min(CAPS["response_bytes"], CAPS["total_bytes"] - total)
        if limit <= 0: raise RuntimeError("total bytes cap")
        try:
            response = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            response = e
        except (urllib.error.URLError, TimeoutError) as e:
            receipt = {"index": index, "label": label, "url": url, "status": None, "error": type(e).__name__, "bytes": 0}
            put(BATCH/"公开来源"/f"request_{index:02d}.json", receipt); receipts.append(receipt); print(label, receipt); continue
        with response:
            data = response.read(limit+1)
            if len(data) > limit: raise RuntimeError("response bytes cap; stop without claiming completeness")
            target = BATCH/"公开来源"/(label+".json")
            with target.open("xb") as f: f.write(data)
            total += len(data)
            receipt = {"index": index, "label": label, "url": url,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "status": response.status if hasattr(response, "status") else response.code,
                       "final_url": response.geturl(), "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "path": target.relative_to(BATCH).as_posix(),
                       "link": response.headers.get("Link"), "elapsed": time.monotonic()-t}
            put(BATCH/"公开来源"/f"request_{index:02d}.json", receipt)
            receipts.append(receipt); print(label, receipt["status"], len(data), flush=True)
    put(BATCH/"元数据/采集汇总_v1.json", {"requests": receipts, "bytes": total, "elapsed": time.monotonic()-start})


if __name__ == "__main__":
    if sys.argv[1:] != ["collect"]: raise SystemExit("usage: original_training_assets_v1.py collect (once)")
    collect()
