"""Independent, read-only EVAL-004 counter/hash audit; does not import the analyzer."""
import ast
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "DATA/processed_data/MassGIS/评测结果"
OUT = BASE / "十五乘十五到达能力漂移诊断_v5"
E2 = BASE / "正式十乘十十五乘十五确认_v1"
E1 = ROOT / "DATA/processed_data/MassGIS/工程准备/正式导航协议冻结_v1"
FEATS = E2 / "特征"
LOGS = E2 / "主对照"
MEAN = ROOT / "DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1/全局拟合均值.npy"
ACTIONS = ("up", "right", "down", "left")
DELTAS = ((-1,0),(0,1),(1,0),(0,-1))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def independently_map(manifests):
    result={}
    for area in {r["area"] for r in manifests[10]["regions"]}:
        a=next(x for x in manifests[10]["regions"] if x["area"]==area)
        b=next(x for x in manifests[15]["regions"] if x["area"]==area)
        lookup={tuple(round(float(v),6) for v in c["bounds_m"]):c for c in b["cells"]}
        mapping={}
        for cell in a["cells"]:
            dest=lookup[tuple(round(float(v),6) for v in cell["bounds_m"])]
            i,j=int(cell["cell"]),int(dest["cell"])
            if i//10!=j//15 or i%10+5!=j%15: raise AssertionError("independent bounds/index mapping failed")
            if cell["file_sha256"]!=dest["file_sha256"]: raise AssertionError("shared payload hash mismatch")
            mapping[i]=j
        result[area]=mapping
    return result


def independent_quality(pos,goal,action,k):
    idx=ACTIONS.index(action); dr,dc=DELTAS[idx]
    r,c=pos; gr,gc=goal; old=abs(r-gr)+abs(c-gc)
    nr,nc=r+dr,c+dc
    if nr<0 or nr>=k or nc<0 or nc>=k: nr,nc=r,c
    new=abs(nr-gr)+abs(nc-gc)
    return "toward" if new<old else "away" if new>old else "side_or_wall"


def reconstruct_sha(mean,current,pos,remaining,visited,k):
    counts=[0]*25
    for idx in visited:
        r,c=divmod(int(idx),k)
        bucket=(r*5//k)*5+(c*5//k)
        counts[bucket]=min(counts[bucket]+1,3)
    state=np.asarray([pos[0]/(k-1),pos[1]/(k-1),remaining/20],dtype=np.float32)
    hist=np.asarray(counts,dtype=np.float32)/3
    vector=np.concatenate((mean,current,state,hist)).astype(np.float32)
    return hashlib.sha256(vector.tobytes()).hexdigest()


def run():
    binding=json.loads((OUT/"输入绑定.json").read_text("utf-8"))
    summary=json.loads((OUT/"诊断汇总.json").read_text("utf-8"))
    seal=json.loads((E2/"核验/阶段封存.json").read_text("utf-8"))
    if len(seal["files_sha256"])!=753 or seal["files"]!=753: raise AssertionError("EVAL-002 seal denominator")
    for name,want in seal["files_sha256"].items():
        if sha(ROOT/name)!=want: raise AssertionError(f"sealed EVAL-002 file changed: {name}")
    seal_inventory={}
    for category in ("主对照","规则基线"):
        nfiles=nrecords=nactions=0
        for path in sorted((E2/category).rglob("*.jsonl")):
            lseal=json.loads(path.with_suffix(".seal.json").read_text("utf-8"))
            if sha(path)!=lseal["sha256"]: raise AssertionError(f"trajectory hash mismatch {path.name}")
            rows=actions=0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    item=json.loads(line);rows+=1;actions+=int(item.get("steps",len(item.get("decisions",[]))))
            if (rows,actions)!=(lseal["records"],lseal["actions"]): raise AssertionError(f"trajectory seal counters mismatch {path.name}")
            nfiles+=1;nrecords+=rows;nactions+=actions
        seal_inventory[category]={"files":nfiles,"records":nrecords,"actions":nactions}
    if seal_inventory!={"主对照":{"files":120,"records":10500,"actions":188246},"规则基线":{"files":40,"records":3500,"actions":66879}}:
        raise AssertionError(f"all trajectory seal inventory mismatch: {seal_inventory}")
    if sum(v["actions"] for v in seal_inventory.values())!=255125: raise AssertionError("full sealed action sum mismatch")
    tasks={k:json.loads((E1/f"元数据/grid{k}任务.json").read_text("utf-8")) for k in (10,15)}
    manifests={k:json.loads((E1/f"工程数据/grid{k}/数据清单.json").read_text("utf-8")) for k in (10,15)}
    mapping=independently_map(manifests)
    region_ids={r["area"]:r["region_id"] for r in manifests[15]["regions"]}
    t10={x["pair_id"]:x for x in tasks[10] if x["cohort"]=="common"}
    t15={x["pair_id"]:x for x in tasks[15] if x["cohort"]=="common"}
    if len(t10)!=750 or set(t10)!=set(t15): raise AssertionError("common pair identity count")
    for pid,a in t10.items():
        b=t15[pid]
        if mapping[a["area"]][int(a["start"])]!=int(b["start"]) or mapping[a["area"]][int(a["goal"])]!=int(b["goal"]):
            raise AssertionError(f"independent endpoint pairing mismatch {pid}")
    means=np.load(MEAN,allow_pickle=False)
    cache={}
    for area in sorted({x["area"] for x in tasks[15]}):
        with np.load(FEATS/f"{area}.npz",allow_pickle=False) as z: cache[area]=z["global_features"]
    first={}
    log_files=records=actions=0
    expected_counts=defaultdict(Counter)
    for path in sorted(LOGS.glob("g*_CueFull_*.jsonl")):
        stem=path.name
        if not (stem.startswith("g10_M0_") or stem.startswith("g10_Coverage3Radial_") or stem.startswith("g15_M0_") or stem.startswith("g15_Coverage3Radial_")): continue
        k=int(stem[1:3]); log_files+=1
        seal_path=path.with_suffix(".seal.json")
        logseal=json.loads(seal_path.read_text("utf-8"))
        if sha(path)!=logseal["sha256"]: raise AssertionError(f"log seal mismatch {stem}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                row=json.loads(line);records+=1; actions+=len(row["decisions"])
                task=next(x for x in tasks[k] if x["episode_id"]==row["episode_id"])
                for d in row["decisions"]:
                    p=tuple(map(int,d["public_position"])); rem=int(d["remaining_budget"]);vis=tuple(d["public_visited"])
                    i=p[0]*k+p[1];i15=i if k==15 else p[0]*15+p[1]+5
                    if reconstruct_sha(means,cache[row["area"]][i15],p,rem,vis,k)!=d["explorer_features_sha256"]:
                        raise AssertionError(f"independent input hash mismatch {row['episode_id']}")
                d=row["decisions"][0]
                record_id=row["pair_id"] if row["pair_id"] is not None else "extra:"+row["episode_id"]
                key=(k,row["policy"],int(row["seed"]),record_id)
                q=independent_quality(tuple(d["public_position"]),divmod(int(task["goal"]),k),d["explorer_action"],k)
                first[key]=(row,task,d,q)
                expected_counts[(k,row["policy"])][q]+=1
    if (log_files,records,actions)!=(120,10500,188246): raise AssertionError(f"independent main denominator {log_files}/{records}/{actions}")
    expected_pairs=set()
    for pid in t10:
        for policy in ("M0","Coverage3Radial"):
            for seed in (0,1,2):
                a=first[(10,policy,seed,pid)];b=first[(15,policy,seed,pid)]
                if a[2]["current_image_sha256"]!=b[2]["current_image_sha256"] or a[2]["target_image_sha256"]!=b[2]["target_image_sha256"]:
                    raise AssertionError("independent first image pairing failed")
                if tuple(a[2]["public_position"])!=(b[2]["public_position"][0],b[2]["public_position"][1]-5):
                    raise AssertionError("independent first state mapping failed")
                expected_pairs.add((policy,seed,pid))
    if len(expected_pairs)!=4500: raise AssertionError("independent first-step pair count")
    details=[]
    with (OUT/"首步配对明细.jsonl").open(encoding="utf-8") as f:
        details=[json.loads(line) for line in f]
    if len(details)!=4500: raise AssertionError("saved first-step detail denominator")
    if any(x["region_id"]!=region_ids[x["area"]] for x in details): raise AssertionError("saved region identity differs from frozen data manifest")
    bykey={(x["grid"],x["policy"],x["seed"],x["pair_id"]):x for x in details}
    quality_pair=defaultdict(Counter); logit_delta=[]
    for pid in t10:
        for policy in ("M0","Coverage3Radial"):
            for seed in (0,1,2):
                a=first[(10,policy,seed,pid)];b=first[(15,policy,seed,pid)]
                x=bykey[(10,policy,seed,pid)];y=bykey[(15,policy,seed,pid)]
                if x["quality10"]!=a[3] or y["quality15"]!=b[3]: raise AssertionError("saved action quality disagrees")
                score={"toward":1,"side_or_wall":0,"away":-1};dv=score[b[3]]-score[a[3]]
                quality_pair[policy]["better" if dv>0 else "worse" if dv<0 else "tie"]+=1
                logit_delta.append([b[2]["explorer_logits"][i]-a[2]["explorer_logits"][i] for i in range(4)])
    for policy in ("M0","Coverage3Radial"):
        got=summary["first_step_pairing"]["proposal_quality"][policy]
        for label in ("better","worse","tie"):
            if got[f"paired_quality_grid15_{label}"]!=quality_pair[policy][label] if label!="tie" else got["paired_quality_tie"]!=quality_pair[policy]["tie"]:
                raise AssertionError("summary paired action quality count mismatch")
    source=Path("project/src/eval/diagnose_massgis_arrival_drift_v5.py").read_text(encoding="utf-8-sig")
    tree=ast.parse(source)
    imported=set();forward_calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imported.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imported.add(node.module.split(".")[0])
        elif isinstance(node,ast.Call):
            name=node.func.attr.lower() if isinstance(node.func,ast.Attribute) else node.func.id.lower() if isinstance(node.func,ast.Name) else ""
            if name in {"forward","predict","act","load_area_policy","step"}: forward_calls.append(name)
    if imported & {"torch","torchvision","transformers"} or forward_calls: raise AssertionError("model runtime/call found in new analyzer source")
    result={"status":"passed","review_scope":"independent reimplementation by same developer; not independent-person review",
        "checks":{"eval2_sealed_files":753,"grid10_tasks":len(tasks[10]),"grid15_tasks":len(tasks[15]),"bounds_mapped_regions":len(mapping),
        "mapped_cells_per_region":100,"common_physical_pairs":len(t10),"first_step_paired_rows":len(details),
        "main_log_files":log_files,"main_records":records,"main_actions":actions,"trajectory_seal_inventory":seal_inventory,"reconstructed_policy_input_hashes":actions,
        "quality_comparisons":dict(quality_pair),"model_forward_calls":0,"training":0,"navigation":0,"network":0,"spares":0},
        "input_binding_sha256":sha(OUT/"输入绑定.json"),"summary_sha256":sha(OUT/"诊断汇总.json")}
    path=OUT/"独立复核.json"
    with path.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True);f.write("\n")
    return result


if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=False))

