"""Independent counters for the frozen EVAL-004 v6 read-only analysis."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/"DATA/processed_data/MassGIS/评测结果"
V6=BASE/"十五乘十五到达能力漂移诊断_v6"
OUT=V6/"核验/审计v3"
E2=BASE/"正式十乘十十五乘十五确认_v1"
E1=ROOT/"DATA/processed_data/MassGIS/工程准备/正式导航协议冻结_v1"
FEATS=E2/"特征"
LOGDIR=E2/"主对照"
MASA=ROOT/"DATA/processed_data/Masa"
TRAIN=MASA/"训练结果/PBRS完整导航与局部匹配对照_v1"
EDGE=MASA/"训练结果/边缘连续性可信线索对照_v1"
SRC=ROOT/"project/src/eval/audit_massgis_arrival_drift_v8.py"
TEST=ROOT/"project/src/tests/test_massgis_arrival_drift_audit_v8.py"
ACTIONS=("up","right","down","left")
DELTA=((-1,0),(0,1),(1,0),(0,-1))
POLICIES=("M0","Coverage3Radial")
SEEDS=(0,1,2)


def jread(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def write_new(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as f:json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True);f.write("\n")
def feature_input_hash(mean,current,pos,remaining,visited,k):
    # Separate implementation: scalar bucket accumulation, not the analyzer's vectorized bincount.
    bins=[0]*25
    for cell in visited:
        r,c=divmod(int(cell),k)
        b=(r*5//k)*5+(c*5//k)
        bins[b]=min(3,bins[b]+1)
    scalar=np.asarray([pos[0]/(k-1),pos[1]/(k-1),remaining/20],dtype=np.float32)
    hist=np.asarray(bins,dtype=np.float32)/3
    payload=np.concatenate((mean,current,scalar,hist)).astype(np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()
def action_class(pos,goal,action,k):
    a=ACTIONS.index(action);dr,dc=DELTA[a];r,c=pos;gr,gc=goal
    old=abs(r-gr)+abs(c-gc);rr,cc=r+dr,c+dc
    if rr<0 or rr>=k or cc<0 or cc>=k:rr,cc=r,c
    new=abs(rr-gr)+abs(cc-gc)
    return "toward" if new<old else "away" if new>old else "side_or_wall"
def pair_detail_index(rows):
    result={}
    for row in rows:
        key=(row["policy"],int(row["seed"]),row["pair_id"])
        if key in result: raise ValueError(f"duplicate paired detail {key}")
        result[key]=row
    return result
def mapped_cells(manifests):
    allmaps={10:{},15:{}}; physical={}
    for k in (10,15):
        for region in manifests[k]["regions"]:
            byid={int(c["cell"]):c for c in region["cells"]}
            bybounds={tuple(round(float(x),6) for x in c["bounds_m"]):int(c["cell"]) for c in region["cells"]}
            if len(byid)!=k*k or len(bybounds)!=k*k:raise ValueError("manifest cell map incomplete")
            allmaps[k][region["area"]]={"region":region,"byid":byid,"bybounds":bybounds}
    for area in allmaps[10]:
        a,b=allmaps[10][area],allmaps[15][area]
        if a["region"]["region_id"]!=b["region"]["region_id"]:raise ValueError("region identity differs")
        mapping={}
        for i,cell in a["byid"].items():
            j=b["bybounds"][tuple(round(float(x),6) for x in cell["bounds_m"])]
            if (i//10,i%10+5)!=(j//15,j%15):raise ValueError("bounds-to-index formula failed")
            if cell["file_sha256"]!=b["byid"][j]["file_sha256"]:raise ValueError("overlap patch hashes differ")
            mapping[i]=j
        physical[area]=mapping
    return allmaps,physical
def distance_change(pos,goal,action,k):
    return action_class(pos,goal,action,k)


def freeze():
    if OUT.exists():raise FileExistsError(f"refusing to overwrite audit output: {OUT}")
    binding=jread(V6/"输入绑定.json")
    outputs={name:sha(V6/name) for name in ("输入绑定.json","诊断汇总.json","诊断报告.md","首步配对明细.jsonl","路径分叉明细.jsonl","下一项单因素方案草案.md")}
    for name,expected in outputs.items():
        if not (V6/name).is_file():raise FileNotFoundError(V6/name)
    sealed=binding.get("eval2_sealed_files_sha256")
    if not isinstance(sealed,dict) or len(sealed)!=753:raise ValueError("v6 binding lacks 753-file EVAL-002 stage seal map")
    for name,want in sealed.items():
        if sha(ROOT/name)!=want:raise ValueError(f"EVAL-002 sealed file differs before audit freeze: {name}")
    OUT.mkdir(parents=True);snap=OUT/"源码快照";snap.mkdir()
    for src in (SRC,TEST): (snap/src.name).write_bytes(src.read_bytes())
    audit_sources={rel(snap/p.name):sha(snap/p.name) for p in (SRC,TEST)}
    audit_binding={"version":"massgis-arrival-drift-independent-audit-binding-v3","date":"2026-10-05",
        "main_analysis_binding_sha256":sha(V6/"输入绑定.json"),"main_analysis_outputs_sha256":outputs,
        "eval2_stage_seal_sha256":binding["eval2_stage_seal_sha256"],
        "eval2_sealed_files_sha256":binding["eval2_sealed_files_sha256"],
        "main_input_sha256":binding["input_sha256"],"audit_source_snapshot_sha256":audit_sources,
        "audit_limits":{"model_forward":0,"training":0,"navigation":0,"network":0,"spares":0}}
    write_new(OUT/"审计绑定.json",audit_binding)
    print(json.dumps({"frozen":True,"audit_out":rel(OUT),"source_snapshot":audit_sources},ensure_ascii=False))


def verify_binding(binding):
    if sha(V6/"输入绑定.json")!=binding["main_analysis_binding_sha256"]:raise ValueError("main v6 binding changed")
    for name,want in binding["main_analysis_outputs_sha256"].items():
        if sha(V6/name)!=want:raise ValueError(f"main result changed: {name}")
    seal_path=E2/"核验/阶段封存.json"
    if sha(seal_path)!=binding["eval2_stage_seal_sha256"]:raise ValueError("EVAL-002 stage seal changed")
    for name,want in binding["eval2_sealed_files_sha256"].items():
        if sha(ROOT/name)!=want:raise ValueError(f"EVAL-002 sealed file changed: {name}")
    for name,want in binding["main_input_sha256"].items():
        if name.startswith(("EVAL2-stage:","EVAL2-source:","EVAL2-input:")):continue
        if sha(ROOT/name)!=want:raise ValueError(f"main analysis input changed: {name}")
    for name,want in binding["audit_source_snapshot_sha256"].items():
        p=ROOT/name
        live=TEST if Path(name).name.startswith("test_") else SRC
        if sha(p)!=want or sha(live)!=want:raise ValueError(f"audit source snapshot differs: {name}")


def run_audit():
    audit_binding=jread(OUT/"审计绑定.json");verify_binding(audit_binding)
    summary=jread(V6/"诊断汇总.json")
    tasks={k:jread(E1/f"元数据/grid{k}任务.json") for k in (10,15)}
    manifests={k:jread(E1/f"工程数据/grid{k}/数据清单.json") for k in (10,15)}
    maps,cellmap=mapped_cells(manifests)
    region_ids={r["area"]:r["region_id"] for r in manifests[15]["regions"]}
    eval2seal=jread(E2/"核验/阶段封存.json")
    if len(eval2seal["files_sha256"])!=753:raise AssertionError("sealed file count")
    trajectory_counts={}
    for cat,expected in (("主对照",(120,10500,188246)),("规则基线",(40,3500,66879))):
        nf=nr=na=0
        folder=E2/cat
        for path in sorted(folder.rglob("*.jsonl")):
            seal=jread(path.with_suffix(".seal.json"))
            if sha(path)!=seal["sha256"]:raise AssertionError(f"trajectory SHA mismatch {path.name}")
            rcount=acount=0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row=json.loads(line);rcount+=1;acount+=int(row.get("steps",len(row.get("decisions",[]))))
            if (rcount,acount)!=(seal["records"],seal["actions"]):raise AssertionError(f"trajectory counters mismatch {path.name}")
            nf+=1;nr+=rcount;na+=acount
        trajectory_counts[cat]={"files":nf,"records":nr,"actions":na}
        if (nf,nr,na)!=expected:raise AssertionError(f"{cat} denominator mismatch: {trajectory_counts[cat]}")
    if sum(x["actions"] for x in trajectory_counts.values())!=255125:raise AssertionError("total actions across main/rules differs")

    mean=np.load(EDGE/"全局拟合均值.npy",allow_pickle=False)
    cache={}
    for area in sorted(maps[15]):
        with np.load(FEATS/f"{area}.npz",allow_pickle=False) as z:cache[area]=z["global_features"]
    episode_task={k:{x["episode_id"]:x for x in rows} for k,rows in tasks.items()}
    records={};first={}
    all_success=defaultdict(Counter);cohort_success=defaultdict(Counter);region_success=defaultdict(Counter);layer_success=defaultdict(Counter)
    file_count=record_count=action_count=0
    for path in sorted(LOGDIR.glob("g*_CueFull_*.jsonl")):
        name=path.name
        if not (name.startswith("g10_M0_") or name.startswith("g10_Coverage3Radial_") or name.startswith("g15_M0_") or name.startswith("g15_Coverage3Radial_")):continue
        k=int(name[1:3]); file_count+=1
        seal=jread(path.with_suffix(".seal.json"))
        if sha(path)!=seal["sha256"]:raise AssertionError(f"main trajectory SHA mismatch {name}")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row=json.loads(line);record_count+=1;action_count+=len(row["decisions"])
                task=episode_task[k][row["episode_id"]]
                pair=row["pair_id"] if row["pair_id"] is not None else "extra:"+row["episode_id"]
                key=(k,row["policy"],int(row["seed"]),pair)
                if key in records:raise AssertionError(f"duplicate main record {key}")
                records[key]={"row":row,"task":task}
                all_success[(k,row["policy"])]["n"]+=1;all_success[(k,row["policy"])]["success"]+=int(row["success"])
                cohort_success[(k,row["policy"],task["cohort"])]["n"]+=1;cohort_success[(k,row["policy"],task["cohort"])]["success"]+=int(row["success"])
                if task["cohort"]=="common":
                    region_success[(k,row["policy"],task["area"])]["n"]+=1;region_success[(k,row["policy"],task["area"])]["success"]+=int(row["success"])
                    layer_success[(k,row["policy"],task["stratum"])]["n"]+=1;layer_success[(k,row["policy"],task["stratum"])]["success"]+=int(row["success"])
                for i,d in enumerate(row["decisions"]):
                    pos=tuple(map(int,d["public_position"]));visited=tuple(d["public_visited"]);rem=int(d["remaining_budget"])
                    cell=pos[0]*k+pos[1];cell15=cell if k==15 else cellmap[row["area"]][cell]
                    rebuilt=feature_input_hash(mean,cache[row["area"]][cell15],pos,rem,visited,k)
                    if rebuilt!=d["explorer_features_sha256"]:raise AssertionError(f"decision input SHA mismatch {row['episode_id']}/{i}")
                if task["cohort"]=="common":
                    d=row["decisions"][0]; keyfirst=(k,row["policy"],int(row["seed"]),task["pair_id"])
                    first[keyfirst]=(row,task,d,action_class(tuple(d["public_position"]),divmod(int(task["goal"]),k),d["explorer_action"],k))
    if (file_count,record_count,action_count)!=(120,10500,188246):raise AssertionError("independent main total mismatch")
    t10={x["pair_id"]:x for x in tasks[10] if x["cohort"]=="common"};t15={x["pair_id"]:x for x in tasks[15] if x["cohort"]=="common"}
    if len(t10)!=750 or set(t10)!=set(t15):raise AssertionError("common physical task pair set")
    details=[]
    with (V6/"首步配对明细.jsonl").open(encoding="utf-8") as stream:
        for line in stream:details.append(json.loads(line))
    by_detail=pair_detail_index(details)
    if len(details)!=4500 or len(by_detail)!=4500:raise AssertionError("paired detail count/identity")
    quality=defaultdict(Counter);mismatch=0;bucketdiff=set();drs=[];dcs=[]
    for pid,a in t10.items():
        b=t15[pid];area=a["area"];mp=cellmap[area]
        if a["area"]!=b["area"] or mp[int(a["start"])]!=int(b["start"]) or mp[int(a["goal"])]!=int(b["goal"]):raise AssertionError(f"task endpoint pairing {pid}")
        start10=divmod(int(a["start"]),10);start15=divmod(int(b["start"]),15)
        bucket10=(start10[0]*5//10)*5+start10[1]*5//10;bucket15=(start15[0]*5//15)*5+start15[1]*5//15
        if bucket10!=bucket15:bucketdiff.add(pid)
        for policy in POLICIES:
            for seed in SEEDS:
                one=first[(10,policy,seed,pid)];two=first[(15,policy,seed,pid)]
                row10,task10,d10,q10=one;row15,task15,d15,q15=two
                record=by_detail[(policy,seed,pid)]
                if d10["current_image_sha256"]!=d15["current_image_sha256"] or d10["target_image_sha256"]!=d15["target_image_sha256"]:raise AssertionError("paired image payload mismatch")
                if record["region_id"]!=region_ids[area]:raise AssertionError("paired detail region id mismatch")
                if record["quality10"]!=q10 or record["quality15"]!=q15:raise AssertionError("paired posthoc direction quality mismatch")
                if record["logits10"]!=d10["explorer_logits"] or record["logits15"]!=d15["explorer_logits"]:raise AssertionError("logit detail differs from saved first row")
                if record["proposal10"]!=d10["explorer_action"] or record["proposal15"]!=d15["explorer_action"]:raise AssertionError("proposal detail mismatch")
                rr,cc=start10
                wantdr=rr/14-rr/9;wantdc=(cc+5)/14-cc/9
                if abs(record["normalized_coordinate_delta"][0]-wantdr)>1e-12 or abs(record["normalized_coordinate_delta"][1]-wantdc)>1e-12:raise AssertionError("coordinate shift sign/formula")
                mismatch+=int(d10["explorer_action"]!=d15["explorer_action"])
                quality[policy][10][q10]+=1;quality[policy][15][q15]+=1
                score={"toward":1,"side_or_wall":0,"away":-1};delta=score[q15]-score[q10]
                quality[policy]["pair"]["better" if delta>0 else "worse" if delta<0 else "tie"]+=1
                drs.append(wantdr);dcs.append(wantdc)
    if mismatch/4500!=summary["first_step_pairing"]["proposal_mismatch_rate"]:raise AssertionError("proposal mismatch rate")
    if len(bucketdiff)/750!=summary["first_step_pairing"]["start_bucket_mismatch_rate"]:raise AssertionError("bucket mismatch rate")
    for policy in POLICIES:
        saved=summary["first_step_pairing"]["proposal_quality"][policy]
        for grid in (10,15):
            for label in ("toward","away","side_or_wall"):
                if saved[f"grid{grid}"][label]!=quality[policy][grid][label]:raise AssertionError("direction category count")
        for name,label in (("paired_quality_grid15_better","better"),("paired_quality_grid15_worse","worse"),("paired_quality_tie","tie")):
            if saved[name]!=quality[policy]["pair"][label]:raise AssertionError("paired direction comparison")
    reach=summary["reachability"]["all_tasks_by_grid_policy"]
    for (k,p),v in all_success.items():
        saved=reach[f"g{k}_{p}_all_tasks"]
        if (saved["successes"],saved["planned_task_weight_runs"])!=(v["success"],v["n"]):raise AssertionError("planned SR numerator/denominator")
    reach_detail=summary["reachability"]
    for (k,p,cohort),v in cohort_success.items():
        saved=reach_detail["common_task_cohort"][f"g{k}_{p}_{cohort}"]
        if (saved["successes"],saved["planned_task_weight_runs"])!=(v["success"],v["n"]):raise AssertionError("cohort SR numerator/denominator")
    for (k,p,area),v in region_success.items():
        saved=reach_detail["common_tasks_by_region"][f"g{k}_{p}_{area}"]
        if (saved["successes"],saved["planned_task_weight_runs"])!=(v["success"],v["n"]):raise AssertionError("region SR numerator/denominator")
    for (k,p,layer),v in layer_success.items():
        saved=reach_detail["common_tasks_by_stratum"][f"g{k}_{p}_{layer}"]
        if (saved["successes"],saved["planned_task_weight_runs"])!=(v["success"],v["n"]):raise AssertionError("stratum SR numerator/denominator")
    # Recalculate logit/coordinate correlations independently from the saved paired rows.
    for i,name in enumerate(ACTIONS):
        dl=[x["logits15"][i]-x["logits10"][i] for x in details]
        dr=[x["normalized_coordinate_delta"][0] for x in details]
        dc=[x["normalized_coordinate_delta"][1] for x in details]
        row_summary=summary["first_step_pairing"]["delta_logit_coordinate_pearson"][name]
        rrow=float(np.corrcoef(dl,dr)[0,1]);rcol=float(np.corrcoef(dl,dc)[0,1])
        if abs(rrow-row_summary["delta_logit_vs_row_delta_pearson"])>1e-10 or abs(rcol-row_summary["delta_logit_vs_column_delta_pearson"])>1e-10:
            raise AssertionError("logit/coordinate correlation mismatch")
    fork_categories=Counter();fork_steps=[]
    for pid,(a,b) in {key:(t10[key],t15[key]) for key in t10}.items():
        for pol in POLICIES:
            for seed in SEEDS:
                ra=records[(10,pol,seed,pid)]["row"];rb=records[(15,pol,seed,pid)]["row"]
                path_a=[tuple(maps[10][ra["area"]]["byid"][int(step["patch_id"])]["bounds_m"][j] for j in (0,1,2,3)) for step in ra["trajectory"]]
                path_b=[tuple(maps[15][rb["area"]]["byid"][int(step["patch_id"])]["bounds_m"][j] for j in (0,1,2,3)) for step in rb["trajectory"]]
                n=0
                while n<min(len(path_a),len(path_b)) and path_a[n]==path_b[n]:n+=1
                if n==len(path_a)==len(path_b):fork_categories["identical_route"]+=1
                elif n<min(len(path_a),len(path_b)):fork_categories["forked"]+=1;fork_steps.append(n)
                else:fork_categories["one_route_ended_first"]+=1;fork_steps.append(n)
    fork_summary=summary["trajectory_fork"]
    if dict(fork_categories)!=fork_summary["route_categories"]:raise AssertionError("route fork category counts")
    fork_dist=fork_summary["first_different_position_index"]
    if len(fork_steps)!=fork_dist["n"]:raise AssertionError("route fork step denominator")
    if abs(float(np.mean(fork_steps))-fork_dist["mean"])>1e-12 or abs(float(np.quantile(fork_steps,.5))-fork_dist["p50"])>1e-12 or abs(float(np.quantile(fork_steps,.9))-fork_dist["p90"])>1e-12:
        raise AssertionError("route fork step distribution")
    audit_src=SRC.read_text(encoding="utf-8-sig");tree=ast.parse(audit_src);imports=set();calls=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split(".")[0])
        elif isinstance(node,ast.Call):
            name=node.func.attr.lower() if isinstance(node.func,ast.Attribute) else node.func.id.lower() if isinstance(node.func,ast.Name) else ""
            if name in {"forward","predict","act","load_area_policy","step"}:calls.add(name)
    if imports&{"torch","torchvision","transformers"} or calls:raise AssertionError("model runtime/call in audit source")
    result={"status":"passed","review_scope":"separate implementation by same developer; not independent-person review",
        "checks":{"eval2_stage_files":753,"all_trajectory_seals":trajectory_counts,"main_input_hashes_reconstructed":action_count,
            "common_physical_tasks":750,"bounds_mapped_regions":10,"shared_cells_per_region":100,"first_step_pair_records":len(details),
            "direction_counts_by_policy":{p:{k:{q:int(quality[p][k][q]) for q in ("toward","away","side_or_wall")} for k in (10,15)} for p in POLICIES},
            "route_categories":dict(fork_categories),"planned_success_counts":{f"g{k}_{p}":{"n":x["n"],"success":x["success"]} for (k,p),x in all_success.items()},
            "model_forward":0,"training":0,"navigation":0,"network":0,"spares":0},
        "main_analysis_binding_sha256":sha(V6/"输入绑定.json"),"diagnostic_summary_sha256":sha(V6/"诊断汇总.json")}
    write_new(OUT/"独立复核_v3.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument("--freeze",action="store_true");modes.add_argument("--audit",action="store_true")
    args=parser.parse_args()
    if args.freeze:freeze()
    else:print(json.dumps(run_audit(),ensure_ascii=False))
if __name__=="__main__":main()

