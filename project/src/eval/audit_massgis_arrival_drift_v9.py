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
OUT=V6/"核验/审计v4"
E2=BASE/"正式十乘十十五乘十五确认_v1"
E1=ROOT/"DATA/processed_data/MassGIS/工程准备/正式导航协议冻结_v1"
FEATS=E2/"特征"
LOGDIR=E2/"主对照"
MASA=ROOT/"DATA/processed_data/Masa"
TRAIN=MASA/"训练结果/PBRS完整导航与局部匹配对照_v1"
EDGE=MASA/"训练结果/边缘连续性可信线索对照_v1"
SRC=ROOT/"project/src/eval/audit_massgis_arrival_drift_v9.py"
TEST=ROOT/"project/src/tests/test_massgis_arrival_drift_audit_v9.py"
ACTIONS=("up","right","down","left")
DELTA=((-1,0),(0,1),(1,0),(0,-1))
POLICIES=("M0","Coverage3Radial")
SEEDS=(0,1,2)
VIRTUAL_PREFIXES=("EVAL2-sealed:","EVAL2-source:","EVAL2-input:")


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
def real_input_entries(entries):
    return {name:want for name,want in entries.items() if not name.startswith(VIRTUAL_PREFIXES)}
def logical_input_entries(entries,prefix):
    return {name[len(prefix):]:want for name,want in entries.items() if name.startswith(prefix)}
def verify_hash_map(entries,label):
    for name,want in entries.items():
        path=ROOT/name
        if not path.is_file() or sha(path)!=want:raise ValueError(f"{label} hash mismatch: {name}")
def verify_v6_input_maps(binding):
    sealed=binding.get("eval2_sealed_files_sha256")
    frozen=binding.get("eval2_frozen_sources_sha256")
    selected=binding.get("eval2_selected_inputs_sha256")
    inputs=binding.get("input_sha256")
    if not isinstance(sealed,dict) or len(sealed)!=753:raise ValueError("v6 binding lacks 753-file EVAL-002 stage seal map")
    if not isinstance(frozen,dict) or len(frozen)!=73:raise ValueError("v6 binding lacks 73 frozen source hashes")
    if not isinstance(selected,dict) or len(selected)!=9:raise ValueError("v6 binding lacks 9 selected EVAL-002 input hashes")
    if len(logical_input_entries(inputs,"EVAL2-sealed:"))!=753:raise ValueError("virtual EVAL2 sealed-file index count mismatch")
    if logical_input_entries(inputs,"EVAL2-sealed:")!=sealed:raise ValueError("virtual EVAL2 sealed-file index differs")
    if logical_input_entries(inputs,"EVAL2-source:")!=frozen:raise ValueError("virtual EVAL2 frozen-source index differs")
    if logical_input_entries(inputs,"EVAL2-input:")!=selected:raise ValueError("virtual EVAL2 selected-input index differs")
    seal_path=E2/"核验/阶段封存.json"
    if sha(seal_path)!=binding.get("eval2_stage_seal_sha256"):raise ValueError("EVAL-002 stage-seal file hash mismatch")
    stage=json.loads(seal_path.read_text(encoding="utf-8"))
    if stage.get("files_sha256")!=sealed or stage.get("files")!=753:raise ValueError("v6 binding differs from EVAL-002 stage seal map")
    verify_hash_map(sealed,"EVAL-002 sealed file")
    verify_hash_map(frozen,"EVAL-002 frozen source")
    verify_hash_map(selected,"EVAL-002 selected input")
    direct=real_input_entries(inputs)
    verify_hash_map(direct,"direct v6 analysis input")
    return {"sealed_files":len(sealed),"frozen_sources":len(frozen),"selected_inputs":len(selected),"direct_inputs":len(direct)}
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
def visit_context(pos,remaining,visited,k):
    bins=[0]*25
    for cell in visited:
        r,c=divmod(int(cell),k);idx=(r*5//k)*5+(c*5//k)
        bins[idx]=min(3,bins[idx]+1)
    r,c=map(int,pos)
    return np.asarray([r/(k-1),c/(k-1),remaining/20]+[x/3 for x in bins],dtype=np.float32)
def quantiles(values):
    a=np.asarray(values,dtype=np.float64)
    if not len(a):return {"n":0,"mean":None,"p50":None,"p90":None,"min":None,"max":None}
    return {"n":int(len(a)),"mean":float(a.mean()),"p50":float(np.quantile(a,.5)),"p90":float(np.quantile(a,.9)),"min":float(a.min()),"max":float(a.max())}
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
    names=("输入绑定.json","诊断汇总.json","诊断报告.md","首步配对明细.jsonl","路径分叉明细.jsonl","下一项单因素方案草案.md")
    for name in names:
        if not (V6/name).is_file():raise FileNotFoundError(V6/name)
    for path in (SRC,TEST):
        if not path.is_file():raise FileNotFoundError(path)
    outputs={name:sha(V6/name) for name in names}
    input_counts=verify_v6_input_maps(binding)
    OUT.mkdir(parents=True);snap=OUT/"源码快照";snap.mkdir()
    for src in (SRC,TEST): (snap/src.name).write_bytes(src.read_bytes())
    audit_sources={rel(snap/p.name):sha(snap/p.name) for p in (SRC,TEST)}
    audit_binding={"version":"massgis-arrival-drift-independent-audit-binding-v4","date":"2026-10-05",
        "main_analysis_binding_sha256":sha(V6/"输入绑定.json"),"main_analysis_outputs_sha256":outputs,
        "eval2_stage_seal_sha256":binding["eval2_stage_seal_sha256"],
        "eval2_sealed_files_sha256":binding["eval2_sealed_files_sha256"],
        "eval2_frozen_sources_sha256":binding["eval2_frozen_sources_sha256"],
        "eval2_selected_inputs_sha256":binding["eval2_selected_inputs_sha256"],
        "main_input_sha256":binding["input_sha256"],"audit_source_snapshot_sha256":audit_sources,
        "preflight_counts":input_counts,
        "audit_limits":{"model_forward":0,"training":0,"navigation":0,"network":0,"spares":0}}
    write_new(OUT/"审计绑定.json",audit_binding)
    print(json.dumps({"frozen":True,"audit_out":rel(OUT),"source_snapshot":audit_sources},ensure_ascii=False))


def verify_binding(binding):
    if sha(V6/"输入绑定.json")!=binding["main_analysis_binding_sha256"]:raise ValueError("main v6 binding changed")
    for name,want in binding["main_analysis_outputs_sha256"].items():
        if sha(V6/name)!=want:raise ValueError(f"main result changed: {name}")
    seal_path=E2/"核验/阶段封存.json"
    if sha(seal_path)!=binding["eval2_stage_seal_sha256"]:raise ValueError("EVAL-002 stage seal changed")
    stage=json.loads(seal_path.read_text(encoding="utf-8"))
    if stage.get("files_sha256")!=binding["eval2_sealed_files_sha256"] or stage.get("files")!=753:raise ValueError("EVAL-002 sealed mapping differs")
    if logical_input_entries(binding["main_input_sha256"],"EVAL2-sealed:")!=binding["eval2_sealed_files_sha256"]:raise ValueError("virtual sealed-file index differs")
    if logical_input_entries(binding["main_input_sha256"],"EVAL2-source:")!=binding["eval2_frozen_sources_sha256"]:raise ValueError("virtual frozen-source index differs")
    if logical_input_entries(binding["main_input_sha256"],"EVAL2-input:")!=binding["eval2_selected_inputs_sha256"]:raise ValueError("virtual selected-input index differs")
    verify_hash_map(binding["eval2_sealed_files_sha256"],"EVAL-002 sealed file")
    verify_hash_map(binding["eval2_frozen_sources_sha256"],"EVAL-002 frozen source")
    verify_hash_map(binding["eval2_selected_inputs_sha256"],"EVAL-002 selected input")
    verify_hash_map(real_input_entries(binding["main_input_sha256"]),"direct v6 analysis input")
    if binding["preflight_counts"]!={"sealed_files":753,"frozen_sources":73,"selected_inputs":9,"direct_inputs":44}:
        raise ValueError("audit preflight input counts differ")
    for name,want in binding["audit_source_snapshot_sha256"].items():
        p=ROOT/name
        live=TEST if Path(name).name.startswith("test_") else SRC
        if sha(p)!=want or sha(live)!=want:raise ValueError(f"audit source snapshot differs: {name}")

def assert_nested_close(saved,calculated,label,tol=1e-10):
    if isinstance(saved,dict):
        if not isinstance(calculated,dict) or set(saved)!=set(calculated):raise AssertionError(f"{label}: object keys differ")
        for key in saved:assert_nested_close(saved[key],calculated[key],f"{label}.{key}",tol)
    elif isinstance(saved,(int,float)) and not isinstance(saved,bool):
        if not isinstance(calculated,(int,float)) or not np.isclose(float(saved),float(calculated),rtol=0,atol=tol):raise AssertionError(f"{label}: {saved} != {calculated}")
    elif saved!=calculated:raise AssertionError(f"{label}: {saved!r} != {calculated!r}")

def summarize_contexts(contexts):
    a=np.asarray(contexts,dtype=np.float32);b=a[:,3:]
    return {"decision_states":int(len(a)),"normalized_row":quantiles(a[:,0]),"normalized_column":quantiles(a[:,1]),
        "remaining_fraction":quantiles(a[:,2]),"occupied_buckets_per_state":quantiles((b>0).sum(1)),
        "saturated_buckets_per_state":quantiles((b>=1-1e-7).sum(1)),"empty_buckets_per_state":quantiles((b==0).sum(1))}

def reconstruct_train_contexts(summary):
    expected=summary["input_distribution"]["training_trace_hashes"]
    contexts=[];trace_hashes={}
    for seed in SEEDS:
        path=TRAIN/f"Small256_NoTarget_s{seed}"/"训练轨迹.npz";name=rel(path)
        if name not in expected or sha(path)!=expected[name]:raise ValueError(f"training trace hash mismatch: {name}")
        trace_hashes[name]=expected[name]
        with np.load(path,allow_pickle=False) as z:
            before=np.asarray(z["before"]);steps=np.asarray(z["step_before"]);after=np.asarray(z["after"])
            if before.shape!=(2560,64) or steps.shape!=before.shape or after.shape!=before.shape or np.asarray(z["wall"]).any():
                raise ValueError(f"training trace schema/boundary mismatch: {name}")
            visits=np.zeros((64,25),dtype=np.float32);last=np.full(64,-1,dtype=np.int16)
            for t in range(before.shape[0]):
                for e in range(before.shape[1]):
                    cell=int(before[t,e]);step=int(steps[t,e])
                    if step==0:visits[e].fill(0);visits[e,cell]=1
                    elif last[e]!=cell:raise ValueError(f"training trajectory discontinuity: seed={seed}, t={t}, episode={e}")
                    r,c=divmod(cell,5)
                    contexts.append(np.concatenate((np.asarray([r/4,c/4,(10-step)/10],dtype=np.float32),np.minimum(visits[e],3)/3)).astype(np.float32))
                    visits[e,int(after[t,e])]+=1;last[e]=int(after[t,e])
    if len(contexts)!=491520:raise AssertionError(f"training context count {len(contexts)} != 491520")
    return np.asarray(contexts,dtype=np.float32),trace_hashes

def audit_context_distribution(summary,contexts_by_grid,contexts_by_region):
    training,trace_hashes=reconstruct_train_contexts(summary)
    unique=np.unique(training,axis=0)
    if training.shape!=(491520,28) or len(unique)!=41919:raise AssertionError("training context shape/unique count")
    observed=summary["input_distribution"]
    if trace_hashes!=observed["training_trace_hashes"]:raise AssertionError("training trace inventory")
    if observed["training_trace_decision_states"]!=len(training):raise AssertionError("training trace decision-state denominator")
    buckets={}
    all_context={}
    for k in (10,15):
        mat=np.asarray(contexts_by_grid[k],dtype=np.float32);all_context[f"g{k}"]=mat
        buckets[f"g{k}"]=summarize_contexts(mat)
    buckets["training"]=summarize_contexts(training)
    buckets["training"]["unique_28d_contexts"]=int(len(unique))
    assert_nested_close(observed["coordinate_bucket_distribution"],buckets,"coordinate/bucket distributions")
    from scipy.spatial import cKDTree
    tree=cKDTree(unique)
    nearest={};byregion={}
    for k in (10,15):nearest[f"g{k}"]=quantiles(tree.query(all_context[f"g{k}"],k=1,workers=1)[0])
    for (k,area),values in contexts_by_region.items():byregion[f"g{k}_{area}"]=quantiles(tree.query(np.asarray(values,dtype=np.float32),k=1,workers=1)[0])
    assert_nested_close(observed["nearest_training_state_by_grid"],nearest,"nearest train-context distance by grid")
    assert_nested_close(observed["nearest_training_state_by_region"],byregion,"nearest train-context distance by region")
    return {"training_trace_states":len(training),"unique_training_contexts":len(unique),"current_contexts":{f"g{k}":len(v) for k,v in all_context.items()},"region_context_groups":len(byregion)}


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
    records={};first={};contexts_by_grid=defaultdict(list);contexts_by_region=defaultdict(list)
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
                    ctx=visit_context(pos,rem,visited,k);contexts_by_grid[k].append(ctx);contexts_by_region[(k,row["area"])].append(ctx)
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
    quality=defaultdict(Counter);region_quality=defaultdict(Counter);mismatch=0;bucketdiff=set();drs=[];dcs=[]
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
                rq=region_quality[(policy,area)];rq["n"]+=1;rq["score10"]+=score[q10];rq["score15"]+=score[q15]
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
        for area in region_ids:
            saved=summary["first_step_pairing"]["per_region_proposal_score"][f"{policy}_{area}"]
            got=region_quality[(policy,area)]
            if saved["n_repeated_rows"]!=got["n"] or not np.isclose(saved["mean_score_g10"],got["score10"]/got["n"],rtol=0,atol=1e-12) or not np.isclose(saved["mean_score_g15"],got["score15"]/got["n"],rtol=0,atol=1e-12):
                raise AssertionError("first-step proposal quality by region")
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
    # Recreate the 10-region paired bootstrap using a fresh draw-index matrix.
    boot=reach_detail["g15_minus_g10_region_bootstrap"]
    region_names=sorted({t["area"] for t in t10.values()})
    if len(region_names)!=10:raise AssertionError("region bootstrap geographic denominator")
    for policy in POLICIES:
        diffs=[]
        for area in region_names:
            left=region_success[(10,policy,area)];right=region_success[(15,policy,area)]
            diffs.append(right["success"]/right["n"]-left["success"]/left["n"])
        seed=int(boot[policy]["seed"]);nrep=int(boot[policy]["resamples"])
        picks=np.random.default_rng(seed).integers(0,len(diffs),size=(nrep,len(diffs)))
        draws=np.asarray(diffs,dtype=np.float64)[picks].mean(axis=1)
        independently={"unit":"whole region","n_regions":len(diffs),"resamples":nrep,"seed":seed,
            "mean_difference":float(np.mean(diffs)),"ci95_percentile":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),],
            "region_differences":dict(zip(region_names,diffs))}
        assert_nested_close(boot[policy],independently,f"{policy} whole-region bootstrap")
    context_checks=audit_context_distribution(summary,contexts_by_grid,contexts_by_region)
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
    result["checks"]["input_context_distribution"] = context_checks
    result["checks"]["whole_region_bootstrap"] = {"regions":10,"resamples":4000,"unit":"whole region"}
    write_new(OUT/"独立复核_v4.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument("--freeze",action="store_true");modes.add_argument("--audit",action="store_true")
    args=parser.parse_args()
    if args.freeze:freeze()
    else:print(json.dumps(run_audit(),ensure_ascii=False))
if __name__=="__main__":main()

