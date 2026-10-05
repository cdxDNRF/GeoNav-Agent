"""Offline, read-only EVAL-004 analysis. No project/model imports or forward calls."""
from __future__ import annotations
import argparse, hashlib, json, math, re, shutil
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E2 = ROOT / "DATA/processed_data/MassGIS/评测结果/正式十乘十十五乘十五确认_v1"
E1 = ROOT / "DATA/processed_data/MassGIS/工程准备/正式导航协议冻结_v1"
MASA = ROOT / "DATA/processed_data/Masa"
TRAIN = MASA / "训练结果/PBRS完整导航与局部匹配对照_v1"
EDGE = MASA / "训练结果/边缘连续性可信线索对照_v1"
OUT = ROOT / "DATA/processed_data/MassGIS/评测结果/十五乘十五到达能力漂移诊断_v5"
SRC = ROOT / "project/src/eval/diagnose_massgis_arrival_drift_v5.py"
AUD = ROOT / "project/src/eval/audit_massgis_arrival_drift_v5.py"
TEST = ROOT / "project/src/tests/test_massgis_arrival_drift_v5.py"
SEAL = E2 / "核验/阶段封存.json"
REG = E2 / "核验/执行预登记.json"
PROTO_COPY = E2 / "元数据/冻结协议副本.json"
TASKS = {10:E1/"元数据/grid10任务.json",15:E1/"元数据/grid15任务.json"}
MANIFESTS = {10:E1/"工程数据/grid10/数据清单.json",15:E1/"工程数据/grid15/数据清单.json"}
FEATS = E2 / "特征"
LOGS = E2 / "主对照"
POLICIES = ("M0","Coverage3Radial")
SEEDS = (0,1,2)
ACTIONS = ("up","right","down","left")
DELTA = {"up":(-1,0),"right":(0,1),"down":(1,0),"left":(0,-1)}

def jread(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def rel(p): return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
def dump_new(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8",newline="\n") as f: json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True); f.write("\n")
def jsonl_new(p,rows):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
def hash_map(expected,label):
    bad=[]
    for name,want in expected.items():
        p=ROOT/name
        if not p.is_file() or digest(p)!=want: bad.append(name)
    if bad: raise ValueError(f"{label} hash mismatch: {bad[:5]}")
def verify_seal():
    s=jread(SEAL); items=s["files_sha256"]
    if s["files"]!=753 or len(items)!=753: raise ValueError("EVAL-002 seal count is not 753")
    hash_map(items,"EVAL-002 stage seal")
    if s["default_upgraded"] or s["formal_regions_consumed"]!=10 or s["spare_regions_model_consumed"]!=0:
        raise ValueError("EVAL-002 protected status changed")
    return items
def verify_all_trajectory_seals():
    result={}
    for category in ("主对照","规则基线"):
        files=sorted((E2/category).rglob("*.jsonl")); nrecords=nactions=0
        for path in files:
            seal_path=path.with_suffix(".seal.json")
            if not seal_path.is_file(): raise ValueError(f"missing trajectory seal {seal_path}")
            seal=jread(seal_path)
            if digest(path)!=seal["sha256"]: raise ValueError(f"trajectory seal hash mismatch {path.name}")
            records=actions=0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row=json.loads(line); records+=1; actions+=int(row.get("steps",len(row.get("decisions",[]))))
            if (records,actions)!=(int(seal["records"]),int(seal["actions"])): raise ValueError(f"trajectory seal count mismatch {path.name}")
            nrecords+=records; nactions+=actions
        result[category]={"files":len(files),"records":nrecords,"actions":nactions}
    if result!={"主对照":{"files":120,"records":10500,"actions":188246},"规则基线":{"files":40,"records":3500,"actions":66879}}:
        raise ValueError(f"trajectory inventory differs from sealed executor totals: {result}")
    if sum(v["records"] for v in result.values())!=14000 or sum(v["actions"] for v in result.values())!=255125:
        raise ValueError("trajectory inventory differs from EVAL-002 full execution seal")
    return result
def verify_source_snapshot(reg):
    out={}
    for name,want in reg["source_sha256"].items():
        if "/源码快照/" in name: p=ROOT/name
        elif name.startswith("project/src/"):
            p=ROOT/"DATA/processed_data/MassGIS/评测结果/正式十乘十十五乘十五确认_v1/源码快照"/name[len("project/src/"):]
        else: p=ROOT/name
        if not p.is_file() or digest(p)!=want: raise ValueError(f"frozen EVAL-002 source differs: {name}")
        out[rel(p)]=want
    return out
def required_eval2_inputs(reg):
    names={rel(p) for p in TASKS.values()}|{rel(p) for p in MANIFESTS.values()}
    names|={rel(E1/"冻结协议.json"),rel(EDGE/"全局拟合均值.npy")}
    names|={rel(TRAIN/f"Small256_NoTarget_s{s}/model.pt") for s in SEEDS}
    d=reg["input_sha256"]
    if not names<=set(d): raise ValueError(f"required EVAL-002 prereg inputs absent: {sorted(names-set(d))}")
    chosen={k:d[k] for k in sorted(names)}; hash_map(chosen,"EVAL-002 selected inputs")
    return chosen
def load_protocol_tasks():
    reg=jread(REG); p=E1/"冻结协议.json"
    if digest(p)!=reg["protocol_sha256"] or jread(PROTO_COPY)!=jread(p): raise ValueError("EVAL-002 protocol copy mismatch")
    tasks={k:jread(v) for k,v in TASKS.items()}; manifests={k:jread(v) for k,v in MANIFESTS.items()}
    if (len(tasks[10]),len(tasks[15]))!=(750,1000): raise ValueError("task denominators changed")
    for k in (10,15):
        m=manifests[k]
        if (m["grid_size"],m["cell_size_m"],m["budget"],len(m["regions"]))!=(k,300,20,10): raise ValueError("grid manifest protocol changed")
        for r in m["regions"]:
            if len(r["cells"])!=k*k or sorted(c["cell"] for c in r["cells"])!=list(range(k*k)): raise ValueError(f"bad cells {k}/{r['area']}")
    return tasks,manifests
def derive_maps(manifests):
    maps={10:{},15:{}}; transforms={}
    for k in (10,15):
        for reg in manifests[k]["regions"]:
            byid={int(c["cell"]):c for c in reg["cells"]}
            bybounds={tuple(round(float(x),6) for x in c["bounds_m"]):int(c["cell"]) for c in reg["cells"]}
            if len(bybounds)!=k*k: raise ValueError("duplicate physical bounds")
            maps[k][reg["area"]]={"region":reg,"byid":byid,"bybounds":bybounds}
    for area in maps[10]:
        a,b=maps[10][area]["region"]["bounds_m"],maps[15][area]["region"]["bounds_m"]
        if maps[10][area]["region"].get("region_id")!=maps[15][area]["region"].get("region_id"): raise ValueError(f"region id differs by grid: {area}")
        if a[2:]!=b[2:] or not math.isclose(a[0]-b[0],1500) or not math.isclose(a[1]-b[1],1500):
            raise ValueError(f"windows bounds do not support stated embedding: {area}")
        mapping={}
        for i,c in maps[10][area]["byid"].items():
            j=maps[15][area]["bybounds"].get(tuple(round(float(x),6) for x in c["bounds_m"]))
            if j is None or divmod(j,15)!=(i//10,i%10+5): raise ValueError(f"bounds-derived map mismatch {area}:{i}->{j}")
            if c["file_sha256"]!=maps[15][area]["byid"][j]["file_sha256"]: raise ValueError("shared patch hash mismatch")
            mapping[i]=j
        transforms[area]={"dx_m":a[0]-b[0],"dy_m":a[1]-b[1],"region_id":maps[10][area]["region"]["region_id"],"map":mapping}
    return maps,transforms
def visit_buckets(visited,k):
    ids=np.asarray(tuple(visited),dtype=np.int64)
    if ids.ndim!=1 or len(ids)<1 or np.any(ids<0) or np.any(ids>=k*k): raise ValueError("invalid visit list")
    pooled=(ids//k*5//k)*5+(ids%k*5//k)
    return np.minimum(np.bincount(pooled,minlength=25).astype(np.float32),3)/3
def policy_input(mean,current,pos,remaining,visited,k,budget=20):
    r,c=map(int,pos); visited=tuple(map(int,visited))
    if k not in (10,15) or budget!=20 or not(0<=r<k and 0<=c<k and 1<=remaining<=budget): raise ValueError("invalid policy state")
    if len(visited)!=budget-remaining+1 or visited[-1]!=r*k+c: raise ValueError("invalid public observation prefix")
    coords=np.asarray([r/(k-1),c/(k-1),remaining/budget],np.float32)
    return np.concatenate((np.asarray(mean,np.float32),np.asarray(current,np.float32),coords,visit_buckets(visited,k))).astype(np.float32)
def context(pos,remaining,visited,k):
    r,c=map(int,pos)
    return np.concatenate((np.asarray([r/(k-1),c/(k-1),remaining/20],np.float32),visit_buckets(visited,k)))
def normalized_coordinate_delta(row10,col10):
    return row10/14-row10/9,(col10+5)/14-col10/9
def qtls(values):
    a=np.asarray(values,dtype=np.float64)
    if not len(a): return {"n":0,"mean":None,"p50":None,"p90":None,"min":None,"max":None}
    return {"n":int(len(a)),"mean":float(a.mean()),"p50":float(np.quantile(a,.5)),"p90":float(np.quantile(a,.9)),"min":float(a.min()),"max":float(a.max())}
def corr(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    return None if len(a)<2 or np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
def action_quality(pos,goal,action,k):
    r,c=map(int,pos); gr,gc=map(int,goal); dr,dc=DELTA[action]
    d0=abs(r-gr)+abs(c-gc); nr,nc=r+dr,c+dc
    if not(0<=nr<k and 0<=nc<k): nr,nc=r,c
    d1=abs(nr-gr)+abs(nc-gc)
    return "toward" if d1<d0 else "away" if d1>d0 else "side_or_wall"
def world_center(idx,k,maps,area):
    b=maps[k][area]["byid"][idx]["bounds_m"]
    return ((b[0]+b[2])/2,(b[1]+b[3])/2)

def read_main(tasks,manifests):
    task_by_episode={k:{x["episode_id"]:x for x in v} for k,v in tasks.items()}
    cache={}
    for area in sorted(r["area"] for r in manifests[15]["regions"]):
        with np.load(FEATS/f"{area}.npz",allow_pickle=False) as z: cache[area]=np.asarray(z["global_features"],np.float32)
        if cache[area].shape!=(225,512): raise ValueError(f"feature cache shape mismatch {area}")
    mean=np.load(EDGE/"全局拟合均值.npy",allow_pickle=False)
    if mean.shape!=(512,) or mean.dtype!=np.float32: raise ValueError("NoTarget mean shape/dtype mismatch")
    records={}; first=[]; contexts={10:[],15:[]}; contexts_area=defaultdict(list)
    metrics=defaultdict(Counter); strata=defaultdict(Counter); logs_hash={}
    nfiles=nrecords=nactions=0
    pat=re.compile(r"^g(10|15)_(M0|Coverage3Radial)_s([0-2])_CueFull_(img_\d+)\.jsonl$")
    for path in sorted(LOGS.glob("g*_CueFull_*.jsonl")):
        m=pat.match(path.name)
        if not m: continue
        k,policy,seed,area=int(m[1]),m[2],int(m[3]),m[4]
        sp=path.with_suffix(".seal.json"); seal=jread(sp); h=digest(path)
        if seal["sha256"]!=h: raise ValueError(f"trajectory seal mismatch {path.name}")
        bind=seal["binding"]
        if (bind["grid"],bind["policy"],bind["seed"],bind["condition"],bind["area"])!=(k,policy,seed,"CueFull",area): raise ValueError(f"trajectory binding mismatch {path.name}")
        if bind["tasks_sha256"]!=digest(TASKS[k]) or bind["feature_sha256"]!=digest(FEATS/f"{area}.npz"): raise ValueError(f"task/cache binding mismatch {path.name}")
        logs_hash[rel(path)]=h; nfiles+=1; local_n=local_a=0
        with path.open(encoding="utf-8") as f:
            for line in f:
                row=json.loads(line); local_n+=1; nrecords+=1; local_a+=len(row["decisions"]); nactions+=len(row["decisions"])
                if (row["grid_size"],row["policy"],row["seed"],row["area"],row["condition"],row["status"])!=(k,policy,seed,area,"CueFull","completed"): raise ValueError("trajectory identity/condition/status mismatch")
                task=task_by_episode[k].get(row["episode_id"])
                if task is None or (task["pair_id"],task["area"],task["dist"],task["stratum"],task["cohort"])!=(row["pair_id"],area,row["distance"],row["stratum"],row["cohort"]): raise ValueError(f"task/list mismatch {row['episode_id']}")
                record_id=row["pair_id"] if row["pair_id"] is not None else "extra:"+row["episode_id"]
                key=(k,policy,seed,record_id)
                if key in records: raise ValueError(f"duplicate record {key}")
                item={"grid":k,"policy":policy,"seed":seed,"area":area,"pair_id":row["pair_id"],"record_id":record_id,"episode_id":row["episode_id"],
                      "success":bool(row["success"]),"sg":int(row["sg"]),"steps":int(row["steps"]),"distance":int(row["distance"]),
                      "stratum":row["stratum"],"cohort":row["cohort"],"region_id":row["region_id"],"task":task,
                      "route":[int(x["patch_id"]) for x in row["trajectory"]],"states":[]}
                records[key]=item; metrics[(k,policy)]["n"]+=1; metrics[(k,policy)]["success"]+=int(row["success"])
                if task["cohort"]=="common":
                    for typ,layer in (("distance",int(task["dist"])),("stratum",task["stratum"])):
                        strata[(k,policy,typ,layer)]["n"]+=1; strata[(k,policy,typ,layer)]["success"]+=int(row["success"])
                if len(row["decisions"])!=int(row["steps"]): raise ValueError("saved decisions != executed steps")
                for i,d in enumerate(row["decisions"]):
                    pos=tuple(map(int,d["public_position"])); visited=tuple(map(int,d["public_visited"])); rem=int(d["remaining_budget"])
                    if d["step"]!=i+1 or rem!=20-i: raise ValueError("invalid public step/budget sequence")
                    idx=pos[0]*k+pos[1]; idx15=idx if k==15 else pos[0]*15+pos[1]+5
                    vec=policy_input(mean,cache[area][idx15],pos,rem,visited,k)
                    if vec.shape!=(1052,) or hashlib.sha256(vec.tobytes()).hexdigest()!=d["explorer_features_sha256"]: raise ValueError(f"policy input hash failed {row['episode_id']}/{i}")
                    logits=np.asarray(d["explorer_logits"],np.float64)
                    if logits.shape!=(4,) or ACTIONS[int(np.argmax(logits))]!=d["explorer_action"]: raise ValueError("logits/argmax mismatch")
                    ctx=context(pos,rem,visited,k); contexts[k].append(ctx); contexts_area[(k,area)].append(ctx)
                    item["states"].append({"position":list(pos),"visited":list(visited),"remaining":rem,"logits":logits.tolist(),"proposal":d["explorer_action"],"executed":d["action"]})
                    if i==0:
                        first.append({"grid":k,"policy":policy,"seed":seed,"area":area,"region_id":row["region_id"],"pair_id":row["pair_id"],
                                      "episode_id":row["episode_id"],"distance":int(task["dist"]),"stratum":task["stratum"],"cohort":task["cohort"],
                                      "start":int(task["start"]),"goal":int(task["goal"]),"position":list(pos),
                                      "current_sha":d["current_image_sha256"],"target_sha":d["target_image_sha256"],
                                      "input_sha":d["explorer_features_sha256"],"logits":logits.tolist(),"proposal":d["explorer_action"],
                                      "quality":action_quality(pos,divmod(int(task["goal"]),k),d["explorer_action"],k)})
        if (local_n,local_a)!=(seal["records"],seal["actions"]) or local_n!=(75 if k==10 else 100): raise ValueError(f"record/action counts differ from seal {path.name}")
    if (nfiles,nrecords,nactions)!=(120,10500,188246): raise ValueError(f"main log totals mismatch {nfiles}/{nrecords}/{nactions}")
    return {"records":records,"first":first,"contexts":contexts,"contexts_area":contexts_area,"metrics":metrics,"strata":strata,
            "log_hashes":logs_hash,"nfiles":nfiles,"nrecords":nrecords,"nactions":nactions,"cache":cache,"mean":mean}

def train_contexts():
    reg=jread(TRAIN/"预登记.json"); split=jread(TRAIN/"源图划分.json"); audit=jread(TRAIN/"独立复核.json")
    fit=sorted(split["fit"]); table=np.load(MASA/"papr_train_sat_embeds_grid_5.npy",allow_pickle=True).item()
    if not set(fit)<=set(table): raise ValueError("training fit features absent")
    for name,want in reg["data_sha256"].items():
        if digest(MASA/name)!=want: raise ValueError(f"PBRS data hash mismatch {name}")
    mean=np.load(TRAIN/"全局拟合均值.npy",allow_pickle=False)
    if digest(TRAIN/"全局拟合均值.npy")!=reg["frozen_inputs_sha256"]["全局拟合均值.npy"]: raise ValueError("training mean hash mismatch")
    if not np.array_equal(mean,np.load(EDGE/"全局拟合均值.npy",allow_pickle=False)): raise ValueError("EVAL-002 NoTarget mean differs")
    contexts=[]; trace_hashes={}; n=0
    for seed in SEEDS:
        folder=TRAIN/f"Small256_NoTarget_s{seed}"; path=folder/"训练轨迹.npz"; resources=jread(folder/"训练资源.json")
        want=resources["trace_sha256"]; audit_key=f"Small256_NoTarget_s{seed}/训练轨迹.npz"
        if digest(path)!=want or audit["files_sha256"].get(audit_key)!=want: raise ValueError(f"training trace not bound, seed={seed}")
        if jread(REG)["input_sha256"].get(rel(folder/"model.pt"))!=resources["final_sha256"]: raise ValueError(f"checkpoint/trace mismatch seed={seed}")
        trace_hashes[rel(path)]=want
        with np.load(path,allow_pickle=False) as z:
            if z["before"].shape!=(2560,64) or z["wall"].any(): raise ValueError("training trace schema or boundary mismatch")
            before,step,after=z["before"],z["step_before"],z["after"]; visits=np.zeros((64,25),np.float32); last=np.full(64,-1,np.int16)
            for t in range(before.shape[0]):
                for e in range(before.shape[1]):
                    pos,s=int(before[t,e]),int(step[t,e])
                    if s==0: visits[e].fill(0); visits[e,pos]=1
                    elif last[e]!=pos: raise ValueError(f"training trajectory discontinuity {seed}/{t}/{e}")
                    r,c=divmod(pos,5)
                    contexts.append(np.concatenate((np.asarray([r/4,c/4,(10-s)/10],np.float32),np.minimum(visits[e],3)/3)).astype(np.float32))
                    visits[e,int(after[t,e])]+=1; last[e]=int(after[t,e]); n+=1
    return np.asarray(contexts,np.float32),table,fit,trace_hashes,n

def bootstrap(diff):
    names=sorted(diff); a=np.asarray([diff[x] for x in names],np.float64)
    if len(a)!=10: raise ValueError("region bootstrap requires 10 regions")
    rng=np.random.default_rng(7317); samples=rng.integers(0,10,size=(4000,10)); means=a[samples].mean(1)
    return {"unit":"whole region","n_regions":10,"resamples":4000,"seed":7317,"mean_difference":float(a.mean()),
            "ci95_percentile":[float(np.quantile(means,.025)),float(np.quantile(means,.975))],
            "region_differences":{x:float(diff[x]) for x in names}}
def pair_task_rows(tasks10,tasks15):
    left={x["pair_id"]:x for x in tasks10 if x["cohort"]=="common"}
    right={x["pair_id"]:x for x in tasks15 if x["cohort"]=="common"}
    if set(left)!=set(right): raise ValueError("common pair ids differ between grids")
    return {pid:(left[pid],right[pid]) for pid in left}
def first_pair_analysis(tasks,transforms,parsed):
    t10={x["pair_id"]:x for x in tasks[10] if x["cohort"]=="common"}; t15={x["pair_id"]:x for x in tasks[15] if x["cohort"]=="common"}
    if len(t10)!=750 or len(t15)!=750 or set(t10)!=set(t15): raise ValueError("common task identities differ")
    for pid,a in t10.items():
        b=t15[pid]; mp=transforms[a["area"]]["map"]
        if a["area"]!=b["area"] or mp[int(a["start"])]!=int(b["start"]) or mp[int(a["goal"])]!=int(b["goal"]): raise ValueError(f"endpoint map mismatch {pid}")
    lookup={(x["grid"],x["policy"],x["seed"],x["pair_id"]):x for x in parsed["first"]}
    rows=[]; bucketdiff=set(); score={"toward":1,"side_or_wall":0,"away":-1}; counters=defaultdict(lambda:{10:Counter(),15:Counter(),"pair":Counter()})
    for pid,a in t10.items():
        b=t15[pid]; r,c=divmod(int(a["start"]),10); r15,c15=divmod(int(b["start"]),15)
        bucket10=(r*5//10)*5+c*5//10; bucket15=(r15*5//15)*5+c15*5//15
        if bucket10!=bucket15: bucketdiff.add(pid)
        for policy in POLICIES:
            for seed in SEEDS:
                x=lookup[(10,policy,seed,pid)]; y=lookup[(15,policy,seed,pid)]
                if x["current_sha"]!=y["current_sha"] or x["target_sha"]!=y["target_sha"]: raise ValueError("paired image hash mismatch")
                if x["position"][0]!=y["position"][0] or x["position"][1]+5!=y["position"][1]: raise ValueError("first state map mismatch")
                dr=y["position"][0]/14-x["position"][0]/9; dc=y["position"][1]/14-x["position"][1]/9
                changed=x["proposal"]!=y["proposal"]; qd=score[y["quality"]]-score[x["quality"]]
                counters[policy][10][x["quality"]]+=1; counters[policy][15][y["quality"]]+=1
                counters[policy]["pair"]["better" if qd>0 else "worse" if qd<0 else "tie"]+=1
                rows.append({"area":a["area"],"region_id":transforms[a["area"]]["region_id"],"pair_id":pid,"policy":policy,"seed":seed,"distance":a["dist"],"stratum":a["stratum"],
                    "start10":a["start"],"goal10":a["goal"],"start15":b["start"],"goal15":b["goal"],"position_delta":[0,5],
                    "normalized_coordinate_delta":[dr,dc],"bucket10":bucket10,"bucket15":bucket15,
                    "logits10":x["logits"],"logits15":y["logits"],"logit_delta_15_minus_10":[y["logits"][i]-x["logits"][i] for i in range(4)],
                    "proposal10":x["proposal"],"proposal15":y["proposal"],"quality10":x["quality"],"quality15":y["quality"],
                    "current_image_sha256_equal":True,"target_image_sha256_equal":True,"proposal_changed":changed})
    if len(rows)!=4500: raise ValueError(f"first pair count {len(rows)} != 4500")
    correlations={}
    for i,name in enumerate(ACTIONS):
        correlations[name]={"delta_logit_vs_row_delta_pearson":corr([x["logit_delta_15_minus_10"][i] for x in rows],[x["normalized_coordinate_delta"][0] for x in rows]),
                            "delta_logit_vs_column_delta_pearson":corr([x["logit_delta_15_minus_10"][i] for x in rows],[x["normalized_coordinate_delta"][1] for x in rows])}
    proposal={}
    for pol in POLICIES:
        sub=[x for x in rows if x["policy"]==pol]
        proposal[pol]={"paired_n":len(sub),"grid10":{q:int(counters[pol][10][q]) for q in ("toward","away","side_or_wall")},
                       "grid15":{q:int(counters[pol][15][q]) for q in ("toward","away","side_or_wall")},
                       "paired_quality_grid15_better":int(counters[pol]["pair"]["better"]),"paired_quality_grid15_worse":int(counters[pol]["pair"]["worse"]),
                       "paired_quality_tie":int(counters[pol]["pair"]["tie"])}
        for label, subset in (("by_distance",sorted({x["distance"] for x in sub})),("by_seed",SEEDS)):
            proposal[pol][label]={}
            for value in subset:
                group=[x for x in sub if x["distance" if label=="by_distance" else "seed"]==value]
                proposal[pol][label][str(value)]={"n":len(group),"toward_rate_g10":sum(x["quality10"]=="toward" for x in group)/len(group),
                                                  "toward_rate_g15":sum(x["quality15"]=="toward" for x in group)/len(group)}
    regions=defaultdict(Counter)
    for x in rows:
        v=regions[(x["policy"],x["area"])]; v["n"]+=1; v["a"]+=score[x["quality10"]]; v["b"]+=score[x["quality15"]]
    unique_dr=[r/14-r/9 for r,c in (divmod(int(x["start"]),10) for x in t10.values())]
    unique_dc=[(c+5)/14-c/9 for r,c in (divmod(int(x["start"]),10) for x in t10.values())]
    return {"matched_first_decisions":len(rows),"unique_physical_tasks":750,
        "proposal_mismatch_rate":sum(x["proposal_changed"] for x in rows)/len(rows),"start_bucket_mismatch_rate":len(bucketdiff)/750,
        "normalized_coordinate_delta_unique_tasks":{"row":qtls(unique_dr),"column":qtls(unique_dc)},
        "normalized_coordinate_delta_repeated_states":{"row":qtls([x["normalized_coordinate_delta"][0] for x in rows]),
            "column":qtls([x["normalized_coordinate_delta"][1] for x in rows]),"l2":qtls([math.hypot(*x["normalized_coordinate_delta"]) for x in rows])},
        "delta_logit_coordinate_pearson":correlations,"proposal_quality":proposal,
        "per_region_proposal_score":{f"{p}_{a}":{"n_repeated_rows":v["n"],"mean_score_g10":v["a"]/v["n"],"mean_score_g15":v["b"]/v["n"]} for (p,a),v in regions.items()},
        "dependence_note":"4500 rows are 750 physical tasks repeated across three weights and two strategies; geography n=10 regions."}, rows, pair_task_rows(tasks[10],tasks[15])
def reachability(tasks,parsed):
    all_metrics={}
    for (k,p),c in parsed["metrics"].items(): all_metrics[f"g{k}_{p}_all_tasks"]={"successes":int(c["success"]),"planned_task_weight_runs":int(c["n"]),"sr":c["success"]/c["n"]}
    common={r["pair_id"] for r in tasks[10]}; records=parsed["records"]; regions={}; layers={}; cohorts={}; cis={}
    for policy in POLICIES:
        rates={10:{},15:{}}
        for k in (10,15):
            for area in sorted({x["area"] for x in tasks[k] if x["cohort"]=="common"}):
                es=[records[(k,policy,s,pid)] for s in SEEDS for pid in common if records[(k,policy,s,pid)]["area"]==area]
                regions[f"g{k}_{policy}_{area}"]={"successes":sum(x["success"] for x in es),"planned_task_weight_runs":len(es),"sr":sum(x["success"] for x in es)/len(es),
                    "tasks_per_seed":sum(x["area"]==area and x["cohort"]=="common" for x in tasks[k])}
                rates[k][area]=regions[f"g{k}_{policy}_{area}"]["sr"]
            for layer in sorted({x["stratum"] for x in tasks[k] if x["cohort"]=="common"}):
                ids={x["pair_id"] for x in tasks[k] if x["cohort"]=="common" and x["stratum"]==layer}
                es=[records[(k,policy,s,pid)] for s in SEEDS for pid in ids]
                layers[f"g{k}_{policy}_{layer}"]={"successes":sum(x["success"] for x in es),"planned_task_weight_runs":len(es),"sr":sum(x["success"] for x in es)/len(es)}
            for cohort in sorted({x["cohort"] for x in tasks[k]}):
                selected_tasks=[x for x in tasks[k] if x["cohort"]==cohort]
                es=[records[(k,policy,s,x["pair_id"] if x["pair_id"] is not None else "extra:"+x["episode_id"])] for s in SEEDS for x in selected_tasks]
                cohorts[f"g{k}_{policy}_{cohort}"]={"successes":sum(x["success"] for x in es),"planned_task_weight_runs":len(es),"sr":sum(x["success"] for x in es)/len(es)}
        cis[policy]=bootstrap({a:rates[15][a]-rates[10][a] for a in rates[10]})
    return {"all_tasks_by_grid_policy":all_metrics,"common_task_cohort":cohorts,"common_tasks_by_region":regions,"common_tasks_by_stratum":layers,"g15_minus_g10_region_bootstrap":cis}
def route_summary(parsed,maps,taskpairs):
    rows=[]; types=Counter(); prop=defaultdict(Counter)
    records=parsed["records"]
    for pid,(ta,tb) in taskpairs.items():
        for pol in POLICIES:
            for seed in SEEDS:
                a=records[(10,pol,seed,pid)]; b=records[(15,pol,seed,pid)]
                ca=[world_center(i,10,maps,a["area"]) for i in a["route"]]; cb=[world_center(i,15,maps,b["area"]) for i in b["route"]]
                shared=0
                while shared<min(len(ca),len(cb)) and ca[shared]==cb[shared]: shared+=1
                if shared==len(ca)==len(cb): fork=None; kind="identical_route"
                elif shared<min(len(ca),len(cb)): fork=shared; kind="forked"
                else: fork=shared; kind="one_route_ended_first"
                types[kind]+=1; n=min(shared,len(a["states"]),len(b["states"])); goals={10:divmod(ta["goal"],10),15:divmod(tb["goal"],15)}
                qualities={10:Counter(),15:Counter()}
                for k,rec in ((10,a),(15,b)):
                    for i,state in enumerate(rec["states"]):
                        phase="prefork" if i<n else "postfork"
                        qualities[k][f"{phase}_{action_quality(state['position'],goals[k],state['proposal'],k)}"]+=1
                for k in (10,15):
                    for label,count in qualities[k].items(): prop[(k,label.split("_")[0])][label.split("_",1)[1]]+=count
                rows.append({"area":a["area"],"region_id":a["region_id"],"pair_id":pid,"policy":pol,"seed":seed,"first_different_physical_position_index":fork,
                    "shared_physical_positions":shared,"route_category":kind,"route10_states":len(ca),"route15_states":len(cb),
                    "success10":a["success"],"success15":b["success"],"sg10":a["sg"],"sg15":b["sg"],
                    "proposal_quality_prefork_g10":{x:y for x,y in qualities[10].items() if x.startswith("prefork_")},
                    "proposal_quality_prefork_g15":{x:y for x,y in qualities[15].items() if x.startswith("prefork_")},
                    "proposal_quality_postfork_g10":{x:y for x,y in qualities[10].items() if x.startswith("postfork_")},
                    "proposal_quality_postfork_g15":{x:y for x,y in qualities[15].items() if x.startswith("postfork_")}})
    forks=[r["first_different_physical_position_index"] for r in rows if r["first_different_physical_position_index"] is not None]
    return {"rows":rows,"summary":{"paired_routes":len(rows),"route_categories":dict(types),"first_different_position_index":qtls(forks),
        "identical_route_fraction":types["identical_route"]/len(rows),"proposal_quality_by_pre_postfork":{f"g{k}_{phase}":dict(v) for (k,phase),v in prop.items()}}}
def input_distances(parsed,maps,transforms,training):
    from scipy.spatial import cKDTree
    tr,table,fit,trace_hashes,ntrain=training; unique=np.unique(tr,axis=0)
    if unique.shape[1]!=28: raise ValueError("expected 28-d training context")
    tree=cKDTree(unique); nearest={}; byarea={}
    for k in (10,15):
        d,_=tree.query(np.asarray(parsed["contexts"][k],np.float32),k=1,workers=1); nearest[f"g{k}"]=qtls(d)
    for (k,area),values in parsed["contexts_area"].items():
        d,_=tree.query(np.asarray(values,np.float32),k=1,workers=1); byarea[f"g{k}_{area}"]=qtls(d)
    bank=np.stack([np.asarray(table[a],np.float32) for a in fit]).reshape(-1,512)
    bn=bank/np.maximum(np.linalg.norm(bank,axis=1,keepdims=True),1e-12); cosine={}
    for k in (10,15):
        for area in maps[k]:
            ids=sorted(maps[k][area]["byid"])
            idx=[transforms[area]["map"][i] for i in ids] if k==10 else ids
            mat=parsed["cache"][area][idx]; norm=mat/np.maximum(np.linalg.norm(mat,axis=1,keepdims=True),1e-12)
            vals=[float(1-np.max(v@bn.T)) for v in norm]
            cosine[(k,area)]=dict(zip(ids,vals))
    cos_states=defaultdict(list); feature_norms=defaultdict(list)
    for key,row in parsed["records"].items():
        k,p,s,pid=key; area=row["area"]
        for st in row["states"]:
            r,c=st["position"]; cos_states[k].append(cosine[(k,area)][r*k+c])
    for k in (10,15):
        for area in maps[k]:
            ids=sorted(maps[k][area]["byid"]); idx=[transforms[area]["map"][i] for i in ids] if k==10 else ids
            feature_norms[k].extend(np.linalg.norm(parsed["cache"][area][idx],axis=1).tolist())
    bucket={}
    for k in (10,15):
        a=np.asarray(parsed["contexts"][k],np.float32); b=a[:,3:]
        bucket[f"g{k}"]={"decision_states":len(a),"normalized_row":qtls(a[:,0]),"normalized_column":qtls(a[:,1]),"remaining_fraction":qtls(a[:,2]),
            "occupied_buckets_per_state":qtls((b>0).sum(1)),"saturated_buckets_per_state":qtls((b>=1-1e-7).sum(1)),"empty_buckets_per_state":qtls((b==0).sum(1))}
    b=tr[:,3:]
    bucket["training"]={"decision_states":len(tr),"unique_28d_contexts":len(unique),"normalized_row":qtls(tr[:,0]),"normalized_column":qtls(tr[:,1]),"remaining_fraction":qtls(tr[:,2]),
        "occupied_buckets_per_state":qtls((b>0).sum(1)),"saturated_buckets_per_state":qtls((b>=1-1e-7).sum(1)),"empty_buckets_per_state":qtls((b==0).sum(1))}
    image={"distance":"nearest cosine distance of cached current-image vector to any of 109 fit-source Masa 5x5 cached vectors",
        "training_fit_patch_l2_norm":qtls(np.linalg.norm(bank,axis=1)),
        "target_channel":{"training_and_eval":"same bound 512-d NoTarget mean","sha256":digest(EDGE/"全局拟合均值.npy"),"cosine_distance":0.0}}
    for k in (10,15):
        image[f"g{k}_current_nearest_training_patch_cosine_distance"]=qtls(cos_states[k]); image[f"g{k}_cached_patch_l2_norm"]=qtls(feature_norms[k])
    return {"training_checkpoint_scope":"Small256_NoTarget seeds 0,1,2, bound to EVAL-002 frozen models",
        "training_trace_hashes":trace_hashes,"training_trace_decision_states":ntrain,"training_fit_sources":len(fit),
        "state_distance_method":"exact nearest Euclidean distance over observed 28-d states: normalized row/column, remaining fraction, 25 visit buckets; cKDTree",
        "nearest_training_state_by_grid":nearest,"nearest_training_state_by_region":byarea,"coordinate_bucket_distribution":bucket,
        "current_image_feature_domain_distance":image,"independence_limit":"repeat states are not independent geography samples"}

def freeze():
    if OUT.exists(): raise FileExistsError(f"refuse to overwrite {OUT}")
    sealed=verify_seal(); trajectory_inventory=verify_all_trajectory_seals(); reg=jread(REG); frozen_sources=verify_source_snapshot(reg); selected=required_eval2_inputs(reg)
    tasks,manifests=load_protocol_tasks(); maps,transforms=derive_maps(manifests)
    if sum(x["cohort"]=="common" for x in tasks[10])!=750 or sum(x["cohort"]=="common" for x in tasks[15])!=750: raise ValueError("common cohort mismatch")
    OUT.mkdir(parents=True); snap=OUT/"源码快照"; snap.mkdir()
    for p in (SRC,AUD,TEST): shutil.copyfile(p,snap/p.name)
    paths={SEAL,REG,PROTO_COPY,E1/"冻结协议.json",E1/"元数据/正式池使用账本.json",*TASKS.values(),*MANIFESTS.values(),
        TRAIN/"预登记.json",TRAIN/"源图划分.json",TRAIN/"独立复核.json",TRAIN/"全局拟合均值.npy",MASA/"papr_train_sat_embeds_grid_5.npy",EDGE/"全局拟合均值.npy"}
    for k in (10,15):
        for area in maps[k]: paths.add(FEATS/f"{area}.npz"); paths.add(FEATS/f"{area}.seal.json")
    for s in SEEDS:
        f=TRAIN/f"Small256_NoTarget_s{s}"; paths.update({f/"model.pt",f/"训练轨迹.npz",f/"训练资源.json"})
    hashes={rel(p):digest(p) for p in sorted(paths,key=str)}
    hashes.update({f"EVAL2-sealed:{k}":v for k,v in sealed.items()}); hashes.update({f"EVAL2-source:{k}":v for k,v in frozen_sources.items()})
    hashes.update({f"EVAL2-input:{k}":v for k,v in selected.items()})
    source_sha={rel(p):digest(snap/p.name) for p in (SRC,AUD,TEST)}
    binding={"version":"massgis-arrival-drift-binding-v5","task_id":"EVAL-004","date":"2026-10-05",
        "limits":{"model_forward":0,"training":0,"navigation":0,"network":0,"spare_regions_consumed":0},
        "eval2_stage_seal_sha256":digest(SEAL),"eval2_sealed_file_count":len(sealed),"eval2_sealed_files_sha256":sealed,
        "eval2_frozen_sources_sha256":frozen_sources,"eval2_selected_inputs_sha256":selected,"input_sha256":hashes,"trajectory_seal_inventory":trajectory_inventory,
        "analysis_source_snapshot_sha256":source_sha,"planned_logs":120,
        "denominators":{"g10":750,"g15":1000,"common_pairs":750},
        "bounds_derived_maps":{a:{"dx_m":v["dx_m"],"dy_m":v["dy_m"],"row_offset":0,"column_offset":5,"g10_to_g15":{str(i):j for i,j in v["map"].items()}} for a,v in transforms.items()},
        "protocol":{"grids":[10,15],"policies":list(POLICIES),"seeds":list(SEEDS),"condition":"CueFull","B":20,"cell_size_m":300}}
    dump_new(OUT/"输入绑定.json",binding)
    print(json.dumps({"frozen":True,"out":rel(OUT),"stage_files":len(sealed),"sources":len(frozen_sources),"selected_inputs":len(selected),"new_sources":source_sha},ensure_ascii=False))
def verify_binding(b):
    if digest(SEAL)!=b["eval2_stage_seal_sha256"]: raise ValueError("EVAL-002 seal changed after freeze")
    hash_map(b["eval2_sealed_files_sha256"],"EVAL-002 stage recheck")
    if verify_all_trajectory_seals()!=b["trajectory_seal_inventory"]: raise ValueError("EVAL-002 trajectory seal inventory changed")
    for name,want in b["input_sha256"].items():
        if name.startswith(("EVAL2-sealed:","EVAL2-source:","EVAL2-input:")): continue
        p=ROOT/name
        if not p.is_file() or digest(p)!=want: raise ValueError(f"input changed after freeze: {name}")
    for name,want in b["analysis_source_snapshot_sha256"].items():
        p=ROOT/name; live=(ROOT/"project/src/tests/test_massgis_arrival_drift_v5.py") if Path(name).name.startswith("test_") else ROOT/"project/src/eval"/Path(name).name
        if not p.is_file() or digest(p)!=want or digest(live)!=want: raise ValueError(f"analysis source changed after freeze: {name}")
def create_report(summary):
    first=summary["first_step_pairing"]; reach=summary["reachability"]; q=first["proposal_quality"]
    sr=reach["all_tasks_by_grid_policy"]; d10=sr["g10_Coverage3Radial_all_tasks"]["sr"]; d15=sr["g15_Coverage3Radial_all_tasks"]["sr"]
    lines=["# EVAL-004：15×15到达能力与输入表征漂移诊断","",
    "离线诊断完成。只读取EVAL-002封存日志、缓存特征和训练轨迹；没有训练、forward、重新编码/预测、导航、下载/API调用或备用区域消费。","",
    "## 结论","",
    f"Coverage3Radial全量SR：10×10 **{d10:.2%}**，15×15 **{d15:.2%}**。首步配对为750道共同物理任务×3权重×2策略，共{first['matched_first_decisions']}行。逐区bounds得出局部索引映射为行不变、列+5，纠正了“索引相同”的旧摘要说法。","",
    f"配对策略首步提议不一致率{first['proposal_mismatch_rate']:.2%}；起点的5×5访问桶不一致率{first['start_bucket_mismatch_rate']:.2%}。动作质量和坐标差/logit差相关见逐策略统计及逐区结果。","",
    "训练分布采用EVAL-002相同Small256 NoTarget三权重绑定的保存训练状态；从轨迹重建位置、预算和访问历史，最近邻只度量28维状态上下文。目标特征均值相同；当前图像512维向量另与训练源缓存计算最近余弦距离。对数/距离一致性支持表征漂移是机制候选，但观察性对照不能单独证明因果。","",
    "## 首步动作质量","",
    "|策略|网格|朝向目标|背离目标|侧向/撞边|15相对10更好/更差/持平|","|---|---:|---:|---:|---:|---:|"]
    for p,v in q.items():
        for k in (10,15):
            x=v[f"grid{k}"]; tail="—" if k==10 else f"{v['paired_quality_grid15_better']}/{v['paired_quality_grid15_worse']}/{v['paired_quality_tie']}"
            lines.append(f"|{p}|{k}×{k}|{x['toward']}|{x['away']}|{x['side_or_wall']}|{tail}|")
    lines += ["","任务按完整区域配对，10区bootstrap结果、各距离层/区域、权重结果均见诊断汇总。4500条首步决策包含任务、权重和策略重复，不是4500个独立地图。","",
    "## 输入和路径","",
    f"从封存向量重建{summary['input_check']['reconstructed_inputs']}条1052维explorer输入，所有SHA与日志吻合；训练状态数为{summary['input_distribution']['training_trace_decision_states']}。路线同轨/分叉及分叉前后提议方向见汇总及路径分叉明细。","",
    "## 后续单因素候选","",
    "仅建议在新开发批次测试位置坐标适配：把两个坐标通道改为floor(5*row/k)/4和floor(5*col/k)/4，使同一5×5访问桶内坐标对齐训练格点。固定所有其它输入、模型、阈值、B20和导航策略。本轮只冻结草案，未执行、未升级默认。见下一项单因素方案草案.md。","",
    "## 限制与审计","",
    "目标只用于事后曼哈顿方向质量计数，不参与策略输入。另实现复核由同一开发Agent完成，不是独立人员审查。状态和图像特征分别报告距离，没有合成模型隐空间整体距离。","",
    "产物包括输入绑定、诊断汇总、首步/路径明细、独立复核与下一项方案草案。",""]
    with (OUT/"诊断报告.md").open("x",encoding="utf-8",newline="\n") as f:f.write("\n".join(lines))
def analyze():
    b=jread(OUT/"输入绑定.json"); verify_binding(b); tasks,manifests=load_protocol_tasks(); maps,transforms=derive_maps(manifests)
    parsed=read_main(tasks,manifests); first,first_rows,pairs=first_pair_analysis(tasks,transforms,parsed)
    routes=route_summary(parsed,maps,pairs); train=train_contexts(); drift=input_distances(parsed,maps,transforms,train); reach=reachability(tasks,parsed)
    result={"version":"massgis-arrival-drift-summary-v5",
      "input_check":{"eval2_sealed_files_reverified":b["eval2_sealed_file_count"],"trajectory_seal_inventory":b["trajectory_seal_inventory"],"main_log_files":parsed["nfiles"],"records":parsed["nrecords"],"actions":parsed["nactions"],
        "reconstructed_inputs":parsed["nactions"],"input_dimension":1052,"all_input_sha_match":True,"mapped_regions":10,"shared_cells_per_region":100,
        "common_tasks":750,"first_step_pairs":len(first_rows),"model_forward":0,"training":0,"navigation":0,"network":0,"spares":0},
      "bounds_mapping":{a:{"g10_bounds_m":maps[10][a]["region"]["bounds_m"],"g15_bounds_m":maps[15][a]["region"]["bounds_m"],
        "row_offset":0,"column_offset":5,"shared_file_hashes_equal":True,"example_map":{str(i):j for i,j in list(v["map"].items())[:5]}} for a,v in transforms.items()},
      "reachability":reach,"first_step_pairing":first,"input_distribution":drift,
      "trajectory_fork":routes["summary"],
      "sha256":{"input_binding":digest(OUT/"输入绑定.json"),"logs":parsed["log_hashes"],"analysis_sources":b["analysis_source_snapshot_sha256"]}}
    dump_new(OUT/"诊断汇总.json",result); jsonl_new(OUT/"首步配对明细.jsonl",first_rows); jsonl_new(OUT/"路径分叉明细.jsonl",routes["rows"])
    plan="""# 下一项单因素方案草案：15×15位置坐标训练格点适配

状态：EVAL-004只读诊断后形成的开发候选；未启动。本方案不更改EVAL-002/DEV-001模型、均值、阈值、默认或历史验收。

## 唯一假设与因素
Small256 NoTarget训练位置通道为5×5格点row/4、col/4；MassGIS执行输入为row/(k−1)、col/(k−1)。若首步logit差、方向提议与由bounds推导的位置偏移相伴，位置分辨率不匹配是可测试候选机制，而非本次观察已经证明的因果关系。

只变更两个位置坐标通道：floor(5*row/k)/4、floor(5*col/k)/4。固定目标/当前512维特征、25访问桶、剩余预算、合法动作、cue头和0.50阈值、Coverage3Radial/M0、B20及seed0/1/2冻结权重。该输入变换须在新开发批次执行前冻结源码和任务。

## 开发数据与基线
使用已经消费的工程区img_6100的15×15/B20开发任务；任务及距离/接缝层须在执行前哈希冻结。若该区没有匹配可用任务则停止并另行登记，不借用正式区或备用区。
同题基线使用原连续坐标编码下同三权重的Coverage3Radial和M0。仅候选臂应用坐标变换，模型输入权限、特征缓存、目标、调用/行动预算、排序和所有其它策略固定。允许的未来DEV批次是只读冻结权重导航，非本轮运行，不训练。

## 门槛和停止
SR相对同题原策略至少+2百分点；三个权重至少2个正收益；全部终局SG不变差；各冻结距离/接缝层SR损失不超过2百分点；轨迹和输入哈希审计通过。任一门槛不通过即收束，不调阈值、不挑权重、不升级默认。开发通过只可申请后续独立确认，不得直接升级；确认需要新的冻结方案和满足空间隔离样本条件。

## 泄漏与报告
目标位置/索引/距离、来源、分层、未访问图块特征只允许事后诊断，禁止进入策略。SR以计划任务为分母，SG成功记0，另报移动米数；不把权重/重复题计作独立地区。备用区不因本草案自动获准。"""
    with (OUT/"下一项单因素方案草案.md").open("x",encoding="utf-8",newline="\n") as f:f.write(plan+"\n")
    create_report(result)
    print(json.dumps({"analyzed":True,"actions":parsed["nactions"],"first_pairs":len(first_rows),"reachability":reach["all_tasks_by_grid_policy"],
      "quality":first["proposal_quality"],"routes":routes["summary"]["route_categories"],"context_nn":drift["nearest_training_state_by_grid"]},ensure_ascii=False))
def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True);g.add_argument("--freeze-inputs",action="store_true");g.add_argument("--analyze",action="store_true")
    a=p.parse_args()
    if a.freeze_inputs: freeze()
    else: analyze()
if __name__=="__main__": main()

