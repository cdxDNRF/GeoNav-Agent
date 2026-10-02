"""Auditable, equal-step local curiosity ablations; train and val only.

This is an adaptation, not a reproduction of GeoExplorer/DynCur. A separate
frozen action-conditioned dynamics head uses a frozen copy of the initial GRU.
It is pretrained on train-source random walks and calibrated on other train
sources. PPO replays recurrent sequences and receives only public features.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.dyncur_tiny import (TinyPolicy, EmbeddingStore, policy_features,
                               set_seed, digest, json_bytes, FEATURE_DIM)
from data.make_episodes import MASA
from env.environment import GridWorldEnv
from env.episode import ACTIONS
from eval.evaluate import verify_task_file, metrics

SRC = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = MASA / "训练结果/好奇心配对验证_v3"
ARMS = ("PBRS", "Curiosity", "GatedCuriosity")


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(data)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refuse to overwrite {path}")
        return
    with path.open("xb") as f:
        f.write(payload)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def training_reward(before, after, goal, done, beta=.5, gamma=.99):
    """Private trainer-only reward. Cell IDs for every argument, never tuples."""
    d0 = np.abs(before // 5 - goal // 5) + np.abs(before % 5 - goal % 5)
    d1 = np.abs(after // 5 - goal // 5) + np.abs(after % 5 - goal % 5)
    success = np.asarray(done) & (after == goal)
    external = success.astype(np.float32) + .1 * (d0 - d1)
    phi_next = np.where(done, 0.0, -d1 / 8.0)
    pbrs = beta * (gamma * phi_next + d0 / 8.0)
    return external.astype(np.float32), pbrs.astype(np.float32), success


def move(cells, actions):
    deltas = np.asarray(list(ACTIONS.values()), dtype=np.int64)[actions]
    rows, cols = cells // 5 + deltas[:, 0], cells % 5 + deltas[:, 1]
    wall = (rows < 0) | (rows >= 5) | (cols < 0) | (cols >= 5)
    return np.where(wall, cells, rows * 5 + cols), wall


class FeatureWorld:
    """Trainer-owned vector environment. The neural network sees observe() only.

    Dynamics match GridWorldEnv; images are replaced with frozen local features.
    No boundary mask, no visit governor, no change to the four-action protocol.
    """
    def __init__(self, store, count, seed):
        self.areas = sorted(store.data)
        self.table = np.stack([store.data[a] for a in self.areas])
        self.count = count
        self.rng = np.random.default_rng(seed)
        self.area = np.zeros(count, np.int64)
        self.position = np.zeros(count, np.int64)
        self.goal = np.zeros(count, np.int64)
        self.initial_distance = np.zeros(count, np.int64)
        self.steps = np.zeros(count, np.int64)
        self.visits = np.zeros((count, 25), np.float32)
        self.reset(np.ones(count, bool))

    def reset(self, mask):
        for i in np.flatnonzero(mask):
            self.area[i] = self.rng.integers(len(self.areas))
            # Dynamic pairs C=4..8, uniform distance then uniform ordered pair.
            distance = int(self.rng.integers(4, 9))
            pairs = PAIRS[distance]
            self.position[i], self.goal[i] = pairs[self.rng.integers(len(pairs))]
            self.initial_distance[i] = distance
            self.steps[i] = 0
            self.visits[i] = 0
            self.visits[i, self.position[i]] = 1

    def observe(self):
        return np.concatenate((self.table[self.area, self.goal],
                               self.table[self.area, self.position],
                               self.position[:, None] // 5 / 4,
                               self.position[:, None] % 5 / 4,
                               (10 - self.steps[:, None]) / 10,
                               np.minimum(self.visits, 3) / 3), axis=1).astype(np.float32)

    def step(self, actions):
        before = self.position.copy()
        self.position, wall = move(before, actions)
        revisited = self.visits[np.arange(self.count), self.position] > 0
        self.visits[np.arange(self.count), self.position] += 1
        self.steps += 1
        done = (self.position == self.goal) | (self.steps == 10)
        external, pbrs, success = training_reward(before, self.position, self.goal, done)
        distance = np.abs(before // 5 - self.goal // 5) + np.abs(before % 5 - self.goal % 5)
        gate = .405 + .595 * np.clip(distance / self.initial_distance, 0, 1)
        return {"done": done, "success": success, "external": external, "pbrs": pbrs,
                "gate": gate.astype(np.float32), "wall": wall, "revisited": revisited,
                "next_feature": self.table[self.area, self.position].copy()}


PAIRS = {d: [(a, b) for a in range(25) for b in range(25)
             if abs(a//5-b//5) + abs(a%5-b%5) == d] for d in range(4, 9)}


class Dynamics(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.encoder = copy.deepcopy(initial).eval().requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(initial.hidden_size + 4, 256), nn.Tanh(), nn.Linear(256, 512))

    def predict(self, hidden, actions):
        action_features = nn.functional.one_hot(actions.long(), 4).to(hidden.dtype)
        return self.head(torch.cat((hidden, action_features), dim=-1))


def normalize_error(error, scale, cap=3.0):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalizer must be a positive train-only scalar")
    return np.clip(error / scale, 0, cap)


def tensor(array, device):
    return torch.as_tensor(array, device=device)


@torch.no_grad()
def dynamics_examples(model, store, seed, count, device):
    world = FeatureWorld(store, 64, seed)
    h = None
    xs, acts, ys, current = [], [], [], []
    rng = np.random.default_rng(seed + 17)
    for _ in range(count // 64):
        features = world.observe()
        _, _, _, h = model.encoder.step(tensor(features, device), h)
        actions = rng.integers(0, 4, 64)
        info = world.step(actions)
        xs.append(h.clone())
        acts.append(tensor(actions, device))
        ys.append(tensor(info["next_feature"], device))
        current.append(tensor(features[:, 512:1024], device))
        h = h * tensor(~info["done"], device)[:, None]
        world.reset(info["done"])
    return torch.cat(xs), torch.cat(acts), torch.cat(ys), torch.cat(current)


def pretrain_dynamics(initial, store, out, device):
    set_seed(1701)
    model = Dynamics(initial).to(device)
    areas = sorted(store.data)
    fit_store, calibration_store = copy.copy(store), copy.copy(store)
    calibration_areas = areas[::5]
    fit_store.data = {a: store.data[a] for a in areas if a not in calibration_areas}
    calibration_store.data = {a: store.data[a] for a in calibration_areas}
    x, a, y, _ = dynamics_examples(model, fit_store, 1701, 32768, device)
    cx, ca, cy, cc = dynamics_examples(model, calibration_store, 1702, 8192, device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-4)
    history = []
    for epoch in range(20):
        losses = []
        for idx in torch.randperm(len(x), device=device).split(512):
            loss = nn.functional.mse_loss(model.predict(x[idx], a[idx]), y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        history.append(float(torch.stack(losses).mean()))
    with torch.no_grad():
        errors = (model.predict(cx, ca) - cy).square().mean(-1)
        scale = float(errors.mean().clamp_min(1e-8))
        persistence = float((cc - cy).square().mean())
    model.eval().requires_grad_(False)
    torch.save(model.state_dict(), out / "冻结动力学.pt")
    details = {"training_sources": len(fit_store.data), "calibration_sources": len(calibration_store.data),
               "calibration_area_ids": calibration_areas, "train_only": True,
               "pretrain_samples": len(x), "calibration_samples": len(cx), "epochs": 20,
               "loss_history": history, "normalization_mean_mse": scale,
               "calibration_persistence_mse": persistence,
               "calibration_vs_persistence_improved": scale < persistence,
               "frozen_encoder": True, "frozen_predictor_during_PPO": True,
               "action_conditioned": True, "checkpoint_sha256": digest(out / "冻结动力学.pt")}
    write_new(out / "动力学预训练.json", details)
    print(json.dumps({"dynamics_calibration_mse": scale, "persistence_mse": persistence}), flush=True)
    return model, scale


def gae(rewards, values, dones, next_value, gamma=.99, lam=.95):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        future_value = next_value if t == len(rewards)-1 else values[t+1]
        keep = (~dones[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * future_value * keep - values[t]
        running = delta + gamma * lam * keep * running
        advantages[t] = running
    return advantages, advantages + values


def recurrent_logits(model, features, starts, hidden):
    logits, values = [], []
    for t in range(len(features)):
        hidden = hidden * (~starts[t])[:, None]
        pi, v, _, hidden = model.step(features[t], hidden)
        logits.append(pi)
        values.append(v)
    return torch.stack(logits), torch.stack(values)


def train_arm(initial, dynamics, scale, store, directory, arm, seed, device, updates=64):
    set_seed(seed)
    model = copy.deepcopy(initial).to(device)
    # The historical prediction head is unused; no auxiliary-loss gradient reaches policy.
    model.next_state.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    world = FeatureWorld(store, 64, seed)
    hidden = torch.zeros(64, model.hidden_size, device=device)
    dyn_hidden = torch.zeros_like(hidden)
    start = np.ones(64, bool)
    for update in range(updates):
        model.eval()
        hx = hidden.detach().clone()
        features, starts, values, actions, old_logs, rewards, dones = [], [], [], [], [], [], []
        ext_logs, pb_logs, int_logs, success_logs, error_logs = [], [], [], [], []
        with torch.no_grad():
            for _ in range(10):
                x = tensor(world.observe(), device)
                begins = tensor(start, device)
                hidden *= (~begins)[:, None]
                dyn_hidden *= (~begins)[:, None]
                logits, value, _, hidden = model.step(x, hidden)
                _, _, _, dyn_hidden = dynamics.encoder.step(x, dyn_hidden)
                distribution = Categorical(logits=logits)
                chosen = distribution.sample()
                prediction = dynamics.predict(dyn_hidden, chosen)
                info = world.step(chosen.cpu().numpy())
                error = (prediction - tensor(info["next_feature"], device)).square().mean(-1).cpu().numpy()
                intrinsic = .1 * normalize_error(error, scale) if arm != "PBRS" else np.zeros(64, np.float32)
                if arm == "GatedCuriosity":
                    intrinsic *= info["gate"]
                features.append(x); starts.append(begins); values.append(value)
                actions.append(chosen); old_logs.append(distribution.log_prob(chosen))
                rewards.append(tensor(info["external"] + info["pbrs"] + intrinsic, device))
                dones.append(tensor(info["done"], device))
                ext_logs.extend(info["external"]); pb_logs.extend(info["pbrs"])
                int_logs.extend(intrinsic); success_logs.extend(info["success"]); error_logs.extend(error)
                world.reset(info["done"])
                start = info["done"]
            _, future_value, _, _ = model.step(tensor(world.observe(), device), hidden * tensor(~start, device)[:, None])
            features, starts, values, actions, old_logs, rewards, dones = [torch.stack(v) for v in (features, starts, values, actions, old_logs, rewards, dones)]
            advantage, returns = gae(rewards, values, dones, future_value)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        model.train()
        losses = []
        for _ in range(2):
            for idx in torch.randperm(64, device=device).split(16):
                pi, v = recurrent_logits(model, features[:, idx], starts[:, idx], hx[idx])
                dist = Categorical(logits=pi)
                ratio = (dist.log_prob(actions[:, idx]) - old_logs[:, idx]).exp()
                policy_loss = -torch.minimum(ratio * advantage[:, idx], ratio.clamp(.8, 1.2) * advantage[:, idx]).mean()
                loss = policy_loss + .5 * (v - returns[:, idx]).square().mean() - .01 * dist.entropy().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss.detach()))
        # Recompute carry state under updated parameters on the retained rollout.
        with torch.no_grad():
            hidden = hx.clone()
            for t in range(10):
                _, _, _, hidden = model.step(features[t], hidden * (~starts[t])[:, None])
        log = {"update": update+1, "steps": (update+1)*640, "successes": int(sum(success_logs)),
               "external_mean": float(np.mean(ext_logs)), "external_abs_mean": float(np.mean(np.abs(ext_logs))),
               "pbrs_mean": float(np.mean(pb_logs)), "intrinsic_mean": float(np.mean(int_logs)),
               "prediction_mse_mean": float(np.mean(error_logs)), "loss": float(np.mean(losses))}
        append(directory / "训练日志.jsonl", log)
        if (update+1) % 16 == 0:
            print(json.dumps({"arm": arm, "seed": seed, **log}), flush=True)
    torch.save(model.state_dict(), directory / "model.pt")
    return model


@torch.no_grad()
def evaluate_and_audit(model, root, episodes, store, directory, device):
    env = GridWorldEnv(root)
    records = []
    model.eval()
    for ep in episodes:
        obs = env.reset(ep)
        hidden = None
        # Evaluator supplies target features exactly as GridWorldEnv supplies target pixels.
        target = store.patch(ep.area, ep.goal).copy()
        while not env.done:
            cell = obs.position[0]*5 + obs.position[1]
            public = policy_features(target, store.patch(ep.area, cell), obs.position, obs.remaining_budget, obs.visited)
            logits, _, _, hidden = model.step(tensor(public, device)[None], hidden)
            obs, _, _ = env.step(tuple(ACTIONS)[int(logits.argmax(-1))])
        result = env.evaluator_result()
        result.update(area=ep.area, distance=ep.dist)
        records.append(result)
        append(directory / "val轨迹.jsonl", result)
    # Second pass replays both the neural decisions and all environment transitions.
    by_id = {e.episode_id:e for e in episodes}
    for record in records:
        ep = by_id[record["episode_id"]]
        obs = env.reset(ep)
        hidden = None
        for event in record["trajectory"][1:]:
            current = obs.position[0]*5 + obs.position[1]
            public = policy_features(store.patch(ep.area, ep.goal), store.patch(ep.area, current), obs.position, obs.remaining_budget, obs.visited)
            logits, _, _, hidden = model.step(tensor(public, device)[None], hidden)
            if tuple(ACTIONS)[int(logits.argmax(-1))] != event["action"]:
                raise ValueError("neural replay mismatch")
            obs, _, _ = env.step(event["action"])
        if any(record[k] != v for k, v in env.evaluator_result().items()):
            raise ValueError("trajectory/metrics mismatch")
    result = {"completed": len(records), "planned": len(episodes), "metrics": metrics(records),
              "by_distance": {str(d):metrics([r for r in records if r["distance"]==d]) for d in range(4,9)},
              "by_area": {a:metrics([r for r in records if r["area"]==a]) for a in sorted(store.data)},
              "audit": {"neural_replay": True, "environment_replay": True,
                        "trajectory_sha256": digest(directory / "val轨迹.jsonl"),
                        "checkpoint_sha256": digest(directory / "model.pt")}}
    write_new(directory / "验证结果.json", result)
    return result


def compare(results):
    summary = {}
    for arm in ARMS:
        runs = [results[f"{arm}_s{s}"] for s in (0,1,2)]
        ms = [r["metrics"] for r in runs]
        summary[arm] = {"sr_mean": float(np.mean([m["sr"] for m in ms])),
                        "sr_by_seed": [m["sr"] for m in ms],
                        "sg_mean": float(np.mean([m["mean_sg_all_episodes"] for m in ms])),
                        "repeat_micro_mean": float(np.mean([m["repeat_visit_rate_micro"] for m in ms])),
                        "q": min(r["completed"]/r["planned"] for r in runs)}
    comparisons = {}
    for a,b in (("Curiosity","PBRS"),("GatedCuriosity","Curiosity")):
        sr = summary[a]["sr_mean"]-summary[b]["sr_mean"]
        sg = summary[a]["sg_mean"]-summary[b]["sg_mean"]
        wins = sum(x>y+1e-12 for x,y in zip(summary[a]["sr_by_seed"], summary[b]["sr_by_seed"]))
        comparisons[f"{a}_minus_{b}"] = {"sr_difference":sr, "sg_difference":sg, "positive_seeds":wins,
                "candidate_signal": sr>=.05-1e-12 and sg<=1e-12 and wins>=2}
    return {"arms": summary, "paired_comparisons": comparisons,
            "S2_passed": False, "note": "Local 4-source validation, three continuation seeds from one historical checkpoint; not a formal S2 replacement or independent pretraining repeats."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root, out, device = MASA, args.output_dir, torch.device(args.device)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "预登记.json").exists():
        raise FileExistsError("registered batch already exists; keep the original and choose a new directory")
    train_episodes, train_manifest = verify_task_file(root, root / "任务清单_v2/episodes_train.jsonl", "train")
    val_episodes, val_manifest = verify_task_file(root, root / "任务清单_v2/episodes_val.jsonl", "val")
    train, val = [EmbeddingStore(root/f"papr_{s}_sat_embeds_grid_5.npy") for s in ("train","val")]
    for episodes, store in ((train_episodes, train), (val_episodes,val)):
        if set(store.data) != {e.area for e in episodes}:
            raise ValueError("embedding/source mapping mismatch")
    if {e.source_tile for e in train_episodes} & {e.source_tile for e in val_episodes}:
        raise ValueError("train/val source overlap")
    initial_path = root / "训练结果/tiny_pbrs_v2/model.pt"
    initial = TinyPolicy().to(device)
    initial.load_state_dict(torch.load(initial_path, map_location=device, weights_only=True))
    torch.save(initial.state_dict(), out / "共同初始化.pt")
    source_files = sorted([p for p in SRC.rglob("*.py") if p.relative_to(SRC).parts[0] in ("train","env","eval","agents","data","tests")])
    source_hash = {p.relative_to(SRC).as_posix():digest(p) for p in source_files}
    for p in source_files:
        target = out/"源码快照"/p.relative_to(SRC)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    registration = {"version":"controlled-curiosity-v3", "utc":datetime.now(timezone.utc).isoformat(),
        "arms":list(ARMS), "seeds":[0,1,2], "env_steps_per_run":40960, "rollout_updates":64,
        "runtime":{"torch":torch.__version__,"python":sys.version,"device":str(device),
                   "cuda":torch.version.cuda,"deterministic_algorithms":True,
                   "CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "PPO":{"gamma":.99,"gae_lambda":.95,"clip":.2,"lr":.0003,"passes":2,"entropy":.01,"vector_envs":64,"horizon":10},
        "reward":{"success":1,"progress_coefficient":.1,"pbrs_beta":.5,"terminal_potential":0,
                  "curiosity_weight":.1,"gate_min":.405,"normalize":"train calibration mean MSE; clip to [0,3]"},
        "initial_checkpoint_sha256":digest(initial_path), "initial_checkpoint":str(initial_path),
        "source_sha256":source_hash,
        "data_sha256":{f: digest(root/f) for f in ("papr_train_sat_embeds_grid_5.npy","papr_val_sat_embeds_grid_5.npy","任务清单_v2/episodes_train.jsonl","任务清单_v2/episodes_val.jsonl","metadata.csv")},
        "train_area_sha256":train_manifest["area_sha256"], "val_area_sha256":val_manifest["area_sha256"],
        "gate_rule":"candidate only if paired SR +5pp, SG no worse, positive SR in >=2/3 seeds",
        "historical_numbers":"not causal baselines: missing arrival reward, joint changes, no equal-budget control",
        "scope":"development repair experiment; no test, no cloud, no scale expansion; no hyperparameter sweep",
        "limitations":"same legacy warm start for all runs; no full DynCur sequence-model reproduction; 4 val sources; 86 unique routes"}
    write_new(out/"预登记.json",registration)
    start_time=time.monotonic()
    try:
        dynamics, scale = pretrain_dynamics(initial, train, out, device)
        frozen_hash=digest(out/"冻结动力学.pt")
        results={}
        for seed in (0,1,2):
            for arm in ARMS:
                directory=out/f"{arm}_s{seed}"
                directory.mkdir(exist_ok=False)
                write_new(directory/"配置.json",{"arm":arm,"seed":seed,"registration_sha256":digest(out/"预登记.json"),"dynamics_sha256":frozen_hash})
                model=train_arm(initial,dynamics,scale,train,directory,arm,seed,device)
                result=evaluate_and_audit(model,root,val_episodes,val,directory,device)
                results[directory.name]=result
                print(json.dumps({"run":directory.name,"SR":result["metrics"]["sr"],"SG":result["metrics"]["mean_sg_all_episodes"]}),flush=True)
                del model
        if any(digest(SRC/name)!=value for name,value in source_hash.items()):
            raise ValueError("source modified during experiment")
        for f, value in registration["data_sha256"].items():
            if digest(root/f)!=value: raise ValueError("data modified during experiment")
        # Freeze check covers every dynamics tensor as well as requires_grad flags.
        saved=torch.load(out/"冻结动力学.pt",map_location=device,weights_only=True)
        if any(not torch.equal(v,saved[k]) for k,v in dynamics.state_dict().items()) or any(p.requires_grad for p in dynamics.parameters()):
            raise ValueError("dynamics was updated during policy training")
        final=compare(results)
        final.update(elapsed_seconds=time.monotonic()-start_time, source_verified=True, dynamics_frozen_verified=True)
        write_new(out/"配对汇总.json",final)
        write_new(out/"执行状态.json",{"status":"completed","runs":9,"val_episodes":900,"training_env_steps":368640})
    except Exception as exc:
        write_new(out/"执行异常.json",{"type":type(exc).__name__,"error":str(exc)})
        raise


if __name__ == "__main__":
    main()
