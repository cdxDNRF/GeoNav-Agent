"""Train/evaluate a small recurrent policy on frozen Sat2Cap features.

The ``bc`` mode is a supervised shortest-path warm start. ``ppo_pbrs`` adds
DynCur-Geo's target-progress potential shaping during training. Target cells
are used only by the trainer to construct labels/rewards; ``policy_features``
contains no goal coordinate or target distance.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.make_episodes import MASA
from env.episode import ACTIONS, Episode, load_episodes, manhattan
from env.environment import GridWorldEnv

ACTIONS_LIST = tuple(ACTIONS)
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS_LIST)}
FEATURE_DIM = 512 + 512 + 2 + 1 + 25


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


class EmbeddingStore:
    """Validated object-dtype dict saved by process_masa.py."""

    def __init__(self, path: Path):
        self.path = Path(path)
        raw = np.load(self.path, allow_pickle=True)
        if raw.shape != () or raw.dtype != object:
            raise ValueError("embedding file must contain a pickled dict")
        data = raw.item()
        if not isinstance(data, dict) or not data:
            raise ValueError("embedding file must contain a non-empty dict")
        self.data = {}
        for area, values in data.items():
            array = np.asarray(values)
            if array.shape != (25, 512) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"invalid embeddings for {area}: expected finite float32 (25,512)")
            self.data[str(area)] = array

    def patch(self, area: str, cell: int) -> np.ndarray:
        if area not in self.data or not 0 <= cell < 25:
            raise KeyError((area, cell))
        return self.data[area][cell]


def policy_features(target: np.ndarray, current: np.ndarray, position,
                    remaining_budget: int, visited) -> np.ndarray:
    """Public inference features; deliberately excludes goal/distance fields."""
    row, col = position
    counts = np.bincount(np.asarray(tuple(visited), dtype=np.int64), minlength=25).astype(np.float32)
    counts = np.minimum(counts, 3.0) / 3.0
    return np.concatenate((target, current,
                           np.asarray([row / 4.0, col / 4.0], dtype=np.float32),
                           np.asarray([remaining_budget / 10.0], dtype=np.float32), counts)).astype(np.float32)


class TinyPolicy(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.input = nn.Sequential(nn.Linear(FEATURE_DIM, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, 4)
        self.critic = nn.Linear(hidden_size, 1)
        self.next_state = nn.Linear(hidden_size, 512)

    def step(self, features, hidden=None):
        x = self.input(features)
        if hidden is None:
            hidden = torch.zeros((features.shape[0], self.hidden_size), device=features.device)
        hidden = self.gru(x, hidden)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), self.next_state(hidden), hidden


@dataclass
class TrainConfig:
    mode: str
    seed: int
    epochs: int
    episodes_per_epoch: int
    learning_rate: float
    pbrs_beta: float
    gamma: float
    curiosity_weight: float
    curiosity_lambda_min: float
    device: str


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_examples(episodes, store):
    features, labels = [], []
    for ep in episodes:
        current = ep.start
        visited = [current]
        remaining = ep.budget
        while current != ep.goal and remaining > 0:
            row, col = divmod(current, ep.grid_size)
            gr, gc = divmod(ep.goal, ep.grid_size)
            if row < gr:
                action = "down"
            elif row > gr:
                action = "up"
            elif col < gc:
                action = "right"
            else:
                action = "left"
            dr, dc = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}[action]
            nr, nc = row + dr, col + dc
            current = nr * ep.grid_size + nc
            visited.append(current)
            remaining -= 1
            features.append(policy_features(store.patch(ep.area, ep.goal), store.patch(ep.area, visited[-2]),
                                             divmod(visited[-2], ep.grid_size), remaining + 1, visited[:-1]))
            labels.append(ACTION_INDEX[action])
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def train_bc(model, episodes, store, cfg, device):
    x, y = teacher_examples(episodes, store)
    features = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(y).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    order = np.arange(len(labels))
    history = []
    for epoch in range(cfg.epochs):
        np.random.shuffle(order)
        losses = []
        for start in range(0, len(order), 512):
            batch = torch.from_numpy(order[start:start + 512]).to(device)
            logits, _, _, _ = model.step(features[batch])
            loss = nn.functional.cross_entropy(logits, labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": statistics.mean(losses), "teacher_samples": len(labels)})
    return history


def _transition_reward(ep, before, after, done, cfg):
    def distance(value):
        if isinstance(value, tuple):
            row, col = value
            goal_row, goal_col = divmod(ep.goal, ep.grid_size)
            return abs(row - goal_row) + abs(col - goal_col)
        return manhattan(value, ep.goal, ep.grid_size)
    before_d = distance(before)
    after_d = distance(after)
    progress = float(before_d - after_d)
    potential_before = -before_d / max(2 * (ep.grid_size - 1), 1)
    # The finite-budget terminal state is absorbing for the discounted return.
    potential_after = 0.0 if done else -after_d / max(2 * (ep.grid_size - 1), 1)
    pbrs = cfg.gamma * potential_after - potential_before
    return float(done and after_d == 0) + .1 * progress + cfg.pbrs_beta * pbrs


def curiosity_gate(ep, position, cfg):
    """DynCur-style gate: curiosity is stronger far from the target."""
    row, col = position
    goal_row, goal_col = divmod(ep.goal, ep.grid_size)
    distance = abs(row - goal_row) + abs(col - goal_col)
    rho = min(max(distance / max(ep.dist, 1), 0.0), 1.0)
    return cfg.curiosity_lambda_min + (1.0 - cfg.curiosity_lambda_min) * rho


def collect_ppo(model, episodes, store, env, cfg, device, rng):
    rows = []
    chosen = episodes if cfg.episodes_per_epoch <= 0 else [episodes[i] for i in rng.sample(range(len(episodes)), min(cfg.episodes_per_epoch, len(episodes)))]
    model.eval()
    for ep in chosen:
        obs = env.reset(ep)
        hidden = None
        while not env.done:
            current = obs.position[0] * ep.grid_size + obs.position[1]
            features = policy_features(store.patch(ep.area, ep.goal), store.patch(ep.area, current),
                                       obs.position, obs.remaining_budget, obs.visited)
            tensor = torch.from_numpy(features).to(device).unsqueeze(0)
            with torch.no_grad():
                logits, value, prediction, next_hidden = model.step(tensor, hidden)
                dist = Categorical(logits=logits)
                action_index = int(dist.sample().item())
                log_prob = float(dist.log_prob(torch.tensor([action_index], device=device)).item())
            action = ACTIONS_LIST[action_index]
            before = obs.position
            next_obs, _, _ = env.step(action)
            after = next_obs.position
            done = env.done
            next_cell = after[0] * ep.grid_size + after[1]
            target_next = store.patch(ep.area, next_cell).copy()
            prediction_error = float(np.mean((prediction.detach().cpu().numpy().squeeze(0) - target_next) ** 2))
            curiosity_reward = cfg.curiosity_weight * curiosity_gate(ep, before, cfg) * prediction_error
            rows.append({"features": features, "hidden": hidden.detach().cpu().numpy().squeeze(0) if hidden is not None else np.zeros(model.hidden_size, dtype=np.float32),
                         "action": action_index, "old_log_prob": log_prob, "old_value": float(value.item()),
                         "reward": _transition_reward(ep, before, after, done, cfg) + curiosity_reward, "done": done,
                         "curiosity_reward": curiosity_reward, "prediction_error": prediction_error,
                         "target_next": target_next,
                         "prediction": prediction.detach().cpu().numpy().squeeze(0), "success": bool(done and next_cell == ep.goal)})
            obs = next_obs
            hidden = next_hidden.detach()
    return rows


def train_ppo(model, episodes, store, env, cfg, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    rng = random.Random(cfg.seed)
    history = []
    for epoch in range(cfg.epochs):
        rows = collect_ppo(model, episodes, store, env, cfg, device, rng)
        if not rows:
            raise ValueError("no PPO transitions")
        features = torch.from_numpy(np.stack([r["features"] for r in rows])).to(device)
        hidden = torch.from_numpy(np.stack([r["hidden"] for r in rows])).to(device)
        actions = torch.tensor([r["action"] for r in rows], device=device)
        old_log_probs = torch.tensor([r["old_log_prob"] for r in rows], device=device)
        old_values = torch.tensor([r["old_value"] for r in rows], device=device)
        rewards = torch.tensor([r["reward"] for r in rows], device=device)
        next_targets = torch.from_numpy(np.stack([r["target_next"] for r in rows])).to(device)
        with torch.no_grad():
            returns = rewards.clone()
            running = torch.tensor(0.0, device=device)
            for i in range(len(rows) - 1, -1, -1):
                running = rewards[i] + (0.0 if rows[i]["done"] else cfg.gamma * running)
                returns[i] = running
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        model.train()
        order = torch.randperm(len(rows), device=device)
        losses = []
        for start in range(0, len(rows), 512):
            idx = order[start:start + 512]
            logits, values, prediction, _ = model.step(features[idx], hidden[idx])
            dist = Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(actions[idx]) - old_log_probs[idx])
            clipped = torch.clamp(ratio, .8, 1.2) * advantages[idx]
            policy_loss = -torch.minimum(ratio * advantages[idx], clipped).mean()
            value_loss = .5 * (values - returns[idx]).square().mean()
            pred_loss = (.05 * nn.functional.mse_loss(prediction, next_targets[idx])
                         if cfg.curiosity_weight > 0 else torch.zeros((), device=device))
            loss = policy_loss + value_loss + pred_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": statistics.mean(losses),
                        "transitions": len(rows), "successes": sum(r["success"] for r in rows)})
    return history


@torch.no_grad()
def evaluate(model, split, episodes, store, env, device):
    records = []
    model.eval()
    for ep in episodes:
        obs = env.reset(ep)
        hidden = None
        while not env.done:
            current = obs.position[0] * ep.grid_size + obs.position[1]
            features = policy_features(store.patch(ep.area, ep.goal), store.patch(ep.area, current),
                                       obs.position, obs.remaining_budget, obs.visited)
            logits, _, _, hidden = model.step(torch.from_numpy(features).to(device).unsqueeze(0), hidden)
            action = ACTIONS_LIST[int(torch.argmax(logits, dim=-1).item())]
            obs, _, _ = env.step(action)
        result = env.evaluator_result()
        records.append(result)
    return {"split": split, "episodes": len(records), "successes": sum(r["success"] for r in records),
            "sr": sum(r["success"] for r in records) / len(records),
            "sg": statistics.mean(r["sg"] for r in records),
            "repeat_visit_rate": statistics.mean(r["repeat_visit_rate"] for r in records),
            "by_distance": {str(d): {"episodes": sum(r["episode_id"].split("_d")[1].split("_")[0] == str(d) for r in records),
                                      "sr": statistics.mean([r["success"] for r in records if f"_d{d}_" in r["episode_id"]]) if any(f"_d{d}_" in r["episode_id"] for r in records) else None,
                                      "sg": statistics.mean([r["sg"] for r in records if f"_d{d}_" in r["episode_id"]]) if any(f"_d{d}_" in r["episode_id"] for r in records) else None}
                             for d in range(4, 9)}}


def load_episodes_file(path):
    return load_episodes(Path(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=MASA)
    parser.add_argument("--mode", choices=("bc", "ppo_pbrs", "ppo_pbrs_curiosity"), default="bc")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--episodes-per-epoch", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--pbrs-beta", type=float, default=.5)
    parser.add_argument("--curiosity-weight", type=float, default=.1)
    parser.add_argument("--curiosity-lambda-min", type=float, default=.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    root = Path(args.dataset_root)
    train_eps = load_episodes_file(root / "任务清单_v2/episodes_train.jsonl")
    val_eps = load_episodes_file(root / "任务清单_v2/episodes_val.jsonl")
    store_train = EmbeddingStore(root / "papr_train_sat_embeds_grid_5.npy")
    store_val = EmbeddingStore(root / "papr_val_sat_embeds_grid_5.npy")
    model = TinyPolicy().to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True))
    cfg = TrainConfig(args.mode, args.seed, args.epochs, args.episodes_per_epoch,
                      args.learning_rate, args.pbrs_beta, .99,
                      args.curiosity_weight if args.mode == "ppo_pbrs_curiosity" else 0.0,
                      args.curiosity_lambda_min, str(device))
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refuse to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    env = GridWorldEnv(root)
    history = train_bc(model, train_eps, store_train, cfg, device) if args.mode == "bc" else train_ppo(model, train_eps, store_train, env, cfg, device)
    checkpoint = output / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    evaluation = evaluate(model, "val", val_eps, store_val, env, device)
    result = {"version": "tiny-dyncur-v1", "config": asdict(cfg),
              "model": {"class": "TinyPolicy", "parameters": sum(p.numel() for p in model.parameters()),
                        "feature_dim": FEATURE_DIM, "inference_goal_fields": False},
              "data": {"train_task_sha256": digest(root / "任务清单_v2/episodes_train.jsonl"),
                       "val_task_sha256": digest(root / "任务清单_v2/episodes_val.jsonl"),
                       "train_embedding_sha256": digest(root / "papr_train_sat_embeds_grid_5.npy"),
                       "val_embedding_sha256": digest(root / "papr_val_sat_embeds_grid_5.npy"),
                       "train_episodes": len(train_eps), "val_episodes": len(val_eps)},
              "history": history, "val": evaluation, "checkpoint": checkpoint.name,
              "target_used_only_for_training": True}
    (output / "训练结果.json").write_bytes(json_bytes(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
