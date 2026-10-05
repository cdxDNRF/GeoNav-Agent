"""四动作局部观测环境。评测端持有任务真值；策略仅收 Observation。"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from .episode import ACTIONS, Episode, inspect_area, manhattan


@dataclass(frozen=True)
class Observation:
    current_image: bytes
    target_image: bytes
    position: tuple[int, int]
    grid_size: int
    remaining_budget: int
    visited: tuple[int, ...]
    legal_actions: tuple[str, ...] = tuple(ACTIONS)
    media_type: str = "image/png"

    def load_image(self) -> Image.Image:
        with Image.open(BytesIO(self.current_image)) as image:
            return image.convert("RGB")


@dataclass(frozen=True)
class StepInfo:
    out_of_bounds: bool
    revisited: bool
    step_count: int


def image_payload(path: Path) -> bytes:
    # 重编码像素以剥离路径、EXIF、PNG 文本及其他原始文件元数据。
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    stream = BytesIO()
    clean.save(stream, format="PNG")
    return stream.getvalue()


class GridWorldEnv:
    def __init__(self, dataset_root: Path):
        self._root = Path(dataset_root)
        self._episode = None
        self._done = True
        self._validated_areas = {}
        self._image_cache = {}

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, episode: Episode | dict) -> Observation:
        ep = Episode.from_dict(episode) if isinstance(episode, dict) else episode
        if not isinstance(ep, Episode):
            raise ValueError("需要 Episode 或任务字典")
        ep.validate()
        key = (ep.split, ep.area)
        signature = self._area_signature(ep)
        if key not in self._validated_areas:
            inspect_area(self._root, *key)
            self._validated_areas[key] = signature
        elif self._validated_areas[key] != signature:
            raise ValueError("环境实例使用不可变数据快照；区域文件已变化，请重新校验数据")
        self._episode = ep
        self._position = ep.start
        self._remaining = ep.budget
        self._visited = [ep.start]
        self._trajectory = [{"step": 0, "patch_id": ep.start, "action": None,
                             "out_of_bounds": False, "revisited": False}]
        self._done = False
        self._success = False
        self._target = self._payload(ep.goal)
        return self._obs()

    def step(self, action: str) -> tuple[Observation, bool, StepInfo]:
        if self._episode is None or self._done:
            raise RuntimeError("请 reset 一个未结束的 episode")
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError(f"仅支持四方向动作: {tuple(ACTIONS)}")
        dr, dc = ACTIONS[action]
        row, col = divmod(self._position, 5)
        nr, nc = row + dr, col + dc
        outside = not (0 <= nr < 5 and 0 <= nc < 5)
        if not outside:
            self._position = nr * 5 + nc
        revisit = self._position in self._visited
        self._remaining -= 1
        self._visited.append(self._position)
        count = self._episode.budget - self._remaining
        self._trajectory.append({"step": count, "patch_id": self._position,
                                 "action": action, "out_of_bounds": outside,
                                 "revisited": revisit})
        self._success = self._position == self._episode.goal
        self._done = self._success or self._remaining == 0
        return self._obs(), self._done, StepInfo(outside, revisit, count)

    def evaluator_result(self) -> dict:
        """仅评测器在终止后调用，返回值不能传给策略。"""
        if self._episode is None or not self._done:
            raise RuntimeError("只允许在 episode 终止后取得评测结果")
        steps = len(self._trajectory) - 1
        revisits = sum(event["revisited"] for event in self._trajectory[1:])
        return {"episode_id": self._episode.episode_id,
                "success": self._success,
                "termination": "goal_reached" if self._success else "budget_exhausted",
                "sg": manhattan(self._position, self._episode.goal),
                "steps": steps, "revisits": revisits,
                "repeat_visit_rate": revisits / steps,
                "out_of_bounds": sum(t["out_of_bounds"] for t in self._trajectory),
                "trajectory": [dict(event) for event in self._trajectory]}

    def _patch_path(self, pid: int) -> Path:
        ep = self._episode
        return self._root / "patches" / ep.split / ep.area / f"patch_{pid}.jpg"

    def _area_signature(self, ep: Episode):
        folder = self._root / "patches" / ep.split / ep.area
        return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                     for p in sorted(folder.glob("patch_*")))

    def _payload(self, pid: int) -> bytes:
        path = self._patch_path(pid)
        stat = path.stat()
        key = (path.name, stat.st_size, stat.st_mtime_ns)
        if key not in self._validated_areas[(self._episode.split, self._episode.area)]:
            raise ValueError("图块在运行期间被修改")
        if path not in self._image_cache:
            self._image_cache[path] = image_payload(path)
        return self._image_cache[path]

    def _obs(self) -> Observation:
        return Observation(self._payload(self._position), self._target,
                           divmod(self._position, 5), 5, self._remaining,
                           tuple(self._visited))
