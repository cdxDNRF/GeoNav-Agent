"""评测端任务与图块清单校验；这些对象不交给策略。"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re

from PIL import Image

PROTOCOL = "masa-local-grid5-v2"
SPLITS = ("train", "val", "test")
ACTIONS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


def manhattan(a: int, b: int, grid: int = 5) -> int:
    ar, ac = divmod(a, grid)
    br, bc = divmod(b, grid)
    return abs(ar - br) + abs(ac - bc)


def safe_area(area: str) -> str:
    if not isinstance(area, str) or re.fullmatch(r"img_\d+", area) is None:
        raise ValueError("area 必须是 img_<整数>，不能包含路径")
    return area


@dataclass(frozen=True)
class Episode:
    episode_id: str
    split: str
    area: str
    start: int
    goal: int
    dist: int
    budget: int = 10
    grid_size: int = 5
    protocol: str = PROTOCOL
    source_tile: str = ""

    @classmethod
    def from_dict(cls, row: dict):
        fields = cls.__dataclass_fields__
        required = ("episode_id", "split", "area", "start", "goal", "dist")
        if any(key not in row for key in required):
            raise ValueError("episode 缺少必需字段")
        allowed = set(fields) | {"start_rc", "goal_rc"}
        if set(row) - allowed:
            raise ValueError("episode 包含未知字段")
        ep = cls(**{k: v for k, v in row.items() if k in fields})
        ep.validate()
        for key, value in (("start_rc", ep.start), ("goal_rc", ep.goal)):
            if key in row and row[key] != list(divmod(value, ep.grid_size)):
                raise ValueError(f"{key} 与格编号不一致")
        return ep

    def validate(self):
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id 不能为空")
        if self.split not in SPLITS or self.protocol != PROTOCOL:
            raise ValueError("未知 split 或协议版本")
        safe_area(self.area)
        for key in ("start", "goal", "dist", "budget", "grid_size"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} 必须是整数")
        if self.grid_size != 5:
            raise ValueError("当前图块仅支持已验证的 5×5，不可只改网格参数")
        if not 0 <= self.start < 25 or not 0 <= self.goal < 25:
            raise ValueError("起终点越界")
        if self.start == self.goal or self.budget <= 0:
            raise ValueError("起终点必须不同且预算为正")
        if self.dist != manhattan(self.start, self.goal):
            raise ValueError("dist 与起终点不一致")
        if not isinstance(self.source_tile, str):
            raise ValueError("source_tile 必须为字符串")


def inspect_area(root: Path, split: str, area: str) -> str:
    folder = Path(root) / "patches" / split / safe_area(area)
    if split not in SPLITS:
        raise ValueError("非法 split")
    expected = {f"patch_{i}.jpg" for i in range(25)}
    if not folder.is_dir() or {p.name for p in folder.glob("patch_*")} != expected:
        raise ValueError(f"图块缺失或数量错误: {folder}")
    digest = sha256()
    for i in range(25):
        path = folder / f"patch_{i}.jpg"
        with Image.open(path) as image:
            image.load()
            if image.size != (300, 300) or image.mode != "RGB":
                raise ValueError(f"图块应为 300×300 RGB: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def load_episodes(path: Path) -> list[Episode]:
    episodes = []
    ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            try:
                ep = Episode.from_dict(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if ep.episode_id in ids:
                raise ValueError(f"重复 episode_id: {ep.episode_id}")
            ids.add(ep.episode_id)
            episodes.append(ep)
    if not episodes:
        raise ValueError("episode 文件为空")
    return episodes


def write_immutable_files(files: dict[Path, bytes]) -> None:
    for path, content in files.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"拒绝覆盖不同内容，使用新目录: {path}")
    for path, content in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
