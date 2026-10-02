"""在线动作治理器：只读公开观测和episode内状态，不读目标真值。"""
from dataclasses import dataclass
from collections import Counter

from env.environment import Observation

COARSE_REGIONS = tuple(f"r{row}c{col}" for row in range(3) for col in range(3))


@dataclass(frozen=True)
class GovernorDecision:
    selected_action: str
    ranked_actions: tuple[str, ...]
    filtered_actions: tuple[str, ...]
    override: bool
    reason: str
    cycle_window_detected: bool
    boundary_filtered: bool
    fallback_used: bool
    memory_override: bool = False
    selected_visit_count: int = 0
    selected_destination: int | None = None
    confirmation_mode: bool = False
    confirmation_override: bool = False
    target_evidence: str | None = None
    ranked_regions: tuple[str, ...] = tuple()
    target_region: str | None = None
    destination_region: str | None = None
    region_rank: int | None = None
    hierarchical_mode: bool = False
    hierarchical_override: bool = False


class ActionGovernor:
    """Minimal治理：避开会立即形成ABAB的候选，不调用Frontier，不看图。"""
    name = "ranked_abab_governor_v1"

    def __init__(self, avoid_boundary: bool = False):
        self.avoid_boundary = avoid_boundary
        self.reset()

    def reset(self):
        self.actions = []

    @staticmethod
    def _in_bounds(obs: Observation, action: str) -> bool:
        dr, dc = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}[action]
        return 0 <= obs.position[0] + dr < obs.grid_size and 0 <= obs.position[1] + dc < obs.grid_size

    def _would_cycle(self, action: str) -> bool:
        # 历史动作 h=[A,B,A]，选B会造成动作ABAB；只在A!=B时触发。
        return len(self.actions) >= 3 and self.actions[-3] == self.actions[-1] and self.actions[-2] != self.actions[-1] and action == self.actions[-2]

    def choose(self, observation: Observation, ranked_actions) -> GovernorDecision:
        ranked = tuple(ranked_actions)
        legal = tuple(observation.legal_actions)
        if set(ranked) != set(legal) or len(ranked) != len(legal) or len(set(ranked)) != len(ranked):
            raise ValueError("Governor收到的动作排序不是合法全排列")
        boundary_filtered = False
        filtered = []
        for action in ranked:
            if self.avoid_boundary and not self._in_bounds(observation, action):
                boundary_filtered = True
                continue
            filtered.append(action)
        if not filtered:
            filtered = list(ranked)
        cycle_candidates = [action for action in filtered if not self._would_cycle(action)]
        cycle_window = len(cycle_candidates) != len(filtered)
        candidates = cycle_candidates or filtered
        selected = candidates[0]
        override = selected != ranked[0]
        if cycle_window and override:
            reason = "avoid_immediate_abab"
        elif boundary_filtered and override:
            reason = "avoid_boundary"
        elif cycle_window:
            reason = "all_candidates_would_cycle_keep_top1"
        elif boundary_filtered:
            reason = "top1_boundary_filtered"
        else:
            reason = "top1"
        self.actions.append(selected)
        return GovernorDecision(selected, ranked, tuple(filtered), override, reason,
                                cycle_window, boundary_filtered, False)


class SpatialMemoryGovernor:
    """用公开访问轨迹抑制重复访问的最小空间记忆治理器。

    该治理器只使用 Observation.visited 和当前网格拓扑，对 VLM 返回的动作排序
    做一步候选重排。它不读取目标坐标、目标距离、未访问图块或评测日志，也不做
    Frontier 的全局回溯搜索，因此可以作为相对于 ranked top-1 的单变量消融。
    """

    name = "ranked_spatial_memory_v1"

    def reset(self):
        # 记忆存于 Observation.visited；reset 仅保留与 ActionGovernor 对称的接口。
        return None

    @staticmethod
    def _destination(obs: Observation, action: str) -> int:
        dr, dc = {"up": (-1, 0), "right": (0, 1),
                  "down": (1, 0), "left": (0, -1)}[action]
        row, col = obs.position
        nr, nc = row + dr, col + dc
        if not (0 <= nr < obs.grid_size and 0 <= nc < obs.grid_size):
            return row * obs.grid_size + col
        return nr * obs.grid_size + nc

    @staticmethod
    def _validate_ranked(obs: Observation, ranked_actions) -> tuple[str, ...]:
        ranked = tuple(ranked_actions)
        legal = tuple(obs.legal_actions)
        if (set(ranked) != set(legal) or len(ranked) != len(legal)
                or len(set(ranked)) != len(ranked)):
            raise ValueError("空间记忆治理器收到的动作排序不是合法全排列")
        return ranked

    def choose(self, observation: Observation, ranked_actions) -> GovernorDecision:
        ranked = self._validate_ranked(observation, ranked_actions)
        counts = Counter(observation.visited)
        destinations = {action: self._destination(observation, action)
                        for action in ranked}
        novel = [action for action in ranked if counts[destinations[action]] == 0]
        candidates = novel or list(ranked)
        selected = min(candidates,
                       key=lambda action: (counts[destinations[action]], ranked.index(action)))
        selected_count = counts[destinations[selected]]
        override = selected != ranked[0]
        if novel:
            reason = "prefer_unvisited_destination" if override else "top1_unvisited"
        else:
            reason = "least_visited_destination" if override else "top1_least_visited"
        return GovernorDecision(
            selected_action=selected,
            ranked_actions=ranked,
            filtered_actions=tuple(candidates),
            override=override,
            reason=reason,
            cycle_window_detected=False,
            boundary_filtered=False,
            fallback_used=False,
            memory_override=override,
            selected_visit_count=selected_count,
            selected_destination=destinations[selected],
        )


class NeighborhoodConfirmationGovernor:
    """在空间记忆上增加证据确认：仅 high 证据允许一次局部回退。

    该策略不增加观察范围，也不读取目标真值。正常情况下完全复用空间记忆
    规则；当 VLM 明确返回 high 且 top1 指向已访问的非当前位置时，允许 top1
    通过，以测试“进入目标邻域后需要局部确认/回退”的假设。
    """

    name = "ranked_spatial_memory_confirmation_v1"

    def __init__(self):
        self._spatial = SpatialMemoryGovernor()

    def reset(self):
        self._spatial.reset()

    def choose(self, observation: Observation, ranked_actions, target_evidence: str) -> GovernorDecision:
        if target_evidence not in ("low", "medium", "high"):
            raise ValueError("target_evidence 必须是 low、medium 或 high")
        base = self._spatial.choose(observation, ranked_actions)
        ranked = base.ranked_actions
        top1 = ranked[0]
        top1_destination = SpatialMemoryGovernor._destination(observation, top1)
        current = observation.position[0] * observation.grid_size + observation.position[1]
        top1_count = Counter(observation.visited)[top1_destination]
        if target_evidence == "high" and top1_destination != current and top1_count > 0:
            return GovernorDecision(
                selected_action=top1,
                ranked_actions=ranked,
                filtered_actions=ranked,
                override=False,
                reason="confirm_target_neighborhood_revisit",
                cycle_window_detected=False,
                boundary_filtered=False,
                fallback_used=False,
                memory_override=False,
                selected_visit_count=top1_count,
                selected_destination=top1_destination,
                confirmation_mode=True,
                confirmation_override=True,
                target_evidence=target_evidence,
            )
        return GovernorDecision(
            selected_action=base.selected_action,
            ranked_actions=base.ranked_actions,
            filtered_actions=base.filtered_actions,
            override=base.override,
            reason=base.reason,
            cycle_window_detected=base.cycle_window_detected,
            boundary_filtered=base.boundary_filtered,
            fallback_used=base.fallback_used,
            memory_override=base.memory_override,
            selected_visit_count=base.selected_visit_count,
            selected_destination=base.selected_destination,
            confirmation_mode=True,
            confirmation_override=False,
            target_evidence=target_evidence,
        )


class HierarchicalSearchGovernor:
    """把 VLM 的粗区域排序用于逐格动作执行。

    5x5 网格按 ``row//2, col//2`` 映射为 3x3 粗区域。每一步只读取公开
    Observation、已访问格和 VLM 返回的区域/动作排序：优先向当前排序中仍有
    未访问格的区域移动，再在局部四方向中选择动作。没有跨格跳跃、额外视野、
    目标坐标或全图特征，因此层级只存在于内部规划，不改变基础环境协议。
    """

    name = "ranked_hierarchical_search_v1"

    def reset(self):
        return None

    @staticmethod
    def region_for_cell(cell_id: int, grid_size: int = 5) -> str:
        if grid_size != 5:
            raise ValueError("当前层级搜索只支持5x5网格")
        row, col = divmod(cell_id, grid_size)
        return f"r{row // 2}c{col // 2}"

    @staticmethod
    def cells_for_region(region: str, grid_size: int = 5) -> tuple[int, ...]:
        if region not in COARSE_REGIONS or grid_size != 5:
            raise ValueError("非法5x5粗区域")
        coarse_row, coarse_col = int(region[1]), int(region[3])
        cells = []
        for row in range(grid_size):
            for col in range(grid_size):
                if row // 2 == coarse_row and col // 2 == coarse_col:
                    cells.append(row * grid_size + col)
        return tuple(cells)

    @staticmethod
    def _validate_regions(ranked_regions) -> tuple[str, ...]:
        regions = tuple(ranked_regions)
        if (len(regions) != len(COARSE_REGIONS)
                or len(set(regions)) != len(regions)
                or set(regions) != set(COARSE_REGIONS)):
            raise ValueError("层级搜索收到的区域排序不是合法3x3全排列")
        return regions

    def choose(self, observation: Observation, ranked_actions, ranked_regions) -> GovernorDecision:
        ranked = SpatialMemoryGovernor._validate_ranked(observation, ranked_actions)
        regions = self._validate_regions(ranked_regions)
        region_rank = {region: index for index, region in enumerate(regions)}
        counts = Counter(observation.visited)
        available_regions = [
            region for region in regions
            if any(counts[cell] == 0 for cell in self.cells_for_region(region, observation.grid_size))
        ]
        target_region = available_regions[0] if available_regions else regions[0]
        target_unvisited = sum(counts[cell] == 0 for cell in
                               self.cells_for_region(target_region, observation.grid_size))
        destinations = {action: SpatialMemoryGovernor._destination(observation, action)
                        for action in ranked}
        novel = [action for action in ranked if counts[destinations[action]] == 0]
        candidates = novel or list(ranked)

        def key(action):
            destination_region = self.region_for_cell(destinations[action], observation.grid_size)
            return (0 if destination_region == target_region else 1,
                    region_rank[destination_region],
                    counts[destinations[action]], ranked.index(action))

        selected = min(candidates, key=key)
        destination = destinations[selected]
        destination_region = self.region_for_cell(destination, observation.grid_size)
        selected_count = counts[destination]
        override = selected != ranked[0]
        if novel and destination_region == target_region:
            reason = "move_within_ranked_target_region"
        elif novel:
            reason = "move_toward_ranked_region"
        else:
            reason = "least_visited_ranked_region"
        return GovernorDecision(
            selected_action=selected,
            ranked_actions=ranked,
            filtered_actions=tuple(candidates),
            override=override,
            reason=reason,
            cycle_window_detected=False,
            boundary_filtered=False,
            fallback_used=False,
            memory_override=override and selected_count > 0,
            selected_visit_count=selected_count,
            selected_destination=destination,
            ranked_regions=regions,
            target_region=target_region,
            destination_region=destination_region,
            region_rank=region_rank[destination_region],
            hierarchical_mode=True,
            hierarchical_override=override or destination_region != self.region_for_cell(
                destinations[ranked[0]], observation.grid_size),
        )
