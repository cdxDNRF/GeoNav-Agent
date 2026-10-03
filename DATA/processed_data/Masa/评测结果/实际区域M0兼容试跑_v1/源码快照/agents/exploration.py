"""无目标真值的规则探索：优先未访相邻格，必要时在已访图上回退。"""
from collections import deque

from env.environment import Observation
from env.episode import ACTIONS


class FrontierPolicy:
    name = "frontier_bfs_backtrack_v1"

    def act(self, observation: Observation) -> str:
        grid = observation.grid_size
        current = observation.position[0] * grid + observation.position[1]
        visited = set(observation.visited)
        queue = deque([(current, None)])
        seen = {current}
        # 拓扑已知但未访问格的视觉内容未知；只用编号、边界及访问记录找前沿。
        while queue:
            node, first_action = queue.popleft()
            row, col = divmod(node, grid)
            for action, (dr, dc) in ACTIONS.items():
                nr, nc = row + dr, col + dc
                if not (0 <= nr < grid and 0 <= nc < grid):
                    continue
                neighbor = nr * grid + nc
                chosen = first_action or action
                if neighbor not in visited:
                    return chosen
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, chosen))
        for action, (dr, dc) in ACTIONS.items():
            if 0 <= observation.position[0] + dr < grid and 0 <= observation.position[1] + dc < grid:
                return action
        raise ValueError("没有合法移动")
