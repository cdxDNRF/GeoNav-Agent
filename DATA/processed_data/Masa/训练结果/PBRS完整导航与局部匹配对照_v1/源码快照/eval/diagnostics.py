"""仅评测端事后分析；不得把目标距离、目标编号返回策略。"""
from env.episode import manhattan


def analyze_positions(positions: list[int], goal: int, grid_size: int = 5):
    if not positions:
        raise ValueError("轨迹不能为空")
    if not 0 <= goal < grid_size**2 or any(not 0 <= p < grid_size**2 for p in positions):
        raise ValueError("格编号越界")
    steps = len(positions) - 1
    distances = [manhattan(p, goal, grid_size) for p in positions]
    revisits = sum(p in positions[:i] for i, p in enumerate(positions) if i)
    reversals = sum(positions[i] == positions[i-2] and positions[i] != positions[i-1]
                    for i in range(2, len(positions)))
    oscillations = sum(positions[i] == positions[i-2] and positions[i-1] == positions[i-3]
                       and positions[i] != positions[i-1] for i in range(3, len(positions)))
    longest = 1
    streak = 1
    for i in range(1, len(positions)):
        if positions[i] == positions[i-1]:
            streak = 1
        elif i >= 2 and positions[i] == positions[i-2]:
            streak += 1
        else:
            streak = 2
        longest = max(longest, streak)
    unique = len(set(positions))
    best_before = distances[0]
    departures_from_adjacent = 0
    moves_away_after_adjacent = 0
    for i in range(1, len(distances)):
        departures_from_adjacent += distances[i-1] == 1 and distances[i] > 1
        moves_away_after_adjacent += best_before <= 1 and distances[i] > distances[i-1]
        best_before = min(best_before, distances[i])
    return {"executed_steps": steps, "unique_cells_including_start": unique,
            "coverage_ratio": unique / grid_size**2,
            "revisit_steps": revisits, "repeat_visit_rate": revisits / steps if steps else None,
            "immediate_reversal_windows_aba": reversals,
            "two_cell_cycle_windows_abab": oscillations,
            "two_cell_cycle_present": oscillations > 0,
            "longest_alternating_run_moves": max(0, longest - 1),
            "initial_distance": distances[0], "last_observed_distance": distances[-1],
            "minimum_distance": min(distances),
            "first_adjacent_step": next((i for i, d in enumerate(distances) if d == 1), None),
            "adjacent_but_not_reached": 1 in distances and 0 not in distances,
            "departures_from_adjacent": departures_from_adjacent,
            "moves_away_after_ever_adjacent": moves_away_after_adjacent,
            "distance_trace_evaluator_only": distances}
