"""Same coverage/radial rule, parameterized by public grid geometry."""
from functools import lru_cache
from itertools import product

ACTIONS = ('up', 'right', 'down', 'left')
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@lru_cache(maxsize=2)
def geometry(k):
    if type(k) is not int or k not in (10, 15):
        raise ValueError('registered grid10/grid15 geometry required')
    neighbors = []
    for cell in range(k * k):
        r, c = divmod(cell, k)
        neighbors.append({a: (r + dr) * k + c + dc for a, (dr, dc) in zip(ACTIONS, DELTAS)
                          if 0 <= r + dr < k and 0 <= c + dc < k})
    closed = tuple(sum(1 << x for x in {cell, *neighbors[cell].values()}) for cell in range(k * k))
    return tuple(neighbors), closed


def choose_area_coverage(position, visited, remaining_budget, explorer_action, cue_action, *, grid_size, budget=20):
    if type(budget) is not int or budget != 20 or type(remaining_budget) is not int or not 1 <= remaining_budget <= budget:
        raise ValueError('registered B20 public budget required')
    neighbors, closed = geometry(grid_size)
    if len(position) != 2 or any(type(x) is not int or not 0 <= x < grid_size for x in position):
        raise ValueError('public position required')
    visited = tuple(visited)
    current = position[0] * grid_size + position[1]
    if len(visited) != budget - remaining_budget + 1 or visited[-1] != current:
        raise ValueError('consecutive public prefix required')
    if any(type(x) is not int or not 0 <= x < grid_size * grid_size for x in visited):
        raise ValueError('invalid visited cells')
    if explorer_action not in neighbors[current] or cue_action not in (None, *neighbors[current]):
        raise ValueError('legal existing proposals required')
    if cue_action is not None:
        result = dict(action=cue_action, changed=False, control='accepted_cue', horizon=0,
            baseline_gain=None, selected_gain=None, opportunity_gain=0, paths_by_first={})
    else:
        covered = 0
        for cell in visited:
            covered |= closed[cell]
        known = sum(1 << cell for cell in set(visited))
        horizon, best = min(3, remaining_budget), {}
        for sequence in product(ACTIONS, repeat=horizon):
            cell, opportunity, seen, repeats, cells = current, covered, known, 0, []
            for index, action in enumerate(sequence, 1):
                dest = neighbors[cell].get(action)
                if dest is None:
                    break
                repeats += bool(seen & (1 << dest))
                seen |= 1 << dest
                cells.append(dest)
                opportunity |= closed[dest] if index < remaining_budget else 1 << dest
                cell = dest
            else:
                record = dict(actions=list(sequence), cells=cells, gain=(opportunity & ~covered).bit_count(), revisit_moves=repeats)
                first = sequence[0]
                if first not in best or (record['gain'], -repeats) > (best[first]['gain'], -best[first]['revisit_moves']):
                    best[first] = record
        winner = max(best, key=lambda a: (best[a]['gain'], -best[a]['revisit_moves'], a == explorer_action, -ACTIONS.index(a)))
        if best[winner]['gain'] < best[explorer_action]['gain'] + 2:
            winner = explorer_action
        result = dict(action=winner, changed=winner != explorer_action,
            control='coverage_lookahead' if winner != explorer_action else 'original_edge', horizon=horizon,
            baseline_gain=best[explorer_action]['gain'], selected_gain=best[winner]['gain'],
            opportunity_gain=best[winner]['gain'] - best[explorer_action]['gain'], paths_by_first=best)
    result.update(unprotected_action=result['action'], radial_guard_blocked=False,
                  original_next_radius=None, unprotected_next_radius=None)
    if cue_action is None:
        start = visited[0]
        def radius(cell):
            return abs(cell // grid_size - start // grid_size) + abs(cell % grid_size - start % grid_size)
        result['original_next_radius'] = radius(neighbors[current][explorer_action])
        result['unprotected_next_radius'] = radius(neighbors[current][result['action']])
        if result['changed'] and result['unprotected_next_radius'] < result['original_next_radius']:
            result.update(action=explorer_action, changed=False, control='radial_guard', radial_guard_blocked=True,
                          selected_gain=result['baseline_gain'], opportunity_gain=0)
    return result
