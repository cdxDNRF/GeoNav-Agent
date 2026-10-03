"""One frozen public-geometry intervention around the original M0 action."""
from itertools import product

ACTIONS = ('up', 'right', 'down', 'left')
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
GRID = 10
BUDGET = 20
HORIZON = 3
MIN_GAIN = 2


def neighbors(cell):
    r, c = divmod(cell, GRID)
    return {a: (r + dr) * GRID + c + dc for a, (dr, dc) in zip(ACTIONS, DELTAS)
            if 0 <= r + dr < GRID and 0 <= c + dc < GRID}


NEIGHBORS = tuple(neighbors(c) for c in range(GRID * GRID))
CLOSED_MASK = tuple(sum(1 << p for p in {c, *NEIGHBORS[c].values()}) for c in range(GRID * GRID))


def choose_coverage_action(position, visited, remaining_budget, explorer_action, cue_action):
    """Accept only public pose/history/budget and the two existing proposals.

    Future cells are geometry, never image predictions or eliminated targets.
    Plans are recomputed at every real observation; no future observation is
    fetched or credited. A terminal arrival cannot supply another cue action.
    """
    if type(remaining_budget) is not int or not 1 <= remaining_budget <= BUDGET:
        raise ValueError('fixed B20 public budget required')
    if len(position) != 2 or any(type(v) is not int or not 0 <= v < GRID for v in position):
        raise ValueError('public grid10 position required')
    visited = tuple(visited)
    current = position[0] * GRID + position[1]
    if len(visited) != BUDGET - remaining_budget + 1 or visited[-1] != current:
        raise ValueError('consecutive public observation prefix required')
    if any(type(v) is not int or not 0 <= v < GRID * GRID for v in visited):
        raise ValueError('invalid public visited cell')
    legal = NEIGHBORS[current]
    if explorer_action not in legal or cue_action not in (None, *legal):
        raise ValueError('existing proposals must be legal')
    if cue_action is not None:
        return dict(action=cue_action, changed=False, control='accepted_cue', horizon=0,
                    baseline_gain=None, selected_gain=None, opportunity_gain=0, paths_by_first={})
    covered = 0
    for c in visited:
        covered |= CLOSED_MASK[c]
    known_mask = sum(1 << c for c in set(visited))
    horizon = min(HORIZON, remaining_budget)
    best = {}
    for sequence in product(ACTIONS, repeat=horizon):
        cell = current
        opportunity = covered
        seen_mask = known_mask
        cells = []
        repeat = 0
        for index, action in enumerate(sequence, 1):
            dest = NEIGHBORS[cell].get(action)
            if dest is None:
                break
            repeat += bool(seen_mask & (1 << dest))
            seen_mask |= 1 << dest
            cells.append(dest)
            # At zero budget only direct arrival can succeed; no extra cue.
            opportunity |= CLOSED_MASK[dest] if index < remaining_budget else 1 << dest
            cell = dest
        else:
            record = dict(actions=list(sequence), cells=cells,
                          gain=(opportunity & ~covered).bit_count(), revisit_moves=repeat)
            first = sequence[0]
            if first not in best or (record['gain'], -repeat) > (best[first]['gain'], -best[first]['revisit_moves']):
                best[first] = record
    baseline = best[explorer_action]
    selected = max(best, key=lambda a: (best[a]['gain'], -best[a]['revisit_moves'],
                                       a == explorer_action, -ACTIONS.index(a)))
    if best[selected]['gain'] < baseline['gain'] + MIN_GAIN:
        selected = explorer_action
    return dict(action=selected, changed=selected != explorer_action,
                control='coverage_lookahead' if selected != explorer_action else 'original_edge',
                horizon=horizon, baseline_gain=baseline['gain'], selected_gain=best[selected]['gain'],
                opportunity_gain=best[selected]['gain'] - baseline['gain'],
                paths_by_first=best)
