"""One public radial guard on the frozen Coverage3 proposal."""
from agents.coverage_lookahead import choose_coverage_action, NEIGHBORS, HORIZON, MIN_GAIN


def choose_guarded_coverage_action(position, visited, remaining_budget, explorer_action, cue_action):
    """Preserve the original actor if a coverage override contracts more.

    Radius is L1 distance from visited[0], never distance from the target.
    This compares one-step proposals, without predicting the future actor.
    Accepted target cues have priority, including inward moves.
    """
    visited = tuple(visited)
    result = choose_coverage_action(position, visited, remaining_budget, explorer_action, cue_action)
    result.update(unprotected_action=result['action'], radial_guard_blocked=False,
                  original_next_radius=None, unprotected_next_radius=None)
    if cue_action is not None:
        return result
    start, current = visited[0], visited[-1]
    def radius(cell):
        return abs(cell // 10 - start // 10) + abs(cell % 10 - start % 10)
    result['original_next_radius'] = radius(NEIGHBORS[current][explorer_action])
    result['unprotected_next_radius'] = radius(NEIGHBORS[current][result['action']])
    if result['changed'] and result['unprotected_next_radius'] < result['original_next_radius']:
        result.update(action=explorer_action, changed=False, control='radial_guard',
                      radial_guard_blocked=True, selected_gain=result['baseline_gain'], opportunity_gain=0)
    return result
