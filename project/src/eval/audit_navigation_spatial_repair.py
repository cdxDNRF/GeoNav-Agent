"""Versioned auditor repair: rule baselines obey the original wall-cost protocol.

Keep the preregistered trainer, evaluator and auditor files byte-for-byte intact.
Only replace the auditor's rule-record branch; neural checks remain unchanged.
"""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import audit_navigation_spatial as original
from train.curiosity_controlled import write_new

_neural_check = original.check_record


def check_record(row, ep, *args, **kwargs):
    rule = kwargs.get('rule')
    if rule is None:
        return _neural_check(row, ep, *args, **kwargs)
    if args or set(kwargs) != {'rule'} or rule not in ('Frontier', 'FixedRegion'):
        raise ValueError('rule audit requires an explicit supported baseline without a neural model')
    assert row['area'] == ep['area'] and row['distance'] == ep['dist']
    visited = [ep['start']]; revisits = 0; walls = 0
    assert row['trajectory'][0] == dict(step=0, patch_id=ep['start'], action=None, out_of_bounds=False, revisited=False)
    assert len(row['trajectory']) == row['steps'] + 1 and 0 < row['steps'] <= ep['budget']
    for step, event in enumerate(row['trajectory'][1:], 1):
        cell = visited[-1]
        assert cell != ep['goal']
        obs = original.Observation(b'', b'', divmod(cell, 5), 5, ep['budget'] + 1 - step, tuple(visited))
        action = original.FrontierPolicy().act(obs) if rule == 'Frontier' else original.HierarchicalSearchGovernor().choose(obs, tuple(original.MOVES), original.COARSE_REGIONS).selected_action
        dr, dc = original.MOVES[action]
        nr, nc = cell // 5 + dr, cell % 5 + dc
        outside = not (0 <= nr < 5 and 0 <= nc < 5)
        dest = cell if outside else nr * 5 + nc
        revisit = dest in visited
        assert event == dict(step=step, patch_id=dest, action=action, out_of_bounds=outside, revisited=revisit)
        walls += outside; revisits += revisit; visited.append(dest)
    assert row['success'] == (visited[-1] == ep['goal'])
    assert row['termination'] == ('goal_reached' if row['success'] else 'budget_exhausted')
    assert row['success'] or row['steps'] == ep['budget']
    assert row['sg'] == original.near(visited[-1], ep['goal'])
    assert row['revisits'] == revisits and row['out_of_bounds'] == walls
    original.close(row['repeat_visit_rate'], revisits / row['steps'])
    return row['steps']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=original.OUTPUT)
    args = parser.parse_args(); out = args.output_dir
    sources = [Path(__file__), original.SRC / 'tests/test_navigation_rule_audit.py']
    repair_hashes = {}
    for source in sources:
        name = source.relative_to(original.SRC).as_posix()
        copied = out / '源码快照' / name
        if copied.exists():
            assert copied.read_bytes() == source.read_bytes()
        else:
            copied.write_bytes(source.read_bytes())
        repair_hashes[name] = original.digest(source)
    write_new(out / '审计修复记录.json', dict(
        issue='Original audit asserted zero boundary moves for the FixedRegion rule; protocol permits stay-in-place wall moves costing one step.',
        original_failure_stage='All 12 neural models, 3360 neural episodes and training transitions passed; FixedRegion rule audit then failed.',
        original_auditor_sha256=original.digest(original.SRC / 'eval/audit_navigation_spatial.py'),
        correction='Replay rule out-of-bounds as stay-in-place; verify wall/revisit counts. Neural zero-wall and checkpoint checks unchanged.',
        changed_training_or_evaluation=False, changed_results=False, rerun_training=False,
        repair_source_sha256=repair_hashes, repair_regression_tests=3))
    original.check_record = check_record
    original.main()


if __name__ == '__main__':
    main()
