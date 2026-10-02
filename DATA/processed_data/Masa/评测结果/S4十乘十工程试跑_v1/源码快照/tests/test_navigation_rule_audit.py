import copy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.governor import HierarchicalSearchGovernor, COARSE_REGIONS
from env.episode import ACTIONS
from eval.audit_navigation_spatial_repair import check_record
from test_foundation import Fixture


class RuleAuditTests(Fixture):
    def record(self):
        ep = self.ep(start=8, goal=15)
        env = self.env(); obs = env.reset(ep)
        while not env.done:
            action = HierarchicalSearchGovernor().choose(obs, tuple(ACTIONS), COARSE_REGIONS).selected_action
            obs, _, _ = env.step(action)
        record = dict(**env.evaluator_result(), area=ep.area, distance=ep.dist)
        return record, asdict(ep)

    def test_real_rule_wall_move_costs_budget_and_revisits_current_cell(self):
        record, ep = self.record()
        self.assertGreater(record['out_of_bounds'], 0)
        self.assertEqual(check_record(record, ep, rule='FixedRegion'), 10)
        wall = next(e for e in record['trajectory'] if e['out_of_bounds'])
        self.assertTrue(wall['revisited'])
        self.assertEqual(wall['patch_id'], record['trajectory'][wall['step']-1]['patch_id'])

    def test_rule_audit_rejects_false_wall_counts_and_boundary_teleport(self):
        record, ep = self.record()
        changed = copy.deepcopy(record); changed['out_of_bounds'] = 0
        with self.assertRaises(AssertionError):
            check_record(changed, ep, rule='FixedRegion')
        changed = copy.deepcopy(record)
        next(e for e in changed['trajectory'] if e['out_of_bounds'])['patch_id'] = 24
        with self.assertRaises(AssertionError):
            check_record(changed, ep, rule='FixedRegion')

    def test_neural_audit_is_delegated_without_relaxing_its_checks(self):
        model = object()
        with patch('eval.audit_navigation_spatial_repair._neural_check', side_effect=AssertionError('strict')) as strict:
            with self.assertRaisesRegex(AssertionError, 'strict'):
                check_record({}, {}, model)
            strict.assert_called_once_with({}, {}, model)


if __name__ == '__main__':
    unittest.main()
