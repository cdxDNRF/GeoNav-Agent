"""Prevent a rule-reset repair from changing neural replay or rule decisions."""
import unittest
from unittest.mock import patch
from eval import audit_spatial_area_confirmation_v2 as repair


class RuleResetRevisionTests(unittest.TestCase):
    def args(self, seed, agent, condition):
        return ['engineering', seed, condition, object(), object(), agent,
                object(), object(), object(), object(), object(), object(), object()]

    def test_original_rule_replay_gets_only_stateless_reset(self):
        def original(*args):
            self.assertIsNone(args[1])
            self.assertIsInstance(args[5], repair.RuleWithoutMemory)
            self.assertIsNone(args[5].reset())
            self.assertFalse(hasattr(args[5], 'act'))
            return 'all original checks'
        with patch.object(repair, 'original_replay', side_effect=original):
            self.assertEqual(repair.replay_episode(*self.args(None, None, 'Frontier')), 'all original checks')
            self.assertEqual(repair.replay_episode(*self.args(None, None, 'FixedRegion')), 'all original checks')

    def test_neural_replay_agent_and_arguments_unchanged(self):
        args = self.args(0, object(), 'CueFull')
        with patch.object(repair, 'original_replay', return_value='same') as original:
            self.assertEqual(repair.replay_episode(*args), 'same')
            self.assertEqual(original.call_args.args, tuple(args))

    def test_no_other_rule_or_nonnull_rule_agent_allowed(self):
        with self.assertRaises(ValueError):
            repair.replay_episode(*self.args(None, None, 'CueFull'))
        with self.assertRaises(ValueError):
            repair.replay_episode(*self.args(None, object(), 'Frontier'))


if __name__ == '__main__':
    unittest.main()
