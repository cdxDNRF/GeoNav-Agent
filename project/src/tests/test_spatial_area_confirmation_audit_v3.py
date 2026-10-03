"""Timing metadata exemption cannot hide metric, key, or gate differences."""
import unittest
from eval.audit_spatial_area_confirmation_v3 import same_values


class TimingSchemaRevisionTests(unittest.TestCase):
    def summary(self):
        return dict(stage='engineering', arms={'sr': .61}, effects={'gain': .2},
                    target_checks={'passed': True}, planned_total=4200, primary_SR=.6)

    def test_only_positive_finite_elapsed_metadata_is_exempted(self):
        expected = self.summary()
        self.assertTrue(same_values(dict(expected, seconds=235.1), expected))
        for seconds in (-1, 0, float('nan'), float('inf'), True):
            self.assertFalse(same_values(dict(expected, seconds=seconds), expected))

    def test_statistic_and_gate_mismatches_still_fail(self):
        expected = self.summary()
        self.assertFalse(same_values(dict(expected, seconds=235.1, arms={'sr': .62}), expected))
        self.assertFalse(same_values(dict(expected, seconds=235.1, target_checks={'passed': False}), expected))
        self.assertFalse(same_values(dict(expected, seconds=235.1, planned_total=4199), expected))

    def test_other_unknown_fields_and_nested_elapsed_not_exempted(self):
        expected = self.summary()
        self.assertFalse(same_values(dict(expected, seconds=235.1, other=1), expected))
        self.assertFalse(same_values({'metrics': {'sr': .61, 'seconds': 1}}, {'metrics': {'sr': .61}}))


if __name__ == '__main__':
    unittest.main()
