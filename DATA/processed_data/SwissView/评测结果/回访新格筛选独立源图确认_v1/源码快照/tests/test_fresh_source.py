"""Boundaries, source isolation and full fixed task-cohort checks."""
import unittest
from eval.fresh_source_confirmation import primary_checks, target_checks, require_unused, validate_cohort
from data.scaled_masa import scaled_tasks


class FreshSourceTests(unittest.TestCase):
    def example(self):
        return dict(sr_gain=.02, positive_seeds=2, sg_change=0, source95_SR=[.001, .04])

    def test_boundary_gain_passes(self):
        self.assertTrue(all(primary_checks(self.example()).values()))

    def test_cross_zero_interval_blocks_upgrade(self):
        for lower in (0, -.001):
            e = self.example(); e['source95_SR'][0] = lower
            self.assertFalse(primary_checks(e)['source95_SR_positive'])

    def test_sg_damage_or_only_one_weight_blocks(self):
        e = self.example(); e.update(positive_seeds=1, sg_change=.001)
        c = primary_checks(e)
        self.assertFalse(c['two_positive_weights']); self.assertFalse(c['SG_no_worse'])

    def test_target_gate_requires_five_points(self):
        e = self.example(); self.assertFalse(target_checks(e)['SR_gain5pp'])
        e['sr_gain'] = .05; self.assertTrue(all(target_checks(e).values()))

    def test_previously_used_or_duplicate_sources_rejected(self):
        selected = [f'img_{i}' for i in range(60, 80)]
        require_unused(selected, {'img_40'})
        with self.assertRaises(ValueError): require_unused(selected, {'img_60'})
        with self.assertRaises(ValueError): require_unused(selected[:-1] + [selected[0]], set())

    def test_fixed_sampler_cohort_and_truncation(self):
        sources = [dict(area=f'img_{i}', split='test', source_tile=f'SwissView100/{i}.png') for i in range(60, 80)]
        episodes = scaled_tasks(sources, 5)
        validate_cohort(episodes)
        with self.assertRaises(ValueError): validate_cohort(episodes[:-1])
        with self.assertRaises(ValueError): validate_cohort(episodes[:-1] + [episodes[0]])


if __name__ == '__main__':
    unittest.main()
