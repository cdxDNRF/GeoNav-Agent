"""Synthetic permission, causal state and goal-use checks; no data/model I/O."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.ground_local_loop_v1 import GroundLocalLoop, feature, neighbors


def vec(a=1., b=0.):
    x = np.zeros(512); x[:2] = [a, b]
    return x


def obs(cell, step=0, x=None, budget=10):
    return dict(position=cell, step_index=step, remaining_steps=budget-step,
                grid_size=5, current_feature=vec() if x is None else x)


class Tests(unittest.TestCase):
    def agent(self, goal=None, budget=10):
        p = GroundLocalLoop(); p.begin(grid_size=5, budget=budget, goal_feature=goal)
        return p

    def test_feature_bank_rejected(self):
        with self.assertRaises(ValueError): feature(np.ones((25, 512)))

    def test_hidden_fields_rejected(self):
        for field in ('goal_index', 'source_id', 'all_features', 'distance', 'visited', 'current_image'):
            p = self.agent(); x = obs(12); x[field] = 0
            with self.assertRaises(ValueError): p.observe(x)

    def test_invalid_features(self):
        for x in (np.zeros(512), np.full(512, np.nan), np.ones(513)):
            with self.assertRaises(ValueError): self.agent().observe(obs(12, x=x))

    def test_budget_and_steps(self):
        for x in (obs(12, 1), obs(12, budget=9), obs(12, 10)):
            with self.assertRaises(ValueError): self.agent().observe(x)

    def test_transition_and_failure_latch(self):
        p = self.agent(); event = p.observe(obs(12)); nxt = dict(neighbors(12, 5))[event['action']]
        with self.assertRaises(ValueError): p.observe(obs((nxt+1)%25, 1))
        with self.assertRaises(ValueError): p.observe(obs(nxt, 1))

    def test_reset_and_finish(self):
        p = self.agent(); p.observe(obs(12)); p.finish()
        with self.assertRaises(ValueError): p.observe(obs(7, 1))
        p.begin(grid_size=5, budget=10, goal_feature=None)
        self.assertEqual(p.observe(obs(24))['observed_cells'], [24])

    def test_feature_and_target_copy(self):
        goal = vec(); p = self.agent(goal); goal[:] = 0
        x = vec(); p.observe(obs(12, x=x)); x[:] = 0
        self.assertAlmostEqual(p.scores[12], 1.)

    def test_disabled_matches_frontier(self):
        a, b = self.agent(), self.agent(vec())
        b.evidence = False
        cell = 12
        for step in range(10):
            ea, eb = a.observe(obs(cell, step)), b.observe(obs(cell, step))
            self.assertEqual(ea['action'], eb['action'])
            cell = dict(neighbors(cell, 5))[ea['action']]

    def test_goal_changes_a_rational_plan(self):
        # Same two legal observations, two opposite target vectors. The first
        # decision is identical; after moving up, increasing similarity leads
        # further up, decreasing similarity selects an equal-cost side frontier.
        a, b = self.agent(vec()), self.agent(-vec())
        self.assertEqual(a.observe(obs(12, x=vec(1, 1)))['action'], 0)
        self.assertEqual(b.observe(obs(12, x=vec(1, 1)))['action'], 0)
        ea, eb = a.observe(obs(7, 1)), b.observe(obs(7, 1))
        self.assertEqual(ea['action'], 0)
        self.assertNotEqual(ea['action'], eb['action'])

    def test_no_out_of_bounds_action(self):
        for cell in (0, 4, 20, 24):
            action = self.agent().observe(obs(cell))['action']
            self.assertIn(action, dict(neighbors(cell, 5)))

    def test_observation_identity_on_revisit(self):
        p = self.agent(); p.observe(obs(12))
        # Public contract rejects a skipped position instead of accepting a
        # caller-provided bank/history as a shortcut to a revisit.
        with self.assertRaises(ValueError): p.observe(obs(12, 1, vec(0, 1)))


if __name__ == '__main__': unittest.main()
