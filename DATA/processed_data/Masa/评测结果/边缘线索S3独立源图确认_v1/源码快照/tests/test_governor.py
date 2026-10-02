"""Action Governor离线测试，不调用云端。"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.governor import (COARSE_REGIONS, ActionGovernor,
                              HierarchicalSearchGovernor,
                              NeighborhoodConfirmationGovernor,
                              SpatialMemoryGovernor)
from env.environment import Observation


class GovernorTests(unittest.TestCase):
    def obs(self, pos=(2, 2), visited=(12,)):
        return Observation(b"current", b"target", pos, 5, 10, visited)

    def test_top1_is_preserved_without_cycle(self):
        g = ActionGovernor()
        d = g.choose(self.obs(), ("right", "up", "left", "down"))
        self.assertEqual(d.selected_action, "right")
        self.assertFalse(d.override)
        self.assertEqual(d.reason, "top1")

    def test_abab_candidate_is_overridden(self):
        g = ActionGovernor()
        g.actions = ["up", "down", "up"]
        d = g.choose(self.obs(), ("down", "right", "left", "up"))
        self.assertEqual(d.selected_action, "right")
        self.assertTrue(d.override)
        self.assertEqual(d.reason, "avoid_immediate_abab")
        self.assertTrue(d.cycle_window_detected)

    def test_single_aba_is_not_abab(self):
        g = ActionGovernor()
        g.actions = ["up", "down"]
        d = g.choose(self.obs(), ("up", "right", "left", "down"))
        self.assertEqual(d.selected_action, "up")
        self.assertFalse(d.cycle_window_detected)

    def test_invalid_rank_is_rejected(self):
        g = ActionGovernor()
        for rank in (("up", "up", "left", "down"), ("up", "left"), ("up", "left", "down", "stop")):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                g.choose(self.obs(), rank)

    def test_boundary_is_explicit_and_default_preserves_protocol(self):
        boundary = self.obs(pos=(0, 0))
        d = ActionGovernor().choose(boundary, ("up", "right", "down", "left"))
        self.assertEqual(d.selected_action, "up")
        self.assertFalse(d.boundary_filtered)
        d = ActionGovernor(avoid_boundary=True).choose(boundary, ("up", "right", "down", "left"))
        self.assertEqual(d.selected_action, "right")
        self.assertTrue(d.boundary_filtered)
        self.assertTrue(d.override)

    def test_all_cycle_candidates_keeps_top1(self):
        g = ActionGovernor()
        g.actions = ["up", "down", "up"]
        d = g.choose(self.obs(), ("down", "right", "left", "up"))
        # Only down would complete ABAB for this history; the next ranked legal candidate wins.
        self.assertEqual(d.selected_action, "right")
        self.assertFalse(d.fallback_used)

    def test_reset_is_episode_local(self):
        g = ActionGovernor()
        g.actions = ["up", "down", "up"]
        g.reset()
        d = g.choose(self.obs(), ("down", "right", "left", "up"))
        self.assertEqual(d.selected_action, "down")
        self.assertFalse(d.cycle_window_detected)

    def test_goal_not_in_decision_object(self):
        g = ActionGovernor()
        decision = g.choose(self.obs(), ("right", "up", "left", "down"))
        self.assertFalse(hasattr(decision, "goal"))
        self.assertFalse(hasattr(decision, "distance"))

    def test_spatial_memory_prefers_unvisited_destination(self):
        g = SpatialMemoryGovernor()
        # right=13 is already visited; up=7 is novel, so memory overrides VLM top1.
        decision = g.choose(self.obs(visited=(12, 13)),
                            ("right", "up", "left", "down"))
        self.assertEqual(decision.selected_action, "up")
        self.assertTrue(decision.memory_override)
        self.assertEqual(decision.reason, "prefer_unvisited_destination")
        self.assertEqual(decision.selected_destination, 7)
        self.assertEqual(decision.selected_visit_count, 0)

    def test_spatial_memory_uses_least_visited_when_no_novel_neighbor(self):
        g = SpatialMemoryGovernor()
        # All four destinations are visited; left=11 has the lowest count.
        decision = g.choose(self.obs(visited=(12, 13, 13, 7, 7, 17, 17, 11)),
                            ("right", "up", "down", "left"))
        self.assertEqual(decision.selected_action, "left")
        self.assertEqual(decision.reason, "least_visited_destination")
        self.assertEqual(decision.selected_visit_count, 1)

    def test_spatial_memory_treats_boundary_as_current_cell(self):
        g = SpatialMemoryGovernor()
        decision = g.choose(self.obs(pos=(0, 0), visited=(0,)),
                            ("up", "right", "down", "left"))
        self.assertEqual(decision.selected_action, "right")
        self.assertEqual(decision.selected_destination, 1)
        self.assertTrue(decision.memory_override)

    def test_spatial_memory_does_not_receive_goal_fields(self):
        g = SpatialMemoryGovernor()
        decision = g.choose(self.obs(), ("right", "up", "left", "down"))
        self.assertFalse(hasattr(decision, "goal"))
        self.assertFalse(hasattr(decision, "distance"))

    def test_confirmation_high_can_allow_non_current_revisit(self):
        g = NeighborhoodConfirmationGovernor()
        # top1=right revisits 13; high evidence permits this local confirmation move.
        decision = g.choose(self.obs(visited=(12, 13, 7)),
                            ("right", "up", "left", "down"), "high")
        self.assertEqual(decision.selected_action, "right")
        self.assertTrue(decision.confirmation_mode)
        self.assertTrue(decision.confirmation_override)
        self.assertEqual(decision.reason, "confirm_target_neighborhood_revisit")
        self.assertEqual(decision.target_evidence, "high")

    def test_confirmation_low_keeps_spatial_memory(self):
        g = NeighborhoodConfirmationGovernor()
        decision = g.choose(self.obs(visited=(12, 13, 7)),
                            ("right", "up", "left", "down"), "low")
        self.assertEqual(decision.selected_action, "left")
        self.assertFalse(decision.confirmation_override)
        self.assertEqual(decision.reason, "prefer_unvisited_destination")

    def test_confirmation_high_does_not_allow_boundary_revisit(self):
        g = NeighborhoodConfirmationGovernor()
        decision = g.choose(self.obs(pos=(0, 0), visited=(0, 1)),
                            ("up", "right", "down", "left"), "high")
        self.assertEqual(decision.selected_action, "down")
        self.assertFalse(decision.confirmation_override)

    def test_confirmation_rejects_invalid_evidence(self):
        with self.assertRaises(ValueError):
            NeighborhoodConfirmationGovernor().choose(
                self.obs(), ("right", "up", "left", "down"), "certain")

    def test_hierarchy_region_mapping_and_cells(self):
        self.assertEqual(HierarchicalSearchGovernor.region_for_cell(0), "r0c0")
        self.assertEqual(HierarchicalSearchGovernor.region_for_cell(24), "r2c2")
        self.assertEqual(HierarchicalSearchGovernor.region_for_cell(12), "r1c1")
        self.assertEqual(len(HierarchicalSearchGovernor.cells_for_region("r1c1")), 4)
        self.assertEqual(set(HierarchicalSearchGovernor.cells_for_region("r2c2")), {24})

    def test_hierarchy_prefers_ranked_region_then_unvisited_action(self):
        g = HierarchicalSearchGovernor()
        regions = ("r0c0",) + tuple(region for region in COARSE_REGIONS if region != "r0c0")
        decision = g.choose(self.obs(pos=(1, 1), visited=(6, 5)),
                            ("right", "up", "down", "left"), regions)
        self.assertEqual(decision.selected_action, "up")
        self.assertEqual(decision.target_region, "r0c0")
        self.assertEqual(decision.destination_region, "r0c0")
        self.assertEqual(decision.reason, "move_within_ranked_target_region")
        self.assertTrue(decision.hierarchical_mode)
        self.assertTrue(decision.hierarchical_override)

    def test_hierarchy_uses_next_ranked_region_when_target_not_adjacent(self):
        g = HierarchicalSearchGovernor()
        regions = ("r2c2", "r0c1") + tuple(region for region in COARSE_REGIONS if region not in ("r2c2", "r0c1"))
        decision = g.choose(self.obs(pos=(0, 0), visited=(0,)),
                            ("up", "right", "down", "left"), regions)
        self.assertEqual(decision.target_region, "r2c2")
        self.assertEqual(decision.selected_action, "right")
        self.assertEqual(decision.destination_region, "r0c0")
        self.assertEqual(decision.reason, "move_toward_ranked_region")

    def test_hierarchy_rejects_invalid_region_permutation(self):
        with self.assertRaises(ValueError):
            HierarchicalSearchGovernor().choose(
                self.obs(), ("right", "up", "left", "down"), ("r0c0",))

    def test_hierarchy_decision_has_no_goal_fields(self):
        regions = tuple(COARSE_REGIONS)
        decision = HierarchicalSearchGovernor().choose(
            self.obs(), ("right", "up", "left", "down"), regions)
        self.assertFalse(hasattr(decision, "goal"))
        self.assertFalse(hasattr(decision, "distance"))


if __name__ == "__main__":
    unittest.main()
