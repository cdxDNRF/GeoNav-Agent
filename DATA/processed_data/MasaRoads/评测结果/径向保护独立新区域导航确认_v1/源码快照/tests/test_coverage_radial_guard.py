"""Behavioral checks for the single radial guard and inherited safety gates."""
import inspect
import random
import unittest
from agents.coverage_lookahead import choose_coverage_action, NEIGHBORS
from agents.coverage_radial_guard import choose_guarded_coverage_action as choose
from eval.audit_coverage_radial import independent_coverage
from eval.coverage_radial_validation import gated, all_gates, STRATA


class RadialGuardTests(unittest.TestCase):
    def test_runtime_signature_excludes_truth_and_labels(self):
        self.assertEqual(tuple(inspect.signature(choose).parameters),
            ('position','visited','remaining_budget','explorer_action','cue_action'))
        with self.assertRaises(TypeError):choose((0,3),(3,),20,'right',None,goal=99)
        with self.assertRaises(TypeError):choose((0,3),(3,),20,'right',None,stratum='long_distance')

    def test_observed_contraction_is_vetoed(self):
        prefix=(41,31,32,33,34,35,36,37,38)
        old=choose_coverage_action((3,8),prefix,12,'right',None)
        self.assertEqual(old['action'],'down')
        guarded=choose((3,8),prefix,12,'right',None)
        self.assertEqual(guarded['action'],'right')
        self.assertTrue(guarded['radial_guard_blocked'])
        self.assertFalse(guarded['changed'])
        self.assertEqual(guarded['unprotected_action'],'down')
        self.assertEqual((guarded['original_next_radius'],guarded['unprotected_next_radius']),(9,7))

    def test_equal_radius_keeps_border_coverage_benefit(self):
        result=choose((0,3),(3,),20,'right',None)
        self.assertEqual(result['action'],'down')
        self.assertTrue(result['changed'])
        self.assertFalse(result['radial_guard_blocked'])
        self.assertEqual(result['opportunity_gain'],2)

    def test_inward_accepted_target_cue_is_never_vetoed(self):
        prefix=(41,31,32,33,34,35,36,37,38)
        result=choose((3,8),prefix,12,'right','left')
        self.assertEqual(result['action'],'left')
        self.assertEqual(result['control'],'accepted_cue')
        self.assertFalse(result['radial_guard_blocked'])
        self.assertIsNone(result['original_next_radius'])

    def test_terminal_budget_keeps_direct_arrival_semantics(self):
        result=choose((0,3),(3,)*20,1,'right',None)
        self.assertEqual(result['action'],'right')
        self.assertEqual(result['horizon'],1)
        self.assertTrue(all(v['gain']==0 for v in result['paths_by_first'].values()))

    def test_invalid_public_prefix_or_proposal_is_rejected(self):
        for args in (((0,3),(4,),20,'right',None),((0,3),(3,),19,'right',None),
                     ((0,3),(3,),20,'up',None),((0,3),(3,),20,'right','up')):
            with self.assertRaises(ValueError):choose(*args)

    def test_no_cross_episode_state_or_mutation(self):
        prefix=[41,31,32,33,34,35,36,37,38]
        saved=list(prefix)
        expected=choose((3,8),prefix,12,'right',None)
        choose((8,8),(88,),20,'left',None)
        self.assertEqual(choose((3,8),prefix,12,'right',None),expected)
        self.assertEqual(prefix,saved)

    def test_independent_planner_and_guard_on_reachable_public_prefixes(self):
        rng=random.Random(7321)
        for start in (0,3,9,33,44,90,99):
            cells=[start]
            for remaining in range(20,0,-1):
                position=divmod(cells[-1],10)
                for action in NEIGHBORS[cells[-1]]:
                    result=choose(position,cells,remaining,action,None)
                    self.assertEqual(result,independent_coverage(position,cells,remaining,action,None))
                    if result['changed']:
                        self.assertGreaterEqual(result['unprotected_next_radius'],result['original_next_radius'])
                cells.append(rng.choice(list(NEIGHBORS[cells[-1]].values())))

    def test_original_actor_may_still_return_inward(self):
        # The guard constrains overrides, not the original actor or a target cue.
        prefix=(44,45)*10
        result=choose((4,5),prefix,1,'left',None)
        self.assertEqual(result['action'],'left')
        self.assertFalse(result['radial_guard_blocked'])

    def test_original_mixed_and_stratum_protection_gates_are_preserved(self):
        effects={k:dict(sr_gain=.03,positive_seeds=3,sg_change=-.1) for k in ('all',*STRATA)}
        self.assertTrue(all_gates(gated(effects)))
        effects['long_distance']['sr_gain']=-.021
        self.assertFalse(all_gates(gated(effects)))
        effects['long_distance']['sr_gain']=0
        effects['interior_target']['sg_change']=.001
        self.assertFalse(all_gates(gated(effects)))


if __name__=='__main__':unittest.main()
