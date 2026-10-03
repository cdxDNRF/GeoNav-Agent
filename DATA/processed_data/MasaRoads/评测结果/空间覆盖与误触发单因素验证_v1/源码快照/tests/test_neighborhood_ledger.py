import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.neighborhood_ledger_option import closed_neighborhood,neighborhood_candidates,neighborhood_choice,NeighborhoodLedgerOption
from agents.evidence_budget import paths_for
from env.environment import Observation


class NeighborhoodTests(unittest.TestCase):
    def test_grid_edges_and_duplicate_observations(self):
        self.assertEqual(closed_neighborhood([0,0],10),{0,1,10})
        self.assertEqual(closed_neighborhood([99],10),{99,89,98})
        with self.assertRaises(ValueError):closed_neighborhood([100],10)

    def test_neighborhood_opportunity_differs_from_new_cell_count(self):
        # New adjacent path cells are not all new target hypotheses; their overlap is removed.
        c=neighborhood_candidates([dict(cells=[44,45])],[43],10)[0]
        self.assertEqual(set(c['new_target_hypotheses']),{34,35,54,55,45,46})
        self.assertEqual(c['new_target_hypothesis_count'],6)
        self.assertNotIn(44,c['new_target_hypotheses'])

    def test_repeat_visit_cannot_add_already_covered_hypotheses(self):
        c=neighborhood_candidates([dict(cells=[11,12])],[11,12],10)[0]
        self.assertEqual(c['new_target_hypothesis_count'],0)
        for c in neighborhood_candidates(paths_for((5,5),[55],10,2,'right'),[55],10):
            self.assertTrue(all(0<=n<100 for n in c['new_target_hypotheses']))

    def test_ties_keep_original_priority_and_empty_abstains(self):
        candidates=paths_for((5,5),[55],10,2,'right');enriched=neighborhood_candidates(candidates,[55],10)
        cid=neighborhood_choice(enriched);selected=next(c for c in enriched if c['candidate_id']==cid)
        self.assertTrue(selected['matches_explorer']);self.assertEqual(selected['actions'],['right','right'])
        self.assertIsNone(neighborhood_choice([]))

    def test_pending_option_updates_every_step_but_accepted_cue_interrupts(self):
        ctrl=NeighborhoodLedgerOption();obs=Observation(b'a',b't',(0,0),10,20,(0,))
        ctrl.decide(obs,dict(cue_action=None,action='right',explorer_action='right'))
        ctrl.pending=['down'];ctrl.used=True
        obs=Observation(b'b',b't',(0,1),10,19,(0,1))
        r=ctrl.decide(obs,dict(cue_action='right',action='right',explorer_action='down'))
        self.assertEqual(r['action'],'right');self.assertEqual(r['pending_after'],[]);self.assertTrue(r['option_interrupted'])
        ctrl.reset();self.assertFalse(ctrl.used);self.assertIsNone(ctrl.previous)


if __name__=='__main__':unittest.main()
