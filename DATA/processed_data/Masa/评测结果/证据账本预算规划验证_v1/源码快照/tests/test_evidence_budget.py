import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.evidence_budget import EvidenceLedger,public_trigger,paths_for,deterministic_choice,geometry_score
from eval.evidence_budget_stage_t import select_tasks,public_views


class EvidenceBudgetTests(unittest.TestCase):
    def test_checkpoint_is_spent7_and_target_truth_is_not_an_argument(self):
        self.assertFalse(public_trigger(list(range(7)),14,20,False,False)[0])
        self.assertEqual(public_trigger(list(range(8)),13,20,False,False),(True,'checkpoint'))
        self.assertFalse(public_trigger(list(range(8)),13,20,False,True)[0])

    def test_repeat_trigger_and_cue_protection(self):
        self.assertEqual(public_trigger([0,1,0],18,20,False,False),(True,'stagnation'))
        self.assertFalse(public_trigger([0,1,2],18,20,False,False)[0])
        self.assertFalse(public_trigger([0,1,0],18,20,True,False)[0])
        self.assertFalse(public_trigger([0,1,0],18,20,False,False,used=True)[0])
        self.assertFalse(public_trigger(list(range(20)),1,20,False,False)[0])

    def test_paths_obey_two_actual_moves_and_public_coverage(self):
        cs=paths_for((0,0),[0],10,20,'right')
        self.assertTrue(cs)
        for c in cs:
            r=k=0
            for a,cell in zip(c['actions'],c['cells']):
                dr,dc={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}[a];r+=dr;k+=dc
                self.assertTrue(0<=r<10 and 0<=k<10);self.assertEqual(cell,r*10+k)
            self.assertLessEqual(len(c['actions']),2)
            self.assertEqual(c['anticipated_new_cells'],len(set(c['cells'])-{0}))
        self.assertTrue(all(len(c['actions'])==1 for c in paths_for((0,0),[0],10,1)))

    def test_visited_transit_can_reach_new_cell(self):
        cs=paths_for((0,0),[0,1,10],10,20)
        self.assertTrue(any(c['known_revisit_moves']==1 and c['anticipated_new_cells']==1 for c in cs))
        chosen=next(c for c in cs if c['candidate_id']==deterministic_choice(cs))
        self.assertEqual(geometry_score(chosen),max(geometry_score(c) for c in cs))

    def test_ledger_has_exact100grid_and_only_observed_images(self):
        ledger=EvidenceLedger(10,20)
        ledger.observe_fields((0,0),20,[0],'a','t',None)
        ledger.observe_fields((0,1),19,[0,1],'b','t',None,'right')
        s=ledger.snapshot();self.assertEqual(len(s['unobserved_cells']),98)
        self.assertEqual({n['cell'] for n in s['observed_nodes']},{0,1})
        self.assertFalse(any(k in s for k in ('goal','distance','area','episode_id','baseline_success')))
        self.assertEqual(s['executed_edges'][0]['to_cell'],1)
        ledger.reset();self.assertEqual(ledger.nodes,{})

    def test_ledger_rejects_discontinuous_or_changed_history(self):
        ledger=EvidenceLedger(10,20)
        with self.assertRaises(ValueError):ledger.observe_fields((0,1),19,[0,1],'b','t',None)
        ledger.observe_fields((0,0),20,[0],'a','t',None)
        with self.assertRaises(ValueError):ledger.observe_fields((0,1),19,[2,1],'b','t',None,'right')

    def test_selection_never_uses_success_labels(self):
        bank=[dict(split='dev',area=f'img_{i}',source_tile=f'{i}.png',dist=d,episode_id=f'{i}-{d}-{j}')
              for i in range(20) for d in range(12,17) for j in range(5)]
        result=select_tasks(bank)
        self.assertEqual(len(result),20);self.assertEqual(len({r['source_tile'] for r in result}),20)
        self.assertEqual(result,select_tasks(list(reversed(bank))))


if __name__=='__main__':unittest.main()
