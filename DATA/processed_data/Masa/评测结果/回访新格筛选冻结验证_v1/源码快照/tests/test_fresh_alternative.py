from pathlib import Path
import sys,unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.fresh_alternative import fresh_alternative

def obs(pos=(1,1),visited=(11,1),remaining=12):return SimpleNamespace(grid_size=10,position=pos,visited=visited,remaining_budget=remaining)
def base(action='up',cue=None,logits=(4,3,2,1)):return dict(action=action,cue_action=cue,explorer_logits=list(logits))

class FreshAlternativeTests(unittest.TestCase):
    def test_selects_highest_original_score_among_fresh_legal_moves(self):
        r=fresh_alternative(obs(),base());self.assertTrue(r['triggered']);self.assertEqual(r['action'],'right')
    def test_valid_cue_keeps_priority(self):
        r=fresh_alternative(obs(),base('right','right'));self.assertFalse(r['triggered']);self.assertEqual(r['action'],'right')
    def test_original_fresh_move_is_unchanged(self):
        r=fresh_alternative(obs(),base('down'));self.assertFalse(r['triggered']);self.assertEqual(r['action'],'down')
    def test_can_revisit_when_no_fresh_neighbor_remains(self):
        r=fresh_alternative(obs(visited=(11,1,12,21,10)),base());self.assertFalse(r['triggered']);self.assertEqual(r['action'],'up')
    def test_boundary_and_tie_use_canonical_order(self):
        r=fresh_alternative(obs((0,0),(0,1)),base('right',logits=(999,4,1,999)))
        self.assertEqual(r['action'],'down');self.assertEqual(r['fresh_actions'],['down'])
        r=fresh_alternative(obs(),base(logits=(9,2,2,2)));self.assertEqual(r['action'],'right')
    def test_only_public_state_and_original_logits_are_needed(self):
        class Restricted:
            grid_size=10;position=(1,1);visited=(11,1);remaining_budget=1
            def __getattr__(self,name):raise AssertionError('forbidden access '+name)
        self.assertEqual(fresh_alternative(Restricted(),base())['action'],'right')
    def test_invalid_original_action_and_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):fresh_alternative(obs((0,0),(0,)),base())
        with self.assertRaises(ValueError):fresh_alternative(obs(),base(logits=(float('nan'),0,0,0)))

if __name__=='__main__':unittest.main()
