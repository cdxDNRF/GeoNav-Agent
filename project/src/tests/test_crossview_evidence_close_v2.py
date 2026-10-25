"""Regression for legacy truth omission: explicit frozen mapping, no label guessing."""
import unittest
from eval.crossview_evidence_close_v2 import adapt_legacy


class LegacyTruthAdapter(unittest.TestCase):
    def setUp(self):
        self.tasks=[{'pair_id':f'p{i}','group_id':f'g{i}','goal_index_evaluator_only':i} for i in range(16)]

    def test_join_uses_frozen_task_and_preserves_input(self):
        data={'rows':[{'pair_id':'p3','group_id':'g3','ranking':[0]*25}]}
        out=adapt_legacy(data,self.tasks)
        self.assertEqual(out['rows'][0]['truth_evaluator_only'],3)
        self.assertNotIn('truth_evaluator_only',data['rows'][0])

    def test_conflicting_or_unknown_pair_group_rejected(self):
        for r in [{'pair_id':'p3','group_id':'g3','truth_evaluator_only':4},
                  {'pair_id':'p3','group_id':'wrong'},{'pair_id':'unknown','group_id':'g3'}]:
            with self.assertRaises(ValueError):adapt_legacy({'rows':[r]},self.tasks)

    def test_incomplete_or_duplicate_label_manifest_rejected(self):
        for tasks in [self.tasks[:-1],self.tasks[:-1]+[self.tasks[0]]]:
            with self.assertRaises(ValueError):adapt_legacy({'rows':[]},tasks)


if __name__=='__main__':unittest.main()
