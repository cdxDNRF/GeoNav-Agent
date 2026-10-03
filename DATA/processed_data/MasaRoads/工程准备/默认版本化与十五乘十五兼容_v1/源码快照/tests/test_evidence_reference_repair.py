import sys,json
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.evidence_reference_repair import normalize_known_refs,parse_plan_repaired


class ReferenceRepairTests(unittest.TestCase):
    def setUp(self):
        self.E=dict(observed_facts=[dict(evidence_id='e0',kind='geometry')],counter_evidence=[],evidence_ids=['e0'])
        self.state=dict(candidates=[dict(candidate_id='c01',anticipated_new_cells=1)])
        self.P=dict(proposal_type='exploration_plan',candidate_id='c01',expected_new_cells=1,evidence_ids=['geometry:e0','ledger'],consumed_message_id='m0',stop_condition='after_option_or_new_cue')

    def test_known_type_and_fact_are_normalized_with_provenance(self):
        out,aliases=normalize_known_refs(self.P,self.E)
        self.assertEqual(out['evidence_ids'],['e0','ledger']);self.assertEqual(len(aliases),1)
        self.assertEqual(self.P['evidence_ids'],['geometry:e0','ledger'])
        self.assertEqual(parse_plan_repaired(json.dumps(self.P),'stop',self.state,self.E)[0],out)

    def test_unknown_or_wrong_type_cannot_be_normalized(self):
        for ref in ('geometry:e99','visual:e0','hidden:e0','geometry:unvisited'):
            bad=dict(self.P,evidence_ids=[ref])
            with self.assertRaises(ValueError):parse_plan_repaired(json.dumps(bad),'stop',self.state,self.E)

    def test_candidate_arithmetic_and_extra_keys_remain_strict(self):
        for bad in (dict(self.P,expected_new_cells=2),dict(self.P,type='text')):
            with self.assertRaises(ValueError):parse_plan_repaired(json.dumps(bad),'stop',self.state,self.E)


if __name__=='__main__':unittest.main()
