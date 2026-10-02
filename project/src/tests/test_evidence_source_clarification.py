from pathlib import Path
import sys,json
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.evidence_source_clarification import clarified_request
from agents.evidence_budget_vlm import parse_evidence
from agents.vlm import APIConfig
from agents.collaboration_pilot import digest_payload


class SourceClarificationTests(unittest.TestCase):
    def test_catalogue_adds_only_names_for_already_provided_sources(self):
        state=dict(candidates=[dict(candidate_id='c00',anticipated_new_cells=1)],ledger=dict(visited_order=[0]))
        images=[('target',b't'),('current',b'c')]
        cfg=APIConfig('https://ollama.com/v1','gemma4:31b','fake',60,512)
        payload,audit=clarified_request(state,images,cfg,'M4','evidence')
        self.assertEqual(audit['public_input']['available_source_ids'],['target','current','ledger','geometry:c00'])
        self.assertEqual(audit['request_sha256'],digest_payload(payload));self.assertEqual(audit['image_count'],2)
        self.assertNotIn('available_source_ids',state)

    def test_empty_numeric_and_unknown_sources_remain_rejected(self):
        state=dict(candidates=[dict(candidate_id='c00')])
        for refs in ([],[0],['unvisited_1']):
            E=dict(observed_facts=[dict(evidence_id='e0',kind='geometry',source_ids=refs,candidate_id='c00',statement='one cell')],
                   counter_evidence=[],unknowns=[],evidence_ids=['e0'],ranked_candidates=['c00'])
            with self.assertRaises(ValueError):parse_evidence(json.dumps(E),'stop',state,['target','current'])


if __name__=='__main__':unittest.main()
