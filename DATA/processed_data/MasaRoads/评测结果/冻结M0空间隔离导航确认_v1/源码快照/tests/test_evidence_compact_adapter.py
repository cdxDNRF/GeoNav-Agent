import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.evidence_budget import paths_for
from agents.evidence_compact_adapter import (geometry_catalog,parse_compact_evidence,parse_compact_plan,
    compact_request,ablate_compact,replay_legacy)
from agents.vlm import APIConfig
from eval.evidence_budget_stage_t import OUT,lines


class CompactAdapterTests(unittest.TestCase):
    def setUp(self):
        self.state=dict(grid_size=10,position=[1,9],remaining_budget=13,
                        ledger={'visited_order':[3,4,5,6,7,8,9,19]},
                        candidates=paths_for((1,9),[3,4,5,6,7,8,9,19],10,13,'down'),
                        local_proposals={})
        self.labels=['target','current','history_0']
        self.evidence=dict(ranked_candidates=['c03'],visual_relation='unknown',source_ids=[],normalization_receipt=[])

    def test_geometry_recomputed_and_tampered_count_rejected(self):
        fact=geometry_catalog(self.state)['c03']
        self.assertEqual(fact['score'],[2,0,3,-2])
        self.state['candidates'][3]['anticipated_new_cells']=99
        with self.assertRaises(ValueError):geometry_catalog(self.state)

    def test_unknown_null_is_explicit_receipt_not_missing_provenance_repair(self):
        obj,norm=parse_compact_evidence(json.dumps(dict(ranked_candidates=['c03'],visual_relation='unknown',source_ids=None)),
                                       'stop',self.state,self.labels)
        self.assertEqual(obj['source_ids'],[]);self.assertEqual(len(obj['normalization_receipt']),1)
        with self.assertRaises(ValueError):parse_compact_evidence(json.dumps(dict(ranked_candidates=['c03'],visual_relation='similar',source_ids=None)),
                                                                'stop',self.state,self.labels)

    def test_visual_sources_must_reference_actual_images(self):
        for refs in ([],[0,1],['target','unseen'],['target'],['target','target']):
            with self.assertRaises(ValueError):parse_compact_evidence(json.dumps(dict(ranked_candidates=['c03'],visual_relation='similar',source_ids=refs)),
                                                                    'stop',self.state,self.labels)
        obj,_=parse_compact_evidence(json.dumps(dict(ranked_candidates=['c03'],visual_relation='different',source_ids=['target','history_0'])),
                                    'stop',self.state,self.labels)
        self.assertEqual(obj['visual_relation'],'different')

    def test_unseen_candidate_and_truncation_rejected(self):
        for rank,finish in ((['c99'],'stop'),(['c03','c03'],'stop'),(['c03'],'length')):
            with self.assertRaises(ValueError):parse_compact_evidence(json.dumps(dict(ranked_candidates=rank,visual_relation='unknown',source_ids=[])),
                                                                    finish,self.state,self.labels)

    def test_plan_program_derives_counts_and_requires_own_geometry(self):
        obj,_=parse_compact_plan('{"candidate_id":"c03","evidence_refs":["g:c03","m0"]}','stop',self.state,self.evidence)
        self.assertEqual(obj['expected_new_cells'],2);self.assertEqual(obj['consumed_message_id'],'m0')
        for refs in ([],['g:c02'],['m0'],['g:c03','target']):
            with self.assertRaises(ValueError):parse_compact_plan(json.dumps(dict(candidate_id='c03',evidence_refs=refs)),
                                                                'stop',self.state,self.evidence)

    def test_hidden_message_cannot_be_cited_and_abstention_normalization(self):
        hidden=ablate_compact(self.evidence,'hidden')
        with self.assertRaises(ValueError):parse_compact_plan('{"candidate_id":"c03","evidence_refs":["g:c03","m0"]}',
                                                            'stop',self.state,hidden)
        obj,_=parse_compact_plan('{"candidate_id":null,"evidence_refs":null}','stop',self.state,None)
        self.assertEqual(obj['proposal_type'],'abstain');self.assertEqual(len(obj['normalization_receipt']),1)

    def test_target_claim_or_extra_fields_cannot_execute(self):
        with self.assertRaises(ValueError):parse_compact_plan('{"candidate_id":"c03","evidence_refs":["g:c03"],"target_claim":true}',
                                                            'stop',self.state,self.evidence)

    def test_same_permission_different_context_and_source_preserving_shuffle(self):
        cfg=APIConfig('https://ollama.com/v1','gemma4:31b','dummy',60,512)
        images=[('target',b'target'),('current',b'current')]
        a,aa=compact_request(self.state,images,cfg,'M2','plan',self.evidence)
        b,bb=compact_request(self.state,images,cfg,'M4','plan',self.evidence)
        self.assertEqual(aa['image_sha256'],bb['image_sha256'])
        self.assertEqual(aa['public_input']['geometry_tools'],bb['public_input']['geometry_tools'])
        self.assertEqual(len(a['messages']),4);self.assertEqual(len(b['messages']),2)
        self.assertEqual(ablate_compact(self.evidence,'shuffled')['source_ids'],self.evidence['source_ids'])

    def test_evaluator_truth_cannot_enter_request(self):
        cfg=APIConfig('https://ollama.com/v1','gemma4:31b','dummy',60,512)
        self.state['goal']=99
        with self.assertRaises(ValueError):compact_request(self.state,[('target',b'a'),('current',b'b')],cfg,'M4','evidence')

    def test_all_14_historical_replies_only_two_semantic_representation_repairs(self):
        records=list(lines(OUT/'API响应.jsonl'));accepted=[];changes=[];rejected=[]
        self.assertEqual(len(records),14)
        for r in records:
            try:
                _,receipt,_=replay_legacy(r);accepted.append(r['call_id'])
                if receipt:changes.append(r['call_id'])
            except ValueError:rejected.append(r['call_id'])
        self.assertEqual(changes,[6,14]);self.assertEqual(rejected,[7,11,12,13]);self.assertEqual(len(accepted),10)


if __name__=='__main__':unittest.main()
