import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.evidence_budget import paths_for
from agents.evidence_budget_vlm import parse_evidence,parse_plan,ablate_message,request
from agents.vlm import APIConfig
from eval.evidence_budget_stage_d import plan_status,diagnostic_summary


class EvidenceVLMTests(unittest.TestCase):
    def setUp(self):
        self.state=dict(grid_size=10,position=[0,0],remaining_budget=20,
            ledger=dict(visited_order=[0],unobserved_cells=list(range(1,100))),
            candidates=paths_for((0,0),[0],10,20),local_proposals=dict(cue_action=None))
        self.E=dict(observed_facts=[dict(evidence_id='e0',kind='geometry',source_ids=['ledger'],candidate_id='c00',statement='One publicly unvisited cell.')],
            counter_evidence=[],unknowns=['target_direction'],evidence_ids=['e0'],ranked_candidates=['c00','c01'])
        self.P=dict(proposal_type='exploration_plan',candidate_id='c00',expected_new_cells=1,evidence_ids=['e0'],consumed_message_id='m0',stop_condition='after_option_or_new_cue')
        self.cfg=APIConfig('https://ollama.com/v1','gemma4:31b','fake',60,512)

    def test_evidence_sources_and_ids_are_enforced(self):
        self.assertEqual(parse_evidence(json.dumps(self.E),'stop',self.state,['target','current'])[0],self.E)
        bad=json.loads(json.dumps(self.E));bad['observed_facts'][0]['source_ids']=['unvisited_image']
        with self.assertRaises(ValueError):parse_evidence(json.dumps(bad),'stop',self.state,['target','current'])
        bad=dict(self.E,evidence_ids=['invented'])
        with self.assertRaises(ValueError):parse_evidence(json.dumps(bad),'stop',self.state,['target','current'])

    def test_plan_coverage_is_public_arithmetic_not_self_report(self):
        self.assertEqual(parse_plan(json.dumps(self.P),'stop',self.state,self.E)[0],self.P)
        with self.assertRaises(ValueError):parse_plan(json.dumps(dict(self.P,expected_new_cells=99)),'stop',self.state,self.E)
        with self.assertRaises(ValueError):parse_plan(json.dumps(dict(self.P,candidate_id='hidden_goal')),'stop',self.state,self.E)

    def test_unknown_target_allows_legal_coverage_and_unverified_claim_is_rejected(self):
        self.assertTrue(plan_status(self.P,self.state)['executable'])
        self.assertFalse(plan_status(dict(self.P,proposal_type='target_claim'),self.state)['executable'])

    def test_ablation_retains_sources_without_mutating_original(self):
        shuffled=ablate_message(self.E,'shuffled')
        self.assertEqual(shuffled['ranked_candidates'],['c01','c00'])
        self.assertEqual(shuffled['observed_facts'][0],self.E['observed_facts'][0])
        self.assertEqual(self.E['ranked_candidates'],['c00','c01'])
        self.assertFalse(ablate_message(self.E,'hidden')['observed_facts'])

    def test_second_request_really_receives_message_in_both_workflows(self):
        images=[('target',b't'),('current',b'c')]
        for arm in ('M2','M4'):
            payload,audit=request(self.state,images,self.cfg,arm,'plan',self.E)
            self.assertIn('e0',json.dumps(payload));self.assertEqual(audit['provided_message'],self.E)
            self.assertEqual(audit['image_count'],2);self.assertEqual(payload['max_tokens'],512)
            self.assertNotIn('data:image',json.dumps(audit))
        single,_=request(self.state,images,self.cfg,'M2','plan',self.E)
        two,_=request(self.state,images,self.cfg,'M4','plan',self.E)
        self.assertNotEqual(single,two)
        self.assertEqual([x['role'] for x in single['messages']],['system','user','assistant','user'])
        self.assertEqual([x['role'] for x in two['messages']],['system','user'])

    def test_truncated_and_extra_json_keys_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):parse_plan(json.dumps(self.P),'length',self.state,self.E)
        with self.assertRaises(ValueError):parse_plan(json.dumps(dict(self.P,type='text')),'stop',self.state,self.E)

    def test_equal_quality_message_changes_do_not_release_G(self):
        rows=[]
        for rep in (0,1):
            for i in range(4):
                for arm in ('M2','M4','M4_ablation'):
                    rows.append(dict(arm=arm,state=i,repeat=rep,all_responses_valid=True,evidence=self.E,plan=self.P,
                        status=dict(executable=True,reason='legal_exploration',candidate_id='c00' if arm!='M4_ablation' else 'c01',score=[1,0,2,-1])))
        report=diagnostic_summary(rows,dict(passed=True))
        self.assertFalse(report['D_numeric_passed']);self.assertFalse(report['checks']['repeated_useful_message_effect'])

    def test_one_random_better_plan_is_not_repeated_mechanism(self):
        rows=[]
        for rep in (0,1):
            for i in range(4):
                for arm in ('M2','M4','M4_ablation'):
                    full=arm!='M4_ablation';score=[2,0,2,-2] if full and rep==0 else [1,0,2,-1]
                    rows.append(dict(arm=arm,state=i,repeat=rep,all_responses_valid=True,evidence=self.E,plan=self.P,
                        status=dict(executable=True,reason='legal_exploration',candidate_id='c02' if full and rep==0 else 'c00',score=score)))
        self.assertFalse(diagnostic_summary(rows,dict(passed=True))['D_numeric_passed'])


if __name__=='__main__':unittest.main()
