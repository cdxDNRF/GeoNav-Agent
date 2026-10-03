import json
from pathlib import Path
import sys
import tempfile
import unittest

import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.collaboration_pilot import (CollaborationLayer, PilotAPI, request_for, parse_recommendation,
    coordinate, review_trigger, boundary_legal)
from agents.vlm import APIConfig
from env.environment import Observation
from eval.collaboration_pilot import select_tasks


class CollaborationTests(unittest.TestCase):
    def setUp(self):
        self.obs=Observation(b'current',b'target',(0,0),5,10,(0,))
        self.cfg=APIConfig('https://ollama.com/v1','gemma4:31b','fake-secret',60,1024)
        self.base=dict(action='right',explorer_action='down',cue_action='right',probabilities=[.1,.6,.1,.1,.1])
        self.yes=dict(action='down',target_evidence='supported',evidence_refs=['target','current'])

    def test_verifier_cannot_receive_current_proposal(self):
        with self.assertRaises(ValueError):request_for(self.obs,self.base,[],[],'verifier',self.cfg,self.yes)
        _,audit=request_for(self.obs,self.base,[],[],'verifier',self.cfg)
        self.assertNotIn('current_proposal',audit['public_input'])
        self.assertFalse(audit['current_proposal_visible'])

    def test_reflection_requires_and_contains_same_agent_proposal(self):
        with self.assertRaises(ValueError):request_for(self.obs,self.base,[],[],'reflector',self.cfg)
        _,audit=request_for(self.obs,self.base,[],[],'reflector',self.cfg,self.yes)
        self.assertEqual(audit['public_input']['current_proposal'],self.yes)

    def test_information_whitelist_and_history_cap(self):
        frames=[dict(pixels=b'image',position=[0,1],remaining_budget=i) for i in range(9,5,-1)]
        payload,audit=request_for(self.obs,self.base,frames,[],'planner',self.cfg)
        self.assertEqual(audit['image_count'],4)
        self.assertEqual(audit['public_input']['legal_actions'],['right','down'])
        self.assertFalse(any(k in audit['public_input'] for k in ('goal','area','distance','episode_id','source_tile')))
        self.assertNotIn('fake-secret',json.dumps(payload))
        self.assertNotIn('data:image',json.dumps(audit))

    def test_invalid_or_non_evidence_recommendations_are_rejected(self):
        bad=[dict(self.yes,action='up'),dict(self.yes,evidence_refs=['unvisited']),dict(self.yes,evidence_refs=['current']),
             dict(self.yes,type='text'),dict(self.yes,target_evidence='high')]
        for value in bad:
            with self.assertRaises(ValueError):parse_recommendation(json.dumps(value),'stop',boundary_legal(self.obs),['target','current'])
        with self.assertRaises(ValueError):parse_recommendation('{"action":"right","action":"down"}','stop',['down'],[])
        with self.assertRaises(ValueError):parse_recommendation(json.dumps(self.yes),'length',['down'],['target','current'])

    def test_joint_agreement_and_abstention_have_identical_arbitration(self):
        for mode in ('SingleReflect','TwoAgent'):
            self.assertEqual(coordinate(mode,'right',[self.yes,self.yes])[0],'down')
            self.assertEqual(coordinate(mode,'right',[self.yes,dict(self.yes,action='right')])[0],'right')
            self.assertEqual(coordinate(mode,'right',[self.yes,None])[0],'right')
            self.assertEqual(coordinate(mode,'right',[self.yes,dict(self.yes,target_evidence='unknown')])[0],'right')

    def test_two_reviews_and_episode_reset(self):
        yes=self.yes
        class API:
            circuit_reason=None
            count=0
            def recommend(self,*args,**kwargs):
                self.count+=1
                return yes,dict(call_id=self.count)
        api=API();layer=CollaborationLayer('TwoAgent',api)
        for _ in range(3):layer.act(self.obs,self.base,{})
        self.assertEqual(api.count,4)
        self.assertEqual(len(layer.memories['planner']),2)
        self.assertEqual(len(layer.memories['verifier']),2)
        self.assertEqual(layer.memories['single'],[])
        layer.reset();self.assertEqual(layer.frames,[]);self.assertEqual(layer.reviews,0)
        self.assertTrue(all(not v for v in layer.memories.values()))

    def test_no_trigger_on_equal_local_proposals(self):
        self.assertFalse(review_trigger(dict(self.base,explorer_action='right'),0))
        self.assertFalse(review_trigger(dict(self.base,cue_action=None),0))
        self.assertFalse(review_trigger(self.base,2))

    def test_provider_failure_stops_without_retry_or_key_log(self):
        class Client:
            count=0
            def request(self,*a,**kw):
                self.count+=1
                return httpx.Response(429,json=dict(error=dict(code='quota',message='fake-secret')))
        with tempfile.TemporaryDirectory() as tmp:
            client=Client();api=PilotAPI(self.cfg,client,Path(tmp))
            answer,record=api.recommend(self.obs,self.base,[],[],'planner',{})
            self.assertIsNone(answer);self.assertEqual(client.count,1);self.assertIsNotNone(api.circuit_reason)
            logs=''.join(p.read_text(encoding='utf-8') for p in Path(tmp).iterdir())
            self.assertNotIn('fake-secret',logs);self.assertNotIn('data:image',logs)

    def test_task_selection_ignores_metrics_and_balances_sources(self):
        bank=[dict(area=f'img_{i}',dist=d,episode_id=f'{i}-{d}') for i in range(28) for d in range(4,9)]
        a=select_tasks(bank)
        b=select_tasks(list(reversed([dict(x,old_success=(i%2==0)) for i,x in enumerate(bank)])))
        self.assertEqual([x['episode_id'] for x in a],[x['episode_id'] for x in b])
        self.assertEqual(len({x['area'] for x in a}),20)

if __name__=='__main__':unittest.main()
